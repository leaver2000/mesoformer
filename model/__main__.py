import random
import sys
import typing

import mesoscaler as ms
import numpy as np
import torch
import torch.nn as nn
from mesoscaler._typing import Array, Nt, Nv, Nx, Ny, Nz
from mesoscaler.enums import (
    GEOPOTENTIAL,
    SPECIFIC_HUMIDITY,
    SPECIFIC_HUMIDITY_2M,
    SURFACE_PRESSURE,
    TEMPERATURE,
    TEMPERATURE_2M,
    U_COMPONENT_OF_WIND,
    U_WIND_COMPONENT_10M,
    V_COMPONENT_OF_WIND,
    V_WIND_COMPONENT_10M,
)
from torch.utils.data import DataLoader, IterableDataset

from .mae import MaskedAutoencoder3d

era5_dvars = [
    GEOPOTENTIAL,
    TEMPERATURE,
    SPECIFIC_HUMIDITY,
    U_COMPONENT_OF_WIND,
    V_COMPONENT_OF_WIND,
]

urma_dvars = [
    SURFACE_PRESSURE,
    TEMPERATURE_2M,
    SPECIFIC_HUMIDITY_2M,
    U_WIND_COMPONENT_10M,
    V_WIND_COMPONENT_10M,
]
assert len(urma_dvars) == len(era5_dvars)
CHANNELS = len(urma_dvars)


class Dataset(IterableDataset[Array[[Nv, Nt, Nz, Ny, Nx], np.float_]]):
    def __init__(self, resampler: ms.ReSampler) -> None:
        super().__init__()

        self.resampler = resampler
        self.indices = ms.AreaOfInterestSampler(resampler.domain, aoi=(-106.6, 25.8, -93.5, 36.5))
        torch.random.manual_seed(0)
        random.shuffle(self.indices.indices)

    def __iter__(self):
        for (lon, lat), time in self.indices:
            yield self.resampler(lon, lat, time)

    def __len__(self):
        return len(self.indices)


def add_weight_decay(model: nn.Module, weight_decay=1e-5, skip_list=(), bias_wd=False):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (not bias_wd) and len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def normalized_gradient(
    parameters: typing.Iterable[torch.Tensor] | torch.Tensor,
    norm_type: float = 2.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    norm_type = float(norm_type)
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    gradients = [p.grad for p in parameters if p.grad is not None]

    if len(gradients) == 0:
        return torch.tensor(0.0)
    if device is None:
        device = gradients[0].device

    if norm_type == torch.inf:
        # return max(grad.detach().abs().max().to(device) for grad in gradients)
        return torch.max(
            torch.stack([torch.max(grad.detach().abs()).to(device) for grad in gradients]),
            # dim=0,
        )

    return torch.norm(
        torch.stack([torch.norm(grad.detach(), norm_type).to(device) for grad in gradients]),
        norm_type,
    )


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(
        self, fp32: bool = False, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self._scaler = torch.cuda.amp.GradScaler(enabled=not fp32)
        self._device = device

    def __call__(
        self,
        loss: torch.Tensor,
        optimizer,
        clip_grad=None,
        parameters: typing.Iterable[torch.Tensor] | torch.Tensor | None = None,
        create_graph: bool = False,
        update_grad: bool = True,
    ) -> torch.Tensor | None:
        typing.cast(torch.Tensor, self._scaler.scale(loss)).backward(create_graph=create_graph)
        if parameters is None:
            norm = None
        elif update_grad and clip_grad is not None:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # un-scale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = normalized_gradient(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self) -> dict[str, typing.Any]:
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict: dict[str, typing.Any]) -> None:
        self._scaler.load_state_dict(state_dict)


def main(
    dataset_sequence: ms.DatasetSequence,
    width: int = 80,
    height: int = 40,
    distance_ratio: float = 2.5,  # km
    patch_ratio=0.2,
    levels: list[float] = [1013.25, 1000, 850, 700, 500, 300],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size: int = 4,
    num_epochs: int = 1,
    num_workers: int = 4,
    lr: float = 1e-3,
    fp32: bool = False,
) -> int:
    input_shape = len(levels), height, width  # (Z, Y, X)
    patch_shape = len(levels) // 2, int(height * patch_ratio), int(width * patch_ratio)

    dx = int(width * distance_ratio)
    dy = int(height * distance_ratio)

    scale = ms.Mesoscale(dx, dy, levels=levels, rate=12)
    model = MaskedAutoencoder3d(batch_size, CHANNELS, input_shape, patch_shape, embed_dim=768, decoder_embed_dim=768)

    model.train(True)
    model.to(device)

    optimizer = torch.optim.AdamW(params=add_weight_decay(model, weight_decay=1e-5, skip_list=["mask"]))
    loss_scaler = NativeScalerWithGradNormCount(fp32=fp32)
    accum_iter = 1
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        dataset = Dataset(scale.resample(dataset_sequence, height=height, width=width))
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        print(f"Epoch {epoch} len(ds): {len(dataset)}")
        for data_iter_step, batch in enumerate(data_loader):
            batch = batch[:, :, 0, :, :, :]  # type: torch.Tensor
            batch = batch.to(device)
            loss, _, _ = model(batch)
            loss /= accum_iter
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
                update_grad=(data_iter_step + 1) % accum_iter == 0,
                # clip_grad=args.clip_grad,
            )
            print(f"loss: {loss}")
            accum_iter += 1

    torch.save(model.state_dict(), "model.pt")
    return 0


if __name__ == "__main__":
    dataset_sequence = ms.open_datasets([("data/urma.zarr", urma_dvars), ("data/era5.zarr", era5_dvars)])
    sys.exit(main(dataset_sequence))
