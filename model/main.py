import mesoscaler as ms
import torch
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
from torch.utils.data import DataLoader

from .data import MesoscalerDataset
from .mae import MaskedAutoencoder4d
from .utils import NativeScalerWithGradNormCount, add_weight_decay

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


def open_dataset():
    return ms.open_datasets([("data/urma.zarr", urma_dvars), ("data/era5.zarr", era5_dvars)])


def main(
    width: int = 80,
    height: int = 40,
    distance_ratio: float = 2.5,  # km
    patch_ratio=0.2,
    levels: list[float] = [1013.25, 1000, 850, 700, 500, 300],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_time: int = 12,
    batch_size: int = 1,
    num_epochs: int = 1,
    num_workers: int = 4,
    lr: float = 1e-3,
    fp32: bool = False,
) -> int:
    batch: torch.Tensor
    dataset_sequence = open_dataset()
    input_shape = num_time, len(levels), height, width  # (T, Z, Y, X)
    patch_shape = num_time, len(levels) // 2, int(height * patch_ratio), int(width * patch_ratio)

    dx = int(width * distance_ratio)
    dy = int(height * distance_ratio)

    scale = ms.Mesoscale(dx, dy, levels=levels, rate=12)

    model = MaskedAutoencoder4d(
        batch_size,
        CHANNELS,
        input_shape,
        patch_shape,
        embed_dim=768,
        decoder_embed_dim=768,
    )
    model.train(True)
    model.to(device)

    optimizer = torch.optim.AdamW(params=add_weight_decay(model, weight_decay=1e-5, skip_list=["mask"]))
    loss_scaler = NativeScalerWithGradNormCount(fp32=fp32)
    accum_iter = 1
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        dataset = MesoscalerDataset(
            scale.resample(dataset_sequence, height=height, width=width), time_batch_size=num_time
        )
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        print(f"Epoch {epoch} len(ds): {len(dataset)}")

        for data_iter_step, batch in enumerate(data_loader):
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

    model.to_disk("model.pt")
    return 0
