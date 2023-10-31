# .utils.py
# ruff: noqa: F401
from __future__ import annotations

import math
import string
import typing
from typing import Any, Callable, Iterable, TypeAlias, TypeVar

import torch

from . import reduce

_T = TypeVar("_T", bound=Any)
DictStrAny: TypeAlias = dict[str, Any]
Single: TypeAlias = tuple[_T]
Pair: TypeAlias = tuple[_T, _T]
Triple: TypeAlias = tuple[_T, _T, _T]
Quadruple: TypeAlias = tuple[_T, _T, _T, _T]


if typing.TYPE_CHECKING:

    def single(x: _T | Single[_T]) -> _T:
        ...

    def pair(x: _T | Pair[_T]) -> Pair[_T]:
        ...

    def triple(x: _T | Triple[_T]) -> Triple[_T]:
        ...

    def quadruple(x: _T | Quadruple[_T]) -> Quadruple[_T]:
        ...

else:
    from torch.nn.modules.utils import _pair as pair
    from torch.nn.modules.utils import _quadruple as quadruple
    from torch.nn.modules.utils import _single as single
    from torch.nn.modules.utils import _triple as triple


def all_equals(x: Iterable[Any], y: Iterable[Any]) -> bool:
    return all(reduce.equals(x, y))


def einsum_transpose(
    unsqueeze: tuple[int, ...], equation: str, output_shape: tuple[int, ...]
) -> Callable[[torch.Tensor], torch.Tensor]:
    x, y = equation.split("->")
    assert set(x) == set(y)
    assert math.prod(unsqueeze) == math.prod(output_shape)
    return lambda x: torch.einsum(equation, x.reshape(unsqueeze)).reshape(output_shape)


def _get_patching_equations(dims: int) -> Pair[str]:
    """This function will generate a pair of equations to be used with `torch.einsum` to patch and unpatch a tensor.

    >>> _get_equations(4)
    ('BCAaDdEeFf->BADEFadefC', 'BADEFadefC->BCAaDdEeFf')
    """
    c = string.ascii_uppercase.replace("B", "").replace("C", "")
    if dims > len(c):
        raise ValueError(f"cannot generate equations for {dims} dimensions")
    c = c[:dims]

    left, right = tuple(c), tuple(c.lower())
    x = "BC" + "".join(f"{l}{r}" for l, r in zip(left, right))
    y = f"B{''.join(left)}{''.join(right)}C"
    return f"{x}->{y}", f"{y}->{x}"


def get_patch_encoding_functions(
    batch_size: int, in_chans: int, input_shape: tuple[int, ...], patch_shape: tuple[int, ...]
) -> Pair[Callable[[torch.Tensor], torch.Tensor]]:
    """
    Returns a pair of functions to encode and decode patches of a given shape from a tensor of a given shape.

    Args:
        batch_size (int): The batch size of the input tensor.
        in_chans (int): The number of channels of the input tensor.
        input_shape (tuple[int, ...]): The shape of the input tensor, excluding the batch and channel dimensions.
        patch_shape (tuple[int, ...]): The shape of the patches to extract from the input tensor.

    Returns:
        A pair of functions (encode, decode), where encode takes a tensor of shape (batch_size, in_chans, *input_shape)
        and returns a tensor of shape (batch_size, num_patches, patch_size * in_chans), and decode takes a tensor
        of shape (batch_size, num_patches, patch_size * in_chans) and returns a tensor of shape
        (batch_size, in_chans, *input_shape), where num_patches is the number of patches that can be extracted from
        the input tensor given the patch shape.

    Raises:
        ValueError: If the input shape and patch shape have different lengths, or if the input shape is not divisible
        by the patch shape.

    ```python
    B = 5
    C = 3
    T = 2
    Z = 4
    Y = 4
    X = 4
    x = torch.randn(B, C, T, Z, Y, X)
    assert len(x.shape) != 3
    patch, unpatch = get_patch_encoding_functions(B, C, (T, Z, Y, X), (2, 2, 2, 2))
    assert len(patch(x).shape) == 3
    assert torch.all(unpatch(patch(x)) == x).item()
    ```
    """

    if len(input_shape) != len(patch_shape):
        raise ValueError(f"input shape {input_shape} and patch shape {patch_shape} must have same length")
    elif any(reduce.mod(input_shape, patch_shape)):
        raise ValueError(f"input shape {input_shape} is not divisible by patch shape {patch_shape}")

    encode_equation, decode_equation = _get_patching_equations(len(input_shape))

    grid_shape = tuple(reduce.floordiv(input_shape, patch_shape))
    patch_unsqueeze = tuple(x for xy in zip(grid_shape, patch_shape) for x in xy)

    output_shape = output_shape = (batch_size, math.prod(grid_shape), math.prod(patch_shape) * in_chans)
    if output_shape != (
        batch_size,
        math.prod(input_shape) // math.prod(patch_shape),
        math.prod(patch_shape) * in_chans,
    ):
        raise ValueError(f"output shape {output_shape} does not match expected shape {output_shape}")
    patch = einsum_transpose(
        (batch_size, in_chans) + patch_unsqueeze,
        encode_equation,
        output_shape,
    )
    unpatch = einsum_transpose(
        (batch_size, *grid_shape, *patch_shape, in_chans),
        decode_equation,
        (batch_size, in_chans, *input_shape),
    )
    return patch, unpatch


# =====================================================================================================================
def normalized_gradient(
    parameters: Iterable[torch.Tensor] | torch.Tensor,
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
        parameters: Iterable[torch.Tensor] | torch.Tensor | None = None,
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


# =====================================================================================================================
def add_weight_decay(
    model: torch.nn.Module, weight_decay: float = 1e-5, skip_list=(), bias_wd=False
) -> list[dict[str, Any]]:
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
