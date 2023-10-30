import pytest
import torch
from model.utils import get_patch_encoding_functions


@pytest.mark.parametrize(
    "batch,channels,input_shape,patch_shape",
    [
        (5, 2, (3, 40, 80), (1, 2, 1)),
        (5, 5, (3, 6, 40, 40), (1, 2, 1, 1)),
    ],
)
def test_get_patch_encoding_functions(
    batch: int, channels: int, input_shape: tuple[int, ...], patch_shape: tuple[int, ...]
) -> None:
    sample = torch.randn(batch, channels, *input_shape)
    encode, decode = get_patch_encoding_functions(batch, channels, input_shape, patch_shape)
    patch_encoding = encode(sample)
    assert len(patch_encoding.shape) == 3
    assert torch.all(decode(patch_encoding) == sample).item()
