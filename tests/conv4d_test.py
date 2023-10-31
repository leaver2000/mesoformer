from __future__ import annotations
import itertools
import torch
import model.conv4d

import pytest
from model.utils import quadruple, Quadruple


def test_get_dimension_size() -> None:
    kernel_size = (1, 1, 1, 1)
    padding = (0, 0, 0, 0)
    stride = (1, 1, 1, 1)
    dilation = (1, 1, 1, 1)
    shape = (Batch, _, l_i, d_i, h_i, w_i) = (10, 5, 8, 12, 40, 80)

    (l_k, d_k, h_k, w_k) = kernel_size
    (l_p, d_p, h_p, w_p) = padding
    (l_d, d_d, h_d, w_d) = dilation
    (l_s, d_s, h_s, w_s) = stride
    l_o = (l_i + 2 * l_p - (l_k) - (l_k - 1) * (l_d - 1)) // l_s + 1
    d_o = (d_i + 2 * d_p - (d_k) - (d_k - 1) * (d_d - 1)) // d_s + 1
    h_o = (h_i + 2 * h_p - (h_k) - (h_k - 1) * (h_d - 1)) // h_s + 1
    w_o = (w_i + 2 * w_p - (w_k) - (w_k - 1) * (w_d - 1)) // w_s + 1
    expect = l_o, d_o, h_o, w_o
    assert (
        expect
        == tuple(
            itertools.starmap(
                model.conv4d._get_dimension_size,
                zip(shape[2:], kernel_size, stride, padding, dilation),
            )
        )
        == model.conv4d.get_dimension_shape(
            shape[2:],
            kernel_size,
            stride,
            padding,
            dilation,
        )
    )


@pytest.mark.parametrize(
    "shape,out_channels,kernel_size,stride,padding,dilation",
    [
        ((10, 5, 8, 12, 40, 80), 5, 1, 1, 1, 1),
        ((10, 5, 8, 12, 40, 80), 10, 1, 1, 1, 1),
        ((10, 5, 8, 12, 40, 80), 10, 2, 2, 2, 2),
    ],
)
def test_conv4d(
    shape: tuple[int, ...],
    out_channels: int,
    kernel_size: Quadruple[int] | int,
    stride: Quadruple[int] | int,
    padding: Quadruple[int] | int,
    dilation: Quadruple[int] | int,
) -> None:
    B, C, T, Z, Y, X = shape
    kernel_size, padding, dilation, stride = map(quadruple, (kernel_size, padding, dilation, stride))
    batch = torch.randn(B, C, T, Z, Y, X)
    input_shape = (T, Z, Y, X)
    m = model.conv4d.Conv4d(
        C, out_channels, input_shape, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )

    assert tuple(m(batch).shape) == (B, out_channels) + model.conv4d.get_dimension_shape(
        input_shape,
        kernel_size,
        padding,
        dilation,
        stride,
    )
