"""based on https://github.com/ZhengyuLiang24/Conv4d-PyTorch/blob/main/Conv4d.py"""
from __future__ import annotations

import itertools
import math
from typing import Callable, Iterable

import torch
import torch.nn as nn

from .generic import GenericModule
from .utils import Quadruple

_get_dimension_size: Callable[[int, int, int, int, int], int] = (
    lambda i, k, p, d, s: (i + 2 * p - (k) - (k - 1) * (d - 1)) // s + 1
)


def _generate3d(
    in_channels, out_channels, kernel_size, padding, dilation, weight, stride, *, times: int
) -> Iterable[nn.Conv3d]:
    for i in range(times):
        m = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
            bias=False,
        )
        m.weight = nn.Parameter(weight[:, :, i, :, :])
        yield m


class Conv4d(GenericModule[[torch.Tensor], torch.Tensor]):
    def __init__(
        self,
        in_channels: int,
        # batch_size: int,
        out_channels: int,
        input_shape: Quadruple[int],
        kernel_size: Quadruple[int],
        stride: Quadruple[int] = (1, 1, 1, 1),
        padding: Quadruple[int] = (0, 0, 0, 0),
        dilation: Quadruple[int] = (1, 1, 1, 1),
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
    ):
        super().__init__()

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_modes = {"zeros"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                "padding_mode must be one of {}, but got padding_mode='{}'".format(valid_padding_modes, padding_mode)
            )

        # Assertions for constructor arguments
        assert len(kernel_size) == 4, "4D kernel size expected!"
        assert len(stride) == 4, "4D Stride size expected!!"
        assert len(padding) == 4, "4D Padding size expected!!"
        assert len(dilation) == 4, "4D dilation size expected!"
        assert groups == 1, "Groups other than 1 not yet implemented!"

        # Store constructor arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.groups = groups
        self.padding_mode = padding_mode

        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        # # # # # self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 3)

        # Construct weight and bias of 4D convolution
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.layers = torch.nn.ModuleList(
            _generate3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size[1::],
                padding=padding[1::],
                dilation=dilation[1::],
                weight=self.weight,
                stride=stride[1::],
                times=kernel_size[0],
            )
        )

        self._shape = (
            out_channels,
            *(itertools.starmap(_get_dimension_size, zip(input_shape, kernel_size, padding, dilation, stride))),
        )

        self._dim_zero = tuple(x[0] for x in (kernel_size, padding, dilation, stride))
        # del self.weight

    def _get_zeros(self, batch_size: int, *, device=None) -> torch.Tensor:
        return torch.zeros((batch_size, *self._shape), device=device)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Define shortcut names for dimensions of input and kernel

        B, T = x.shape[0:3:2]

        out = self._get_zeros(B, device=x.device)

        k, p, d, s = self._dim_zero
        # Convolve each kernel frame i with each input frame j
        for i in range(k):
            # Calculate the zero-offset of kernel frame i
            zero_offset = -p + (i * d)
            # Calculate the range of input frame j corresponding to kernel frame i
            j_start = max(zero_offset % s, zero_offset)
            j_end = min(T, T + p - (k - i - 1) * d)
            # Convolve each kernel frame i with corresponding input frame j
            for j in range(j_start, j_end, s):
                # Calculate the output frame
                out_frame = (j - zero_offset) // s
                # Add results to this output frame
                out[:, :, out_frame, :, :, :] += self.layers[i](x[:, :, j, :, :])

        # Add bias to output
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1, 1)

        return out
