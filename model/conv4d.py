"""based on https://github.com/ZhengyuLiang24/Conv4d-PyTorch/blob/main/Conv4d.py"""
from __future__ import annotations

import itertools
import math
import typing

import torch
import torch.nn as nn
import torch.nn.modules.conv

from .generic import GenericModule
from .utils import Quadruple, Triple, quadruple

_get_dimension_size = typing.cast(
    typing.Callable[[int, int, int, int, int], int],
    lambda i, k, s, p, d: ((i + 2 * p - (k) - (k - 1) * (d - 1)) // s + 1),
)


def get_dimension_shape(
    input_shape: Quadruple[int],
    kernel_size: Quadruple[int],
    stride: Quadruple[int],
    padding: Quadruple[int],
    dilation: Quadruple[int],
) -> Quadruple[int]:
    assert min(stride) > 0
    t, z, y, x = itertools.starmap(_get_dimension_size, zip(input_shape, kernel_size, stride, padding, dilation))
    return t, z, y, x


def conv3d_sequence(
    in_channels: int,
    out_channels: int,
    kernel_size: Triple[int] | int,
    stride: Triple[int] | int = 1,
    padding: Triple[int] | int = 0,
    dilation: Triple[int] | int = 1,
    groups: int = 1,
    padding_mode: str = "zeros",
    device: torch.device | str | None = None,
    dtype: torch.dtype | str | None = None,
    *,
    times: int,
    weight: torch.Tensor,
) -> nn.ModuleList:
    """
    Returns a sequence of 3D convolutional layers with shared weights along the time dimension.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Triple[int] | int): Size of the convolving kernel.
        stride (Triple[int] | int, optional): Stride of the convolution. Default: 1
        padding (Triple[int] | int, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (Triple[int] | int, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        padding_mode (str, optional): 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
        device (torch.device | str | None, optional): Device where the tensor will be allocated. Default: None
        dtype (torch.dtype | str | None, optional): Data type of the tensor. Default: None
        times (int): Number of times to repeat the convolutional layer.
        weight (torch.Tensor): Weight tensor of shape (out_channels, in_channels, times, kernel_size[0], kernel_size[1]).

    Returns:
        nn.ModuleList: A sequence of `times` 3D convolutional layers with shared weights along the time dimension.
    """
    x = nn.ModuleList()
    for i in range(times):
        m = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
            bias=False,
            groups=groups,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        m.weight = nn.Parameter(weight[:, :, i, :, :])
        x.append(m)
    return x


class Conv4d(GenericModule[[torch.Tensor], torch.Tensor]):
    hyper_cube: Quadruple[Quadruple[int]]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_shape: Quadruple[int],
        kernel_size: Quadruple[int] | int = 1,
        stride: Quadruple[int] | int = 1,
        padding: Quadruple[int] | int = 0,
        dilation: Quadruple[int] | int = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        elif out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        # -
        self.hyper_cube = kernel_size, stride, padding, dilation = (
            quadruple(kernel_size),
            quadruple(stride),
            quadruple(padding),
            quadruple(dilation),
        )
        if len(kernel_size) != 4:
            raise ValueError(f"kernel_size must be a 4-tuple, but got {len(kernel_size) =}")
        elif len(stride) != 4:
            raise ValueError(f"stride must be a 4-tuple, but got {len(stride) =}")
        elif len(padding) != 4:
            raise ValueError(f"padding must be a 4-tuple, but got {len(padding) =}")
        elif len(dilation) != 4:
            raise ValueError(f"dilation must be a 4-tuple, but got {len(dilation) =}")
        elif groups != 1:
            raise NotImplementedError("groups other than 1 not yet implemented")

        valid_padding_modes = {"zeros"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(f"padding_mode must be one of {valid_padding_modes}, but got {padding_mode =}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = quadruple(0)
        self.groups = groups
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.

        # Construct weight and bias of 4D convolution
        self.weight = weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.layers = conv3d_sequence(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size[1:],
            padding=padding[1:],
            dilation=dilation[1:],
            stride=stride[1:],
            weight=weight,
            times=kernel_size[0],
        )

        self._shape = (out_channels,) + get_dimension_shape(input_shape, kernel_size, stride, padding, dilation)
        assert len(self._shape) == 5

        self.reset_parameters()

        # del self.weight

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 6, f"unexpected {x.ndim = }"
        B, T = x.shape[0:3:2]
        out = torch.zeros((B, *self._shape), device=x.device)
        kernel, stride, pad, dilation = (x[0] for x in self.hyper_cube)  # kernel_size, padding, dilation, stride
        # Convolve each kernel frame i with each input frame j
        for i in range(kernel):
            net = self.layers[i]
            # Calculate the zero-offset of kernel frame i
            offset = -pad + (i * dilation)
            # Calculate the range of input frame j corresponding to kernel frame i
            start = max(offset % stride, offset)
            stop = min(T, T + pad - (kernel - i - 1) * dilation)
            # Convolve each kernel frame i with corresponding input frame j
            for j in range(start, stop, stride):
                # Calculate the output frame
                idx = (j - offset) // stride
                # Add results to this output frame
                out[:, :, idx, :, :, :] += net(x[:, :, j, :, :, :])

        # Add bias to output
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1, 1, 1)

        return out

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}" ", stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"
