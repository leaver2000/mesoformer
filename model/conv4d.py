from __future__ import annotations
import torch
import torch.nn as nn
import math
from .utils import Quadruple, quadruple


class Conv4d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Quadruple[int] | int,
        stride: Quadruple[int] | int = (1, 1, 1, 1),
        padding: Quadruple[int] | int = (0, 0, 0, 0),
        dilation: Quadruple[int] | int = (1, 1, 1, 1),
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        kernel_size = quadruple(kernel_size)
        stride = quadruple(stride)
        padding = quadruple(padding)
        dilation = quadruple(dilation)

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
        self.conv3d_layers = torch.nn.ModuleList()

        for i in range(self.kernel_size[0]):
            # Initialize a Conv3D layer
            conv3d_layer = nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size[1::],
                padding=self.padding[1::],
                dilation=self.dilation[1::],
                stride=self.stride[1::],
                bias=False,
            )
            conv3d_layer.weight = nn.Parameter(self.weight[:, :, i, :, :])

            # Store the layer
            self.conv3d_layers.append(conv3d_layer)

        del self.weight

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, __x: torch.Tensor) -> torch.Tensor:
        # Define shortcut names for dimensions of input and kernel
        B, C, T, Z, Y, X = __x.shape

        (l_k, d_k, h_k, w_k) = self.kernel_size
        (l_p, d_p, h_p, w_p) = self.padding
        (l_d, d_d, h_d, w_d) = self.dilation
        (l_s, d_s, h_s, w_s) = self.stride

        # Compute the size of the output tensor based on the zero padding
        l_o = (T + 2 * l_p - (l_k) - (l_k - 1) * (l_d - 1)) // l_s + 1
        d_o = (Z + 2 * d_p - (d_k) - (d_k - 1) * (d_d - 1)) // d_s + 1
        h_o = (Y + 2 * h_p - (h_k) - (h_k - 1) * (h_d - 1)) // h_s + 1
        w_o = (X + 2 * w_p - (w_k) - (w_k - 1) * (w_d - 1)) // w_s + 1

        # Pre-define output tensors
        out = torch.zeros(B, self.out_channels, l_o, d_o, h_o, w_o).to(__x.device)

        # Convolve each kernel frame i with each input frame j
        for i in range(l_k):
            # Calculate the zero-offset of kernel frame i
            zero_offset = -l_p + (i * l_d)
            # Calculate the range of input frame j corresponding to kernel frame i
            j_start = max(zero_offset % l_s, zero_offset)
            j_end = min(T, T + l_p - (l_k - i - 1) * l_d)
            # Convolve each kernel frame i with corresponding input frame j
            for j in range(j_start, j_end, l_s):
                # Calculate the output frame
                out_frame = (j - zero_offset) // l_s
                # Add results to this output frame
                out[:, :, out_frame, :, :, :] += self.conv3d_layers[i](__x[:, :, j, :, :])

        # Add bias to output
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1, 1)

        return out


if __name__ == "__main__":
    B = 10
    C = 5

    T = 20
    Z = 12
    Y = 40
    X = 80

    input = torch.randn(B, C, T, Z, Y, X).cuda()

    net = Conv4d(
        C, C, kernel_size=(T, 1, 1, 1), padding=(0, 0, 0, 0), stride=(1, 1, 1, 1), dilation=(1, 1, 1, 1), bias=True
    ).cuda()
    out1 = net(input)
    print(out1.shape)
