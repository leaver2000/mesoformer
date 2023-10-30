from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable, overload

import torch
import torch.nn as nn

from .conv4d import Conv4d
from .generic import GenericModule
from .utils import Pair, Quadruple, Triple, all_equals, pair


def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(GenericModule[[torch.Tensor], torch.Tensor]):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class MultiLayerPerceptron(GenericModule[[torch.Tensor], torch.Tensor]):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        activation: Callable[[], nn.Module] = nn.GELU,
        normalization: Callable[[int], nn.Module] | None = None,
        bias: Pair[bool] = pair(True),
        drop_prob: float = 0.0,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        normalization = normalization or nn.Identity

        linear_layer = functools.partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = activation()
        self.drop1, self.drop2 = map(nn.Dropout, pair(drop_prob))

        self.norm = normalization(hidden_features)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(GenericModule[[torch.Tensor], torch.Tensor]):
    if TYPE_CHECKING:
        q: Callable[[torch.Tensor], torch.Tensor]
        k: Callable[[torch.Tensor], torch.Tensor]
        v: Callable[[torch.Tensor], torch.Tensor]
        scale: float

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        input_size: Triple[int] = (4, 14, 14),
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert input_size[1] == input_size[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x


# =====================================================================================================================
#
# =====================================================================================================================
class Block(GenericModule[[torch.Tensor], torch.Tensor]):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_prob: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_func: Callable[..., nn.Module] = Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop_prob,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MultiLayerPerceptron(
            in_features=dim, hidden_features=mlp_hidden_dim, activation=act_layer, drop_prob=drop_prob
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def block_list(
    dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = False,
    qk_scale: float | None = None,
    drop_prob: float = 0.0,
    attn_drop: float = 0.0,
    drop_path: float = 0.0,
    act_layer: Callable[..., nn.Module] = nn.GELU,
    norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    attn_func: Callable[..., nn.Module] = Attention,
    *,
    times: int,
) -> nn.ModuleList:
    return nn.ModuleList(
        Block(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_prob=drop_prob,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attn_func=attn_func,
        )
        for _ in range(times)
    )


# =====================================================================================================================
#
# =====================================================================================================================


class PatchEmbed3d(GenericModule[[torch.Tensor], torch.Tensor]):
    """Image to Patch Embedding"""

    if TYPE_CHECKING:
        net: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        input_shape: Triple[int],
        patch_shape: Triple[int],
    ) -> None:
        super().__init__()
        self.input_shape = input_shape, "Input shape must be 3D"
        self.net = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_shape, stride=patch_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, time_steps, height*width, channels).
        """
        if not all_equals(x.shape[2:], self.input_shape):
            raise ValueError(f"Input image size ({x.shape}) doesn't match model ({self.input_shape}).")
        return self.net(x).flatten(3).moveaxis(1, -1)


class PatchEmbed4d(GenericModule[[torch.Tensor], torch.Tensor]):
    """Image to Patch Embedding"""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        input_shape: Quadruple[int],
        patch_shape: Quadruple[int],
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        assert len(patch_shape) == 4, "4D kernel size expected!"
        self.net = Conv4d(in_channels, embed_dim, input_shape=input_shape, kernel_size=patch_shape, stride=patch_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, time_steps, height*width, channels).
        """

        if not all_equals(x.shape[2:], self.input_shape):
            raise ValueError(f"Input image size ({x.shape}) doesn't match model ({self.input_shape}).")

        return self.net(x).flatten(3).moveaxis(1, -1)


class PatchEmbedNd(GenericModule[[torch.Tensor], torch.Tensor]):
    @overload
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        input_shape: Triple[int],
        patch_shape: Triple[int],
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        input_shape: Quadruple[int],
        patch_shape: Quadruple[int],
    ) -> None:
        ...

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        input_shape: Triple[int] | Quadruple[int],
        patch_shape: Triple[int] | Quadruple[int],
    ) -> None:
        super().__init__()
        self.input_shape = input_shape

        if len(input_shape) == 3 and len(patch_shape) == 3:
            self.net = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_shape, stride=patch_shape)
        elif len(input_shape) == 4 and len(patch_shape) == 4:
            self.net = Conv4d(
                in_channels, embed_dim, input_shape=input_shape, kernel_size=patch_shape, stride=patch_shape
            )
        else:
            raise ValueError(f"Input shape {input_shape} and patch shape {patch_shape} must have same length")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, time_steps, height*width, channels).
        """

        if not all_equals(x.shape[2:], self.input_shape):
            raise ValueError(f"Input image size ({x.shape}) doesn't match model ({self.input_shape}).")

        return self.net(x).flatten(3).moveaxis(1, -1)
