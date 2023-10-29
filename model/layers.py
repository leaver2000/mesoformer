from __future__ import annotations

from typing import Callable, TYPE_CHECKING
import functools


import torch
import torch.nn as nn

from .utils import Pair, Triple, pair, all_equals


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


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class MultiLayerPerceptron(nn.Module):
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


class Attention(nn.Module):
    if TYPE_CHECKING:
        q: Callable[[torch.Tensor], torch.Tensor]
        k: Callable[[torch.Tensor], torch.Tensor]
        v: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        dim: int,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
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
class Block(nn.Module):
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
class PatchEmbed3d(nn.Module):
    """Image to Patch Embedding"""

    if TYPE_CHECKING:
        proj: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self, input_shape: Triple[int], patch_shape: Triple[int], in_chans: int, embed_dim: int = 768
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_shape, stride=patch_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, time_steps, height*width, channels).
        """
        if not all_equals(x.shape[-3:], self.input_shape):
            raise ValueError(f"Input image size ({x.shape}) doesn't match model ({self.input_shape}).")
        x = self.proj(x).flatten(3)
        x = torch.einsum("bczs->bzsc", x)  # [N, T, H*W, C]
        return x
