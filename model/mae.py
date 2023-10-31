"""based onhttps://github.com/facebookresearch/mae_st/blob/main/models_mae.py"""
from __future__ import annotations

import math
from typing import Callable, Concatenate, ParamSpec

import torch
import torch.nn as nn

from . import reduce
from .generic import GenericModule, IOModule
from .layers import PatchEmbedNd, block_list
from .utils import DictStrAny, Quadruple, Triple, get_patch_encoding_functions

_P = ParamSpec("_P")


class MaskedAutoencoder3d(GenericModule[Concatenate[torch.Tensor, _P], Triple[torch.Tensor]]):
    def __init__(
        self,
        batch_size: int,
        in_channels: int,
        input_shape: Triple[int],
        patch_shape: Triple[int],
        *,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        norm_pix_loss: bool = False,
        no_qkv_bias: bool = False,
        pred_t_dim: int | None = None,
    ) -> None:
        super().__init__()
        B, C = batch_size, in_channels
        Z, Y, X = input_shape
        Pz, Py, Px = patch_shape
        if pred_t_dim is None:
            # predict just the last frame
            pred_t_dim = Z
        # - sanity check -
        for i, p in zip(input_shape, patch_shape):
            if i % p != 0:
                raise ValueError(f"Input image size ({input_shape}) doesn't match patch size ({patch_shape}).")

        # - shape info -
        self.grid_shape = z, y, x = Z // Pz, Y // Py, X // Px
        self.grid_size = grid_size = z * y * x
        self.output_shape = B, C, Z, Y, X

        #  - patch embedding -
        self.pred_t_dim = pred_t_dim
        self.t_pred_patch_size = z = z * pred_t_dim // Z
        self.patch_embed = PatchEmbedNd(
            in_channels,
            embed_dim,
            input_shape,
            patch_shape,
        )
        self.patch_encode, self.patch_decode = get_patch_encoding_functions(
            B,
            C,
            input_shape,
            patch_shape,
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size, embed_dim))

        self.blocks = block_list(
            embed_dim,
            num_heads,
            mlp_ratio,
            qkv_bias=not no_qkv_bias,
            qk_scale=None,
            norm_layer=norm_layer,
            times=depth,
        )
        self.norm = norm_layer(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, grid_size, decoder_embed_dim))

        self.decoder_blocks = block_list(
            decoder_embed_dim,
            decoder_num_heads,
            mlp_ratio,
            qkv_bias=not no_qkv_bias,
            qk_scale=None,
            norm_layer=norm_layer,
            #
            times=decoder_depth,
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, math.prod(patch_shape) * in_channels, bias=True)

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        print("model initialized")

    def initialize_weights(self) -> None:
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        w = self.patch_embed.net.weight.data
        # if self.trunc_init:
        #     torch.nn.init.trunc_normal_(w)
        #     torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        # else:
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            # if self.trunc_init:
            #     nn.init.trunc_normal_(m.weight, std=0.02)
            # else:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> Quadruple[torch.Tensor]:
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # un-shuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def encode(self, x: torch.Tensor, mask_ratio: float) -> Triple[torch.Tensor]:
        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape

        x = x.reshape(N, T * L, C)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        x = x.view(N, -1, C)

        pos_embed = self.pos_embed[:, 0:, :].expand(x.shape[0], -1, -1)
        pos_embed = torch.gather(
            pos_embed,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
        )
        x = x.view([N, -1, C]) + pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = x[:, :, :]

        return x, mask, ids_restore

    def decode(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        B, *_, C = x.shape

        Z, Y, X = self.grid_shape

        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(B, Z * Y * X + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([B, Z * Y * X, C])
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]))  # unshuffle
        x = x_.view([B, Z * Y * X, C])

        x += self.decoder_pos_embed[:, :, :]

        attn = self.decoder_blocks[0].attn
        requires_t_shape = bool(getattr(attn, "requires_t_shape", False))
        if requires_t_shape:
            x = x.view([B, Z, Y * X, C])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if requires_t_shape:
            x = x.view([B, Z * Y * X, -1])

        else:
            x = x[:, :, :]

        return x

    def loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        imgs: [N, 3, T, H, W]
        pred: [N, t*h*w, u*p*p*3]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """
        _imgs = torch.index_select(
            imgs, 2, torch.linspace(0, imgs.shape[2] - 1, self.pred_t_dim).long().to(imgs.device)
        )
        target = self.patch_encode(_imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        return (loss * mask).sum() / mask.sum()  # mean loss on removed patches

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.75) -> Triple[torch.Tensor]:
        latent, mask, ids_restore = self.encode(x, mask_ratio)
        pred = self.decode(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.loss(x, pred, mask)
        return loss, pred, mask


class MaskedAutoencoder4d(IOModule[Concatenate[torch.Tensor, _P], Triple[torch.Tensor]]):
    @property
    def _constructor_kwargs(self) -> DictStrAny:
        return {
            "batch_size": self.batch_size,
            "in_channels": self.in_channels,
            "input_shape": self.input_shape,
            "patch_shape": self.patch_shape,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "decoder_embed_dim": self.decoder_embed_dim,
            "decoder_depth": self.decoder_depth,
            "decoder_num_heads": self.decoder_num_heads,
            "mlp_ratio": self.mlp_ratio,
            "norm_pix_loss": self.norm_pix_loss,
            "no_qkv_bias": self.no_qkv_bias,
            "pred_t_dim": self.pred_t_dim,
        }

    def __init__(
        self,
        batch_size: int,
        in_channels: int,
        input_shape: Quadruple[int],
        patch_shape: Quadruple[int],
        *,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_pix_loss: bool = False,
        no_qkv_bias: bool = False,
        pred_t_dim: int | None = None,
    ) -> None:
        super().__init__()
        # - set constructor arguments -
        self.batch_size = B = batch_size
        self.in_channels = C = in_channels
        self.input_shape = T, Z, Y, X = input_shape
        self.patch_shape = patch_shape
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_pix_loss = norm_pix_loss
        self.no_qkv_bias = no_qkv_bias
        self.pred_t_dim = pred_t_dim = pred_t_dim if pred_t_dim is not None else T

        # - sanity check -
        for i, p in zip(input_shape, patch_shape):
            if i % p != 0:
                raise ValueError(f"Input image size ({input_shape}) doesn't match patch size ({patch_shape}).")

        # - shape info -
        self.grid_shape = t, z, y, x = tuple(reduce.floordiv(input_shape, patch_shape))
        self.grid_size = grid_size = math.prod(self.grid_shape)
        self.output_shape = B, C, T, Z, Y, X

        #  - patch embedding -
        self.t_pred_patch_size = t * self.pred_t_dim // T
        self.patch_embed = PatchEmbedNd(in_channels, embed_dim, input_shape, patch_shape)
        self.patch_encode, self.patch_decode = get_patch_encoding_functions(
            batch_size, in_channels, input_shape, patch_shape
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size, embed_dim))

        self.blocks = block_list(
            embed_dim,
            num_heads,
            mlp_ratio,
            qkv_bias=not no_qkv_bias,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            times=depth,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, grid_size, decoder_embed_dim))

        self.decoder_blocks = block_list(
            decoder_embed_dim,
            decoder_num_heads,
            mlp_ratio,
            qkv_bias=not no_qkv_bias,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            #
            times=decoder_depth,
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, math.prod(patch_shape) * in_channels, bias=True)

        self.initialize_weights()
        print("model initialized")

    def initialize_weights(self) -> None:
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        w = self.patch_embed.net.weight.data

        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> Quadruple[torch.Tensor]:
        B, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # un-shuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def encode(self, x: torch.Tensor, mask_ratio: float) -> Triple[torch.Tensor]:
        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape

        x = x.reshape(N, T * L, C)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        x = x.view(N, -1, C)

        pos_embed = self.pos_embed[:, 0:, :].expand(x.shape[0], -1, -1)
        pos_embed = torch.gather(
            pos_embed,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
        )
        x = x.view([N, -1, C]) + pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = x[:, :, :]

        return x, mask, ids_restore

    def decode(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        B, *_, C = x.shape

        T, Z, Y, X = self.grid_shape

        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(B, T * Z * Y * X + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([B, T * Z * Y * X, C])
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]))  # unshuffle
        x = x_.view([B, T * Z * Y * X, C])

        x += self.decoder_pos_embed[:, :, :]

        attn = self.decoder_blocks[0].attn
        requires_t_shape = bool(getattr(attn, "requires_t_shape", False))
        if requires_t_shape:
            x = x.view([B, T, Z * Y * X, C])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if requires_t_shape:
            x = x.view([B, Z * Y * X, -1])

        else:
            x = x[:, :, :]

        return x

    def loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        imgs: [B, C
        pred: [N, t*h*w, u*p*p*3]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """
        B, C, T, Z, Y, X = imgs.shape

        _imgs = torch.index_select(imgs, 2, torch.linspace(0, T - 1, self.pred_t_dim).long().to(imgs.device))

        target = self.patch_encode(_imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        return (loss * mask).sum() / mask.sum()  # mean loss on removed patches

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.75) -> Triple[torch.Tensor]:
        latent, mask, ids_restore = self.encode(x, mask_ratio)
        pred = self.decode(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.loss(x, pred, mask)
        return loss, pred, mask
