from __future__ import annotations

import math
from typing import Callable, TYPE_CHECKING


import torch
import torch.nn as nn

from .utils import Triple, einsum_transpose, Quadruple
from .layers import block_list, PatchEmbed3d


class MaskedAutoencoder3d(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    if TYPE_CHECKING:
        to_patches: Callable[[torch.Tensor], torch.Tensor]
        from_patches: Callable[[torch.Tensor], torch.Tensor]
        __call__: Callable[[torch.Tensor], Triple[torch.Tensor]]

    def __init__(
        self,
        input_shape: Triple[int],
        patch_shape: Triple[int],
        batch_size: int,
        in_chans: int,
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
        sep_pos_embed: bool = False,
        trunc_init: bool = False,
        cls_embed: bool = False,
        pred_t_dim: int | None = None,
    ) -> None:
        super().__init__()
        B, C = batch_size, in_chans
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
        # print(z, pred_t_dim)
        self.t_pred_patch_size = z = z * pred_t_dim // Z
        # print(z)

        self.patch_embed = PatchEmbed3d(input_shape, patch_shape, in_chans, embed_dim)
        self.to_patches = einsum_transpose(
            (B, C, z, Pz, y, Py, x, Px),  # unsqueeze
            "BCzZyYxX->BzyxZYXC",
            (B, grid_size, math.prod(patch_shape) * C),  # reshape
        )
        self.from_patches = einsum_transpose(
            (B, z, y, x, Pz, Py, Px, C),  # unsqueeze
            "BzyxZYXC->BCzZyYxX",
            (B, C, Z, Y, X),  # reshape
        )

        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(torch.zeros(1, Y * X, embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, Z, embed_dim))
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = grid_size + 1
            else:
                _num_patches = grid_size

            self.pos_embed = nn.Parameter(torch.zeros(1, _num_patches, embed_dim))

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

        if sep_pos_embed:
            self.decoder_pos_embed_spatial = nn.Parameter(torch.zeros(1, Y * X, decoder_embed_dim))
            self.decoder_pos_embed_temporal = nn.Parameter(torch.zeros(1, Z, decoder_embed_dim))
            if self.cls_embed:
                self.decoder_pos_embed_class = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        else:
            if self.cls_embed:
                _num_patches = grid_size + 1
            else:
                _num_patches = grid_size

            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, _num_patches, decoder_embed_dim))

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
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            math.prod(patch_shape) * in_chans,
            bias=True,
        )

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        print("model initialized")

    def initialize_weights(self) -> None:
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
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

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float) -> Triple[torch.Tensor]:
        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape

        x = x.reshape(N, T * L, C)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        x = x.view(N, -1, C)
        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(1, self.input_size[0], 1) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat([self.pos_embed_class.expand(pos_embed.shape[0], -1, -1), pos_embed], 1)
        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat([self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1), pos_embed], 1)
        x = x.view([N, -1, C]) + pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
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
        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal, self.input_size[1] * self.input_size[2], dim=1
            )
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [self.decoder_pos_embed_class.expand(decoder_pos_embed.shape[0], -1, -1), decoder_pos_embed], 1
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        x = x + decoder_pos_embed

        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
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

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
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
        target = self.to_patches(_imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75) -> Triple[torch.Tensor]:
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.decode(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.loss(imgs, pred, mask)
        return loss, pred, mask
