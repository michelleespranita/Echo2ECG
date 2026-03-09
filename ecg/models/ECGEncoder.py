# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE:  https://github.com/facebookresearch/mae?tab=readme-ov-file
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

import timm.models.vision_transformer
from timm.layers import trunc_normal_

from ecg.models.Components import Attention, Block, DyT, PatchEmbed
from util.pos_embed import get_1d_sincos_pos_embed


class ECGEncoder(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, img_size, patch_size=(1, 100), 
                 use_adapter=False, adapter_bottleneck_dim=64,
                 use_checkpoint=False,
                 **kwargs):
        # img_size: (input_channels, input_variates, time_steps) = (C, V, T)

        super(ECGEncoder, self).__init__(**kwargs)

        self.use_checkpoint = use_checkpoint

        embed_dim = kwargs['embed_dim']
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size[0], patch_size, embed_dim, flatten=False) # set flatten to False
        self.patch_size = patch_size
        self.patch_height = patch_size[0]
        self.patch_width = patch_size[1]

        self.grid_height = img_size[1] // self.patch_height # number of variates
        self.grid_width = img_size[2] // self.patch_width # number of time steps
        self.max_num_patches_x = self.grid_width
        num_patches = self.grid_height * self.grid_width

        assert embed_dim % 2 == 0
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.grid_width + 1, embed_dim // 2), requires_grad=False) # +1 cls embed

        self.pos_embed_y = nn.Embedding(self.grid_height + 1, embed_dim // 2, padding_idx=0) # +1 padding embed

        # split into pos_embed_x and pos_embed_y
        del self.pos_embed

        dpr = [x.item() for x in torch.linspace(0, kwargs['drop_path_rate'], kwargs['depth'])]  # stochastic depth decay rule
        i = 0
        for i in range(len(self.blocks)):
            self.blocks[i] = Block(
                dim=embed_dim,
                num_heads=kwargs['num_heads'],
                mlp_ratio=kwargs['mlp_ratio'],
                qkv_bias=kwargs['qkv_bias'],
                drop_path=dpr[i],
                norm_layer=kwargs['norm_layer'],
                use_adapter=use_adapter,
                adapter_bottleneck_dim=adapter_bottleneck_dim
            )
            self.blocks[i].attn = Attention(embed_dim, kwargs['num_heads'], qkv_bias=kwargs['qkv_bias'])
            i += 1

        del self.norm

        self.initialize_weights()

        # Placeholders
        self.mask = None
        self.token_probs = None

    def initialize_weights(self):
        # initialize learnable pos_embed for the vertical axis
        _pos_embed_y = torch.nn.Parameter(torch.randn(self.pos_embed_y.num_embeddings-1, 
                                                      self.pos_embed_y.embedding_dim) * .02)
        trunc_normal_(_pos_embed_y, std=.02)
        with torch.no_grad():
            self.pos_embed_y.weight[1:] = _pos_embed_y
                
        # initialize (and freeze) pos_embed for the horizontal axis by sin-cos embedding
        _pos_embed_x = get_1d_sincos_pos_embed(self.pos_embed_x.shape[-1], 
                                               self.pos_embed_x.shape[-2]-1, 
                                               cls_token=True)
        self.pos_embed_x.data.copy_(torch.from_numpy(_pos_embed_x).float().unsqueeze(0))

        # # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward_features(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, return_mask: bool = False):
        """
        x: [B, C, V, T], sequence

        Note: patch_size: (p, q) 
        """
        # embed patches
        # (B, D, C', T')
        x = self.patch_embed(x)

        # add pos embed X w/o cls token
        # (1, 1+T'_max, D/2)
        pos_embed_x = self.pos_embed_x
        # (1, 1+T'_max, D), padding left
        pos_embed_x = torch.nn.functional.pad(pos_embed_x, (x.shape[1]//2, 0), "constant", 0)
        # (1, D, 1, 1+T'_max)
        pos_embed_x_batch = torch.permute(pos_embed_x, (0, 2, 1)).unsqueeze(-2)
        # (1, D, 1, T')
        pos_embed_x_batch = pos_embed_x_batch[..., 1:x.shape[-1]+1]
        # (1, D, C', T')
        pos_embed_x_batch = pos_embed_x_batch.expand(-1, -1, x.shape[2], -1)

        # (B, D, C', T')
        x = x + pos_embed_x_batch

        # add pos embed Y
        # (B, C', T', D/2)
        B = x.shape[0]
        pos_embed_y = torch.LongTensor(
            torch.arange(self.grid_height).view(-1, 1).repeat(1, self.grid_width) + 1
        ).to(x.device).expand(B, -1, -1)
        pos_embed_y_batch = self.pos_embed_y(pos_embed_y)
        # (B, C', T', D), padding right
        pos_embed_y_batch = torch.nn.functional.pad(pos_embed_y_batch, (0, x.shape[1]//2), "constant", 0)
        # (B, D, C', T')
        pos_embed_y_batch = torch.permute(pos_embed_y_batch, (0, 3, 1, 2))
        
        # (B, D, C', T')
        x = x + pos_embed_y_batch

        # flatten
        # (B, N, D), with N=C'*T'
        x = x.flatten(2).transpose(1, 2)

        self.token_probs = None
        
        self.mask = mask

        # apply mask if available
        if mask is not None:
            B, _, C = x.shape
            x = x[~mask].reshape(B, -1, C)

        # append cls token
        # (1, 1, D)
        cls_token = self.cls_token + pos_embed_x[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # (B, 1+N, D)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.pos_drop(x)

        # add cls token to attn mask
        if attn_mask is not None:
            attn_mask = torch.cat((torch.ones(size=(attn_mask.shape[0], 1).bool(), device=x.device), attn_mask), dim=1) # (B, 1+N)

        # apply Transformer blocks
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(partial(blk, attn_mask=attn_mask), x, use_reentrant=False)
                # x = checkpoint.checkpoint(blk, x, use_reentrant=False)
        else:   
            for blk in self.blocks:
                x = blk(x, attn_mask=attn_mask)
                # x = blk(x)
        
        if return_mask:
            return x, mask
        else:
            return x # (B, 1+N, D)


def vit_baseDeep_patchX(**kwargs):
    model = ECGEncoder(
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        # norm_layer=DyT,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model

def vit_largeDeep_patchX(**kwargs):
    model = ECGEncoder(
        embed_dim=384, depth=18, num_heads=6, mlp_ratio=4, qkv_bias=True,
        # norm_layer=DyT,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model

def vit_hugeDeep_patchX(**kwargs):
    model = ECGEncoder(
        embed_dim=576, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        # norm_layer=DyT,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


models = {
    'vit_baseDeep_patchX': vit_baseDeep_patchX,
    'vit_largeDeep_patchX': vit_largeDeep_patchX,
    'vit_hugeDeep_patchX': vit_hugeDeep_patchX
}