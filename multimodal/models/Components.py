import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import drop_path, to_2tuple

class Adapter(nn.Module):
    def __init__(self, dim, bottleneck_dim=64):
        super(Adapter, self).__init__()
        self.down_proj = nn.Linear(dim, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, dim)
    
    def forward(self, x):
        return self.up_proj(self.activation(self.down_proj(x)))

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_map = None

    def forward(self, x, attn_mask=None):
        B, N, D = x.shape # D = embed_dim
        qkv = self.qkv(x).reshape(B, N, 3, D).permute(2, 0, 1, 3) # (QKV, B, N, D)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) (B, N, D)

        # attn_mask: 1/True (ignore), 0/False (attend)
        attn, attn_weights = self.mha(q, k, v, key_padding_mask=attn_mask)
        self.attn_map = attn_weights

        x = self.proj(attn)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, use_adapter=False, adapter_bottleneck_dim=64):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        if use_adapter:
            self.adapter1 = Adapter(dim, adapter_bottleneck_dim)
            self.adapter2 = Adapter(dim, adapter_bottleneck_dim)
        else:
            self.adapter1, self.adapter2 = None, None

    def forward(self, x, attn_mask=None, return_attn=False):
        if self.gamma_1 is None:
            if return_attn:
                _x, attn = self.attn(self.norm1(x), attn_mask=attn_mask, return_attn=True)
                x = x + self.drop_path(_x)
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), attn_mask=attn_mask))

            if self.adapter1:
                x = x + self.drop_path(self.adapter1(x))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            if self.adapter2:
                x = x + self.drop_path(self.adapter2(x))

            if return_attn:
                return x, attn

        else:
            if return_attn:
                _x, attn = self.attn(self.norm1(x), attn_mask=attn_mask, return_attn=True)
                x = x + self.drop_path(self.gamma_1 * _x)
            else:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), attn_mask=attn_mask))

            if self.adapter1:
                x = x + self.drop_path(self.adapter1(x))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            if self.adapter2:
                x = x + self.drop_path(self.adapter2(x))
            
            if return_attn:
                return x, attn
                
        return x



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x