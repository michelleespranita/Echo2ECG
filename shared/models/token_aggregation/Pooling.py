from typing import Optional

import torch
import torch.nn as nn

# Class interface

class AveragePooling(nn.Module):
    """
    Average Pooling class for global token
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if attn_mask is not None:
            inv_attn_mask = 1 - attn_mask
            inv_attn_mask = inv_attn_mask.unsqueeze(-1) # (B, N, 1)
            x = (x * inv_attn_mask).sum(dim=1) / inv_attn_mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return x

class MaxPooling(nn.Module):
    """
    Max Pooling class for global token
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if attn_mask is not None:
            mask_bool = attn_mask.bool().unsqueeze(-1)
            x = x.masked_fill(mask_bool, float('-inf'))
        x, _ = x.max(dim=1)
        return x
    
class AttentionPooling(nn.Module):
    """
    Attention Pooling class for global token
    """
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [batch, seq_len, embed_dim]
            mask: [batch, seq_len] (float tensor, 1 where padding)
        """
        # 1. Calculate the Query (mean of ONLY valid views)
        if attn_mask is not None:
            # Create mask where 0 is padding, 1 is valid
            inv_attn_mask = 1 - attn_mask
            valid_float = inv_attn_mask.unsqueeze(-1) # [B, seq_len, 1]
            
            # Sum up valid views and divide by the count of valid views
            q = (x * valid_float).sum(dim=1, keepdim=True)
            num_valid = valid_float.sum(dim=1, keepdim=True).clamp(min=1)
            q = q / num_valid # [B, 1, D]
            mha_key_mask = attn_mask.bool()
        else:
            q = x.mean(dim=1, keepdim=True)
            mha_key_mask = None

        # 2. Key and Value are the original sequence
        k = x
        v = x

        # 3. Perform MHA with key_padding_mask
        # key_padding_mask: [B, seq_len]. True values are ignored.
        attn_output, attn_weights = self.mha(q, k, v, key_padding_mask=mha_key_mask)

        # 4. Remove the sequence dimension
        x = attn_output.squeeze(dim=1) # [B, embed_dim]
        
        return x

class CLSPooling(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        num_layers: int = 1
    ):
        super().__init__()
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mha": nn.MultiheadAttention(
                    embed_dim=embed_dim, 
                    num_heads=num_heads, 
                    dropout=dropout, 
                    batch_first=True
                ),
                "ln1": nn.LayerNorm(embed_dim),
                "ln2": nn.LayerNorm(embed_dim),
                "ffn": nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout)
                )
            }) for _ in range(self.num_layers)
        ])
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: Input tensor of shape (B, seq_len, D)
            attn_mask: Float tensor of shape (B, seq_len). (1 where padding)
        Returns:
            Global token of shape (B, D)
        """
        B, seq_len, D = x.shape
        
        # Expand CLS token to (B, 1, D)
        cls_token = self.cls_token.expand(B, -1, -1)

        mha_key_mask = attn_mask.bool() if attn_mask is not None else None
        
        for layer in self.layers:
            attn_out, _ = layer['mha'](
                query=cls_token,
                key=x,
                value=x,
                key_padding_mask=mha_key_mask
            )
            cls_token = layer['ln1'](cls_token + attn_out)
            ffn_out = layer['ffn'](cls_token)
            cls_token = layer['ln2'](cls_token + ffn_out)
        
        return cls_token.squeeze(1)

# Function inferface

def avg_pool(x, attn_mask=None):
    return AveragePooling()(x, attn_mask)