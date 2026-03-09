import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Optional

from shared.models.token_aggregation.Pooling import AveragePooling, MaxPooling, AttentionPooling

class TokenAggregator(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        echo_encoder_cfg: DictConfig
    ):
        super(TokenAggregator, self).__init__()
        self.cfg = cfg
        self.echo_encoder_cfg = echo_encoder_cfg

        self._init_layer_norms()
        self._init_echo_token_aggregator()
    
    def _init_layer_norms(self):
        # self.echo_layernorm_local = nn.LayerNorm(self.echo_encoder_cfg.embed_dim, dtype=torch.float32)
        self.echo_layernorm_global = nn.LayerNorm(self.echo_encoder_cfg.embed_dim, dtype=torch.float32)

    def _init_echo_token_aggregator(self):
        if self.cfg.model.token_aggregation.echo.strategy == 'mean':
            self.echo_token_aggregator = AveragePooling()
        elif self.cfg.model.token_aggregation.echo.strategy == 'max':
            self.echo_token_aggregator = MaxPooling()
        elif self.cfg.model.token_aggregation.echo.strategy == 'attention':
            self.echo_token_aggregator = AttentionPooling(
                embed_dim=self.echo_encoder_cfg.embed_dim,
                num_heads=self.cfg.model.token_aggregation.echo.num_heads
            )
        else:
            raise ValueError(
                f"Unknown echo token aggregation strategy: {self.cfg.model.token_aggregation.echo.strategy}"
            )
    
    def _aggregate_echo_tokens(self, tokens: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):        
        if self.cfg.model.token_aggregation.echo.strategy == 'cls':
            all_tokens, global_token = tokens[:, 1:], tokens[:, 0]
        else:
            all_tokens = tokens
            global_token = self.echo_token_aggregator(all_tokens, attn_mask)

        # all_tokens = self.echo_layernorm_local(all_tokens)
        global_token = self.echo_layernorm_global(global_token)

        return {
            # 'echo_all_tokens': all_tokens,
            'echo_global_token': global_token
        }
    
    def forward(self, tokens: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self._aggregate_echo_tokens(tokens, attn_mask)
    
