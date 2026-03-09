from multiprocessing import Value
import torch
import torch.nn as nn
from omegaconf import DictConfig

from shared.models.token_aggregation.Pooling import AveragePooling, MaxPooling, AttentionPooling

class TokenAggregator(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        ecg_encoder_cfg: DictConfig,
        echo_encoder_cfg: DictConfig
    ):
        super(TokenAggregator, self).__init__()
        self.cfg = cfg
        self.ecg_encoder_cfg = ecg_encoder_cfg
        self.echo_encoder_cfg = echo_encoder_cfg

        self._init_layer_norms()
        self._init_ecg_token_aggregator()
    
    def _init_layer_norms(self):
        self.ecg_layernorm_global = nn.LayerNorm(self.ecg_encoder_cfg.embed_dim, dtype=torch.float32)

    def _init_ecg_token_aggregator(self):
        if self.cfg.model.token_aggregation.ecg.strategy == 'mean':
            self.ecg_token_aggregator = AveragePooling()
        elif self.cfg.model.token_aggregation.ecg.strategy == 'max':
            self.ecg_token_aggregator = MaxPooling()
        elif self.cfg.model.token_aggregation.ecg.strategy == 'attention':
            self.ecg_token_aggregator = AttentionPooling(
                embed_dim=self.ecg_encoder_cfg.embed_dim,
                num_heads=self.cfg.model.token_aggregation.ecg.num_heads
            )
        else:
            raise ValueError(
                f"Unknown ECG token aggregation strategy: {self.cfg.model.token_aggregation.ecg.strategy}"
            )
    
    def _aggregate_ecg_tokens(self, tokens: torch.Tensor):
        global_token = None
        all_tokens = None
        
        if self.cfg.model.token_aggregation.ecg.strategy == 'cls':
            all_tokens, global_token = tokens[:, 1:], tokens[:, 0]
        else:
            all_tokens = tokens[:, 1:] # the ECG encoder has a CLS token
            global_token = self.ecg_token_aggregator(all_tokens)

        global_token = self.ecg_layernorm_global(global_token)

        return {
            'ecg_global_token': global_token
        }
    
    def forward(self, tokens: torch.Tensor, modality: str):
        if modality == 'ecg':
            return self._aggregate_ecg_tokens(tokens)
        elif modality == 'echo':
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
