from omegaconf import DictConfig
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.models.token_aggregation.Pooling import AveragePooling, AttentionPooling, CLSPooling

class EchoViewAggregator(nn.Module):
    """
    Combines view-type information with view embeddings and performs aggregation.
    """
    def __init__(
        self,
        cfg: DictConfig,
        echo_encoder_cfg: DictConfig
    ):
        super(EchoViewAggregator, self).__init__()

        self.cfg = cfg
        self.echo_encoder_cfg = echo_encoder_cfg
        self.embed_dim = echo_encoder_cfg.embed_dim
        self.aggregation_strategy = self.cfg.model.echo.view_aggregation.strategy
        assert self.aggregation_strategy in ['att', 'mean', 'cls']

        if self.aggregation_strategy == 'att':
            self.aggregator = AttentionPooling(
                self.embed_dim,
                num_heads=self.cfg.model.echo.view_aggregation.num_heads
            )
        
        elif self.aggregation_strategy == 'mean':
            self.aggregator = AveragePooling()
        
        elif self.aggregation_strategy == 'cls':
            self.aggregator = CLSPooling(
                self.embed_dim,
                num_heads=self.cfg.model.echo.view_aggregation.num_heads,
                dropout=self.cfg.model.echo.view_aggregation.dropout,
                num_layers=self.cfg.model.echo.view_aggregation.num_layers
            )

    def forward(self, view_features: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Generates the study embedding by aggregating view features.

        Args:
            view_features (torch.Tensor): Image features from the view encoder.
                                          Shape: (B, max_num_views, D)
            mask (torch.Tensor, optional): Boolean mask where True indicates padding/missing views.
                                           Shape: (B, max_num_views)
                                           If None, all views are treated as present.
        
        Returns:
            torch.Tensor: The final study embedding (B, study_embedding_dim)
        """

        study_embedding = self.aggregator(view_features, mask)
        
        return study_embedding