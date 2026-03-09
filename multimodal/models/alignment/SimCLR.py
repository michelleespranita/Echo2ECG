from typing import Optional

import torch
import torch.nn as nn
from lightly.models.modules import SimCLRProjectionHead

class SimCLRProjectionLayer(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        target_dimension: int = 1024,
        cast_dtype: Optional[torch.dtype] = None,
        width_factor: int = 8,
    ):
        super(SimCLRProjectionLayer, self).__init__()
        assert input_dimension is not None
        self.cast_dtype = cast_dtype

        self.mapping_network = SimCLRProjectionHead(
            input_dimension, input_dimension * width_factor, target_dimension
        )
        self.layer_norm = nn.LayerNorm(input_dimension)

    def forward(self, tokens: torch.Tensor):

        tokens = tokens.to(self.cast_dtype)
        tokens = self.layer_norm(tokens)
        tokens = self.mapping_network(tokens)

        return tokens