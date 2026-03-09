# https://github.com/lezhang7/SAIL
# with modifications

import torch
from typing import Optional
import torch.nn as nn
import torch
from multimodal.models.alignment.MLP import StarMLP, SiglipMLP, ShareLockMLP
from functools import partial
import numpy as np

class SAIL(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        target_dimension: int = 1024,
        linear_type: str = "star",
        cast_dtype: Optional[torch.dtype] = None,
        logit_scale: float = 10.0,
        logit_bias: float = -10.0,
        width_factor: int = 8,
    ):
        super(SAIL, self).__init__()
        assert input_dimension is not None

        self.cast_dtype = cast_dtype
        self.linear_type = linear_type
        if linear_type == "star":
            LinearLayer = partial(StarMLP, width_factor=width_factor, activation=nn.ReLU6())
        elif linear_type == "mlp":
            LinearLayer = SiglipMLP
        else:
            LinearLayer = nn.Linear

        self.mapping_network = LinearLayer(
            input_dimension, target_dimension
        )
        self.layer_norm = nn.LayerNorm(input_dimension)

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))
        self._initialize_weights(logit_scale, logit_bias)

    def _initialize_weights(self, scale: float, bias: float):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

        # Initialize logit_scale and logit_bias
        logit_scale_init = torch.log(torch.tensor(scale))                           
        self.logit_scale.data.fill_(logit_scale_init)
        self.logit_bias.data.fill_(torch.tensor(bias))
    
    @property
    def get_logit_scale(self):
        return self.logit_scale.exp()
    
    @property
    def get_logit_bias(self):
        return self.logit_bias
     
    def forward(self, tokens: torch.Tensor):
        tokens = tokens.to(self.cast_dtype)
        tokens = self.layer_norm(tokens)
        tokens = self.mapping_network(tokens)

        return tokens

class ShareLock(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        target_dimension: int = None,
        linear_type: str = None,
        cast_dtype: Optional[torch.dtype] = None,
        logit_scale: float = np.log(1 / 0.07),
        logit_bias: float = -10.0,
        *args,
        **kwargs,
    ):
        super(ShareLock, self).__init__()
        self.cast_dtype = cast_dtype
        # no ecg alignment layer overhead
        target_dimension = input_dimension 
        hidden_dim = 4096
        self.mapping_network = ShareLockMLP(
            input_dimension, hidden_dim, target_dimension
        )
     
        self.layer_norm = nn.LayerNorm(input_dimension)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_bias = nn.Parameter(torch.ones([]) * -10.0)
        self._initialize_weights(logit_scale, logit_bias)

    def _initialize_weights(self, scale: float, bias: float):
    
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)                     

    @property
    def get_logit_scale(self):
        return self.logit_scale.exp()
    
    @property
    def get_logit_bias(self):
        return self.logit_bias

    def forward(self, tokens: torch.Tensor):
    
        tokens = tokens.to(self.cast_dtype)
        tokens = self.layer_norm(tokens)
        tokens = self.mapping_network(tokens)

        return tokens