# https://github.com/lezhang7/SAIL

import torch
from torch import nn
from typing import Optional, Callable
import torch.nn.functional as F


class StarMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width_factor: int ,
        intermediate_dim: Optional[int] = None,
        activation: Optional[Callable] = None,
    ):
        super().__init__()
        self.f1 = nn.Linear(input_dim, width_factor * input_dim)
        self.f2 = nn.Linear(input_dim, width_factor * input_dim)
        self.act = activation  # 传入的激活函数
        self.g = nn.Linear(width_factor * input_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.f1(hidden_states), self.f2(hidden_states)
        x1 = torch.clamp(x1, min=-1e3, max=1e3)
        x2 = torch.clamp(x2, min=-1e3, max=1e3)
        if self.act:
            x = self.act(x1) * x2
        else:
            x = x1 * x2
        x = self.g(x)

        assert not torch.isnan(x).any(), "Output contains NaN"
        assert not torch.isinf(x).any(), "Output contains infinite values"

        return x


class ShareLockMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.2):
        super(ShareLockMLP, self).__init__()
        
        # Define the layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)





# v2
class SwiGLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SwiGLU, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear1(x) * F.silu(self.linear2(x))

class SiglipMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        intermediate_dim: Optional[int] = None,
    ):
        super().__init__()
        intermediate_dim = intermediate_dim if intermediate_dim is not None else 4*input_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, output_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return  self.proj(hidden_states)