from typing import Optional
from omegaconf import DictConfig

import torch.nn as nn

from multimodal.losses.CLIPLoss import CLIPLoss

def init_loss_fn(task_type: str, downstream_task_cfg: Optional[DictConfig] = None) -> nn.Module:
    if 'regression' in task_type:
        return nn.MSELoss()
    elif 'binary_classification' in task_type:
        return nn.BCEWithLogitsLoss()
    elif 'multiclass_classification' in task_type:
        return nn.CrossEntropyLoss()
    elif 'multilabel_classification' in task_type:
        return nn.BCEWithLogitsLoss()

def init_contrastive_loss_fn(cfg: DictConfig) -> nn.Module:
    return CLIPLoss(
        cfg.train.clip_loss.temperature,
        cfg.train.clip_loss.lambda_0,
        cfg.train.clip_loss.learnable_temperature
    )