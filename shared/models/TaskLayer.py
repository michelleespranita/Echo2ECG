from omegaconf import DictConfig

import torch
import torch.nn as nn

activation_map = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "elu": nn.ELU(),
    "gelu": nn.GELU()
}

class TaskLayer(nn.Module):
    def __init__(
        self,
        downstream_task_cfg: DictConfig
    ):
        super(TaskLayer, self).__init__()

        self.modality = downstream_task_cfg.modality

        self.embed_dim = downstream_task_cfg.embed_dim

        self.task_type = downstream_task_cfg.task_type

        if self.task_type in ['regression', 'multilabel_classification', 'multiclass_classification', 'binary_classification']:
        
            if downstream_task_cfg.num_head_layers == 1:
                self.head = nn.Linear(self.embed_dim, downstream_task_cfg.num_classes)

            elif downstream_task_cfg.num_head_layers == 2:
                self.head = nn.Sequential(
                    nn.Linear(self.embed_dim, int(self.embed_dim // 2)),
                    activation_map[downstream_task_cfg.non_linearity] if downstream_task_cfg.non_linearity else nn.Identity(),
                    nn.Dropout(downstream_task_cfg.head_dropout) if downstream_task_cfg.head_dropout > 0 else nn.Identity(),
                    nn.Linear(int(self.embed_dim // 2), downstream_task_cfg.num_classes)
                )
            
            elif downstream_task_cfg.num_head_layers == 3:
                self.head = nn.Sequential(
                    nn.Linear(self.embed_dim, int(self.embed_dim // 2)),
                    activation_map[downstream_task_cfg.non_linearity] if downstream_task_cfg.non_linearity else nn.Identity(),
                    nn.Dropout(downstream_task_cfg.head_dropout) if downstream_task_cfg.head_dropout > 0 else nn.Identity(),
                    nn.Linear(int(self.embed_dim) // 2, int(self.embed_dim) // 4),
                    activation_map[downstream_task_cfg.non_linearity] if downstream_task_cfg.non_linearity else nn.Identity(),
                    nn.Dropout(downstream_task_cfg.head_dropout) if downstream_task_cfg.head_dropout > 0 else nn.Identity(),
                    nn.Linear(int(self.embed_dim // 4), downstream_task_cfg.num_classes)
                )
            else:
                print(f'Cannot implement {downstream_task_cfg.num_head_layers} layers for task head')
                raise NotImplementedError
                    
        print(f'Initialized a task head for {downstream_task_cfg.num_classes} classes')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)