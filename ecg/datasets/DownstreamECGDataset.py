import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig
import pandas as pd
import os

from ecg.datasets.ECGAugmentations import ECGAugmentations
    
class DownstreamECGDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig, # cfg
        mode: str,
        target: str = None,
        first_n_samples: int = None
    ):
        super(DownstreamECGDataset, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.target = target
        self.first_n_samples = first_n_samples
        self.task_type = cfg.downstream_task_ecg.task_type

        data_path = cfg.downstream_task_ecg.paths[f'data_{self.mode}']
        label_path = cfg.downstream_task_ecg.paths[f'labels_{self.mode}']
        if data_path.endswith('.pt'):
            self.data = torch.load(data_path)
            self.data_ext = 'pt'
        elif data_path.endswith('.csv'):
            filelist = pd.read_csv(cfg.downstream_task_ecg.paths[f'data_{self.mode}'])
            self.paths = list(filelist['filename'])
            self.data_ext = 'csv'
        
        self.labels = torch.load(label_path)

        if self.first_n_samples:
            if hasattr(self, 'data'):
                self.data = self.data[0:self.first_n_samples]
            elif hasattr(self, 'paths'):
                self.paths = self.paths[0:self.first_n_samples]
            self.labels = self.labels[0:self.first_n_samples]

        if self.task_type == 'regression':
            if 'target_list' in cfg.downstream_task_ecg:
                self.lower_bound = cfg.downstream_task_ecg.target_list[target].lower_bound
                self.upper_bound = cfg.downstream_task_ecg.target_list[target].upper_bound
                self.num_classes = self.upper_bound - self.lower_bound
            else:
                self.num_classes = len(self.labels[0])
        elif self.task_type in ['binary_classification', 'multiclass_classification', 'multilabel_classification']:
            self.num_classes = len(self.labels[0])
    
        self.transform_and_augment_fn = ECGAugmentations(cfg.downstream_task_ecg, sig_len=cfg.downstream_task_ecg.time_steps)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.data_ext == 'pt':
            ecg = self.data[idx][1].float() # self.data[idx]: ('ecg', torch.Tensor)
        elif self.data_ext == 'csv':
            ecg = torch.load(self.paths[idx]).float()
        
        if self.task_type == 'regression':
            if 'target_list' in self.cfg.downstream_task_ecg:
                label = self.labels[idx][..., self.lower_bound:self.upper_bound] # (nb_classes,)
            else:
                label = self.labels[idx].squeeze()
        elif self.task_type == 'binary_classification':
            label = self.labels[idx].argmax(dim=-1) # e.g. tensor(0)
        elif self.task_type in ['multiclass_classification', 'multilabel_classification']:
            label = self.labels[idx]

        ecg = ecg.unsqueeze(0) # add channel dimension
        if self.mode == 'train':
            ecg = self.transform_and_augment_fn(ecg, use_augmentations=True)
        else:
            ecg = self.transform_and_augment_fn(ecg, use_augmentations=False)

        ret =  {
            'ecg': ecg, # (C, V, T)
            'label': label
        }

        if self.data_ext == 'csv':
            ret.update({
                'filename': os.path.basename(self.paths[idx])
            })

        return ret
