from torch.utils.data import Dataset
import torch
import os
import pandas as pd
from omegaconf import DictConfig

from ecg.datasets.ECGAugmentations import ECGAugmentations

class ECGEchoDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        mode: str
    ):
        super(ECGEchoDataset, self).__init__()
        self.cfg = cfg
        self.mode = mode

        # load data
        pairs = pd.read_csv(cfg.dataset.paths[f'data_{self.mode}'])
        self.echo_paths = list(pairs['echo_filepath'])
        self.ecg_paths = list(pairs['ecg_filepath'])
        if cfg.dataset.echo.use_precomputed_embeds:
            self.echo_embeds = torch.load(os.path.join(cfg.dataset.echo.embeddings_filepath),
                                          weights_only=False, map_location=torch.device('cpu'))
            self.echo_embeds_inds = list(pairs['echo_embed_idx'])
        if cfg.dataset.ecg.use_precomputed_embeds:
            self.ecg_embeds = torch.load(os.path.join(cfg.dataset.ecg.embeddings_filepath),
                                          weights_only=False, map_location=torch.device('cpu'))
            self.ecg_embeds_inds = list(pairs['ecg_embed_idx'])
        
        self._init_augmentations()
        self.use_augmentations = True if self.mode == 'train' else False
    
    def _init_augmentations(self):
        self.ecg_transform_and_augment_fn = ECGAugmentations(self.cfg.dataset.ecg, sig_len=self.cfg.model.ecg.time_steps)

    def __len__(self):
        return len(self.echo_paths)

    def __getitem__(self, idx):
        ret = dict()

        # load echo
        if not self.cfg.dataset.echo.use_precomputed_embeds:
            raise NotImplementedError
        
        else:
            echo_path = self.echo_paths[idx]
            embed_idx = self.echo_embeds_inds[idx]
            echo_augm_embed = self.echo_embeds[embed_idx]
            ret.update({
                'echo_embed': echo_augm_embed # (embed_dim)
            })
            
        # load ecg
        if not self.cfg.dataset.ecg.use_precomputed_embeds:
            ecg_path = self.ecg_paths[idx]
            ecg = torch.load(ecg_path).unsqueeze(0) # (C, V, T)

            ecg_augm = self.ecg_transform_and_augment_fn(ecg, use_augmentations=self.use_augmentations)
        
            ret.update({
                'ecg': ecg_augm, # (C, V, T)
            })
            
        else:
            ecg_path = self.ecg_paths[idx]
            embed_idx = self.ecg_embeds_inds[idx]
            ecg_augm_embed = self.ecg_embeds[embed_idx]
            ret.update({
                'ecg_embed': ecg_augm_embed # (embed_dim)
            })

        ret.update({
            'echo_filename': echo_path,
            'ecg_filename': ecg_path
        })

        return ret