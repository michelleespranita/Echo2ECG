from torch.utils.data import Dataset
import torch
import os
import pandas as pd
from omegaconf import DictConfig

from ecg.datasets.ECGAugmentations import ECGAugmentations

class ECGEchoStudyDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        mode: str
    ):
        super(ECGEchoStudyDataset, self).__init__()
        self.cfg = cfg
        self.mode = mode

        # load dataset
        pairs = pd.read_csv(cfg.dataset.paths[f'data_{self.mode}'])
        self.grouped_studies = pairs.groupby(['ecg_study_id', 'echo_study_id'])
        self.study_ids = list(self.grouped_studies.groups.keys())
        if cfg.dataset.echo.use_precomputed_embeds:
            self.echo_embeds = torch.load(os.path.join(cfg.dataset.echo.embeddings_filepath),
                                          weights_only=False, map_location=torch.device('cpu'))
            self.echo_embed_dim = self.echo_embeds.shape[-1]
        else:
            raise NotImplementedError
        if cfg.dataset.ecg.use_precomputed_embeds:
            self.ecg_embeds = torch.load(os.path.join(cfg.dataset.ecg.embeddings_filepath),
                                          weights_only=False, map_location=torch.device('cpu'))
        
        self.echo_max_n_views = self.cfg.dataset.echo.max_n_views
        
        self._init_augmentations()
        self.use_augmentations = True if self.mode == 'train' else False
    
    def _init_augmentations(self):
        self.ecg_transform_and_augment_fn = ECGAugmentations(self.cfg.dataset.ecg, sig_len=self.cfg.model.ecg.time_steps)

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        ret = dict()

        # get the current study
        study_id = self.study_ids[idx]
        ecg_study_id, echo_study_id = study_id
        study_data = self.grouped_studies.get_group(study_id)

        # load ECG (there is only 1 ECG per study)
        ecg_path = study_data['ecg_filepath'].unique().tolist()[0]

        if not self.cfg.dataset.ecg.use_precomputed_embeds:
            ecg = torch.load(ecg_path).unsqueeze(0)
            ecg_augm = self.ecg_transform_and_augment_fn(ecg, use_augmentations=self.use_augmentations)
            ret.update({
                'ecg': ecg_augm # (C, V, T)
            })
        
        else:
            raise NotImplementedError
        
        # load all echos (there are multiple echos per study)
        echo_embeds = []
        for _, row in study_data.iterrows():
            embed_idx = row['echo_embed_idx']
            echo_embed = self.echo_embeds[embed_idx]
            echo_embeds.append(echo_embed)
        
        echo_embeds = torch.stack(echo_embeds) 

        n_views = min(len(echo_embeds), self.echo_max_n_views)
        padded_echo_embeds = torch.zeros((self.echo_max_n_views, self.echo_embed_dim), dtype=torch.float32)
        padded_echo_embeds[:n_views] = echo_embeds[:n_views]

        ret.update({
            'echo_embed': padded_echo_embeds # (max_n_views, D),
        })

        echo_view_masks = torch.ones(self.echo_max_n_views, dtype=bool)
        echo_view_masks[:n_views] = False # 0: view available, 1: view unavailable

        ret.update({
            'ecg_study_id': ecg_study_id,
            'echo_study_id': echo_study_id,
            'echo_n_views': n_views,
            'attn_mask': echo_view_masks
        })

        return ret