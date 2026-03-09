import os
import pandas as pd
import numpy as np
from omegaconf import DictConfig
import torch
import pytorchvideo.transforms as TV
import torchvision.transforms as T
from torch.utils.data import Dataset

from echo.datasets.EchoAugmentations import MinMaxNormalization, ImageNetNormalization
from echo.utils.load_video import loadvideo_decord
from util.echoprime import ECHOPRIME_MEAN, ECHOPRIME_STD, crop_and_scale

class EchoStudyDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        echo_study_df: pd.DataFrame
    ):
        self.cfg = cfg
        self.echo_study_df = echo_study_df # columns: echo_study_id, echo_filepath, echo_view
        self.studies = self.echo_study_df['echo_study_id'].unique().tolist()
        self.echo_max_n_views = cfg.dataset.echo.max_n_views

        self.img_size = cfg.dataset.echo.img_size
        self.num_frames = cfg.dataset.echo.num_frames
        self.sampling_rate = cfg.dataset.echo.sampling_rate
        self.num_channels = 3

        if cfg.dataset.echo.use_precomputed_embeds:
            self.echo_embeds = torch.load(os.path.join(cfg.dataset.echo.embeddings_filepath),
                                          weights_only=False, map_location=torch.device('cpu'))
            self.echo_embed_dim = self.echo_embeds.shape[-1]
        else:
            raise NotImplementedError

        transforms_to_apply = []
        transforms_to_apply.append(T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR))
        transforms_to_apply.append(MinMaxNormalization())
        transforms_to_apply.append(ImageNetNormalization())
        self.transform_fn = TV.ApplyTransformToKey(
            key="video",
            transform=T.Compose(transforms_to_apply)
        )

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        ret = dict()

        echo_study_id = self.studies[idx]
        study_data = self.echo_study_df[self.echo_study_df['echo_study_id'] == echo_study_id]

        if not self.cfg.dataset.echo.use_precomputed_embeds:
            echo_paths = study_data['echo_filepath'].tolist()
            echos = []
            for echo_path in echo_paths:
                # same logic as the original code, but faster
                out = loadvideo_decord(echo_path, 'val', self.cfg.dataset.echo.sampling_rate, self.cfg.dataset.echo.num_frames, repeat_or_pad='pad')
                echo = out['video']
                echo = np.array(echo)
                echo_augm = np.zeros((len(echo), self.cfg.dataset.echo.img_size, self.cfg.dataset.echo.img_size, self.num_channels))
                for i in range(len(echo_augm)):
                    echo_augm[i] = crop_and_scale(echo[i])

                echo_augm = torch.from_numpy(echo_augm).float()
                echo_augm = echo_augm.permute(3, 0, 1, 2)
                # normalize
                echo_augm.sub_(ECHOPRIME_MEAN).div_(ECHOPRIME_STD)

                echo = echo_augm

                echos.append(echo)
            
            echos = torch.stack(echos) # (num_views, C, T, H, W)
            _, C, T, H, W = echos.shape
            n_views = min(len(echos), self.echo_max_n_views)
            padded_echos = torch.zeros((self.echo_max_n_views, C, T, H, W), dtype=torch.float32)
            padded_echos[:n_views] = echos[:n_views]

            ret.update({
                'echo': padded_echos
            })
        
        else:
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
            'echo_study_id': echo_study_id,
            'attn_mask': echo_view_masks
        })

        return ret