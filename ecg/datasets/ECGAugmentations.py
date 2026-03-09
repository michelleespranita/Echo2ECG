from omegaconf import DictConfig

import torch
import torchvision.transforms as T

from ecg.utils.augmentations import CropResizing, FTSurrogate, Jitter, Rescaling, TimeFlip, SignFlip

class ECGAugmentations():
    def __init__(self,
        cfg: DictConfig,
        sig_len: int = 1008
    ):
        super(ECGAugmentations, self).__init__()

        self.cfg = cfg
        self.sig_len = sig_len
        self.apply_augmentations = cfg.apply_augmentations

        self.crop_resize_fn_for_transform = T.Compose([
            CropResizing(
                fixed_crop_len=self.sig_len,
                start_idx=0,
                resize=False
            )
        ])
        self.crop_resize_fn_for_augment = T.Compose([
            CropResizing(
                fixed_resize_len=self.sig_len,
                lower_bnd=self.cfg.transforms.crop_lower_bnd,
                upper_bnd=self.cfg.transforms.crop_upper_bnd,
                resize=True
            )
        ])

        self.augmentations_to_apply = []
        
        if self.apply_augmentations:
            self._add_ft_surrogate_fn()
            self._add_jitter_fn()
            self._add_rescaling_fn()
            self._add_time_flip_fn()
            self._add_sign_flip_fn()
            
        self.augment_fn = T.Compose(self.augmentations_to_apply)
    
    def _add_ft_surrogate_fn(self):
        self.augmentations_to_apply.append(FTSurrogate(phase_noise_magnitude=self.cfg.augmentations.ft_surr_phase_noise, prob=0.5))
    
    def _add_jitter_fn(self):
        self.augmentations_to_apply.append(Jitter(sigma=self.cfg.augmentations.jitter_sigma))
    
    def _add_rescaling_fn(self):
        self.augmentations_to_apply.append(Rescaling(sigma=self.cfg.augmentations.rescaling_sigma))
    
    def _add_time_flip_fn(self):
        self.augmentations_to_apply.append(TimeFlip(prob=self.cfg.augmentations.time_flip_prob))
    
    def _add_sign_flip_fn(self):
        self.augmentations_to_apply.append(SignFlip(prob=self.cfg.augmentations.sign_flip_prob))

    def __call__(self, ecg: torch.Tensor, use_augmentations: bool = True) -> torch.Tensor:
        if use_augmentations:
            ecg = self.crop_resize_fn_for_augment(ecg)
            ecg = self.augment_fn(ecg)
        else:
            ecg = self.crop_resize_fn_for_transform(ecg)
        return ecg