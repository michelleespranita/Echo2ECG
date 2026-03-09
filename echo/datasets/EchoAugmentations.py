from typing import Tuple, Optional
from omegaconf import DictConfig
import random

import torch
import pytorchvideo.transforms as TV
import torchvision.transforms as T

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class MinMaxNormalization(object):
    def __call__(self, img):
        """Convert image from [0, 255] to [0, 1]."""
        img = img.float()
        return (img - 0) / 255

class ImageNetNormalization:
    def __init__(self):
        self.mean = IMAGENET_DEFAULT_MEAN
        self.std = IMAGENET_DEFAULT_STD
        self.transform = TV.Normalize(mean=self.mean, std=self.std)

    def __call__(self, x):
        """Apply ImageNet normalization to a tensor or video tensor."""
        return self.transform(x)

class EchoAugmentations():
    def __init__(
        self,
        cfg: DictConfig,
        crop_size: int,
        task_type: str
    ):
        super(EchoAugmentations, self).__init__()

        self.cfg = cfg
        self.crop_size = crop_size
        self.task_type = task_type
        self.apply_augmentations = cfg.apply_augmentations

        self.transforms_to_apply = []
        self.transforms_to_apply_for_segmap = []

        self._add_resize_fn()
        self._add_minmax_norm_fn()
        self._add_imagenet_norm_fn()

        self.transform_fn = TV.ApplyTransformToKey(
            key="video",
            transform=T.Compose(self.transforms_to_apply)
        )

        if self.task_type == 'segmentation':
            self.transform_fn_for_segmap = TV.ApplyTransformToKey(
                key="seg",
                transform=T.Compose(self.transforms_to_apply_for_segmap)
            )

        self.augmentations_to_apply = []

        if self.apply_augmentations:
            if self.task_type == 'segmentation':
                self._add_synchronized_flip_fn()
                self.augment_fn = T.Compose(self.augmentations_to_apply)
            else:
                self._add_random_horizontal_flip_fn()
                self._add_random_vertical_flip_fn()
                # self._add_random_rotate_fn()
                # self._add_random_crop_fn()

                self.augment_fn = TV.ApplyTransformToKey(
                    key="video",
                    transform=T.Compose(self.augmentations_to_apply)
                )
        
    def _add_resize_fn(self):
        self.transforms_to_apply.append(T.Resize(self.crop_size, interpolation=T.InterpolationMode.BILINEAR))

        if self.task_type == 'segmentation':
            self.transforms_to_apply_for_segmap.append(T.Resize(self.crop_size, interpolation=T.InterpolationMode.NEAREST))
    
    def _add_minmax_norm_fn(self):
        self.transforms_to_apply.append(MinMaxNormalization())

    def _add_imagenet_norm_fn(self):
        self.transforms_to_apply.append(TV.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))
    
    def _add_random_horizontal_flip_fn(self):
        self.augmentations_to_apply.append(T.RandomHorizontalFlip(p=self.cfg.augmentations.random_horizontal_flip))
    
    def _add_random_vertical_flip_fn(self):
        self.augmentations_to_apply.append(T.RandomVerticalFlip(p=self.cfg.augmentations.random_vertical_flip))

    def _add_random_rotate_fn(self):
        self.augmentations_to_apply.append(T.RandomRotation(
            degrees=self.cfg.augmentations.random_rotation,
            interpolation=T.InterpolationMode.BILINEAR,
        ))
    
    def _add_random_crop_fn(self):
        self.augmentations_to_apply.append(T.RandomCrop(
            (self.cfg.augmentations.random_crop_size,
            self.cfg.augmentations.random_crop_size)
        ))
    
    def _add_synchronized_flip_fn(self):
        self.augmentations_to_apply.append(SynchronizedFlip(
            p_horizontal=self.cfg.augmentations.random_horizontal_flip,
            p_vertical=self.cfg.augmentations.random_vertical_flip
        ))

    def __call__(self, echo: torch.Tensor, seg: Optional[torch.Tensor] = None, use_augmentations: bool = True) -> torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]: 
        inp = {'video': echo}
        out = self.transform_fn(inp)
        echo = out['video']

        if seg is not None:
            inp.update({'seg': seg})
            out = self.transform_fn_for_segmap(inp)
            seg = out['seg']

        if use_augmentations:
            inp = {'video': echo}
            if seg is not None:
                inp.update({'seg': seg})
            out = self.augment_fn(inp)
            echo = out['video']
            if seg is not None:
                seg = out['seg']
        
        if seg is not None:
            return echo, seg
        else:
            return echo

class SynchronizedFlip(object):
    def __init__(self, p_horizontal=0.5, p_vertical=0.5):
        self.p_horizontal = p_horizontal
        self.p_vertical = p_vertical

    def __call__(self, x):
        flip_h = random.random() < self.p_horizontal
        flip_v = random.random() < self.p_vertical

        if flip_h:
            x['video'] = T.functional.hflip(x['video'])
            x['seg'] = T.functional.hflip(x['seg'])
        if flip_v:
            x['video'] = T.functional.vflip(x['video'])
            x['seg'] = T.functional.vflip(x['seg'])
        return x