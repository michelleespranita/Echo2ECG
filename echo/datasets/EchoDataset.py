import numpy as np
import torch
import pytorchvideo.transforms as TV
import torchvision.transforms as T
from torch.utils.data import Dataset

from echo.datasets.EchoAugmentations import MinMaxNormalization, ImageNetNormalization
from echo.utils.load_video import loadvideo_decord
from util.echoprime import ECHOPRIME_MEAN, ECHOPRIME_STD, crop_and_scale

class EchoDataset(Dataset):
    def __init__(self, cfg, filenames):
        self.cfg = cfg
        
        # Create full filepaths
        self.filenames = filenames

        self.img_size = cfg.dataset.echo.img_size
        self.num_frames = cfg.dataset.echo.num_frames
        self.sampling_rate = cfg.dataset.echo.sampling_rate
        self.num_channels = 3

        transforms_to_apply = []
        transforms_to_apply.append(T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR))
        transforms_to_apply.append(MinMaxNormalization())
        transforms_to_apply.append(ImageNetNormalization())
        self.transform_fn = TV.ApplyTransformToKey(
            key="video",
            transform=T.Compose(transforms_to_apply)
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # same logic as the original code, but faster
        out = loadvideo_decord(self.filenames[idx], 'val', self.cfg.dataset.echo.sampling_rate, self.cfg.dataset.echo.num_frames, repeat_or_pad='pad')
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
        
        return {'echo': echo}