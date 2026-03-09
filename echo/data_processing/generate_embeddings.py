import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorchvideo.transforms as TV
import torchvision.transforms as T

from echo.datasets.EchoAugmentations import MinMaxNormalization, ImageNetNormalization
from echo.utils.load_video import loadvideo_decord
from util.echoprime import ECHOPRIME_MEAN, ECHOPRIME_STD, crop_and_scale

class EchoDataset(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames

        self.img_size = 224
        self.num_frames = 16
        self.sampling_rate = 2
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
        out = loadvideo_decord(self.filenames[idx], 'val', self.sampling_rate, self.num_frames, repeat_or_pad='pad')
        echo = out['video']
        echo = np.array(echo)
        echo_augm = np.zeros((len(echo), self.img_size, self.img_size, self.num_channels))
        for i in range(len(echo_augm)):
            echo_augm[i] = crop_and_scale(echo[i])

        echo_augm = torch.from_numpy(echo_augm).float()
        echo_augm = echo_augm.permute(3, 0, 1, 2)
        # normalize
        echo_augm.sub_(ECHOPRIME_MEAN).div_(ECHOPRIME_STD)

        echo = echo_augm
        
        return {'echo': echo}

if __name__ == '__main__':
    argparse = ArgumentParser()
    argparse.add_argument('--echo_encoder_path', type=str)
    argparse.add_argument('--input_dir', type=str)
    argparse.add_argument('--output_dir', type=str)
    argparse.add_argument('--normalize', action='store_true')
    args = argparse.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # load dataset
    filenames = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)]
    filenames.sort()
    dataset = EchoDataset(filenames)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # load model
    echo_encoder = torchvision.models.video.mvit_v2_s()
    checkpoint = torch.load(args.echo_encoder_path, map_location='cpu', weights_only=False)
    msg = echo_encoder.load_state_dict(checkpoint, strict=True)
    print(msg)
    echo_encoder.to(device)
    echo_encoder.eval()

    # generate embeddings
    with torch.no_grad():
        embeddings = None
        for batch in tqdm(dataloader):
            echo = batch['echo'].float().to(device)
            global_token = echo_encoder(echo)
            if embeddings is None:
                embeddings = global_token
            else:
                embeddings = torch.cat([embeddings, global_token], dim=0)
        if args.normalize:
            embeddings = F.normalize(embeddings, dim=-1)
    
    print('embeddings', embeddings.shape)

    # save embeddings & filenames
    torch.save(filenames, os.path.join(args.output_dir, 'echoprime_echo_filepaths.pt'))
    if args.normalize:
        suffix = ''
    else:
        suffix = '_unnorm'
    torch.save(embeddings, os.path.join(args.output_dir, f'echoprime_echo_embeddings{suffix}.pt'))


