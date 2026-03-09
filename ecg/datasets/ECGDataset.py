import os

import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, cfg, filenames):
        self.cfg = cfg
        
        # Create full filepaths
        # self.filenames = [os.path.join(cfg.dataset.ecg.base_path, f) for f in filenames]
        self.filenames = filenames

        self.sig_len = cfg.dataset.ecg.sig_len

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        ecg = torch.load(self.filenames[idx]).unsqueeze(0).float()
        ecg = ecg[:, :, :self.sig_len]
        return {'ecg': ecg}