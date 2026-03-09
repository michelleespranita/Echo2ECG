import lightning as L
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from ecg.datasets.DownstreamECGDataset import DownstreamECGDataset

class DownstreamECGDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.downstream_task_ecg.batch_size
        self.num_workers = cfg.downstream_task_ecg.num_workers
        self.pin_memory = cfg.dataset.pin_memory

    def setup(self, stage: str):
        if stage == 'fit':
            self.dataset_train = DownstreamECGDataset(cfg=self.cfg, mode='train', target=self.cfg.downstream_task_ecg.target)
            self.dataset_val = DownstreamECGDataset(cfg=self.cfg, mode='val', target=self.cfg.downstream_task_ecg.target)
        elif stage == 'validate':
            self.dataset_val = DownstreamECGDataset(cfg=self.cfg, mode='val', target=self.cfg.downstream_task_ecg.target)
        elif stage == 'test':
            self.dataset_test = DownstreamECGDataset(cfg=self.cfg, mode='test', target=self.cfg.downstream_task_ecg.target)
        elif stage == 'linearprobe':
            self.dataset_train = DownstreamECGDataset(cfg=self.cfg, mode='train', target=self.cfg.downstream_task_ecg.target)
            self.dataset_train.mode = 'val' # disable aug
            self.dataset_val = DownstreamECGDataset(cfg=self.cfg, mode='val', target=self.cfg.downstream_task_ecg.target)
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def train_dataloader_linearprobe(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
