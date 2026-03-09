import lightning as L
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from multimodal.datasets.ECGEchoDataset import ECGEchoDataset
from ecg.datasets.DownstreamECGDataset import DownstreamECGDataset

class ECGEchoDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.dataset.batch_size
        self.num_workers = cfg.dataset.num_workers
        self.pin_memory = cfg.dataset.pin_memory

        self.downstream_ecg_batch_size = cfg.downstream_task_ecg.batch_size
        self.downstream_ecg_num_workers = cfg.downstream_task_ecg.num_workers

        # The time_steps for ECG downstream task has to be the same as for ECG pre-training
        OmegaConf.set_struct(self.cfg, False)
        self.cfg.downstream_task_ecg.time_steps = self.cfg.model.ecg.time_steps
        OmegaConf.set_struct(self.cfg, True)

    def setup(self, stage: str):
        if stage == 'fit':
            self.dataset_train = ECGEchoDataset(cfg=self.cfg, mode='train')
            self.dataset_val = ECGEchoDataset(cfg=self.cfg, mode='val')
        
        elif stage == 'validate':
            self.dataset_val = ECGEchoDataset(cfg=self.cfg, mode='val')
        
        elif stage == 'test':
            self.dataset_test = ECGEchoDataset(cfg=self.cfg, mode='test')
        
        elif stage == 'online_eval':
            self.dataset_train_downstream_ecg_online_eval = DownstreamECGDataset(
                cfg=self.cfg,
                mode='train',
                target=self.cfg.downstream_task_ecg.target,
                first_n_samples=self.cfg.online_eval_first_n_samples_train
            )
            self.dataset_train_downstream_ecg_online_eval.mode = 'val' # to turn off augmentations
        
            self.dataset_val_downstream_ecg_online_eval = DownstreamECGDataset(
                cfg=self.cfg,
                mode='val',
                target=self.cfg.downstream_task_ecg.target,
                first_n_samples=self.cfg.online_eval_first_n_samples_val
            )

        elif stage == 'linear_probe':
            self.dataset_train_downstream_ecg_linear_probe = DownstreamECGDataset(
                cfg=self.cfg,
                mode='train',
                target=self.cfg.downstream_task_ecg.target
            )
            self.dataset_train_downstream_ecg_linear_probe.mode = 'val' # to turn off augmentations

            self.dataset_val_downstream_ecg_linear_probe = DownstreamECGDataset(
                cfg=self.cfg,
                mode='val',
                target=self.cfg.downstream_task_ecg.target
            )

    def train_dataloader(self):
        dataloader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader
    
    def val_dataloader(self):
        dataloader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader
    
    def test_dataloader(self):
        dataloader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader
    
    def online_eval_train_dataloader(self):
        downstream_ecg_dataloader = DataLoader(
            self.dataset_train_downstream_ecg_online_eval,
            batch_size=self.downstream_ecg_batch_size,
            shuffle=False,
            num_workers=self.downstream_ecg_num_workers
        )
        return {'downstream_ecg': downstream_ecg_dataloader}
        
    def online_eval_val_dataloader(self):
        downstream_ecg_dataloader = DataLoader(
            self.dataset_val_downstream_ecg_online_eval,
            batch_size=self.downstream_ecg_batch_size,
            shuffle=False,
            num_workers=self.downstream_ecg_num_workers
        )
        return {'downstream_ecg': downstream_ecg_dataloader}
    
    def linear_probe_train_dataloader(self):
        downstream_ecg_dataloader = DataLoader(
            self.dataset_train_downstream_ecg_linear_probe,
            batch_size=self.downstream_ecg_batch_size,
            shuffle=False,
            num_workers=self.downstream_ecg_num_workers
        )
        return {'downstream_ecg': downstream_ecg_dataloader}
        
    def linear_probe_val_dataloader(self):
        downstream_ecg_dataloader = DataLoader(
            self.dataset_val_downstream_ecg_linear_probe,
            batch_size=self.downstream_ecg_batch_size,
            shuffle=False,
            num_workers=self.downstream_ecg_num_workers
        )
        return {'downstream_ecg': downstream_ecg_dataloader}