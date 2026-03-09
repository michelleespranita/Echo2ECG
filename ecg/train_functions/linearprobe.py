from omegaconf import DictConfig

import torch
import wandb
from lightning.pytorch.loggers import WandbLogger

from shared.models.linear_probing.LinearProbe import LinearProbe
from ecg.models.DownstreamECGEncoder import DownstreamECGEncoder
from ecg.datasets.DownstreamECGDataModule import DownstreamECGDataModule

@torch.no_grad()
def linearprobe(
    cfg: DictConfig,
    wandb_logger: WandbLogger,
    save_dir: str,
    devices: int = 1
):
    wandb_run = wandb_logger.experiment

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    datamodule = DownstreamECGDataModule(cfg=cfg)
    datamodule.setup(stage='linearprobe')
    dataloader_train = datamodule.train_dataloader_linearprobe()
    dataloader_val = datamodule.val_dataloader()

    model = DownstreamECGEncoder(cfg=cfg, save_dir=save_dir)
    model.to(device)
    model.eval()
    
    def extract_ecg_features(batch):
        if cfg.token_aggregator_path is not None:
            ecg_global_token = model.forward_features(batch['ecg'].to(model.device))
        else:
            ecg_tokens = model.ecg_encoder.forward_features(batch['ecg'].to(model.device))
            ecg_global_token = ecg_tokens.mean(dim=1)
        return {
            'global_token': ecg_global_token,
            'label': batch['label']
        }
    
    probe = LinearProbe(
        task_type=cfg.downstream_task_ecg.task_type,
        device=device,
        num_classes=cfg.downstream_task_ecg.num_classes,
        save_dir=save_dir
    )
    probe.fit(dataloader_train, extract_ecg_features)
    metrics = probe.evaluate(dataloader_val, extract_ecg_features, 'val')

    metrics_final_val = dict()
    for metric_name, metric_value in metrics.items():
        if 'per_output' in metric_name:
            metric_name = metric_name.replace('_per_output', '')
            if metric_value.dim() != 0:
                for i, val in enumerate(metric_value):
                    metrics_final_val[f'final_val/{metric_name}_output_{i}'] = val
        else:
            metrics_final_val[f'final_val/{metric_name}'] = metric_value
    
    print(f'Final validation metrics: {metrics_final_val}')

    wandb_run.log(metrics_final_val)

    if cfg.test:
        datamodule.setup(stage='test')
        dataloader_test = datamodule.test_dataloader()
        metrics = probe.evaluate(dataloader_test, extract_ecg_features, 'test')

        metrics_test = dict()
        for k, v in metrics.items():
            metrics_test[f'test/{k}'] = v
        
        print(f'Test metrics: {metrics_test}')

        wandb_run.log(metrics_test)
    
    wandb.finish()