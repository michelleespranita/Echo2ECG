from omegaconf import DictConfig

import torch
import wandb
from lightning.pytorch.loggers import WandbLogger

from multimodal.models.Retriever import Retriever

@torch.no_grad()
def retrieval(
    cfg: DictConfig,
    wandb_logger: WandbLogger,
    save_dir: str = None,
    devices: int = 1,
):
    wandb_run = wandb_logger.experiment

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Retriever(cfg=cfg, save_dir=save_dir)
    model.to(device)
    model.device = device

    if cfg.eval_mode == 'all':
        eval_modes = ['train', 'val', 'test']
    else:
        eval_modes = [cfg.eval_mode]
    
    for mode in eval_modes:
        model.eval_mode = mode
        model.setup()
        out = model()

        print(f'Eval mode: {mode}')
        for k, v in out['metrics'].items():
            print(f'{k}: {v}')
        
        wandb_run.log(out['metrics'])

        if 'umap' in out:
            wandb_run.log({f'{mode}/umap': [wandb.Image(out['umap'])]})
    
    wandb.finish()