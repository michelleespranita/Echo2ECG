from omegaconf import DictConfig, OmegaConf

import os
from lightning import Trainer

from ecg.models.DownstreamECGEncoder import DownstreamECGEncoder
from ecg.datasets.DownstreamECGDataModule import DownstreamECGDataModule

def inference(
    cfg: DictConfig,
    save_dir: str,
    devices: int = 1,
):

    if os.path.exists(os.path.join(save_dir, 'config.yaml')):
        model_cfg = OmegaConf.load(
            os.path.join(save_dir, 'config.yaml')
        )
    else:
        model_cfg = cfg

    OmegaConf.set_struct(model_cfg, False)
    model_cfg.ecg_encoder_checkpoint_path = None
    model_cfg.token_aggregator_path = None
    model_cfg.ecg_alignment_path = None
    OmegaConf.set_struct(model_cfg, True)

    datamodule = DownstreamECGDataModule(cfg=cfg)

    if cfg.use_best_checkpoint:
        checkpoint_names = [f for f in os.listdir(os.path.join(save_dir, 'checkpoints')) if f != 'last.ckpt']
        checkpoint_name = checkpoint_names[0]
    else:
        checkpoint_name = 'last.ckpt'

    print(f'Loading checkpoint {os.path.join(save_dir, "checkpoints", checkpoint_name)}...')
    
    model = DownstreamECGEncoder.load_from_checkpoint(
        os.path.join(save_dir, 'checkpoints', checkpoint_name),
        cfg=model_cfg,
        save_dir=save_dir,
        strict=False
    )
    
    strategy = "ddp" if devices > 1 else "auto"
    
    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        precision="bf16-mixed",
        logger=False
    )

    if cfg.validate:
        model.is_final_eval = True
        trainer.validate(model=model, datamodule=datamodule)
    
    if cfg.test:
        trainer.test(model=model, datamodule=datamodule)