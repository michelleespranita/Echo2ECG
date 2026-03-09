from omegaconf import DictConfig

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from multimodal.models.Multimodal_CLIP import MultimodalECGEchoCLIP
from multimodal.datasets.ECGEchoDataModule import ECGEchoDataModule
from multimodal.datasets.ECGEchoStudyDataModule import ECGEchoStudyDataModule

def train_clip(
    cfg: DictConfig,
    wandb_logger: WandbLogger,
    save_dir: str,
    devices: int = 1,
):
    if cfg.model.echo.view_aggregation.use:
        datamodule = ECGEchoStudyDataModule(cfg=cfg)
    else:
        datamodule = ECGEchoDataModule(cfg=cfg)
    
    model = MultimodalECGEchoCLIP(cfg=cfg, save_dir=save_dir)
    
    # wandb_logger.watch(model, log_graph=True)

    strategy = "ddp_find_unused_parameters_true" if devices > 1 else "auto"
    
    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        precision="bf16-mixed",
        logger=wandb_logger,
        max_epochs=cfg.max_epochs,
        accumulate_grad_batches=cfg.dataset.accum_iter,
        log_every_n_steps=cfg.log_every_n_steps,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        default_root_dir=save_dir,
        num_sanity_val_steps=0,
        profiler="simple",
    )

    if 'resume_from_checkpoint_path' in cfg and cfg.resume_from_checkpoint_path is not None:
        model.strict_loading = False # to enable torch.load(checkpoint, strict=False) instead of strict=True

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.resume_from_checkpoint_path)