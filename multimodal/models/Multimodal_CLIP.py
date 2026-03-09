import os
from typing import Sequence, Dict, Any
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from einops import rearrange
import lightning as L
from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from util.init_loss import init_contrastive_loss_fn, init_loss_fn
from util.init_metrics import init_metrics_fn
from util.init_model import init_ecg_encoder, init_echo_encoder, init_alignment_layer, init_task_layer
from util.optimizer import create_optimizer_and_scheduler_multiple_lr, create_optimizer_and_scheduler
from util.model import get_grad_norm_

from echo.models.EchoViewAggregator import EchoViewAggregator
from multimodal.models.token_aggregation.TokenAggregator import TokenAggregator
from shared.models.linear_probing.LinearProbe import LinearProbe

class MultimodalECGEchoCLIP(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        save_dir: str
    ) -> None:
        super(MultimodalECGEchoCLIP, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.save_dir = save_dir

        self.batch_size = cfg.dataset.batch_size

        if 'online_eval_warmup_epochs' in self.cfg:
            OmegaConf.set_struct(self.cfg, False)
            self.cfg.downstream_task_ecg.params.scheduler.warmup_cosine.warmup_steps = self.cfg.online_eval_warmup_epochs
            OmegaConf.set_struct(self.cfg, True)

        if 'online_eval_max_epochs' in self.cfg:
            OmegaConf.set_struct(self.cfg, False)
            self.cfg.downstream_task_ecg.max_epochs = self.cfg.online_eval_max_epochs
            OmegaConf.set_struct(self.cfg, True)

        self.ecg_encoder, self.ecg_encoder_cfg = init_ecg_encoder(cfg)
        self.echo_encoder, self.echo_encoder_cfg = init_echo_encoder(cfg)

        # Delete unused components
        del self.ecg_encoder.head, self.ecg_encoder.head_drop, self.ecg_encoder.fc_norm
        
        if self.cfg.model.echo.view_dropout.use:
            self.echo_max_n_views = self.cfg.dataset.echo.max_n_views
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.echo_encoder.embed_dim))
        
        if self.cfg.model.echo.view_aggregation.use:
            self.echo_projection_for_agg = nn.Linear(self.echo_encoder.embed_dim, cfg.model.echo.view_aggregation.proj_embed_dim)
            self.echo_encoder_cfg.embed_dim = cfg.model.echo.view_aggregation.proj_embed_dim # temporary for the initialization of other layers
            self.echo_view_aggregator = EchoViewAggregator(cfg, self.echo_encoder_cfg)

        self.token_aggregator = TokenAggregator(cfg, self.ecg_encoder_cfg, self.echo_encoder_cfg)

        self.ecg_alignment_global_token = init_alignment_layer(cfg, self.ecg_encoder_cfg)
        self.echo_alignment_global_token = init_alignment_layer(cfg, self.echo_encoder_cfg)

        # Reset
        self.echo_encoder_cfg.embed_dim = self.echo_encoder.embed_dim

        # Define loss function
        self.loss_fn = init_contrastive_loss_fn(cfg)

        # Necessary for training using multiple optimizers!
        # self.automatic_optimization = False
    
    def setup(self, stage: str) -> None: # executed after init, the model is already moved to desired device
        self._init_task_metrics()
    
    def _init_task_layers(self, modality: str) -> None:
        if modality == 'ecg':
            OmegaConf.set_struct(self.cfg.downstream_task_ecg, False)
            self.cfg.downstream_task_ecg.embed_dim = self.ecg_encoder_cfg.embed_dim
            OmegaConf.set_struct(self.cfg.downstream_task_ecg, True)

            self.ecg_task_layer = init_task_layer(self.cfg.downstream_task_ecg).to(self.device)
            self.ecg_task_layernorm = nn.LayerNorm(self.ecg_encoder_cfg.embed_dim).to(self.device)
        else:
            raise NotImplementedError
    
    def _init_task_loss_fn(self, modality: str) -> None:
        if modality == 'ecg':
            self.ecg_task_loss_fn = init_loss_fn(self.cfg.downstream_task_ecg.task_type)
        else:
            raise NotImplementedError

    def _init_task_metrics(self) -> None:
        self.ecg_task_metrics = init_metrics_fn(
            self.cfg.downstream_task_ecg.task_type, self.device, self.cfg.downstream_task_ecg.num_classes
        )

    def forward_ecg(self, ecg: torch.Tensor):
        """
        Args:
            ecg: ECG input tensor of shape (B, C, V, T)
        """
        ecg_tokens = self.ecg_encoder.forward_features(ecg)
        ecg_tokens_dict = self.token_aggregator(ecg_tokens, modality='ecg')
        ecg_global_token_aligned = self.ecg_alignment_global_token(ecg_tokens_dict['ecg_global_token'])
        return {
            'ecg_global_token': ecg_tokens_dict['ecg_global_token'],
            'ecg_global_token_aligned': ecg_global_token_aligned
        }

    def forward_echo(self, echo: torch.Tensor):
        """
        Args:
            echo: Echo input tensor of shape (B, C, T, H, W)
        """
        # extract echo features
        echo_global_token = self.echo_encoder(echo)
        
        # project for view aggregation (if view agg is used)
        if self.cfg.model.echo.view_aggregation.use:
            echo_global_token = self.echo_projection_for_agg(echo_global_token)

        # align
        echo_global_token_aligned = self.echo_alignment_global_token(echo_global_token)

        return {
            'echo_global_token': echo_global_token,
            'echo_global_token_aligned': echo_global_token_aligned
        }
    
    def forward_echo_study(self, echo: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            echo: Echo input tensor of shape (B, num_views, C, T, H, W)
            mask: (B, num_views)
        """
        B = len(echo)
        echo = rearrange(echo, 'B num_views C T H W -> (B num_views) C T H W')

        # extract echo features
        echo_global_token = self.echo_encoder(echo) # (B * num_views, D)
        
        echo_view_tokens = rearrange(echo_global_token, '(B num_views) D -> B num_views D', B=B) # (B, num_views, D)

        # view dropout (if necessary)
        if self.cfg.model.echo.view_dropout.use:
            B, max_n_views, D = echo_view_tokens.shape
            x_flat = echo_view_tokens.view(B * max_n_views, 1, D) # (B * max_n_views, 1, D)
            if mask is not None: # 0: view available, 1: view unavailable (B, max_n_views)
                mask_flat = mask.view(-1) # (B * max_n_views,)
            else: # all views are available
                mask_flat = torch.zeros(B * max_n_views, dtype=torch.bool) # (B * max_n_views,)
            x_flat, mask_flat = self._apply_view_dropout(x_flat, mask_flat) # (B * max_n_views, 1, D)
            echo_view_tokens = x_flat.view(B, max_n_views, D) # (B, max_n_views, D)
            mask = mask_flat.view(B, max_n_views).float() # (B, max_n_views)
        
        # view aggregation
        if self.cfg.model.echo.view_aggregation.strategy == 'mean' and not hasattr(self, 'echo_projection_for_agg'): # !!! Only for retrieval purposes
            echo_study_token = echo_view_tokens.mean(dim=1)
        else:
            echo_view_tokens = self.echo_projection_for_agg(echo_view_tokens)
            echo_study_token = self.echo_view_aggregator(echo_view_tokens, mask=mask)
        
        # align
        echo_global_token_aligned = self.echo_alignment_global_token(echo_study_token)

        return {
            'echo_global_token': echo_study_token,
            'echo_global_token_aligned': echo_global_token_aligned,
            'echo_local_tokens': echo_view_tokens,
            'echo_view_mask': mask
        }
    
    def _apply_view_dropout(self, x_flat, mask_flat):
        # mask -> 0: view available, 1: view unavailable

        total_elements = x_flat.shape[0] # B * max_n_views
        
        should_mask = torch.rand(total_elements, device=self.device) < self.cfg.model.echo.view_dropout.mask_ratio
        should_mask = should_mask.bool() | mask_flat.bool()
        
        if should_mask.any():
            x_flat = x_flat.clone()
            mask_tokens = self.mask_token.expand(should_mask.sum(), 1, self.echo_encoder.embed_dim).to(self.device)
            x_flat[should_mask] = mask_tokens
        
        batch_size = total_elements // self.echo_max_n_views
        should_mask = should_mask.view(batch_size, self.echo_max_n_views)
        
        return x_flat, should_mask

    def forward(self, batch: dict):            
        if 'echo_embed' in batch and 'ecg_embed' not in batch:
            # Process ECG, use precomputed echo embeddings
            ecg = batch['ecg'].float()
            ecg_token_dict = self.forward_ecg(ecg)

            echo_embed = batch['echo_embed'].float()

            if len(echo_embed.shape) == 2: # (B, D)
                echo_global_token = echo_embed
                echo_token_dict = {
                    'echo_global_token': echo_global_token,
                    'echo_global_token_aligned': self.echo_alignment_global_token(echo_global_token)
                }
            
            elif len(echo_embed.shape) == 3: # (B, num_views, D)
                echo_view_tokens = echo_embed
                mask = batch['attn_mask'].float()

                # view dropout (if necessary)
                if self.cfg.model.echo.view_dropout.use:
                    B, max_n_views, D = echo_view_tokens.shape
                    x_flat = echo_view_tokens.view(B * max_n_views, 1, D) # (B * max_n_views, 1, D)
                    if mask is not None: # 0: view available, 1: view unavailable (B, max_n_views)
                        mask_flat = mask.view(-1) # (B * max_n_views,)
                    else: # all views are available
                        mask_flat = torch.zeros(B * max_n_views, dtype=torch.bool) # (B * max_n_views,)
                    x_flat, mask = self._apply_view_dropout(x_flat, mask_flat) # (B * max_n_views, 1, D)

                    echo_view_tokens = x_flat.view(B, max_n_views, D) # (B, max_n_views, D)
                    mask = mask.float() # (B, max_n_views)
                
                echo_view_tokens = self.echo_projection_for_agg(echo_view_tokens)
                echo_study_token = self.echo_view_aggregator(echo_view_tokens, mask=mask)
                
                echo_global_token_aligned = self.echo_alignment_global_token(echo_study_token)

                echo_token_dict = {
                    'echo_global_token': echo_study_token,
                    'echo_global_token_aligned': echo_global_token_aligned,
                    'echo_local_tokens': echo_view_tokens,
                    'echo_view_mask': mask
                }
            
        elif 'ecg_embed' not in batch and 'echo_embed' not in batch:
            # Process both modalities
            ecg = batch['ecg'].float()
            ecg_token_dict = self.forward_ecg(ecg)

            echo = batch['echo'].float()

            if len(echo.shape) == 5: # (B, C, T, H, W)
                echo_token_dict = self.forward_echo(echo)
            else: # (B, num_views, C, T, H, W)
                mask = batch['attn_mask'].float()
                echo_token_dict = self.forward_echo_study(echo, mask)
        
        else:
            raise NotImplementedError

        return ecg_token_dict, echo_token_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=self.cfg.train.use_autocast):
            ecg_token_dict, echo_token_dict = self(batch)
            loss = self.loss_fn(ecg_token_dict, echo_token_dict)
            self.log('train/loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size, prog_bar=True, sync_dist=True)

        # Calculate grad norm
        grad_norm_ecg_encoder = get_grad_norm_(self.ecg_encoder.parameters(), norm_type=2.0)
        grad_norm_echo_encoder = get_grad_norm_(self.echo_encoder.parameters(), norm_type=2.0)
        
        self.log('train/grad_norm_ecg_encoder', grad_norm_ecg_encoder, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size, prog_bar=True, sync_dist=True)
        self.log('train/grad_norm_echo_encoder', grad_norm_echo_encoder, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size, prog_bar=True, sync_dist=True)

        return loss
    
    def on_train_epoch_end(self) -> None:
        """
        Executes at the end of each training epoch
        """

        # Online eval on first epoch and every n epochs
        if (self.trainer.current_epoch == 0) or ((self.trainer.current_epoch + 1) % self.cfg.online_eval_every_n_epoch == 0):
            self.trainer.datamodule.setup(stage='online_eval')
            dataloaders_train = self.trainer.datamodule.online_eval_train_dataloader()
            dataloaders_val = self.trainer.datamodule.online_eval_val_dataloader()
            downstream_ecg_dataloader_train = dataloaders_train['downstream_ecg']
            downstream_ecg_dataloader_val = dataloaders_val['downstream_ecg']

            print('Doing online evaluation on ECG task...')
            self._online_eval_ecg(downstream_ecg_dataloader_train, downstream_ecg_dataloader_val)
        
        # Linear probing on first epoch and every n epochs
        if (self.trainer.current_epoch == 0) or ((self.trainer.current_epoch + 1) % self.cfg.linear_probe_every_n_epoch == 0):
            self.trainer.datamodule.setup(stage='linear_probe')
            dataloaders_train = self.trainer.datamodule.linear_probe_train_dataloader()
            dataloaders_val = self.trainer.datamodule.linear_probe_val_dataloader()
            downstream_ecg_dataloader_train = dataloaders_train['downstream_ecg']
            downstream_ecg_dataloader_val = dataloaders_val['downstream_ecg']

            print('Doing linear probing on ECG task...')
            self._linear_probe_ecg(downstream_ecg_dataloader_train, downstream_ecg_dataloader_val)
    
    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=self.cfg.train.use_autocast):
            ecg_token_dict, echo_token_dict = self(batch)
            loss = self.loss_fn(ecg_token_dict, echo_token_dict)
            self.log('val/loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size, prog_bar=True, sync_dist=True)
        
        return loss
        
    def _online_eval_ecg_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        ecg, label = batch['ecg'].to(self.device).float(), batch['label'].to(self.device).float()
        with torch.no_grad():
            ecg_tokens = self.ecg_encoder.forward_features(ecg)
            if self.cfg.model.token_aggregation.ecg.strategy == 'cls':
                ecg_global_token = ecg_tokens[:, 0]
            else:
                ecg_global_token = self.token_aggregator.ecg_token_aggregator(ecg_tokens)
        ecg_token = self.ecg_task_layernorm(ecg_global_token)
        pred = self.ecg_task_layer(ecg_token)
        pred = pred.squeeze(-1)
        loss = self.ecg_task_loss_fn(pred, label)
        self.ecg_task_metrics.update(pred, label)
        return loss

    def _online_eval_ecg(self, dataloader_train: DataLoader, dataloader_val: DataLoader) -> None:
        # Initialize task layers and loss function
        self._init_task_layers(modality='ecg')
        self._init_task_loss_fn(modality='ecg')

        # Initialize optimizer and scheduler
        ecg_task_optimizer, ecg_task_scheduler = create_optimizer_and_scheduler(
            models=[self.ecg_task_layernorm, self.ecg_task_layer],
            optimizer_params=self.cfg.downstream_task_ecg.params,
            num_opt_steps_per_epoch=self.num_steps(mode='downstream_ecg_online_eval'),
        )
        if ecg_task_scheduler is not None:
            ecg_task_scheduler = ecg_task_scheduler['scheduler']

        self.ecg_encoder.eval()
        self.token_aggregator.eval()

        for epoch in tqdm(range(self.cfg.downstream_task_ecg.max_epochs), desc='Online eval ECG task - training'):
            for batch_idx, batch in enumerate(dataloader_train):
                loss = self._online_eval_ecg_step(batch, batch_idx)

                ecg_task_optimizer.zero_grad()
                loss.backward()

                ecg_task_optimizer.step()
                if ecg_task_scheduler is not None:
                    ecg_task_scheduler.step()
        
        self.ecg_task_layernorm.eval()
        self.ecg_task_layer.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader_val), desc='Online eval ECG task - validation'):
                loss = self._online_eval_ecg_step(batch, batch_idx)

            ecg_task_metrics_dict = self.ecg_task_metrics.compute()
            self.ecg_task_metrics.reset()
            for metric_name, metric_value in ecg_task_metrics_dict.items():
                metric_value = metric_value.mean().item()
                self.log(
                    f'online_eval/ecg_{self.cfg.downstream_task_ecg.task_type}_{self.cfg.downstream_task_ecg.dataset}_{self.cfg.downstream_task_ecg.target}_{metric_name}',
                    metric_value,
                    on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True
                )
        
        self.ecg_encoder.train()
        self.token_aggregator.train()
        self.ecg_task_layernorm.train()
        self.ecg_task_layer.train()

    @torch.no_grad()
    def _linear_probe_ecg(self, dataloader_train: DataLoader, dataloader_val: DataLoader):
        def extract_ecg_features(batch):
            ecg_token_dict = self.forward_ecg(batch['ecg'].to(self.device))
            return {
                'global_token': ecg_token_dict['ecg_global_token'],
                'global_token_aligned': ecg_token_dict['ecg_global_token_aligned'],
                'label': batch['label']
            }
        
        self.ecg_encoder.eval()
        self.token_aggregator.eval()
        self.ecg_alignment_global_token.eval()

        probe = LinearProbe(
            task_type=self.cfg.downstream_task_ecg.task_type,
            device=self.device,
            num_classes=self.cfg.downstream_task_ecg.num_classes,
        )
        probe.fit(dataloader_train, extract_ecg_features)
        metrics = probe.evaluate(dataloader_val, extract_ecg_features, 'val')
        
        log_dict = {
            f'linear_probe/ecg_{self.cfg.downstream_task_ecg.task_type}_'
            f'{self.cfg.downstream_task_ecg.dataset}_'
            f'{self.cfg.downstream_task_ecg.target}_{metric_name}': metric_value
            for metric_name, metric_value in metrics.items()
        }
        self.logger.experiment.log(log_dict, step=self.trainer.global_step)

        self.ecg_encoder.train()
        self.token_aggregator.train()
        self.ecg_alignment_global_token.train()

    def configure_optimizers(self): 
        model_group_dict = {
            'ecg_encoder': self.ecg_encoder,
            'echo_encoder': self.echo_encoder,
            'token_aggregator': self.token_aggregator
        }
        if hasattr(self, 'echo_view_aggregator') and self.cfg.model.echo.view_aggregation.strategy in ['att', 'cls']:
            model_group_dict['echo_projection_for_agg'] = self.echo_projection_for_agg
            model_group_dict['echo_view_aggregator'] = self.echo_view_aggregator
        if self.cfg.model.echo.view_dropout.use:
            model_group_dict['mask_token'] = self.mask_token
        model_group_dict['ecg_alignment_global_token'] = self.ecg_alignment_global_token
        model_group_dict['echo_alignment_global_token'] = self.echo_alignment_global_token

        main_optimizer, main_scheduler = create_optimizer_and_scheduler_multiple_lr(
            model_group_dict=model_group_dict,
            optimizer_params=self.cfg.train.params,
            num_opt_steps_per_epoch=self.num_steps(mode='main')
        )

        if main_scheduler is not None:
            return [main_optimizer], [main_scheduler]
        else:
            return main_optimizer

    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        checkpoint_folder = os.path.join(self.save_dir, 'checkpoints')
        os.makedirs(checkpoint_folder, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_folder,
            auto_insert_metric_name=True,
            monitor='val/loss',
            mode='min',
            save_top_k=self.cfg.save_top_k,
            every_n_epochs=self.cfg.save_model_every_n_epochs,
            # save_on_train_epoch_end=True,
            save_on_train_epoch_end=False,
            save_weights_only=False,
            verbose=True,
            save_last=True,
            # save_last=False
        )
        
        learning_rate_callback = LearningRateMonitor(
            logging_interval='epoch',
            log_weight_decay=True,
            log_momentum=False
        )

        callbacks = [learning_rate_callback, checkpoint_callback]
        
        if self.cfg.train.params.early_stopping.use:
            early_stopping_callback = EarlyStopping(
                monitor='val/loss',
                mode='min',
                patience=self.cfg.train.params.early_stopping.patience
            )
            callbacks.append(early_stopping_callback)

        return callbacks
    
    def num_steps(self, mode='main') -> int:
        """Get number of steps per epoch"""
        if mode == 'main':
            dataset = self.trainer.fit_loop._data_source.dataloader()
        elif mode == 'downstream_ecg_online_eval':
            dataset = self.trainer.datamodule.online_eval_train_dataloader()['downstream_ecg']
        dataset_size = len(dataset) # num_batches
        num_devices = max(1, self.trainer.num_devices)
        num_steps = dataset_size // (self.trainer.accumulate_grad_batches * num_devices)
        return num_steps

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        os.makedirs(os.path.join(self.save_dir, 'checkpoints'), exist_ok=True)

        # Save only the ECG encoder
        ecg_encoder_state_dict = self.ecg_encoder.state_dict()
        for k in ['norm.weight', 'norm.bias', 'head.weight', 'head.bias']: # these model components were not used during training
            if k in ecg_encoder_state_dict:
                del ecg_encoder_state_dict[k]
        torch.save({
            'model': ecg_encoder_state_dict,
            'cfg': self.cfg,
            'epoch': self.current_epoch
        }, os.path.join(self.save_dir, 'checkpoints', 'ecg_multimodal.ckpt'))

        # Save only the echo encoder
        echo_encoder_state_dict = self.echo_encoder.state_dict()
        torch.save({
            'model': echo_encoder_state_dict,
            'cfg': self.cfg,
            'epoch': self.current_epoch
        }, os.path.join(self.save_dir, 'checkpoints', 'echo_multimodal.ckpt'))

        # Save only the token aggregator
        torch.save({
            'model': self.token_aggregator.state_dict(),
            'cfg': self.cfg,
            'epoch': self.current_epoch
        }, os.path.join(self.save_dir, 'checkpoints', 'token_aggregator.ckpt'))

        # Save echo projection for agg and echo view aggregator if available
        if hasattr(self, 'echo_view_aggregator') and self.cfg.model.echo.view_aggregation.strategy in ['att', 'cls']:
            torch.save({
                'model': self.echo_projection_for_agg.state_dict(),
                'cfg': self.cfg,
                'epoch': self.current_epoch
            }, os.path.join(self.save_dir, 'checkpoints', 'echo_projection_for_agg.ckpt'))
            torch.save({
                'model': self.echo_view_aggregator.state_dict(),
                'cfg': self.cfg,
                'epoch': self.current_epoch
            }, os.path.join(self.save_dir, 'checkpoints', 'echo_view_aggregator.ckpt'))

        # Save only the ECG alignment layers
        torch.save({
            'model': self.ecg_alignment_global_token.state_dict(),
            'cfg': self.cfg,
            'epoch': self.current_epoch
        }, os.path.join(self.save_dir, 'checkpoints', 'ecg_alignment_global_token.ckpt'))

        # Save only the echo alignment layers
        torch.save({
            'model': self.echo_alignment_global_token.state_dict(),
            'cfg': self.cfg,
            'epoch': self.current_epoch
        }, os.path.join(self.save_dir, 'checkpoints', 'echo_alignment_global_token.ckpt'))

        # Save the config
        OmegaConf.save(self.cfg, os.path.join(self.save_dir, 'config.yaml'))
