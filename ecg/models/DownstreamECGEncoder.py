from typing import Sequence, Dict, Any
from omegaconf import DictConfig, OmegaConf

import os
import pandas as pd
import torch
import torch.nn.functional as F
import lightning as L
from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torchmetrics.functional import confusion_matrix

from util.init_loss import init_loss_fn
from util.init_metrics import init_metrics_fn
from util.init_model import init_ecg_encoder, init_token_aggregator, init_task_layer, init_alignment_layer
from util.optimizer import create_optimizer_and_scheduler

class DownstreamECGEncoder(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        save_dir: str
    ):
        super(DownstreamECGEncoder, self).__init__()

        self.cfg = cfg
        self.save_dir = save_dir
        self.batch_size = cfg.downstream_task_ecg.batch_size
        self.task_type = cfg.downstream_task_ecg.task_type

        # Init ecg encoder
        self.cfg.model.ecg.time_steps = self.cfg.downstream_task_ecg.time_steps
        self.cfg.model.ecg.input_size[-1] = self.cfg.downstream_task_ecg.time_steps
        if self.cfg.ecg_encoder_checkpoint_path:
            OmegaConf.set_struct(self.cfg, False)
            self.cfg.train.encoder.ecg.checkpoint_path = self.cfg.ecg_encoder_checkpoint_path
            OmegaConf.set_struct(self.cfg, True)
        self.ecg_encoder, self.ecg_encoder_cfg = init_ecg_encoder(self.cfg)

        # Delete unused components
        del self.ecg_encoder.head, self.ecg_encoder.head_drop, self.ecg_encoder.fc_norm

        # Init alignment layer (from CLIP pretraining if necessary)
        if self.cfg.ecg_alignment_path:
            self.ecg_alignment_global_token = init_alignment_layer(self.cfg, self.ecg_encoder_cfg)
        else:
            print('No alignment is used!')
            self.ecg_alignment_global_token = None
        
        # Init token aggregator
        self.token_aggregator = init_token_aggregator(self.cfg, 'ecg', self.ecg_encoder_cfg)

        # Define task-related head and loss
        self._init_task_layers()
        self._init_task_loss_fn()

        self.is_final_eval = False
    
    def setup(self, stage: str) -> None: # executed after init, the model is already moved to desired device
        self._init_task_metrics()

    def _init_task_layers(self) -> None:
        OmegaConf.set_struct(self.cfg.downstream_task_ecg, False)
        if self.ecg_alignment_global_token:
            self.cfg.downstream_task_ecg.embed_dim = self.cfg.model.alignment.proj_embed_dim
        else:
            self.cfg.downstream_task_ecg.embed_dim = self.ecg_encoder_cfg.embed_dim
        OmegaConf.set_struct(self.cfg.downstream_task_ecg, True)

        self.ecg_task_layer = init_task_layer(self.cfg.downstream_task_ecg)
    
    def _init_task_loss_fn(self) -> None:
        self.ecg_task_loss_fn = init_loss_fn(self.task_type, self.cfg.downstream_task_ecg)
        
    def _init_task_metrics(self) -> None:
        if 'num_tasks' not in self.cfg.downstream_task_ecg:
            num_tasks = 1
        else:
            num_tasks = self.cfg.downstream_task_ecg.num_tasks

        self.ecg_task_metrics_train = init_metrics_fn(
            self.task_type, self.device, self.cfg.downstream_task_ecg.num_classes, num_tasks
        )
        self.ecg_task_metrics_val = init_metrics_fn(
            self.task_type, self.device, self.cfg.downstream_task_ecg.num_classes, num_tasks
        )
        self.ecg_task_metrics_test = init_metrics_fn(
            self.task_type, self.device, self.cfg.downstream_task_ecg.num_classes, num_tasks
        )
    
    def forward_features(self, ecg: torch.Tensor) -> torch.Tensor:
        ecg_tokens = self.ecg_encoder.forward_features(ecg)

        ecg_global_token = self.token_aggregator(ecg_tokens)['ecg_global_token']
        if self.ecg_alignment_global_token:
            ecg_global_token = self.ecg_alignment_global_token(ecg_global_token)
        
        return ecg_global_token
    
    def forward_head(self, ecg_global_token: torch.Tensor) -> torch.Tensor:
        out = self.ecg_task_layer(ecg_global_token)
        return out
    
    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        ecg_global_token = self.forward_features(ecg)
        out = self.forward_head(ecg_global_token)
        return out
    
    def on_train_batch_start(self, batch: dict, batch_idx: int):
        if self.cfg.train.encoder.ecg.freeze_first_n_layers == len(self.ecg_encoder.blocks):
            self.ecg_encoder.eval()

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        ecg, label = batch['ecg'].float(), batch['label'].float()
        pred = self(ecg)
        pred = pred.squeeze(-1)
        loss = self.ecg_task_loss_fn(pred, label)
        self.ecg_task_metrics_train.update(pred, label)
        self.log('train/loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size, prog_bar=True)

        return loss
    
    def on_train_epoch_end(self) -> None:
        ecg_task_metrics_dict = self.ecg_task_metrics_train.compute()
        for metric_name, metric_value in ecg_task_metrics_dict.items():
            if ('per_output' not in metric_name) and (metric_name != 'confmat'):
                self.log(f'train/{metric_name}', metric_value.mean().item(), batch_size=self.batch_size)
        self.ecg_task_metrics_train.reset()
    
    def on_validation_epoch_start(self) -> None:
        # Create buffers
        self.logits_val = []
        self.labels_val = []
        self.filenames_val = []
    
    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        ecg, label = batch['ecg'].float(), batch['label'].float()
        pred = self(ecg)
        pred = pred.squeeze(-1)
        loss = self.ecg_task_loss_fn(pred, label)
        self.ecg_task_metrics_val.update(pred, label)
        if not self.is_final_eval:
            self.log('val/loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size, prog_bar=True)
        else:
            self.log('final_val/loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size, prog_bar=True)

        self.logits_val.append(pred.detach().cpu())
        self.labels_val.append(label.detach().cpu())

        if 'filename' in batch:
            self.filenames_val += batch['filename']
        
        return loss
    
    def on_validation_epoch_end(self) -> None:
        ecg_task_metrics_dict = self.ecg_task_metrics_val.compute()
        for metric_name, metric_value in ecg_task_metrics_dict.items():
            if ('per_output' not in metric_name) and (metric_name != 'confmat'):
                if not self.is_final_eval:
                    self.log(f'val/{metric_name}', metric_value.mean().item(), batch_size=self.batch_size)
                else:
                    self.log(f'final_val/{metric_name}', metric_value.mean().item(), batch_size=self.batch_size)
            elif 'per_output' in metric_name:
                metric_name = metric_name.replace('_per_output', '')
                if self.is_final_eval:
                    if metric_value.dim() != 0:
                        for i, val in enumerate(metric_value):
                            self.log(f'final_val/{metric_name}_output_{i}', val, on_step=False, on_epoch=True)
        if self.task_type == 'regression':
            avg = (self.ecg_task_metrics_val.get('pcc').mean().item() + self.ecg_task_metrics_val.get('r2').mean().item()) / 2
            if not self.is_final_eval:
                self.log('val/avg', avg, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size, prog_bar=True)
            else:
                self.log('final_val/avg', avg, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size, prog_bar=True)
        self.ecg_task_metrics_val.reset()

        if self.is_final_eval:
            self._save_predictions('val')
    
    def on_test_epoch_start(self) -> None:
        # Create buffers
        self.logits_test = []
        self.labels_test = []
        self.filenames_test = []
    
    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        ecg, label = batch['ecg'].float(), batch['label'].float() # label: (B, num_classes)
        pred = self(ecg)
        pred = pred.squeeze(-1) # (B, num_classes) if num_classes > 1 else (B,)
        loss = self.ecg_task_loss_fn(pred, label)
        self.ecg_task_metrics_test.update(pred, label)
        self.log('test/loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size, prog_bar=True)

        self.logits_test.append(pred.detach().cpu())
        self.labels_test.append(label.detach().cpu())
        
        if 'filename' in batch:
            self.filenames_test += batch['filename']

        return loss

    def on_test_epoch_end(self) -> None:
        ecg_task_metrics_dict = self.ecg_task_metrics_test.compute()
        for metric_name, metric_value in ecg_task_metrics_dict.items():
            if 'per_output' in metric_name:
                if metric_value.dim() != 0:
                    for i, val in enumerate(metric_value):
                        self.log(f'test/{metric_name}_output_{i}', val, on_step=False, on_epoch=True)
            else:
                self.log(f'test/{metric_name}', metric_value.mean().item(), batch_size=self.batch_size)
        self.ecg_task_metrics_test.reset()
        
        self._save_predictions('test')
    
    def _save_predictions(self, split_name: str):
        logits = getattr(self, f'logits_{split_name}')
        labels = getattr(self, f'labels_{split_name}')

        logits = torch.cat(logits, dim=0) # concat along batch dimension

        filenames = getattr(self, f'filenames_{split_name}')

        if self.task_type == 'segmentation':
            probs = torch.softmax(logits, dim=1) # (B, num_classes, T, H, W)
            preds = torch.argmax(probs, dim=1) # (B, T, H, W)

            # Save predicted segmentation maps
            torch.save(preds, os.path.join(self.save_dir, f'preds_{split_name}.pt'))
            if self.logger is not None:
                self.logger.experiment.save(os.path.join(self.save_dir, f'preds_{split_name}.pt'))

        else:
            # Save confusion matrix
            if self.task_type in ['multiclass_classification']:
                preds = logits.argmax(dim=-1)
                labels = torch.cat(labels, dim=0).argmax(dim=-1)
            elif self.task_type in ['binary_classification', 'multilabel_classification']:
                preds = (F.sigmoid(logits) > 0.5).long()
                labels = torch.cat(labels, dim=0).long()

            if self.task_type == 'binary_classification':
                cm = confusion_matrix(preds, labels, task='binary')
            elif self.task_type == 'multilabel_classification':
                cm = confusion_matrix(preds, labels, task='multilabel', num_labels=self.cfg.downstream_task_ecg.num_classes)
            elif self.task_type == 'multiclass_classification':
                cm = confusion_matrix(preds, labels, task='multiclass', num_classes=self.cfg.downstream_task_ecg.num_classes)
            else:
                cm = None
            
            if cm is not None:
                torch.save(cm, os.path.join(self.save_dir, f'confmat_{split_name}.pt'))
                if self.logger is not None:
                    self.logger.experiment.save(os.path.join(self.save_dir, f'confmat_{split_name}.pt'))

            # Save raw predictions and argmaxed predictions
            torch.save(logits, os.path.join(self.save_dir, f'logits_{split_name}.pt'))
            if self.task_type != 'regression':
                torch.save(preds, os.path.join(self.save_dir, f'preds_{split_name}.pt'))
            if self.logger is not None:
                self.logger.experiment.save(os.path.join(self.save_dir, f'logits_{split_name}.pt'))
                if self.task_type != 'regression':
                    self.logger.experiment.save(os.path.join(self.save_dir, f'preds_{split_name}.pt'))
            
            # Save as csv
            if len(filenames) > 0:
                results = pd.DataFrame({'filename': filenames, 'label': labels.tolist(), 'pred': preds.tolist()})
                results.to_csv(os.path.join(self.save_dir, f'results_{split_name}.csv'), index=False)
    
    def configure_optimizers(self):
        ecg_task_optimizer, ecg_task_scheduler = create_optimizer_and_scheduler(
            models=[self],
            optimizer_params=self.cfg.downstream_task_ecg.params,
            num_opt_steps_per_epoch=self.num_steps(),
        )

        if ecg_task_scheduler is not None:
            return [ecg_task_optimizer], [ecg_task_scheduler]
        else:
            return ecg_task_optimizer
    
    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        checkpoint_folder = os.path.join(self.save_dir, 'checkpoints')
        os.makedirs(checkpoint_folder, exist_ok=True)

        callbacks = []

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_folder,
            auto_insert_metric_name=True,
            monitor='val/avg' if self.task_type == 'regression' else 'val/auroc',
            mode='max',
            save_top_k=self.cfg.save_top_k,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            save_weights_only=False,
            verbose=True,
            save_last=True,
        )
        
        learning_rate_callback = LearningRateMonitor(
            logging_interval='epoch',
            log_weight_decay=True,
            log_momentum=False
        )

        callbacks = [checkpoint_callback, learning_rate_callback]

        if self.cfg.downstream_task_ecg.params.early_stopping.use:
            early_stopping_callback = EarlyStopping(
                monitor='val/avg' if self.task_type == 'regression' else 'val/auroc',
                patience=self.cfg.downstream_task_ecg.params.early_stopping.patience,
                min_delta=self.cfg.downstream_task_ecg.params.early_stopping.min_delta,
                mode='max',
                verbose=True
            )
            callbacks.append(early_stopping_callback)

        return callbacks
    
    def num_steps(self) -> int:
        """Get number of steps per epoch and batch size"""
        dataloader = self.trainer.fit_loop._data_source.dataloader()
        num_batches = len(dataloader)
        num_devices = max(1, self.trainer.num_devices)
        num_steps = num_batches // (self.trainer.accumulate_grad_batches * num_devices)
        return num_steps
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Save the config
        OmegaConf.save(self.cfg, os.path.join(self.save_dir, 'config.yaml'))
