import os
import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
from omegaconf import DictConfig, OmegaConf
from typing import Tuple, Optional

import ecg.models.ECGEncoder as ecg_enc
from multimodal.models.alignment.SAIL import SAIL, ShareLock
from multimodal.models.alignment.SimCLR import SimCLRProjectionLayer
from shared.models.TaskLayer import TaskLayer
from util.pos_embed import interpolate_pos_embed_x

ECG_DOMAIN_OFFSET = 4 # from OTiS

def init_ecg_encoder(cfg: DictConfig) -> Tuple[nn.Module, DictConfig]:
    
    ecg_encoder = ecg_enc.__dict__[cfg.model.ecg.model_name](
        img_size=cfg.model.ecg.input_size,
        patch_size=cfg.model.ecg.patch_size, 
        drop_path_rate=cfg.model.ecg.drop_path_rate,
        use_adapter=cfg.model.ecg.use_adapter,
        adapter_bottleneck_dim=cfg.model.ecg.adapter_bottleneck_dim,
        use_checkpoint=cfg.model.ecg.use_checkpoint
    )

    OmegaConf.set_struct(cfg, False)
    cfg.model.ecg.embed_dim = ecg_encoder.embed_dim
    OmegaConf.set_struct(cfg, True)

    if cfg.train.encoder.ecg.checkpoint_path and os.path.exists(cfg.train.encoder.ecg.checkpoint_path):

        if not 'multimodal' in cfg.train.encoder.ecg.checkpoint_path: # unimodal (OTiS)
            checkpoint = torch.load(cfg.train.encoder.ecg.checkpoint_path, map_location='cpu', weights_only=False)
            ECG_DOMAIN_OFFSET = checkpoint['domain_offsets']['ecg']
            if cfg.model.ecg.ignore_pos_embed_y:
                del checkpoint['model']['pos_embed_y.weight']
            else:
                checkpoint['model']['pos_embed_y.weight'] = checkpoint['model']['pos_embed_y.weight'][ECG_DOMAIN_OFFSET+1:ECG_DOMAIN_OFFSET+1+cfg.model.ecg.input_variates, :] # (39, 96) -> (12, 96)
                checkpoint['model']['pos_embed_y.weight'] = torch.cat([
                    torch.zeros(1, checkpoint['model']['pos_embed_y.weight'].size(1)),
                    checkpoint['model']['pos_embed_y.weight']
                ], dim=0) # add padding embed

            # Don't load pos_embed_x just yet
            exclude_keys = ['pos_embed_x']
            filtered_state_dict = {
                k: v for k, v in checkpoint['model'].items()
                if not any(excluded in k for excluded in exclude_keys)
            }

            msg = ecg_encoder.load_state_dict(filtered_state_dict, strict=False)
            print(f'Successfully loaded ECG encoder checkpoint: {cfg.train.encoder.ecg.checkpoint_path}')
            print(msg)

            # Interpolate pos_embed_x if necessary and load it into model
            interpolate_pos_embed_x(ecg_encoder, checkpoint['model'])

        else:
            checkpoint = torch.load(cfg.train.encoder.ecg.checkpoint_path, map_location='cpu', weights_only=False)

            # Don't load pos_embed_x just yet
            exclude_keys = ['pos_embed_x']
            filtered_state_dict = {
                k: v for k, v in checkpoint['model'].items()
                if not any(excluded in k for excluded in exclude_keys)
            }

            msg = ecg_encoder.load_state_dict(filtered_state_dict, strict=False)
            print(f'Successfully loaded ECG encoder checkpoint: {cfg.train.encoder.ecg.checkpoint_path}')
            print(msg)
    
            # Interpolate pos_embed_x if necessary and load it into model
            interpolate_pos_embed_x(ecg_encoder, checkpoint['model'])
    
    else:
        print('The ECG encoder is randomly initialized!')
            
    if cfg.train.encoder.ecg.freeze_first_n_layers > len(ecg_encoder.blocks):
        raise ValueError(f"ECG encoder: num_blocks ({cfg.train.encoder.ecg.freeze_first_n_layers}) exceeds number of blocks ({len(ecg_encoder.blocks)})")

    # Freeze encoder layers
    if cfg.train.encoder.ecg.freeze_first_n_layers > 0:
        for layer_name in ['patch_embed']: # pos_embed is not trainable anw so it's not included here
            layer = getattr(ecg_encoder, layer_name, None)
            if layer is not None:
                for param in layer.parameters():
                    param.requires_grad = False
        for name, param in ecg_encoder.named_parameters():
            if 'blocks.' in name and int(name.split('.')[1]) < cfg.train.encoder.ecg.freeze_first_n_layers:
                param.requires_grad = False

    ecg_encoder_cfg = cfg.model.ecg
    
    return ecg_encoder, ecg_encoder_cfg

def init_echo_encoder(cfg: DictConfig) -> Tuple[nn.Module, DictConfig]:

    echo_encoder = torchvision.models.video.mvit_v2_s()
    echo_encoder.embed_dim = 512
    echo_encoder.head[-1] = nn.Linear(echo_encoder.head[-1].in_features, echo_encoder.embed_dim)

    OmegaConf.set_struct(cfg, False)
    cfg.model.echo.embed_dim = echo_encoder.embed_dim
    OmegaConf.set_struct(cfg, True)

    if cfg.train.encoder.echo.checkpoint_path and os.path.exists(cfg.train.encoder.echo.checkpoint_path):

        if not 'multimodal' in cfg.train.encoder.echo.checkpoint_path: # unimodal (EchoPrime)
            checkpoint = torch.load(cfg.train.encoder.echo.checkpoint_path, map_location='cpu', weights_only=False)
            msg = echo_encoder.load_state_dict(checkpoint, strict=True)
            print(f'Successfully loaded echo encoder checkpoint: {cfg.train.encoder.echo.checkpoint_path}')
            print(msg)
        
        else:
            checkpoint = torch.load(cfg.train.encoder.echo.checkpoint_path, map_location='cpu', weights_only=False)
            msg = echo_encoder.load_state_dict(checkpoint['model'], strict=True)
            print(f'Successfully loaded echo encoder checkpoint: {cfg.train.encoder.echo.checkpoint_path}')
            print(msg)
    
    else:
        print(f'The echo encoder is randomly initialized!')
        
    if cfg.train.encoder.echo.freeze_first_n_layers > len(echo_encoder.blocks):
        raise ValueError(f"Echo encoder: num_blocks ({cfg.train.encoder.echo.freeze_first_n_layers}) exceeds number of blocks ({len(echo_encoder.blocks)})")
    
    # Freeze encoder layers
    if cfg.train.encoder.echo.freeze_first_n_layers > 0:
        for layer_name in ['patch_embed', 'conv_proj', 'pos_encoding']: # pos_embed is not trainable anw so it's not included here
            layer = getattr(echo_encoder, layer_name, None)
            if layer is not None:
                for param in layer.parameters():
                    param.requires_grad = False
        for name, param in echo_encoder.named_parameters():
            if 'blocks.' in name and int(name.split('.')[1]) < cfg.train.encoder.echo.freeze_first_n_layers:
                param.requires_grad = False
        if (cfg.train.encoder.echo.freeze_first_n_layers == len(echo_encoder.blocks)):
            for layer_name in ['norm', 'head']:
                layer = getattr(echo_encoder, layer_name, None)
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = False

    echo_encoder_cfg = cfg.model.echo

    return echo_encoder, echo_encoder_cfg

def init_alignment_layer(cfg: DictConfig, modality_encoder_cfg: DictConfig, alignment_level: Optional[str] = None) -> nn.Module:

    if alignment_level is None:
        alignment_cfg = cfg.model.alignment
    else:
        alignment_cfg = cfg.model.alignment.get(alignment_level)
    
    alignment_type = alignment_cfg.get('alignment_type', 'linear')
    
    if alignment_type == 'simclr':
        alignment = SimCLRProjectionLayer(
            input_dimension=modality_encoder_cfg.embed_dim,
            target_dimension=alignment_cfg.proj_embed_dim,
            width_factor=alignment_cfg.width_factor
        )
    elif alignment_type == 'sail':
        alignment = SAIL(
            input_dimension=modality_encoder_cfg.embed_dim,
            target_dimension=alignment_cfg.proj_embed_dim,
            linear_type=alignment_cfg.sail.linear_type,
            width_factor=alignment_cfg.width_factor
        )
    elif alignment_type == 'sharelock':
        alignment = ShareLock(
            input_dimension=modality_encoder_cfg.embed_dim,
            target_dimension=alignment_cfg.proj_embed_dim,
            linear_type=alignment_cfg.sail.linear_type,
            width_factor=alignment_cfg.width_factor
        )
    elif alignment_type == 'linear':
        alignment = nn.Linear(modality_encoder_cfg.embed_dim, alignment_cfg.proj_embed_dim, bias=False)
    else:
        raise ValueError(f"Unknown alignment type: {alignment_type}")

    # Load checkpoint if available
    if 'ecg_alignment_path' in cfg:
        if cfg.ecg_alignment_path:
            checkpoint = torch.load(cfg.ecg_alignment_path, map_location='cpu', weights_only=False)
            msg = alignment.load_state_dict(checkpoint['model'], strict=False)
            print(f'Successfully loaded ECG alignment: {cfg.ecg_alignment_path}')
            print(msg)
    
    elif 'echo_alignment_path' in cfg:
        if cfg.echo_alignment_path:
            checkpoint = torch.load(cfg.echo_alignment_path, map_location='cpu', weights_only=False)
            msg = alignment.load_state_dict(checkpoint['model'], strict=False)
            print(f'Successfully loaded echo alignment: {cfg.echo_alignment_path}')
            print(msg)
    
    else:
        print('The alignment layer is randomly initialized!')
    
    return alignment

def init_task_layer(cfg: DictConfig) -> nn.Module:
    task_layer = TaskLayer(cfg)
    return task_layer

def init_token_aggregator(cfg: DictConfig, modality: str, modality_cfg: DictConfig) -> nn.Module:
    if modality == 'ecg':
        from ecg.models.token_aggregation.TokenAggregator import TokenAggregator
    
    elif modality == 'echo':
        from echo.models.token_aggregation.TokenAggregator import TokenAggregator
        
    token_aggregator = TokenAggregator(cfg, modality_cfg)

    # Load weights if available
    if cfg.token_aggregator_path:
        checkpoint = torch.load(cfg.token_aggregator_path, map_location='cpu', weights_only=False)
        msg = token_aggregator.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
    else:
        print('The token aggregator is randomly initialized!')

    return token_aggregator