import hydra
import os
import torch

from omegaconf import DictConfig, OmegaConf
from importlib import import_module

from lightning.pytorch.loggers import WandbLogger

from util.misc import set_seed

def configure_experiment_name(cfg: DictConfig):
    if cfg.train.task == 'multimodal_pretrain_clip':
        bs = f'B{cfg.dataset.batch_size * cfg.dataset.accum_iter * cfg.num_gpus}'
        e = f'e{cfg.max_epochs}'

        if cfg.train.encoder.ecg.checkpoint_path is None:
            init = 'Rand'
        else:
            init = 'Uni'
        
        if cfg.train.encoder.echo.checkpoint_path is None:
            init += '_Rand'
        else:
            init += '_Uni'
        
        data_ecg = f'T{cfg.model.ecg.time_steps}'
        data_echo = f'{cfg.model.echo.num_frames}x{cfg.dataset.echo.sampling_rate}'
        
        freeze_n_ecg = f'f{cfg.train.encoder.ecg.freeze_first_n_layers}'
        freeze_n_echo = f'f{cfg.train.encoder.echo.freeze_first_n_layers}'
        token_agg_ecg = f'tok{cfg.model.token_aggregation.ecg.strategy}'
        proj_embed_dim = f'd{cfg.model.alignment.proj_embed_dim}'
        align = f'{cfg.model.alignment.alignment_type}'

        lr = f'lr{cfg.train.params.lr}'
        lr_ecg = f'lrECG{cfg.train.params.lr_ecg_encoder}'
        lr_echo = f'lrEcho{cfg.train.params.lr_echo_encoder}'
        wd = f'wd{cfg.train.params.weight_decay}'
        wd_ecg = f'wdECG{cfg.train.params.weight_decay_ecg_encoder}'
        wd_echo = f'wdEcho{cfg.train.params.weight_decay_echo_encoder}'
        lrd = f'lrd{cfg.train.params.layer_decay}'
        warmup = 'T' if cfg.train.params.scheduler.warmup_cosine.use else 'F'
        early_stop = '' if cfg.train.params.early_stopping.use else 'noES'

        loss = f't{cfg.train.clip_loss.temperature}_L{cfg.train.clip_loss.lambda_0}'

        if cfg.model.echo.view_aggregation.use:
            suffix = f'_std_{cfg.model.echo.view_aggregation.strategy}'
            if cfg.model.echo.view_aggregation.strategy == 'cls':
                suffix += f'{cfg.model.echo.view_aggregation.num_layers}'
            suffix += f'_D{cfg.model.echo.view_aggregation.proj_embed_dim}'
            suffix += f'_v{cfg.model.echo.view_dropout.mask_ratio}' if cfg.model.echo.view_dropout.use else ''
        else:
            suffix = ''
        
        return f'{bs}_{e}_{init}_{data_ecg}_{data_echo}_{freeze_n_ecg}_{freeze_n_echo}_{token_agg_ecg}_{proj_embed_dim}_{align}_{lr}_{lr_ecg}_{lr_echo}_{wd}_{wd_ecg}_{wd_echo}_{lrd}_{warmup}{early_stop}_{loss}{suffix}'
    
    elif cfg.train.task == 'ecg_linearprobe':
        aug = 'aug' if cfg.downstream_task_ecg.apply_augmentations else 'xAug'
        timesteps = cfg.downstream_task_ecg.time_steps

        if cfg.ecg_encoder_checkpoint_path:
            if 'unimodal' in cfg.ecg_encoder_checkpoint_path:
                encoder_type = 'uni'
            else:
                encoder_type = 'mul'
                
            if cfg.encoder_experiment_name is not None:
                encoder_experiment_name = cfg.encoder_experiment_name
            else:
                encoder_experiment_name = ''
        else:
            encoder_type = 'random'
            encoder_experiment_name = ''
        token_agg_layernorm = 'xTkLN' if cfg.token_aggregator_path is None else 'TkLN'
        align_type = 'xAlgn' if cfg.ecg_alignment_path is None else 'algn'

        return f'{aug}_{timesteps}_{encoder_type}_{encoder_experiment_name}_{token_agg_layernorm}_{align_type}'

    elif cfg.train.task == 'multimodal_retrieval':
        model_experiment_name = cfg.model.experiment_name
        if cfg.dataset.phenotype:
            phenotype = f'_{cfg.dataset.phenotype}_tol{cfg.train.tolerance}'
        else:
            phenotype = ''
        return f'{model_experiment_name}{phenotype}'

def configure_save_dir(save_dir: str, project_name: str, experiment_name: str):
    return os.path.join(save_dir, project_name, experiment_name)

def configure_wandb_logger(cfg: DictConfig, project_name: str, experiment_name: str, save_dir: str):
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    os.environ["WANDB__SERVICE_WAIT"] = "300"

    wandb_logger = WandbLogger(
        project=project_name,
        name=experiment_name,
        save_dir=save_dir,
        config=wandb_cfg,
        log_model=False,
        resume='never'
    )

    return wandb_logger

def determine_train_function(cfg):
    task = cfg.train.task

    if 'downstream_task_ecg' in cfg:
        downstream_ecg_dataset = f'-{cfg.downstream_task_ecg.dataset.lower().capitalize()}'
        downstream_ecg_task_type = f'-{cfg.downstream_task_ecg.task_type.lower().capitalize()}'
        if cfg.downstream_task_ecg.target is not None:
            downstream_ecg_target = f'-{cfg.downstream_task_ecg.target.lower().capitalize()}'
        else:
            downstream_ecg_target = ''
    else:
        downstream_ecg_dataset = ''
        downstream_ecg_task_type = ''
        downstream_ecg_target = ''

    task_to_details = {
        'multimodal_pretrain_clip': {
            'module': 'multimodal.train_functions.train_clip',
            'function': 'train_clip',
            'project_name': 'Multimodal-Pretrain-CLIP',
        },
        'ecg_linearprobe': {
            'module': 'ecg.train_functions.linearprobe',
            'function': 'linearprobe',
            'project_name': f'ECG-LinearProbe{downstream_ecg_dataset}{downstream_ecg_task_type}{downstream_ecg_target}'
        },
        'multimodal_retrieval': {
            'module': 'multimodal.train_functions.retrieval',
            'function': 'retrieval',
            'project_name': 'Multimodal-Retrieval',
        }
    }

    details = task_to_details.get(task)
    if details is None:
        raise ValueError(f"Unknown training task specified: {task}")

    module = import_module(details['module'])
    function = getattr(module, details['function'])

    return {
        'function': function,
        'project_name': details['project_name'],
    }

@hydra.main(config_path='configs', config_name='base', version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    
    training_details = determine_train_function(cfg)
    train_function = training_details['function']
    project_name = training_details['project_name']

    if cfg.experiment_name is None:
        experiment_name = configure_experiment_name(cfg)
    else:
        experiment_name = cfg.experiment_name

    if 'resume_from_checkpoint_path' in cfg and cfg.resume_from_checkpoint_path is not None:
        experiment_name += '_res'
    
    save_dir = configure_save_dir(cfg.save_dir, project_name, experiment_name)

    if not os.path.exists(save_dir):
        print(f"{save_dir} doesn't exist")
        os.makedirs(save_dir, exist_ok=True)
    else:
        print(f'{save_dir} already exists')

    wandb_logger = configure_wandb_logger(cfg, project_name, experiment_name, save_dir)

    print('Start training...', flush=True)

    torch.set_float32_matmul_precision('medium')

    if 'resume_from_checkpoint_path' in cfg and cfg.resume_from_checkpoint_path is not None:
        # Load the config of this checkpoint
        ckpt_path = cfg.resume_from_checkpoint_path
        max_epochs = cfg.max_epochs
        accum_iter = cfg.dataset.accum_iter

        config_dir = os.path.dirname(ckpt_path).replace('/checkpoints', '')
        cfg = OmegaConf.load(os.path.join(config_dir, 'config.yaml'))
        OmegaConf.set_struct(cfg, False)
        cfg.resume_from_checkpoint_path = ckpt_path
        cfg.max_epochs = max_epochs
        cfg.dataset.accum_iter = accum_iter
        OmegaConf.set_struct(cfg, True)
    
    print(cfg)

    try:
        train_function(
            cfg,
            wandb_logger,
            save_dir,
            devices=cfg.num_gpus
        )
    finally:
        wandb_logger.experiment.finish()

if __name__ == "__main__":
    main()
