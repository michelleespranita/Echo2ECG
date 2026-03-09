# Copyright 2023 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
from torch.optim.lr_scheduler import _LRScheduler


class LARS(Optimizer):
    r"""Implements LARS (Layer-wise Adaptive Rate Scaling).
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        eta (float, optional): LARS coefficient as used in the paper (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        epsilon (float, optional): epsilon to prevent zero division (default: 0)
    Example:
        >>> optimizer = torch.optim.LARS(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self, params, lr=required, momentum=0, eta=1e-3, dampening=0,
                 weight_decay=0, nesterov=False, epsilon=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, eta=eta, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, epsilon=epsilon)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            dampening = group['dampening']
            nesterov = group['nesterov']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                w_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)
                if w_norm * g_norm > 0:
                    local_lr = eta * w_norm / (g_norm +
                        weight_decay * w_norm + epsilon)
                else:
                    local_lr = 1
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-local_lr * group['lr'])

        return loss


class Lion(Optimizer):
  r"""Implements Lion algorithm."""

  def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
    """Initialize the hyperparameters.

    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
      lr (float, optional): learning rate (default: 1e-4)
      betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.99))
      weight_decay (float, optional): weight decay coefficient (default: 0)
    """

    if not 0.0 <= lr:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
    defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.

    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.

    Returns:
      the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        # Perform stepweight decay
        p.data.mul_(1 - group['lr'] * group['weight_decay'])

        grad = p.grad
        state = self.state[p]
        # State initialization
        if len(state) == 0:
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(p)

        exp_avg = state['exp_avg']
        beta1, beta2 = group['betas']

        # Weight update
        update = exp_avg * beta1 + grad * (1 - beta1)

        p.add_(update.sign_(), alpha=-group['lr'])

        # Decay the momentum running average coefficient
        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

    return loss

class WarmupCosineStepScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_factor=0.1, last_step=-1):
        """
        min_lr_factor: It's a scalar multiplier (typically between 0.0 and 1.0)
        that defines how low the learning rate can go relative to the base_lr (aka initial_lr).

        Specifically:
        During warmup:
        The LR increases linearly from min_lr_factor × base_lr to base_lr.

        During cosine decay:
        The LR decreases from base_lr to min_lr_factor × base_lr following a cosine curve.

        After total_steps:
        The LR remains fixed at min_lr_factor × base_lr.
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer, last_epoch=last_step)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup from min_lr_factor * base_lr to base_lr
            lr_scale = self.min_lr_factor + (1 - self.min_lr_factor) * (self.last_epoch / self.warmup_steps)
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        elif self.last_epoch <= self.total_steps:
            # Cosine decay after warmup
            cos_step = self.last_epoch - self.warmup_steps
            cos_steps = self.total_steps - self.warmup_steps
            cos_theta = cos_step / cos_steps
            cos_theta = min(max(cos_theta, 0), 1)
            return [base_lr * (self.min_lr_factor + (1 - self.min_lr_factor) * 0.5 * (1 + math.cos(math.pi * cos_theta))) for base_lr in self.base_lrs]
        else:
            # After total_steps, maintain the minimum learning rate
            return [base_lr * self.min_lr_factor for base_lr in self.base_lrs]

    def step(self, epoch=None):
        super().step(epoch)

def define_param_groups(models, weight_decay: float, lr: float, layer_decay: float = 1.0):
    def exclude_from_wd(name, module):
        return (
            'bias' in name or
            'layernorm' in name or
            'position_embeddings' in name or
            'mask_token' in name or
            'cls_token' in name or
            'log_temperature' in name or
            'projection_head' in name or
            'linear_probe_layer' in name or
            'linear_layer_reduction' in name or
            'attention_pooling' in name or
            'layerscale' in name or 
            'bn' in name or isinstance(module, (nn.LayerNorm,
                                                nn.BatchNorm1d,
                                                nn.BatchNorm2d,
                                                nn.BatchNorm3d))
        )

    def get_layer_id_for_vit(name, num_layers):
        """
        Assign a parameter with its layer id
        Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        """
        if 'cls_token' in name or 'pos_embed' in name or 'patch_embed' in name or 'conv_proj' in name or 'pos_encoding' in name:
            return 0 # gets the smallest lr
        elif 'blocks' in name:
            return int(name.split('.')[2]) # expects a name like 'echo_encoder.blocks.0' or 'ecg_encoder.blocks.1'
        else:
            return num_layers - 1 # gets base lr (the biggest lr)
        
    if not isinstance(models, list):
        models = [models]

    param_groups = []
    
    for model in models:
        if hasattr(model, 'ecg_encoder'):
            encoder_num_layers = len(model.ecg_encoder.blocks)
        elif hasattr(model, 'echo_encoder'):
            encoder_num_layers = len(model.echo_encoder.blocks)
        else:
            encoder_num_layers = 1
            
        lrs = list(lr * (layer_decay ** (encoder_num_layers - (i + 1))) for i in range(encoder_num_layers)) # apply layer decay just to encoders
        print(lrs)
        # +1 is for the rest of the model after the encoder
        # the first (n-1) lrs are for the encoder, the last (n) lr is for the rest of the model
        
        for name, module in model.named_modules():
            for n, p in module.named_parameters(recurse=False):
                if p.requires_grad:
                    full_name = f"{name}.{n}" if name else n
                    # Apply layer decay?
                    if ('ecg_encoder' in full_name) or ('echo_encoder' in full_name):
                        layer_id = get_layer_id_for_vit(full_name, encoder_num_layers)
                    else:
                        layer_id = -1
                    lr = lrs[layer_id]
                    wd = 0.0 if exclude_from_wd(full_name, module) else weight_decay
                    param_groups.append({
                        'name': full_name,
                        'weight_decay': wd,
                        'lr': lr,
                        'params': p
                    })

        # Display
        for param_group in param_groups:
            for k, v in param_group.items():
                if k != 'params':
                    print(f'{k}: {v}')
            
    return param_groups


def create_optimizer_and_scheduler(models, optimizer_params, num_opt_steps_per_epoch):
    try:
        optimizer_name = next(
            n for n, config in optimizer_params['optimizer'].items() if config.get('use', True)
        )
    except StopIteration:
        raise ValueError("Configuration error: No optimizer is marked for use.")

    try:
        scheduler_name = next(
            n for n, config in optimizer_params['scheduler'].items() if config.get('use', True)
        )
    except StopIteration:
        scheduler_name = None

    param_groups = define_param_groups(models, optimizer_params['weight_decay'], optimizer_params['lr'], optimizer_params['layer_decay'])

    optimizer_constructors = {
        'adamw': lambda params, config: AdamW(params, lr=optimizer_params['lr'], **config),
        'lars': lambda params, config: LARS(params, lr=optimizer_params['lr'], **config),
        'adam': lambda params, config: Adam(params, lr=optimizer_params['lr'], **config),
        'sgd': lambda params, config: SGD(params, lr=optimizer_params['lr'], **config),
    }

    if optimizer_name not in optimizer_constructors:
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported.")

    opt_config = optimizer_params['optimizer'][optimizer_name]
    opt_config = {k: v for k, v in opt_config.items() if k != 'use'}

    optimizer = optimizer_constructors[optimizer_name](param_groups, opt_config)

    if scheduler_name is not None:
        scheduler_constructors = {
            'cosine': lambda opt: CosineAnnealingLR(
                opt,
                T_max=num_opt_steps_per_epoch * optimizer_params['scheduler']['cosine']['T_max'],
                eta_min=optimizer_params['scheduler']['cosine'].get('eta_min', 0.01),
            ),
            'step': lambda opt: StepLR(
                opt,
                step_size=num_opt_steps_per_epoch * optimizer_params['scheduler']['step']['step_size'],
                gamma=optimizer_params['scheduler']['step'].get('gamma', 0.1),
            ),
            'exponential': lambda opt: ExponentialLR(
                opt,
                gamma=optimizer_params['scheduler']['exponential']['gamma']
            ),
            'warmup_cosine': lambda opt: WarmupCosineStepScheduler(
                opt,
                min_lr_factor=optimizer_params['scheduler']['warmup_cosine']['min_lr_factor'],
                warmup_steps=num_opt_steps_per_epoch * optimizer_params['scheduler']['warmup_cosine']['warmup_steps'],
                total_steps=num_opt_steps_per_epoch * optimizer_params['scheduler']['warmup_cosine']['total_steps'],
            ),
        }

        if scheduler_name not in scheduler_constructors:
            raise ValueError(f"Scheduler '{scheduler_name}' is not supported.")

        scheduler = {
            'scheduler': scheduler_constructors[scheduler_name](optimizer),
            'name': f"{scheduler_name}",
            'interval': 'step',
        }
        
    else:
        print('No scheduler is used')
        scheduler = None

    return optimizer, scheduler

def define_param_groups_multiple_lr(model_group_dict, weight_decay_dict, lr_rates, layer_decay):
    def exclude_from_wd(name, module):
        return (
            'bias' in name or
            'layernorm' in name or
            'position_embeddings' in name or
            'mask_token' in name or
            'cls_token' in name or
            'log_temperature' in name or
            'projection_head' in name or
            'linear_probe_layer' in name or
            'linear_layer_reduction' in name or
            'attention_pooling' in name or
            'layerscale' in name or 
            'bn' in name or isinstance(module, (nn.LayerNorm,
                                                nn.BatchNorm1d,
                                                nn.BatchNorm2d,
                                                nn.BatchNorm3d))
        )
    
    def get_layer_id_for_vit(name, num_layers):
        """
        Assign a parameter with its layer id
        Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        """
        if 'cls_token' in name or 'pos_embed' in name or 'patch_embed' in name:
            return 0 # gets the smallest lr
        elif 'blocks' in name:
            return int(name.split('.')[1]) # expects a name like 'blocks.0' or 'blocks.1'
        else:
            return num_layers - 1 # gets base lr (the biggest lr)

    param_groups = []
    
    for group_key, models in model_group_dict.items():
        if not isinstance(models, list):
            models = [models]

        for model in models:
            if group_key == 'ecg_encoder' or group_key == 'echo_encoder':
                encoder_num_layers = len(model.blocks)
                lr = lr_rates[group_key]
                weight_decay = weight_decay_dict[group_key]
            elif group_key == 'multimodal_encoder' or group_key == 'decoder':
                encoder_num_layers = 1
                lr = lr_rates[group_key]
                weight_decay = weight_decay_dict[group_key]
            else:
                encoder_num_layers = 1
                lr = lr_rates['lr']
                weight_decay = weight_decay_dict['lr']
            
            lrs = list(lr * (layer_decay ** (encoder_num_layers - (i + 1))) for i in range(encoder_num_layers)) # apply layer decay just to encoders
            print(lrs)

            if isinstance(model, nn.Parameter): # for things outside of the encoder and decoder
                p = model
                if p.requires_grad:
                    # Apply layer decay?
                    if 'encoder' in group_key:
                        layer_id = get_layer_id_for_vit(name, encoder_num_layers)
                    else:
                        layer_id = 0
                    lr = lrs[layer_id]
                    wd = 0.0 if exclude_from_wd(name, p) else weight_decay
                    param_groups.append({
                        'name': f'{group_key}.{name}',
                        'weight_decay': wd,
                        'lr': lr,
                        'params': p
                    })
            
            elif isinstance(model, nn.Module):
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        # Apply layer decay?
                        if 'ecg_encoder' in group_key or 'echo_encoder' in group_key:
                            layer_id = get_layer_id_for_vit(name, encoder_num_layers)
                        else:
                            layer_id = 0
                        lr = lrs[layer_id]
                        wd = 0.0 if exclude_from_wd(name, p) else weight_decay
                        param_groups.append({
                            'name': f'{group_key}.{name}',
                            'weight_decay': wd,
                            'lr': lr,
                            'params': p
                        })

    # Display
    for param_group in param_groups:
        for k, v in param_group.items():
            if k != 'params':
                print(f'{k}: {v}')
    
    return param_groups

def create_optimizer_and_scheduler_multiple_lr(model_group_dict, optimizer_params, num_opt_steps_per_epoch):
    lr_rates = {
        'lr': optimizer_params['lr'],
        'ecg_encoder': optimizer_params['lr_ecg_encoder'],
        'echo_encoder': optimizer_params['lr_echo_encoder']
    }

    if 'lr_multimodal_encoder' in optimizer_params:
        lr_rates['multimodal_encoder'] = optimizer_params['lr_multimodal_encoder']

    if 'lr_decoder' in optimizer_params:
        lr_rates['decoder'] = optimizer_params['lr_decoder']

    weight_decay_dict = {
        'lr': optimizer_params['weight_decay'],
        'ecg_encoder': optimizer_params['weight_decay_ecg_encoder'],
        'echo_encoder': optimizer_params['weight_decay_echo_encoder']
    }

    if 'weight_decay_multimodal_encoder' in optimizer_params:
        weight_decay_dict['multimodal_encoder'] = optimizer_params['weight_decay_multimodal_encoder']

    if 'weight_decay_decoder' in optimizer_params:
        weight_decay_dict['decoder'] = optimizer_params['weight_decay_decoder']

    try:
        optimizer_name = next(
            n for n, config in optimizer_params['optimizer'].items() if config.get('use', True)
        )
    except StopIteration:
        raise ValueError("Configuration error: No optimizer is marked for use.")

    try:
        scheduler_name = next(
            n for n, config in optimizer_params['scheduler'].items() if config.get('use', True)
        )
    except StopIteration:
        scheduler_name = None

    param_groups = define_param_groups_multiple_lr(
        model_group_dict=model_group_dict, weight_decay_dict=weight_decay_dict,
        lr_rates=lr_rates, layer_decay=optimizer_params['layer_decay']
    )

    optimizer_constructors = {
        'adamw': lambda params, config: AdamW(params, **config),
        'lars': lambda params, config: LARS(params, **config),
        'adam': lambda params, config: Adam(params, **config),
        'sgd': lambda params, config: SGD(params, **config),
    }

    if optimizer_name not in optimizer_constructors:
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported.")

    opt_config = optimizer_params['optimizer'][optimizer_name]
    opt_config = {k: v for k, v in opt_config.items() if k != 'use'}

    optimizer = optimizer_constructors[optimizer_name](param_groups, opt_config)

    if scheduler_name is not None:
        scheduler_constructors = {
            'cosine': lambda opt: CosineAnnealingLR(
                opt,
                T_max=num_opt_steps_per_epoch * optimizer_params['scheduler']['cosine']['T_max'],
                eta_min=optimizer_params['scheduler']['cosine'].get('eta_min', 0.01),
            ),
            'step': lambda opt: StepLR(
                opt,
                step_size=num_opt_steps_per_epoch * optimizer_params['scheduler']['step']['step_size'],
                gamma=optimizer_params['scheduler']['step'].get('gamma', 0.1),
            ),
            'exponential': lambda opt: ExponentialLR(
                opt,
                gamma=optimizer_params['scheduler']['exponential']['gamma']
            ),
            'warmup_cosine': lambda opt: WarmupCosineStepScheduler(
                opt,
                min_lr_factor=optimizer_params['scheduler']['warmup_cosine']['min_lr_factor'],
                warmup_steps=num_opt_steps_per_epoch * optimizer_params['scheduler']['warmup_cosine']['warmup_steps'],
                total_steps=num_opt_steps_per_epoch * optimizer_params['scheduler']['warmup_cosine']['total_steps'],
            ),
        }

        if scheduler_name not in scheduler_constructors:
            raise ValueError(f"Scheduler '{scheduler_name}' is not supported.")

        scheduler = {
            'scheduler': scheduler_constructors[scheduler_name](optimizer),
            'name': f"{scheduler_name}",
            'interval': 'step',
        }
        
    else:
        scheduler = None

    return optimizer, scheduler