from math import inf

import torch

def count_model_parameters(model):
  total_params = sum(p.numel() for p in model.parameters())

  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

  print(f"Total parameters: {total_params}")
  print(f"Trainable parameters: {trainable_params}")


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm