import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
  """
  # https://github.com/oetu/MMCL-ECG-CMR
  Loss function for multimodal contrastive learning based off of the CLIP paper.
  
  Embeddings are taken, L2 normalized and dot product between modalities is calculated to generate a cosine
  similarity between all combinations of subjects in a cross-modal fashion. Tempered by temperature.
  Loss is calculated attempting to match each subject's embeddings between the modalities i.e. the diagonal. 
  """
  def __init__(
        self, 
        temperature: float,
        lambda_0: float = 0.5,
        learnable_temperature: bool = False
    ) -> None:
    super(CLIPLoss, self).__init__()

    self.learnable_temperature = learnable_temperature
    if self.learnable_temperature:
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
    else:
        self.temperature = temperature
    self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    if lambda_0 > 1 or lambda_0 < 0:
      raise ValueError('lambda_0 must be a float between 0 and 1.')
    self.lambda_0 = lambda_0
    self.lambda_1 = 1 - lambda_0

  def forward(self, ecg_token_dict: dict, echo_token_dict: dict):
    ecg_global_token_aligned = ecg_token_dict['ecg_global_token_aligned']
    echo_global_token_aligned = echo_token_dict['echo_global_token_aligned']

    # normalize the embedding onto the unit hypersphere
    out0 = F.normalize(ecg_global_token_aligned, dim=1) # (B, D)
    out1 = F.normalize(echo_global_token_aligned, dim=1) # (B, D)

    if self.learnable_temperature:
        self.logit_scale.data.clamp_(0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits = torch.matmul(out0, out1.T) * logit_scale
    else:
        logits = torch.matmul(out0, out1.T) / self.temperature # (B, B)
    labels = torch.arange(len(out0), device=out0.device)
    
    loss_0 = self.lambda_0 * self.cross_entropy(logits, labels)
    loss_1 = self.lambda_1 * self.cross_entropy(logits.T, labels)
    loss = loss_0 + loss_1
  
    return loss
