from typing import Optional, List

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from io import BytesIO
from PIL import Image

def plot_ecg(
    ecg: np.ndarray | torch.Tensor,
    lead_names: Optional[List[str]] = None,
    fs: int = 500,
    save_path: str = None
):
    """
    Plot a 12-lead ECG signal.

    Args:
        ecg (np.ndarray / torch.Tensor): ECG signal of shape (12, num_timesteps).
        lead_names (list or None): List of 12 lead names. If None, default names are used.
        fs (int): Sampling frequency in Hz (used for x-axis time).
    """
    num_leads, num_timesteps = ecg.shape
    assert num_leads == 12, "Expected 12 leads in the ECG input."

    if lead_names is None:
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    time = np.arange(num_timesteps) / fs  # convert to seconds

    fig, axes = plt.subplots(12, 1, figsize=(24, 20), sharex=True)
    axes = axes.flatten()

    for i in range(12):
        axes[i].plot(time, ecg[i], linewidth=2)
        axes[i].grid(True)
        axes[i].set_ylabel(f"{lead_names[i]}/mV")
        if i >= 10:  # only show x-axis labels on the bottom plots
            axes[i].set_xlabel("Time (s)")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=50)
    else:
        plt.show()
    
    plt.close()

def plot_pred_and_gt_ecg(
    pred_ecg: np.ndarray | torch.Tensor,
    gt_ecg: np.ndarray | torch.Tensor,
    mask: Optional[np.ndarray | torch.Tensor] = None,
    lead_names: Optional[List[str]] = None,
    fs: int = 500
):
    """
    Plot predicted and ground truth ECG signals with masked regions highlighted.

    Args:
        pred_ecg (np.ndarray / torch.Tensor): Predicted ECG of shape (C=1, V, T)
        gt_ecg (np.ndarray / torch.Tensor): Ground truth ECG of shape (C=1, V, T)
        mask (np.ndarray / torch.Tensor): Mask of shape (V, T), 1 for masked, 0 otherwise
        lead_names (list or None): List of 12 lead names
        fs (int): Sampling frequency in Hz
    """
    # Convert tensors to numpy if needed
    if isinstance(pred_ecg, torch.Tensor):
        pred_ecg = pred_ecg.detach().cpu().float().numpy()
    if isinstance(gt_ecg, torch.Tensor):
        gt_ecg = gt_ecg.detach().cpu().float().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Remove channel dimension (assume C=1)
    pred_ecg = pred_ecg[0]
    gt_ecg = gt_ecg[0]

    num_leads, num_timesteps = pred_ecg.shape
    assert num_leads == 12, "Expected 12 leads"

    if lead_names is None:
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    time = np.arange(num_timesteps) / fs

    fig, axes = plt.subplots(12, 1, figsize=(24, 20), sharex=True)
    axes = axes.flatten()

    for i in range(12):
        axes[i].plot(time, gt_ecg[i], label="GT", color="blue", linewidth=2)
        axes[i].plot(time, pred_ecg[i], label="Pred", color="orange", linewidth=2, alpha=0.7)
        axes[i].set_ylabel(f"{lead_names[i]}/mV")
        axes[i].grid(True)

        if mask is not None:
            # Highlight masked regions
            masked_indices = np.where(mask[i] > 0)[0]
            for idx in masked_indices:
                rect_width = 1 / fs
                rect = patches.Rectangle(
                    (time[idx], axes[i].get_ylim()[0]),  # bottom-left corner
                    rect_width,
                    axes[i].get_ylim()[1] - axes[i].get_ylim()[0],
                    linewidth=1,
                    edgecolor='red',
                    facecolor='none'
                )
                axes[i].add_patch(rect)

        if i >= 10:
            axes[i].set_xlabel("Time (s)")
        if i == 0:
            axes[i].legend(loc="upper right")

    plt.tight_layout()
    
    # Save the plot to a BytesIO object
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)  # Free up memory

    # Convert to PIL image
    image = Image.open(buf)
    return image