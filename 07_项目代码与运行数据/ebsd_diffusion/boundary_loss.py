"""Pixel-aligned soft Dice on RGB boundary proxies, for experimental use."""
from __future__ import annotations

import torch


def boundary_dice_loss(predicted: torch.Tensor, target: torch.Tensor, *,
                       threshold: float = 0.2, temperature: float = 0.05,
                       eps: float = 1e-6) -> torch.Tensor:
    """Compare black boundary pixels; images have BCHW layout and [-1, 1] range.

    The target is a hard mask without dilation, while the prediction is soft so
    gradients pass through the frozen VAE decoder.  A score of zero is not a
    guarantee of physically correct grain topology.
    """
    if predicted.ndim != 4 or predicted.shape != target.shape or predicted.shape[1] != 3:
        raise ValueError("Expected matching BCHW RGB tensors")
    if not 0 < threshold < 1 or temperature <= 0 or eps <= 0:
        raise ValueError("Invalid threshold, temperature or epsilon")
    pred_rgb = (predicted.float() + 1.0) * 0.5
    target_rgb = (target.detach().float() + 1.0) * 0.5
    pred_boundary = torch.sigmoid((threshold - pred_rgb.amax(dim=1)) / temperature)
    true_boundary = (target_rgb.amax(dim=1) < threshold).to(pred_boundary.dtype)
    intersection = (pred_boundary * true_boundary).sum(dim=(1, 2))
    mass = pred_boundary.sum(dim=(1, 2)) + true_boundary.sum(dim=(1, 2))
    return (1.0 - (2.0 * intersection + eps) / (mass + eps)).mean()
