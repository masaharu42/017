"""Frozen surrogate interface for differentiable mechanics feedback.

The surrogate must be trained from real composition/curve labels before this
loss is enabled.  No synthetic mechanics targets are created here.
"""
from __future__ import annotations
import torch
from torch import nn
from .differentiable_descriptors import image_descriptors


class MechanicsSurrogate(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 9, hidden: int = 128):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, output_dim))

    def freeze(self) -> 'MechanicsSurrogate':
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self


def mechanics_feedback_loss(predicted_image: torch.Tensor,
                            target_curve: torch.Tensor,
                            surrogate: MechanicsSurrogate,
                            condition: torch.Tensor) -> torch.Tensor:
    """Backpropagate through image descriptors but keep surrogate frozen."""
    features = torch.cat([condition.float(), image_descriptors(predicted_image)], dim=1)
    prediction = surrogate(features)
    return torch.nn.functional.smooth_l1_loss(prediction, target_curve.detach())
