"""Orientation-map encodings used by project 017.

The direct Euler-to-RGB mode is included for paper reproduction.  The
sin/cos mode is safer around angular wrap-around and is the recommended mode
for the first experiment once raw Euler maps are available.
"""
from __future__ import annotations
import torch


def euler_to_channels(euler: torch.Tensor, mode: str = 'sincos') -> torch.Tensor:
    """Convert BCHW Euler angles in radians to normalized image channels."""
    if euler.ndim != 4 or euler.shape[1] != 3:
        raise ValueError('Expected Euler tensor with shape Bx3xHxW')
    if mode == 'rgb':
        return (euler.remainder(2 * torch.pi) / (2 * torch.pi)) * 2 - 1
    if mode == 'sincos':
        return torch.cat((torch.sin(euler), torch.cos(euler)), dim=1)
    raise ValueError("mode must be 'rgb' or 'sincos'")
