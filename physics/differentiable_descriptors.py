"""Differentiable image statistics for optional physical feedback."""
from __future__ import annotations
import torch
from torch.nn import functional as F


def _sobel(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8
    ky = kx.transpose(-1, -2)
    rgb = x[:, :3]
    gx = F.conv2d(rgb, kx.expand(3, 1, 3, 3), groups=3)
    gy = F.conv2d(rgb, ky.expand(3, 1, 3, 3), groups=3)
    return gx, gy


def image_descriptors(image: torch.Tensor) -> torch.Tensor:
    """Return smooth RGB/GB descriptors suitable for a training loss.

    These are proxy descriptors, not a replacement for EBSD grain
    segmentation.  Hard connected-component operations are deliberately
    avoided so gradients remain defined.
    """
    if image.ndim != 4 or image.shape[1] < 3:
        raise ValueError('Expected BCHW image with at least RGB channels')
    rgb = image[:, :3].float()
    gx, gy = _sobel(image)
    edge = (gx.square() + gy.square() + 1e-8).sqrt().mean((1, 2, 3))
    smooth = F.avg_pool2d(rgb, 3, stride=1, padding=1)
    texture = (rgb - smooth).abs().mean((1, 2, 3))
    mean = rgb.mean((1, 2, 3))
    std = rgb.std((1, 2, 3), unbiased=False)
    values = [mean, std, edge[:, None], texture[:, None]]
    if image.shape[1] >= 4:
        gb = image[:, 3:4].float()
        values.append(torch.sigmoid(-8 * gb).mean((1, 2, 3))[:, None])
    return torch.cat([v if v.ndim == 2 else v[:, None] for v in values], dim=1)


def descriptor_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Huber loss between predicted-image and target descriptors."""
    return F.smooth_l1_loss(image_descriptors(predicted), target.detach())
