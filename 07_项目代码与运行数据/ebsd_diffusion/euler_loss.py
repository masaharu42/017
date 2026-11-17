"""VAE loss for Euler+GB channels. The RGB trainer does not call this."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ebsd_diffusion.models import sobel_magnitude
from physics.orientation_encoding import normalized_euler_gb


def euler_gb_vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    posterior,
    config: dict,
    kl_weight: float,
) -> dict[str, torch.Tensor]:
    encoding = str(config.get("phi_encoding", "sincos"))
    policy = str(config.get("boundary_euler", "fill_neighbors"))
    predicted_orientation, predicted_boundary = normalized_euler_gb(reconstruction, encoding)
    target_orientation = target[:, :-1]
    target_boundary = target[:, -1:]
    if predicted_orientation.shape != target_orientation.shape:
        raise ValueError("Euler reconstruction and target channel counts differ.")
    per_pixel = (predicted_orientation - target_orientation).abs().mean(dim=1, keepdim=True)
    if policy == "mask_loss":
        grain = (target_boundary < 0.5).to(per_pixel.dtype)
        orientation_l1 = (per_pixel * grain).sum() / grain.sum().clamp_min(1.0)
    elif policy == "fill_neighbors":
        orientation_l1 = per_pixel.mean()
    else:
        raise ValueError("boundary_euler must be 'fill_neighbors' or 'mask_loss'.")

    gb_kind = str(config.get("gb_loss", "bce"))
    if gb_kind == "bce":
        gb_loss = F.binary_cross_entropy_with_logits(predicted_boundary, target_boundary)
    elif gb_kind == "dice":
        probability = torch.sigmoid(predicted_boundary)
        intersection = (probability * target_boundary).sum(dim=(1, 2, 3))
        denominator = probability.sum(dim=(1, 2, 3)) + target_boundary.sum(dim=(1, 2, 3))
        gb_loss = (1.0 - (2.0 * intersection + 1.0e-6) / (denominator + 1.0e-6)).mean()
    else:
        raise ValueError("gb_loss must be 'bce' or 'dice'.")

    probability = torch.sigmoid(predicted_boundary)
    edge_loss = F.l1_loss(sobel_magnitude(probability), sobel_magnitude(target_boundary))
    kl_loss = posterior.kl().mean()
    loss = (
        float(config.get("orientation_l1_weight", 1.0)) * orientation_l1
        + float(config.get("gb_loss_weight", 1.0)) * gb_loss
        + float(config.get("gb_edge_weight", 0.0)) * edge_loss
        + float(kl_weight) * kl_loss
    )
    return {
        "loss": loss,
        "orientation_l1": orientation_l1,
        "gb_loss": gb_loss,
        "gb_edge_l1": edge_loss,
        "kl": kl_loss,
    }
