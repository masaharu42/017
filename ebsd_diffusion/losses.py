"""Opt-in timestep weighting; the default preserves the historical MSE path."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def validate_snr_gamma(gamma: float | None, prediction_type: str) -> None:
    if gamma is None:
        return
    if not math.isfinite(float(gamma)) or float(gamma) <= 0:
        raise ValueError("snr_gamma must be a finite positive number, or null to disable.")
    if prediction_type != "epsilon":
        raise ValueError("Min-SNR in this experiment supports epsilon prediction only.")


def min_snr_weights(
    alphas_cumprod: torch.Tensor, timesteps: torch.Tensor, gamma: float
) -> torch.Tensor:
    validate_snr_gamma(gamma, "epsilon")
    alpha = alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)[timesteps]
    # min(SNR, gamma) / SNR, with the continuous limit 1 at SNR=0.
    return (float(gamma) * (1 - alpha) / alpha.clamp_min(torch.finfo(torch.float32).tiny)).clamp(max=1)


def diffusion_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    *,
    snr_gamma: float | None = None,
    prediction_type: str = "epsilon",
) -> torch.Tensor:
    validate_snr_gamma(snr_gamma, prediction_type)
    if snr_gamma is None:
        return F.mse_loss(prediction.float(), target.float())
    per_sample = F.mse_loss(prediction.float(), target.float(), reduction="none")
    per_sample = per_sample.flatten(1).mean(1)
    weights = min_snr_weights(alphas_cumprod, timesteps, snr_gamma)
    return (per_sample * weights).mean()
