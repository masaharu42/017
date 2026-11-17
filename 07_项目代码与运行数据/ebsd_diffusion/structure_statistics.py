"""Differentiable finite-window boundary density and two-point statistics.

S2 is a spatial statistic, not a guarantee of grain topology. RGB in [-1, 1].
Offsets never wrap across patch edges; no hard segmentation or RNG is used.
"""
from __future__ import annotations

import math
import torch


def validate_statistics_config(cfg: dict) -> dict:
    result = dict(cfg)
    for name in ("max_t", "max_samples", "calibration_batches"):
        value = result.get(name, {"max_t": 300, "max_samples": 8, "calibration_batches": 8}[name])
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "max_t" else 1):
            raise ValueError(f"Invalid structure statistics {name}")
        result[name] = value
    for name, default in (("threshold", .20), ("temperature", .05), ("density_floor", .02),
                          ("gradient_ratio", .1)):
        value = float(result.get(name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid structure statistics {name}")
        result[name] = value
    if result["threshold"] >= 1 or result["density_floor"] > 1:
        raise ValueError("Threshold/density floor must be fractions")
    distances = list(result.get("distances", [1, 2, 4, 8, 16, 32, 64]))
    if not distances or any(isinstance(r, bool) or not isinstance(r, int) or r <= 0 for r in distances):
        raise ValueError("Distances must be positive integers")
    if len(set(distances)) != len(distances):
        raise ValueError("Distances must be unique")
    result["distances"] = distances
    return result


def soft_boundary(images: torch.Tensor, threshold: float, temperature: float) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 3 or images.shape[0] == 0:
        raise ValueError("Expected nonempty BCHW RGB images")
    if not 0 < threshold < 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid soft boundary parameters")
    rgb = (images.float() + 1) / 2
    return torch.sigmoid((threshold - rgb.amax(dim=1)) / temperature)


def boundary_statistics(boundary: torch.Tensor, distances) -> tuple[torch.Tensor, torch.Tensor]:
    if boundary.ndim != 3:
        raise ValueError("Expected BHW boundary probabilities")
    height, width = boundary.shape[-2:]
    correlations = []
    for r in distances:
        if not isinstance(r, int) or isinstance(r, bool) or r <= 0 or r >= min(height, width):
            raise ValueError("Each distance must fit strictly within both patch dimensions")
        # Four directions. Diagonal displacements have Euclidean length sqrt(2)*r.
        for dy, dx in ((0, r), (r, 0), (r, r), (r, -r)):
            a = boundary[:, max(0, dy):height + min(0, dy), max(0, dx):width + min(0, dx)]
            b = boundary[:, max(0, -dy):height - max(0, dy), max(0, -dx):width - max(0, dx)]
            correlations.append((a * b).mean(dim=(1, 2)))
    if not correlations:
        raise ValueError("At least one distance required")
    return boundary.mean(dim=(1, 2)), torch.stack(correlations, dim=1)


def structure_statistics_loss(predicted: torch.Tensor, target: torch.Tensor, *,
                              distances=(1, 2, 4, 8, 16, 32, 64), threshold=.20,
                              temperature=.05, density_floor=.02):
    """L=(|rho_p-rho_t| + mean_delta|S2_p-S2_t|)/max(rho_t,floor).

    Both images use the SAME soft transform. Target statistics/scale are detached.
    Target-density normalization avoids suppressing sparse-boundary windows.
    """
    if predicted.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")
    if not math.isfinite(density_floor) or not 0 < density_floor <= 1:
        raise ValueError("Invalid density floor")
    rho_p, s2_p = boundary_statistics(soft_boundary(predicted, threshold, temperature), distances)
    with torch.no_grad():
        rho_t, s2_t = boundary_statistics(soft_boundary(target.detach(), threshold, temperature), distances)
        scale = rho_t.clamp_min(density_floor)
    density = ((rho_p - rho_t).abs() / scale).mean()
    correlation = ((s2_p - s2_t).abs().mean(dim=1) / scale).mean()
    return density + correlation, {"density_loss": density, "correlation_loss": correlation,
                                    "pred_density": rho_p.mean(), "target_density": rho_t.mean()}


def x0_from_epsilon(noisy, prediction, timesteps, alphas_cumprod):
    alpha_bar = alphas_cumprod.to(noisy.device)[timesteps].float().view(-1, 1, 1, 1)
    return (noisy.float() - (1 - alpha_bar).sqrt() * prediction.float()) / alpha_bar.sqrt()


def gradient_measurement(mse, auxiliary, parameters):
    """Observe gradients without changing .grad, model parameters or RNG."""
    parameters = list(parameters)
    gm = torch.autograd.grad(mse, parameters, retain_graph=True, allow_unused=True)
    ga = torch.autograd.grad(auxiliary, parameters, retain_graph=True, allow_unused=True)
    zero = mse.new_zeros((), dtype=torch.float32)
    nm = sum((g.detach().float().square().sum() for g in gm if g is not None), zero).sqrt()
    na = sum((g.detach().float().square().sum() for g in ga if g is not None), zero).sqrt()
    dot = sum(((m.detach().float() * a.detach().float()).sum()
               for m, a in zip(gm, ga) if m is not None and a is not None), zero)
    if not bool(torch.isfinite(torch.stack([nm, na, dot])).all()) or min(float(nm), float(na)) <= 0:
        raise RuntimeError("Structure statistics gradient is non-finite or zero")
    return {"mse_grad_norm": float(nm), "statistics_grad_norm": float(na),
            "gradient_cosine": float(dot / (nm * na)), "ratio": float(nm / na)}
