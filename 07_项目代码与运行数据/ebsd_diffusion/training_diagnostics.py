"""Detached interval statistics; never change losses, gradients, or RNG streams."""

from __future__ import annotations

import numpy as np
import torch

from ebsd_diffusion.losses import min_snr_weights


class TrainingDiagnostics:
    def __init__(self, num_timesteps: int, bins: int = 10):
        self.num_timesteps = int(num_timesteps)
        self.bins = min(int(bins), self.num_timesteps)
        if self.bins <= 0:
            raise ValueError("Diagnostic bins must be positive.")
        self.total = np.zeros((4, self.bins), dtype=np.float64)
        self.interval = np.zeros_like(self.total)
        self.steps = self.clipped_steps = 0
        self.gradient_sum = 0.0
        self.interval_steps = self.interval_clipped = 0
        self.interval_gradient_sum = 0.0
        self.last_flush = 0

    @torch.no_grad()
    def record(self, prediction, target, timesteps, alphas_cumprod, gamma, gradient_norm):
        errors = (prediction.detach().float() - target.detach().float()).square().flatten(1).mean(1)
        weights = torch.ones_like(errors) if gamma is None else min_snr_weights(alphas_cumprod, timesteps, gamma)
        # CPU binning avoids nondeterministic CUDA weighted-bincount kernels.
        indices = (timesteps.detach().cpu().numpy() * self.bins // self.num_timesteps).astype(int)
        values = errors.cpu().double().numpy()
        scales = weights.cpu().double().numpy()
        norm = float(gradient_norm)
        if not np.isfinite(norm) or not np.isfinite(values).all() or not np.isfinite(scales).all():
            raise ValueError("Non-finite training diagnostics.")
        update = np.stack([np.bincount(indices, weights=v, minlength=self.bins)
                           for v in (np.ones_like(values), values, values * scales, scales)])
        self.total += update
        self.interval += update
        self.steps += 1
        self.clipped_steps += int(norm > 1.0)
        self.gradient_sum += norm
        self.interval_steps += 1
        self.interval_clipped += int(norm > 1.0)
        self.interval_gradient_sum += norm

    def flush(self, step: int) -> tuple[dict, list[dict]]:
        if not self.interval_steps:
            return {}, []
        samples = float(self.interval[0].sum())
        scalars = {
            "grad_norm_mean": self.interval_gradient_sum / self.interval_steps,
            "clip_fraction": self.interval_clipped / self.interval_steps,
            "mean_snr_weight": float(self.interval[3].sum() / samples),
            "interval_unweighted_mse": float(self.interval[1].sum() / samples),
            "interval_weighted_mse": float(self.interval[2].sum() / samples),
        }
        rows = []
        for index in range(self.bins):
            count = int(self.interval[0, index])
            rows.append({
                "step_start": self.last_flush + 1, "step_end": step,
                "t_min": (index * self.num_timesteps + self.bins - 1) // self.bins,
                "t_max": ((index + 1) * self.num_timesteps + self.bins - 1) // self.bins - 1,
                "sample_count": count,
                "unweighted_mse": float(self.interval[1, index] / count) if count else None,
                "weighted_mse": float(self.interval[2, index] / count) if count else None,
                "mean_weight": float(self.interval[3, index] / count) if count else None,
            })
        self.last_flush = step
        self.interval.fill(0)
        self.interval_steps = self.interval_clipped = 0
        self.interval_gradient_sum = 0.0
        return scalars, rows

    def summary(self) -> dict:
        return {
            "successful_optimizer_steps": self.steps,
            "sample_count": int(self.total[0].sum()),
            "sample_counts_by_t_bin": self.total[0].astype(int).tolist(),
            "grad_norm_mean": self.gradient_sum / self.steps if self.steps else None,
            "clipped_steps": self.clipped_steps,
            "clip_fraction": self.clipped_steps / self.steps if self.steps else None,
            "gradient_norm_scope": "U-Net and encoder combined, before global clipping at 1.0",
            "amp_skipped_steps": 0,
            "amp_skip_policy": "controlled runs stop on the first skipped update",
        }
