"""Portable ridge fitting and a frozen differentiable two-target readout."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

TARGET_NAMES = ("peak_stress_MPa", "endpoint_strain_pct")
ALPHAS = (0.001, 0.01, 0.1, 1., 10., 100., 1000.)
SCHEMA = "linear_mechanics_v1"


def fit_ridge(x, y, alpha):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all() or alpha <= 0:
        raise ValueError("Ridge requires finite training data and positive alpha")
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale < 1e-12, 1., scale)
    ym, ys = float(y.mean()), float(y.std())
    if ys < 1e-12:
        ys = 1.
    z = (x - mean) / scale
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ ((y - ym) / ys))
    return {"input_mean": mean.tolist(), "input_scale": scale.tolist(),
            "target_mean": ym, "target_scale": ys, "coefficient": coef.tolist(), "alpha": float(alpha)}


def predict_ridge(model, x):
    return (model["target_mean"] + model["target_scale"] *
            (((np.asarray(x) - model["input_mean"]) / model["input_scale"]) @ model["coefficient"]))


def select_alpha(x, y, alphas=ALPHAS):
    scores = []
    for a in alphas:
        errors = [float(predict_ridge(fit_ridge(np.delete(x, i, 0), np.delete(y, i), a), x[i]) - y[i]) ** 2
                  for i in range(len(y))]
        scores.append(np.mean(errors))
    return float(alphas[int(np.argmin(scores))])


def nested_oof(x, y, alphas=ALPHAS):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(y) < 4:
        raise ValueError("At least four alloys required even for a software-level nested check")
    predictions = []
    for i in range(len(y)):
        tx, ty = np.delete(x, i, 0), np.delete(y, i)
        a = select_alpha(tx, ty, alphas)
        predictions.append(predict_ridge(fit_ridge(tx, ty, a), x[i]))
    return np.asarray(predictions)


def compare_proxy(compositions, features, target, *, influence=True, alphas=ALPHAS):
    c, d, y = map(lambda v: np.asarray(v, dtype=np.float64), (compositions, features, target))
    x = np.column_stack([c, d])
    base, joint = nested_oof(c, y, alphas), nested_oof(x, y, alphas)
    rmse_b, rmse_j = float(np.sqrt(np.mean((base-y)**2))), float(np.sqrt(np.mean((joint-y)**2)))
    gain = 1 - rmse_j / rmse_b if rmse_b > 1e-12 else 0.
    deletion = []
    if influence:
        for i in range(len(y)):
            yy = np.delete(y, i)
            pb = nested_oof(np.delete(c, i, 0), yy, alphas)
            pj = nested_oof(np.delete(x, i, 0), yy, alphas)
            deletion.append(float(np.mean((pb-yy)**2) - np.mean((pj-yy)**2)))
    improved = int(np.sum(np.abs(joint-y) < np.abs(base-y)))
    required = int(np.ceil(12*len(y)/21))
    passed = bool(gain >= .15 and improved >= required and influence and all(v > 0 for v in deletion))
    model = fit_ridge(x, y, select_alpha(x, y, alphas))
    return {"rmse_composition": rmse_b, "rmse_joint": rmse_j, "relative_gain": gain,
            "improved_alloys": improved, "required_improved": required,
            "composition_oof": base.tolist(), "joint_oof": joint.tolist(),
            "deletion_mse_gains": deletion, "prediction_gate_passed": passed}, model


class LinearMechanics(nn.Module):
    """Buffers only: coefficients are frozen, input autograd is retained."""
    def __init__(self, artifact):
        super().__init__()
        if artifact.get("schema") != SCHEMA or artifact.get("target_names") != list(TARGET_NAMES):
            raise ValueError("Unsupported mechanics artifact")
        self.artifact = artifact
        self.composition_names = list(artifact["composition_names"])
        from .soft_mechanics_features import build_feature_extractor
        self.features = build_feature_extractor(artifact["feature_extractor"])
        width = len(self.composition_names) + len(self.features.names)
        models = artifact["models"]
        if len(models) != 2:
            raise ValueError("Expected two mechanics heads")
        if any(m.get('disabled_reason') and active for m, active in
               zip(models, artifact['approved_targets'])):
            raise ValueError('A disabled mechanics head cannot be activated')
        for key in ("input_mean", "input_scale", "coefficient"):
            value = torch.tensor([m[key] for m in models], dtype=torch.float32)
            if value.shape != (2, width) or not torch.isfinite(value).all():
                raise ValueError("Malformed mechanics coefficients")
            self.register_buffer(key, value)
        for key in ("target_mean", "target_scale"):
            value = torch.tensor([m[key] for m in models], dtype=torch.float32)
            if value.shape != (2,) or not torch.isfinite(value).all():
                raise ValueError("Malformed target scales")
            self.register_buffer(key, value)
        if bool((self.input_scale <= 0).any() or (self.target_scale <= 0).any()):
            raise ValueError("Scales must be positive")
        self.register_buffer("active", torch.tensor(artifact["approved_targets"], dtype=torch.bool))
        if self.active.shape != (2,):
            raise ValueError("Expected two target activation flags")
        self.requires_grad_(False).eval()

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text("utf-8")))

    def forward(self, compositions, descriptors):
        if compositions.ndim != 2 or descriptors.ndim != 2 or len(compositions) != len(descriptors):
            raise ValueError("Expected paired composition/feature matrices")
        x = torch.cat([compositions.detach().float(), descriptors.float()], dim=1)
        z = (x[:, None, :] - self.input_mean) / self.input_scale
        return self.target_mean + self.target_scale * (z * self.coefficient).sum(-1)
