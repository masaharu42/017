"""Versioned, differentiable RGB proxies; these are not physical grain labels.

An extractor is an nn.Module mapping BCHW RGB in [-1,1] to BF features.
A future CNN can implement that contract, but must have a new version and a
separately validated checkpoint. No CNN is silently substituted here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
from torch import nn

FEATURE_VERSION = "soft_boundary_three_v1"
FEATURE_NAMES = ("soft_boundary_fraction", "soft_boundary_variation", "interior_s2")


@dataclass(frozen=True)
class FeatureSettings:
    threshold: float = 0.20
    temperature: float = 0.05
    lag: int = 32

    def __post_init__(self):
        import math
        if not 0 < self.threshold < 1 or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid soft-boundary threshold/temperature")
        if isinstance(self.lag, bool) or not isinstance(self.lag, int) or self.lag < 1:
            raise ValueError("lag must be a positive integer")


class SoftMechanicsFeatures(nn.Module):
    names = FEATURE_NAMES
    version = FEATURE_VERSION

    def __init__(self, **settings):
        super().__init__()
        self.settings = FeatureSettings(**settings)

    def metadata(self):
        return {"type": self.version, "names": list(self.names), "settings": asdict(self.settings)}

    def boundary(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[0] == 0:
            raise ValueError("Expected nonempty BCHW RGB")
        if not torch.isfinite(images).all():
            raise ValueError("Non-finite image")
        value = ((images.float() + 1) / 2).amax(dim=1)
        return torch.sigmoid((self.settings.threshold - value) / self.settings.temperature)

    def forward(self, images):
        b = self.boundary(images)
        r = self.settings.lag
        if min(b.shape[-2:]) <= r:
            raise ValueError("S2 lag must fit strictly inside the patch")
        dx = b[:, :-1, 1:] - b[:, :-1, :-1]
        dy = b[:, 1:, :-1] - b[:, :-1, :-1]
        # Subtract the smoothing floor: a constant map has exactly zero variation.
        variation = ((dx.square() + dy.square() + 1e-8).sqrt() - 1e-4).mean((1, 2))
        interior = 1 - b
        s2 = ((interior[:, :, r:] * interior[:, :, :-r]).mean((1, 2))
              + (interior[:, r:, :] * interior[:, :-r, :]).mean((1, 2))) / 2
        return torch.stack([b.mean((1, 2)), variation, s2], dim=1)


def build_feature_extractor(metadata):
    if metadata.get("type") != FEATURE_VERSION:
        raise ValueError("Unknown/unvalidated feature extractor (CNN needs separate validation)")
    module = SoftMechanicsFeatures(**metadata.get("settings", {}))
    if metadata != module.metadata():
        raise ValueError("Feature names/version/settings do not match")
    return module


def group_mean(features, group_size):
    if group_size < 1 or features.ndim != 2 or len(features) % group_size:
        raise ValueError("Expected a whole number of complete feature groups")
    return features.reshape(-1, group_size, features.shape[-1]).mean(1)
