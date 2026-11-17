"""Small residual descriptor embedding; preserves the old composition branch."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from ebsd_diffusion.models import ContinuousConditionEmbedding


class DescriptorConditionEmbedding(nn.Module):
    def __init__(self, composition: ContinuousConditionEmbedding, hidden_dim: int = 32):
        super().__init__()
        if hidden_dim != 32:
            raise ValueError("This pilot fixes descriptor hidden_dim=32")
        self.composition = composition
        self.input_dim = composition.input_dim
        self.output_dim = composition.output_dim
        self.hidden_dim = composition.hidden_dim
        # Construction must not advance the matched training RNG stream.
        with torch.random.fork_rng(devices=[]):
            # torch.manual_seed also seeds CUDA; seed only the CPU generator
            # inside this CPU-only fork, preserving every device's RNG stream.
            torch.random.default_generator.manual_seed(20261027)
            self.projection = nn.Sequential(nn.Linear(5, hidden_dim), nn.SiLU(),
                                            nn.Linear(hidden_dim, self.output_dim))
            nn.init.zeros_(self.projection[-1].weight)
            nn.init.zeros_(self.projection[-1].bias)

    def forward(self, condition, drop_mask=None, *, descriptors=None, descriptor_valid=None):
        base = self.composition(condition)
        if descriptors is None or descriptors.shape != (len(condition), 5):
            raise ValueError("Five normalized descriptor values are required")
        if not torch.isfinite(descriptors).all():
            raise ValueError("Non-finite descriptor input")
        valid = (torch.ones(len(condition), device=condition.device, dtype=torch.bool)
                 if descriptor_valid is None else descriptor_valid.bool())
        if valid.shape != (len(condition),):
            raise ValueError("Invalid descriptor validity mask")
        delta = self.projection(descriptors) * valid[:, None]
        combined = base + delta
        if drop_mask is not None:
            null = self.unconditional(len(condition), device=base.device, dtype=base.dtype)
            combined = torch.where(drop_mask[:, None], null, combined)
        return combined

    def unconditional(self, batch_size, device=None, dtype=None):
        return self.composition.unconditional(batch_size, device=device, dtype=dtype)

    def save_pretrained(self, directory):
        directory = Path(directory)
        self.composition.save_pretrained(directory / "composition")
        torch.save(self.projection.state_dict(), directory / "descriptor_projection.bin")
        (directory / "config.json").write_text(json.dumps({
            "model_type": "composition_plus_local5", "descriptor_hidden_dim": 32,
            "descriptor_input_dim": 5, "output_dim": self.output_dim,
            "uncertainty_inputs": False}), "utf-8")

    @classmethod
    def from_pretrained(cls, directory):
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text("utf-8"))
        if config["model_type"] != "composition_plus_local5" or config["uncertainty_inputs"]:
            raise ValueError("Unsupported descriptor encoder")
        composition = ContinuousConditionEmbedding.from_pretrained(directory / "composition")
        if config["descriptor_input_dim"] != 5 or config["output_dim"] != composition.output_dim:
            raise ValueError("Descriptor encoder dimension mismatch")
        model = cls(composition, config["descriptor_hidden_dim"])
        model.projection.load_state_dict(torch.load(directory / "descriptor_projection.bin",
                                                    map_location="cpu", weights_only=True), strict=True)
        return model


class FixedDescriptorCondition(nn.Module):
    """Bind one target to the existing DDIM sampler without changing its RNG."""
    def __init__(self, encoder, normalized):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("normalized", torch.as_tensor(normalized).reshape(1, 5))

    def forward(self, condition):
        return self.encoder(condition, descriptors=self.normalized.to(condition).expand(len(condition), -1))

    def unconditional(self, *args, **kwargs):
        return self.encoder.unconditional(*args, **kwargs)
