"""Load raw U-Net weights and the matching EMA shadow without training."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".safetensors":
        return load_file(str(path))
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dict in {path}, got {type(state).__name__}.")
    return state


def load_ema_shadow(path: str | Path) -> tuple[float, list[torch.Tensor]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "shadow" not in payload:
        raise TypeError(f"EMA checkpoint {path} does not contain a shadow list.")
    shadow = payload["shadow"]
    if not isinstance(shadow, list) or not shadow:
        raise TypeError(f"EMA shadow in {path} is empty or not a list.")
    return float(payload.get("decay", 0.0)), shadow


def apply_ema_shadow_(module: torch.nn.Module, shadow: list[torch.Tensor]) -> None:
    """Copy an EMA shadow into trainable parameters, in registration order."""
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if len(trainable) != len(shadow):
        raise RuntimeError(
            f"EMA shadow has {len(shadow)} tensors, but the module has {len(trainable)} trainable parameters."
        )
    with torch.no_grad():
        for parameter, value in zip(trainable, shadow, strict=True):
            if tuple(parameter.shape) != tuple(value.shape):
                raise RuntimeError(
                    f"EMA shape {tuple(value.shape)} does not match parameter shape {tuple(parameter.shape)}."
                )
            parameter.copy_(value.to(dtype=parameter.dtype))


def max_abs_difference(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    left_keys = set(left)
    right_keys = set(right)
    if left_keys != right_keys:
        missing = sorted(left_keys.symmetric_difference(right_keys))
        raise KeyError(f"State dict keys differ: {missing[:8]}")
    difference = 0.0
    for key in left_keys:
        value = (left[key].float() - right[key].float()).abs().max().item()
        difference = max(difference, float(value))
    return difference
