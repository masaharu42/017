"""Provenance and strict initialization for matched continuation experiments."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel

from ebsd_diffusion.composition import CompositionConditioner
from ebsd_diffusion.config import resolve_project_path
from ebsd_diffusion.models import ContinuousConditionEmbedding
from ebsd_diffusion.weight_loading import load_state_dict


INITIALIZATION_FILES = {
    "raw_unet": None,
    "unet_config": "config.json",
    "condition_encoder_dir": "config.json",
    "condition_encoder_weights": None,
    "conditioner": None,
    "scheduler_dir": "scheduler_config.json",
    "run_metadata": None,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_initialization(config: dict) -> tuple[dict[str, Path], dict] | tuple[None, None]:
    source = config["train_diffusion"].get("initialization")
    if source is None:
        return None, None
    paths = {}
    hashes = {}
    for key, child in INITIALIZATION_FILES.items():
        path = resolve_project_path(config, source[key]).resolve()
        file_path = path / child if child else path
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing initialization {key}: {file_path}")
        paths[key] = path
        hashes[key] = file_sha256(file_path)
    state = json.loads(paths["run_metadata"].read_text(encoding="utf-8"))
    scaling = float(state["latent_scaling"])
    if not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("Source latent_scaling must be finite and positive.")
    step = int(state["global_step"])
    if step < 0:
        raise ValueError("Source global_step must be nonnegative.")
    return paths, {
        "mode": "raw_weights_only_fresh_optimizer_and_lr_scheduler",
        "source_weight_type": "raw",
        "source_global_step": step,
        "latent_scaling": scaling,
        "paths": {key: str(value) for key, value in paths.items()},
        "sha256": hashes,
    }


def load_frozen_conditioner(paths: dict[str, Path], train_ids: set[int], compositions):
    conditioner = CompositionConditioner.load(paths["conditioner"])
    ids = conditioner.state.training_alloy_ids
    if set(ids) != train_ids:
        raise ValueError("Training alloy IDs differ from the source conditioner; refusing to refit.")
    normalized = [conditioner.transform_by_alloy_id(compositions, value) for value in ids]
    if not np.allclose(normalized, conditioner.state.normalized_training_conditions, rtol=1e-6, atol=1e-6):
        raise ValueError("Composition values differ from the source conditioner training values.")
    return conditioner


def load_initialized_models(paths: dict[str, Path], model_cfg: dict, input_dim: int):
    unet_config = json.loads((paths["unet_config"] / "config.json").read_text("utf-8"))
    unet = UNet2DModel.from_config(unet_config)
    unet.load_state_dict(load_state_dict(paths["raw_unet"]), strict=True)
    encoder_config = json.loads((paths["condition_encoder_dir"] / "config.json").read_text("utf-8"))
    encoder = ContinuousConditionEmbedding(**encoder_config)
    encoder.load_state_dict(load_state_dict(paths["condition_encoder_weights"]), strict=True)
    if encoder.input_dim != input_dim or encoder.output_dim != unet.config.block_out_channels[0] * 4:
        raise ValueError("Source condition encoder dimensions do not match the U-Net/conditioner.")
    if encoder.hidden_dim != int(model_cfg["condition_hidden_dim"]):
        raise ValueError("Condition encoder hidden dimension differs from the training config.")
    if list(unet.config.block_out_channels) != list(model_cfg["block_out_channels"]):
        raise ValueError("Source U-Net block widths differ from the training config.")
    if int(unet.config.layers_per_block) != int(model_cfg["layers_per_block"]):
        raise ValueError("Source U-Net layers_per_block differs from the training config.")
    scheduler = DDPMScheduler.from_pretrained(paths["scheduler_dir"])
    for key in ("num_train_timesteps", "beta_schedule", "prediction_type"):
        if scheduler.config[key] != model_cfg[key]:
            raise ValueError(f"Source scheduler {key} differs from the training config.")
    return unet, encoder, scheduler


def tensor_fingerprint(**tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().contiguous().cpu()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
