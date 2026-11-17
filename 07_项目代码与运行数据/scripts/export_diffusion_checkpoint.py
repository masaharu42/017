"""Export an Accelerate diffusion checkpoint for sampling.

The checkpoint contains raw U-Net weights in model.safetensors and the EMA
shadow weights in custom_checkpoint_0.pkl. This utility exports either one to
the same diffusers layout consumed by sample.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from diffusers import UNet2DModel
from safetensors.torch import load_file

from ebsd_diffusion.models import ContinuousConditionEmbedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", choices=("raw", "ema"), required=True)
    return parser.parse_args()


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    reference = args.reference_final.resolve()
    output = args.output.resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing raw U-Net weights: {checkpoint / 'model.safetensors'}")
    if not (checkpoint / "model_1.safetensors").is_file():
        raise FileNotFoundError(f"Missing condition encoder weights: {checkpoint / 'model_1.safetensors'}")
    if args.weights == "ema" and not (checkpoint / "custom_checkpoint_0.pkl").is_file():
        raise FileNotFoundError(f"Missing EMA state: {checkpoint / 'custom_checkpoint_0.pkl'}")

    output.mkdir(parents=True, exist_ok=True)
    for name in ("composition_conditioner.json", "run_metadata.json"):
        copy_file(reference / name, output / name)
    copy_file(reference / "scheduler" / "scheduler_config.json", output / "scheduler" / "scheduler_config.json")
    copy_file(reference / "unet" / "config.json", output / "unet" / "config.json")
    copy_file(reference / "condition_encoder" / "config.json", output / "condition_encoder" / "config.json")

    unet = UNet2DModel.from_config(reference / "unet" / "config.json")
    unet.load_state_dict(load_file(str(checkpoint / "model.safetensors")))
    if args.weights == "ema":
        ema_state = torch.load(checkpoint / "custom_checkpoint_0.pkl", map_location="cpu", weights_only=False)
        shadow = ema_state["shadow"]
        parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]
        if len(parameters) != len(shadow):
            raise RuntimeError(f"EMA parameter count mismatch: model={len(parameters)}, shadow={len(shadow)}")
        with torch.no_grad():
            for parameter, value in zip(parameters, shadow, strict=True):
                parameter.copy_(value.to(dtype=parameter.dtype))
    unet.save_pretrained(output / "unet", safe_serialization=True)

    condition_encoder = ContinuousConditionEmbedding.from_pretrained(reference / "condition_encoder")
    condition_encoder.load_state_dict(load_file(str(checkpoint / "model_1.safetensors")))
    condition_encoder.save_pretrained(output / "condition_encoder")

    metadata_path = output / "export_metadata.json"
    metadata_path.write_text(json.dumps({"checkpoint": str(checkpoint), "weights": args.weights}, indent=2), encoding="utf-8")
    print(f"exported {args.weights} weights to {output}")


if __name__ == "__main__":
    main()
