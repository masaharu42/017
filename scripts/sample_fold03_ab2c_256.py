"""Sample Fold 03 A/B2/C2 20k checkpoints at 256 and evaluate raw versus EMA.

This command only samples and scores existing checkpoints. It does not train.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DModel
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TF

from ebsd_diffusion.composition import CompositionConditioner, load_compositions
from ebsd_diffusion.metrics import compare_descriptor_sets, describe_image, descriptors_as_dict
from ebsd_diffusion.models import ContinuousConditionEmbedding
from ebsd_diffusion.utils import center_crop_pil, tensor_to_pil
from ebsd_diffusion.weight_loading import (
    apply_ema_shadow_,
    load_ema_shadow,
    load_state_dict,
    max_abs_difference,
)
from scripts.evaluate_ablation import scalar_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def resolve_path(project_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def code_version(project_root: Path) -> dict[str, Any]:
    tracked = [
        Path("scripts/sample_fold03_ab2c_256.py"),
        Path("ebsd_diffusion/weight_loading.py"),
        Path("ebsd_diffusion/metrics.py"),
        Path("sample.py"),
    ]
    return {
        "git_revision": git_revision(project_root),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "file_sha256": {
            str(path): file_sha256(project_root / path) for path in tracked
        },
    }


def require_file(path: Path | None, label: str) -> Path:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def build_unet(config_dir: Path, weights: dict[str, torch.Tensor]) -> UNet2DModel:
    config = json.loads((config_dir / "config.json").read_text(encoding="utf-8"))
    unet = UNet2DModel.from_config(config)
    unet.load_state_dict(weights)
    return unet


def load_condition_encoder(config_dir: Path, weights_path: Path) -> tuple[ContinuousConditionEmbedding, float]:
    encoder = ContinuousConditionEmbedding.from_pretrained(config_dir)
    exported = {
        key: value.detach().cpu()
        for key, value in encoder.state_dict().items()
    }
    weights = load_state_dict(weights_path)
    difference = max_abs_difference(exported, weights)
    encoder.load_state_dict(weights)
    return encoder, difference


def prepare_weight_pair(model: dict[str, Any]) -> dict[str, Any]:
    """Return raw and EMA states, plus the difference between EMA sources."""
    raw_path = require_file(model["raw_unet"], "raw U-Net")
    shadow_path = require_file(model["ema_shadow"], "EMA shadow")
    raw_state = load_state_dict(raw_path)
    decay, shadow = load_ema_shadow(shadow_path)
    probe = build_unet(model["unet_config"], raw_state)
    apply_ema_shadow_(probe, shadow)
    ema_from_shadow = probe.state_dict()

    reference_difference = None
    reference_path = model.get("raw_reference_unet")
    if reference_path is not None:
        reference_difference = max_abs_difference(raw_state, load_state_dict(reference_path))

    export_path = model.get("ema_export_unet")
    export_difference = None
    if export_path is not None:
        export_state = load_state_dict(require_file(export_path, "EMA export"))
        export_difference = max_abs_difference(ema_from_shadow, export_state)
        ema_state = export_state
        ema_source = str(export_path)
        ema_weight_type = "ema_export"
    else:
        ema_state = ema_from_shadow
        ema_source = str(shadow_path)
        ema_weight_type = "ema_shadow"

    return {
        "raw_state": raw_state,
        "ema_state": ema_state,
        "ema_decay": decay,
        "raw_source": str(raw_path),
        "ema_source": ema_source,
        "ema_weight_type": ema_weight_type,
        "raw_reference_max_abs_diff": reference_difference,
        "ema_shadow_versus_export_max_abs_diff": export_difference,
    }


@torch.no_grad()
def sample_image(
    unet: UNet2DModel,
    vae: AutoencoderKL,
    condition_encoder: ContinuousConditionEmbedding,
    scheduler: DDIMScheduler,
    condition: torch.Tensor,
    latent_scaling: float,
    seed: int,
    steps: int,
    guidance_scale: float,
    generate_height: int,
    generate_width: int,
    crop_height: int,
    crop_width: int,
) -> Image.Image:
    device = condition.device
    dtype = condition.dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler.set_timesteps(steps, device=device)
    vae_scale = 2 ** (len(vae.config.block_out_channels) - 1)
    if generate_height % vae_scale or generate_width % vae_scale:
        raise ValueError(
            f"Generation size {generate_width}x{generate_height} must be divisible by {vae_scale}."
        )
    latent_height = generate_height // vae_scale
    latent_width = generate_width // vae_scale
    latents = torch.randn(
        (1, int(vae.config.latent_channels), latent_height, latent_width),
        generator=generator,
        device=device,
        dtype=dtype,
    ) * scheduler.init_noise_sigma
    conditional = condition_encoder(condition)
    unconditional = condition_encoder.unconditional(1, device=device, dtype=dtype)
    for timestep in scheduler.timesteps:
        latent_input = torch.cat([latents, latents], dim=0)
        class_embedding = torch.cat([unconditional, conditional], dim=0)
        prediction = unet(latent_input, timestep, class_labels=class_embedding).sample
        unguided, guided = prediction.chunk(2)
        prediction = unguided + guidance_scale * (guided - unguided)
        latents = scheduler.step(prediction, timestep, latents, generator=generator).prev_sample
    decoded = vae.decode(latents / float(latent_scaling)).sample
    return center_crop_pil(tensor_to_pil(decoded[0]), crop_height, crop_width)


def replication_pad(height: int, width: int, multiple: int) -> tuple[int, int, int, int]:
    """Return top, bottom, left, right padding that reaches a multiple."""
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    extra_height = (multiple - height % multiple) % multiple
    extra_width = (multiple - width % multiple) % multiple
    top = extra_height // 2
    left = extra_width // 2
    return top, extra_height - top, left, extra_width - left


@torch.no_grad()
def reconstruct_center(vae: AutoencoderKL, image: Image.Image, multiple: int = 8) -> Image.Image:
    tensor = TF.to_tensor(image).unsqueeze(0).to(device=vae.device, dtype=vae.dtype)
    tensor = tensor * 2.0 - 1.0
    top, bottom, left, right = replication_pad(image.height, image.width, multiple)
    padded = F.pad(tensor, (left, right, top, bottom), mode="replicate")
    latents = vae.encode(padded).latent_dist.mode()
    decoded = vae.decode(latents).sample
    cropped = decoded[:, :, top:top + image.height, left:left + image.width]
    return tensor_to_pil(cropped[0])


def save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def build_montage(
    output_path: Path,
    rows: list[tuple[str, list[tuple[str, Path]]]],
    tile_size: tuple[int, int] = (256, 256),
) -> None:
    tile_width, tile_height = tile_size
    label_height = 28
    column_count = max(len(columns) for _, columns in rows)
    canvas = Image.new(
        "RGB",
        (tile_width * column_count, (tile_height + label_height) * len(rows)),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row_index, (row_label, columns) in enumerate(rows):
        for column_index, (name, path) in enumerate(columns):
            left = column_index * tile_width
            top = row_index * (tile_height + label_height)
            image = Image.open(path).convert("RGB")
            image.thumbnail((tile_width, tile_height))
            paste_left = left + (tile_width - image.width) // 2
            paste_top = top + label_height + (tile_height - image.height) // 2
            canvas.paste(image, (paste_left, paste_top))
            if row_index == 0:
                draw.text((left + 4, top + 2), name, fill=(20, 20, 20), font=font)
            if column_index == 0:
                draw.text((left + 4, top + 14), row_label, fill=(20, 20, 20), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def metric_record(path: Path, gb_threshold: float) -> dict[str, Any]:
    descriptor = describe_image(path, gb_threshold=gb_threshold)
    return descriptors_as_dict(descriptor)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "weight_type",
        "image_count",
        "gb_fraction_mean",
        "gb_fraction_std",
        "edge_density_mean",
        "edge_density_std",
        "grain_count_mean",
        "grain_count_std",
        "grain_area_q25_mean",
        "grain_area_median_mean",
        "grain_area_q75_mean",
        "hue_jensen_shannon",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = resolve_path(project_root, config["output_dir"])
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(seed) for seed in config["seeds"]]
    count_per_seed = int(config["count_per_seed"])
    if count_per_seed != 1:
        raise ValueError("This comparison fixes one image per seed so the initial noise matches.")
    crop_height = int(config["height"])
    crop_width = int(config["width"])
    generate_height = int(config.get("generate_height", crop_height))
    generate_width = int(config.get("generate_width", crop_width))
    if generate_height % 16 or generate_width % 16:
        raise ValueError("Generation height and width must be divisible by 16.")
    steps = int(config["steps"])
    guidance_scale = float(config["guidance_scale"])
    gb_threshold = float(config["gb_threshold"])
    alloy_id = int(config["alloy_id"])

    real_path = require_file(resolve_path(project_root, config["real_image"]), "real image")
    vae_dir = require_file(resolve_path(project_root, config["vae_dir"]) / "config.json", "VAE config").parent
    scheduler_dir = require_file(
        resolve_path(project_root, config["scheduler_dir"]) / "scheduler_config.json",
        "scheduler config",
    ).parent
    compositions_csv = require_file(resolve_path(project_root, config["compositions_csv"]), "compositions")

    for model in config["models"]:
        for key in (
            "unet_config",
            "raw_unet",
            "ema_shadow",
            "condition_encoder_dir",
            "condition_encoder_weights",
            "conditioner",
            "run_metadata",
            "previous_checkpoint",
        ):
            model[key] = resolve_path(project_root, model[key])
        for key in ("raw_reference_unet", "ema_export_unet"):
            model[key] = resolve_path(project_root, model.get(key))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    version = code_version(project_root)

    with Image.open(real_path) as source:
        real_rgb = source.convert("RGB")
    real_crop = center_crop_pil(real_rgb, crop_height, crop_width)
    left = (real_rgb.width - crop_width) // 2
    top = (real_rgb.height - crop_height) // 2
    real_crop_path = output_dir / "real_reference.png"
    save_png(real_crop, real_crop_path)

    vae = AutoencoderKL.from_pretrained(vae_dir).to(device=device).eval()
    reconstruction = reconstruct_center(vae, real_crop)
    reconstruction_path = output_dir / "vae_reconstruction.png"
    save_png(reconstruction, reconstruction_path)
    vae = vae.to(dtype=dtype)

    scheduler = DDIMScheduler.from_pretrained(scheduler_dir)
    per_image: list[dict[str, Any]] = []
    model_reports: list[dict[str, Any]] = []
    montage_rows: list[tuple[str, list[tuple[str, Path]]]] = []
    saved_by_seed: dict[int, list[tuple[str, Path]]] = {seed: [] for seed in seeds}

    for model in config["models"]:
        weights = prepare_weight_pair(model)
        metadata = json.loads(Path(model["run_metadata"]).read_text(encoding="utf-8"))
        latent_scaling = float(metadata["latent_scaling"])
        conditioner = CompositionConditioner.load(model["conditioner"])
        composition_frame = load_compositions(compositions_csv)
        normalized = conditioner.transform_by_alloy_id(composition_frame, alloy_id)
        condition_cpu = torch.as_tensor(normalized, dtype=torch.float32).unsqueeze(0)
        encoder, encoder_difference = load_condition_encoder(
            model["condition_encoder_dir"],
            model["condition_encoder_weights"],
        )
        encoder = encoder.to(device=device, dtype=dtype).eval()
        report = {
            "name": model["name"],
            "global_step": metadata.get("global_step"),
            "latent_scaling": latent_scaling,
            "exclude_alloy_id": metadata.get("exclude_alloy_id"),
            "previously_sampled_weight": model["previously_sampled_weight"],
            "previous_checkpoint": str(model["previous_checkpoint"]),
            "condition_encoder_exported_versus_checkpoint_max_abs_diff": encoder_difference,
            "raw_reference_max_abs_diff": weights["raw_reference_max_abs_diff"],
            "ema_decay": weights["ema_decay"],
            "ema_shadow_versus_export_max_abs_diff": weights["ema_shadow_versus_export_max_abs_diff"],
            "columns": {},
        }
        for weight_type, state, source in (
            ("raw", weights["raw_state"], weights["raw_source"]),
            (weights["ema_weight_type"], weights["ema_state"], weights["ema_source"]),
        ):
            column_name = f"{model['name']}_{weight_type if weight_type == 'raw' else 'ema'}"
            unet = build_unet(model["unet_config"], state).to(device=device, dtype=dtype).eval()
            column_dir = output_dir / column_name
            files = []
            for seed in seeds:
                image = sample_image(
                    unet,
                    vae,
                    encoder,
                    scheduler,
                    condition_cpu.to(device=device, dtype=dtype),
                    latent_scaling,
                    seed,
                    steps,
                    guidance_scale,
                    generate_height,
                    generate_width,
                    crop_height,
                    crop_width,
                )
                path = column_dir / f"seed{seed}_n1" / f"alloy_{alloy_id:02d}_seed{seed}_000.png"
                save_png(image, path)
                print(f"saved {column_name} seed {seed}", flush=True)
                files.append(str(path))
                saved_by_seed[seed].append((column_name, path))
                per_image.append(
                    {
                        "method": column_name,
                        "model": model["name"],
                        "weight_type": "raw" if weight_type == "raw" else "ema",
                        "weight_source": source,
                        "checkpoint_metadata": str(model["run_metadata"]),
                        "global_step": metadata.get("global_step"),
                        "latent_scaling": latent_scaling,
                        "seed": seed,
                        "sample_index": 0,
                        "path": str(path),
                        "descriptors": metric_record(path, gb_threshold),
                        "code_version": version,
                    }
                )
            report["columns"][column_name] = {
                "weight_type": "raw" if weight_type == "raw" else "ema",
                "weight_source": source,
                "files": files,
            }
            del unet
            if device.type == "cuda":
                torch.cuda.empty_cache()
        model_reports.append(report)

    real_descriptor = describe_image(real_crop_path, gb_threshold=gb_threshold)
    vae_descriptor = describe_image(reconstruction_path, gb_threshold=gb_threshold)
    per_image.extend(
        [
            {
                "method": "real_center",
                "weight_type": "real",
                "path": str(real_crop_path),
                "seed": None,
                "descriptors": descriptors_as_dict(real_descriptor),
                "crop_box": {"left": left, "top": top, "width": crop_width, "height": crop_height},
                "source_image": str(real_path),
                "source_size": [real_rgb.width, real_rgb.height],
            },
            {
                "method": "vae_reconstruction",
                "weight_type": "vae",
                "path": str(reconstruction_path),
                "seed": None,
                "vae_dir": str(vae_dir),
                "descriptors": descriptors_as_dict(vae_descriptor),
            },
        ]
    )

    summary: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    real_scalar = scalar_summary([real_descriptor])
    vae_scalar = scalar_summary([vae_descriptor])
    vae_comparison = compare_descriptor_sets([real_descriptor], [vae_descriptor])
    summary["real_center"] = {"image_count": 1, "scalar": real_scalar}
    summary["vae_reconstruction"] = {
        "image_count": 1,
        "scalar": vae_scalar,
        "comparison_to_real_center": vae_comparison,
    }
    csv_rows.append(csv_row("real_center", "real", real_scalar, None, 1))
    csv_rows.append(csv_row("vae_reconstruction", "vae", vae_scalar, vae_comparison, 1))

    methods = [
        f"{model['name']}_{suffix}"
        for model in config["models"]
        for suffix in ("raw", "ema")
    ]
    for method in methods:
        descriptors = [
            describe_image(item["path"], gb_threshold=gb_threshold)
            for item in per_image
            if item["method"] == method
        ]
        scalar = scalar_summary(descriptors)
        comparison = compare_descriptor_sets([real_descriptor], descriptors)
        summary[method] = {
            "image_count": len(descriptors),
            "scalar": scalar,
            "comparison_to_real_center": comparison,
        }
        weight_type = next(item["weight_type"] for item in per_image if item["method"] == method)
        csv_rows.append(csv_row(method, weight_type, scalar, comparison, len(descriptors)))

    for seed in seeds:
        columns = [("real_center", real_crop_path), ("vae_reconstruction", reconstruction_path)]
        columns.extend(saved_by_seed[seed])
        montage_rows.append((f"seed {seed}", columns))
    tile_width = 256 if crop_width == crop_height else 375
    tile_height = max(1, round(tile_width * crop_height / crop_width))
    build_montage(output_dir / "comparison_grid.png", montage_rows, (tile_width, tile_height))

    write_summary_csv(output_dir / "summary.csv", csv_rows)
    (output_dir / "per_image_metrics.json").write_text(
        json.dumps(per_image, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": config.get("task", "fold03_ab2c2_20k_raw_versus_ema"),
        "training_started": False,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "fold": int(config["fold"]),
        "alloy_id": alloy_id,
        "held_out_target_image_used": False,
        "evaluation_space": config.get("evaluation_space", "unspecified"),
        "generate_height": generate_height,
        "generate_width": generate_width,
        "crop_height": crop_height,
        "crop_width": crop_width,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "sampler": "DDIM",
        "seeds": seeds,
        "count_per_seed": count_per_seed,
        "gb_threshold": gb_threshold,
        "edge_density_definition": "mean channel-wise Sobel magnitude on RGB in [0, 1], outer pixel excluded, kernel unnormalized",
        "real_image": str(real_path),
        "real_reference_crop": {
            "left": left,
            "top": top,
            "width": crop_width,
            "height": crop_height,
        },
        "vae_input_padding": {
            "multiple": 8,
            "top_bottom_left_right": list(replication_pad(crop_height, crop_width, 8)),
        },
        "vae_dir": str(vae_dir),
        "scheduler_dir": str(scheduler_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "sampling_dtype": str(dtype),
        "vae_reconstruction_dtype": "torch.float32",
        "code_version": version,
        "models": model_reports,
        "excluded_from_main_table": ["B3", "C3"],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_metadata


def csv_row(
    name: str,
    weight_type: str,
    scalar: dict[str, dict[str, float]],
    comparison: dict[str, Any] | None,
    image_count: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "weight_type": weight_type,
        "image_count": image_count,
        "gb_fraction_mean": scalar["gb_fraction"]["mean"],
        "gb_fraction_std": scalar["gb_fraction"]["std"],
        "edge_density_mean": scalar["edge_density"]["mean"],
        "edge_density_std": scalar["edge_density"]["std"],
        "grain_count_mean": scalar["grain_count"]["mean"],
        "grain_count_std": scalar["grain_count"]["std"],
        "grain_area_q25_mean": scalar["grain_area_q25_px"]["mean"],
        "grain_area_median_mean": scalar["grain_area_median_px"]["mean"],
        "grain_area_q75_mean": scalar["grain_area_q75_px"]["mean"],
        "hue_jensen_shannon": None if comparison is None else comparison["hue_jensen_shannon"],
    }


def main() -> None:
    metadata = run(parse_args().config)
    print(json.dumps({"output_dir": metadata["output_dir"], "device": metadata["device"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
