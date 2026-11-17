"""Prepare, train, or evaluate ONE matched raw50k MSE/Min-SNR pilot.

Default phase is prepare (read-only model checks, new protocol files, no training).
Run phases explicitly and sequentially; never overwrite historical or partial runs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from diffusers import AutoencoderKL, DDIMScheduler
from PIL import Image

from ebsd_diffusion.composition import load_compositions
from ebsd_diffusion.data import EbsdPatchDataset
from ebsd_diffusion.metrics import compare_descriptor_sets, describe_image, descriptors_as_dict
from ebsd_diffusion.training_control import (
    INITIALIZATION_FILES,
    file_sha256,
    load_frozen_conditioner,
    load_initialized_models,
    resolve_initialization,
)
from ebsd_diffusion.utils import save_json
from scripts.sample_fold03_ab2c_256 import build_montage, sample_image, save_png


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARMS = {"A_mse": None, "B_minsnr5": 5.0}


def absolute(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_hparams(path: Path) -> dict:
    flat = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = {}
    for dotted, value in flat.items():
        if dotted.startswith("_"):
            continue
        if isinstance(value, str) and value.startswith(("[", "{")):
            value = json.loads(value)
        target = config
        keys = dotted.split(".")
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value
    return config


def make_arm_configs(protocol: dict, root: Path) -> tuple[dict, dict[str, dict]]:
    source_path = absolute(root, protocol["source_sampling_config"])
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    base = read_hparams(absolute(root, protocol["historical_hparams"]))
    output = absolute(root, protocol["output_root"])
    data_root = absolute(root, protocol.get("data_root") or root)
    for key in ("manifest", "compositions", "heat_treatment"):
        if key in base["data"]:
            base["data"][key] = str(absolute(data_root, base["data"][key]))
    settings = protocol["training"]
    overrides = settings.get("performance", {})
    allowed = {"batch_size", "mixed_precision", "gradient_checkpointing", "num_workers",
               "vae_encode_batch_size", "allow_tf32", "fused_adamw"}
    if set(overrides) - allowed:
        raise ValueError(f"Unknown performance overrides: {sorted(set(overrides) - allowed)}")
    for key, value in overrides.items():
        if key in {"batch_size", "vae_encode_batch_size", "num_workers"}:
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "num_workers" else 1):
                raise ValueError(f"Invalid performance {key}: {value}")
        elif key == "mixed_precision":
            if value not in {"no", "fp16", "bf16"}:
                raise ValueError("mixed_precision must be no, fp16, or bf16.")
        elif not isinstance(value, bool):
            raise ValueError(f"performance {key} must be boolean.")
        target = base["data"] if key == "num_workers" else base["train_diffusion"]
        target[key] = value
    steps = int(settings["additional_steps"])
    warmup = int(settings["warmup_steps"])
    if steps <= 0 or not 0 <= warmup < steps:
        raise ValueError("Require additional_steps > warmup_steps >= 0.")
    if not math.isfinite(float(settings["learning_rate"])) or float(settings["learning_rate"]) <= 0:
        raise ValueError("Pilot learning_rate must be positive.")
    if int(settings["save_every"]) <= 0 or int(settings["log_every"]) <= 0:
        raise ValueError("save_every/log_every must be positive.")
    if base["diffusion"]["prediction_type"] != "epsilon":
        raise ValueError("This pilot is restricted to epsilon prediction.")
    if int(base["train_diffusion"]["gradient_accumulation_steps"]) != 1:
        raise ValueError("Matched pilot currently requires accumulation=1.")
    evaluation = protocol["evaluation"]
    if len(set(evaluation["seeds"])) != len(evaluation["seeds"]) or not evaluation["seeds"]:
        raise ValueError("Evaluation seeds must be non-empty and unique.")
    if len(set(evaluation["alloy_ids"])) != len(evaluation["alloy_ids"]) or not evaluation["alloy_ids"]:
        raise ValueError("Evaluation alloy IDs must be non-empty and unique.")
    heldout_ids = evaluation.get("heldout_alloy_ids", [])
    if (len(set(heldout_ids)) != len(heldout_ids)
            or set(heldout_ids) & set(evaluation["alloy_ids"])):
        raise ValueError("Held-out report IDs must be unique and separate from monitor IDs.")
    if heldout_ids and heldout_ids != [int(protocol["exclude_alloy_id"])]:
        raise ValueError("The separate held-out report must contain only the excluded alloy.")
    if evaluation["sampling_dtype"] not in {"float16", "float32"}:
        raise ValueError("sampling_dtype must be float16 or float32.")
    if evaluation["height"] != base["data"]["patch_size"] or evaluation["width"] != base["data"]["patch_size"]:
        raise ValueError("Primary evaluation must use the training patch size.")
    base["seed"] = int(settings["seed"])
    base["train_diffusion"].update({
        "vae_dir": str(absolute(data_root, source["vae_dir"])),
        "max_steps": steps,
        "learning_rate": float(settings["learning_rate"]),
        "warmup_steps": warmup,
        "save_every": int(settings["save_every"]),
        "log_every": int(settings["log_every"]),
        "controlled_experiment": True,
        "deterministic": bool(settings["deterministic"]),
        "export_raw": True,
        "lambda_gb": 0.0, "lambda_edge": 0.0,
        "initialization": {key: str(absolute(data_root, source[key])) for key in INITIALIZATION_FILES},
    })
    arms = {}
    for name, gamma in ARMS.items():
        config = deepcopy(base)
        config["train_diffusion"]["snr_gamma"] = gamma
        config["train_diffusion"]["output_dir"] = str(output / name)
        arms[name] = config
    assert_matched_configs(arms)
    return source, arms


def assert_matched_configs(arms: dict[str, dict]) -> None:
    normalized = []
    for name, gamma in ARMS.items():
        config = deepcopy(arms[name])
        train = config["train_diffusion"]
        if train.pop("snr_gamma") != gamma:
            raise ValueError(f"Wrong loss weighting for {name}.")
        train.pop("output_dir")
        normalized.append(config)
    if normalized[0] != normalized[1]:
        raise ValueError("A/B training configs differ beyond snr_gamma and output_dir.")


def current_code_files(root: Path) -> list[Path]:
    files = [root / "train_diffusion.py"]
    files += sorted((root / "ebsd_diffusion").glob("*.py"))
    files += [root / "scripts" / name for name in (
        "__init__.py", "run_minsnr_ablation.py", "sample_fold03_ab2c_256.py",
        "evaluate_ablation.py",
        "tune_minsnr_throughput.py",
        "pack_minsnr_review.py",
    )]
    return files


def preflight(protocol: dict, root: Path, arms: dict[str, dict]) -> dict:
    config = deepcopy(arms["A_mse"])
    config["_project_root"] = str(root)
    paths, provenance = resolve_initialization(config)
    assert paths is not None and provenance is not None
    if provenance["source_global_step"] != int(protocol["expected_source_step"]):
        raise ValueError("Source checkpoint step does not match the protocol.")
    if "expected_latent_scaling" in protocol and not math.isclose(
            provenance["latent_scaling"], float(protocol["expected_latent_scaling"]), rel_tol=1e-12):
        raise ValueError("Source latent_scaling differs from the approved checkpoint.")
    for key, expected in protocol.get("expected_source_sha256", {}).items():
        if key not in provenance["sha256"] or provenance["sha256"][key] != expected:
            raise ValueError(f"Source identity differs from the approved raw50k checkpoint: {key}")
    manifest_path = Path(config["data"]["manifest"])
    frame = pd.read_csv(manifest_path)
    selected = frame["use_for_diffusion"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
    ids = set(frame.loc[selected, "alloy_id"].astype(int)) - {int(protocol["exclude_alloy_id"])}
    compositions = load_compositions(config["data"]["compositions"])
    conditioner = load_frozen_conditioner(paths, ids, compositions)
    dataset = EbsdPatchDataset(
        manifest_path, config["data"]["patch_size"], config["data"]["patches_per_image"],
        target="diffusion", conditions={key: np.zeros(conditioner.dimension) for key in ids},
        exclude_alloy_id=int(protocol["exclude_alloy_id"]),
        crop_margins=tuple(config["data"]["crop_margins"]),
    )
    if len(dataset) < int(config["train_diffusion"]["batch_size"]):
        raise ValueError("Dataset is smaller than batch_size with drop_last=True.")
    for row in dataset.rows:
        with Image.open(row["image_path"]) as image:
            top, right, bottom, left = dataset.crop_margins
            if min(image.width - left - right, image.height - top - bottom) < dataset.patch_size:
                raise ValueError(f"Image cannot provide the training patch: {row['image_path']}")
    vae_dir = Path(config["train_diffusion"]["vae_dir"])
    for name, expected in protocol.get("expected_vae_sha256", {}).items():
        if file_sha256(vae_dir / name) != expected:
            raise ValueError(f"VAE identity differs from the approved shared VAE: {name}")
    vae = AutoencoderKL.from_pretrained(vae_dir).cpu().eval()
    unet, encoder, scheduler = load_initialized_models(paths, config["diffusion"], conditioner.dimension)
    scale = 2 ** (len(vae.config.block_out_channels) - 1)
    if int(unet.config.sample_size) != dataset.patch_size // scale:
        raise ValueError("Source U-Net sample size differs from training patch/latent scale.")
    if int(unet.config.in_channels) != int(vae.config.latent_channels):
        raise ValueError("Source U-Net and VAE channels differ.")
    if scheduler.config.clip_sample:
        raise ValueError("Unexpected scheduler clip_sample=True.")
    if not torch.isfinite(torch.cat([p.detach().flatten() for p in encoder.parameters()])).all():
        raise ValueError("Source encoder contains non-finite values.")
    evaluation_dataset = EbsdPatchDataset(
        manifest_path, dataset.patch_size, 1, "evaluation", crop_margins=dataset.crop_margins,
    )
    for alloy_id in (*protocol["evaluation"]["alloy_ids"], *protocol["evaluation"].get("heldout_alloy_ids", [])):
        rows = [row for row in evaluation_dataset.rows if int(row["alloy_id"]) == int(alloy_id)]
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one real image for evaluation alloy {alloy_id}.")
        conditioner.transform_by_alloy_id(compositions, int(alloy_id))
    files = [manifest_path, Path(config["data"]["compositions"])]
    files += [vae_dir / "config.json", *sorted(vae_dir.glob("*.safetensors")), *sorted(vae_dir.glob("*.bin"))]
    files += [Path(row["image_path"]) for row in evaluation_dataset.rows]
    files += [paths[key] / child if child else paths[key] for key, child in INITIALIZATION_FILES.items()]
    files += [absolute(root, protocol["source_sampling_config"]), absolute(root, protocol["historical_hparams"])]
    files += current_code_files(root)
    return {
        "initialization": provenance,
        "training_alloy_ids": sorted(ids),
        "training_dataset_patches_per_epoch": len(dataset),
        "frozen_conditioner": True,
        "checks": {"weights_load_strictly": True, "vae_loads": True, "training_images_accessible": True},
        "file_sha256": {str(path): file_sha256(path) for path in sorted(set(files))},
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()},
    }


def prepare(protocol_path: Path, root: Path = PROJECT_ROOT) -> dict:
    protocol_path = protocol_path.resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    output = absolute(root, protocol["output_root"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output_root; refusing to overwrite {output}")
    _, arms = make_arm_configs(protocol, root)
    report = preflight(protocol, root, arms)
    output.mkdir(parents=True, exist_ok=True)
    config_paths = {}
    for name, config in arms.items():
        path = output / "configs" / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        config_paths[name] = str(path)
        report["file_sha256"][str(path)] = file_sha256(path)
    for path in current_code_files(root):
        destination = output / "source_snapshot" / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        report["file_sha256"][str(destination)] = file_sha256(destination)
    shutil.copy2(protocol_path, output / "configs" / "protocol.yaml")
    report["file_sha256"][str(output / "configs" / "protocol.yaml")] = file_sha256(protocol_path)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "prepared_not_trained",
        "name": protocol["name"],
        "project_root": str(root),
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "protocol": protocol,
        "configs": config_paths,
        "preflight": report,
        "scientific_limit": "One matched continuation pilot; no causal proof from the 013 comparison.",
    }
    try:
        manifest["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        manifest["git_status"] = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL).splitlines()
    except (OSError, subprocess.CalledProcessError):
        manifest["git_revision"] = None
    save_json(manifest, output / "experiment_manifest.json")
    return manifest


def load_prepared(protocol_path: Path, root: Path = PROJECT_ROOT) -> tuple[Path, dict]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    output = absolute(root, protocol["output_root"])
    path = output / "experiment_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["project_root"] != str(root):
        raise ValueError("Prepare again on this machine with a new output_root; paths are machine-local.")
    if file_sha256(protocol_path) != manifest["protocol_sha256"]:
        raise ValueError("Protocol changed after prepare; refusing to run a different experiment.")
    for name, expected in manifest["preflight"]["file_sha256"].items():
        if file_sha256(Path(name)) != expected:
            raise ValueError(f"Input/config/source changed after prepare: {name}")
    arms = {name: yaml.safe_load(Path(path).read_text("utf-8")) for name, path in manifest["configs"].items()}
    assert_matched_configs(arms)
    return output, manifest


def completed_metadata(output: Path, arm: str, manifest: dict) -> dict:
    final = output / arm / "final"
    metadata = json.loads((final / "run_metadata.json").read_text("utf-8"))
    config = yaml.safe_load(Path(manifest["configs"][arm]).read_text("utf-8"))
    if metadata["global_step"] != config["train_diffusion"]["max_steps"]:
        raise ValueError(f"Wrong continuation step count in {arm}.")
    if metadata["snr_gamma"] != ARMS[arm]:
        raise ValueError(f"Wrong loss configuration in {arm}.")
    if metadata["initialization"] != manifest["preflight"]["initialization"]:
        raise ValueError(f"Source initialization differs in {arm}.")
    if metadata["exclude_alloy_id"] != int(manifest["protocol"]["exclude_alloy_id"]):
        raise ValueError(f"Wrong held-out alloy in {arm}.")
    if metadata["training_alloy_ids"] != manifest["preflight"]["training_alloy_ids"]:
        raise ValueError(f"Training alloy IDs differ in {arm}.")
    if metadata["unet_weight_types"].get("unet_raw") != "raw":
        raise ValueError(f"Raw weights are not explicitly identified in {arm}.")
    for path in (final / "unet_raw" / "diffusion_pytorch_model.safetensors",
                 final / "condition_encoder" / "pytorch_model.bin"):
        if not path.is_file():
            raise FileNotFoundError(path)
    return metadata


def verify_pair(output: Path, manifest: dict) -> dict:
    records = {}
    initialization = []
    for arm in ARMS:
        completed_metadata(output, arm, manifest)
        with (output / arm / "input_fingerprints.csv").open(encoding="utf-8", newline="") as handle:
            records[arm] = list(csv.DictReader(handle))
        initialization.append(json.loads((output / arm / "initialization_metadata.json").read_text("utf-8")))
    steps = int(manifest["protocol"]["training"]["additional_steps"])
    interval = int(manifest["protocol"]["training"]["log_every"])
    expected_steps = [step for step in range(1, steps + 1) if step <= 5 or step % interval == 0]
    for rows in records.values():
        if [int(row["additional_step"]) for row in rows] != expected_steps:
            raise ValueError("Missing or duplicated input fingerprints; cannot verify matched inputs.")
    if manifest["protocol"]["training"].get("require_diagnostics", False):
        for arm in ARMS:
            diagnostics = json.loads((output / arm / "training_diagnostics.json").read_text("utf-8"))
            config = yaml.safe_load(Path(manifest["configs"][arm]).read_text("utf-8"))
            samples = steps * int(config["train_diffusion"]["batch_size"])
            if (diagnostics["successful_optimizer_steps"] != steps
                    or diagnostics["sample_count"] != samples
                    or sum(diagnostics["sample_counts_by_t_bin"]) != samples
                    or diagnostics["amp_skipped_steps"] != 0):
                raise ValueError(f"Incomplete training diagnostics: {arm}")
    if records["A_mse"] != records["B_minsnr5"]:
        raise ValueError("A/B sampled input fingerprints differ; do not interpret this as a single-variable result.")
    for key in ("torch", "python", "accelerate", "diffusers", "cuda", "device", "gpu_name",
                "deterministic_algorithms", "world_size", "optimizer_state_loaded"):
        if initialization[0][key] != initialization[1][key]:
            raise ValueError(f"A/B runtime mismatch: {key}")
    return {"status": "passed", "audited_input_batches": len(expected_steps),
            "scope": "first five batches and every logged batch; not an exhaustive all-step audit"}


def train(protocol_path: Path, root: Path = PROJECT_ROOT) -> dict:
    output, manifest = load_prepared(protocol_path, root)
    if manifest["protocol"].get("require_cuda", True) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real pilot. Prepare is allowed on CPU; train on the server.")
    for arm in ARMS:
        directory = output / arm
        if directory.exists() and any(directory.iterdir()):
            if (directory / "final" / "run_metadata.json").is_file():
                completed_metadata(output, arm, manifest)
                print(f"Already complete, preserving {arm}", flush=True)
                continue
            raise FileExistsError(f"Partial run found: {directory}. No automatic resume/overwrite; review first.")
        command = [sys.executable, str(root / "train_diffusion.py"), "--config", manifest["configs"][arm],
                   "--exclude-alloy-id", str(manifest["protocol"]["exclude_alloy_id"]),
                   "--output-dir", str(directory)]
        print(f"Starting {arm}: {command}", flush=True)
        # Keep logs outside the arm: train_diffusion refuses non-empty outputs.
        log_path = output / "console_logs" / f"{arm}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("x", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
        completed_metadata(output, arm, manifest)
    verification = verify_pair(output, manifest)
    save_json(verification, output / "matched_input_verification.json")
    manifest["status"] = "trained_pending_evaluation"
    save_json(manifest, output / "experiment_manifest.json")
    return verification


def scalar_distribution(descriptors) -> dict:
    dictionaries = [descriptors_as_dict(item) for item in descriptors]
    result = {}
    for key, example in dictionaries[0].items():
        if isinstance(example, (float, int)):
            values = np.asarray([item[key] for item in dictionaries], dtype=float)
            result[key] = {"mean": float(values.mean()), "std_population": float(values.std()),
                           "q10": float(np.quantile(values, .1)), "median": float(np.median(values)),
                           "q90": float(np.quantile(values, .9))}
    return result


def reference_boxes(width: int, height: int, size: int, margins: list[int]) -> list[tuple[int, int, int, int]]:
    top, right, bottom, left = margins
    xmax, ymax = width - right - size, height - bottom - size
    if xmax < left or ymax < top:
        raise ValueError("Real image is too small for the evaluation patch.")
    positions = [(left, top), (xmax, top), (left, ymax), (xmax, ymax),
                 ((left + xmax) // 2, (top + ymax) // 2)]
    return list(dict.fromkeys((x, y, x + size, y + size) for x, y in positions))


def paired_comparisons(per_image: list[dict], real_descriptors: dict) -> list[dict]:
    """Seed-paired descriptive errors, not an automatic scientific pass/fail."""
    generated = {(row["alloy_id"], row["seed"], row["method"]): row
                 for row in per_image if "seed" in row}
    rows = []
    for alloy, seed in sorted({(key[0], key[1]) for key in generated}):
        reference_valid = all(item.grain_area_median_px > 0 for item in real_descriptors[alloy])
        record = {"alloy_id": alloy, "seed": seed, "real_windows_valid_regions": reference_valid}
        for field in ("grain_count", "gb_fraction", "edge_density"):
            reference = float(np.median([getattr(item, field) for item in real_descriptors[alloy]]))
            record[f"{field}_real_window_median"] = reference
            errors = {}
            for method in ("start_raw50k", *ARMS):
                item = generated[alloy, seed, method]
                value = float(item["descriptors"][field])
                record[f"{field}_{method}"] = value
                valid = reference_valid and item["valid_regions"] and value > 0 and reference > 0
                errors[method] = abs(math.log(value / reference)) if valid else None
                record[f"{field}_{method}_abs_log_error"] = errors[method]
            a, b = errors["A_mse"], errors["B_minsnr5"]
            record[f"{field}_delta_mse_minus_minsnr"] = a - b if a is not None and b is not None else None
        rows.append(record)
    return rows


def evaluate(protocol_path: Path, root: Path = PROJECT_ROOT, *, heldout: bool = False,
             review_note: Path | None = None) -> dict:
    output, manifest = load_prepared(protocol_path, root)
    verification = verify_pair(output, manifest)
    evaluation_dir = output / ("evaluation_heldout" if heldout else "evaluation")
    if evaluation_dir.exists() and any(evaluation_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite evaluation: {evaluation_dir}")
    config = yaml.safe_load(Path(manifest["configs"]["A_mse"]).read_text("utf-8"))
    config["_project_root"] = str(root)
    paths, provenance = resolve_initialization(config)
    protocol = manifest["protocol"]
    settings = deepcopy(protocol["evaluation"])
    if heldout:
        ids = settings.get("heldout_alloy_ids", [])
        if not ids:
            raise ValueError("This protocol has no separate held-out report.")
        if not (output / "evaluation" / "run_metadata.json").is_file():
            raise ValueError("Complete and review the training-alloy evaluation first.")
        monitor = json.loads((output / "evaluation" / "run_metadata.json").read_text("utf-8"))
        for name, expected in monitor.get("artifact_sha256", {}).items():
            if file_sha256(output / "evaluation" / name) != expected:
                raise ValueError(f"Monitor evaluation artifact changed before ID03: {name}")
        if review_note is None or not review_note.read_text("utf-8").strip():
            raise ValueError("Supply a non-empty --review-note locking the monitor conclusion before ID03.")
        settings["alloy_ids"] = ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if protocol.get("require_cuda", True) and device.type != "cuda":
        raise RuntimeError("CUDA is required for server evaluation; no silent CPU/FP32 fallback.")
    dtype = torch.float16 if settings["sampling_dtype"] == "float16" and device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(config["train_diffusion"]["vae_dir"]).to(device=device, dtype=dtype).eval()
    compositions = load_compositions(config["data"]["compositions"])
    conditioner = load_frozen_conditioner(paths, set(manifest["preflight"]["training_alloy_ids"]), compositions)
    dataset = EbsdPatchDataset(config["data"]["manifest"], settings["height"], 1, "evaluation")
    per_image, summaries, real_descriptors = [], {}, {}
    model_files, sampling_inputs = {}, {}
    for alloy_id in settings["alloy_ids"]:
        row = next(row for row in dataset.rows if int(row["alloy_id"]) == int(alloy_id))
        folder = evaluation_dir / f"alloy_{alloy_id:02d}"
        descriptors, references = [], []
        with Image.open(row["image_path"]) as source:
            source = source.convert("RGB")
            boxes = reference_boxes(source.width, source.height, settings["height"], config["data"]["crop_margins"])
            for index, box in enumerate(boxes):
                path = folder / "real" / f"window_{index}.png"
                save_png(source.crop(box), path)
                descriptor = describe_image(path, settings["gb_threshold"])
                descriptors.append(descriptor)
                references.append((f"window {index}", path))
                per_image.append({"method": "real_reference", "alloy_id": alloy_id,
                                  "path": str(path), "image_sha256": file_sha256(path),
                                  "relative_path": path.relative_to(output).as_posix(),
                                  "box": list(box), "descriptors": descriptors_as_dict(descriptor)})
        real_descriptors[alloy_id] = descriptors
        build_montage(folder / "real_windows.png", [("real windows (not paired seed targets)", references)],
                      (settings["width"], settings["height"]))
        summaries[str(alloy_id)] = {"real_reference": scalar_distribution(descriptors), "methods": {}}
    grids = {alloy_id: [] for alloy_id in settings["alloy_ids"]}
    for method in ("start_raw50k", *ARMS):
        model_paths = dict(paths)
        if method != "start_raw50k":
            final = output / method / "final"
            model_paths.update({"raw_unet": final / "unet_raw" / "diffusion_pytorch_model.safetensors",
                                "unet_config": final / "unet_raw", "condition_encoder_dir": final / "condition_encoder",
                                "condition_encoder_weights": final / "condition_encoder" / "pytorch_model.bin"})
        unet, encoder, _ = load_initialized_models(model_paths, config["diffusion"], conditioner.dimension)
        unet.to(device=device, dtype=dtype).eval()
        encoder.to(device=device, dtype=dtype).eval()
        scheduler = DDIMScheduler.from_pretrained(paths["scheduler_dir"])
        model_files[method] = {key: {"path": str(model_paths[key]), "sha256": file_sha256(model_paths[key])}
                               for key in ("raw_unet", "condition_encoder_weights")}
        for alloy_id in settings["alloy_ids"]:
            condition = torch.tensor(conditioner.transform_by_alloy_id(compositions, alloy_id),
                                     device=device, dtype=dtype).unsqueeze(0)
            descriptors, images = [], []
            for seed in settings["seeds"]:
                sampling_audit = {}
                generated = sample_image(unet, vae, encoder, scheduler, condition, provenance["latent_scaling"],
                                         int(seed), int(settings["steps"]), float(settings["guidance_scale"]),
                                         settings["height"], settings["width"], settings["height"], settings["width"],
                                         sampling_audit=sampling_audit)
                key = f"alloy_{alloy_id}_seed_{seed}"
                if key in sampling_inputs and sampling_inputs[key] != sampling_audit:
                    raise ValueError(f"Initial noise or DDIM grid differs across methods: {key}")
                sampling_inputs[key] = sampling_audit
                path = evaluation_dir / f"alloy_{alloy_id:02d}" / method / f"seed{seed}.png"
                save_png(generated, path)
                descriptor = describe_image(path, settings["gb_threshold"])
                descriptors.append(descriptor)
                images.append((f"seed {seed}", path))
                per_image.append({"method": method, "weight_type": "raw", "alloy_id": alloy_id,
                                  "seed": seed, "path": str(path), "image_sha256": file_sha256(path),
                                  "relative_path": path.relative_to(output).as_posix(),
                                  "valid_regions": descriptor.grain_area_median_px > 0,
                                  "sampling": sampling_audit, "descriptors": descriptors_as_dict(descriptor)})
                print(f"Sampled {method} alloy {alloy_id:02d} seed {seed}", flush=True)
            summaries[str(alloy_id)]["methods"][method] = {
                "image_count": len(descriptors), "scalar": scalar_distribution(descriptors),
                "comparison_to_real_windows": compare_descriptor_sets(real_descriptors[alloy_id], descriptors)}
            grids[alloy_id].append((method, images))
        del unet, encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for alloy_id, rows in grids.items():
        build_montage(evaluation_dir / f"alloy_{alloy_id:02d}" / "comparison_grid.png", rows,
                      (settings["width"], settings["height"]))
    save_json(per_image, evaluation_dir / "per_image_metrics.json")
    save_json(summaries, evaluation_dir / "summary.json")
    save_json(sampling_inputs, evaluation_dir / "sampling_inputs.json")
    paired = paired_comparisons(per_image, real_descriptors)
    save_json(paired, evaluation_dir / "paired_comparison.json")
    with (evaluation_dir / "paired_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    csv_rows = []
    for alloy, entry in summaries.items():
        for method, values in entry["methods"].items():
            csv_rows.append({"alloy_id": int(alloy), "method": method,
                             **{f"{key}_{stat}": value for key, stats in values["scalar"].items()
                                for stat, value in stats.items()}})
    with (evaluation_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "mode": "pure_noise_free_generation", "weight_type": "raw_only",
              "settings": settings, "device": str(device), "actual_dtype": str(dtype),
              "matched_input_verification": verification, "model_files": model_files,
              "report_scope": "exploratory_heldout_after_monitor_review" if heldout else "configured_alloys",
              "protocol_sha256": manifest["protocol_sha256"], "git_revision": manifest.get("git_revision"),
              "generated_image_count": len([row for row in per_image if "seed" in row]),
              "limits": ["One training seed, one fold; pilot only.",
                         "ID03 has been inspected during development; it is not a blind test or proof of whole-chain holdout.",
                         "Five real corner/center windows overlap; they are not independent samples.",
                         "Connected regions after threshold/dilation are not physical grain counts.",
                         "Invalid region masks are flagged; their sentinel count is not a valid log-ratio observation.",
                         "Positive paired delta means lower proxy error for Min-SNR, not a physical acceptance threshold.",
                         "Hue histogram/JS distance is an RGB proxy, not a crystallographic orientation distribution.",
                         "No pixelwise target or target SSIM for free generation; review images and distributions jointly.",
                         "Lower weighted training loss is not an A/B quality comparison."]}
    if heldout:
        shutil.copy2(review_note, evaluation_dir / "monitor_review.txt")
        report["monitor_review_sha256"] = file_sha256(evaluation_dir / "monitor_review.txt")
    report["artifact_sha256"] = {
        path.relative_to(evaluation_dir).as_posix(): file_sha256(path)
        for path in sorted(evaluation_dir.rglob("*")) if path.is_file()
    }
    save_json(report, evaluation_dir / "run_metadata.json")
    manifest["status"] = "heldout_reported_pending_review" if heldout else "evaluated_pending_scientific_review"
    save_json(manifest, output / "experiment_manifest.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fold03_minsnr_pilot.yaml"))
    parser.add_argument("--phase", choices=("prepare", "train", "evaluate", "evaluate-heldout"), default="prepare")
    parser.add_argument("--review-note", type=Path, help="Monitor conclusion file, required only for evaluate-heldout.")
    args = parser.parse_args()
    if args.phase == "evaluate-heldout":
        result = evaluate(args.config.resolve(), heldout=True, review_note=args.review_note)
    else:
        result = {"prepare": prepare, "train": train, "evaluate": evaluate}[args.phase](args.config.resolve())
    protocol = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    summary = {"phase": args.phase, "output_root": str(absolute(PROJECT_ROOT, protocol["output_root"]))}
    if args.phase == "prepare":
        summary.update({"status": result["status"], "checks": result["preflight"]["checks"]})
    elif args.phase == "train":
        summary.update(result)
    else:
        summary.update({"mode": result["mode"], "weight_type": result["weight_type"],
                        "matched_input_verification": result["matched_input_verification"]})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
