"""Local-five RGB proxy pilot: prepare on CPU; explicit server train/evaluate.

Reuses A_mse after checks, or trains a fresh same-device MSE/local-five pair.
Never launches the old Min-SNR phases, trains ID03, or overwrites a run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
import yaml

from ebsd_diffusion.data import EbsdPatchDataset
from ebsd_diffusion.local_descriptors import (
    NAMES, SCHEMA, describe_rgb, segment_native, validate_settings,
)
from ebsd_diffusion.training_control import file_sha256
from ebsd_diffusion.utils import save_json
from scripts.run_minsnr_ablation import (
    absolute, completed_metadata, current_code_files, preflight,
)

ROOT = Path(__file__).resolve().parents[1]
NEW_CODE = ["scripts/run_descriptor_pilot.py", "ebsd_diffusion/local_descriptors.py",
            "ebsd_diffusion/descriptor_conditioning.py"]


def read_json(path):
    return json.loads(Path(path).read_text("utf-8"))


def read_protocol(path, root=ROOT):
    protocol = yaml.safe_load(Path(path).read_text("utf-8"))
    if protocol.get("comparison_mode", "reuse_historical") not in {"reuse_historical", "fresh_pair"}:
        raise ValueError("Unknown comparison_mode")
    if protocol["descriptor_hidden_dim"] != 32 or protocol.get("uncertainty_inputs") is not False:
        raise ValueError("Pilot fixes hidden_dim=32 and excludes uncertainty inputs")
    if not protocol.get("require_cuda", True) and not protocol.get("software_test_only", False):
        raise ValueError("Only synthetic software tests may disable the server CUDA requirement")
    settings = protocol["evaluation"]
    ids, seeds = settings["alloy_ids"], settings["seeds"]
    if not ids or len(set(ids)) != len(ids) or 3 in ids:
        raise ValueError("Only unique training-alloy monitor IDs; no ID03")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Unique evaluation seeds required")
    if settings["target_quantiles"] != [0.1, 0.5, 0.9]:
        raise ValueError("Pilot fixes three observed target vectors at D50 quantiles")
    validate_settings(protocol.get("descriptor_settings", {}))
    if protocol["normalizer_windows_per_alloy"] < 3:
        raise ValueError("Need at least three normalization windows per alloy")
    execution = protocol.get("execution", {})
    allowed = {"gradient_checkpointing", "min_total_vram_gib", "min_free_vram_gib"}
    if set(execution) - allowed:
        raise ValueError("Only checkpointing and memory guards may change; batch/precision remain matched")
    if "gradient_checkpointing" in execution and not isinstance(execution["gradient_checkpointing"], bool):
        raise ValueError("gradient_checkpointing must be boolean")
    for key in ("min_total_vram_gib", "min_free_vram_gib"):
        if key in execution and (isinstance(execution[key], bool) or not isinstance(execution[key], (int, float))
                                 or not np.isfinite(execution[key]) or execution[key] <= 0):
            raise ValueError(f"Invalid execution {key}")
    return protocol, absolute(root, protocol["output_root"])


def execution_config(config, protocol):
    result = deepcopy(config)
    execution = protocol.get("execution", {})
    if "gradient_checkpointing" in execution:
        result["train_diffusion"]["gradient_checkpointing"] = execution["gradient_checkpointing"]
    return result


def device_profile(protocol):
    """Read real device capacity; a larger VRAM number is never fabricated."""
    execution = protocol.get("execution", {})
    if not execution:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("Large-VRAM profile requires a server CUDA device")
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    gib = 1024 ** 3
    report = {"gpu_name": props.name, "total_vram_gib": total / gib,
              "free_vram_gib": free / gib, "compute_capability": list(torch.cuda.get_device_capability(device)),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "compiled_architectures": torch.cuda.get_arch_list()}
    for threshold, observed in (("min_total_vram_gib", total / gib), ("min_free_vram_gib", free / gib)):
        if observed < execution.get(threshold, 0):
            raise RuntimeError(f"GPU memory guard failed: {threshold}={execution[threshold]}, actual={observed:.2f} GiB; {props.name}")
    # A tiny nonrandom kernel catches incompatible CUDA builds without advancing
    # the training RNG or loading any experiment model onto the GPU.
    try:
        a = torch.ones((32, 32), device=device, dtype=torch.float16)
        result = a @ a
        if not bool(torch.isfinite(result).all()) or float(result[0, 0]) != 32:
            raise RuntimeError("FP16 kernel self-check failed")
        del a, result
    except RuntimeError as exc:
        raise RuntimeError(f"GPU/runtime compatibility check failed on {props.name}; verify this GPU's PyTorch/CUDA build") from exc
    report["fp16_kernel_check"] = "passed"
    return report


def verify_baseline(protocol, root=ROOT):
    old_root = absolute(root, protocol["baseline_run"])
    old_manifest = read_json(old_root / "experiment_manifest.json")
    config_path = Path(old_manifest["configs"]["A_mse"])
    config = yaml.safe_load(config_path.read_text("utf-8"))
    report = old_manifest["preflight"]
    if file_sha256(config_path) != report["file_sha256"][str(config_path)]:
        raise ValueError("Baseline arm config changed")
    # Code is deliberately being extended. Verify the archived old source rather
    # than pretending that current train_diffusion.py still has the old hash.
    old_project = Path(old_manifest["project_root"])
    for name, expected in report["file_sha256"].items():
        source = Path(name)
        if (source.suffix == ".py" and source.is_relative_to(old_project)
                and not source.is_relative_to(old_root / "source_snapshot")):
            relative = source.relative_to(old_project)
            source = old_root / "source_snapshot" / relative
        if not source.is_file() or file_sha256(source) != expected:
            raise ValueError(f"Baseline source/data/config archive mismatch: {source}")
    metadata = completed_metadata(old_root, "A_mse", old_manifest)
    t = config["train_diffusion"]
    if metadata["source_global_step"] != protocol["expected_source_step"]:
        raise ValueError("Wrong raw source step")
    if t["max_steps"] != protocol["expected_additional_steps"] or t.get("snr_gamma") is not None:
        raise ValueError("Need the approved completed ordinary-MSE baseline")
    if t.get("lambda_gb", 0) or t.get("lambda_edge", 0) or t.get("local_descriptors"):
        raise ValueError("Baseline must be composition-only without structural losses")
    if old_manifest["protocol"]["exclude_alloy_id"] != 3:
        raise ValueError("Pilot is Fold03 only")
    if config["data"]["patch_size"] != 256 and not protocol.get("software_test_only", False):
        raise ValueError("This pilot uses 256 patches")
    if not set(protocol["evaluation"]["alloy_ids"]) <= set(report["training_alloy_ids"]):
        raise ValueError("Monitor contains an unseen/held-out alloy")
    fingerprints = old_root / "A_mse" / "input_fingerprints.csv"
    if not fingerprints.is_file():
        raise FileNotFoundError("Baseline input fingerprints are required")
    initialization = read_json(old_root / "A_mse" / "initialization_metadata.json")
    for key in ("config", "initialization", "snr_gamma"):
        expected = config if key == "config" else report["initialization"] if key == "initialization" else None
        if key == "config":
            actual = {k: v for k, v in initialization[key].items() if not k.startswith("_")}
            expected = {k: v for k, v in expected.items() if not k.startswith("_")}
        else:
            actual = initialization[key]
        if actual != expected:
            raise ValueError(f"Baseline initialization metadata mismatch: {key}")
    return old_root, old_manifest, config, metadata, fingerprints


def prepare_labels(config, protocol, output, training_ids):
    """Native rasters only. Fixed private NumPy RNG never touches training RNG."""
    settings = validate_settings(protocol.get("descriptor_settings", {}))
    d = config["data"]
    dataset = EbsdPatchDataset(d["manifest"], d["patch_size"], 1, "evaluation",
        exclude_alloy_id=3, crop_margins=tuple(d["crop_margins"]))
    dataset.rows = [r for r in dataset.rows if int(r["alloy_id"]) in training_ids
                    and str(r["use_for_diffusion"]).strip().lower() in {"true", "1", "yes", "y"}]
    if set(dataset.alloy_ids) != set(training_ids) or len(dataset.rows) != len(training_ids):
        raise ValueError("Need exactly one approved source image per training alloy")
    folder = output / "labels"
    folder.mkdir(parents=True)
    images, catalog, valid_vectors = [], [], []
    size = d["patch_size"]
    for index, row in enumerate(dataset.rows):
        aid, path = int(row["alloy_id"]), Path(row["image_path"])
        with Image.open(path) as im:
            pixels = np.asarray(im.convert("RGB"), dtype=np.uint8)
        regions, quality = segment_native(pixels, settings)
        np.save(folder / f"{aid}_rgb.npy", pixels, allow_pickle=False)
        np.save(folder / f"{aid}_regions.npy", regions, allow_pickle=False)
        # Human-readable QA overlay: white region interfaces, no generated images.
        border = np.zeros(regions.shape, bool)
        border[:, 1:] |= regions[:, 1:] != regions[:, :-1]
        border[1:] |= regions[1:] != regions[:-1]
        overlay = pixels.copy(); overlay[border] = 255
        Image.fromarray(overlay).save(folder / f"{aid}_segmentation_QA.png")
        save_json({"alloy_id": aid, "image_path": str(path), "quality": quality,
                   "settings": settings}, folder / f"{aid}_QA.json")
        if quality["region_count"] < 3 or quality["largest_region_fraction"] > 0.9:
            raise ValueError(f"Native segmentation QA failed for alloy {aid}: {quality}; inspect {folder / f'{aid}_segmentation_QA.png'}")
        top, right, bottom, left = d["crop_margins"]
        max_y = pixels.shape[0] - bottom - size
        max_x = pixels.shape[1] - right - size
        if max_y < top or max_x < left:
            raise ValueError("Native image cannot supply the exact training crop")
        rng = np.random.default_rng(np.random.SeedSequence([20261027, aid, index]))
        for _ in range(protocol["normalizer_windows_per_alloy"]):
            y, x = int(rng.integers(top, max_y + 1)), int(rng.integers(left, max_x + 1))
            values, valid, _ = describe_rgb(pixels[y:y + size, x:x + size], settings)
            record = {"alloy_id": aid, "box": [x, y, x + size, y + size],
                      "valid": valid, "values": values.tolist() if valid else None}
            catalog.append(record)
            if valid:
                valid_vectors.append(values)
        images.append({"alloy_id": aid, "image_path": str(path.resolve()),
            "image_sha256": file_sha256(path), "shape": list(pixels.shape),
            "pixels": f"{aid}_rgb.npy", "labels": f"{aid}_regions.npy", "quality": quality})
    for row in images:
        if not any(c["valid"] and c["alloy_id"] == row["alloy_id"] for c in catalog):
            raise ValueError(f"No valid calibration windows for alloy {row['alloy_id']}")
    values = np.asarray(valid_vectors)
    mean, raw_std = values.mean(0), values.std(0)
    metadata = {"schema": SCHEMA, "names": NAMES, "settings": settings,
        "units": "native pixel proxies, not AZtec/CTF physical grain statistics",
        "region_rule": "clipped connected pieces; boundary pixels assigned to nearest region; no crop-frame perimeter",
        "training_alloy_ids": sorted(dataset.alloy_ids), "images": images,
        "normalizer": {"mean": mean.tolist(), "std": np.maximum(raw_std, 1e-6).tolist(),
                       "constant_columns": [NAMES[i] for i in np.flatnonzero(raw_std < 1e-6)]},
        "calibration_valid": len(valid_vectors), "calibration_total": len(catalog),
        "invalid_training_rule": "retain original image/batch; zero descriptor residual via validity mask"}
    save_json(metadata, folder / "metadata.json")
    save_json(catalog, folder / "calibration_windows.json")
    targets = []
    for aid in protocol["evaluation"]["alloy_ids"]:
        candidates = [c for c in catalog if c["valid"] and c["alloy_id"] == aid]
        if len(candidates) < 3:
            raise ValueError(f"Insufficient valid monitor windows for alloy {aid}")
        candidates.sort(key=lambda c: (c["values"][0], c["box"]))
        for name, q in zip(("fine", "middle", "coarse"), protocol["evaluation"]["target_quantiles"]):
            record = deepcopy(candidates[round(q * (len(candidates) - 1))])
            record["target"] = name
            record["normalized"] = ((np.asarray(record["values"]) - mean) / np.maximum(raw_std, 1e-6)).tolist()
            targets.append(record)
    save_json(targets, folder / "monitor_targets.json")
    return metadata


def baseline_samples(old_root, manifest, protocol):
    """Re-use exactly the existing raw A_mse PNGs, after checking their identity."""
    folder = old_root / "evaluation"
    report = read_json(folder / "run_metadata.json")
    settings = protocol["evaluation"]
    previous = manifest["protocol"]["evaluation"]
    for key in ("steps", "guidance_scale", "height", "width", "sampling_dtype", "gb_threshold"):
        if settings[key] != previous[key]:
            raise ValueError(f"Baseline evaluation setting differs: {key}")
    model = report["model_files"]["A_mse"]
    final = old_root / "A_mse" / "final"
    for key, path in (("raw_unet", final / "unet_raw/diffusion_pytorch_model.safetensors"),
                      ("condition_encoder_weights", final / "condition_encoder/pytorch_model.bin")):
        if model[key]["sha256"] != file_sha256(path):
            raise ValueError("Baseline evaluation used different weights")
    records = read_json(folder / "per_image_metrics.json")
    selected = []
    for aid in settings["alloy_ids"]:
        for seed in settings["seeds"]:
            matches = [r for r in records if r["method"] == "A_mse" and r["alloy_id"] == aid and r["seed"] == seed]
            if len(matches) != 1:
                raise ValueError("Missing or duplicated baseline monitor image")
            record = deepcopy(matches[0])
            p = old_root / record["relative_path"]
            if not p.is_file() or file_sha256(p) != record["image_sha256"]:
                raise ValueError(f"Baseline PNG changed: {p}")
            record["path"] = str(p)
            selected.append(record)
    return selected


def prepare(protocol_path, root=ROOT):
    protocol, output = read_protocol(protocol_path, root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output_root: {output}")
    old_root, old, config, _, fingerprints = verify_baseline(protocol, root)
    hardware = device_profile(protocol)
    if hardware:
        old_environment = read_json(old_root / "A_mse/initialization_metadata.json")
        hardware["baseline_gpu_name"] = old_environment.get("gpu_name")
        hardware["device_changed"] = hardware["gpu_name"] != hardware["baseline_gpu_name"]
        hardware["comparison_limit"] = "Checkpointing-off is an execution override. Cross-device numerical identity is not assumed; audited training fingerprints must still match."
    baseline_records = baseline_samples(old_root, old, protocol)
    # Reuse the old scientific preflight; no Min-SNR training is launched.
    audit = preflight(old["protocol"], root, {"A_mse": config})
    for key in ("initialization", "training_alloy_ids"):
        if audit[key] != old["preflight"][key]:
            raise ValueError(f"Current preflight differs from baseline: {key}")
    output.mkdir(parents=True, exist_ok=True)
    metadata = prepare_labels(config, protocol, output, audit["training_alloy_ids"])
    new_config = execution_config(config, protocol)
    for key in list(new_config):
        if key.startswith("_"):
            new_config.pop(key)
    paired = protocol.get("comparison_mode") == "fresh_pair"
    control_config = deepcopy(new_config)
    control_config["train_diffusion"]["output_dir"] = str(output / "A_mse")
    if paired:
        fingerprints = output / "A_mse/input_fingerprints.csv"
    new_config["train_diffusion"].update({"output_dir": str(output / "B_local5"),
        "local_descriptors": {"metadata": str(output / "labels/metadata.json"),
                              "reference_fingerprints": str(fingerprints)}})
    (output / "configs").mkdir()
    if paired:
        (output / "configs/A_mse.yaml").write_text(
            yaml.safe_dump(control_config, allow_unicode=True, sort_keys=False), "utf-8")
    config_path = output / "configs/B_local5.yaml"
    config_path.write_text(yaml.safe_dump(new_config, allow_unicode=True, sort_keys=False), "utf-8")
    shutil.copy2(protocol_path, output / "configs/protocol.yaml")
    for path in sorted(set(current_code_files(root) + [root / name for name in NEW_CODE])):
        destination = output / "source_snapshot" / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    frozen = {str(path): file_sha256(path) for path in output.rglob("*") if path.is_file()}
    frozen.update({str(path): file_sha256(path) for path in current_code_files(root) + [root / n for n in NEW_CODE]})
    # Freeze source/weights/data and the existing control independently of code.
    frozen.update({str(Path(k)): file_sha256(Path(k)) for k in audit["file_sha256"]})
    extra = [old_root / "experiment_manifest.json", old_root / "A_mse/input_fingerprints.csv",
             old_root / "A_mse/final/run_metadata.json", old_root / "evaluation/run_metadata.json",
             old_root / "evaluation/per_image_metrics.json", Path(protocol_path).resolve()]
    extra += [Path(r["path"]) for r in baseline_records]
    extra += [old_root / "A_mse/final/unet_raw/diffusion_pytorch_model.safetensors",
              old_root / "A_mse/final/unet_raw/config.json",
              old_root / "A_mse/final/condition_encoder/pytorch_model.bin"]
    frozen.update({str(p): file_sha256(p) for p in extra})
    result = {"schema_version": 1, "status": "prepared_not_trained", "project_root": str(root),
        "protocol": protocol, "baseline_run": str(old_root), "config": str(config_path),
        "preflight": audit, "baseline_samples": [] if paired else baseline_records, "frozen_sha256": frozen,
        "comparison_mode": "fresh_pair" if paired else "reuse_historical",
        "control_config": str(output / "configs/A_mse.yaml") if paired else None,
        "descriptor_schema": metadata["schema"], "device_profile": hardware,
        "execution_override": {"baseline_gradient_checkpointing": config["train_diffusion"]["gradient_checkpointing"],
                               "new_gradient_checkpointing": new_config["train_diffusion"]["gradient_checkpointing"]},
        "scientific_scope": "training-alloy known-local-RGB-proxy control screening, not composition-only generalization"}
    try:
        result["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        result["git_revision"] = None
    save_json(result, output / "experiment_manifest.json")
    return result


def load_prepared(protocol_path, root=ROOT):
    _, output = read_protocol(protocol_path, root)
    manifest = read_json(output / "experiment_manifest.json")
    if manifest["project_root"] != str(root):
        raise ValueError("Prepare on this server; do not copy prepared absolute paths")
    for name, digest in manifest["frozen_sha256"].items():
        if not Path(name).is_file() or file_sha256(Path(name)) != digest:
            raise ValueError(f"Prepared input changed: {name}")
    return output, manifest


def require_cuda(manifest):
    if manifest["protocol"].get("require_cuda", True) and not torch.cuda.is_available():
        raise RuntimeError("Server CUDA is required; no automatic Mac/CPU training or sampling")


def train_arm(output, root, name, config):
    arm = output / name
    if arm.exists() and any(arm.iterdir()):
        raise FileExistsError(f"{name} already has output; no automatic retraining/resume")
    command = [sys.executable, str(root / "train_diffusion.py"), "--config", config,
               "--exclude-alloy-id", "3"]
    (output / "console_logs").mkdir(exist_ok=True)
    with (output / f"console_logs/{name}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True); log.write(line); log.flush()
        if process.wait():
            raise RuntimeError(f"{name} failed; inspect console log; no automatic retry")


def verify_control(manifest):
    for name, digest in manifest["control_artifact_sha256"].items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f"Fresh control changed: {name}")


def train(protocol_path, root=ROOT):
    output, manifest = load_prepared(protocol_path, root)
    require_cuda(manifest)
    hardware = device_profile(manifest["protocol"])
    if hardware:
        prepared = manifest["device_profile"]
        if hardware["gpu_name"] != prepared["gpu_name"] or hardware["total_vram_gib"] != prepared["total_vram_gib"]:
            raise RuntimeError("GPU changed after prepare; prepare a new run on this GPU")
        save_json(hardware, output / "device_at_train.json")
    paired = manifest.get("comparison_mode") == "fresh_pair"
    if paired:
        if "control_artifact_sha256" not in manifest:
            train_arm(output, root, "A_mse", manifest["control_config"])
            control_metadata = read_json(output / "A_mse/final/run_metadata.json")
            if (control_metadata["global_step"] != manifest["protocol"]["expected_additional_steps"]
                    or control_metadata["source_global_step"] != manifest["protocol"]["expected_source_step"]):
                raise ValueError("Fresh control incomplete or wrong source")
            files = list((output / "A_mse/final").rglob("*"))
            files += [output / "A_mse/input_fingerprints.csv", output / "A_mse/initialization_metadata.json"]
            manifest["control_artifact_sha256"] = {str(p): file_sha256(p) for p in files if p.is_file()}
            manifest["status"] = "control_complete_pending_local5"
            save_json(manifest, output / "experiment_manifest.json")
        verify_control(manifest)
    train_arm(output, root, "B_local5", manifest["config"])
    arm = output / "B_local5"
    audit = read_json(arm / "matched_baseline_inputs.json")
    metadata = read_json(arm / "final/run_metadata.json")
    if audit["status"] != "passed" or metadata["global_step"] != manifest["protocol"]["expected_additional_steps"]:
        raise ValueError("Incomplete training/input audit")
    manifest["status"] = "trained_pending_evaluation"
    manifest["trained_artifact_sha256"] = {str(p): file_sha256(p) for p in (arm / "final").rglob("*") if p.is_file()}
    manifest["trained_artifact_sha256"][str(arm / "matched_baseline_inputs.json")] = file_sha256(arm / "matched_baseline_inputs.json")
    save_json(manifest, output / "experiment_manifest.json")
    return {"status": manifest["status"], "new_training_arms": ["A_mse", "B_local5"] if paired else ["B_local5"],
            "reused_baseline": None if paired else manifest["baseline_run"]}


def evaluate(protocol_path, root=ROOT):
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DModel
    from ebsd_diffusion.composition import CompositionConditioner, load_compositions
    from ebsd_diffusion.descriptor_conditioning import DescriptorConditionEmbedding, FixedDescriptorCondition
    from ebsd_diffusion.metrics import describe_image, descriptors_as_dict
    from scripts.sample_fold03_ab2c_256 import build_montage, sample_image, save_png

    output, manifest = load_prepared(protocol_path, root)
    require_cuda(manifest)
    if manifest["status"] != "trained_pending_evaluation":
        raise ValueError("Complete training once before evaluation")
    for name, digest in manifest["trained_artifact_sha256"].items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f"Trained artifact changed: {name}")
    folder = output / "evaluation"
    if folder.exists():
        raise FileExistsError("Evaluation already exists; no automatic resampling")
    folder.mkdir()
    config = yaml.safe_load(Path(manifest["config"]).read_text("utf-8"))
    settings = manifest["protocol"]["evaluation"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if settings["sampling_dtype"] == "float16" and device.type == "cuda" else torch.float32
    final = output / "B_local5/final"
    unet = UNet2DModel.from_pretrained(final / "unet_raw").to(device=device, dtype=dtype).eval()
    vae = AutoencoderKL.from_pretrained(config["train_diffusion"]["vae_dir"]).to(device=device, dtype=dtype).eval()
    encoder = DescriptorConditionEmbedding.from_pretrained(final / "condition_encoder").to(device=device, dtype=dtype).eval()
    scheduler = DDIMScheduler.from_pretrained(final / "scheduler")
    conditioner = CompositionConditioner.load(final / "composition_conditioner.json")
    compositions = load_compositions(config["data"]["compositions"])
    metadata = read_json(final / "run_metadata.json")
    paired = manifest.get("comparison_mode") == "fresh_pair"
    controls_all = manifest["baseline_samples"]
    if paired:
        from ebsd_diffusion.models import ContinuousConditionEmbedding
        verify_control(manifest)
        control_final = output / "A_mse/final"
        control_unet = UNet2DModel.from_pretrained(control_final / "unet_raw").to(device=device, dtype=dtype).eval()
        control_encoder = ContinuousConditionEmbedding.from_pretrained(control_final / "condition_encoder").to(device=device, dtype=dtype).eval()
        control_scheduler = DDIMScheduler.from_pretrained(control_final / "scheduler")
        control_metadata = read_json(control_final / "run_metadata.json")
        controls_all = []
        for aid in settings["alloy_ids"]:
            condition = torch.tensor(conditioner.transform_by_alloy_id(compositions, aid), device=device, dtype=dtype).unsqueeze(0)
            for seed in settings["seeds"]:
                audit = {}
                generated = sample_image(control_unet, vae, control_encoder, control_scheduler, condition,
                    control_metadata["latent_scaling"], seed, settings["steps"], settings["guidance_scale"],
                    settings["height"], settings["width"], settings["height"], settings["width"], sampling_audit=audit)
                path = folder / f"fresh_controls/alloy_{aid:02d}_seed{seed}.png"
                save_png(generated, path)
                controls_all.append({"alloy_id": aid, "seed": seed, "path": str(path),
                    "image_sha256": file_sha256(path), "sampling": audit,
                    "metrics": descriptors_as_dict(describe_image(path, settings["gb_threshold"]))})
        save_json(controls_all, folder / "control_images.json")
        del control_unet, control_encoder
    descriptor_meta = read_json(output / "labels/metadata.json")
    targets = read_json(output / "labels/monitor_targets.json")
    image_entries = {r["alloy_id"]: r for r in descriptor_meta["images"]}
    summaries, pairs, inputs = [], [], []
    for aid in settings["alloy_ids"]:
        grid = []
        controls = [r for r in controls_all if r["alloy_id"] == aid]
        control_paths = []
        for r in controls:
            dest = folder / f"alloy_{aid:02d}/A_mse" / Path(r["path"]).name
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(r["path"], dest)
            control_paths.append((f"seed {r['seed']}", dest))
        grid.append(("A_mse fresh same-device" if paired else "A_mse reused (composition only)", control_paths))
        condition = torch.tensor(conditioner.transform_by_alloy_id(compositions, aid), device=device, dtype=dtype).unsqueeze(0)
        for target in [r for r in targets if r["alloy_id"] == aid]:
            with Image.open(image_entries[aid]["image_path"]) as source:
                reference = source.convert("RGB").crop(tuple(target["box"]))
            reference_path = folder / f"alloy_{aid:02d}" / f"target_{target['target']}.png"
            save_png(reference, reference_path)
            images = []
            bound = FixedDescriptorCondition(encoder, target["normalized"]).to(device=device, dtype=dtype).eval()
            for seed in settings["seeds"]:
                audit = {}
                generated = sample_image(unet, vae, bound, scheduler, condition, metadata["latent_scaling"],
                    seed, settings["steps"], settings["guidance_scale"], settings["height"], settings["width"],
                    settings["height"], settings["width"], sampling_audit=audit)
                control = next(r for r in controls if r["seed"] == seed)
                if audit != control["sampling"]:
                    raise ValueError("Initial noise/DDIM sequence differs from existing A_mse image")
                path = folder / f"alloy_{aid:02d}/B_local5/{target['target']}_seed{seed}.png"
                save_png(generated, path)
                images.append((f"seed {seed}", path))
                a_rgb = np.asarray(Image.open(control["path"]).convert("RGB"))
                b_rgb = np.asarray(generated.convert("RGB"))
                av, a_valid, _ = describe_rgb(a_rgb, descriptor_meta["settings"])
                bv, b_valid, _ = describe_rgb(b_rgb, descriptor_meta["settings"])
                std = np.asarray(descriptor_meta["normalizer"]["std"])
                target_values = np.asarray(target["values"])
                row = {"alloy_id": aid, "target": target["target"], "seed": seed,
                       "A_valid": a_valid, "B_valid": b_valid,
                       "A_mean_standardized_abs_error": float(np.mean(np.abs((av - target_values) / std))) if a_valid else None,
                       "B_mean_standardized_abs_error": float(np.mean(np.abs((bv - target_values) / std))) if b_valid else None}
                for i, name in enumerate(NAMES):
                    row[f"target_{name}"] = target_values[i]
                    row[f"A_{name}"] = float(av[i]) if a_valid else None
                    row[f"B_{name}"] = float(bv[i]) if b_valid else None
                pairs.append(row)
                summaries.append({"alloy_id": aid, "target": target["target"], "seed": seed,
                    "B_metrics": descriptors_as_dict(describe_image(path, settings["gb_threshold"])), "relative_path": path.relative_to(folder).as_posix(),
                    "sha256": file_sha256(path), "A_image_sha256": control["image_sha256"],
                    "known_descriptor_condition": target, "proxy_valid": b_valid})
                inputs.append({"alloy_id": aid, "seed": seed, "target": target["target"], **audit})
            grid.append((f"B_local5: observed {target['target']} target", images))
        build_montage(folder / f"alloy_{aid:02d}/comparison_grid.png", grid, (settings["width"], settings["height"]))
    save_json(summaries, folder / "summary.json")
    save_json(inputs, folder / "sampling_inputs.json")
    with (folder / "paired_descriptor_errors.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0])); writer.writeheader(); writer.writerows(pairs)
    save_json(pairs, folder / "paired_descriptor_errors.json")
    scalar_rows = [{"alloy_id": r["alloy_id"], "target": r["target"], "seed": r["seed"],
                    "proxy_valid": r["proxy_valid"],
                    **{k: v for k, v in r["B_metrics"].items() if isinstance(v, (int, float))}}
                   for r in summaries]
    with (folder / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0])); writer.writeheader(); writer.writerows(scalar_rows)
    report = {"new_samples": len(summaries) + (len(controls_all) if paired else 0),
        "new_local5_samples": len(summaries), "new_control_images": len(controls_all) if paired else 0,
        "comparison_mode": manifest.get("comparison_mode", "reuse_historical"),
        "reused_control_images": len(manifest["baseline_samples"]),
        "weight_type": "raw_only", "uncertainty_inputs": False,
        "scope": manifest["scientific_scope"], "primary_measure": "five individual proxy errors and morphology; scalar mean is secondary",
        "limits": "two seeds; known training-alloy descriptors; correlated windows; threshold-based RGB regions, not physical grains; ID03 not evaluated",
        "artifact_sha256": {p.relative_to(folder).as_posix(): file_sha256(p) for p in folder.rglob("*") if p.is_file()}}
    save_json(report, folder / "run_metadata.json")
    manifest["status"] = "evaluated_pending_scientific_review"
    save_json(manifest, output / "experiment_manifest.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=["prepare", "train", "evaluate"], default="prepare")
    args = parser.parse_args()
    result = {"prepare": prepare, "train": train, "evaluate": evaluate}[args.phase](args.config.resolve())
    print(json.dumps({k: v for k, v in result.items() if k in ("status", "descriptor_schema", "device_profile", "execution_override", "new_samples", "reused_control_images", "new_training_arms")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
