"""Explicit prepare/calibrate/train/evaluate phases for a matched MSE/S2 pilot."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import yaml
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler
from skimage import measure, morphology

from ebsd_diffusion.composition import load_compositions
from ebsd_diffusion.data import EbsdPatchDataset
from ebsd_diffusion.metrics import describe_image, descriptors_as_dict
from ebsd_diffusion.structure_statistics import validate_statistics_config
from ebsd_diffusion.training_control import file_sha256, resolve_initialization, load_frozen_conditioner, load_initialized_models
from ebsd_diffusion.utils import save_json
from scripts.run_descriptor_pilot import verify_baseline
from scripts.run_minsnr_ablation import absolute, preflight, reference_boxes
from scripts.sample_fold03_ab2c_256 import sample_image, save_png, build_montage

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("A_mse", "B_statistics")


def read_json(path):
    return json.loads(Path(path).read_text("utf-8"))


def assert_pair(configs):
    a, b = (deepcopy(configs[name]) for name in ARMS)
    for cfg in (a, b):
        cfg["train_diffusion"].pop("output_dir")
    if "structure_statistics" in a["train_diffusion"] or not b["train_diffusion"].pop("structure_statistics", None):
        raise ValueError("Only B may contain structure statistics")
    if a != b:
        raise ValueError("Pair differs beyond output directory and the statistics module")


def prepare(path):
    protocol = yaml.safe_load(Path(path).read_text("utf-8"))
    output = absolute(ROOT, protocol["output_root"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output root: {output}")
    old_root, old, source_cfg, metadata, _ = verify_baseline(protocol, ROOT)
    final = old_root / "A_mse/final"
    if metadata["effective_total_step"] != protocol["expected_baseline_effective_step"]:
        raise ValueError("Wrong reviewed A baseline effective step")
    source = {"raw_unet": final / "unet_raw/diffusion_pytorch_model.safetensors",
        "unet_config": final / "unet_raw", "condition_encoder_dir": final / "condition_encoder",
        "condition_encoder_weights": final / "condition_encoder/pytorch_model.bin",
        "conditioner": final / "composition_conditioner.json", "scheduler_dir": final / "scheduler",
        "run_metadata": final / "run_metadata.json"}
    for key, digest in protocol["expected_baseline_sha256"].items():
        if file_sha256(source[key]) != digest:
            raise ValueError(f"Reviewed baseline hash differs: {key}")
    base = {k: deepcopy(v) for k, v in source_cfg.items() if not k.startswith("_")}
    t = base["train_diffusion"]
    for key in ("boundary_dice", "local_descriptors", "structure_statistics"):
        t.pop(key, None)
    settings = protocol["training"]
    if not 0 <= settings["warmup_steps"] < settings["additional_steps"] or settings["learning_rate"] <= 0:
        raise ValueError("Invalid continuation schedule")
    if base["diffusion"]["prediction_type"] != "epsilon" or t["gradient_accumulation_steps"] != 1:
        raise ValueError("This pilot requires epsilon prediction and accumulation=1")
    t.update(initialization={k: str(p) for k, p in source.items()}, controlled_experiment=True,
        export_raw=True, snr_gamma=None, lambda_gb=0., lambda_edge=0.,
        fixed_latent_scaling=metadata["latent_scaling"], track_source_effective_step=True, max_steps=settings["additional_steps"],
        learning_rate=settings["learning_rate"], warmup_steps=settings["warmup_steps"],
        save_every=settings["save_every"], log_every=settings["log_every"])
    base["seed"] = settings["seed"]
    statistics_cfg = validate_statistics_config(protocol["structure_statistics"])
    statistics_cfg["calibration_file"] = str(output / "calibration/statistics_calibration.json")
    configs = {}
    for name in ARMS:
        cfg = deepcopy(base)
        cfg["train_diffusion"]["output_dir"] = str(output / name)
        if name == "B_statistics":
            cfg["train_diffusion"]["structure_statistics"] = statistics_cfg
        configs[name] = cfg
    assert_pair(configs)
    # Reuse read-only validation, loading the reviewed A final rather than its old starting point.
    checks = deepcopy(protocol)
    checks["expected_source_step"] = metadata["global_step"]
    checks["expected_source_sha256"] = protocol["expected_baseline_sha256"]
    report = preflight(checks, ROOT, configs)
    for p in (Path(path).resolve(), ROOT / "scripts/run_structure_statistics.py",
              ROOT / "scripts/run_descriptor_pilot.py"):
        report["file_sha256"][str(p)] = file_sha256(p)
    output.mkdir(parents=True)
    config_paths = {}
    for name, cfg in configs.items():
        p = output / "configs" / f"{name}.yaml"
        p.parent.mkdir(exist_ok=True)
        p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), "utf-8")
        config_paths[name] = str(p)
        report["file_sha256"][str(p)] = file_sha256(p)
    manifest = {"status": "prepared_not_trained", "protocol": protocol,
        "protocol_path": str(Path(path).resolve()), "project_root": str(ROOT),
        "configs": config_paths, "preflight": report,
        "source_effective_step": metadata["effective_total_step"]}
    save_json(manifest, output / "experiment_manifest.json")
    return {"status": manifest["status"], "output": str(output)}


def load_prepared(path):
    protocol = yaml.safe_load(Path(path).read_text("utf-8"))
    output = absolute(ROOT, protocol["output_root"])
    m = read_json(output / "experiment_manifest.json")
    if m["protocol"] != protocol or m["project_root"] != str(ROOT):
        raise ValueError("Protocol or machine changed after preparation")
    for p, digest in m["preflight"]["file_sha256"].items():
        if file_sha256(Path(p)) != digest:
            raise ValueError(f"Prepared input changed: {p}")
    assert_pair({k: yaml.safe_load(Path(p).read_text("utf-8")) for k, p in m["configs"].items()})
    return output, m


def completed(output, m, arm):
    final = output / arm / "final"
    record = read_json(final / "run_metadata.json")
    cfg = yaml.safe_load(Path(m["configs"][arm]).read_text("utf-8"))
    if (record["global_step"] != m["protocol"]["training"]["additional_steps"]
            or record["effective_total_step"] != m["source_effective_step"] + record["global_step"]
            or record["initialization"] != m["preflight"]["initialization"]
            or record["exclude_alloy_id"] != 3 or record["snr_gamma"] is not None
            or record["training_alloy_ids"] != m["preflight"]["training_alloy_ids"]
            or record["unet_weight_types"].get("unet_raw") != "raw"):
        raise ValueError(f"Completion provenance mismatch: {arm}")
    expected = cfg["train_diffusion"].get("structure_statistics")
    actual = record.get("structure_statistics")
    if arm == "B_statistics":
        calibration = read_json(output / "calibration/statistics_calibration.json")
        if actual != {**expected, "lambda": calibration["lambda"]}:
            raise ValueError("Final statistics settings differ from calibration")
    elif actual is not None:
        raise ValueError("MSE control unexpectedly contains statistics")
    return record


def execute(path, phase):
    output, m = load_prepared(path)
    if not torch.cuda.is_available():
        raise RuntimeError("Server CUDA required; no CPU training fallback")
    if phase == "calibrate" and m["status"] != "prepared_not_trained":
        raise ValueError("Calibration already attempted/completed; preserve prior output")
    if phase == "train":
        if m["status"] != "calibrated":
            raise ValueError("Complete zero-update calibration first")
        if file_sha256(output / "calibration/statistics_calibration.json") != m["calibration_sha256"]:
            raise ValueError("Calibration changed")
    arms = ["B_statistics"] if phase == "calibrate" else ARMS
    for arm in arms:
        directory = output / ("calibration" if phase == "calibrate" else arm)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Existing/partial result preserved: {directory}")
        command = [sys.executable, "-u", str(ROOT / "train_diffusion.py"), "--config", m["configs"][arm],
                   "--exclude-alloy-id", "3", "--output-dir", str(directory)]
        if phase == "calibrate":
            command.append("--statistics-calibrate")
        log = output / "console_logs" / f"{phase}_{arm}.log"
        log.parent.mkdir(exist_ok=True)
        with log.open("x") as handle:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as proc:
                for line in proc.stdout:
                    print(line, end="", flush=True); handle.write(line); handle.flush()
                if proc.wait():
                    raise RuntimeError(f"Failed: inspect {log}; partial run preserved")
        if phase == "train":
            completed(output, m, arm)
    if phase == "calibrate":
        record = read_json(output / "calibration/statistics_calibration.json")
        if record["optimizer_updates"] != 0 or record["status"] != "passed":
            raise ValueError("Calibration invalid")
        m["calibration_sha256"] = file_sha256(output / "calibration/statistics_calibration.json")
        m["status"] = "calibrated"
    else:
        records = []
        for arm in ARMS:
            with (output / arm / "input_fingerprints.csv").open() as f:
                records.append(list(csv.DictReader(f)))
        steps = m["protocol"]["training"]["additional_steps"]
        interval = m["protocol"]["training"]["log_every"]
        expected = [i for i in range(1, steps + 1) if i <= 5 or i % interval == 0]
        if records[0] != records[1] or [int(r["additional_step"]) for r in records[0]] != expected:
            raise ValueError("Pair sampled inputs differ or audit is incomplete")
        runtime = [read_json(output / arm / "initialization_metadata.json") for arm in ARMS]
        for k in ("torch", "python", "cuda", "gpu_name", "world_size", "optimizer_state_loaded"):
            if runtime[0][k] != runtime[1][k]:
                raise ValueError(f"Pair runtime mismatch: {k}")
        m["final_sha256"] = {str(p): file_sha256(p) for arm in ARMS
                             for p in (output / arm / "final").rglob("*") if p.is_file()}
        m["status"] = "trained_pending_evaluation"
    save_json(m, output / "experiment_manifest.json")
    return {"status": m["status"], "output": str(output)}


def evaluate(path):
    output, m = load_prepared(path)
    if m["status"] != "trained_pending_evaluation" or not torch.cuda.is_available():
        raise RuntimeError("Complete both arms and evaluate on server CUDA")
    for p, digest in m["final_sha256"].items():
        if file_sha256(Path(p)) != digest:
            raise ValueError(f"Trained artifact changed: {p}")
    folder = output / "evaluation"
    if folder.exists():
        raise FileExistsError(f"Evaluation preserved: {folder}")
    cfg = yaml.safe_load(Path(m["configs"]["A_mse"]).read_text("utf-8"))
    cfg["_project_root"] = str(ROOT)
    paths, provenance = resolve_initialization(cfg)
    settings = m["protocol"]["evaluation"]
    vae = AutoencoderKL.from_pretrained(cfg["train_diffusion"]["vae_dir"]).to("cuda", dtype=torch.float16).eval()
    compositions = load_compositions(cfg["data"]["compositions"])
    conditioner = load_frozen_conditioner(paths, set(m["preflight"]["training_alloy_ids"]), compositions)
    dataset = EbsdPatchDataset(cfg["data"]["manifest"], 256, 1, "evaluation", crop_margins=tuple(cfg["data"]["crop_margins"]))
    rows, audits, grids = [], {}, {aid: [] for aid in settings["alloy_ids"]}
    def record(p, aid, method, seed=None):
        values = descriptors_as_dict(describe_image(p, settings["gb_threshold"]))
        rgb = np.asarray(Image.open(p).convert("RGB")) / 255.
        boundary = morphology.binary_dilation(rgb.max(2) <= settings["gb_threshold"], morphology.disk(1))
        labels = measure.label(~boundary, connectivity=1)
        areas = [r.area for r in measure.regionprops(labels) if r.area >= 9]
        values["largest_region_area_fraction"] = max(areas, default=0) / labels.size
        rows.append({"alloy_id": aid, "method": method, "seed": seed,
                     "path": str(p.relative_to(folder)), **values})
    for aid in settings["alloy_ids"]:
        real = next(r for r in dataset.rows if int(r["alloy_id"]) == aid)
        references = []
        with Image.open(real["image_path"]) as im:
            im = im.convert("RGB")
            for j, box in enumerate(reference_boxes(im.width, im.height, 256, cfg["data"]["crop_margins"])):
                p = folder / f"alloy_{aid:02d}/real/window_{j}.png"
                save_png(im.crop(box), p); record(p, aid, "real")
                references.append((f"real window {j}", p))
        grids[aid].append(("real windows (not seed-paired targets)", references))
    for method in ("start_A", *ARMS):
        model = dict(paths)
        if method != "start_A":
            final = output / method / "final"
            model.update(raw_unet=final / "unet_raw/diffusion_pytorch_model.safetensors", unet_config=final / "unet_raw",
                condition_encoder_dir=final / "condition_encoder", condition_encoder_weights=final / "condition_encoder/pytorch_model.bin")
        unet, encoder, _ = load_initialized_models(model, cfg["diffusion"], conditioner.dimension)
        unet.to("cuda", dtype=torch.float16).eval(); encoder.to("cuda", dtype=torch.float16).eval()
        scheduler = DDIMScheduler.from_pretrained(paths["scheduler_dir"])
        for aid in settings["alloy_ids"]:
            cond = torch.tensor(conditioner.transform_by_alloy_id(compositions, aid), device="cuda", dtype=torch.float16)[None]
            generated = []
            for seed in settings["seeds"]:
                audit = {}
                image = sample_image(unet, vae, encoder, scheduler, cond, provenance["latent_scaling"], seed,
                    settings["steps"], settings["guidance_scale"], 256, 256, 256, 256, sampling_audit=audit)
                key = f"{aid}_{seed}"
                if key in audits and audits[key] != audit:
                    raise ValueError("Sampling inputs differ")
                audits[key] = audit
                p = folder / f"alloy_{aid:02d}/{method}/seed_{seed}.png"
                save_png(image, p); record(p, aid, method, seed); generated.append((f"seed {seed}", p))
            grids[aid].append((method, generated))
        del unet, encoder
        torch.cuda.empty_cache()
    for aid, grid in grids.items():
        build_montage(folder / f"alloy_{aid:02d}/comparison_grid.png", grid, (256, 256))
    save_json(rows, folder / "per_image_metrics.json")
    scalars = [{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in rows]
    with (folder / "per_image_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scalars[0])); writer.writeheader(); writer.writerows(scalars)
    save_json(audits, folder / "sampling_inputs.json")
    save_json({"scope": "Fold03 training-alloy screening; RGB regions are not physical grains",
        "settings": settings, "generated_count": sum(r["seed"] is not None for r in rows),
        "source_effective_step": m["source_effective_step"],
        "artifact_sha256": {str(p.relative_to(folder)): file_sha256(p) for p in folder.rglob("*") if p.is_file()}},
        folder / "run_metadata.json")
    m["status"] = "evaluated_pending_review"
    save_json(m, output / "experiment_manifest.json")
    return {"status": m["status"], "output": str(folder)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fold03_structure_statistics_5k.yaml"))
    parser.add_argument("--phase", choices=("prepare", "calibrate", "train", "evaluate"), default="prepare")
    args = parser.parse_args()
    result = prepare(args.config) if args.phase == "prepare" else evaluate(args.config) if args.phase == "evaluate" else execute(args.config, args.phase)
    print(json.dumps(result, ensure_ascii=False, indent=2))
