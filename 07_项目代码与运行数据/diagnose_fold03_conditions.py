"""Read-only Fold03 condition-response screening; run on the GPU server."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from diffusers import AutoencoderKL
from PIL import Image

from ebsd_diffusion.composition import load_compositions
from ebsd_diffusion.data import EbsdPatchDataset
from ebsd_diffusion.training_control import (
    file_sha256, load_frozen_conditioner, load_initialized_models, tensor_fingerprint,
)
from scripts.run_minsnr_ablation import reference_boxes

IDS = [3, 5, 7, 20]
SEEDS = [20261012, 20261013]
TIMESTEPS = [100, 300, 600, 900]
ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Empty diagnostic table")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def distances(a, b):
    a, b = a.detach().double().reshape(-1), b.detach().double().reshape(-1)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Non-finite diagnostic tensor")
    na, nb = torch.linalg.vector_norm(a).item(), torch.linalg.vector_norm(b).item()
    delta = torch.linalg.vector_norm(a - b).item()
    return {
        "l2": delta, "delta_rms": delta / math.sqrt(a.numel()),
        "relative_l2": delta / max((na + nb) / 2, 1e-12),
        "cosine": (torch.dot(a, b).item() / (na * nb)) if na * nb > 1e-24 else None,
        "rms_a": na / math.sqrt(a.numel()), "rms_b": nb / math.sqrt(b.numel()),
    }


@torch.inference_mode()
def diagnose(args, output):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this lightweight forward-only diagnostic")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    experiment = args.experiment.resolve()
    manifest_path = experiment / "experiment_manifest.json"
    manifest = read_json(manifest_path)
    hashes = {}

    def record(path, expected=None):
        path = Path(path).resolve()
        value = file_sha256(path)
        if expected is not None and value != expected:
            raise ValueError(f"Hash mismatch: {path}")
        hashes[str(path)] = value

    record(manifest_path)
    for name, digest in manifest["frozen_sha256"].items():
        record(name, digest)
    if args.audit:
        audit_path = args.audit.resolve()
    else:
        candidates = list(experiment.glob("evaluation_50k_*/evaluation_audit.json"))
        if not candidates:
            raise FileNotFoundError("No evaluation_audit.json; supply --audit")
        signatures = {json.dumps(read_json(p)["final_sha256"], sort_keys=True) for p in candidates}
        if len(signatures) != 1:
            raise ValueError("Multiple evaluated models; select the correct --audit explicitly")
        audit_path = max(candidates, key=lambda p: p.stat().st_mtime)
    audit = read_json(audit_path)
    record(audit_path)
    record(manifest_path, audit["original_manifest_sha256"])
    for name, digest in audit["final_sha256"].items():
        record(name, digest)
    final = Path(audit["final"])
    final_metadata = read_json(final / "run_metadata.json")
    expected_ids = set(range(1, 23)) - {3}
    if set(manifest["training_alloy_ids"]) != expected_ids:
        raise ValueError("Unexpected Fold03 training IDs")
    if (final_metadata["global_step"] != 50000 or final_metadata["exclude_alloy_id"] != 3
            or set(final_metadata["training_alloy_ids"]) != expected_ids):
        raise ValueError("Expected the evaluated Fold03 50k model")

    config_path = Path(manifest["config_path"])
    record(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    def project_path(value):
        p = Path(value)
        return p if p.is_absolute() else ROOT / p
    compositions_path = project_path(config["data"]["compositions"])
    record(compositions_path)
    compositions = load_compositions(compositions_path)
    base = {key: Path(value) for key, value in manifest["baseline_source"]["paths"].items()}
    if read_json(base["run_metadata"])["global_step"] != 50000:
        raise ValueError("Baseline must be A raw50k")
    scale = float(manifest["baseline_source"]["latent_scaling"])
    if not math.isclose(scale, float(final_metadata["latent_scaling"]), rel_tol=1e-6):
        raise ValueError("VAE latent scaling differs")
    variant = dict(base)
    variant.update({
        "raw_unet": final / "unet_raw/diffusion_pytorch_model.safetensors",
        "unet_config": final / "unet_raw",
        "condition_encoder_dir": final / "condition_encoder",
        "condition_encoder_weights": final / "condition_encoder/pytorch_model.bin",
        "conditioner": final / "composition_conditioner.json",
        "scheduler_dir": final / "scheduler",
        "run_metadata": final / "run_metadata.json",
    })
    for model in (base, variant):
        for key in ("raw_unet", "condition_encoder_weights", "conditioner", "run_metadata"):
            record(model[key])
        for key, name in (("unet_config", "config.json"), ("condition_encoder_dir", "config.json"),
                          ("scheduler_dir", "scheduler_config.json")):
            record(model[key] / name)
    vae_path = project_path(config["train_diffusion"]["vae_dir"])
    for p in vae_path.iterdir():
        if p.is_file() and p.suffix in (".json", ".bin", ".safetensors"):
            record(p)
    vae = AutoencoderKL.from_pretrained(vae_path).float().to("cuda").eval().requires_grad_(False)

    dataset = EbsdPatchDataset(project_path(config["data"]["manifest"]), 256, 1, "evaluation",
                               crop_margins=tuple(config["data"]["crop_margins"]))
    record(project_path(config["data"]["manifest"]))
    matches = [r for r in dataset.rows if int(r["alloy_id"]) == 5]
    if len(matches) != 1:
        raise ValueError("Expected exactly one ID05 reference image")
    image_path = Path(matches[0]["image_path"])
    record(image_path)
    anchors, anchor_records = {}, []
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        boxes = reference_boxes(im.width, im.height, 256, config["data"]["crop_margins"])
        if len(boxes) < 2:
            raise ValueError("Need distinct center and top-left reference windows")
        for label, box in (("ID05_TL", boxes[0]), ("ID05_center", boxes[-1])):
            crop = im.crop(box)
            crop.save(output / f"{label}.png")
            x = torch.from_numpy(np.asarray(crop, dtype=np.float32).copy()).permute(2, 0, 1)[None].to("cuda") / 127.5 - 1
            z = vae.encode(x).latent_dist.mode() * scale
            if not torch.isfinite(z).all():
                raise ValueError("Non-finite VAE latent")
            anchors[label] = z.cpu()
            anchor_records.append({"name": label, "box": list(box), "latent_sha256": tensor_fingerprint(z=z)})
    del vae
    torch.cuda.empty_cache()

    vectors, distance_rows, response_rows, input_rows = {}, [], [], []
    reference_alphas = None
    forwards = 0
    for method, model in (("A_raw50k", base), ("Dice_raw50k", variant)):
        conditioner = load_frozen_conditioner(model, expected_ids, compositions)
        cond = torch.tensor(np.stack([conditioner.transform_by_alloy_id(compositions, aid) for aid in IDS]),
                            device="cuda", dtype=torch.float32)
        unet, encoder, scheduler = load_initialized_models(model, config["diffusion"], conditioner.dimension)
        unet = unet.float().to("cuda").eval().requires_grad_(False)
        encoder = encoder.float().to("cuda").eval().requires_grad_(False)
        if reference_alphas is None:
            reference_alphas = scheduler.alphas_cumprod.clone()
        elif not torch.equal(reference_alphas, scheduler.alphas_cumprod):
            raise ValueError("Noise schedules differ")
        embeddings = encoder(cond)
        null = encoder.unconditional(1, device="cuda", dtype=torch.float32)
        vectors[method] = {
            "active_features": conditioner.active_features,
            "raw_active_values": {str(a): compositions.loc[compositions["alloy_id"].astype(int) == a,
                conditioner.active_features].values[0].astype(float).tolist() for a in IDS},
            "normalized": {str(a): cond[i].cpu().tolist() for i, a in enumerate(IDS)},
            "embeddings": {str(a): embeddings[i].cpu().tolist() for i, a in enumerate(IDS)},
            "null_embedding": null[0].cpu().tolist(),
        }
        for i, j in itertools.combinations(range(4), 2):
            for stage, values in (("normalized_composition", cond), ("encoder", embeddings)):
                distance_rows.append({"method": method, "stage": stage, "id_a": IDS[i], "id_b": IDS[j],
                                      **distances(values[i], values[j])})
        batch_embeddings = torch.cat([embeddings, null])
        for label, z_cpu in anchors.items():
            z = z_cpu.to("cuda")
            for seed in SEEDS:
                gen = torch.Generator(device="cpu").manual_seed(seed)
                noise = torch.randn(z_cpu.shape, generator=gen).to("cuda")
                for t in TIMESTEPS:
                    xt = scheduler.add_noise(z, noise, torch.tensor([t], device="cuda"))
                    fingerprint = tensor_fingerprint(xt=xt)
                    input_rows.append({"method": method, "reference": label, "seed": seed, "t": t,
                                       "xt_sha256": fingerprint})
                    pred = unet(xt.repeat(5, 1, 1, 1), t, class_labels=batch_embeddings).sample
                    if not torch.isfinite(pred).all():
                        raise ValueError("Non-finite U-Net prediction")
                    forwards += 1
                    for i, j in itertools.combinations(range(5), 2):
                        response_rows.append({"method": method, "reference": label, "seed": seed, "t": t,
                            "id_a": str(IDS[i]), "id_b": str(IDS[j]) if j < 4 else "null",
                            **distances(pred[i], pred[j])})
                    print(f"{method} {label} seed={seed} t={t} done", flush=True)
        del unet, encoder, embeddings, null, pred, batch_embeddings
        torch.cuda.empty_cache()
    grouped = {}
    for row in input_rows:
        grouped.setdefault((row["reference"], row["seed"], row["t"]), set()).add(row["xt_sha256"])
    if any(len(v) != 1 for v in grouped.values()):
        raise ValueError("A and Dice did not receive identical noisy inputs")
    write_json(output / "vectors.json", vectors)
    write_csv(output / "condition_distances.csv", distance_rows)
    write_csv(output / "unet_responses.csv", response_rows)
    write_json(output / "input_audit.json", {"anchors": anchor_records, "inputs": input_rows})
    for name, expected in hashes.items():
        if file_sha256(Path(name)) != expected:
            raise ValueError(f"Input changed during diagnostic: {name}")
    write_json(output / "run_metadata.json", {
        "status": "completed_screening", "training_updates": 0, "generated_images": 0,
        "batched_unet_forwards": forwards, "batch_size": 5, "ids": IDS, "seeds": SEEDS,
        "timesteps": TIMESTEPS, "dtype": "FP32", "cfg": "Not applied; conditional and null predictions compared directly",
        "latent_encoding": "ID05 real reference, VAE posterior mode; forward-noised, not free sampling",
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "python": platform.python_version(), "audit_path": str(audit_path), "input_sha256": hashes,
        "script_sha256": file_sha256(Path(__file__)), "inputs_unchanged": True,
        "limitations": "Two possibly overlapping training windows; no universal response threshold; ID03 exploratory only; no accuracy or causality claim.",
    })
    (output / "README.txt").write_text(
        "Read-only fixed-input response screening. No training, no generation.\n"
        "relative_l2 = ||a-b|| / max((||a||+||b||)/2, 1e-12). delta_rms is RMS(a-b).\n"
        "Compare nearby vs distant compositions within each model and timestep; examine null differences too.\n"
        "Near-zero responses can indicate weak local sensitivity, not prove condition failure.\n"
        "Do not compare embedding magnitudes across independently trained encoders as an accuracy score.\n"
        "A common ID05 reference may favor ID05; this is a small mechanism screen, not validation.\n", encoding="utf-8")
    print(f"Completed: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path,
        default=ROOT / "08_训练与生成结果/12_Fold03_scratch_boundary_50k_运行")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    import tempfile
    output = Path(tempfile.mkdtemp(prefix="condition_response_", dir=args.experiment.resolve()))
    print(f"Diagnostic directory: {output}", flush=True)
    try:
        diagnose(args, output)
        import shutil
        import tarfile
        shutil.copy2(Path(__file__), output / "diagnose_fold03_conditions.py")
        archive = output.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(output, arcname=output.name)
        print(f"Download: {archive}", flush=True)
    except Exception:
        (output / "FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Diagnostic failed; record: {output}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
