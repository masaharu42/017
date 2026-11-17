"""Benchmark disposable A/B runs, then emit a fixed-budget hardware protocol.

Run on the training server before prepare. No benchmark weights are exported or
used by the formal pilot. Only explicit CUDA OOMs are recoverable failures.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import yaml
import torch

from ebsd_diffusion.utils import save_json
from scripts.run_minsnr_ablation import ARMS, PROJECT_ROOT, absolute, make_arm_configs


def with_batch(protocol: dict, batch: int, samples: int = 160000) -> dict:
    if batch <= 0 or samples % batch:
        raise ValueError("Each candidate batch must divide the sample budget exactly.")
    result = deepcopy(protocol)
    settings = result["training"]
    settings.setdefault("performance", {})["batch_size"] = batch
    settings.update({"additional_steps": samples // batch,
                     "warmup_steps": max(1, round(samples * .02 / batch)),
                     "save_every": max(1, round(samples * .2 / batch))})
    return result


def choose_candidate(candidates: list[dict], headroom: float) -> dict:
    safe = []
    for candidate in candidates:
        if candidate["status"] != "ok":
            continue
        if set(candidate["arms"]) != set(ARMS):
            continue
        reports = candidate["arms"].values()
        if all(report["peak_reserved_bytes"] <= report["initial_free_bytes"] - headroom * report["total_bytes"]
               for report in reports):
            safe.append(candidate)
    if not safe:
        raise RuntimeError("No safe batch found; inspect tuning_report.json and reduce candidates or VAE microbatch.")
    # Prefer the smaller batch on an exact tie; both arms must perform well.
    return max(safe, key=lambda item: (min(r["images_per_second"] for r in item["arms"].values()),
                                      -item["batch_size"]))


def tune(protocol_path: Path, output: Path, batches: list[int], warmup: int = 3,
         measured: int = 6, headroom: float = .08, samples: int = 160000,
         root: Path = PROJECT_ROOT) -> dict:
    if not batches or len(set(batches)) != len(batches):
        raise ValueError("Provide nonempty unique candidate batches.")
    if not 0 < headroom < .5 or warmup < 1 or measured < 1:
        raise ValueError("Require positive warmup/measurement steps and headroom between 0 and .5.")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not protocol.get("require_cuda"):
        raise ValueError("Hardware tuning requires a CUDA training protocol.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run this tuner inside the server PyTorch environment.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable in the current GPU/runtime.")
    for batch in batches:
        make_arm_configs(with_batch(protocol, batch, samples), root)
    formal_output = absolute(root, protocol["output_root"])
    if output.resolve() == formal_output or formal_output in output.resolve().parents:
        raise ValueError("Keep benchmark output separate from the formal output_root.")
    if formal_output.exists() and any(formal_output.iterdir()):
        raise FileExistsError("Formal output already prepared/populated; choose a new output_root.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite benchmark results: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report = {"protocol": str(protocol_path.resolve()), "sample_budget_per_arm": samples,
              "headroom_fraction_of_total_vram": headroom, "candidates": [],
              "limits": ["Short local throughput estimate, not guaranteed global optimum.",
                         "Bigger batches change optimizer dynamics even with equal patch exposures.",
                         "No weights exported; formal A/B restart from original raw50k."]}
    for batch in sorted(batches):
        candidate_protocol = with_batch(protocol, batch, samples)
        candidate_protocol["output_root"] = str(output / f"batch_{batch}")
        _, configs = make_arm_configs(candidate_protocol, root)
        candidate = {"batch_size": batch, "status": "ok", "arms": {}}
        report["candidates"].append(candidate)
        for arm in ARMS:
            directory = output / f"batch_{batch}" / arm
            config_path = output / f"batch_{batch}_{arm}.yaml"
            config_path.write_text(yaml.safe_dump(configs[arm], allow_unicode=True, sort_keys=False), encoding="utf-8")
            log_path = output / f"batch_{batch}_{arm}.log"
            print(f"Benchmark batch={batch} {arm}; log={log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run([
                    sys.executable, str(root / "train_diffusion.py"), "--config", str(config_path),
                    "--exclude-alloy-id", str(protocol["exclude_alloy_id"]),
                    "--benchmark-steps", str(warmup + measured), "--benchmark-warmup", str(warmup),
                ], cwd=root, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-16000:]
                candidate["status"] = "oom" if "CUDA out of memory" in tail or "CUDA error: out of memory" in tail else "error"
                candidate["failed_log"] = str(log_path)
                save_json(report, output / "tuning_report.json")
                if candidate["status"] != "oom":
                    raise RuntimeError(f"Benchmark failed (not OOM); inspect {log_path}\n{tail[-2500:]}")
                print(f"batch={batch}: CUDA OOM, candidate rejected.", flush=True)
                break
            result = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
            if not result.get("total_bytes"):
                raise RuntimeError("Benchmark did not run on CUDA; check ACCELERATE_USE_CPU / CUDA_VISIBLE_DEVICES.")
            candidate["arms"][arm] = result
            print(f"  {arm}: {result['images_per_second']:.2f} patches/s, "
                  f"{result['peak_reserved_bytes'] / 1024**3:.1f} GiB reserved", flush=True)
        if candidate["status"] == "ok":
            audits = [(output / f"batch_{batch}" / arm / "input_fingerprints.csv").read_bytes() for arm in ARMS]
            if audits[0] != audits[1]:
                candidate["status"] = "input_mismatch"
                save_json(report, output / "tuning_report.json")
                raise RuntimeError("A/B benchmark input hashes differ; do not start formal training.")
        save_json(report, output / "tuning_report.json")
    selected = choose_candidate(report["candidates"], headroom)
    selected_protocol = with_batch(protocol, selected["batch_size"], samples)
    # Keep references valid when this generated YAML lives outside configs/.
    for key in ("source_sampling_config", "historical_hparams", "output_root"):
        selected_protocol[key] = str(absolute(root, selected_protocol[key]))
    selected_protocol["hardware_tuning"] = {"report": str(output / "tuning_report.json"),
                                            "sample_budget_per_arm": samples}
    selected_path = output / "selected_protocol.yaml"
    selected_path.write_text(yaml.safe_dump(selected_protocol, allow_unicode=True, sort_keys=False), encoding="utf-8")
    report.update({"selected_batch_size": selected["batch_size"], "selected_protocol": str(selected_path)})
    save_json(report, output / "tuning_report.json")
    print(json.dumps({key: report[key] for key in ("selected_batch_size", "selected_protocol")}, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fold03_minsnr_pro6000.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("performance/pro6000_tuning"))
    parser.add_argument("--batches", nargs="+", type=int, default=[32, 64, 128, 256, 640, 1000, 1280])
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=6)
    parser.add_argument("--headroom", type=float, default=.08)
    args = parser.parse_args()
    tune(args.config.resolve(), args.output_dir.resolve(), args.batches,
         args.warmup_steps, args.measure_steps, args.headroom)


if __name__ == "__main__":
    main()
