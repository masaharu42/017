"""Replay exactly the first batch, with optimizer updates explicitly blocked.

Compare archived composition-only code and current local-five code on this
server. Distinguishes a new-code input-stream change from cross-environment
failure to reproduce the historical combined SHA256. No samples are generated.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
import subprocess
import sys

import yaml


class CapturedFirstBatch(Exception):
    pass


def worker(job_path):
    job = json.loads(Path(job_path).read_text("utf-8"))
    sys.path.insert(0, job["code_root"])
    import torch
    from ebsd_diffusion import training_control
    original = training_control.tensor_fingerprint

    def capture(**tensors):
        record = {"combined_sha256": original(**tensors), "components": {
            name: {"sha256": original(**{name: t}), "shape": list(t.shape),
                   "dtype": str(t.dtype), "mean": float(t.detach().float().mean())}
            for name, t in tensors.items()}, "optimizer_updates": 0,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
        Path(job["record"]).write_text(json.dumps(record, indent=2), "utf-8")
        raise CapturedFirstBatch()

    def no_update(*args, **kwargs):
        raise RuntimeError("Input audit forbids optimizer updates")

    training_control.tensor_fingerprint = capture
    torch.optim.AdamW.step = no_update
    sys.argv = [str(Path(job["code_root"]) / "train_diffusion.py"), "--config", job["config"],
                "--exclude-alloy-id", "3"]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except CapturedFirstBatch:
        print("Captured first batch; stopped before U-Net forward/backward/update", flush=True)
        return
    raise RuntimeError("No first-batch capture; audit incomplete")


def audit(protocol_path, root=None, control_run=None):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    protocol = yaml.safe_load(Path(protocol_path).read_text("utf-8"))
    output = Path(protocol["output_root"])
    if not output.is_absolute():
        output = root / output
    manifest = json.loads((output / "experiment_manifest.json").read_text("utf-8"))
    baseline = Path(control_run) if control_run is not None else Path(manifest["baseline_run"])
    paired_control = control_run is not None
    old_manifest = json.loads((baseline / "experiment_manifest.json").read_text("utf-8"))
    configs = {"archived_mse": Path(old_manifest["control_config"] if paired_control else old_manifest["configs"]["A_mse"]),
               "current_local5": Path(manifest["config"])}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = output / "input_audits" / stamp
    folder.mkdir(parents=True)
    results = {}
    for name, config_path in configs.items():
        config = deepcopy(yaml.safe_load(config_path.read_text("utf-8")))
        config["train_diffusion"]["output_dir"] = str(folder / name / "runtime")
        path = folder / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), "utf-8")
        code_root = baseline / "source_snapshot" if name == "archived_mse" else root
        record = folder / f"{name}.json"
        job = folder / f"{name}_job.json"
        job.write_text(json.dumps({"code_root": str(code_root), "config": str(path), "record": str(record)}), "utf-8")
        print(f"Replay {name}: first batch only, optimizer blocked", flush=True)
        with (folder / f"{name}.log").open("w", encoding="utf-8") as handle:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(job)],
                           cwd=root, stdout=handle, stderr=subprocess.STDOUT, check=True)
        results[name] = json.loads(record.read_text("utf-8"))
    with (baseline / "A_mse/input_fingerprints.csv").open(newline="") as handle:
        expected = next(r["sha256"] for r in csv.DictReader(handle) if int(r["additional_step"]) == 1)
    old, current = results["archived_mse"], results["current_local5"]
    report = {"optimizer_updates": 0, "historical_step1_sha256": expected,
        "control_run": str(baseline),
        "archived_replay_matches_history": old["combined_sha256"] == expected,
        "new_code_matches_archived_replay_here": current["combined_sha256"] == old["combined_sha256"],
        "components_equal_here": {k: v["sha256"] == current["components"][k]["sha256"] for k,v in old["components"].items()},
        "artifacts": str(folder), "results": results}
    (folder / "summary.json").write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps({k: v for k,v in report.items() if k != "results"}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--control-run", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.config:
        audit(args.config, control_run=args.control_run)
    else:
        parser.error("Supply --config")


if __name__ == "__main__":
    main()
