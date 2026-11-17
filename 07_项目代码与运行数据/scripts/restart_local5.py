"""Restart B only, preserving a completed A and explicitly recording device changes.

This is a weights-only restart from the original raw50k source, not checkpoint
resume. Existing scientific inputs, A hashes, and training fingerprints remain
mandatory. No historical manifest/source hashes are rewritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

from ebsd_diffusion.training_control import file_sha256
from ebsd_diffusion.utils import save_json
from scripts.run_descriptor_pilot import (
    ROOT, device_profile, load_prepared, read_json, require_cuda, train_arm,
    verify_control,
)


def restart(config, *, accept_device_change=False, archive_failed=False, root=ROOT):
    output, manifest = load_prepared(config, root)
    if manifest.get("status") != "control_complete_pending_local5":
        raise ValueError("Need a completed A pending B; never rerun completed arms")
    require_cuda(manifest)
    verify_control(manifest)
    hardware = device_profile(manifest["protocol"])
    previous = manifest.get("device_profile")
    changed = hardware != previous if hardware is None or previous is None else any(
        hardware[k] != previous[k] for k in ("gpu_name", "total_vram_gib"))
    if changed and not accept_device_change:
        raise ValueError("Device changed; explicitly accept recorded cross-device screening")
    arm = output / "B_local5"
    if (arm / "final").exists():
        raise ValueError("B has a final directory; inspect it instead of restarting")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived = None
    if arm.exists() and any(arm.iterdir()):
        if not archive_failed:
            raise FileExistsError("B has partial output; explicitly archive it")
        archived = output / f"B_local5_interrupted_{stamp}"
        arm.rename(archived)
        log = output / "console_logs/B_local5.log"
        if log.exists():
            log.rename(log.with_name(f"B_local5_interrupted_{stamp}.log"))
    record = {"mode": "restart_B_only_from_original_raw50k", "completed_A_reused": True,
        "previous_device": previous, "current_device": hardware, "device_changed": changed,
        "accepted_cross_device_screening": accept_device_change,
        "archived_partial_B": str(archived) if archived else None,
        "input_fingerprint_policy": "unchanged: first five and every logged batch must match A",
        "runner_sha256": file_sha256(Path(__file__)), "runner_python": sys.executable}
    save_json(record, output / f"restart_B_{stamp}.json")
    print("Reusing completed A. Starting B only from original raw50k; input checks retained.", flush=True)
    train_arm(output, root, "B_local5", manifest["config"])
    audit = read_json(arm / "matched_baseline_inputs.json")
    metadata = read_json(arm / "final/run_metadata.json")
    if (audit["status"] != "passed"
            or metadata["global_step"] != manifest["protocol"]["expected_additional_steps"]
            or metadata["source_global_step"] != manifest["protocol"]["expected_source_step"]):
        raise ValueError("B incomplete or input audit failed")
    manifest["status"] = "trained_pending_evaluation"
    files = [p for p in (arm / "final").rglob("*") if p.is_file()]
    files.append(arm / "matched_baseline_inputs.json")
    manifest["trained_artifact_sha256"] = {str(p): file_sha256(p) for p in files}
    manifest.setdefault("execution_restarts", []).append(record)
    save_json(manifest, output / "experiment_manifest.json")
    return {"status": manifest["status"], "new_training_arms": ["B_local5"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--accept-device-change", action="store_true")
    parser.add_argument("--archive-failed-B", action="store_true")
    args = parser.parse_args()
    print(restart(args.config.resolve(), accept_device_change=args.accept_device_change,
                  archive_failed=args.archive_failed_B))


if __name__ == "__main__":
    main()
