"""Reuse completed A for explicitly recorded cross-device screening; train B only.

First-batch A/B replay on the current GPU must match. Historical GPU tensor
hashes are recorded as diagnostics, not claimed to be identical. Old runs stay
immutable; the derived run uses a symlink to the completed A.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil

import yaml

from ebsd_diffusion.training_control import file_sha256
from ebsd_diffusion.utils import save_json
from scripts.audit_descriptor_inputs import audit
from scripts.run_descriptor_pilot import (
    ROOT, NEW_CODE, current_code_files, device_profile, evaluate, load_prepared,
    read_json, read_protocol, require_cuda, train_arm, verify_control,
)


def prepare(config_path, root=ROOT):
    protocol, source = read_protocol(config_path, root)
    old = read_json(source / "experiment_manifest.json")
    if old["project_root"] != str(root) or old["status"] != "control_complete_pending_local5":
        raise ValueError("Need this project's completed A, pending B")
    # Only the deliberately updated trainer may differ. Its original archived
    # version must still match the original manifest; no data/config exception.
    migrations = []
    for name, expected in old["frozen_sha256"].items():
        path = Path(name)
        if path.is_file() and file_sha256(path) == expected:
            continue
        archived = source / "source_snapshot/train_diffusion.py"
        if path != root / "train_diffusion.py" or file_sha256(archived) != expected:
            raise ValueError(f"Original prepared input changed: {name}")
        migrations.append({"path": name, "old_sha256": expected, "new_sha256": file_sha256(path),
            "original_archive": str(archived), "reason": "explicit cross-device input audit policy"})
    verify_control(old)
    require_cuda(old)
    hardware = device_profile(protocol)
    if shutil.disk_usage(root).free < 8 * 1024**3:
        raise RuntimeError("Need at least 8 GiB free disk space")
    replay = audit(config_path, root=root, control_run=source)
    if (not replay["new_code_matches_archived_replay_here"]
            or not all(replay["components_equal_here"].values())):
        raise ValueError("Same-device A/B first batch differs; inspect replay before training")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = source.parent / f"10_Fold03_local5_cross_device_{stamp}_运行"
    output.mkdir()
    (output / "A_mse").symlink_to(source / "A_mse", target_is_directory=True)
    shutil.copytree(source / "labels", output / "labels")
    (output / "configs").mkdir()
    replay_path = output / "zero_update_replay.json"
    save_json(replay, replay_path)
    new_config = yaml.safe_load(Path(old["config"]).read_text("utf-8"))
    new_config["train_diffusion"]["output_dir"] = str(output / "B_local5")
    desc = new_config["train_diffusion"]["local_descriptors"]
    desc.update({"metadata": str(output / "labels/metadata.json"),
        "input_comparison_policy": "cross_device_screening", "zero_update_replay": str(replay_path),
        "zero_update_replay_sha256": file_sha256(replay_path)})
    config = output / "configs/B_local5.yaml"
    config.write_text(yaml.safe_dump(new_config, allow_unicode=True, sort_keys=False), "utf-8")
    protocol = deepcopy(protocol); protocol["output_root"] = str(output)
    protocol_path = output / "configs/protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, allow_unicode=True, sort_keys=False), "utf-8")
    code = sorted(set(current_code_files(root) + [root / n for n in NEW_CODE] + [
        Path(__file__).resolve(), root / "scripts/audit_descriptor_inputs.py"]))
    for path in code:
        destination = output / "source_snapshot" / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, destination)
    manifest = deepcopy(old)
    manifest.update({"status": "control_complete_pending_local5", "protocol": protocol,
        "config": str(config), "device_profile": hardware, "control_reused_from": str(source),
        "code_migrations": migrations, "input_comparison_policy": "cross_device_screening",
        "scientific_scope": "cross-device training-alloy RGB-proxy screening; first batch A/B replay verified; historical per-step tensor identity not claimed",
        "new_training_arms": ["B_local5"]})
    # The derived manifest records the new code honestly; old hashes/manifests
    # remain in the source run and source_snapshot, never patched in place.
    frozen = dict(old["frozen_sha256"])
    frozen.update({str(p): file_sha256(p) for p in code})
    frozen.update({str(p): file_sha256(p) for p in output.rglob("*") if p.is_file()})
    frozen[str(source / "experiment_manifest.json")] = file_sha256(source / "experiment_manifest.json")
    manifest["frozen_sha256"] = frozen
    save_json(manifest, output / "experiment_manifest.json")
    print(f"RUN_CONFIG={protocol_path}", flush=True)
    return protocol_path


def train(config_path, root=ROOT):
    output, manifest = load_prepared(config_path, root)
    require_cuda(manifest); verify_control(manifest)
    train_arm(output, root, "B_local5", manifest["config"])
    arm = output / "B_local5"
    result = read_json(arm / "matched_baseline_inputs.json")
    metadata = read_json(arm / "final/run_metadata.json")
    if (result["status"] != "completed_cross_device_screening"
            or metadata["global_step"] != manifest["protocol"]["expected_additional_steps"]
            or metadata["source_global_step"] != manifest["protocol"]["expected_source_step"]):
        raise ValueError("Incomplete B training or cross-device audit")
    files = [p for p in (arm / "final").rglob("*") if p.is_file()]
    files += [arm / "matched_baseline_inputs.json", arm / "cross_device_inputs.csv"]
    manifest["trained_artifact_sha256"] = {str(p): file_sha256(p) for p in files}
    manifest["status"] = "trained_pending_evaluation"
    save_json(manifest, output / "experiment_manifest.json")
    return {"status": manifest["status"], "new_training_arms": ["B_local5"], "config": str(config_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("run", "train", "evaluate"), default="run")
    args = parser.parse_args()
    if args.phase == "run":
        result = train(prepare(args.config.resolve()))
    elif args.phase == "train":
        result = train(args.config.resolve())
    else:
        result = evaluate(args.config.resolve())
    print(result, flush=True)


if __name__ == "__main__":
    main()
