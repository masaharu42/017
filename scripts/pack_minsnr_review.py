"""Package server reports for review without model weights or checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def pack_review(run_dir: Path, destination: Path | None = None) -> dict:
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "experiment_manifest.json").read_text("utf-8"))
    if not (run_dir / "evaluation" / "run_metadata.json").is_file():
        raise ValueError("Complete the monitor evaluation before packaging results.")
    for name in ("evaluation", "evaluation_heldout"):
        report_path = run_dir / name / "run_metadata.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text("utf-8"))
            for relative, expected in report.get("artifact_sha256", {}).items():
                if digest(run_dir / name / relative) != expected:
                    raise ValueError(f"Evaluation artifact changed before packaging: {name}/{relative}")
    suffix = "with_id03" if (run_dir / "evaluation_heldout" / "run_metadata.json").is_file() else "monitor"
    destination = destination.resolve() if destination else run_dir / f"review_bundle_{suffix}.tar.gz"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite review bundle: {destination}")
    files = {run_dir / "experiment_manifest.json", run_dir / "matched_input_verification.json"}
    for folder in ("configs", "source_snapshot", "evaluation", "evaluation_heldout", "console_logs"):
        files.update(path for path in (run_dir / folder).rglob("*") if path.is_file())
    files.update(run_dir.glob("*.log"))
    for arm in ("A_mse", "B_minsnr5"):
        for name in ("training_metrics.csv", "timestep_metrics.csv", "training_diagnostics.json",
                     "input_fingerprints.csv", "initialization_metadata.json", "final/run_metadata.json"):
            path = run_dir / arm / name
            if path.is_file():
                files.add(path)
    index = {}
    for path in sorted(files):
        if path.is_symlink() or not path.resolve().is_relative_to(run_dir):
            raise ValueError(f"Review file escapes the run directory: {path}")
        if path.suffix.lower() in {".safetensors", ".bin", ".pkl", ".pt", ".pth"}:
            raise ValueError(f"Model weights do not belong in the review package: {path}")
        index[path.relative_to(run_dir).as_posix()] = {"sha256": digest(path), "size": path.stat().st_size}
    payload = json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": manifest["name"], "git_revision": manifest.get("git_revision"),
        "scope": "reports, PNGs, configs, logs and source; no model weights or training data",
        "files": index,
    }, ensure_ascii=False, indent=2).encode("utf-8")
    with tarfile.open(destination, "x:gz") as archive:
        for path in sorted(files):
            archive.add(path, arcname=path.relative_to(run_dir).as_posix(), recursive=False)
        entry = tarfile.TarInfo("REVIEW_FILES_SHA256.json")
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))
    checksum = digest(destination)
    with Path(str(destination) + ".sha256").open("x", encoding="utf-8") as handle:
        handle.write(f"{checksum}  {destination.name}\n")
    return {"path": str(destination), "sha256": checksum, "file_count": len(files),
            "size_bytes": destination.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(pack_review(args.run_dir, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
