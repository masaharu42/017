"""Evaluate existing generated-image ablations under one reproducible protocol.

The script intentionally evaluates files that already exist. It does not load a
diffusion model or start sampling, so historical results remain immutable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from ebsd_diffusion.metrics import (
    ImageDescriptors,
    compare_descriptor_sets,
    describe_image,
    descriptors_as_dict,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SCALAR_FIELDS = (
    "gb_fraction",
    "grain_count",
    "grain_area_median_px",
    "grain_area_q25_px",
    "grain_area_q75_px",
    "lab_mean_l",
    "lab_mean_a",
    "lab_mean_b",
    "lab_std_l",
    "lab_std_a",
    "lab_std_b",
    "edge_density",
)
SEED_PATTERN = re.compile(r"seed(?P<seed>\d+)", re.IGNORECASE)
SAMPLE_PATTERN = re.compile(r"_(?P<sample>\d+)(?=\.[^.]+$)")


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def extract_seed(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = SEED_PATTERN.search(part)
        if match:
            return int(match.group("seed"))
    return None


def extract_sample_index(path: Path) -> int | None:
    match = SAMPLE_PATTERN.search(path.name)
    return int(match.group("sample")) if match else None


def collect_image_records(
    project_root: Path,
    method: dict[str, Any],
    expected_seeds: Iterable[int] | None = None,
    expected_count_per_seed: int | None = None,
) -> list[dict[str, Any]]:
    directory = resolve_path(project_root, method["image_dir"])
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    pattern = str(method.get("glob", "**/*"))
    paths = sorted(
        path
        for path in directory.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No images matched {pattern!r} under {directory}")

    records: list[dict[str, Any]] = []
    seen_keys: set[tuple[int | None, int | None]] = set()
    for path in paths:
        seed = extract_seed(path)
        sample_index = extract_sample_index(path)
        key = (seed, sample_index)
        if key in seen_keys:
            raise ValueError(f"Duplicate seed/sample image for {method['name']}: {key}")
        seen_keys.add(key)
        records.append(
            {
                "method": str(method["name"]),
                "path": str(path.resolve()),
                "seed": seed,
                "sample_index": sample_index,
            }
        )

    if expected_seeds is not None:
        expected = {int(seed) for seed in expected_seeds}
        actual = {record["seed"] for record in records}
        if actual != expected:
            raise ValueError(
                f"Method {method['name']} has seeds {sorted(actual)}, expected {sorted(expected)}."
            )
    if expected_count_per_seed is not None:
        counts: dict[int | None, int] = {}
        for record in records:
            counts[record["seed"]] = counts.get(record["seed"], 0) + 1
        bad = {seed: count for seed, count in counts.items() if count != expected_count_per_seed}
        if bad:
            raise ValueError(
                f"Method {method['name']} has unexpected per-seed image counts: {bad}; "
                f"expected {expected_count_per_seed}."
            )
    return records


def choose_crop_size(
    paths: Iterable[Path], configured: list[int] | tuple[int, int] | None
) -> tuple[int, int]:
    if configured is not None:
        if len(configured) != 2 or any(int(value) <= 0 for value in configured):
            raise ValueError("crop_size must be [width, height] with positive values")
        return int(configured[0]), int(configured[1])
    sizes = [Image.open(path).size for path in paths]
    return min(size[0] for size in sizes), min(size[1] for size in sizes)


def scalar_summary(descriptors: list[ImageDescriptors]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for field in SCALAR_FIELDS:
        values = np.asarray([getattr(item, field) for item in descriptors], dtype=float)
        summary[field] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.quantile(values, 0.50)),
            "q75": float(np.quantile(values, 0.75)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def _group_by_seed(records: list[dict[str, Any]]) -> dict[int | None, list[dict[str, Any]]]:
    groups: dict[int | None, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["seed"], []).append(record)
    for group in groups.values():
        group.sort(key=lambda item: (item["sample_index"] is None, item["sample_index"] or 0))
    return groups


def build_montage(
    real_path: Path,
    method_records: dict[str, list[dict[str, Any]]],
    output_path: Path,
    seeds: list[int],
    sample_index: int = 0,
    tile_size: tuple[int, int] = (320, 240),
) -> None:
    columns = ["real", *method_records.keys()]
    rows: list[tuple[int, dict[str, Path]]] = []
    grouped = {name: _group_by_seed(records) for name, records in method_records.items()}
    for seed in seeds:
        selected: dict[str, Path] = {"real": real_path}
        for name in method_records:
            candidates = grouped[name].get(seed, [])
            match = next(
                (record for record in candidates if record["sample_index"] == sample_index),
                candidates[0] if candidates else None,
            )
            if match is None:
                raise ValueError(f"No image found for method {name!r}, seed {seed}.")
            selected[name] = Path(match["path"])
        rows.append((seed, selected))

    label_height = 24
    canvas = Image.new(
        "RGB",
        (tile_size[0] * len(columns), (tile_size[1] + label_height) * len(rows)),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row_index, (seed, selected) in enumerate(rows):
        for column_index, name in enumerate(columns):
            left = column_index * tile_size[0]
            top = row_index * (tile_size[1] + label_height)
            image = Image.open(selected[name]).convert("RGB")
            image.thumbnail(tile_size)
            paste_left = left + (tile_size[0] - image.width) // 2
            paste_top = top + label_height + (tile_size[1] - image.height) // 2
            canvas.paste(image, (paste_left, paste_top))
            label = "Real" if name == "real" else name
            if row_index == 0:
                draw.text((left + 6, top + 5), label, fill=(20, 20, 20), font=font)
        draw.text((6, top + label_height + 5), f"seed {seed}", fill=(20, 20, 20), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_summary_csv(
    output_path: Path,
    summary: dict[str, Any],
    seed_summary: list[dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    for method, value in summary.items():
        comparison = value["comparison_to_real"]
        for metric, distribution in value["scalar"].items():
            row = {
                "scope": "all_images",
                "method": method,
                "seed": "",
                "image_count": value["image_count"],
                "metric": metric,
                **distribution,
                "real_mean": comparison[metric]["real_mean"],
                "generated_mean": comparison[metric]["generated_mean"],
                "relative_error": comparison[metric]["relative_error"],
            }
            rows.append(row)
    for value in seed_summary:
        comparison = value["comparison_to_real"]
        for metric, distribution in value["scalar"].items():
            rows.append(
                {
                    "scope": "per_seed",
                    "method": value["method"],
                    "seed": value["seed"],
                    "image_count": value["image_count"],
                    "metric": metric,
                    **distribution,
                    "real_mean": comparison[metric]["real_mean"],
                    "generated_mean": comparison[metric]["generated_mean"],
                    "relative_error": comparison[metric]["relative_error"],
                }
            )
    fieldnames = [
        "scope",
        "method",
        "seed",
        "image_count",
        "metric",
        "mean",
        "std",
        "q25",
        "median",
        "q75",
        "min",
        "max",
        "real_mean",
        "generated_mean",
        "relative_error",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    project_root = config_path.parent.parent.resolve()
    output_dir = resolve_path(project_root, output_override or config["output_dir"])
    expected_seeds = [int(value) for value in config.get("seeds", [])] or None
    expected_count = config.get("count_per_seed")
    expected_count = int(expected_count) if expected_count is not None else None

    real_paths = [resolve_path(project_root, value) for value in config["real_images"]]
    for path in real_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Real image does not exist: {path}")
    crop_size = choose_crop_size(real_paths, config.get("crop_size"))
    gb_threshold = float(config.get("gb_threshold", 0.18))
    real_descriptors = [describe_image(path, gb_threshold, crop_size=crop_size) for path in real_paths]

    method_records: dict[str, list[dict[str, Any]]] = {}
    method_descriptors: dict[str, list[ImageDescriptors]] = {}
    per_image: list[dict[str, Any]] = []
    methods_metadata: list[dict[str, Any]] = []
    for method in config["methods"]:
        name = str(method["name"])
        if name in method_records:
            raise ValueError(f"Duplicate method name: {name}")
        records = collect_image_records(project_root, method, expected_seeds, expected_count)
        descriptors: list[ImageDescriptors] = []
        for record in records:
            descriptor = describe_image(record["path"], gb_threshold, crop_size=crop_size)
            descriptors.append(descriptor)
            per_image.append({**record, "descriptors": descriptors_as_dict(descriptor)})
        method_records[name] = records
        method_descriptors[name] = descriptors
        metadata = {"name": name, "image_dir": str(resolve_path(project_root, method["image_dir"]))}
        for key in ("checkpoint", "vae_dir", "run_metadata", "config"):
            if key in method:
                path = resolve_path(project_root, method[key])
                metadata[key] = str(path)
                metadata[f"{key}_exists"] = path.exists()
        methods_metadata.append(metadata)

    seed_summary: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for name, descriptors in method_descriptors.items():
        summary[name] = {
            "image_count": len(descriptors),
            "scalar": scalar_summary(descriptors),
            "comparison_to_real": compare_descriptor_sets(real_descriptors, descriptors),
        }
        record_index = {id(record): index for index, record in enumerate(method_records[name])}
        for seed, records in _group_by_seed(method_records[name]).items():
            seed_descriptors = [descriptors[record_index[id(record)]] for record in records]
            seed_summary.append(
                {
                    "method": name,
                    "seed": seed,
                    "image_count": len(seed_descriptors),
                    "scalar": scalar_summary(seed_descriptors),
                    "comparison_to_real": compare_descriptor_sets(real_descriptors, seed_descriptors),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_image_metrics.json").write_text(
        json.dumps(per_image, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "real_metrics.json").write_text(
        json.dumps(
            [
                {"path": str(path), "descriptors": descriptors_as_dict(descriptor)}
                for path, descriptor in zip(real_paths, real_descriptors)
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "seed_summary.json").write_text(
        json.dumps(seed_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary_csv(output_dir / "summary.csv", summary, seed_summary)
    if expected_seeds:
        build_montage(
            real_paths[0],
            method_records,
            output_dir / "comparison_grid.png",
            seeds=expected_seeds,
            sample_index=int(config.get("montage_sample_index", 0)),
        )

    run_metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "project_root": str(project_root),
        "fold": config.get("fold"),
        "evaluation_space": config.get("evaluation_space", "unspecified"),
        "gb_threshold": gb_threshold,
        "crop_width": crop_size[0],
        "crop_height": crop_size[1],
        "real_images": [str(path) for path in real_paths],
        "methods": methods_metadata,
        "seeds": expected_seeds,
        "count_per_seed": expected_count,
        "generated_images": sum(len(records) for records in method_records.values()),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"summary": summary, "seed_summary": seed_summary, "run_metadata": run_metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = evaluate(args.config, args.output_dir)
    print(json.dumps(result["run_metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
