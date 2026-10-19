"""Compare A-baseline and B3 raw samples with simple EBSD structure descriptors.

This is an image-only evaluation.  It does not load a model or require a GPU.
The descriptors are deliberately reported as proxies: black-pixel ratio is a
GB-area proxy, gradient density is an edge proxy, and connected regions are a
rough grain-count proxy.  They should be used for matched comparisons, not as
standalone physical measurements.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-image", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--b3-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261012, 20261013, 20261014, 20261015, 20261016])
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if (width, height) == (750, 451):
        image = image.crop((0, 0, 750, 450))
    elif image.size != (750, 450):
        image = image.resize((750, 450), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def descriptors(array: np.ndarray) -> dict[str, float]:
    gray = 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
    black = np.all(array <= (35.0 / 255.0), axis=-1)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge_x = gx > (20.0 / 255.0)
    edge_y = gy > (20.0 / 255.0)
    edge_density = 0.5 * (float(edge_x.mean()) + float(edge_y.mean()))

    # Dilating the dark boundary proxy makes the region count less sensitive
    # to one-pixel gaps. It is only a rough grain-count proxy.
    boundary = ndimage.binary_dilation(black, iterations=1)
    regions, count = ndimage.label(~boundary, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(regions.ravel())[1:]
    grain_count_proxy = float(np.count_nonzero(sizes >= 100))

    quantized = np.floor(array * 8.0).astype(np.int16)
    colors = quantized.reshape(-1, 3)
    _, counts = np.unique(colors, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    color_entropy = float(-(probabilities * np.log2(probabilities)).sum())
    return {
        "gb_area_ratio_proxy": float(black.mean()),
        "edge_density_proxy": edge_density,
        "grain_count_proxy": grain_count_proxy,
        "gray_std": float(gray.std()),
        "color_entropy": color_entropy,
    }


def add_record(records: list[dict[str, object]], group: str, checkpoint: str, seed: str, path: Path, reference: dict[str, float]) -> None:
    metrics = descriptors(load_rgb(path))
    record: dict[str, object] = {"group": group, "checkpoint": checkpoint, "seed": seed, "path": str(path)}
    record.update(metrics)
    relative_errors = [
        abs(metrics[name] - reference[name]) / max(abs(reference[name]), 1.0e-6)
        for name in reference
    ]
    record["descriptor_distance_to_real"] = float(np.mean(relative_errors))
    records.append(record)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    real = descriptors(load_rgb(args.real_image))
    records: list[dict[str, object]] = []
    add_record(records, "real", "reference", "reference", args.real_image, real)

    for seed in args.seeds:
        seed_text = str(seed)
        baseline = args.baseline_dir / f"seed{seed_text}_n5" / f"alloy_03_seed{seed_text}_000.png"
        if baseline.is_file():
            add_record(records, "A_baseline_raw", "20000", seed_text, baseline, real)
        for checkpoint in ("10000", "15000", "20000"):
            sample = args.b3_dir / f"step{checkpoint}_raw" / f"seed_{seed_text}" / f"alloy_03_seed{seed_text}_000.png"
            if sample.is_file():
                add_record(records, "B3_raw", checkpoint, seed_text, sample, real)

    fieldnames = list(records[0])
    with (args.output / "descriptor_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    groups = [record for record in records if record["group"] != "real"]
    summary_fields = ["group", "checkpoint", "n", *real.keys(), "descriptor_distance_to_real"]
    summary_rows: list[dict[str, object]] = []
    for group in sorted({(row["group"], row["checkpoint"]) for row in groups}):
        selected = [row for row in groups if (row["group"], row["checkpoint"]) == group]
        row: dict[str, object] = {"group": group[0], "checkpoint": group[1], "n": len(selected)}
        for name in real:
            row[name] = float(np.mean([float(item[name]) for item in selected]))
        row["descriptor_distance_to_real"] = float(np.mean([float(item["descriptor_distance_to_real"]) for item in selected]))
        summary_rows.append(row)
    with (args.output / "descriptor_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {args.output / 'descriptor_metrics.csv'}")
    print(f"wrote {args.output / 'descriptor_summary.csv'}")


if __name__ == "__main__":
    main()
