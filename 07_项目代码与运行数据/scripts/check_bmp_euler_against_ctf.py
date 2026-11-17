"""Compare a BMP Euler reading with CTF Euler angles at the same grid position.

This check is mandatory when a CTF file is expected. A missing CTF is an error,
not a skipped test. The pixel formula is still an assumption until this report
says the errors are small.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ebsd_diffusion.euler_images import (
    circular_difference,
    default_euler_gb_config,
    rgb_to_euler_degrees,
    thin_boundary_mask,
)


def find_ctf_files(path: str | Path) -> list[Path]:
    candidate = Path(path)
    if candidate.is_file() and candidate.suffix.lower() == ".ctf":
        return [candidate]
    files = sorted(candidate.rglob("*.ctf")) if candidate.is_dir() else []
    if not files:
        raise FileNotFoundError(
            f"No CTF files at {candidate}. The pixel-to-Euler check cannot be skipped. "
            "Pass the AZtec .ctf file or its directory."
        )
    return files


def read_ctf(path: str | Path) -> dict[str, np.ndarray | float]:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    header: dict[str, float] = {}
    columns: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if columns is None and "Euler1" in parts and "X" in parts and "Y" in parts:
            columns = parts
            continue
        if columns is None:
            if len(parts) >= 2 and parts[0] in {"XCells", "YCells", "XStep", "YStep"}:
                header[parts[0]] = float(parts[1])
            continue
        if len(parts) < len(columns):
            continue
        rows.append(dict(zip(columns, parts)))
    if not rows or columns is None:
        raise ValueError(f"CTF file has no Euler rows: {path}")
    euler = np.asarray(
        [[float(row["Euler1"]), float(row["Euler2"]), float(row["Euler3"])] for row in rows],
        dtype=np.float64,
    )
    horizontal = np.asarray([float(row["X"]) for row in rows], dtype=np.float64)
    vertical = np.asarray([float(row["Y"]) for row in rows], dtype=np.float64)
    x_step = header.get("XStep", 1.0) or 1.0
    y_step = header.get("YStep", 1.0) or 1.0
    columns_px = np.rint(horizontal / x_step).astype(int)
    rows_px = np.rint(vertical / y_step).astype(int)
    width = int(header.get("XCells", int(columns_px.max()) + 1))
    height = int(header.get("YCells", int(rows_px.max()) + 1))
    grid = np.full((height, width, 3), np.nan, dtype=np.float64)
    valid = (rows_px >= 0) & (rows_px < height) & (columns_px >= 0) & (columns_px < width)
    grid[rows_px[valid], columns_px[valid]] = euler[valid]
    return {"grid": grid, "x_cells": width, "y_cells": height, "filled": int(valid.sum())}


def _one(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def _alignment_errors(bmp_euler: np.ndarray, ctf_grid: np.ndarray, use: np.ndarray) -> dict | None:
    height = min(bmp_euler.shape[0], ctf_grid.shape[0])
    width = min(bmp_euler.shape[1], ctf_grid.shape[1])
    reference = bmp_euler[:height, :width]
    candidate = ctf_grid[:height, :width]
    selected = use[:height, :width] & np.isfinite(candidate).all(axis=-1)
    if int(selected.sum()) < 10:
        return None
    errors = np.stack(
        [
            circular_difference(reference[..., 0], candidate[..., 0], 360.0),
            np.abs(reference[..., 1] - candidate[..., 1]),
            circular_difference(reference[..., 2], candidate[..., 2], 90.0),
        ],
        axis=-1,
    )[selected]
    summary = {"pixels": int(selected.sum()), "phi1": _one(errors[:, 0]), "Phi": _one(errors[:, 1]), "phi2": _one(errors[:, 2])}
    summary["median_of_medians"] = float(np.median([summary["phi1"]["median"], summary["Phi"]["median"], summary["phi2"]["median"]]))
    return summary


def compare_bmp_with_ctf(bmp_path: str | Path, ctf_path: str | Path, config: dict | None = None) -> dict:
    settings = default_euler_gb_config()
    if config:
        settings.update(config)
    with Image.open(bmp_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bmp_euler = rgb_to_euler_degrees(rgb, settings)
    boundary = thin_boundary_mask(rgb, settings)
    ctf = read_ctf(ctf_path)
    grid = ctf["grid"]
    direct = _alignment_errors(bmp_euler, grid, ~boundary)
    flipped = _alignment_errors(bmp_euler, grid[::-1], ~boundary)
    return {
        "bmp": str(Path(bmp_path).resolve()),
        "ctf": str(Path(ctf_path).resolve()),
        "bmp_shape": list(rgb.shape[:2]),
        "ctf_shape": [int(ctf["y_cells"]), int(ctf["x_cells"])],
        "shapes_equal": list(rgb.shape[:2]) == [int(ctf["y_cells"]), int(ctf["x_cells"])],
        "formula_degrees": {
            "phi1": f"R/255*{settings['phi1_degrees']}",
            "Phi": f"G/255*{settings['Phi_degrees']}",
            "phi2": f"B/255*{settings['phi2_degrees']}",
        },
        "alpha_ignored": True,
        "excluded_boundary_pixels": int(boundary.sum()),
        "direct_top_left_overlap": direct,
        "vertical_flip_top_left_overlap": flipped,
        "note": "The smaller error is a candidate alignment, not proof that the formula or the row direction is correct.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check BMP Euler pixels against CTF Euler angles.")
    parser.add_argument("--ctf", type=Path, required=True)
    parser.add_argument("--bmp", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        ctf_files = find_ctf_files(args.ctf)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not args.bmp.is_file():
        print(f"BMP does not exist: {args.bmp}", file=sys.stderr)
        return 2
    reports = [compare_bmp_with_ctf(args.bmp, path) for path in ctf_files]
    text = json.dumps(reports if len(reports) > 1 else reports[0], ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
