"""Compare one reconstructed Euler+GB window with the original window."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ebsd_diffusion.orientation_metrics import reconstruction_report


def load_map(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    euler = np.asarray(data["euler_degrees"], dtype=np.float64)
    boundary = np.asarray(data["boundary"], dtype=np.float64)
    if euler.ndim != 3 or euler.shape[-1] != 3:
        raise ValueError(f"{path} euler_degrees must have shape HxWx3.")
    return euler, boundary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VAE acceptance numbers for one Euler+GB reconstruction.")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--reconstructed", type=Path, required=True)
    parser.add_argument("--threshold-degrees", type=float, default=15.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    original_euler, original_boundary = load_map(args.original)
    reconstructed_euler, reconstructed_boundary = load_map(args.reconstructed)
    report = reconstruction_report(
        original_euler,
        original_boundary,
        reconstructed_euler,
        reconstructed_boundary,
        threshold_degrees=args.threshold_degrees,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
