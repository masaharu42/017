"""Nearest training window of a generated Euler+GB map.

Orientation distance and boundary distance are reported separately. The search
uses a fixed stride over the training maps. It does not rank seeds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ebsd_diffusion.orientation_metrics import nearest_training_window


def load_map(path: Path) -> dict:
    data = np.load(path)
    return {
        "alloy_id": int(data["alloy_id"]) if "alloy_id" in data.files else -1,
        "euler_degrees": np.asarray(data["euler_degrees"], dtype=np.float64),
        "boundary": np.asarray(data["boundary"], dtype=np.float64),
        "path": str(path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find the nearest training window of one generated map.")
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    query = load_map(args.query)
    library = [load_map(path) for path in sorted(args.library.glob("*.npz"))]
    report = nearest_training_window(query["euler_degrees"], query["boundary"], library, args.stride)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
