from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from ebsd_diffusion.utils import RunProgressTracker, save_json


def seed_for_alloy(base_seed: int, alloy_id: int, count_per_alloy: int) -> int:
    """Return a deterministic, non-overlapping seed range for one alloy."""
    return base_seed + (alloy_id - 1) * count_per_alloy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a fixed number of EBSD images for each selected alloy."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--alloy-ids", type=int, nargs="+", default=list(range(1, 23)))
    parser.add_argument("--count-per-alloy", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.8)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260721,
        help="Base seed; each alloy receives a deterministic, distinct RNG seed.",
    )
    parser.add_argument("--vae-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("08_训练与生成结果/04_22种合金正式生成图"),
    )
    args = parser.parse_args()

    session_started_monotonic = time.perf_counter()
    alloy_ids = [int(value) for value in args.alloy_ids]
    if len(set(alloy_ids)) != len(alloy_ids):
        raise ValueError("--alloy-ids contains duplicates")
    if any(value < 1 or value > 22 for value in alloy_ids):
        raise ValueError("--alloy-ids must be between 1 and 22")
    if args.count_per_alloy <= 0:
        raise ValueError("--count-per-alloy must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_images = len(alloy_ids) * args.count_per_alloy
    tracker = RunProgressTracker(
        output_dir=args.output_dir,
        run_name="All-alloy EBSD image generation",
        total_units=total_images,
        unit_name="saved_image",
        session_started_monotonic=session_started_monotonic,
        file_prefix="all_alloys",
    )

    completed_images = 0
    alloy_seeds: dict[str, int] = {}
    for position, alloy_id in enumerate(alloy_ids, start=1):
        alloy_seed = seed_for_alloy(args.seed, alloy_id, args.count_per_alloy)
        alloy_seeds[str(alloy_id)] = alloy_seed
        tracker.set_stage(
            f"generating_alloy_{alloy_id:02d}",
            completed_images,
            details={
                "alloy_position": position,
                "total_alloys": len(alloy_ids),
                "count_per_alloy": args.count_per_alloy,
                "alloy_seed": alloy_seed,
            },
        )
        command = [
            sys.executable,
            "sample.py",
            "--checkpoint",
            str(args.checkpoint),
            "--alloy-id",
            str(alloy_id),
            "--count",
            str(args.count_per_alloy),
            "--batch-size",
            str(args.batch_size),
            "--steps",
            str(args.steps),
            "--guidance-scale",
            str(args.guidance_scale),
            "--seed",
            str(alloy_seed),
            "--output-dir",
            str(args.output_dir / f"alloy_{alloy_id:02d}"),
        ]
        if args.vae_dir is not None:
            command += ["--vae-dir", str(args.vae_dir)]
        subprocess.run(command, check=True)
        completed_images += args.count_per_alloy
        tracker.update(
            completed_images,
            details={"completed_alloy_id": alloy_id},
        )

    final_timing = tracker.finish(completed_images)
    save_json(
        {
            "alloy_ids": alloy_ids,
            "count_per_alloy": args.count_per_alloy,
            "total_output_images": total_images,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "base_seed": args.seed,
            "alloy_seeds": alloy_seeds,
            "timing": final_timing,
        },
        args.output_dir / "generation_summary.json",
    )


if __name__ == "__main__":
    main()
