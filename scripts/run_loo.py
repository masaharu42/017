from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ebsd_diffusion.config import load_config, resolve_project_path
from ebsd_diffusion.utils import RunProgressTracker, save_json


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
RELATIVE_ERROR_METRICS = (
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
)
DIRECT_METRICS = ("two_point_rmse", "radial_power_rmse", "hue_jensen_shannon")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run complete leave-one-alloy-out folds: strict training, held-out "
            "generation, descriptor evaluation, and aggregate reporting."
        )
    )
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--compositions", type=Path)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Set both VAE and diffusion steps. Intended only for code checks.",
    )
    parser.add_argument("--vae-max-steps", type=int)
    parser.add_argument("--diffusion-max-steps", type=int)
    parser.add_argument(
        "--retrain-vae",
        action="store_true",
        help="Strict image holdout: retrain the VAE without each held-out alloy.",
    )
    parser.add_argument("--samples-per-fold", type=int)
    parser.add_argument("--sample-batch-size", type=int, default=1)
    parser.add_argument("--sample-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gb-threshold", type=float, default=0.18)
    parser.add_argument(
        "--progress-dir",
        type=Path,
        default=Path("08_训练与生成结果/05_留一验证"),
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Do not resume an incomplete training stage from its newest checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    session_started_monotonic = time.perf_counter()
    args = parse_args()
    config = load_config(args.config)
    project_root = Path(config["_project_root"])
    progress_dir = resolve_from_root(project_root, args.progress_dir)
    progress_dir.mkdir(parents=True, exist_ok=True)

    folds = validate_folds(args.folds)
    compositions_path = (
        resolve_from_root(project_root, args.compositions)
        if args.compositions is not None
        else resolve_project_path(config, config["data"]["compositions"])
    )
    manifest_path = resolve_project_path(config, config["data"]["manifest"])
    validate_conditional_compositions(folds, compositions_path)
    held_out_images = load_held_out_images(folds, manifest_path)

    vae_steps, diffusion_steps = resolve_training_steps(args, config)
    sampling_cfg = config["sampling"]
    samples_per_fold = positive_int(
        args.samples_per_fold
        if args.samples_per_fold is not None
        else sampling_cfg["preview_count"],
        "samples_per_fold",
    )
    sample_steps = positive_int(
        args.sample_steps
        if args.sample_steps is not None
        else sampling_cfg["num_inference_steps"],
        "sample_steps",
    )
    sample_batch_size = positive_int(args.sample_batch_size, "sample_batch_size")
    guidance_scale = float(
        args.guidance_scale
        if args.guidance_scale is not None
        else sampling_cfg["guidance_scale"]
    )
    base_seed = int(args.seed if args.seed is not None else config["seed"])

    base_vae_final = resolve_project_path(config, config["train_diffusion"]["vae_dir"])

    state_path = progress_dir / "loo_state.json"
    state = load_json_or_default(
        state_path,
        {
            "status": "not_started",
            "folds": folds,
            "strict_vae_retraining": bool(args.retrain_vae),
            "fold_status": {},
        },
    )
    state.update(
        {
            "status": "running",
            "folds": folds,
            "strict_vae_retraining": bool(args.retrain_vae),
            "last_update_local": local_now(),
            "error": None,
        }
    )
    save_json(state, state_path)

    existing_summaries = {
        fold: load_valid_fold_summary(
            progress_dir / f"fold_{fold:02d}" / "fold_summary.json",
            fold=fold,
            vae_steps=vae_steps,
            diffusion_steps=diffusion_steps,
            samples_per_fold=samples_per_fold,
            sample_steps=sample_steps,
            guidance_scale=guidance_scale,
            strict_vae_retraining=bool(args.retrain_vae),
        )
        for fold in folds
    }
    previous_elapsed = read_previous_elapsed(progress_dir / "loo_progress.json")
    tracker = RunProgressTracker(
        output_dir=progress_dir,
        run_name="Complete representative leave-one-alloy-out validation",
        total_units=len(folds),
        unit_name="completed_fold",
        initial_units=0,
        previous_elapsed_seconds=previous_elapsed,
        session_started_monotonic=session_started_monotonic,
        file_prefix="loo",
    )

    completed_summaries: list[dict[str, Any]] = []
    completed_count = 0
    active_fold: int | None = None
    try:
        for fold_position, fold in enumerate(folds, start=1):
            active_fold = fold
            existing = existing_summaries[fold]
            if existing is not None:
                completed_summaries.append(existing)
                completed_count += 1
                tracker.update(
                    completed_count,
                    details={"completed_fold": fold, "resumed_from_existing_summary": True},
                )
                print(f"LOO fold {fold:02d}: complete result already exists; skipping.", flush=True)
                continue

            fold_dir = progress_dir / f"fold_{fold:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            model_scope = (
                "models_strict"
                if args.retrain_vae
                else "models_condition_only"
            )
            vae_output = fold_dir / model_scope / "vae"
            diffusion_output = fold_dir / model_scope / "diffusion"
            vae_final = vae_output / "final" if args.retrain_vae else base_vae_final
            diffusion_final = diffusion_output / "final"
            fold_seed = base_seed + fold
            run_name = generation_run_name(
                seed=fold_seed,
                count=samples_per_fold,
                steps=sample_steps,
                guidance_scale=guidance_scale,
            )
            generation_dir = fold_dir / "generated" / run_name
            evaluation_path = fold_dir / f"evaluation_{run_name}.json"

            update_fold_state(
                state,
                state_path,
                fold,
                status="running",
                stage="initializing",
                fold_position=fold_position,
                total_folds=len(folds),
            )

            if args.retrain_vae:
                tracker.set_stage(
                    f"fold_{fold:02d}_vae_training",
                    completed_count,
                    details={
                        "fold": fold,
                        "fold_position": fold_position,
                        "total_folds": len(folds),
                        "child_progress": str(vae_output / "run_progress.json"),
                    },
                )
                update_fold_state(
                    state,
                    state_path,
                    fold,
                    status="running",
                    stage="vae_training",
                )
                ensure_vae_model(
                    project_root=project_root,
                    config_path=Path(config["_config_path"]),
                    output_dir=vae_output,
                    final_dir=vae_final,
                    fold=fold,
                    max_steps=vae_steps,
                    auto_resume=not args.no_auto_resume,
                )
            elif not base_vae_is_available(base_vae_final):
                raise FileNotFoundError(
                    f"Base VAE is unavailable at {base_vae_final}. "
                    "Use --retrain-vae for strict LOO."
                )

            tracker.set_stage(
                f"fold_{fold:02d}_diffusion_training",
                completed_count,
                details={
                    "fold": fold,
                    "fold_position": fold_position,
                    "total_folds": len(folds),
                    "child_progress": str(diffusion_output / "run_progress.json"),
                },
            )
            update_fold_state(
                state,
                state_path,
                fold,
                status="running",
                stage="diffusion_training",
            )
            ensure_diffusion_model(
                project_root=project_root,
                config_path=Path(config["_config_path"]),
                output_dir=diffusion_output,
                final_dir=diffusion_final,
                vae_final=vae_final,
                fold=fold,
                max_steps=diffusion_steps,
                auto_resume=not args.no_auto_resume,
            )

            tracker.set_stage(
                f"fold_{fold:02d}_held_out_generation",
                completed_count,
                details={
                    "fold": fold,
                    "requested_images": samples_per_fold,
                    "output_dir": str(generation_dir),
                },
            )
            update_fold_state(
                state,
                state_path,
                fold,
                status="running",
                stage="held_out_generation",
            )
            generation_report = ensure_held_out_generation(
                project_root=project_root,
                checkpoint=diffusion_final,
                vae_final=vae_final,
                compositions_path=compositions_path,
                output_dir=generation_dir,
                fold=fold,
                count=samples_per_fold,
                batch_size=sample_batch_size,
                steps=sample_steps,
                guidance_scale=guidance_scale,
                seed=fold_seed,
                sampling_cfg=sampling_cfg,
                max_nearest_distance=float(
                    config["composition"]["max_normalized_extrapolation"]
                ),
            )

            tracker.set_stage(
                f"fold_{fold:02d}_held_out_evaluation",
                completed_count,
                details={
                    "fold": fold,
                    "real_image": str(held_out_images[fold]),
                    "generated_dir": str(generation_dir),
                },
            )
            update_fold_state(
                state,
                state_path,
                fold,
                status="running",
                stage="held_out_evaluation",
            )
            evaluation = ensure_evaluation(
                project_root=project_root,
                real_image=held_out_images[fold],
                generation_dir=generation_dir,
                output_path=evaluation_path,
                expected_count=samples_per_fold,
                gb_threshold=float(args.gb_threshold),
            )

            fold_summary = build_fold_summary(
                fold=fold,
                fold_position=fold_position,
                total_folds=len(folds),
                strict_vae_retraining=bool(args.retrain_vae),
                vae_steps=vae_steps,
                diffusion_steps=diffusion_steps,
                samples_per_fold=samples_per_fold,
                sample_steps=sample_steps,
                guidance_scale=guidance_scale,
                seed=fold_seed,
                real_image=held_out_images[fold],
                vae_final=vae_final,
                diffusion_final=diffusion_final,
                generation_dir=generation_dir,
                evaluation_path=evaluation_path,
                generation_report=generation_report,
                evaluation=evaluation,
            )
            save_json(fold_summary, fold_dir / "fold_summary.json")
            completed_summaries.append(fold_summary)
            completed_count += 1
            update_fold_state(
                state,
                state_path,
                fold,
                status="completed",
                stage="completed",
                fold_summary=str(fold_dir / "fold_summary.json"),
            )
            tracker.update(
                completed_count,
                details={
                    "completed_fold": fold,
                    "fold_summary": str(fold_dir / "fold_summary.json"),
                },
            )

        completed_summaries.sort(key=lambda value: int(value["held_out_alloy_id"]))
        aggregate = aggregate_fold_summaries(completed_summaries)
        final_timing = tracker.finish(
            len(folds),
            details={"completed_folds": folds, "aggregate_report_ready": True},
        )
        summary = {
            "status": "completed",
            "created_at_local": local_now(),
            "folds": folds,
            "strict_vae_retraining": bool(args.retrain_vae),
            "scientific_scope": (
                "Strict image-and-condition holdout"
                if args.retrain_vae
                else "Condition holdout only; base VAE saw held-out images"
            ),
            "parameters": {
                "vae_max_steps": vae_steps,
                "diffusion_max_steps": diffusion_steps,
                "samples_per_fold": samples_per_fold,
                "sample_steps": sample_steps,
                "guidance_scale": guidance_scale,
                "base_seed": base_seed,
                "gb_threshold": float(args.gb_threshold),
            },
            "aggregate_metrics": aggregate,
            "per_fold": completed_summaries,
            "timing": final_timing,
            "interpretation_note": (
                "This report provides diagnostic descriptor errors only. "
                "No automatic pass/fail threshold was imposed."
            ),
        }
        save_json(summary, progress_dir / "loo_summary.json")
        write_summary_csv(completed_summaries, progress_dir / "loo_summary.csv")
        state.update(
            {
                "status": "completed",
                "current_fold": None,
                "current_stage": "completed",
                "last_update_local": local_now(),
                "error": None,
            }
        )
        save_json(state, state_path)
        print(f"LOO validation completed: {progress_dir / 'loo_summary.json'}", flush=True)
    except BaseException as error:
        error_text = f"{type(error).__name__}: {error}"
        if active_fold is not None:
            fold_status = state.setdefault("fold_status", {}).setdefault(
                str(active_fold), {}
            )
            fold_status.update(
                {
                    "status": "failed",
                    "last_update_local": local_now(),
                    "error": error_text,
                }
            )
        state.update(
            {
                "status": "failed",
                "last_update_local": local_now(),
                "error": error_text,
            }
        )
        save_json(state, state_path)
        tracker.update(
            completed_count,
            status="failed",
            details={"error": state["error"], "completed_folds": completed_count},
        )
        raise


def validate_folds(values: list[int]) -> list[int]:
    folds = [int(value) for value in values]
    if not folds:
        raise ValueError("At least one LOO fold is required.")
    if len(set(folds)) != len(folds):
        raise ValueError("--folds contains duplicates.")
    if any(value <= 0 for value in folds):
        raise ValueError("LOO fold identifiers must be positive integers.")
    return folds


def validate_conditional_compositions(folds: list[int], compositions_path: Path) -> None:
    compositions = pd.read_csv(compositions_path)
    compositions["alloy_id"] = compositions["alloy_id"].astype(int)
    ready = set(
        compositions.loc[
            compositions["conditional_ready"]
            .astype(str)
            .str.lower()
            .isin({"1", "true", "yes", "y"}),
            "alloy_id",
        ].tolist()
    )
    missing = sorted(set(folds) - set(compositions["alloy_id"].tolist()))
    invalid = sorted(set(folds) - ready)
    if missing:
        raise ValueError(f"These folds are absent from the composition table: {missing}")
    if invalid:
        raise ValueError(f"These folds do not have usable point compositions: {invalid}")


def load_held_out_images(folds: list[int], manifest_path: Path) -> dict[int, Path]:
    manifest = pd.read_csv(manifest_path)
    manifest["alloy_id"] = manifest["alloy_id"].astype(int)
    root = manifest_path.parent
    result: dict[int, Path] = {}
    for fold in folds:
        rows = manifest.loc[manifest["alloy_id"] == fold]
        if len(rows) != 1:
            raise ValueError(
                f"Expected exactly one manifest image for alloy {fold}, found {len(rows)}."
            )
        image_path = Path(str(rows.iloc[0]["image_path"]))
        if not image_path.is_absolute():
            image_path = (root / image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Held-out image does not exist: {image_path}")
        result[fold] = image_path
    return result


def resolve_training_steps(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[int, int]:
    if args.max_steps is not None and (
        args.vae_max_steps is not None or args.diffusion_max_steps is not None
    ):
        raise ValueError(
            "--max-steps cannot be combined with --vae-max-steps or "
            "--diffusion-max-steps."
        )
    if args.max_steps is not None:
        shared = positive_int(args.max_steps, "max_steps")
        return shared, shared
    vae_steps = positive_int(
        args.vae_max_steps
        if args.vae_max_steps is not None
        else config["train_vae"]["max_steps"],
        "vae_max_steps",
    )
    diffusion_steps = positive_int(
        args.diffusion_max_steps
        if args.diffusion_max_steps is not None
        else config["train_diffusion"]["max_steps"],
        "diffusion_max_steps",
    )
    return vae_steps, diffusion_steps


def ensure_vae_model(
    *,
    project_root: Path,
    config_path: Path,
    output_dir: Path,
    final_dir: Path,
    fold: int,
    max_steps: int,
    auto_resume: bool,
) -> None:
    if vae_model_is_complete(final_dir, fold, max_steps):
        print(f"LOO fold {fold:02d}: VAE already complete; skipping.", flush=True)
        return
    command = [
        sys.executable,
        "train_vae.py",
        "--config",
        str(config_path),
        "--exclude-alloy-id",
        str(fold),
        "--max-steps",
        str(max_steps),
        "--output-dir",
        str(output_dir),
    ]
    checkpoint = latest_checkpoint(output_dir, max_steps) if auto_resume else None
    if checkpoint is not None:
        command += ["--resume-from", str(checkpoint)]
        print(f"LOO fold {fold:02d}: resuming VAE from {checkpoint.name}.", flush=True)
    run_command(command, project_root)
    if not vae_model_is_complete(final_dir, fold, max_steps):
        raise RuntimeError(f"LOO fold {fold:02d} VAE did not produce a valid final model.")


def ensure_diffusion_model(
    *,
    project_root: Path,
    config_path: Path,
    output_dir: Path,
    final_dir: Path,
    vae_final: Path,
    fold: int,
    max_steps: int,
    auto_resume: bool,
) -> None:
    if diffusion_model_is_complete(final_dir, fold, max_steps, vae_final):
        print(f"LOO fold {fold:02d}: diffusion model already complete; skipping.", flush=True)
        return
    command = [
        sys.executable,
        "train_diffusion.py",
        "--config",
        str(config_path),
        "--exclude-alloy-id",
        str(fold),
        "--vae-dir",
        str(vae_final),
        "--max-steps",
        str(max_steps),
        "--output-dir",
        str(output_dir),
    ]
    checkpoint = latest_checkpoint(output_dir, max_steps) if auto_resume else None
    if checkpoint is not None:
        command += ["--resume-from", str(checkpoint)]
        print(
            f"LOO fold {fold:02d}: resuming diffusion from {checkpoint.name}.",
            flush=True,
        )
    run_command(command, project_root)
    if not diffusion_model_is_complete(final_dir, fold, max_steps, vae_final):
        raise RuntimeError(
            f"LOO fold {fold:02d} diffusion training did not produce a valid final model."
        )


def ensure_held_out_generation(
    *,
    project_root: Path,
    checkpoint: Path,
    vae_final: Path,
    compositions_path: Path,
    output_dir: Path,
    fold: int,
    count: int,
    batch_size: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    sampling_cfg: dict[str, Any],
    max_nearest_distance: float,
) -> dict[str, Any]:
    report_path = output_dir / f"alloy_{fold:02d}_generation_report.json"
    existing = load_json_or_default(report_path, {})
    if generation_is_complete(
        existing,
        output_dir=output_dir,
        fold=fold,
        checkpoint=checkpoint,
        count=count,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
    ):
        print(f"LOO fold {fold:02d}: held-out images already exist; skipping.", flush=True)
        return existing

    command = [
        sys.executable,
        "sample.py",
        "--checkpoint",
        str(checkpoint),
        "--alloy-id",
        str(fold),
        "--compositions-csv",
        str(compositions_path),
        "--vae-dir",
        str(vae_final),
        "--output-dir",
        str(output_dir),
        "--count",
        str(count),
        "--batch-size",
        str(batch_size),
        "--height",
        str(sampling_cfg["height"]),
        "--width",
        str(sampling_cfg["width"]),
        "--crop-height",
        str(sampling_cfg["crop_height"]),
        "--crop-width",
        str(sampling_cfg["crop_width"]),
        "--steps",
        str(steps),
        "--guidance-scale",
        str(guidance_scale),
        "--seed",
        str(seed),
        "--max-nearest-distance",
        str(max_nearest_distance),
        "--allow-extrapolation",
    ]
    run_command(command, project_root)
    report = load_json_or_default(report_path, {})
    if not generation_is_complete(
        report,
        output_dir=output_dir,
        fold=fold,
        checkpoint=checkpoint,
        count=count,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
    ):
        raise RuntimeError(
            f"LOO fold {fold:02d} generation did not produce {count} valid images."
        )
    return report


def ensure_evaluation(
    *,
    project_root: Path,
    real_image: Path,
    generation_dir: Path,
    output_path: Path,
    expected_count: int,
    gb_threshold: float,
) -> dict[str, Any]:
    existing = load_json_or_default(output_path, {})
    if evaluation_is_complete(existing, real_image, generation_dir, expected_count):
        print(f"Evaluation already exists at {output_path}; skipping.", flush=True)
        return existing
    command = [
        sys.executable,
        "scripts/evaluate_generated.py",
        "--real",
        str(real_image),
        "--generated-dir",
        str(generation_dir),
        "--output",
        str(output_path),
        "--gb-threshold",
        str(gb_threshold),
    ]
    run_command(command, project_root)
    report = load_json_or_default(output_path, {})
    if not evaluation_is_complete(report, real_image, generation_dir, expected_count):
        raise RuntimeError(f"Evaluation report is incomplete: {output_path}")
    return report


def build_fold_summary(
    *,
    fold: int,
    fold_position: int,
    total_folds: int,
    strict_vae_retraining: bool,
    vae_steps: int,
    diffusion_steps: int,
    samples_per_fold: int,
    sample_steps: int,
    guidance_scale: float,
    seed: int,
    real_image: Path,
    vae_final: Path,
    diffusion_final: Path,
    generation_dir: Path,
    evaluation_path: Path,
    generation_report: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "completed",
        "created_at_local": local_now(),
        "held_out_alloy_id": fold,
        "fold_position": fold_position,
        "total_folds": total_folds,
        "strict_vae_retraining": strict_vae_retraining,
        "scientific_scope": (
            "Strict image-and-condition holdout"
            if strict_vae_retraining
            else "Condition holdout only; base VAE saw the held-out image"
        ),
        "training": {
            "vae_max_steps": vae_steps,
            "diffusion_max_steps": diffusion_steps,
            "vae_final": str(vae_final.resolve()),
            "diffusion_final": str(diffusion_final.resolve()),
        },
        "generation": {
            "samples": samples_per_fold,
            "ddim_steps": sample_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "output_dir": str(generation_dir.resolve()),
            "nearest_training_distance": generation_report.get(
                "nearest_training_distance"
            ),
            "is_extrapolation": generation_report.get("is_extrapolation"),
            "range_violations": generation_report.get("raw_range_violations", {}),
        },
        "evaluation": {
            "real_image": str(real_image.resolve()),
            "report": str(evaluation_path.resolve()),
            "metrics": compact_evaluation(evaluation),
        },
    }


def aggregate_fold_summaries(
    fold_summaries: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    if not fold_summaries:
        raise ValueError("No completed fold summaries are available.")
    metric_rows = [flatten_metrics(item["evaluation"]["metrics"]) for item in fold_summaries]
    metric_names = sorted(metric_rows[0])
    aggregate: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = [float(row[name]) for row in metric_rows]
        aggregate[name] = {
            "mean": float(statistics.fmean(values)),
            "median": float(statistics.median(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }
    return aggregate


def compact_evaluation(report: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for name in RELATIVE_ERROR_METRICS:
        value = report.get(name)
        if not isinstance(value, dict) or "relative_error" not in value:
            raise ValueError(f"Evaluation report is missing {name}.relative_error")
        compact[name] = {
            "real_mean": float(value["real_mean"]),
            "generated_mean": float(value["generated_mean"]),
            "relative_error": float(value["relative_error"]),
        }
    for name in DIRECT_METRICS:
        if name not in report:
            raise ValueError(f"Evaluation report is missing {name}")
        compact[name] = float(report[name])
    return compact


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    result = {
        f"{name}_relative_error": float(metrics[name]["relative_error"])
        for name in RELATIVE_ERROR_METRICS
    }
    result.update({name: float(metrics[name]) for name in DIRECT_METRICS})
    return result


def write_summary_csv(fold_summaries: list[dict[str, Any]], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for summary in fold_summaries:
        row: dict[str, Any] = {
            "held_out_alloy_id": summary["held_out_alloy_id"],
            "strict_vae_retraining": summary["strict_vae_retraining"],
            "nearest_training_distance": summary["generation"][
                "nearest_training_distance"
            ],
            "is_extrapolation": summary["generation"]["is_extrapolation"],
        }
        row.update(flatten_metrics(summary["evaluation"]["metrics"]))
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def vae_model_is_complete(final_dir: Path, fold: int, max_steps: int) -> bool:
    metadata = load_json_or_default(final_dir / "training_metadata.json", {})
    if not isinstance(metadata, dict):
        return False
    return (
        (final_dir / "config.json").is_file()
        and (final_dir / "diffusion_pytorch_model.safetensors").is_file()
        and safe_int(metadata.get("global_step")) >= max_steps
        and safe_int(metadata.get("exclude_alloy_id")) == fold
        and fold not in {
            safe_int(value) for value in metadata.get("training_alloy_ids", [])
        }
    )


def base_vae_is_available(final_dir: Path) -> bool:
    return (
        (final_dir / "config.json").is_file()
        and (final_dir / "diffusion_pytorch_model.safetensors").is_file()
    )


def diffusion_model_is_complete(
    final_dir: Path, fold: int, max_steps: int, vae_final: Path
) -> bool:
    metadata = load_json_or_default(final_dir / "run_metadata.json", {})
    if not isinstance(metadata, dict):
        return False
    recorded_vae = Path(str(metadata.get("vae_dir", "")))
    return (
        (final_dir / "unet" / "config.json").is_file()
        and (final_dir / "condition_encoder" / "config.json").is_file()
        and (final_dir / "scheduler" / "scheduler_config.json").is_file()
        and (final_dir / "composition_conditioner.json").is_file()
        and safe_int(metadata.get("global_step")) >= max_steps
        and safe_int(metadata.get("exclude_alloy_id")) == fold
        and fold not in {
            safe_int(value) for value in metadata.get("training_alloy_ids", [])
        }
        and recorded_vae.resolve() == vae_final.resolve()
    )


def latest_checkpoint(output_dir: Path, max_steps: int) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    if not output_dir.is_dir():
        return None
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        state = load_json_or_default(path / "training_state.json", {})
        try:
            step = int(state["global_step"])
        except (KeyError, TypeError, ValueError):
            continue
        if step <= max_steps:
            candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda value: value[0])[1]


def generation_is_complete(
    report: dict[str, Any],
    *,
    output_dir: Path,
    fold: int,
    checkpoint: Path,
    count: int,
    steps: int,
    guidance_scale: float,
    seed: int,
) -> bool:
    images = sorted(
        path
        for path in output_dir.glob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not isinstance(report, dict):
        return False
    try:
        recorded_checkpoint = Path(str(report["checkpoint"]))
        recorded_files = {
            Path(value).resolve() for value in report["generated_files"]
        }
        return (
            report.get("condition_label") == f"alloy_{fold:02d}"
            and int(report.get("output_count", -1)) == count
            and int(report.get("steps", -1)) == steps
            and int(report.get("seed", -1)) == seed
            and abs(float(report.get("guidance_scale", -1.0)) - guidance_scale)
            < 1.0e-9
            and recorded_checkpoint.resolve() == checkpoint.resolve()
            and len(images) == count
            and recorded_files == {path.resolve() for path in images}
        )
    except (KeyError, TypeError, ValueError):
        return False


def evaluation_is_complete(
    report: dict[str, Any],
    real_image: Path,
    generation_dir: Path,
    expected_count: int,
) -> bool:
    try:
        real_files = {Path(value).resolve() for value in report["real_files"]}
        generated_files = [Path(value).resolve() for value in report["generated_files"]]
        compact_evaluation(report)
    except (KeyError, TypeError, ValueError):
        return False
    return (
        int(report.get("real_count", -1)) == 1
        and int(report.get("generated_count", -1)) == expected_count
        and real_image.resolve() in real_files
        and len(generated_files) == expected_count
        and all(
            path.parent == generation_dir.resolve() and path.is_file()
            for path in generated_files
        )
    )


def load_valid_fold_summary(
    path: Path,
    *,
    fold: int,
    vae_steps: int,
    diffusion_steps: int,
    samples_per_fold: int,
    sample_steps: int,
    guidance_scale: float,
    strict_vae_retraining: bool,
) -> dict[str, Any] | None:
    value = load_json_or_default(path, {})
    try:
        matches = (
            value["status"] == "completed"
            and int(value["held_out_alloy_id"]) == fold
            and bool(value["strict_vae_retraining"]) == strict_vae_retraining
            and int(value["training"]["vae_max_steps"]) == vae_steps
            and int(value["training"]["diffusion_max_steps"]) == diffusion_steps
            and int(value["generation"]["samples"]) == samples_per_fold
            and int(value["generation"]["ddim_steps"]) == sample_steps
            and abs(float(value["generation"]["guidance_scale"]) - guidance_scale)
            < 1.0e-9
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not matches:
        return None
    try:
        evaluation_report = Path(value["evaluation"]["report"])
        generation_dir = Path(value["generation"]["output_dir"])
    except (KeyError, TypeError):
        return None
    if not evaluation_report.is_file() or not generation_dir.is_dir():
        return None
    generated_count = sum(
        path.suffix.lower() in IMAGE_EXTENSIONS
        for path in generation_dir.iterdir()
        if path.is_file()
    )
    return value if generated_count == samples_per_fold else None


def generation_run_name(
    *, seed: int, count: int, steps: int, guidance_scale: float
) -> str:
    scale = f"{guidance_scale:g}".replace("-", "m").replace(".", "p")
    return f"seed{seed}_n{count}_steps{steps}_cfg{scale}"


def run_command(command: list[str], project_root: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(project_root), existing_pythonpath) if value
    )
    print("\nRUNNING: " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        check=True,
    )


def update_fold_state(
    state: dict[str, Any],
    state_path: Path,
    fold: int,
    *,
    status: str,
    stage: str,
    **details: Any,
) -> None:
    fold_status = state.setdefault("fold_status", {})
    fold_status[str(fold)] = {
        "status": status,
        "stage": stage,
        "last_update_local": local_now(),
        **details,
    }
    state.update(
        {
            "status": "running" if status != "failed" else "failed",
            "current_fold": fold,
            "current_stage": stage,
            "last_update_local": local_now(),
        }
    )
    save_json(state, state_path)


def resolve_from_root(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_json_or_default(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def read_previous_elapsed(path: Path) -> float:
    value = load_json_or_default(path, {})
    if not isinstance(value, dict):
        return 0.0
    try:
        return max(float(value.get("elapsed_seconds", 0.0)), 0.0)
    except (TypeError, ValueError):
        return 0.0


def positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def local_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
