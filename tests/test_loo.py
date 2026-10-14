from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.run_loo import (
    aggregate_fold_summaries,
    ensure_vae_model,
    generation_is_complete,
    generation_run_name,
    latest_checkpoint,
    vae_model_is_complete,
)


def evaluation_metrics(scale: float) -> dict:
    names = (
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
    result = {
        name: {
            "real_mean": 1.0,
            "generated_mean": 1.0 + scale,
            "relative_error": scale,
        }
        for name in names
    }
    result.update(
        {
            "two_point_rmse": scale / 10.0,
            "radial_power_rmse": scale / 20.0,
            "hue_jensen_shannon": scale / 2.0,
        }
    )
    return result


class LooWorkflowTests(unittest.TestCase):
    def test_generation_run_name_is_filesystem_safe(self) -> None:
        self.assertEqual(
            generation_run_name(seed=123, count=8, steps=50, guidance_scale=1.8),
            "seed123_n8_steps50_cfg1p8",
        )

    def test_latest_checkpoint_uses_highest_valid_step(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (2500, 5000, 7500):
                checkpoint = root / f"checkpoint-{step:07d}"
                checkpoint.mkdir()
                (checkpoint / "training_state.json").write_text(
                    json.dumps({"global_step": step}),
                    encoding="utf-8",
                )
            invalid = root / "checkpoint-0099999"
            invalid.mkdir()
            self.assertEqual(
                latest_checkpoint(root, 6000),
                root / "checkpoint-0005000",
            )

    def test_vae_completion_rejects_held_out_training_id(self) -> None:
        with TemporaryDirectory() as directory:
            final_dir = Path(directory)
            (final_dir / "config.json").write_text("{}", encoding="utf-8")
            (final_dir / "diffusion_pytorch_model.safetensors").touch()
            metadata = {
                "global_step": 30000,
                "exclude_alloy_id": 3,
                "training_alloy_ids": [1, 2, 4],
            }
            (final_dir / "training_metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            self.assertTrue(vae_model_is_complete(final_dir, 3, 30000))
            metadata["training_alloy_ids"].append(3)
            (final_dir / "training_metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            self.assertFalse(vae_model_is_complete(final_dir, 3, 30000))

    def test_aggregate_reports_mean_and_extremes(self) -> None:
        summaries = [
            {"evaluation": {"metrics": evaluation_metrics(0.1)}},
            {"evaluation": {"metrics": evaluation_metrics(0.3)}},
        ]
        aggregate = aggregate_fold_summaries(summaries)
        self.assertAlmostEqual(
            aggregate["gb_fraction_relative_error"]["mean"],
            0.2,
        )
        self.assertAlmostEqual(
            aggregate["hue_jensen_shannon"]["max"],
            0.15,
        )

    def test_generation_completion_checks_exact_file_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model" / "final"
            checkpoint.mkdir(parents=True)
            output = root / "generated"
            output.mkdir()
            image = output / "alloy_03_seed126_000.png"
            image.touch()
            report = {
                "condition_label": "alloy_03",
                "checkpoint": str(checkpoint),
                "output_count": 1,
                "steps": 5,
                "seed": 126,
                "guidance_scale": 1.8,
                "generated_files": [str(image.resolve())],
            }
            self.assertTrue(
                generation_is_complete(
                    report,
                    output_dir=output,
                    fold=3,
                    checkpoint=checkpoint,
                    count=1,
                    steps=5,
                    guidance_scale=1.8,
                    seed=126,
                )
            )
            report["generated_files"] = [str(output / "wrong.png")]
            self.assertFalse(
                generation_is_complete(
                    report,
                    output_dir=output,
                    fold=3,
                    checkpoint=checkpoint,
                    count=1,
                    steps=5,
                    guidance_scale=1.8,
                    seed=126,
                )
            )

    def test_invalid_vae_metadata_is_incomplete(self) -> None:
        with TemporaryDirectory() as directory:
            final_dir = Path(directory)
            (final_dir / "config.json").write_text("{}", encoding="utf-8")
            (final_dir / "diffusion_pytorch_model.safetensors").touch()
            (final_dir / "training_metadata.json").write_text(
                '{"global_step": null}',
                encoding="utf-8",
            )
            self.assertFalse(vae_model_is_complete(final_dir, 3, 30000))

    def test_vae_stage_uses_explicit_output_and_latest_checkpoint(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "fold_03" / "models_strict" / "vae"
            checkpoint = output / "checkpoint-0002500"
            checkpoint.mkdir(parents=True)
            (checkpoint / "training_state.json").write_text(
                '{"global_step": 2500}',
                encoding="utf-8",
            )
            final_dir = output / "final"

            def complete_fake_training(command, project_root):
                self.assertEqual(project_root, root)
                final_dir.mkdir(parents=True)
                (final_dir / "config.json").write_text("{}", encoding="utf-8")
                (final_dir / "diffusion_pytorch_model.safetensors").touch()
                (final_dir / "training_metadata.json").write_text(
                    json.dumps(
                        {
                            "global_step": 30000,
                            "exclude_alloy_id": 3,
                            "training_alloy_ids": [1, 2, 4],
                        }
                    ),
                    encoding="utf-8",
                )

            with patch(
                "scripts.run_loo.run_command",
                side_effect=complete_fake_training,
            ) as mocked:
                ensure_vae_model(
                    project_root=root,
                    config_path=root / "configs" / "base.yaml",
                    output_dir=output,
                    final_dir=final_dir,
                    fold=3,
                    max_steps=30000,
                    auto_resume=True,
                )

            command = mocked.call_args.args[0]
            self.assertEqual(command[command.index("--output-dir") + 1], str(output))
            self.assertEqual(
                command[command.index("--resume-from") + 1],
                str(checkpoint),
            )


if __name__ == "__main__":
    unittest.main()
