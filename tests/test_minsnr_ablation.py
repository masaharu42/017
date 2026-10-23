from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
from safetensors.torch import load_file, save_file
import torch
import yaml
from diffusers import DDPMScheduler

from ebsd_diffusion.composition import CompositionConditioner
from ebsd_diffusion.models import ContinuousConditionEmbedding, build_unet, build_vae
from ebsd_diffusion.training_control import load_frozen_conditioner, resolve_initialization
from scripts.run_minsnr_ablation import (
    ARMS,
    assert_matched_configs,
    completed_metadata,
    evaluate,
    load_prepared,
    make_arm_configs,
    prepare,
    reference_boxes,
    train,
    verify_pair,
)


@pytest.fixture
def tiny_protocol(tmp_path):
    """Synthetic fixture for software checks, never a scientific training result."""
    source = tmp_path / "source"
    source.mkdir()
    model_cfg = {"block_out_channels": [32], "layers_per_block": 1,
                 "condition_hidden_dim": 8, "condition_dropout": .15, "ema_decay": .9999,
                 "num_train_timesteps": 10, "beta_schedule": "squaredcos_cap_v2", "prediction_type": "epsilon"}
    vae = build_vae({"block_out_channels": [32], "layers_per_block": 1, "latent_channels": 4}, 8)
    vae.save_pretrained(source / "vae")
    unet = build_unet(model_cfg, 4, 8)
    unet.save_pretrained(source / "unet")
    save_file(unet.state_dict(), str(source / "raw.safetensors"))
    encoder = ContinuousConditionEmbedding(1, 8, 128)
    encoder.save_pretrained(source / "encoder")
    save_file(encoder.state_dict(), str(source / "encoder.safetensors"))
    scheduler = DDPMScheduler(num_train_timesteps=10, beta_schedule="squaredcos_cap_v2", clip_sample=False)
    scheduler.save_pretrained(source / "scheduler")
    compositions = pd.DataFrame({"alloy_id": [1, 2, 3], "conditional_ready": [True] * 3, "Ni": [1., 2., 3.]})
    compositions.to_csv(source / "compositions.csv", index=False)
    conditioner = CompositionConditioner.fit(compositions, ["Ni"], [1, 2], closure_drop_feature=None)
    conditioner.save(source / "conditioner.json")
    (source / "state.json").write_text(json.dumps({"global_step": 50000, "latent_scaling": 1.125}), "utf-8")
    rows = []
    for alloy_id in (1, 2, 3):
        path = source / f"real_{alloy_id}.png"
        pixels = np.full((12, 12, 3), [100 + alloy_id * 10, 190, 90], dtype=np.uint8)
        pixels[:, 4:5] = 0
        Image.fromarray(pixels).save(path)
        rows.append({"alloy_id": alloy_id, "image_path": str(path), "use_for_vae": True, "use_for_diffusion": True})
    pd.DataFrame(rows).to_csv(source / "manifest.csv", index=False)
    sampling = {
        "raw_unet": str(source / "raw.safetensors"), "unet_config": str(source / "unet"),
        "condition_encoder_dir": str(source / "encoder"), "condition_encoder_weights": str(source / "encoder.safetensors"),
        "conditioner": str(source / "conditioner.json"), "scheduler_dir": str(source / "scheduler"),
        "run_metadata": str(source / "state.json"), "vae_dir": str(source / "vae"),
    }
    (source / "sampling.yaml").write_text(yaml.safe_dump(sampling), "utf-8")
    hparams = {
        "seed": 123,
        "data.manifest": str(source / "manifest.csv"), "data.compositions": str(source / "compositions.csv"),
        "data.patch_size": 8, "data.patches_per_image": 2, "data.num_workers": 0,
        "data.crop_margins": "[0, 0, 0, 0]", "data.horizontal_flip": False,
        "data.vertical_flip": False, "data.rotate_180": False,
        "composition.candidate_features": '["Ni"]', "composition.closure_drop_feature": None,
        "composition.min_nonzero_samples": 2, "composition.min_standard_deviation": 1e-8,
        "composition.vicinal_jitter_std": .05,
        "train_diffusion.batch_size": 2, "train_diffusion.gradient_accumulation_steps": 1,
        "train_diffusion.mixed_precision": "no", "train_diffusion.weight_decay": 1e-4,
        "train_diffusion.gradient_checkpointing": False,
        **{f"diffusion.{key}": json.dumps(value) if isinstance(value, list) else value for key, value in model_cfg.items()},
    }
    (source / "hparams.yaml").write_text(yaml.safe_dump(hparams), "utf-8")
    protocol = {
        "name": "software_smoke_only", "source_sampling_config": str(source / "sampling.yaml"),
        "historical_hparams": str(source / "hparams.yaml"), "output_root": str(tmp_path / "run"),
        "exclude_alloy_id": 3, "expected_source_step": 50000, "require_cuda": False,
        "training": {"seed": 123, "additional_steps": 2, "learning_rate": 1e-3, "warmup_steps": 0,
                     "save_every": 2, "log_every": 1, "deterministic": True},
        "evaluation": {"alloy_ids": [1, 3], "seeds": [101, 102], "steps": 2, "guidance_scale": 1.8,
                       "height": 8, "width": 8, "gb_threshold": .18, "sampling_dtype": "float32"},
    }
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol), "utf-8")
    return path, protocol


def test_hparams_and_only_one_training_variable(tiny_protocol):
    path, protocol = tiny_protocol
    from scripts.run_minsnr_ablation import PROJECT_ROOT
    _, arms = make_arm_configs(protocol, PROJECT_ROOT)
    assert_matched_configs(arms)
    assert arms["A_mse"]["data"]["crop_margins"] == [0, 0, 0, 0]
    assert arms["A_mse"]["train_diffusion"]["max_steps"] == 2
    assert arms["B_minsnr5"]["train_diffusion"]["snr_gamma"] == 5
    bad = deepcopy(arms)
    bad["B_minsnr5"]["diffusion"]["condition_dropout"] = .5
    with pytest.raises(ValueError, match="differ beyond"):
        assert_matched_configs(bad)


def test_preflight_freezes_conditioner_and_refuses_overwrite(tiny_protocol):
    path, _ = tiny_protocol
    manifest = prepare(path)
    assert manifest["status"] == "prepared_not_trained"
    assert manifest["preflight"]["training_alloy_ids"] == [1, 2]
    assert manifest["preflight"]["initialization"]["latent_scaling"] == 1.125
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare(path)
    load_prepared(path)


def test_detects_source_tampering(tiny_protocol):
    path, protocol = tiny_protocol
    prepare(path)
    state_path = Path(protocol["historical_hparams"])
    state_path.write_text(state_path.read_text("utf-8") + "\n# changed\n", "utf-8")
    with pytest.raises(ValueError, match="changed after prepare"):
        load_prepared(path)


def test_conditioner_cannot_silently_refit_to_different_alloys(tiny_protocol):
    path, protocol = tiny_protocol
    from scripts.run_minsnr_ablation import PROJECT_ROOT
    _, arms = make_arm_configs(protocol, PROJECT_ROOT)
    config = arms["A_mse"]
    config["_project_root"] = str(PROJECT_ROOT)
    paths, _ = resolve_initialization(config)
    compositions = pd.read_csv(config["data"]["compositions"])
    with pytest.raises(ValueError, match="alloy IDs differ"):
        load_frozen_conditioner(paths, {1, 3}, compositions)
    compositions.loc[0, "Ni"] += 1
    with pytest.raises(ValueError, match="Composition values differ"):
        load_frozen_conditioner(paths, {1, 2}, compositions)


def test_reference_boxes_honor_margins_and_remove_duplicates():
    assert reference_boxes(256, 256, 256, [0, 0, 0, 0]) == [(0, 0, 256, 256)]
    boxes = reference_boxes(750, 450, 256, [1, 2, 3, 4])
    assert len(boxes) == 5
    assert boxes[0] == (4, 1, 260, 257)
    assert boxes[-1] == (248, 96, 504, 352)


def test_two_arm_training_and_free_generation_smoke(tiny_protocol, monkeypatch):
    path, protocol = tiny_protocol
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        manifest = prepare(path)
        result = train(path)
        assert result == {"status": "passed", "audited_input_batches": 2,
                          "scope": "first five batches and every logged batch; not an exhaustive all-step audit"}
        output = Path(protocol["output_root"])
        raw_states = {}
        for arm in ARMS:
            metadata = completed_metadata(output, arm, manifest)
            assert metadata["global_step"] == 2
            assert metadata["source_global_step"] == 50000
            assert metadata["effective_total_step"] == 50002
            assert metadata["latent_scaling"] == 1.125
            assert (output / arm / "checkpoint-0000002" / "training_state.json").is_file()
            raw_states[arm] = load_file(str(output / arm / "final" / "unet_raw" / "diffusion_pytorch_model.safetensors"))
            checkpoint_state = load_file(str(output / arm / "checkpoint-0000002" / "model.safetensors"))
            assert all(torch.equal(value, checkpoint_state[key]) for key, value in raw_states[arm].items())
        assert any(not torch.equal(value, raw_states["B_minsnr5"][key]) for key, value in raw_states["A_mse"].items())
        report = evaluate(path)
        assert report["mode"] == "pure_noise_free_generation"
        assert report["weight_type"] == "raw_only"
        assert report["actual_dtype"] == "torch.float32"
        entries = json.loads((output / "evaluation" / "per_image_metrics.json").read_text("utf-8"))
        assert len([item for item in entries if "seed" in item]) == 12
        assert (output / "evaluation" / "alloy_03" / "comparison_grid.png").is_file()
        with pytest.raises(FileExistsError, match="overwrite evaluation"):
            evaluate(path)
        # Corrupted audit data must prevent a scientific A/B comparison.
        audit = output / "B_minsnr5" / "input_fingerprints.csv"
        audit.write_text(audit.read_text("utf-8").replace(",", ",changed", 1), "utf-8")
        with pytest.raises((ValueError, KeyError)):
            verify_pair(output, manifest)
    finally:
        torch.set_num_threads(previous_threads)
