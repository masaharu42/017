from __future__ import annotations

import argparse
import json
import os
import platform
import time
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import accelerate
import diffusers
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ebsd_diffusion.composition import CompositionConditioner, load_compositions
from ebsd_diffusion.config import flatten_config_for_logging, load_config, resolve_project_path
from ebsd_diffusion.data import EbsdPatchDataset, build_condition_mapping
from ebsd_diffusion.descriptor_conditioning import DescriptorConditionEmbedding
from ebsd_diffusion.local_descriptors import LocalDescriptorStore
from ebsd_diffusion.losses import diffusion_mse_loss, validate_snr_gamma
from ebsd_diffusion.boundary_loss import boundary_dice_loss
from ebsd_diffusion.training_diagnostics import TrainingDiagnostics
from ebsd_diffusion.training_control import (
    file_sha256,
    load_frozen_conditioner,
    load_initialized_models,
    resolve_initialization,
    tensor_fingerprint,
)
from ebsd_diffusion.models import (
    ContinuousConditionEmbedding,
    ExponentialMovingAverage,
    build_unet,
    unet_condition_dimension,
)
from ebsd_diffusion.utils import (
    RunProgressTracker,
    append_csv_row,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the composition-conditioned latent diffusion model.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--vae-dir", type=Path)
    parser.add_argument("--exclude-alloy-id", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--lambda-gb", type=float)
    parser.add_argument("--lambda-edge", type=float)
    parser.add_argument("--structure-max-t", type=int)
    parser.add_argument("--boundary-calibrate", action="store_true",
                        help="Measure boundary/MSE gradients without optimizer updates.")
    parser.add_argument("--software-test-cpu", action="store_true",
                        help="CPU only for synthetic software tests.")
    parser.add_argument("--benchmark-steps", type=int, default=0,
                        help="Disposable optimizer updates; writes timing only, never exports weights.")
    parser.add_argument("--benchmark-warmup", type=int, default=3)
    return parser.parse_args()


def soft_structure_descriptors(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable proxies for black GB fraction and RGB edge density."""
    rgb = (images + 1.0) * 0.5
    max_rgb = rgb.amax(dim=1, keepdim=True)
    gb = torch.sigmoid((0.22 - max_rgb) / 0.035).mean(dim=(1, 2, 3))
    gray = rgb.mean(dim=1, keepdim=True)
    kernel_x = gray.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(2, 3)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    edge = torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-8).mean(dim=(1, 2, 3))
    return gb, edge


def main() -> None:
    session_started_monotonic = time.perf_counter()
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    composition_cfg = config["composition"]
    model_cfg = config["diffusion"]
    train_cfg = config["train_diffusion"]
    if args.software_test_cpu and int(data_cfg["patch_size"]) != 8:
        raise ValueError("--software-test-cpu is restricted to 8x8 synthetic fixtures.")
    benchmark_steps = int(getattr(args, "benchmark_steps", 0))
    benchmark_warmup = int(getattr(args, "benchmark_warmup", 3))
    if benchmark_steps < 0 or (benchmark_steps and not 0 < benchmark_warmup < benchmark_steps):
        raise ValueError("Benchmark requires 0 < warmup < steps.")
    if benchmark_steps and args.resume_from:
        raise ValueError("Benchmark must start fresh, not resume.")
    controlled = bool(train_cfg.get("controlled_experiment", False))
    boundary_cfg = train_cfg.get("boundary_dice")
    if args.boundary_calibrate and not boundary_cfg:
        raise ValueError("Boundary calibration requires boundary_dice configuration.")
    if boundary_cfg:
        if not controlled or args.resume_from or benchmark_steps or train_cfg.get("local_descriptors"):
            raise ValueError("Boundary pilot requires fresh composition-only controlled training.")
        if train_cfg.get("snr_gamma") is not None or train_cfg.get("lambda_gb", 0) or train_cfg.get("lambda_edge", 0):
            raise ValueError("Boundary pilot uses ordinary MSE without other structure losses.")
        if boundary_cfg.get("max_t") != 300 or boundary_cfg.get("max_samples") != 8:
            raise ValueError("Boundary pilot fixes max_t=300 and max_samples=8.")
        if not args.boundary_calibrate:
            calibration_path = resolve_project_path(config, boundary_cfg["calibration_file"])
            calibration = json.loads(calibration_path.read_text("utf-8"))
            if calibration.get("optimizer_updates") != 0 or calibration.get("status") != "passed":
                raise ValueError("A passing zero-update boundary calibration is required.")
            if calibration.get("config_sha256") != file_sha256(args.config):
                raise ValueError("Calibration belongs to a different training configuration.")
            if calibration.get("gpu_name") != (torch.cuda.get_device_name() if torch.cuda.is_available() else None):
                raise ValueError("Calibration and training must use the same GPU.")
            boundary_weight = float(calibration["lambda"])
            if not math.isfinite(boundary_weight) or boundary_weight <= 0:
                raise ValueError("Invalid calibrated boundary weight.")
    calibration_ratios = []
    descriptor_cfg = train_cfg.get("local_descriptors")
    descriptor_store = None
    reference_fingerprints = None
    compared_steps = []
    fingerprint_mismatches = []
    input_policy = "strict_tensor_identity"
    cross_replay = None
    if descriptor_cfg:
        if not controlled or args.resume_from or benchmark_steps:
            raise ValueError("Local-five pilot requires a fresh controlled run")
        if train_cfg.get("snr_gamma") is not None:
            raise ValueError("Local-five pilot uses ordinary MSE, not Min-SNR")
        descriptor_store = LocalDescriptorStore(resolve_project_path(config, descriptor_cfg["metadata"]))
        input_policy = descriptor_cfg.get("input_comparison_policy", "strict_tensor_identity")
        if input_policy not in {"strict_tensor_identity", "cross_device_screening"}:
            raise ValueError("Unknown local-five input comparison policy")
        if input_policy == "cross_device_screening":
            replay_path = resolve_project_path(config, descriptor_cfg["zero_update_replay"])
            if file_sha256(replay_path) != descriptor_cfg["zero_update_replay_sha256"]:
                raise ValueError("Cross-device replay report changed")
            cross_replay = json.loads(replay_path.read_text("utf-8"))
            if (cross_replay.get("optimizer_updates") != 0
                    or not cross_replay.get("new_code_matches_archived_replay_here")
                    or not cross_replay.get("components_equal_here")
                    or not all(cross_replay["components_equal_here"].values())):
                raise ValueError("Cross-device screening requires a passing zero-update A/B replay")
            actual_gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else None
            if cross_replay["results"]["current_local5"]["gpu"] != actual_gpu:
                raise ValueError("GPU differs from the zero-update replay")
        with resolve_project_path(config, descriptor_cfg["reference_fingerprints"]).open(newline="") as handle:
            import csv
            rows = list(csv.DictReader(handle))
        reference_fingerprints = {int(row["additional_step"]): row["sha256"] for row in rows}
        expected_steps = set(range(1, 6)) & set(range(1, int(train_cfg["max_steps"]) + 1))
        expected_steps |= set(range(int(train_cfg["log_every"]), int(train_cfg["max_steps"]) + 1, int(train_cfg["log_every"])))
        if set(reference_fingerprints) != expected_steps or len(rows) != len(expected_steps):
            raise ValueError("Reference fingerprints do not cover the exact audit steps")
    if controlled and args.resume_from:
        raise ValueError("This matched pilot uses fresh weights-only starts, not --resume-from.")
    if bool(train_cfg.get("deterministic", False)):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    seed_everything(int(config["seed"]))
    if "allow_tf32" in train_cfg:
        torch.backends.cuda.matmul.allow_tf32 = bool(train_cfg["allow_tf32"])
        torch.backends.cudnn.allow_tf32 = bool(train_cfg["allow_tf32"])
    snr_gamma = train_cfg.get("snr_gamma")
    validate_snr_gamma(snr_gamma, str(model_cfg["prediction_type"]))
    if args.max_steps is not None:
        train_cfg["max_steps"] = args.max_steps
    if args.lambda_gb is not None:
        train_cfg["lambda_gb"] = args.lambda_gb
    if args.lambda_edge is not None:
        train_cfg["lambda_edge"] = args.lambda_edge
    if args.structure_max_t is not None:
        train_cfg["structure_max_t"] = args.structure_max_t
    lambda_gb = float(train_cfg.get("lambda_gb", 0.0))
    lambda_edge = float(train_cfg.get("lambda_edge", 0.0))
    structure_max_t = int(train_cfg.get("structure_max_t", 0))
    if controlled and (lambda_gb != 0.0 or lambda_edge != 0.0):
        raise ValueError("The matched Min-SNR pilot requires both structure loss weights to be zero.")
    loop_steps = (int(boundary_cfg.get("calibration_batches", 8)) if args.boundary_calibrate
                  else benchmark_steps or int(train_cfg["max_steps"]))
    initialization, initialization_report = resolve_initialization(config)

    output_dir = args.output_dir.resolve() if args.output_dir else resolve_project_path(config, train_cfg["output_dir"])
    if controlled and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty pilot directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        cpu=args.software_test_cpu,
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        mixed_precision=str(train_cfg["mixed_precision"]),
        log_with="tensorboard",
        project_dir=output_dir / "logs",
    )
    if controlled and accelerator.num_processes != 1:
        raise ValueError("The matched pilot currently requires one process/GPU per arm.")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    benchmark_memory = {}
    if benchmark_steps and accelerator.device.type == "cuda":
        free, total = torch.cuda.mem_get_info(accelerator.device)
        benchmark_memory = {"initial_free_bytes": free, "total_bytes": total}
        if train_cfg["mixed_precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Selected GPU/runtime does not support BF16.")
    if accelerator.is_main_process:
        accelerator.init_trackers(
            "ebsd_latent_diffusion", config=flatten_config_for_logging(config)
        )

    manifest_path = resolve_project_path(config, data_cfg["manifest"])
    manifest = pd.read_csv(manifest_path)
    manifest["alloy_id"] = manifest["alloy_id"].astype(int)
    use_diffusion = manifest["use_for_diffusion"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
    train_ids = set(manifest.loc[use_diffusion, "alloy_id"].tolist())
    if args.exclude_alloy_id is not None:
        train_ids.discard(int(args.exclude_alloy_id))

    compositions = load_compositions(resolve_project_path(config, data_cfg["compositions"]))
    if initialization:
        conditioner = load_frozen_conditioner(initialization, train_ids, compositions)
    else:
        conditioner = CompositionConditioner.fit(
            compositions,
            candidate_features=composition_cfg["candidate_features"],
            alloy_ids=train_ids,
            closure_drop_feature=composition_cfg.get("closure_drop_feature"),
            min_nonzero_samples=int(composition_cfg["min_nonzero_samples"]),
            min_standard_deviation=float(composition_cfg["min_standard_deviation"]),
        )
    condition_mapping = build_condition_mapping(compositions, conditioner, train_ids)

    dataset = EbsdPatchDataset(
        manifest_path=manifest_path,
        patch_size=int(data_cfg["patch_size"]),
        patches_per_image=int(data_cfg["patches_per_image"]),
        target="diffusion",
        conditions=condition_mapping,
        exclude_alloy_id=args.exclude_alloy_id,
        crop_margins=tuple(data_cfg["crop_margins"]),
        horizontal_flip=bool(data_cfg["horizontal_flip"]),
        vertical_flip=bool(data_cfg["vertical_flip"]),
        rotate_180=bool(data_cfg["rotate_180"]),
        descriptor_store=descriptor_store,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(data_cfg["num_workers"]) > 0,
    )
    if not len(loader):
        raise ValueError("Dataset cannot supply a complete batch; reduce batch_size.")

    vae_dir = args.vae_dir or resolve_project_path(config, train_cfg["vae_dir"])
    vae = AutoencoderKL.from_pretrained(vae_dir)
    vae.requires_grad_(False).eval().to(accelerator.device)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_size = int(data_cfg["patch_size"]) // vae_scale_factor

    if initialization:
        unet, condition_encoder, noise_scheduler = load_initialized_models(
            initialization, model_cfg, conditioner.dimension
        )
        if int(unet.config.in_channels) != int(vae.config.latent_channels):
            raise ValueError("Source U-Net and VAE latent channels differ.")
        if int(unet.config.sample_size) != latent_size:
            raise ValueError("Source U-Net sample_size differs from the configured training patch.")
    else:
        unet = build_unet(model_cfg, int(vae.config.latent_channels), latent_size)
        condition_encoder = ContinuousConditionEmbedding(
            input_dim=conditioner.dimension,
            hidden_dim=int(model_cfg["condition_hidden_dim"]),
            output_dim=unet_condition_dimension(unet),
        )
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=int(model_cfg["num_train_timesteps"]),
            beta_schedule=str(model_cfg["beta_schedule"]),
            prediction_type=str(model_cfg["prediction_type"]),
            clip_sample=False,
        )
    if bool(train_cfg.get("gradient_checkpointing", False)):
        unet.enable_gradient_checkpointing()

    if descriptor_cfg:
        if sorted(dataset.alloy_ids) != descriptor_store.metadata["training_alloy_ids"]:
            raise ValueError("Descriptor normalizer/maps must use exactly the training fold")
        condition_encoder = DescriptorConditionEmbedding(condition_encoder)

    optimizer_options = {}
    if train_cfg.get("fused_adamw", False):
        if accelerator.device.type != "cuda":
            raise ValueError("fused_adamw requires CUDA for this training protocol.")
        optimizer_options["fused"] = True
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(condition_encoder.parameters()),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        **optimizer_options,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(train_cfg["warmup_steps"]),
        num_training_steps=int(train_cfg["max_steps"]),
    )
    unet, condition_encoder, optimizer, loader, lr_scheduler = accelerator.prepare(
        unet, condition_encoder, optimizer, loader, lr_scheduler
    )
    ema = ExponentialMovingAverage(
        accelerator.unwrap_model(unet).parameters(), decay=float(model_cfg["ema_decay"])
    )
    accelerator.register_for_checkpointing(ema)

    latent_scaling = (
        initialization_report["latent_scaling"] if initialization_report
        else estimate_latent_scaling(vae, loader, accelerator.device, max_batches=16)
    )
    global_step = 0
    previous_elapsed_seconds = 0.0
    if args.resume_from:
        accelerator.load_state(str(args.resume_from))
        state_path = args.resume_from / "training_state.json"
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            global_step = int(state["global_step"])
            latent_scaling = float(state["latent_scaling"])
            previous_elapsed_seconds = float(state.get("elapsed_seconds", 0.0))

    if controlled and int(train_cfg["gradient_accumulation_steps"]) != 1:
        raise ValueError("The matched pilot records one batch per optimizer step; accumulation must be 1.")
    if accelerator.is_main_process and controlled:
        save_json({
            "config": config,
            "initialization": initialization_report,
            "snr_gamma": snr_gamma,
            "step_unit": "additional_optimizer_steps",
            "optimizer_state_loaded": False,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "accelerate": accelerate.__version__,
            "diffusers": diffusers.__version__,
            "cuda": torch.version.cuda,
            "device": str(accelerator.device),
            "gpu_name": torch.cuda.get_device_name(accelerator.device) if accelerator.device.type == "cuda" else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "world_size": accelerator.num_processes,
        }, output_dir / "initialization_metadata.json")

    tracker = None
    if accelerator.is_local_main_process:
        tracker = RunProgressTracker(
            output_dir=output_dir,
            run_name="EBSD composition-conditioned diffusion training",
            total_units=int(train_cfg["max_steps"]),
            unit_name="optimizer_step",
            initial_units=global_step,
            previous_elapsed_seconds=previous_elapsed_seconds,
            session_started_monotonic=session_started_monotonic,
        )
        tracker.set_stage(
            "diffusion_training",
            global_step,
            details={
                "exclude_alloy_id": args.exclude_alloy_id,
                "latent_scaling": latent_scaling,
            },
        )

    progress = tqdm(
        total=loop_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="Diffusion",
    )
    epoch = 0
    step_durations = []
    benchmark_tick = None
    diagnostics = TrainingDiagnostics(noise_scheduler.config.num_train_timesteps) if controlled and not benchmark_steps else None
    while global_step < loop_steps:
        torch.manual_seed(int(config["seed"]) + epoch)
        for batch in loader:
            with accelerator.accumulate(unet, condition_encoder):
                images = batch["pixel_values"]
                conditions = batch["condition"]
                with torch.no_grad():
                    latents = encode_training_latents(
                        vae, images, int(train_cfg.get("vae_encode_batch_size", 0))
                    ) * latent_scaling

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                jitter_std = float(composition_cfg["vicinal_jitter_std"])
                vicinal_condition = conditions + torch.randn_like(conditions) * jitter_std
                drop_mask = torch.rand(conditions.shape[0], device=conditions.device) < float(
                    model_cfg["condition_dropout"]
                )
                if descriptor_cfg:
                    class_embedding = condition_encoder(
                        vicinal_condition, drop_mask=drop_mask,
                        descriptors=batch["descriptors"], descriptor_valid=batch["descriptor_valid"])
                else:
                    class_embedding = condition_encoder(vicinal_condition, drop_mask=drop_mask)
                if controlled and (global_step < 5 or (global_step + 1) % int(train_cfg["log_every"]) == 0):
                    fingerprint = tensor_fingerprint(
                        images=images, conditions=conditions, latents=latents,
                        noise=noise, timesteps=timesteps,
                        vicinal_condition=vicinal_condition, drop_mask=drop_mask)
                    if reference_fingerprints is not None:
                        step = global_step + 1
                        matched = reference_fingerprints.get(step) == fingerprint
                        if not matched and input_policy == "strict_tensor_identity":
                            raise RuntimeError(f"Baseline input mismatch at step {step}; stopping before update")
                        if cross_replay is not None:
                            if step == 1 and fingerprint != cross_replay["results"]["current_local5"]["combined_sha256"]:
                                raise RuntimeError("First batch differs from the verified same-device replay")
                            if not matched:
                                fingerprint_mismatches.append(step)
                            append_csv_row(output_dir / "cross_device_inputs.csv", {
                                "additional_step": step, "historical_tensor_hash_matches": matched,
                                "images_conditions_sha256": tensor_fingerprint(images=images, conditions=conditions),
                                "actual_tensor_sha256": fingerprint,
                                "historical_tensor_sha256": reference_fingerprints[step]})
                        compared_steps.append(step)
                        save_json({"status": "in_progress", "checked_steps": compared_steps,
                                   "policy": input_policy, "historical_mismatch_steps": fingerprint_mismatches,
                                   "scope": "first five and logged batches, not every batch"},
                                  output_dir / "matched_baseline_inputs.json")
                        append_csv_row(output_dir / "descriptor_inputs.csv", {
                            "additional_step": step,
                            "valid_fraction": float(batch["descriptor_valid"].float().mean()),
                            "sha256": tensor_fingerprint(descriptors=batch["descriptors"],
                                valid=batch["descriptor_valid"], native_box=batch["native_box"]),
                        })
                    append_csv_row(output_dir / "input_fingerprints.csv", {
                        "additional_step": global_step + 1,
                        "sha256": fingerprint,
                    })
                prediction = unet(
                    noisy_latents,
                    timesteps,
                    class_labels=class_embedding,
                ).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported prediction type: {noise_scheduler.config.prediction_type}")

                loss = diffusion_mse_loss(
                    prediction, target, timesteps, noise_scheduler.alphas_cumprod,
                    snr_gamma=snr_gamma,
                    prediction_type=noise_scheduler.config.prediction_type,
                )
                boundary_loss = prediction.new_zeros(())
                if boundary_cfg:
                    selected = torch.nonzero(timesteps <= boundary_cfg["max_t"], as_tuple=True)[0][:boundary_cfg["max_samples"]]
                    if len(selected):
                        alpha_bar = noise_scheduler.alphas_cumprod.to(latents.device)[timesteps[selected]]
                        alpha = alpha_bar.sqrt().view(-1, 1, 1, 1)
                        sigma = (1 - alpha_bar).sqrt().view(-1, 1, 1, 1)
                        x0 = (noisy_latents[selected].float() - sigma * prediction[selected].float()) / alpha
                        decoded = vae.decode(x0 / latent_scaling).sample
                        boundary_loss = boundary_dice_loss(decoded, images[selected], threshold=0.2, temperature=0.05)
                    if args.boundary_calibrate:
                        if len(selected):
                            parameters = [p for p in accelerator.unwrap_model(unet).parameters() if p.requires_grad]
                            def grad_norm(objective):
                                gradients = torch.autograd.grad(objective, parameters, retain_graph=True, allow_unused=True)
                                squares = [g.detach().float().square().sum() for g in gradients if g is not None]
                                return float(torch.stack(squares).sum().sqrt()) if squares else 0.0
                            mse_norm = grad_norm(loss)
                            dice_norm = grad_norm(boundary_loss)
                            if not all(math.isfinite(v) and v > 0 for v in (mse_norm, dice_norm)):
                                raise RuntimeError("Non-finite or zero calibration gradient; no training started.")
                            calibration_ratios.append({"mse_grad_norm": mse_norm, "dice_grad_norm": dice_norm,
                                                       "selected": len(selected), "ratio": mse_norm / dice_norm})
                        global_step += 1
                        if global_step >= int(boundary_cfg.get("calibration_batches", 8)):
                            if len(calibration_ratios) < 4:
                                raise RuntimeError("Insufficient nonempty calibration batches; no training started.")
                            result = {"status": "passed", "optimizer_updates": 0,
                                      "config_sha256": file_sha256(args.config),
                                      "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                                      "batches": calibration_ratios,
                                      "lambda": 0.1 * statistics.median(x["ratio"] for x in calibration_ratios),
                                      "scope": "unet gradient norms before clipping; same batches; median ratio"}
                            save_json(result, output_dir / "boundary_calibration.json")
                            progress.close()
                            accelerator.end_training()
                            return
                        continue
                    loss = loss + boundary_weight * boundary_loss
                gb_loss = prediction.new_zeros(())
                edge_loss = prediction.new_zeros(())
                structure_fraction = float((timesteps <= structure_max_t).float().mean()) if structure_max_t >= 0 else 0.0
                if (lambda_gb > 0.0 or lambda_edge > 0.0) and structure_max_t >= 0:
                    gate = timesteps <= structure_max_t
                    if bool(gate.any()):
                        # Preserve the existing low-noise structure-loss path for non-pilot runs.
                        alpha_bar = noise_scheduler.alphas_cumprod.to(latents.device)[timesteps]
                        alpha = alpha_bar.sqrt().view(-1, 1, 1, 1)
                        sigma = (1.0 - alpha_bar).sqrt().view(-1, 1, 1, 1)
                        x0 = (noisy_latents - sigma * prediction) / alpha
                        decoded = vae.decode(x0[gate] / latent_scaling).sample
                        pred_gb, pred_edge = soft_structure_descriptors(decoded)
                        with torch.no_grad():
                            target_gb, target_edge = soft_structure_descriptors(images[gate])
                        if lambda_gb > 0.0:
                            gb_loss = F.mse_loss(pred_gb, target_gb)
                            loss = loss + lambda_gb * gb_loss
                        if lambda_edge > 0.0:
                            edge_loss = F.mse_loss(pred_edge, target_edge)
                            loss = loss + lambda_edge * edge_loss
                if controlled and not torch.isfinite(loss).item():
                    raise RuntimeError("Non-finite pilot loss; stopping instead of exporting invalid weights.")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    gradient_norm = accelerator.clip_grad_norm_(
                        list(unet.parameters()) + list(condition_encoder.parameters()), 1.0
                    )
                    if controlled and not torch.isfinite(gradient_norm).item():
                        raise RuntimeError("Non-finite pilot gradients; review both arms before continuing.")
                descriptor_grad_norm = None
                if descriptor_cfg and (global_step < 5 or (global_step + 1) % int(train_cfg["log_every"]) == 0 or global_step + 1 == loop_steps):
                    projection = accelerator.unwrap_model(condition_encoder).projection
                    descriptor_grad_norm = float(torch.sqrt(sum(
                        p.grad.detach().float().square().sum() for p in projection.parameters() if p.grad is not None)))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                if controlled and accelerator.optimizer_step_was_skipped:
                    raise RuntimeError("AMP skipped a pilot update. Stop and review both arms; do not compare unequal updates.")
                global_step += 1
                if diagnostics is not None:
                    diagnostics.record(prediction, target, timesteps, noise_scheduler.alphas_cumprod,
                                       snr_gamma, gradient_norm.detach().item())
                ema.step(accelerator.unwrap_model(unet).parameters())
                progress.update(1)
                if global_step % int(train_cfg["log_every"]) == 0 or (controlled and global_step == loop_steps):
                    diagnostic_logs = {}
                    if diagnostics is not None:
                        diagnostic_logs, bin_rows = diagnostics.flush(global_step)
                        for row in bin_rows:
                            append_csv_row(output_dir / "timestep_metrics.csv", row)
                        save_json(diagnostics.summary(), output_dir / "training_diagnostics.json")
                    logs = {
                        "loss": float(loss.detach()),
                        "noise_loss": float(F.mse_loss(prediction.float(), target.float()).detach()),
                        "gb_loss": float(gb_loss.detach()),
                        "edge_loss": float(edge_loss.detach()),
                        "structure_fraction": structure_fraction,
                        "learning_rate": float(lr_scheduler.get_last_lr()[0]),
                        "latent_scaling": latent_scaling,
                        "condition_drop_fraction": float(drop_mask.float().mean()),
                        **diagnostic_logs,
                    }
                    if boundary_cfg:
                        logs["boundary_dice_loss"] = float(boundary_loss.detach())
                        logs["boundary_dice_selected"] = len(selected)
                        logs["boundary_dice_lambda"] = boundary_weight
                    if descriptor_cfg:
                        logs["descriptor_valid_fraction"] = float(batch["descriptor_valid"].float().mean())
                        logs["descriptor_projection_grad_norm_after_clip"] = descriptor_grad_norm
                    if controlled or snr_gamma is not None:
                        logs["unweighted_mse"] = float(
                            (prediction.detach().float() - target.float()).square().mean()
                        )
                    timing = tracker.update(
                        global_step,
                        details={
                            "loss": logs["loss"],
                            "latent_scaling": latent_scaling,
                        },
                    ) if tracker is not None else None
                    if timing is not None:
                        logs["elapsed_seconds"] = timing["elapsed_seconds"]
                        logs["estimated_remaining_seconds"] = (
                            timing["estimated_remaining_seconds"]
                            if timing["estimated_remaining_seconds"] is not None
                            else -1.0
                        )
                        append_csv_row(
                            output_dir / "training_metrics.csv",
                            {
                                "global_step": global_step,
                                "loss": logs["loss"],
                                "noise_loss": logs["noise_loss"],
                                "gb_loss": logs["gb_loss"],
                                "edge_loss": logs["edge_loss"],
                                "structure_fraction": logs["structure_fraction"],
                                **({key: logs[key] for key in ("boundary_dice_loss", "boundary_dice_selected", "boundary_dice_lambda")} if boundary_cfg else {}),
                                **({"unweighted_mse": logs["unweighted_mse"]} if "unweighted_mse" in logs else {}),
                                **diagnostic_logs,
                                "learning_rate": logs["learning_rate"],
                                "latent_scaling": logs["latent_scaling"],
                                "condition_drop_fraction": logs[
                                    "condition_drop_fraction"
                                ],
                                "elapsed_seconds": timing["elapsed_seconds"],
                                "estimated_remaining_seconds": timing[
                                    "estimated_remaining_seconds"
                                ],
                                "estimated_finish_local": timing["estimated_finish_local"],
                                "cuda_memory_allocated_gb": (
                                    float(torch.cuda.memory_allocated() / 1024**3)
                                    if torch.cuda.is_available()
                                    else 0.0
                                ),
                            },
                        )
                    accelerator.log(logs, step=global_step)
                    progress.set_postfix(
                        loss=f"{logs['loss']:.4f}",
                        stage="diffusion training",
                        elapsed=timing["elapsed"] if timing is not None else "--:--:--",
                        remaining=(
                            timing["estimated_remaining"] if timing is not None else "--:--:--"
                        ),
                    )

                if not benchmark_steps and global_step % int(train_cfg["save_every"]) == 0:
                    checkpoint = output_dir / f"checkpoint-{global_step:07d}"
                    if tracker is not None:
                        tracker.set_stage(
                            "saving_checkpoint",
                            global_step,
                            details={"checkpoint": str(checkpoint)},
                        )
                    accelerator.save_state(str(checkpoint))
                    if accelerator.is_main_process:
                        save_json(
                            {
                                "global_step": global_step,
                                "latent_scaling": latent_scaling,
                                "initialization": initialization_report,
                                "snr_gamma": snr_gamma,
                                "elapsed_seconds": (
                                    tracker.elapsed_seconds if tracker is not None else 0.0
                                ),
                            },
                            checkpoint / "training_state.json",
                        )
                    if tracker is not None:
                        tracker.set_stage("diffusion_training", global_step)

                if benchmark_steps:
                    if accelerator.device.type == "cuda":
                        torch.cuda.synchronize(accelerator.device)
                    now = time.perf_counter()
                    if benchmark_tick is not None:
                        step_durations.append(now - benchmark_tick)
                    if global_step >= benchmark_warmup:
                        benchmark_tick = now
                    if global_step >= benchmark_steps:
                        seconds = sum(step_durations)
                        report = {
                            "kind": "disposable_throughput_benchmark_not_scientific_training",
                            "batch_size": int(train_cfg["batch_size"]),
                            "warmup_steps": benchmark_warmup,
                            "measured_steps": len(step_durations),
                            "step_seconds": step_durations,
                            "images_per_second": len(step_durations) * int(train_cfg["batch_size"]) / seconds,
                            "seconds_per_step": seconds / len(step_durations),
                            "device": str(accelerator.device),
                            "gpu_name": torch.cuda.get_device_name(accelerator.device) if accelerator.device.type == "cuda" else None,
                            "torch": torch.__version__, "cuda": torch.version.cuda,
                            "mixed_precision": train_cfg["mixed_precision"],
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(accelerator.device) if accelerator.device.type == "cuda" else 0,
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(accelerator.device) if accelerator.device.type == "cuda" else 0,
                            **benchmark_memory,
                            "includes": "data loading, FP32 VAE, input audits, backward, optimizer, EMA, logging; excludes startup/checkpoint IO",
                        }
                        save_json(report, output_dir / "benchmark.json")
                        progress.close()
                        accelerator.end_training()
                        return
                if global_step >= loop_steps:
                    break
        epoch += 1

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        if tracker is not None:
            tracker.set_stage("saving_final_diffusion_model", global_step)
        raw_unet = accelerator.unwrap_model(unet)
        raw_condition_encoder = accelerator.unwrap_model(condition_encoder)
        if bool(train_cfg.get("export_raw", False)):
            raw_unet.save_pretrained(final_dir / "unet_raw", safe_serialization=True)
        ema.store(raw_unet.parameters())
        ema.copy_to(raw_unet.parameters())
        raw_unet.save_pretrained(final_dir / "unet", safe_serialization=True)
        ema.restore(raw_unet.parameters())
        raw_condition_encoder.save_pretrained(final_dir / "condition_encoder")
        if descriptor_cfg:
            save_json({key: descriptor_store.metadata[key] for key in (
                "schema", "names", "settings", "units", "region_rule",
                "normalizer", "training_alloy_ids")},
                final_dir / "condition_encoder" / "descriptor_normalizer.json")
        noise_scheduler.save_pretrained(final_dir / "scheduler")
        conditioner.save(final_dir / "composition_conditioner.json")
        if descriptor_cfg:
            if set(compared_steps) != set(reference_fingerprints):
                raise RuntimeError("Incomplete baseline input audit")
            save_json({"status": "passed" if cross_replay is None else "completed_cross_device_screening",
                       "checked_steps": compared_steps, "policy": input_policy,
                       "historical_mismatch_steps": fingerprint_mismatches,
                       "limits": "Same-device first batch replay verified; historical per-step tensor identity is not claimed" if cross_replay else None,
                       "scope": "first five and logged batches, not every batch"},
                      output_dir / "matched_baseline_inputs.json")
        final_timing = tracker.finish(global_step) if tracker is not None else None
        save_json(
            {
                "global_step": global_step,
                "source_global_step": initialization_report["source_global_step"] if initialization_report else 0,
                "effective_total_step": global_step + (initialization_report["source_global_step"] if initialization_report else 0),
                "initialization": initialization_report,
                "snr_gamma": snr_gamma,
                "local_descriptors": descriptor_cfg,
                "unet_weight_types": {
                    "unet": "ema_unet_with_raw_condition_encoder",
                    **({"unet_raw": "raw"} if train_cfg.get("export_raw") else {}),
                },
                "latent_scaling": latent_scaling,
                "vae_dir": str(Path(vae_dir).resolve()),
                "vae_scale_factor": vae_scale_factor,
                "patch_size": int(data_cfg["patch_size"]),
                "exclude_alloy_id": args.exclude_alloy_id,
                "training_alloy_ids": dataset.alloy_ids,
                "active_features": conditioner.active_features,
                "dropped_features": conditioner.state.dropped_features,
                "structure_loss": {
                    "lambda_gb": lambda_gb,
                    "lambda_edge": lambda_edge,
                    "structure_max_t": structure_max_t,
                },
                "boundary_dice": ({**boundary_cfg, "lambda": boundary_weight} if boundary_cfg else None),
                "max_cuda_memory_gb": (
                    float(torch.cuda.max_memory_allocated() / 1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                ),
                "timing": final_timing,
            },
            final_dir / "run_metadata.json",
        )
    progress.close()
    accelerator.end_training()


@torch.no_grad()
def encode_training_latents(vae, images, microbatch: int = 0):
    if microbatch <= 0 or microbatch >= len(images):
        return vae.encode(images).latent_dist.sample()
    # Encode in FP32 chunks, but draw posterior noise once to preserve RNG ordering.
    parameters = torch.cat([
        vae.encode(chunk).latent_dist.parameters for chunk in images.split(microbatch)
    ])
    return DiagonalGaussianDistribution(parameters).sample()


@torch.no_grad()
def estimate_latent_scaling(vae, loader, device, max_batches: int = 16) -> float:
    count = 0
    total = torch.zeros((), device=device, dtype=torch.float64)
    total_square = torch.zeros((), device=device, dtype=torch.float64)
    for batch_index, batch in enumerate(loader):
        images = batch["pixel_values"].to(device)
        latents = vae.encode(images).latent_dist.sample().double()
        total += latents.sum()
        total_square += latents.square().sum()
        count += latents.numel()
        if batch_index + 1 >= max_batches:
            break
    mean = total / max(count, 1)
    variance = total_square / max(count, 1) - mean.square()
    std = torch.sqrt(variance.clamp_min(1.0e-12))
    return float(1.0 / std)


if __name__ == "__main__":
    main()
