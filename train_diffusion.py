from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ebsd_diffusion.composition import CompositionConditioner, load_compositions
from ebsd_diffusion.config import flatten_config_for_logging, load_config, resolve_project_path
from ebsd_diffusion.data import EbsdPatchDataset, build_condition_mapping
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
    seed_everything(int(config["seed"]))
    data_cfg = config["data"]
    composition_cfg = config["composition"]
    model_cfg = config["diffusion"]
    train_cfg = config["train_diffusion"]
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

    output_dir = args.output_dir.resolve() if args.output_dir else resolve_project_path(config, train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        mixed_precision=str(train_cfg["mixed_precision"]),
        log_with="tensorboard",
        project_dir=output_dir / "logs",
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
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

    vae_dir = args.vae_dir or resolve_project_path(config, train_cfg["vae_dir"])
    vae = AutoencoderKL.from_pretrained(vae_dir)
    vae.requires_grad_(False).eval().to(accelerator.device)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_size = int(data_cfg["patch_size"]) // vae_scale_factor

    unet = build_unet(model_cfg, int(vae.config.latent_channels), latent_size)
    if bool(train_cfg.get("gradient_checkpointing", False)):
        unet.enable_gradient_checkpointing()
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

    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(condition_encoder.parameters()),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
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

    latent_scaling = estimate_latent_scaling(vae, loader, accelerator.device, max_batches=16)
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
        total=int(train_cfg["max_steps"]),
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="Diffusion",
    )
    epoch = 0
    while global_step < int(train_cfg["max_steps"]):
        torch.manual_seed(int(config["seed"]) + epoch)
        for batch in loader:
            with accelerator.accumulate(unet, condition_encoder):
                images = batch["pixel_values"]
                conditions = batch["condition"]
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample() * latent_scaling

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
                class_embedding = condition_encoder(vicinal_condition, drop_mask=drop_mask)
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

                loss = F.mse_loss(prediction.float(), target.float())
                gb_loss = prediction.new_zeros(())
                edge_loss = prediction.new_zeros(())
                structure_fraction = float((timesteps <= structure_max_t).float().mean()) if structure_max_t >= 0 else 0.0
                if (lambda_gb > 0.0 or lambda_edge > 0.0) and structure_max_t >= 0:
                    gate = timesteps <= structure_max_t
                    if bool(gate.any()):
                        # x0 estimate is only decoded for the low-noise subset; VAE weights stay frozen.
                        alpha = noise_scheduler.alphas_cumprod.to(latents.device)[timesteps]
                        alpha = alpha.sqrt().view(-1, 1, 1, 1)
                        sigma = (1.0 - noise_scheduler.alphas_cumprod.to(latents.device)[timesteps]).sqrt().view(-1, 1, 1, 1)
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
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(unet.parameters()) + list(condition_encoder.parameters()), 1.0
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                ema.step(accelerator.unwrap_model(unet).parameters())
                progress.update(1)
                if global_step % int(train_cfg["log_every"]) == 0:
                    logs = {
                        "loss": float(loss.detach()),
                        "noise_loss": float(F.mse_loss(prediction.float(), target.float()).detach()),
                        "gb_loss": float(gb_loss.detach()),
                        "edge_loss": float(edge_loss.detach()),
                        "structure_fraction": structure_fraction,
                        "learning_rate": float(lr_scheduler.get_last_lr()[0]),
                        "latent_scaling": latent_scaling,
                        "condition_drop_fraction": float(drop_mask.float().mean()),
                    }
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

                if global_step % int(train_cfg["save_every"]) == 0:
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
                                "elapsed_seconds": (
                                    tracker.elapsed_seconds if tracker is not None else 0.0
                                ),
                            },
                            checkpoint / "training_state.json",
                        )
                    if tracker is not None:
                        tracker.set_stage("diffusion_training", global_step)

                if global_step >= int(train_cfg["max_steps"]):
                    break
        epoch += 1

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        if tracker is not None:
            tracker.set_stage("saving_final_diffusion_model", global_step)
        raw_unet = accelerator.unwrap_model(unet)
        raw_condition_encoder = accelerator.unwrap_model(condition_encoder)
        ema.store(raw_unet.parameters())
        ema.copy_to(raw_unet.parameters())
        raw_unet.save_pretrained(final_dir / "unet", safe_serialization=True)
        ema.restore(raw_unet.parameters())
        raw_condition_encoder.save_pretrained(final_dir / "condition_encoder")
        noise_scheduler.save_pretrained(final_dir / "scheduler")
        conditioner.save(final_dir / "composition_conditioner.json")
        final_timing = tracker.finish(global_step) if tracker is not None else None
        save_json(
            {
                "global_step": global_step,
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
