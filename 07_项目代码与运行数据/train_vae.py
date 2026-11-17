from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ebsd_diffusion.config import flatten_config_for_logging, load_config, resolve_project_path
from ebsd_diffusion.data import EbsdPatchDataset, resolve_exclude_alloy_id
from ebsd_diffusion.euler_images import channel_count, file_sha256
from ebsd_diffusion.euler_loss import euler_gb_vae_loss
from ebsd_diffusion.models import build_vae, parameter_count, sobel_magnitude
from ebsd_diffusion.train_status import cuda_memory_gb, format_step_line, require_fresh_output_dir
from ebsd_diffusion.utils import (
    RunProgressTracker,
    append_csv_row,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EBSD-specific variational autoencoder.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--exclude-alloy-id", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    session_started_monotonic = time.perf_counter()
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    data_cfg = config["data"]
    train_cfg = config["train_vae"]
    args.exclude_alloy_id = resolve_exclude_alloy_id(data_cfg, args.exclude_alloy_id)
    image_mode = str(data_cfg.get("image_mode", "rgb_ipf"))
    if image_mode == "euler_gb":
        expected_channels = channel_count(str(config["euler_gb"]["phi_encoding"]))
        if int(config["vae"].get("in_channels", 3)) != expected_channels:
            raise ValueError("vae.in_channels does not match the Euler+GB encoding.")
        if int(config["vae"].get("out_channels", expected_channels)) != expected_channels:
            raise ValueError("vae.out_channels does not match the Euler+GB encoding.")
    if args.max_steps is not None:
        train_cfg["max_steps"] = args.max_steps

    output_dir = args.output_dir.resolve() if args.output_dir else resolve_project_path(config, train_cfg["output_dir"])
    if args.output_dir is None and args.exclude_alloy_id is not None and "output_root" not in config:
        output_dir = output_dir.parent / f"vae_loo_{args.exclude_alloy_id:02d}"
    if image_mode == "euler_gb":
        require_fresh_output_dir(output_dir, args.resume_from)
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
        accelerator.init_trackers("ebsd_vae", config=flatten_config_for_logging(config))

    dataset = EbsdPatchDataset(
        manifest_path=resolve_project_path(config, data_cfg["manifest"]),
        patch_size=int(data_cfg["patch_size"]),
        patches_per_image=int(data_cfg["patches_per_image"]),
        target="vae",
        exclude_alloy_id=args.exclude_alloy_id,
        crop_margins=tuple(data_cfg["crop_margins"]),
        horizontal_flip=bool(data_cfg["horizontal_flip"]),
        vertical_flip=bool(data_cfg["vertical_flip"]),
        rotate_180=bool(data_cfg["rotate_180"]),
        image_mode=image_mode,
        euler_gb=config.get("euler_gb"),
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

    vae = build_vae(config["vae"], int(data_cfg["patch_size"]))
    vae_parameter_count = parameter_count(vae)
    if accelerator.is_main_process:
        print(f"VAE parameters {vae_parameter_count}", flush=True)
    if bool(train_cfg.get("gradient_checkpointing", False)):
        vae.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(train_cfg["warmup_steps"]),
        num_training_steps=int(train_cfg["max_steps"]),
    )
    vae, optimizer, loader, lr_scheduler = accelerator.prepare(
        vae, optimizer, loader, lr_scheduler
    )

    global_step = 0
    previous_elapsed_seconds = 0.0
    if args.resume_from:
        accelerator.load_state(str(args.resume_from))
        state_path = args.resume_from / "training_state.json"
        if state_path.is_file():
            state = __import__("json").loads(state_path.read_text())
            global_step = int(state["global_step"])
            previous_elapsed_seconds = float(state.get("elapsed_seconds", 0.0))

    tracker = None
    if accelerator.is_local_main_process:
        tracker = RunProgressTracker(
            output_dir=output_dir,
            run_name="EBSD VAE training",
            total_units=int(train_cfg["max_steps"]),
            unit_name="optimizer_step",
            initial_units=global_step,
            previous_elapsed_seconds=previous_elapsed_seconds,
            session_started_monotonic=session_started_monotonic,
        )
        tracker.set_stage(
            "vae_training",
            global_step,
            details={"exclude_alloy_id": args.exclude_alloy_id},
        )

    progress = tqdm(
        total=int(train_cfg["max_steps"]),
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="VAE",
    )
    epoch = 0
    while global_step < int(train_cfg["max_steps"]):
        dataset_epoch_seed = int(config["seed"]) + epoch
        torch.manual_seed(dataset_epoch_seed)
        for batch in loader:
            with accelerator.accumulate(vae):
                images = batch["pixel_values"]
                posterior = vae.encode(images).latent_dist
                latents = posterior.sample()
                reconstruction = vae.decode(latents).sample
                if image_mode == "rgb_ipf":
                    reconstruction_loss = F.l1_loss(reconstruction, images)
                    edge_loss = F.l1_loss(sobel_magnitude(reconstruction), sobel_magnitude(images))
                    kl_loss = posterior.kl().mean()
                    loss = (
                        reconstruction_loss
                        + float(config["vae"]["edge_weight"]) * edge_loss
                        + float(config["vae"]["kl_weight"]) * kl_loss
                    )
                    loss_parts = {
                        "reconstruction_l1": reconstruction_loss,
                        "edge_l1": edge_loss,
                        "kl": kl_loss,
                    }
                elif image_mode == "euler_gb":
                    loss_parts = euler_gb_vae_loss(
                        reconstruction,
                        images,
                        posterior,
                        config["euler_gb"],
                        float(config["vae"]["kl_weight"]),
                    )
                    loss = loss_parts["loss"]
                else:
                    raise ValueError(f"Unsupported image_mode: {image_mode}")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                if global_step % int(train_cfg["log_every"]) == 0:
                    logs = {
                        "loss": float(loss.detach()),
                        "learning_rate": float(lr_scheduler.get_last_lr()[0]),
                    }
                    if image_mode == "rgb_ipf":
                        logs["reconstruction_l1"] = float(loss_parts["reconstruction_l1"].detach())
                        logs["edge_l1"] = float(loss_parts["edge_l1"].detach())
                        logs["kl"] = float(loss_parts["kl"].detach())
                    else:
                        logs["orientation_l1"] = float(loss_parts["orientation_l1"].detach())
                        logs["gb_loss"] = float(loss_parts["gb_loss"].detach())
                        logs["gb_edge_l1"] = float(loss_parts["gb_edge_l1"].detach())
                        logs["kl"] = float(loss_parts["kl"].detach())
                    timing = tracker.update(
                        global_step,
                        details={"loss": logs["loss"]},
                    ) if tracker is not None else None
                    if timing is not None:
                        logs["elapsed_seconds"] = timing["elapsed_seconds"]
                        logs["estimated_remaining_seconds"] = (
                            timing["estimated_remaining_seconds"]
                            if timing["estimated_remaining_seconds"] is not None
                            else -1.0
                        )
                        allocated_gb, peak_gb = cuda_memory_gb()
                        if image_mode == "rgb_ipf":
                            metric_row = {
                                "global_step": global_step,
                                "loss": logs["loss"],
                                "reconstruction_l1": logs["reconstruction_l1"],
                                "sobel_edge_l1": logs["edge_l1"],
                                "kl": logs["kl"],
                                "learning_rate": logs["learning_rate"],
                                "elapsed_seconds": timing["elapsed_seconds"],
                                "estimated_remaining_seconds": timing[
                                    "estimated_remaining_seconds"
                                ],
                                "estimated_finish_local": timing["estimated_finish_local"],
                                "cuda_memory_allocated_gb": allocated_gb,
                            }
                            step_extras = [
                                f"recon {logs['reconstruction_l1']:.4f}",
                                f"edge {logs['edge_l1']:.4f}",
                                f"kl {logs['kl']:.4f}",
                            ]
                        else:
                            metric_row = {
                                "global_step": global_step,
                                "loss": logs["loss"],
                                "orientation_l1": logs["orientation_l1"],
                                "gb_loss": logs["gb_loss"],
                                "gb_edge_l1": logs["gb_edge_l1"],
                                "kl": logs["kl"],
                                "learning_rate": logs["learning_rate"],
                                "elapsed_seconds": timing["elapsed_seconds"],
                                "estimated_remaining_seconds": timing[
                                    "estimated_remaining_seconds"
                                ],
                                "estimated_finish_local": timing["estimated_finish_local"],
                                "cuda_memory_allocated_gb": allocated_gb,
                                "cuda_memory_peak_gb": peak_gb,
                            }
                            step_extras = [
                                f"ori {logs['orientation_l1']:.4f}",
                                f"gb {logs['gb_loss']:.4f}",
                                f"edge {logs['gb_edge_l1']:.4f}",
                                f"kl {logs['kl']:.4f}",
                            ]
                        step_extras.append(f"lr {logs['learning_rate']:.3e}")
                        append_csv_row(output_dir / "training_metrics.csv", metric_row)
                        if accelerator.is_local_main_process:
                            print(
                                format_step_line(
                                    global_step,
                                    int(train_cfg["max_steps"]),
                                    logs["loss"],
                                    timing,
                                    step_extras,
                                ),
                                flush=True,
                            )
                    accelerator.log(logs, step=global_step)
                    allocated_gb, _peak_gb = cuda_memory_gb()
                    progress.set_postfix(
                        loss=f"{logs['loss']:.4f}",
                        stage="VAE training",
                        elapsed=timing["elapsed"] if timing is not None else "--:--:--",
                        remaining=(
                            timing["estimated_remaining"] if timing is not None else "--:--:--"
                        ),
                        gpu=f"{allocated_gb:.2f}GB",
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
                                "elapsed_seconds": (
                                    tracker.elapsed_seconds if tracker is not None else 0.0
                                ),
                            },
                            checkpoint / "training_state.json",
                        )
                    if tracker is not None:
                        tracker.set_stage("vae_training", global_step)

                if global_step >= int(train_cfg["max_steps"]):
                    break
        epoch += 1

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        if tracker is not None:
            tracker.set_stage("saving_final_vae", global_step)
        accelerator.unwrap_model(vae).save_pretrained(final_dir, safe_serialization=True)
        final_timing = tracker.finish(global_step) if tracker is not None else None
        save_json(
            {
                "global_step": global_step,
                "exclude_alloy_id": args.exclude_alloy_id,
                "training_alloy_ids": dataset.alloy_ids,
                "patch_size": int(data_cfg["patch_size"]),
                "image_mode": image_mode,
                "vae_parameter_count": vae_parameter_count,
                "config_sha256": file_sha256(args.config),
                "manifest_sha256": file_sha256(resolve_project_path(config, data_cfg["manifest"])),
                "max_cuda_memory_gb": (
                    float(torch.cuda.max_memory_allocated() / 1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                ),
                "timing": final_timing,
            },
            final_dir / "training_metadata.json",
        )
    progress.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
