from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DModel
from tqdm.auto import tqdm

from ebsd_diffusion.composition import (
    CompositionConditioner,
    load_compositions,
    parse_composition_argument,
)
from ebsd_diffusion.models import ContinuousConditionEmbedding
from ebsd_diffusion.utils import RunProgressTracker, center_crop_pil, tensor_to_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RGB IPF+GB images from an alloy composition.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--alloy-id", type=int)
    source.add_argument("--composition", help="Comma-separated values, e.g. Ni=45,Cr=17,Nb=3.5,Mo=2,Al=0.5,Ti=1")
    parser.add_argument("--compositions-csv", type=Path, default=Path("data/compositions.csv"))
    parser.add_argument("--vae-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("08_训练与生成结果/03_单次预览图"),
    )
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--crop-height", type=int, default=450)
    parser.add_argument("--crop-width", type=int, default=750)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.8)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--max-nearest-distance", type=float, default=3.0)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    session_started_monotonic = time.perf_counter()
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    metadata = json.loads((checkpoint / "run_metadata.json").read_text(encoding="utf-8"))
    conditioner = CompositionConditioner.load(checkpoint / "composition_conditioner.json")
    composition_frame = load_compositions(args.compositions_csv)

    if args.alloy_id is not None:
        normalized = conditioner.transform_by_alloy_id(composition_frame, args.alloy_id)
        raw_row = composition_frame.loc[composition_frame["alloy_id"] == args.alloy_id].iloc[0]
        raw_composition = {feature: float(raw_row[feature]) for feature in conditioner.active_features}
        condition_label = f"alloy_{args.alloy_id:02d}"
    else:
        raw_composition = parse_composition_argument(args.composition)
        normalized = conditioner.transform_mapping(raw_composition)
        condition_label = "custom"

    nearest_distance = conditioner.nearest_training_distance(normalized)
    range_violations = conditioner.raw_range_violations(raw_composition)
    if (nearest_distance > args.max_nearest_distance or range_violations) and not args.allow_extrapolation:
        raise ValueError(
            "Requested composition is outside the reliable training neighborhood. "
            f"nearest normalized distance={nearest_distance:.3f}, range violations={range_violations}. "
            "Use --allow-extrapolation only for explicitly labeled exploratory output."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracker = RunProgressTracker(
        output_dir=args.output_dir,
        run_name=f"EBSD DDIM generation: {condition_label}",
        total_units=args.count,
        unit_name="saved_image",
        session_started_monotonic=session_started_monotonic,
        file_prefix=condition_label,
    )
    tracker.set_stage(
        "loading_models",
        details={"condition_label": condition_label, "requested_images": args.count},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae_dir = args.vae_dir or Path(metadata["vae_dir"])
    vae = AutoencoderKL.from_pretrained(vae_dir).to(device=device, dtype=dtype).eval()
    unet = UNet2DModel.from_pretrained(checkpoint / "unet").to(device=device, dtype=dtype).eval()
    condition_encoder = ContinuousConditionEmbedding.from_pretrained(
        checkpoint / "condition_encoder"
    ).to(device=device, dtype=dtype).eval()
    scheduler = DDIMScheduler.from_pretrained(checkpoint / "scheduler")
    scheduler.set_timesteps(args.steps, device=device)
    tracker.set_stage("preparing_latent_noise")

    vae_scale_factor = int(metadata["vae_scale_factor"])
    unet_scale_factor = 2 ** (len(unet.config.block_out_channels) - 1)
    divisor = vae_scale_factor * unet_scale_factor
    if args.height % divisor or args.width % divisor:
        raise ValueError(f"height and width must be divisible by {divisor}.")

    latent_height = args.height // vae_scale_factor
    latent_width = args.width // vae_scale_factor
    generator = torch.Generator(device=device).manual_seed(args.seed)
    suffix = "_EXTRAPOLATION" if nearest_distance > args.max_nearest_distance or range_violations else ""
    generated_count = 0
    generated_files: list[str] = []
    image_progress = tqdm(total=args.count, desc="Generated images")
    while generated_count < args.count:
        current_batch = min(args.batch_size, args.count - generated_count)
        batch_number = generated_count // max(args.batch_size, 1) + 1
        tracker.set_stage(
            "ddim_denoising",
            generated_count,
            details={"batch": batch_number, "ddim_steps": args.steps},
        )
        condition = torch.as_tensor(normalized, device=device, dtype=dtype).unsqueeze(0)
        condition = condition.expand(current_batch, -1)
        conditional_embedding = condition_encoder(condition)
        unconditional_embedding = condition_encoder.unconditional(current_batch, device=device, dtype=dtype)
        latents = torch.randn(
            (current_batch, int(vae.config.latent_channels), latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=dtype,
        ) * scheduler.init_noise_sigma

        for timestep in tqdm(
            scheduler.timesteps,
            desc=f"DDIM batch {batch_number}",
            leave=False,
        ):
            latent_input = torch.cat([latents, latents], dim=0)
            class_embedding = torch.cat([unconditional_embedding, conditional_embedding], dim=0)
            noise_prediction = unet(
                latent_input,
                timestep,
                class_labels=class_embedding,
            ).sample
            unconditional, conditional = noise_prediction.chunk(2)
            guided = unconditional + args.guidance_scale * (conditional - unconditional)
            latents = scheduler.step(guided, timestep, latents, generator=generator).prev_sample

        tracker.set_stage(
            "vae_decoding_and_saving",
            generated_count,
            details={"batch": batch_number},
        )
        decoded = vae.decode(latents / float(metadata["latent_scaling"])).sample
        for local_index, tensor in enumerate(decoded):
            index = generated_count + local_index
            image = tensor_to_pil(tensor)
            image = center_crop_pil(image, args.crop_height, args.crop_width)
            path = args.output_dir / f"{condition_label}_seed{args.seed}_{index:03d}{suffix}.png"
            image.save(path)
            generated_files.append(str(path.resolve()))
        generated_count += current_batch
        timing = tracker.update(
            generated_count,
            details={"last_saved_batch": batch_number},
        )
        image_progress.update(current_batch)
        image_progress.set_postfix(
            stage="saved",
            elapsed=timing["elapsed"],
            remaining=timing["estimated_remaining"],
        )

    image_progress.close()
    final_timing = tracker.finish(generated_count)

    report = {
        "condition_label": condition_label,
        "checkpoint": str(checkpoint),
        "vae_dir": str(Path(vae_dir).resolve()),
        "raw_composition": raw_composition,
        "active_features": conditioner.active_features,
        "normalized_condition": normalized.tolist(),
        "nearest_training_distance": nearest_distance,
        "raw_range_violations": range_violations,
        "is_extrapolation": bool(
            nearest_distance > args.max_nearest_distance or range_violations
        ),
        "guidance_scale": args.guidance_scale,
        "steps": args.steps,
        "seed": args.seed,
        "output_count": args.count,
        "generated_files": generated_files,
        "generated_height": args.crop_height,
        "generated_width": args.crop_width,
        "timing": final_timing,
    }
    (args.output_dir / f"{condition_label}_generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
