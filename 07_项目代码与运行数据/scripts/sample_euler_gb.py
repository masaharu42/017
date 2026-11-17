"""Sample Euler+GB windows from the fixed seed list in the training config.

There is no seed argument and no comparison against a held-out image.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DModel
from PIL import Image

from ebsd_diffusion.composition import CompositionConditioner, load_compositions
from ebsd_diffusion.config import load_config, resolve_project_path
from ebsd_diffusion.euler_images import decode_orientation_channels, shared_sample_seeds
from ebsd_diffusion.models import ContinuousConditionEmbedding
from ebsd_diffusion.train_status import require_fresh_output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Euler+GB maps with the shared fixed seeds.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--alloy-id", type=int, required=True)
    parser.add_argument("--vae-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int)
    return parser.parse_args(argv)


def _preview(euler_degrees: np.ndarray, boundary: np.ndarray, config: dict) -> Image.Image:
    spans = (
        float(config["phi1_degrees"]),
        float(config["Phi_degrees"]),
        float(config["phi2_degrees"]),
    )
    rgb = np.stack([np.clip(euler_degrees[..., index] / spans[index] * 255.0, 0, 255) for index in range(3)], axis=-1)
    rgb[np.asarray(boundary) >= 0.5] = 0
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    seeds = shared_sample_seeds(config)
    count = len(seeds) if args.count is None else int(args.count)
    if count < 1 or count > len(seeds):
        raise ValueError(f"count must be between 1 and the {len(seeds)} configured seeds.")
    output_dir = args.output_dir.resolve()
    require_fresh_output_dir(output_dir, None)
    output_dir.mkdir(parents=True)
    checkpoint = args.checkpoint.resolve()
    metadata = json.loads((checkpoint / "run_metadata.json").read_text(encoding="utf-8"))
    conditioner = CompositionConditioner.load(checkpoint / "composition_conditioner.json")
    compositions = load_compositions(resolve_project_path(config, config["data"]["compositions"]))
    normalized = conditioner.transform_by_alloy_id(compositions, args.alloy_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae_dir = args.vae_dir or Path(metadata["vae_dir"])
    vae = AutoencoderKL.from_pretrained(vae_dir).to(device=device, dtype=dtype).eval()
    unet = UNet2DModel.from_pretrained(checkpoint / "unet").to(device=device, dtype=dtype).eval()
    encoder = ContinuousConditionEmbedding.from_pretrained(checkpoint / "condition_encoder").to(device=device, dtype=dtype).eval()
    scheduler = DDIMScheduler.from_pretrained(checkpoint / "scheduler")
    steps = int(config["sampling"]["num_inference_steps"])
    guidance = float(config["sampling"]["guidance_scale"])
    scheduler.set_timesteps(steps, device=device)
    height = int(config["sampling"]["height"])
    width = int(config["sampling"]["width"])
    latent_height = height // int(metadata["vae_scale_factor"])
    latent_width = width // int(metadata["vae_scale_factor"])
    condition = torch.as_tensor(normalized, device=device, dtype=dtype).unsqueeze(0)
    saved = []
    for index, seed in enumerate(seeds[:count]):
        generator = torch.Generator(device=device).manual_seed(int(seed))
        latents = torch.randn(
            (1, int(vae.config.latent_channels), latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=dtype,
        ) * scheduler.init_noise_sigma
        conditional = encoder(condition)
        unconditional = encoder.unconditional(1, device=device, dtype=dtype)
        for timestep in scheduler.timesteps:
            latent_input = torch.cat([latents, latents], dim=0)
            class_embedding = torch.cat([unconditional, conditional], dim=0)
            noise_prediction = unet(latent_input, timestep, class_labels=class_embedding).sample
            unconditional_noise, conditional_noise = noise_prediction.chunk(2)
            guided = unconditional_noise + guidance * (conditional_noise - unconditional_noise)
            latents = scheduler.step(guided, timestep, latents, generator=generator).prev_sample
        decoded = vae.decode(latents / float(metadata["latent_scaling"])).sample[0].float().cpu().numpy()
        euler = decode_orientation_channels(decoded, config["euler_gb"])
        boundary = 1.0 / (1.0 + np.exp(-decoded[-1]))
        path = output_dir / f"alloy_{args.alloy_id:02d}_seed_{int(seed)}.npz"
        np.savez(
            path,
            euler_degrees=euler.astype(np.float32),
            boundary=boundary.astype(np.float32),
            channels=decoded.astype(np.float32),
            alloy_id=np.int32(args.alloy_id),
            seed=np.int64(seed),
        )
        _preview(euler, boundary >= 0.5, config["euler_gb"]).save(output_dir / f"alloy_{args.alloy_id:02d}_seed_{int(seed)}.png")
        saved.append({"seed": int(seed), "index": index, "file": str(path)})
        print(f"saved seed {seed} ({index + 1}/{count})", flush=True)
    report = {
        "alloy_id": args.alloy_id,
        "seeds": [int(seed) for seed in seeds[:count]],
        "seed_policy": "fixed list from config; no selection from a held-out image",
        "files": saved,
    }
    (output_dir / "sample_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
