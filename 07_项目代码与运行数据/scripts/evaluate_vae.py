from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ebsd_diffusion.data import EbsdPatchDataset
from ebsd_diffusion.models import sobel_magnitude
from ebsd_diffusion.utils import RunProgressTracker, tensor_to_pil


def main() -> None:
    session_started_monotonic = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Save deterministic original/reconstruction comparisons for the trained VAE."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument(
        "--vae-dir",
        type=Path,
        default=Path("08_训练与生成结果/01_VAE模型/final"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("08_训练与生成结果/01_VAE模型/重建检查"),
    )
    parser.add_argument("--patch-size", type=int, default=192)
    args = parser.parse_args()

    if not args.vae_dir.is_dir():
        raise FileNotFoundError(
            f"Trained VAE not found: {args.vae_dir.resolve()}. Run 03_正式训练VAE.py first."
        )

    dataset = EbsdPatchDataset(
        manifest_path=args.manifest,
        patch_size=args.patch_size,
        patches_per_image=1,
        target="evaluation",
        deterministic=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracker = RunProgressTracker(
        output_dir=args.output_dir,
        run_name="VAE reconstruction inspection",
        total_units=len(dataset),
        unit_name="alloy_image",
        session_started_monotonic=session_started_monotonic,
        file_prefix="vae_reconstruction",
    )
    tracker.set_stage("loading_trained_vae")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(args.vae_dir).to(device=device, dtype=dtype).eval()

    records: list[dict[str, object]] = []
    progress = tqdm(loader, total=len(dataset), desc="VAE reconstruction check")
    tracker.set_stage("encoding_and_reconstructing")
    with torch.no_grad():
        for index, batch in enumerate(progress, start=1):
            images = batch["pixel_values"].to(device=device, dtype=dtype)
            latents = vae.encode(images).latent_dist.mode()
            reconstructed = vae.decode(latents).sample
            rgb_l1 = float(F.l1_loss(reconstructed.float(), images.float()).item())
            edge_l1 = float(
                F.l1_loss(
                    sobel_magnitude(reconstructed.float()),
                    sobel_magnitude(images.float()),
                ).item()
            )

            alloy_id = int(batch["alloy_id"].item())
            original_image = tensor_to_pil(images[0])
            reconstructed_image = tensor_to_pil(reconstructed[0])
            comparison = make_comparison(original_image, reconstructed_image, alloy_id)
            output_path = args.output_dir / f"alloy_{alloy_id:02d}_comparison.png"
            comparison.save(output_path)
            records.append(
                {
                    "alloy_id": alloy_id,
                    "source_image": str(batch["image_path"][0]),
                    "comparison_image": str(output_path.resolve()),
                    "rgb_l1": rgb_l1,
                    "sobel_edge_l1": edge_l1,
                }
            )
            timing = tracker.update(
                index,
                details={"alloy_id": alloy_id, "rgb_l1": rgb_l1, "sobel_edge_l1": edge_l1},
            )
            progress.set_postfix(
                alloy=alloy_id,
                rgb_l1=f"{rgb_l1:.4f}",
                edge_l1=f"{edge_l1:.4f}",
                remaining=timing["estimated_remaining"],
            )

    final_timing = tracker.finish(len(dataset))
    report = {
        "vae_dir": str(args.vae_dir.resolve()),
        "patch_size": args.patch_size,
        "image_count": len(records),
        "mean_rgb_l1": sum(float(row["rgb_l1"]) for row in records) / max(len(records), 1),
        "mean_sobel_edge_l1": (
            sum(float(row["sobel_edge_l1"]) for row in records) / max(len(records), 1)
        ),
        "timing": final_timing,
        "per_alloy": records,
    }
    (args.output_dir / "vae_reconstruction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(records)} VAE comparisons to {args.output_dir.resolve()}")


def make_comparison(original: Image.Image, reconstructed: Image.Image, alloy_id: int) -> Image.Image:
    header_height = 28
    canvas = Image.new(
        "RGB",
        (original.width + reconstructed.width, max(original.height, reconstructed.height) + header_height),
        color=(245, 245, 245),
    )
    canvas.paste(original, (0, header_height))
    canvas.paste(reconstructed, (original.width, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"Alloy {alloy_id:02d} - Original", fill=(20, 20, 20))
    draw.text((original.width + 8, 8), "VAE reconstruction", fill=(20, 20, 20))
    return canvas


if __name__ == "__main__":
    main()
