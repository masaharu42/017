"""Dense, read-only gradient probe for Fold03 structure-loss experiments.

Run with the 017 project environment.  It does not update model weights.
It reproduces train_diffusion.py's epsilon/GB/edge losses and reports
unweighted and configured weighted gradient ratios for t <= 300.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel


def soft_desc(images: torch.Tensor):
    rgb = (images + 1.0) * 0.5 if images.min() < 0 else images
    max_rgb = rgb.amax(dim=1, keepdim=True)
    gb = torch.sigmoid((0.22 - max_rgb) / 0.035).mean(dim=(1, 2, 3))
    gray = rgb.mean(dim=1, keepdim=True)
    kx = rgb.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    edge = torch.sqrt(gx.square() + gy.square() + 1e-8).mean(dim=(1, 2, 3))
    return gb, edge


def flat_grad(model):
    return torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])


def crop_image(image: Image.Image, size: int, name: str):
    w, h = image.size
    boxes = {
        "center": ((w-size)//2, (h-size)//2),
        "left": (0, max(0, (h-size)//2)),
        "right": (max(0, w-size), max(0, (h-size)//2)),
        "top": ((w-size)//2, 0),
    }
    x, y = boxes[name]
    return image.crop((x, y, x+size, y+size))


def cosine(a, b):
    return float(F.cosine_similarity(a[None], b[None]).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--vae", type=Path, required=True)
    ap.add_argument("--alloy-id", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lambda-gb", type=float, default=1.0)
    ap.add_argument("--lambda-edge", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[20261017, 20261018, 20261019])
    args = ap.parse_args()
    sys.path.insert(0, str(args.project))
    from ebsd_diffusion.composition import CompositionConditioner, load_compositions
    from ebsd_diffusion.models import ContinuousConditionEmbedding

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    meta_path = args.model / "run_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    scale = float(meta.get("latent_scaling", 1.0069132457274694))
    unet = UNet2DModel.from_pretrained(args.model / "unet", torch_dtype=dtype).to(device).eval()
    cond = ContinuousConditionEmbedding.from_pretrained(
        args.model / "condition_encoder", map_location="cpu"
    ).to(device).eval()
    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=dtype).to(device).eval()
    vae.requires_grad_(False)
    conditioner = CompositionConditioner.load(args.model / "composition_conditioner.json")
    compositions = load_compositions(args.project / "data" / "compositions.csv")
    c = torch.tensor(conditioner.transform_by_alloy_id(compositions, args.alloy_id), device=device, dtype=dtype).unsqueeze(0)
    cemb = cond(c)
    scheduler = DDPMScheduler.from_pretrained(args.model / "scheduler")
    alphas = scheduler.alphas_cumprod.to(device=device, dtype=dtype)
    image_path = args.project / "data" / "原始EBSD导出数据" / str(args.alloy_id) / f"{args.alloy_id}-IPF + GB.bmp"
    image = Image.open(image_path).convert("RGB")
    timesteps = [50, 100, 150, 200, 250, 299, 300]
    rows = []
    for crop_name in ["center", "left", "right", "top"]:
        arr = np.asarray(crop_image(image, 256, crop_name), dtype=np.float32) / 255.0
        target = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        with torch.no_grad():
            target_z = vae.encode(target).latent_dist.mean * scale
            target_gb, target_edge = soft_desc(target)
        for seed in args.seeds:
            for t_int in timesteps:
                torch.manual_seed(seed)
                noise = torch.randn_like(target_z)
                t = torch.tensor([t_int], device=device, dtype=torch.long)
                a = alphas[t].sqrt().view(1, 1, 1, 1)
                b = (1 - alphas[t]).sqrt().view(1, 1, 1, 1)
                noisy = a * target_z + b * noise
                unet.zero_grad(set_to_none=True)
                pred = unet(noisy, t, class_labels=cemb).sample
                noise_loss = F.mse_loss(pred.float(), noise.float())
                x0 = (noisy - b * pred) / a
                decoded = vae.decode(x0 / scale).sample
                pred_gb, pred_edge = soft_desc(decoded)
                gb_loss = F.mse_loss(pred_gb, target_gb.detach())
                edge_loss = F.mse_loss(pred_edge, target_edge.detach())
                losses = {"noise": noise_loss, "gb": gb_loss, "edge": edge_loss}
                grads = {}
                for name, loss in losses.items():
                    unet.zero_grad(set_to_none=True)
                    loss.backward(retain_graph=True)
                    grads[name] = flat_grad(unet)
                gn = grads["noise"].norm().item()
                gg = grads["gb"].norm().item()
                ge = grads["edge"].norm().item()
                rows.append({
                    "crop": crop_name, "seed": seed, "t": t_int,
                    "noise_loss": float(noise_loss.detach()), "gb_loss": float(gb_loss.detach()),
                    "edge_loss": float(edge_loss.detach()), "grad_noise": gn,
                    "grad_gb": gg, "grad_edge": ge, "gb_noise_ratio": gg/max(gn,1e-12),
                    "edge_noise_ratio": ge/max(gn,1e-12),
                    "weighted_gb_noise_ratio": args.lambda_gb*gg/max(gn,1e-12),
                    "weighted_edge_noise_ratio": args.lambda_edge*ge/max(gn,1e-12),
                    "cos_noise_gb": cosine(grads["noise"], grads["gb"]),
                    "cos_noise_edge": cosine(grads["noise"], grads["edge"]),
                    "cos_gb_edge": cosine(grads["gb"], grads["edge"]),
                    "pred_gb": float(pred_gb.detach()), "target_gb": float(target_gb.detach()),
                    "pred_edge": float(pred_edge.detach()), "target_edge": float(target_edge.detach()),
                })
    out = args.out.resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(out, index=False)
    summary = df.groupby("t").agg({c: ["mean", "std"] for c in ["gb_noise_ratio", "edge_noise_ratio", "weighted_gb_noise_ratio", "weighted_edge_noise_ratio", "cos_noise_gb", "cos_noise_edge"]})
    summary.to_csv(out.with_name(out.stem + "_summary.csv"))
    print(df.groupby("t")[["weighted_gb_noise_ratio", "weighted_edge_noise_ratio", "cos_noise_gb", "cos_noise_edge"]].mean().to_string())
    print("saved", out, out.with_name(out.stem + "_summary.csv"))


if __name__ == "__main__":
    main()
