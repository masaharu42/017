from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from diffusers import DDPMScheduler

from ebsd_diffusion.models import (
    ContinuousConditionEmbedding,
    build_unet,
    build_vae,
    unet_condition_dimension,
)


def main() -> None:
    torch.manual_seed(7)
    vae = build_vae(
        {"block_out_channels": [32, 64], "layers_per_block": 1, "latent_channels": 4},
        sample_size=64,
    )
    images = torch.randn(2, 3, 64, 64).clamp(-1, 1)
    posterior = vae.encode(images).latent_dist
    latents = posterior.sample()
    reconstructed = vae.decode(latents).sample
    assert reconstructed.shape == images.shape

    unet = build_unet(
        {"block_out_channels": [32, 64], "layers_per_block": 1},
        latent_channels=4,
        sample_size=32,
    )
    condition_encoder = ContinuousConditionEmbedding(6, 32, unet_condition_dimension(unet))
    condition = torch.randn(2, 6)
    class_embedding = condition_encoder(condition)
    scheduler = DDPMScheduler(num_train_timesteps=20)
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, 20, (2,))
    noisy = scheduler.add_noise(latents, noise, timesteps)
    prediction = unet(noisy, timesteps, class_labels=class_embedding).sample
    assert prediction.shape == noise.shape

    with tempfile.TemporaryDirectory() as directory:
        condition_encoder.save_pretrained(Path(directory))
        restored = ContinuousConditionEmbedding.from_pretrained(Path(directory))
        assert torch.allclose(restored(condition), class_embedding)
    print("Smoke test passed: VAE, continuous conditioner, and conditioned U-Net forward pass.")


if __name__ == "__main__":
    main()

