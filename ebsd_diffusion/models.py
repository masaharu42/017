from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers import AutoencoderKL, UNet2DModel
from torch import nn


class ContinuousConditionEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.null_embedding = nn.Parameter(torch.zeros(output_dim))
        nn.init.normal_(self.null_embedding, std=0.02)

    def forward(
        self,
        condition: torch.Tensor,
        drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = self.network(condition)
        if drop_mask is not None:
            null = self.null_embedding.unsqueeze(0).expand_as(embedding)
            embedding = torch.where(drop_mask[:, None], null, embedding)
        return embedding

    def unconditional(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        value = self.null_embedding.to(device=device, dtype=dtype)
        return value.unsqueeze(0).expand(batch_size, -1)

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "pytorch_model.bin")
        config = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
        }
        (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, directory: str | Path, map_location="cpu"):
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        model = cls(**config)
        state = torch.load(directory / "pytorch_model.bin", map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model


class ExponentialMovingAverage:
    def __init__(self, parameters, decay: float = 0.9999):
        self.decay = float(decay)
        self.shadow = [parameter.detach().clone() for parameter in parameters if parameter.requires_grad]
        self.backup: list[torch.Tensor] = []

    @staticmethod
    def _match_parameter_device(value: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        """Move a restored EMA tensor to the parameter's device and dtype.

        Accelerate can restore checkpoint state tensors on CPU even when the
        model has already been prepared on CUDA.  Keeping the conversion here
        makes resumed training safe for both old and new checkpoints.
        """
        if value.device != parameter.device or value.dtype != parameter.dtype:
            return value.to(device=parameter.device, dtype=parameter.dtype)
        return value

    @torch.no_grad()
    def step(self, parameters) -> None:
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        if len(trainable) != len(self.shadow):
            raise RuntimeError("EMA parameter structure changed during training.")
        for index, (shadow, parameter) in enumerate(zip(self.shadow, trainable, strict=True)):
            if shadow.device != parameter.device or shadow.dtype != parameter.dtype:
                shadow = self._match_parameter_device(shadow, parameter)
                self.shadow[index] = shadow
            shadow.lerp_(parameter.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def store(self, parameters) -> None:
        self.backup = [parameter.detach().clone() for parameter in parameters if parameter.requires_grad]

    @torch.no_grad()
    def copy_to(self, parameters) -> None:
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        for parameter, shadow in zip(trainable, self.shadow, strict=True):
            shadow = self._match_parameter_device(shadow, parameter)
            parameter.copy_(shadow)

    @torch.no_grad()
    def restore(self, parameters) -> None:
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        for parameter, backup in zip(trainable, self.backup, strict=True):
            backup = self._match_parameter_device(backup, parameter)
            parameter.copy_(backup)
        self.backup = []

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = float(state_dict["decay"])
        self.shadow = [value.clone() for value in state_dict["shadow"]]



def build_vae(config: dict, sample_size: int) -> AutoencoderKL:
    channels = tuple(int(value) for value in config["block_out_channels"])
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=tuple("DownEncoderBlock2D" for _ in channels),
        up_block_types=tuple("UpDecoderBlock2D" for _ in channels),
        block_out_channels=channels,
        layers_per_block=int(config["layers_per_block"]),
        latent_channels=int(config["latent_channels"]),
        norm_num_groups=32,
        sample_size=int(sample_size),
        scaling_factor=1.0,
    )


def build_unet(config: dict, latent_channels: int, sample_size: int) -> UNet2DModel:
    channels = tuple(int(value) for value in config["block_out_channels"])
    return UNet2DModel(
        sample_size=int(sample_size),
        in_channels=int(latent_channels),
        out_channels=int(latent_channels),
        down_block_types=tuple("DownBlock2D" for _ in channels),
        up_block_types=tuple("UpBlock2D" for _ in channels),
        block_out_channels=channels,
        layers_per_block=int(config["layers_per_block"]),
        norm_num_groups=32,
        add_attention=False,
        class_embed_type="identity",
    )


def unet_condition_dimension(unet: UNet2DModel) -> int:
    return int(unet.config.block_out_channels[0]) * 4


def sobel_magnitude(image: torch.Tensor) -> torch.Tensor:
    channels = image.shape[1]
    kernel_x = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = kernel_x.transpose(0, 1)
    kernel_x = kernel_x.expand(channels, 1, 3, 3)
    kernel_y = kernel_y.expand(channels, 1, 3, 3)
    grad_x = torch.nn.functional.conv2d(image, kernel_x, padding=1, groups=channels)
    grad_y = torch.nn.functional.conv2d(image, kernel_y, padding=1, groups=channels)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-8)
