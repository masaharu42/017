from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class EbsdPatchDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        patch_size: int,
        patches_per_image: int,
        target: str,
        conditions: Mapping[int, np.ndarray] | None = None,
        exclude_alloy_id: int | None = None,
        crop_margins: tuple[int, int, int, int] = (0, 0, 0, 0),
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        rotate_180: bool = False,
        deterministic: bool = False,
    ) -> None:
        if target not in {"vae", "diffusion", "evaluation"}:
            raise ValueError(f"Unsupported dataset target: {target}")

        manifest_path = Path(manifest_path).resolve()
        frame = pd.read_csv(manifest_path)
        required = {"alloy_id", "image_path", "use_for_vae", "use_for_diffusion"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        frame["alloy_id"] = frame["alloy_id"].astype(int)
        if exclude_alloy_id is not None:
            frame = frame.loc[frame["alloy_id"] != int(exclude_alloy_id)]
        if target == "vae":
            frame = frame.loc[frame["use_for_vae"].map(_as_bool)]
        elif target == "diffusion":
            frame = frame.loc[frame["use_for_diffusion"].map(_as_bool)]

        root = manifest_path.parent
        rows: list[dict[str, object]] = []
        for record in frame.to_dict("records"):
            image_path = Path(str(record["image_path"]))
            if not image_path.is_absolute():
                image_path = (root / image_path).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"EBSD image does not exist: {image_path}")
            record["image_path"] = image_path
            rows.append(record)

        if not rows:
            raise ValueError(f"No images remain for target={target!r}.")
        if target == "diffusion" and not conditions:
            raise ValueError("Diffusion dataset requires normalized composition conditions.")

        self.rows = rows
        self.patch_size = int(patch_size)
        self.patches_per_image = max(1, int(patches_per_image))
        self.target = target
        self.conditions = conditions or {}
        self.crop_margins = tuple(int(x) for x in crop_margins)
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotate_180 = rotate_180
        self.deterministic = deterministic

    @property
    def alloy_ids(self) -> list[int]:
        return sorted({int(row["alloy_id"]) for row in self.rows})

    def __len__(self) -> int:
        return len(self.rows) * self.patches_per_image

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        row = self.rows[index % len(self.rows)]
        alloy_id = int(row["alloy_id"])
        image_path = Path(row["image_path"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image = self._content_crop(image)
        image = self._patch_crop(image, index)

        if not self.deterministic:
            if self.horizontal_flip and torch.rand(()) < 0.5:
                image = TF.hflip(image)
            if self.vertical_flip and torch.rand(()) < 0.5:
                image = TF.vflip(image)
            if self.rotate_180 and torch.rand(()) < 0.5:
                image = TF.rotate(image, 180)

        tensor = TF.to_tensor(image) * 2.0 - 1.0
        result: dict[str, torch.Tensor | int | str] = {
            "pixel_values": tensor,
            "alloy_id": alloy_id,
            "image_path": str(image_path),
        }
        if self.target == "diffusion":
            if alloy_id not in self.conditions:
                raise KeyError(f"No normalized condition for alloy {alloy_id}.")
            result["condition"] = torch.as_tensor(self.conditions[alloy_id], dtype=torch.float32)
        return result

    def _content_crop(self, image: Image.Image) -> Image.Image:
        top, right, bottom, left = self.crop_margins
        width, height = image.size
        if top + bottom >= height or left + right >= width:
            raise ValueError(f"Crop margins {self.crop_margins} remove the entire image {image.size}.")
        return image.crop((left, top, width - right, height - bottom))

    def _patch_crop(self, image: Image.Image, index: int) -> Image.Image:
        width, height = image.size
        if width < self.patch_size or height < self.patch_size:
            raise ValueError(
                f"Image {image.size} is smaller than patch_size={self.patch_size}."
            )
        if self.deterministic:
            left = (width - self.patch_size) // 2
            top = (height - self.patch_size) // 2
        else:
            left = int(torch.randint(0, width - self.patch_size + 1, ()).item())
            top = int(torch.randint(0, height - self.patch_size + 1, ()).item())
        return image.crop((left, top, left + self.patch_size, top + self.patch_size))


def build_condition_mapping(composition_frame, conditioner, alloy_ids) -> dict[int, np.ndarray]:
    mapping: dict[int, np.ndarray] = {}
    for alloy_id in sorted({int(value) for value in alloy_ids}):
        mapping[alloy_id] = conditioner.transform_by_alloy_id(composition_frame, alloy_id)
    return mapping


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

