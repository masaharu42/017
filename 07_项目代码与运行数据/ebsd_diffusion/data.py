from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from ebsd_diffusion.euler_images import (
    assert_no_geometric_augmentation,
    encode_rgb_image,
    file_sha256,
)


def resolve_manifest_image_path(manifest_path: str | Path, value: str | Path) -> Path:
    """Resolve image paths stored either relative to the manifest or project root."""
    raw = Path(value)
    if raw.is_absolute():
        candidate = raw.resolve()
        if candidate.is_file():
            return candidate
    manifest = Path(manifest_path).resolve()
    candidates = [manifest.parent / raw, manifest.parent.parent / raw]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    return candidates[0].resolve()


def load_manifest_frame(manifest_path: str | Path) -> pd.DataFrame:
    """CSV manifests stay as they are. A JSON manifest is the Euler+GB list."""
    path = Path(manifest_path)
    if path.suffix.lower() != ".json":
        return pd.read_csv(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    rows = []
    for record in records:
        image = record.get("image_path", record.get("image"))
        if not image:
            raise ValueError(f"JSON manifest record has no image: {record!r}")
        rows.append(
            {
                "alloy_id": int(record["alloy_id"]),
                "image_path": image,
                "use_for_vae": True,
                "use_for_diffusion": True,
                "sha256": record.get("sha256", ""),
            }
        )
    if not rows:
        raise ValueError(f"JSON manifest has no records: {path}")
    return pd.DataFrame(rows)


def resolve_exclude_alloy_id(data_cfg: Mapping[str, object], cli_value: int | None) -> int | None:
    configured = data_cfg.get("exclude_alloy_id")
    if cli_value is None and configured is None:
        return None
    if cli_value is None:
        return int(configured)
    if configured is None:
        return int(cli_value)
    if int(cli_value) != int(configured):
        raise ValueError("CLI --exclude-alloy-id disagrees with data.exclude_alloy_id.")
    return int(cli_value)


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
        descriptor_store=None,
        image_mode: str = "rgb_ipf",
        euler_gb: dict | None = None,
    ) -> None:
        if target not in {"vae", "diffusion", "evaluation"}:
            raise ValueError(f"Unsupported dataset target: {target}")
        if image_mode not in {"rgb_ipf", "euler_gb"}:
            raise ValueError("data.image_mode must be 'rgb_ipf' or 'euler_gb'.")
        assert_no_geometric_augmentation(image_mode, horizontal_flip, vertical_flip, rotate_180)
        if image_mode == "euler_gb" and not euler_gb:
            raise ValueError("euler_gb settings are required when data.image_mode is euler_gb.")
        if image_mode == "euler_gb" and descriptor_store is not None:
            raise ValueError("Local RGB descriptors are not defined for Euler+GB channels.")

        manifest_path = Path(manifest_path).resolve()
        frame = load_manifest_frame(manifest_path)
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

        rows: list[dict[str, object]] = []
        for record in frame.to_dict("records"):
            image_path = resolve_manifest_image_path(manifest_path, record["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(f"EBSD image does not exist: {image_path}")
            expected_hash = str(record.get("sha256") or "")
            if expected_hash and file_sha256(image_path) != expected_hash:
                raise ValueError(f"Image bytes differ from the manifest sha256: {image_path}")
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
        self.descriptor_store = descriptor_store
        self.image_mode = image_mode
        self.euler_gb = euler_gb
        self._euler_cache: dict[str, np.ndarray] = {}

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
        if self.image_mode == "euler_gb":
            encoded = self._euler_channels(image_path, image)
            left, top, right, bottom = self._sample_box(image.size[0], image.size[1], index)
            patch_box = (left, top, right, bottom)
            tensor = torch.from_numpy(np.ascontiguousarray(encoded[:, top:bottom, left:right]))
        else:
            image, patch_box = self._patch_with_box(image, index)

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
            if self.descriptor_store is not None:
                top, _, _, left = self.crop_margins
                x0, y0, x1, y1 = patch_box
                native_box = (x0 + left, y0 + top, x1 + left, y1 + top)
                values, valid = self.descriptor_store.lookup(image_path, native_box, rgb=np.asarray(image))
                result["descriptors"] = torch.as_tensor(values, dtype=torch.float32)
                result["descriptor_valid"] = torch.tensor(valid, dtype=torch.bool)
                result["native_box"] = torch.tensor(native_box, dtype=torch.long)
        return result

    def _content_crop(self, image: Image.Image) -> Image.Image:
        top, right, bottom, left = self.crop_margins
        width, height = image.size
        if top + bottom >= height or left + right >= width:
            raise ValueError(f"Crop margins {self.crop_margins} remove the entire image {image.size}.")
        return image.crop((left, top, width - right, height - bottom))

    def _patch_crop(self, image: Image.Image, index: int) -> Image.Image:
        return self._patch_with_box(image, index)[0]

    def _euler_channels(self, image_path: Path, image: Image.Image) -> np.ndarray:
        key = f"{image_path}|{self.crop_margins}"
        cached = self._euler_cache.get(key)
        if cached is None:
            cached = encode_rgb_image(np.asarray(image, dtype=np.uint8), self.euler_gb)
            self._euler_cache[key] = cached
        return cached

    def _sample_box(self, width: int, height: int, index: int) -> tuple[int, int, int, int]:
        del index
        if width < self.patch_size or height < self.patch_size:
            raise ValueError(
                f"Image {(width, height)} is smaller than patch_size={self.patch_size}."
            )
        if self.deterministic:
            left = (width - self.patch_size) // 2
            top = (height - self.patch_size) // 2
        else:
            left = int(torch.randint(0, width - self.patch_size + 1, ()).item())
            top = int(torch.randint(0, height - self.patch_size + 1, ()).item())
        return (left, top, left + self.patch_size, top + self.patch_size)

    def _patch_with_box(self, image: Image.Image, index: int):
        box = self._sample_box(image.size[0], image.size[1], index)
        return image.crop(box), box


def build_condition_mapping(composition_frame, conditioner, alloy_ids) -> dict[int, np.ndarray]:
    mapping: dict[int, np.ndarray] = {}
    for alloy_id in sorted({int(value) for value in alloy_ids}):
        mapping[alloy_id] = conditioner.transform_by_alloy_id(composition_frame, alloy_id)
    return mapping


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

