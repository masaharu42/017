"""Versioned, observable 256-window RGB proxies, not physical EBSD grain labels.

Segment the native raster BEFORE any resizing. No Lanczos/threshold round trip.
The five statistics describe connected pieces visible inside a window; they do
not use unobservable mother-grain sizes outside a generated image.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops


NAMES = ["visible_D50_px", "visible_log_spread", "visible_area_weighted_aspect",
         "visible_coarse_area_fraction_20px", "visible_boundary_length_density_per_px"]
SCHEMA = "native_rgb_visible_window_v1"
DEFAULTS = {"black_threshold": 51, "boundary_dilation_px": 1,
            "min_region_area_px": 10, "coarse_diameter_px": 20.0,
            "min_regions": 3}


def validate_settings(settings: dict) -> dict:
    if set(settings) - set(DEFAULTS):
        raise ValueError(f"Unknown descriptor settings: {sorted(set(settings) - set(DEFAULTS))}")
    s = {**DEFAULTS, **settings}
    for k in ("black_threshold", "boundary_dilation_px", "min_region_area_px", "min_regions"):
        if isinstance(s[k], bool) or not isinstance(s[k], int):
            raise ValueError(f"{k} must be an integer")
    if not 0 < s["black_threshold"] < 256 or not 0 <= s["boundary_dilation_px"] <= 4:
        raise ValueError("Invalid threshold or boundary dilation")
    if s["min_region_area_px"] < 1 or s["min_regions"] < 3:
        raise ValueError("Require positive region area and at least three regions")
    if not np.isfinite(s["coarse_diameter_px"]) or s["coarse_diameter_px"] <= 0:
        raise ValueError("Invalid coarse diameter")
    return s


def segment_native(rgb: np.ndarray, settings: dict) -> tuple[np.ndarray, dict]:
    s = validate_settings(settings)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected native uint8 HWC RGB")
    gb = rgb.max(axis=2) < s["black_threshold"]
    radius = s["boundary_dilation_px"]
    if radius:
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        gb = ndimage.binary_dilation(gb, structure=xx * xx + yy * yy <= radius * radius)
    regions, count = ndimage.label(~gb)
    if count == 0:
        return regions.astype(np.int32), {"region_count": 0, "largest_region_fraction": 0.0}
    if (regions == 0).any():
        nearest = ndimage.distance_transform_edt(regions == 0, return_distances=False, return_indices=True)
        regions = regions[tuple(nearest)]
    areas = np.bincount(regions.ravel())[1:]
    return regions.astype(np.int32), {"region_count": int(count),
        "largest_region_fraction": float(areas.max() / regions.size)}


def window_statistics(regions: np.ndarray, settings: dict) -> tuple[np.ndarray, bool]:
    s = validate_settings(settings)
    if regions.ndim != 2 or not regions.size:
        raise ValueError("Expected nonempty 2D labels")
    # Split pieces that share a parent ID but are disconnected inside this crop.
    visible = label(regions, background=0, connectivity=1)
    props = [p for p in regionprops(visible) if p.area >= s["min_region_area_px"]]
    if len(props) < s["min_regions"]:
        return np.full(5, np.nan, dtype=np.float64), False
    area = np.asarray([p.area for p in props], dtype=np.float64)
    diameter = np.sqrt(4 * area / np.pi)
    aspect = []
    for p in props:
        eigen = np.clip(p.inertia_tensor_eigvals, 0, None) + 1 / 12
        aspect.append(np.sqrt(eigen.max() / eigen.min()))
    d10, d50, d90 = np.percentile(diameter, [10, 50, 90])
    retained = np.isin(visible, [p.label for p in props])
    horizontal = (visible[:, 1:] != visible[:, :-1]) & retained[:, 1:] & retained[:, :-1]
    vertical = (visible[1:] != visible[:-1]) & retained[1:] & retained[:-1]
    length = (horizontal.sum() + vertical.sum()) * np.pi / 4
    result = np.asarray([d50, np.log(d90 / d10) / 2.5631,
                         np.average(aspect, weights=area),
                         area[diameter >= s["coarse_diameter_px"]].sum() / area.sum(),
                         length / area.sum()], dtype=np.float64)
    return result, bool(np.isfinite(result).all())


def describe_rgb(rgb: np.ndarray, settings: dict) -> tuple[np.ndarray, bool, dict]:
    regions, quality = segment_native(rgb, settings)
    values, valid = window_statistics(regions, settings)
    return values, valid, quality


class LocalDescriptorStore:
    """Native RGB cache; measure the actual augmented window without RNG/filtering."""
    def __init__(self, metadata_path: str | Path):
        self.path = Path(metadata_path).resolve()
        self.metadata = json.loads(self.path.read_text("utf-8"))
        if self.metadata["schema"] != SCHEMA or self.metadata["names"] != NAMES:
            raise ValueError("Descriptor schema/column order mismatch")
        self.settings = validate_settings(self.metadata["settings"])
        self.entries = {str(Path(r["image_path"]).resolve()): r for r in self.metadata["images"]}
        self.mean = np.asarray(self.metadata["normalizer"]["mean"], dtype=np.float64)
        self.std = np.asarray(self.metadata["normalizer"]["std"], dtype=np.float64)
        if self.mean.shape != (5,) or self.std.shape != (5,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Invalid training-only normalizer")
        self.cache = {}

    def lookup(self, image_path, box, rgb=None) -> tuple[np.ndarray, bool]:
        key = str(Path(image_path).resolve())
        entry = self.entries[key]
        if key not in self.cache:
            self.cache[key] = np.load(self.path.parent / entry["pixels"], allow_pickle=False)
        pixels = self.cache[key]
        x0, y0, x1, y1 = map(int, box)
        if not (0 <= x0 < x1 <= pixels.shape[1] and 0 <= y0 < y1 <= pixels.shape[0]):
            raise ValueError("Descriptor crop is outside native image")
        # Exactly the same observable-window algorithm as generated RGB scoring.
        crop = pixels[y0:y1, x0:x1] if rgb is None else rgb
        if crop.shape != (y1 - y0, x1 - x0, 3):
            raise ValueError("Descriptor pixels do not match the crop geometry")
        raw, valid, _ = describe_rgb(crop, self.settings)
        normalized = (raw - self.mean) / self.std if valid else np.zeros(5)
        return normalized.astype(np.float32), valid
