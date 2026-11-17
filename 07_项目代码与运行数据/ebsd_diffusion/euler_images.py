"""Read-only Euler+GB BMP encoding.

Pixel scaling is configured, not fixed in the reader. The current working
assumption is phi1 = R/255*360, Phi = G/255*90, phi2 = B/255*90. A separate
CTF comparison has to confirm it; this module does not treat that assumption
as already verified.

The boundary channel keeps only connected thin dark lines. A dark pixel can
also be a legal orientation, so filled dark regions stay in the orientation
channels.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy import ndimage


def default_euler_gb_config() -> dict:
    return {
        "phi1_degrees": 360.0,
        "Phi_degrees": 90.0,
        "phi2_degrees": 90.0,
        "phi_encoding": "sincos",
        "boundary_threshold": 15,
        "boundary_max_half_width": 2.0,
        "boundary_min_component": 8,
        "boundary_max_local_fraction": 0.45,
        "boundary_local_radius": 3,
        "boundary_euler": "fill_neighbors",
        "gb_loss": "bce",
        "orientation_l1_weight": 1.0,
        "gb_loss_weight": 1.0,
        "gb_edge_weight": 0.20,
    }


def channel_count(phi_encoding: str) -> int:
    if phi_encoding == "sincos":
        return 7
    if phi_encoding == "linear":
        return 6
    raise ValueError("phi_encoding must be 'sincos' or 'linear'")


def assert_no_geometric_augmentation(image_mode: str, horizontal: bool, vertical: bool, rotate: bool) -> None:
    if image_mode != "euler_gb":
        return
    if horizontal or vertical or rotate:
        raise ValueError(
            "Euler+GB flips and rotations are disabled until the Euler angles are transformed with the image."
        )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def thin_boundary_mask(rgb: np.ndarray, config: dict) -> np.ndarray:
    """Connected dark lines only. Thick dark regions are not boundaries."""
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("Expected an RGB image.")
    dark = array[..., :3].max(axis=2) <= int(config["boundary_threshold"])
    if not np.any(dark):
        return np.zeros(dark.shape, dtype=bool)
    distance = ndimage.distance_transform_edt(dark)
    thin = dark & (distance <= float(config["boundary_max_half_width"]))
    radius = int(config["boundary_local_radius"])
    density = ndimage.uniform_filter(dark.astype(np.float32), size=2 * radius + 1, mode="constant")
    line_like = thin & (density <= float(config["boundary_max_local_fraction"]))
    labeled, count = ndimage.label(line_like, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        return np.zeros(dark.shape, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    keep = sizes >= int(config["boundary_min_component"])
    keep[0] = False
    return keep[labeled]


def fill_from_neighbors(rgb: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Replace boundary pixels with the nearest pixel that is not on the boundary."""
    filled = np.array(rgb, copy=True)
    if not np.any(boundary) or not np.any(~boundary):
        return filled
    indices = ndimage.distance_transform_edt(boundary, return_distances=False, return_indices=True)
    filled[boundary] = rgb[indices[0][boundary], indices[1][boundary]]
    return filled


def rgb_to_euler_degrees(rgb: np.ndarray, config: dict) -> np.ndarray:
    scale = np.asarray(rgb[..., :3], dtype=np.float64) / 255.0
    return np.stack(
        [
            scale[..., 0] * float(config["phi1_degrees"]),
            scale[..., 1] * float(config["Phi_degrees"]),
            scale[..., 2] * float(config["phi2_degrees"]),
        ],
        axis=-1,
    )


def encode_orientation_degrees(euler_degrees: np.ndarray, config: dict) -> np.ndarray:
    phi1 = np.deg2rad(euler_degrees[..., 0])
    cap_phi = np.deg2rad(euler_degrees[..., 1])
    phi2 = np.deg2rad(euler_degrees[..., 2])
    sine_1, cosine_1 = np.sin(phi1), np.cos(phi1)
    sine_2, cosine_2 = np.sin(4.0 * phi2), np.cos(4.0 * phi2)
    encoding = str(config["phi_encoding"])
    if encoding == "sincos":
        channels = [sine_1, cosine_1, np.sin(cap_phi), np.cos(cap_phi), sine_2, cosine_2]
    elif encoding == "linear":
        span = float(config["Phi_degrees"])
        if span <= 0:
            raise ValueError("Phi_degrees must be positive.")
        linear = euler_degrees[..., 1] / span * 2.0 - 1.0
        channels = [sine_1, cosine_1, linear, sine_2, cosine_2]
    else:
        raise ValueError("phi_encoding must be 'sincos' or 'linear'")
    return np.stack(channels, axis=-1).astype(np.float32)


def _unit_pair(sine: np.ndarray, cosine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = np.maximum(np.sqrt(sine * sine + cosine * cosine), 1.0e-8)
    return sine / norm, cosine / norm


def decode_orientation_channels(channels: np.ndarray, config: dict) -> np.ndarray:
    """Invert CHW or HWC orientation channels to degrees, shape HxWx3.

    Sin/cos pairs are normalized first. phi2 is recovered on a 90 degree period.
    """
    array = np.asarray(channels, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("Expected CHW or HWC channels.")
    channel_first = array.shape[0] in {5, 6, 7} and array.shape[-1] not in {5, 6, 7}
    hwc = np.moveaxis(array, 0, -1) if channel_first else array
    encoding = str(config["phi_encoding"])
    sine_1, cosine_1 = _unit_pair(hwc[..., 0], hwc[..., 1])
    phi1 = np.rad2deg(np.arctan2(sine_1, cosine_1)) % 360.0
    if encoding == "sincos":
        sine_phi, cosine_phi = _unit_pair(hwc[..., 2], hwc[..., 3])
        cap_phi = np.clip(np.rad2deg(np.arctan2(sine_phi, cosine_phi)), 0.0, float(config["Phi_degrees"]))
        sine_2, cosine_2 = _unit_pair(hwc[..., 4], hwc[..., 5])
    elif encoding == "linear":
        cap_phi = (hwc[..., 2] + 1.0) * 0.5 * float(config["Phi_degrees"])
        sine_2, cosine_2 = _unit_pair(hwc[..., 3], hwc[..., 4])
    else:
        raise ValueError("phi_encoding must be 'sincos' or 'linear'")
    phi2 = (np.rad2deg(np.arctan2(sine_2, cosine_2)) % 360.0) / 4.0
    # A negative epsilon at a period boundary otherwise lands on the period instead of 0.
    phi1 = np.where(phi1 >= 360.0 - 1.0e-6, 0.0, phi1)
    phi2 = np.where(phi2 >= 90.0 - 1.0e-6, 0.0, phi2)
    return np.stack((phi1, cap_phi, phi2), axis=-1)


def encode_rgb_image(rgb: np.ndarray, config: dict) -> np.ndarray:
    """Return float32 CHW channels. The last channel is the boundary mask in {0, 1}."""
    merged = default_euler_gb_config()
    merged.update(config)
    policy = str(merged["boundary_euler"])
    if policy not in {"fill_neighbors", "mask_loss"}:
        raise ValueError("boundary_euler must be 'fill_neighbors' or 'mask_loss'.")
    mask = thin_boundary_mask(rgb, merged)
    source = fill_from_neighbors(rgb, mask) if policy == "fill_neighbors" else np.asarray(rgb)
    orientation = encode_orientation_degrees(rgb_to_euler_degrees(source, merged), merged)
    packed = np.concatenate([orientation, mask.astype(np.float32)[..., None]], axis=-1)
    expected = channel_count(str(merged["phi_encoding"]))
    if packed.shape[-1] != expected:
        raise ValueError(f"Euler+GB encoding produced {packed.shape[-1]} channels, expected {expected}.")
    return np.moveaxis(packed, -1, 0).astype(np.float32, copy=False)


def shared_sample_seeds(config: dict) -> list[int]:
    """The only sample seeds. Nothing in here picks a seed from a held-out image."""
    sampling = config.get("sampling") or {}
    if "select_seed_by_target" in sampling or sampling.get("choose_seed_from_test"):
        raise ValueError("Selecting a sample seed from the held-out image is not available.")
    seeds = sampling.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("sampling.seeds must be the shared fixed list.")
    values = [int(item) for item in seeds]
    if len(set(values)) != len(values):
        raise ValueError("sampling.seeds contains duplicates.")
    return values


def circular_difference(left: np.ndarray, right: np.ndarray, period: float) -> np.ndarray:
    delta = np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)) % period
    return np.minimum(delta, period - delta)
