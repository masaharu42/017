from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial.distance import jensenshannon
from skimage import color, measure, morphology


@dataclass
class ImageDescriptors:
    gb_fraction: float
    grain_count: int
    grain_area_median_px: float
    grain_area_q25_px: float
    grain_area_q75_px: float
    lab_mean_l: float
    lab_mean_a: float
    lab_mean_b: float
    lab_std_l: float
    lab_std_a: float
    lab_std_b: float
    two_point: list[float]
    radial_power: list[float]
    hue_histogram: list[float]
    edge_density: float = 0.0


def describe_image(
    path: str | Path,
    gb_threshold: float = 0.18,
    radial_bins: int = 32,
    crop_size: tuple[int, int] | None = None,
) -> ImageDescriptors:
    image = Image.open(path).convert("RGB")
    if crop_size is not None:
        crop_width, crop_height = crop_size
        if image.width < crop_width or image.height < crop_height:
            raise ValueError(f"Image {path} is smaller than evaluation crop {crop_size}.")
        left = (image.width - crop_width) // 2
        top = (image.height - crop_height) // 2
        image = image.crop((left, top, left + crop_width, top + crop_height))
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    boundary = rgb.max(axis=2) <= gb_threshold
    boundary = morphology.binary_dilation(boundary, morphology.disk(1))
    interior = ~boundary

    labels = measure.label(interior, connectivity=1)
    areas = np.asarray([region.area for region in measure.regionprops(labels)], dtype=float)
    areas = areas[areas >= 9]
    if areas.size == 0:
        areas = np.asarray([0.0])

    lab = color.rgb2lab(rgb)
    pixels = lab[interior]
    if pixels.size == 0:
        pixels = lab.reshape(-1, 3)

    hsv = color.rgb2hsv(rgb)
    hue = hsv[..., 0][interior]
    hue_hist, _ = np.histogram(hue, bins=36, range=(0.0, 1.0), density=False)
    hue_hist = normalize_histogram(hue_hist)

    return ImageDescriptors(
        gb_fraction=float(boundary.mean()),
        grain_count=int(len(areas)),
        grain_area_median_px=float(np.median(areas)),
        grain_area_q25_px=float(np.quantile(areas, 0.25)),
        grain_area_q75_px=float(np.quantile(areas, 0.75)),
        lab_mean_l=float(pixels[:, 0].mean()),
        lab_mean_a=float(pixels[:, 1].mean()),
        lab_mean_b=float(pixels[:, 2].mean()),
        lab_std_l=float(pixels[:, 0].std()),
        lab_std_a=float(pixels[:, 1].std()),
        lab_std_b=float(pixels[:, 2].std()),
        two_point=radial_two_point(interior.astype(np.float32), radial_bins).tolist(),
        radial_power=radial_power_spectrum(boundary.astype(np.float32), radial_bins).tolist(),
        hue_histogram=hue_hist.tolist(),
        edge_density=rgb_edge_density(rgb),
    )


def compare_descriptor_sets(real: list[ImageDescriptors], generated: list[ImageDescriptors]) -> dict:
    if not real or not generated:
        raise ValueError("Both real and generated descriptor lists must be non-empty.")

    scalar_fields = [
        "gb_fraction",
        "grain_count",
        "grain_area_median_px",
        "grain_area_q25_px",
        "grain_area_q75_px",
        "lab_mean_l",
        "lab_mean_a",
        "lab_mean_b",
        "lab_std_l",
        "lab_std_a",
        "lab_std_b",
        "edge_density",
    ]
    result: dict[str, object] = {"real_count": len(real), "generated_count": len(generated)}
    for field in scalar_fields:
        real_values = np.asarray([getattr(item, field) for item in real], dtype=float)
        generated_values = np.asarray([getattr(item, field) for item in generated], dtype=float)
        denominator = max(abs(real_values.mean()), 1.0e-8)
        result[field] = {
            "real_mean": float(real_values.mean()),
            "generated_mean": float(generated_values.mean()),
            "relative_error": float(abs(generated_values.mean() - real_values.mean()) / denominator),
        }

    real_two_point = np.mean([item.two_point for item in real], axis=0)
    gen_two_point = np.mean([item.two_point for item in generated], axis=0)
    real_power = np.mean([item.radial_power for item in real], axis=0)
    gen_power = np.mean([item.radial_power for item in generated], axis=0)
    real_hue = normalize_histogram(np.mean([item.hue_histogram for item in real], axis=0))
    gen_hue = normalize_histogram(np.mean([item.hue_histogram for item in generated], axis=0))
    result["two_point_rmse"] = float(np.sqrt(np.mean((real_two_point - gen_two_point) ** 2)))
    result["radial_power_rmse"] = float(np.sqrt(np.mean((real_power - gen_power) ** 2)))
    result["hue_jensen_shannon"] = float(jensenshannon(real_hue, gen_hue, base=2.0))
    return result


def descriptors_as_dict(value: ImageDescriptors) -> dict:
    return asdict(value)


def rgb_edge_density(rgb: np.ndarray) -> float:
    """Mean Sobel magnitude of an RGB image stored in [0, 1].

    The kernel matches ``ebsd_diffusion.models.sobel_magnitude``. It is a
    cross-correlation, not divided by 8. The outer pixel is excluded so
    zero padding does not dominate the mean.
    """
    array = np.asarray(rgb, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("Expected an RGB image with shape HxWx3.")
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = kernel_x.T
    magnitudes = []
    for channel in range(3):
        gradient_x = ndimage.correlate(array[:, :, channel], kernel_x, mode="constant", cval=0.0)
        gradient_y = ndimage.correlate(array[:, :, channel], kernel_y, mode="constant", cval=0.0)
        magnitude = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y + 1.0e-8)
        magnitudes.append(magnitude[1:-1, 1:-1])
    return float(np.mean(magnitudes))


def radial_two_point(field: np.ndarray, bins: int) -> np.ndarray:
    centered = field - field.mean()
    spectrum = np.fft.fft2(centered)
    correlation = np.fft.fftshift(np.fft.ifft2(spectrum * np.conj(spectrum)).real)
    correlation /= max(field.size, 1)
    return radial_average(correlation, bins)


def radial_power_spectrum(field: np.ndarray, bins: int) -> np.ndarray:
    centered = field - field.mean()
    power = np.fft.fftshift(np.abs(np.fft.fft2(centered)) ** 2)
    profile = radial_average(np.log1p(power), bins)
    norm = np.linalg.norm(profile)
    return profile / norm if norm > 0 else profile


def radial_average(array: np.ndarray, bins: int) -> np.ndarray:
    height, width = array.shape
    yy, xx = np.indices(array.shape)
    radius = np.sqrt((xx - (width - 1) / 2.0) ** 2 + (yy - (height - 1) / 2.0) ** 2)
    edges = np.linspace(0, min(height, width) / 2.0, bins + 1)
    result = np.zeros(bins, dtype=float)
    for index in range(bins):
        mask = (radius >= edges[index]) & (radius < edges[index + 1])
        result[index] = float(array[mask].mean()) if mask.any() else 0.0
    return result


def normalize_histogram(histogram: np.ndarray) -> np.ndarray:
    histogram = np.asarray(histogram, dtype=float) + 1.0e-12
    return histogram / histogram.sum()
