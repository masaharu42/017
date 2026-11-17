"""Cubic-symmetry grain and copy metrics for Euler maps.

RGB image descriptors stay in ``ebsd_diffusion.metrics``. These functions take
Euler degrees in HxWx3 and a boundary mask, and they do not read image files.
"""
from __future__ import annotations

from itertools import permutations, product

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import jensenshannon


def cubic_symmetry_matrices() -> np.ndarray:
    matrices = []
    for permutation in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if np.linalg.det(matrix) > 0.5:
                matrices.append(matrix)
    stacked = np.stack(matrices)
    if stacked.shape != (24, 3, 3):
        raise RuntimeError("Cubic symmetry produced an unexpected number of rotations.")
    return stacked


_CUBIC = cubic_symmetry_matrices()


def bunge_matrices(phi1: np.ndarray, cap_phi: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    """Bunge ZXZ matrices. Angles are radians and broadcast together."""
    cosine_1, sine_1 = np.cos(phi1), np.sin(phi1)
    cosine, sine = np.cos(cap_phi), np.sin(cap_phi)
    cosine_2, sine_2 = np.cos(phi2), np.sin(phi2)
    row_0 = np.stack(
        [cosine_1 * cosine_2 - sine_1 * sine_2 * cosine, sine_1 * cosine_2 + cosine_1 * sine_2 * cosine, sine_2 * sine],
        axis=-1,
    )
    row_1 = np.stack(
        [-cosine_1 * sine_2 - sine_1 * cosine_2 * cosine, -sine_1 * sine_2 + cosine_1 * cosine_2 * cosine, cosine_2 * sine],
        axis=-1,
    )
    row_2 = np.stack([sine_1 * sine, -cosine_1 * sine, cosine], axis=-1)
    return np.stack((row_0, row_1, row_2), axis=-2)


def misorientation_degrees(left: np.ndarray, right: np.ndarray, symmetries: np.ndarray | None = None) -> np.ndarray:
    """Smallest cubic misorientation in degrees. The last two axes are 3x3."""
    group = _CUBIC if symmetries is None else symmetries
    transformed = np.einsum("sij,...jk->...sik", group, right)
    mixed = np.einsum("...ij,...sjk->...sik", np.swapaxes(left, -1, -2), transformed)
    trace = mixed[..., 0, 0] + mixed[..., 1, 1] + mixed[..., 2, 2]
    cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine)).min(axis=-1)


def _neighbor_misorientations(euler_degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(np.asarray(euler_degrees, dtype=np.float64))
    if radians.ndim != 3 or radians.shape[-1] != 3:
        raise ValueError("Expected Euler degrees with shape HxWx3.")
    matrices = bunge_matrices(radians[..., 0], radians[..., 1], radians[..., 2])
    right = misorientation_degrees(matrices[:, :-1], matrices[:, 1:])
    down = misorientation_degrees(matrices[:-1, :], matrices[1:, :])
    return right, down


def neighbor_misorientation_samples(euler_degrees: np.ndarray) -> np.ndarray:
    right, down = _neighbor_misorientations(euler_degrees)
    return np.concatenate((right.ravel(), down.ravel()))


def segment_grains(euler_degrees: np.ndarray, threshold_degrees: float = 15.0) -> np.ndarray:
    """Label grains from 4-neighbor misorientation. 0 is not a background class."""
    right, down = _neighbor_misorientations(euler_degrees)
    height, width = euler_degrees.shape[:2]
    count = height * width
    links = []
    right_keep = right <= float(threshold_degrees)
    if np.any(right_keep):
        yy, xx = np.nonzero(right_keep)
        start = yy * width + xx
        links.append((start, start + 1))
    down_keep = down <= float(threshold_degrees)
    if np.any(down_keep):
        yy, xx = np.nonzero(down_keep)
        start = yy * width + xx
        links.append((start, start + width))
    if links:
        left = np.concatenate([item[0] for item in links])
        right_index = np.concatenate([item[1] for item in links])
        graph = csr_matrix(
            (np.ones(left.size * 2, dtype=np.uint8), (np.concatenate((left, right_index)), np.concatenate((right_index, left)))),
            shape=(count, count),
        )
    else:
        graph = csr_matrix((count, count))
    _component_count, labels = connected_components(graph, directed=False, return_labels=True)
    return labels.reshape(height, width)


def grain_report(
    euler_degrees: np.ndarray,
    threshold_degrees: float = 15.0,
    min_grain_pixels: int = 9,
    drawn_boundary: np.ndarray | None = None,
) -> dict[str, object]:
    labels = segment_grains(euler_degrees, threshold_degrees)
    areas = np.bincount(labels.ravel())
    kept = areas[areas >= int(min_grain_pixels)]
    if kept.size == 0:
        kept = np.asarray([0.0])
    right, down = _neighbor_misorientations(euler_degrees)
    boundary = np.zeros(euler_degrees.shape[:2], dtype=bool)
    boundary[:, :-1] |= right > float(threshold_degrees)
    boundary[:, 1:] |= right > float(threshold_degrees)
    boundary[:-1, :] |= down > float(threshold_degrees)
    boundary[1:, :] |= down > float(threshold_degrees)
    edges = np.array([1, 10, 50, 100, 500, 1000, 5000, 10**9], dtype=float)
    histogram, _edges = np.histogram(kept, bins=edges)
    report = {
        "threshold_degrees": float(threshold_degrees),
        "grain_count": int(np.count_nonzero(areas >= int(min_grain_pixels))),
        "area_median_px": float(np.median(kept)),
        "area_q25_px": float(np.quantile(kept, 0.25)),
        "area_q75_px": float(np.quantile(kept, 0.75)),
        "area_histogram_bins_px": edges[:-1].tolist(),
        "area_histogram_counts": histogram.astype(int).tolist(),
        "misorientation_boundary_fraction": float(boundary.mean()),
    }
    if drawn_boundary is not None:
        report["drawn_gb_fraction"] = float(np.asarray(drawn_boundary).astype(bool).mean())
    return report


def misorientation_histogram(euler_degrees: np.ndarray, bins: int = 32, maximum_degrees: float = 62.8) -> np.ndarray:
    samples = neighbor_misorientation_samples(euler_degrees)
    histogram, _edges = np.histogram(samples, bins=bins, range=(0.0, float(maximum_degrees)))
    histogram = histogram.astype(np.float64) + 1.0e-12
    return histogram / histogram.sum()


def misorientation_histogram_distance(original: np.ndarray, reconstructed: np.ndarray, bins: int = 32) -> dict[str, float]:
    left = misorientation_histogram(original, bins=bins)
    right = misorientation_histogram(reconstructed, bins=bins)
    return {
        "jensen_shannon": float(jensenshannon(left, right, base=2.0)),
        "l1": float(np.abs(left - right).sum()),
    }


def binary_dice(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left).astype(bool)
    second = np.asarray(right).astype(bool)
    intersection = np.logical_and(first, second).sum()
    total = first.sum() + second.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def reconstruction_report(
    original_euler: np.ndarray,
    original_boundary: np.ndarray,
    reconstructed_euler: np.ndarray,
    reconstructed_boundary: np.ndarray,
    threshold_degrees: float = 15.0,
    boundary_threshold: float = 0.5,
) -> dict[str, object]:
    original_report = grain_report(original_euler, threshold_degrees, drawn_boundary=original_boundary >= boundary_threshold)
    reconstructed_report = grain_report(
        reconstructed_euler, threshold_degrees, drawn_boundary=reconstructed_boundary >= boundary_threshold
    )
    return {
        "threshold_degrees": float(threshold_degrees),
        "original": original_report,
        "reconstructed": reconstructed_report,
        "grain_count_difference": int(reconstructed_report["grain_count"]) - int(original_report["grain_count"]),
        "gb_dice": binary_dice(original_boundary >= boundary_threshold, reconstructed_boundary >= boundary_threshold),
        "misorientation_distance": misorientation_histogram_distance(original_euler, reconstructed_euler),
    }


def window_distances(query_euler: np.ndarray, query_boundary: np.ndarray, window_euler: np.ndarray, window_boundary: np.ndarray) -> tuple[float, float]:
    if query_euler.shape != window_euler.shape:
        raise ValueError("Orientation windows must have the same shape.")
    radians_query = np.deg2rad(query_euler)
    radians_window = np.deg2rad(window_euler)
    left = bunge_matrices(radians_query[..., 0], radians_query[..., 1], radians_query[..., 2])
    right = bunge_matrices(radians_window[..., 0], radians_window[..., 1], radians_window[..., 2])
    orientation = float(misorientation_degrees(left, right).mean())
    boundary = float(np.mean(np.abs(np.asarray(query_boundary, dtype=np.float64) - np.asarray(window_boundary, dtype=np.float64))))
    return orientation, boundary


def nearest_training_window(
    query_euler: np.ndarray,
    query_boundary: np.ndarray,
    library: list[dict],
    stride: int,
) -> dict[str, object]:
    """Search every library map. Orientation and boundary nearest neighbors are separate."""
    if stride < 1:
        raise ValueError("stride must be positive.")
    if not library:
        raise ValueError("The training window library is empty.")
    best_orientation = None
    best_boundary = None
    query_height, query_width = query_euler.shape[:2]
    for item in library:
        image = np.asarray(item["euler_degrees"])
        boundary = np.asarray(item["boundary"], dtype=np.float64)
        height, width = image.shape[:2]
        if height < query_height or width < query_width:
            continue
        for top in range(0, height - query_height + 1, stride):
            for left in range(0, width - query_width + 1, stride):
                orientation, boundary_distance = window_distances(
                    query_euler,
                    query_boundary,
                    image[top:top + query_height, left:left + query_width],
                    boundary[top:top + query_height, left:left + query_width],
                )
                orientation_hit = {
                    "distance_degrees": orientation,
                    "alloy_id": int(item["alloy_id"]),
                    "top": int(top),
                    "left": int(left),
                }
                boundary_hit = {
                    "distance": boundary_distance,
                    "alloy_id": int(item["alloy_id"]),
                    "top": int(top),
                    "left": int(left),
                }
                if best_orientation is None or orientation_hit["distance_degrees"] < best_orientation["distance_degrees"]:
                    best_orientation = orientation_hit
                if best_boundary is None or boundary_hit["distance"] < best_boundary["distance"]:
                    best_boundary = boundary_hit
    if best_orientation is None or best_boundary is None:
        raise ValueError("No training window was large enough for the query.")
    return {"orientation": best_orientation, "boundary": best_boundary, "stride": int(stride)}
