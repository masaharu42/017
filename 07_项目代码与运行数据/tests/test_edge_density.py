from pathlib import Path

import numpy as np
from PIL import Image

from ebsd_diffusion.metrics import describe_image, rgb_edge_density


def test_step_edge_is_denser_than_a_flat_image() -> None:
    flat = np.full((32, 32, 3), 0.5, dtype=np.float32)
    stepped = flat.copy()
    stepped[:, 16:, :] = 0.0
    assert rgb_edge_density(flat) < 1.0e-3
    assert rgb_edge_density(stepped) > rgb_edge_density(flat)


def test_describe_image_records_edge_density(tmp_path: Path) -> None:
    array = np.full((32, 32, 3), 220, dtype=np.uint8)
    array[:, 15:17, :] = 0
    path = tmp_path / "boundary.png"
    Image.fromarray(array, mode="RGB").save(path)

    descriptor = describe_image(path)

    assert descriptor.edge_density > 0.0
    assert descriptor.gb_fraction > 0.0
