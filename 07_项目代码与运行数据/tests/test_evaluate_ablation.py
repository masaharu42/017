from pathlib import Path

from ebsd_diffusion.metrics import ImageDescriptors
from scripts.evaluate_ablation import extract_sample_index, extract_seed, scalar_summary


def test_extract_seed_and_sample_index() -> None:
    path = Path("seed20261012_n5/alloy_03_seed20261012_004.png")
    assert extract_seed(path) == 20261012
    assert extract_sample_index(path) == 4


def test_scalar_summary_reports_distribution() -> None:
    base = dict(
        grain_count=10,
        grain_area_median_px=20.0,
        grain_area_q25_px=5.0,
        grain_area_q75_px=30.0,
        lab_mean_l=1.0,
        lab_mean_a=2.0,
        lab_mean_b=3.0,
        lab_std_l=4.0,
        lab_std_a=5.0,
        lab_std_b=6.0,
        two_point=[0.0],
        radial_power=[0.0],
        hue_histogram=[1.0],
    )
    first = ImageDescriptors(gb_fraction=0.1, **base)
    second = ImageDescriptors(gb_fraction=0.3, **base)
    result = scalar_summary([first, second])
    assert result["gb_fraction"]["mean"] == 0.2
    assert result["gb_fraction"]["q25"] == 0.15
    assert result["grain_count"]["std"] == 0.0
