from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from ebsd_diffusion.euler_images import (
    assert_no_geometric_augmentation,
    channel_count,
    circular_difference,
    decode_orientation_channels,
    default_euler_gb_config,
    encode_orientation_degrees,
    encode_rgb_image,
    fill_from_neighbors,
    rgb_to_euler_degrees,
    shared_sample_seeds,
    thin_boundary_mask,
)
from ebsd_diffusion.orientation_metrics import (
    bunge_matrices,
    cubic_symmetry_matrices,
    grain_report,
    misorientation_degrees,
    misorientation_histogram_distance,
    nearest_training_window,
    reconstruction_report,
)
from scripts.check_bmp_euler_against_ctf import compare_bmp_with_ctf, find_ctf_files, main as ctf_main

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
BMP = Path("/Users/masaharu/Desktop/1.材料开发/data/data/images/01.bmp")
FULL_CONFIGS = [
    "euler_gb_fold19_wide.yaml",
    "euler_gb_fold19_narrow.yaml",
    "euler_gb_fold14_wide.yaml",
    "euler_gb_fold14_narrow.yaml",
    "euler_gb_fold03_wide.yaml",
    "euler_gb_fold03_narrow.yaml",
]


def _angles() -> np.ndarray:
    generator = np.random.default_rng(0)
    return np.stack(
        [
            generator.uniform(0.0, 360.0, (16, 16)),
            generator.uniform(0.0, 90.0, (16, 16)),
            generator.uniform(0.0, 90.0, (16, 16)),
        ],
        axis=-1,
    )


def test_reduced_zone_roundtrip_and_period_boundaries():
    config = default_euler_gb_config()
    original = _angles()
    decoded = decode_orientation_channels(encode_orientation_degrees(original, config), config)
    assert circular_difference(decoded[..., 0], original[..., 0], 360.0).max() < 1.0e-3
    assert np.abs(decoded[..., 1] - original[..., 1]).max() < 1.0e-3
    assert circular_difference(decoded[..., 2], original[..., 2], 90.0).max() < 1.0e-3

    zero = np.zeros((1, 1, 3), dtype=np.float64)
    full_turn = zero.copy()
    full_turn[..., 0] = 360.0
    phi2_period = zero.copy()
    phi2_period[..., 2] = 90.0
    assert np.allclose(encode_orientation_degrees(zero, config), encode_orientation_degrees(full_turn, config))
    assert np.allclose(encode_orientation_degrees(zero, config), encode_orientation_degrees(phi2_period, config))
    recovered = decode_orientation_channels(encode_orientation_degrees(phi2_period, config), config)
    assert float(np.max(recovered[..., 2])) < 1.0e-4
    assert float(np.max(decode_orientation_channels(encode_orientation_degrees(full_turn, config), config)[..., 0])) < 1.0e-4

    linear = dict(config)
    linear["phi_encoding"] = "linear"
    assert channel_count("linear") == 6
    decoded_linear = decode_orientation_channels(encode_orientation_degrees(original, linear), linear)
    assert np.abs(decoded_linear[..., 1] - original[..., 1]).max() < 1.0e-3


def test_boundary_keeps_a_line_and_not_a_dark_grain():
    config = default_euler_gb_config()
    line = np.full((32, 32, 3), 180, dtype=np.uint8)
    line[16, :, :] = 0
    line_mask = thin_boundary_mask(line, config)
    assert line_mask[16].mean() > 0.8
    assert line_mask.sum() == 32

    square = np.full((32, 32, 3), 180, dtype=np.uint8)
    square[6:26, 6:26] = 0
    square_mask = thin_boundary_mask(square, config)
    assert not square_mask[16, 16]
    assert square_mask.sum() < 20

    filled = fill_from_neighbors(line, line_mask)
    assert filled[16, 8].max() > 0
    assert np.array_equal(filled[~line_mask], line[~line_mask])
    encoded = encode_rgb_image(line, config)
    assert encoded.shape[0] == 7
    assert set(np.unique(encoded[-1]).tolist()) <= {0.0, 1.0}


def test_geometric_augmentation_is_refused_for_euler():
    with pytest.raises(ValueError, match="disabled"):
        assert_no_geometric_augmentation("euler_gb", True, False, False)
    assert_no_geometric_augmentation("rgb_ipf", True, True, True)


def test_cubic_grain_metrics_and_copy_search():
    assert cubic_symmetry_matrices().shape == (24, 3, 3)
    identity = bunge_matrices(0.0, 0.0, 0.0)
    turned = bunge_matrices(np.deg2rad(30.0), 0.0, 0.0)
    assert abs(float(misorientation_degrees(identity, turned)) - 30.0) < 1.0e-4
    assert float(misorientation_degrees(identity, identity)) < 1.0e-5

    uniform = np.zeros((8, 8, 3), dtype=np.float64)
    assert grain_report(uniform, min_grain_pixels=1)["grain_count"] == 1
    split = uniform.copy()
    split[:, 4:, 0] = 40.0
    assert grain_report(split, min_grain_pixels=1)["grain_count"] == 2
    assert misorientation_histogram_distance(uniform, uniform)["jensen_shannon"] < 1.0e-8

    boundary = np.zeros((8, 8), dtype=np.float64)
    query = split[:4, :4]
    found = nearest_training_window(
        query,
        boundary[:4, :4],
        [
            {"alloy_id": 5, "euler_degrees": split, "boundary": boundary},
            {"alloy_id": 9, "euler_degrees": np.full_like(split, 50.0), "boundary": boundary},
        ],
        stride=4,
    )
    assert found["orientation"]["alloy_id"] == 5
    assert found["orientation"]["distance_degrees"] < 1.0e-4
    report = reconstruction_report(split, boundary, split, boundary, threshold_degrees=15.0)
    assert report["gb_dice"] == 1.0
    assert report["grain_count_difference"] == 0


def test_missing_ctf_is_an_error_and_a_matching_pair_is_measured(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="cannot be skipped"):
        find_ctf_files(tmp_path)
    bmp = np.full((8, 8, 3), (200, 100, 50), dtype=np.uint8)
    image_path = tmp_path / "sample.bmp"
    Image.fromarray(bmp, mode="RGB").save(image_path)
    expected = rgb_to_euler_degrees(bmp, default_euler_gb_config())
    ctf_path = tmp_path / "sample.ctf"
    rows = ["Channel Text File", "XCells\t8", "YCells\t8", "XStep\t1", "YStep\t1", "Phase\tX\tY\tEuler1\tEuler2\tEuler3"]
    for y in range(8):
        for x in range(8):
            angle = expected[y, x]
            rows.append(f"1\t{x:.1f}\t{y:.1f}\t{angle[0]:.6f}\t{angle[1]:.6f}\t{angle[2]:.6f}")
    ctf_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    report = compare_bmp_with_ctf(image_path, ctf_path)
    assert report["direct_top_left_overlap"]["phi1"]["median"] < 1.0e-4
    assert ctf_main(["--ctf", str(tmp_path / "absent"), "--bmp", str(image_path)]) == 2


def test_new_configs_share_seeds_and_leave_rgb_config_unchanged():
    base = (CONFIG_DIR / "base.yaml").read_text(encoding="utf-8")
    assert "image_mode" not in base
    assert "euler_gb:" not in base
    loaded = [(name, yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))) for name in FULL_CONFIGS]
    seeds = shared_sample_seeds(loaded[0][1])
    assert len(seeds) == 32
    outputs = set()
    focus = {
        "euler_gb_fold19_wide.yaml": [18, 17, 20, 5],
        "euler_gb_fold19_narrow.yaml": [18, 17, 20, 5],
        "euler_gb_fold14_wide.yaml": [13, 12, 5],
        "euler_gb_fold14_narrow.yaml": [13, 12, 5],
        "euler_gb_fold03_wide.yaml": [5, 7, 9],
        "euler_gb_fold03_narrow.yaml": [5, 7, 9],
    }
    for name, config in loaded:
        assert shared_sample_seeds(config) == seeds
        assert config["data"]["image_mode"] == "euler_gb"
        assert config["data"]["horizontal_flip"] is False
        assert config["euler_gb"]["boundary_euler"] == "fill_neighbors"
        assert config["vae"]["in_channels"] == 7
        assert config["vae"]["out_channels"] == 7
        assert config["train_diffusion"]["lambda_gb"] == 0.0
        assert config["train_diffusion"]["lambda_edge"] == 0.0
        assert config["evaluation"]["focus_alloy_ids"] == focus[name]
        width = config["diffusion"]["block_out_channels"]
        assert width == ([128, 256, 256] if "wide" in name else [64, 128, 128])
        assert config["diffusion"]["layers_per_block"] == 2
        outputs.add(config["output_root"])
        assert "euler_gb" in config["output_root"]
        assert not (ROOT / config["train_vae"]["output_dir"]).exists()
        assert not (ROOT / config["train_diffusion"]["output_dir"]).exists()
    assert len(outputs) == 6
    smoke = yaml.safe_load((CONFIG_DIR / "euler_gb_smoke.yaml").read_text(encoding="utf-8"))
    assert smoke["data"]["patch_size"] == 64
    assert smoke["train_vae"]["max_steps"] == 1
    assert smoke["train_diffusion"]["max_steps"] == 1
    assert smoke["vae"]["in_channels"] == 7


def test_torch_encoding_vae_and_unet_shapes():
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from ebsd_diffusion.euler_loss import euler_gb_vae_loss
    from ebsd_diffusion.models import build_unet, build_vae, parameter_count, unet_condition_dimension
    from physics.orientation_encoding import decode_euler_channels, euler_to_channels

    euler = torch.rand(2, 3, 8, 8) * torch.tensor([6.2832, 1.5708, 1.5708]).view(1, 3, 1, 1)
    unchanged = euler_to_channels(euler, "sincos")
    assert torch.allclose(unchanged, torch.cat((torch.sin(euler), torch.cos(euler)), dim=1))
    encoded = euler_to_channels(euler, "euler_gb", boundary=torch.zeros(2, 8, 8))
    assert encoded.shape == (2, 7, 8, 8)
    recovered = decode_euler_channels(encoded, "sincos")
    phi1_error = torch.remainder(recovered[:, 0] - euler[:, 0] + torch.pi, 2 * torch.pi) - torch.pi
    phi2_error = torch.remainder(recovered[:, 2] - euler[:, 2] + torch.pi / 4, torch.pi / 2) - torch.pi / 4
    assert torch.rad2deg(phi1_error).abs().max() < 1.0e-2
    assert torch.rad2deg(recovered[:, 1] - euler[:, 1]).abs().max() < 1.0e-2
    assert torch.rad2deg(phi2_error).abs().max() < 1.0e-2

    class _Posterior:
        def kl(self):
            return encoded.new_zeros(2)

    target = encoded.detach()
    prediction = target + 0.1 * torch.randn_like(target)
    prediction.requires_grad_(True)
    parts = euler_gb_vae_loss(prediction, target, _Posterior(), default_euler_gb_config(), 1.0e-6)
    parts["loss"].backward()
    assert prediction.grad is not None

    rgb_vae = build_vae({"block_out_channels": [32, 64], "layers_per_block": 1, "latent_channels": 4}, 16)
    assert rgb_vae.config.in_channels == 3 and rgb_vae.config.out_channels == 3
    full = {
        "in_channels": 7,
        "out_channels": 7,
        "block_out_channels": [64, 128, 256],
        "layers_per_block": 2,
        "latent_channels": 4,
    }
    vae = build_vae(full, 256)
    with torch.no_grad():
        latent = vae.encode(torch.randn(1, 7, 256, 256)).latent_dist.mode()
    assert tuple(latent.shape) == (1, 4, 64, 64)

    counts = {}
    for name, widths in {"wide": [128, 256, 256], "narrow": [64, 128, 128]}.items():
        model_cfg = {"block_out_channels": widths, "layers_per_block": 2}
        unet = build_unet(model_cfg, 4, 16)
        condition_dim = unet_condition_dimension(unet)
        assert condition_dim == widths[0] * 4
        assert unet.config.layers_per_block == 2
        assert unet.config.add_attention is False
        output = unet(torch.randn(1, 4, 16, 16), torch.tensor([5]), class_labels=torch.randn(1, condition_dim)).sample
        assert tuple(output.shape) == (1, 4, 16, 16)
        counts[name] = parameter_count(unet)
    assert counts["narrow"] < counts["wide"]


def test_euler_dataset_reads_channels_without_writing_the_bmp(tmp_path: Path):
    pytest.importorskip("torch")
    from ebsd_diffusion.data import EbsdPatchDataset

    rgb = np.full((24, 24, 3), 180, dtype=np.uint8)
    rgb[8, :, :] = 0
    image_path = tmp_path / "01.bmp"
    Image.fromarray(rgb, mode="RGB").save(image_path)
    before = image_path.read_bytes()
    manifest = {
        "records": [{"alloy_id": 1, "image": "01.bmp", "sha256": __import__("hashlib").sha256(before).hexdigest()}]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="disabled"):
        EbsdPatchDataset(
            manifest_path,
            patch_size=8,
            patches_per_image=1,
            target="vae",
            horizontal_flip=True,
            image_mode="euler_gb",
            euler_gb=default_euler_gb_config(),
        )
    dataset = EbsdPatchDataset(
        manifest_path,
        patch_size=8,
        patches_per_image=1,
        target="vae",
        deterministic=True,
        image_mode="euler_gb",
        euler_gb=default_euler_gb_config(),
    )
    sample = dataset[0]["pixel_values"]
    assert tuple(sample.shape) == (7, 8, 8)
    assert image_path.read_bytes() == before


@pytest.mark.skipif(not BMP.is_file(), reason="Euler BMP is not on this machine")
def test_real_bmp_boundary_is_a_thin_network():
    rgb = np.asarray(Image.open(BMP).convert("RGB"), dtype=np.uint8)
    mask = thin_boundary_mask(rgb, default_euler_gb_config())
    fraction = float(mask.mean())
    assert 0.05 < fraction < 0.25


def test_smoke_pipeline_one_step(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("accelerate")
    if not BMP.is_file():
        pytest.skip("Euler BMP is not on this machine")
    import subprocess
    import sys

    vae_dir = tmp_path / "vae"
    diffusion_dir = tmp_path / "diffusion"
    config = CONFIG_DIR / "euler_gb_smoke.yaml"
    vae = subprocess.run(
        [sys.executable, str(ROOT / "train_vae.py"), "--config", str(config), "--output-dir", str(vae_dir), "--max-steps", "1"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert vae.returncode == 0, vae.stderr[-4000:]
    assert "VAE parameters" in vae.stdout
    diffusion = subprocess.run(
        [
            sys.executable,
            str(ROOT / "train_diffusion_scratch.py"),
            "--config",
            str(config),
            "--vae-dir",
            str(vae_dir / "final"),
            "--output-dir",
            str(diffusion_dir),
            "--max-steps",
            "1",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert diffusion.returncode == 0, diffusion.stderr[-4000:]
    assert "U-Net parameters" in diffusion.stdout
    metadata = json.loads((diffusion_dir / "final" / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["image_mode"] == "euler_gb"
    assert metadata["unet_parameter_count"] > 0
    assert metadata["vae_parameter_count"] > 0
    assert not (ROOT / "08_训练与生成结果" / "euler_gb" / "smoke").exists()
