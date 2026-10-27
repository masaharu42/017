import json

import numpy as np
from PIL import Image
import pytest
import torch

from ebsd_diffusion.data import EbsdPatchDataset
from ebsd_diffusion.descriptor_conditioning import DescriptorConditionEmbedding, FixedDescriptorCondition
from ebsd_diffusion.local_descriptors import DEFAULTS, NAMES, SCHEMA, LocalDescriptorStore, describe_rgb
from ebsd_diffusion.models import ContinuousConditionEmbedding


def grid(size=24):
    rgb = np.full((size, size, 3), [140, 180, 210], np.uint8)
    rgb[3::4] = 0; rgb[:, 3::4] = 0
    return rgb


def test_small_projection_preserves_rng_and_base_then_learns(tmp_path):
    base = ContinuousConditionEmbedding(6, 128, 512)
    before = torch.get_rng_state().clone()
    model = DescriptorConditionEmbedding(base)
    assert torch.equal(before, torch.get_rng_state())
    assert sum(p.numel() for p in model.projection.parameters()) == 17088
    c, d = torch.randn(3, 6), torch.randn(3, 5)
    assert torch.equal(model(c, descriptors=d), base(c))
    loss = model(c, descriptors=d).square().mean()
    loss.backward()
    assert model.projection[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.projection[-1].weight.fill_(.1)
    assert not torch.equal(model(c, descriptors=d), base(c))
    assert torch.equal(model(c, descriptors=d, descriptor_valid=torch.zeros(3, dtype=torch.bool)), base(c))
    dropped = model(c, descriptors=d, drop_mask=torch.ones(3, dtype=torch.bool))
    assert torch.equal(dropped, base.unconditional(3))
    model.save_pretrained(tmp_path / 'encoder')
    loaded = DescriptorConditionEmbedding.from_pretrained(tmp_path / 'encoder')
    assert torch.equal(model(c, descriptors=d), loaded(c, descriptors=d))
    bound = FixedDescriptorCondition(loaded, d[0])
    assert torch.equal(bound(c), loaded(c, descriptors=d[0].expand(3, -1)))


def test_native_window_statistics_and_invalid_are_explicit():
    settings = {**DEFAULTS, 'min_region_area_px': 1, 'boundary_dilation_px': 0, 'coarse_diameter_px': 3.}
    rgb = grid()
    values, valid, qa = describe_rgb(rgb, settings)
    assert valid and qa['largest_region_fraction'] < .1
    assert np.isfinite(values).all() and values[0] > 0 and 0 <= values[3] <= 1
    rotated, valid, _ = describe_rgb(np.rot90(rgb), settings)
    # Nearest-boundary tie-breaking on a pixel lattice is not exactly rotation
    # invariant; labels must therefore be measured AFTER the actual augmentation.
    assert valid and np.isfinite(rotated).all() and np.isclose(values[0], rotated[0])
    missing, valid, _ = describe_rgb(np.full((24, 24, 3), 180, np.uint8), settings)
    assert not valid and np.isnan(missing).all()


def test_dataset_metadata_does_not_change_crop_augmentation_or_rng(tmp_path):
    import pandas as pd
    rgb = grid(32)
    image = tmp_path / 'source.png'; Image.fromarray(rgb).save(image)
    manifest = tmp_path / 'manifest.csv'
    pd.DataFrame([{'alloy_id': 5, 'image_path': str(image), 'use_for_diffusion': True,
                   'use_for_vae': True}]).to_csv(manifest, index=False)
    np.save(tmp_path / 'pixels.npy', rgb)
    metadata = {'schema': SCHEMA, 'names': NAMES, 'settings': {
        **DEFAULTS, 'min_region_area_px': 1, 'boundary_dilation_px': 0},
        'normalizer': {'mean': [0]*5, 'std': [1]*5}, 'training_alloy_ids': [5],
        'images': [{'image_path': str(image), 'pixels': 'pixels.npy'}]}
    p = tmp_path / 'metadata.json'; p.write_text(json.dumps(metadata))
    common = dict(manifest_path=manifest, patch_size=12, patches_per_image=2, target='diffusion',
        conditions={5: np.zeros(6)}, crop_margins=(1,2,3,4), horizontal_flip=True,
        vertical_flip=True, rotate_180=True)
    original = EbsdPatchDataset(**common)
    enhanced = EbsdPatchDataset(**common, descriptor_store=LocalDescriptorStore(p))
    torch.manual_seed(17); a = original[0]; after_a = torch.get_rng_state()
    torch.manual_seed(17); b = enhanced[0]; after_b = torch.get_rng_state()
    assert torch.equal(a['pixel_values'], b['pixel_values'])
    assert torch.equal(after_a, after_b)
    assert b['native_box'][0] >= 4 and b['native_box'][1] >= 1
    actual_rgb = np.rint((b['pixel_values'].numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
    expected, valid = LocalDescriptorStore(p).lookup(image, b['native_box'].tolist(), rgb=actual_rgb)
    assert np.array_equal(expected, b['descriptors'].numpy()) and bool(b['descriptor_valid']) == valid


def test_reject_nonfinite_or_wrong_descriptor_input():
    model = DescriptorConditionEmbedding(ContinuousConditionEmbedding(2, 8, 32))
    with pytest.raises(ValueError, match='Five normalized'):
        model(torch.ones(1, 2))
    with pytest.raises(ValueError, match='Non-finite'):
        model(torch.ones(1, 2), descriptors=torch.full((1, 5), float('nan')))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA RNG check requires server GPU')
def test_constructor_also_preserves_cuda_rng():
    before = torch.cuda.get_rng_state_all()
    DescriptorConditionEmbedding(ContinuousConditionEmbedding(6, 128, 512))
    assert all(torch.equal(a, b) for a, b in zip(before, torch.cuda.get_rng_state_all()))
