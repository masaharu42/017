import torch
from torch import nn

from ebsd_diffusion.weight_loading import apply_ema_shadow_, max_abs_difference
from scripts.sample_fold03_ab2c_256 import replication_pad


def test_apply_ema_shadow_replaces_trainable_parameters() -> None:
    module = nn.Linear(2, 1, bias=True)
    shadow = [torch.full_like(module.weight, 3.0), torch.full_like(module.bias, -1.0)]

    apply_ema_shadow_(module, shadow)

    assert torch.equal(module.weight, shadow[0])
    assert torch.equal(module.bias, shadow[1])


def test_apply_ema_shadow_rejects_a_shape_mismatch() -> None:
    module = nn.Linear(2, 1, bias=False)
    try:
        apply_ema_shadow_(module, [torch.zeros(1, 3)])
    except RuntimeError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("shape mismatch was accepted")


def test_original_size_crop_is_padded_to_a_vae_multiple() -> None:
    top, bottom, left, right = replication_pad(450, 750, 8)
    assert (450 + top + bottom) % 8 == 0
    assert (750 + left + right) % 8 == 0
    assert replication_pad(256, 256, 8) == (0, 0, 0, 0)


def test_max_abs_difference_is_zero_for_identical_states() -> None:
    state = {"weight": torch.tensor([1.0, 2.0])}
    assert max_abs_difference(state, state) == 0.0
