import pytest
import torch
import torch.nn.functional as F

from ebsd_diffusion.losses import diffusion_mse_loss, min_snr_weights, validate_snr_gamma


def test_disabled_path_is_exact_historical_mse():
    prediction = torch.randn(3, 4, 8, 8, requires_grad=True)
    target = torch.randn_like(prediction)
    loss = diffusion_mse_loss(prediction, target, torch.arange(3), torch.ones(3))
    assert torch.equal(loss, F.mse_loss(prediction, target))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_epsilon_weights_and_endpoint_limits():
    alpha = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
    actual = min_snr_weights(alpha, torch.arange(5), 5.0)
    assert torch.allclose(actual, torch.tensor([1.0, 1.0, 1.0, 5 / 9, 0.0]))
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()


def test_weights_reduce_per_sample_not_per_batch_snr():
    prediction = torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1).requires_grad_()
    target = torch.zeros_like(prediction)
    alpha = torch.tensor([0.99, 0.5])
    loss = diffusion_mse_loss(prediction, target, torch.arange(2), alpha, snr_gamma=5.0)
    expected = (5 / 99 + 4) / 2
    assert loss.item() == pytest.approx(expected, rel=1e-6)
    loss.backward()
    assert torch.allclose(prediction.grad.flatten(), torch.tensor([5 / 99, 2.0]))


@pytest.mark.parametrize("gamma", [0, -1, float("nan"), float("inf")])
def test_invalid_gamma_is_rejected(gamma):
    with pytest.raises(ValueError, match="snr_gamma"):
        validate_snr_gamma(gamma, "epsilon")


def test_min_snr_does_not_silently_apply_epsilon_formula_to_velocity():
    with pytest.raises(ValueError, match="epsilon"):
        validate_snr_gamma(5.0, "v_prediction")
    validate_snr_gamma(None, "v_prediction")


def test_half_precision_loss_is_accumulated_in_float32():
    loss = diffusion_mse_loss(
        torch.full((2, 4, 2, 2), 0.1, dtype=torch.float16),
        torch.zeros(2, 4, 2, 2, dtype=torch.float16),
        torch.tensor([0, 1]), torch.tensor([0.9, 0.1]), snr_gamma=5.0,
    )
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
