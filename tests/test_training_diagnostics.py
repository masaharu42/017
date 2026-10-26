import math

import pytest
import torch

from ebsd_diffusion.training_diagnostics import TrainingDiagnostics


def test_diagnostics_cover_all_updates_preserve_rng_and_gradients():
    stats = TrainingDiagnostics(1000)
    prediction = torch.tensor([1., 2., 3.]).reshape(3, 1, 1, 1).requires_grad_()
    target = torch.zeros_like(prediction)
    timesteps = torch.tensor([0, 100, 999])
    alphas = torch.full((1000,), 0.5)
    prediction.square().mean().backward()
    gradient = prediction.grad.clone()
    rng = torch.get_rng_state().clone()
    stats.record(prediction, target, timesteps, alphas, None, 2.)
    stats.record(prediction, target, timesteps, alphas, 0.5, 0.5)
    scalars, rows = stats.flush(2)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(gradient, prediction.grad)
    assert sum(row['sample_count'] for row in rows) == 6
    assert rows[0]['t_min'] == 0 and rows[-1]['t_max'] == 999
    assert rows[2]['unweighted_mse'] is None
    assert math.isclose(scalars['interval_unweighted_mse'], 14 / 3)
    assert math.isclose(scalars['interval_weighted_mse'], 3.5)
    assert scalars['clip_fraction'] == 0.5
    assert scalars['grad_norm_mean'] == 1.25
    assert scalars['mean_snr_weight'] == .75
    assert stats.summary()['sample_count'] == 6
    assert stats.flush(2) == ({}, [])
    stats.record(prediction, target, timesteps, alphas, None, 0.2)
    _, rows = stats.flush(3)
    assert rows[0]['step_start'] == 3
    assert stats.summary()['successful_optimizer_steps'] == 3


def test_diagnostics_reject_nonfinite_stats():
    stats = TrainingDiagnostics(10)
    with pytest.raises(ValueError, match='Non-finite'):
        stats.record(torch.ones(1, 1), torch.zeros(1, 1), torch.tensor([0]), torch.ones(10) * .5,
                     None, float('nan'))
