from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml
from test_minsnr_ablation import tiny_protocol

from ebsd_diffusion.structure_statistics import (
    boundary_statistics, gradient_measurement, structure_statistics_loss,
    validate_statistics_config, x0_from_epsilon,
)


def test_finite_window_correlations_have_no_wrap_and_match_manual_products():
    b = torch.tensor([[[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]]])
    rho, s = boundary_statistics(b, [1])
    expected = torch.stack([(b[:, :, 1:] * b[:, :, :-1]).mean(),
                            (b[:, 1:, :] * b[:, :-1, :]).mean(),
                            (b[:, 1:, 1:] * b[:, :-1, :-1]).mean(),
                            (b[:, 1:, :-1] * b[:, :-1, 1:]).mean()])
    assert torch.allclose(s[0], expected)
    assert rho.item() == pytest.approx(1 / 3)


def test_statistics_distinguish_spacing_at_equal_boundary_density_and_backpropagate():
    # Two images have equal black-pixel fraction but different line spacing.
    target = torch.ones(1, 3, 16, 16)
    target[:, :, :, [2, 6, 10, 14]] = -1
    pred = torch.ones_like(target)
    pred[:, :, :, 2:6] = -1
    pred.requires_grad_(); target.requires_grad_()
    equal, _ = structure_statistics_loss(target, target, distances=[1, 2, 4])
    loss, components = structure_statistics_loss(pred, target, distances=[1, 2, 4])
    assert equal.item() == 0
    assert components['density_loss'].item() < 1e-6
    assert components['correlation_loss'].item() > .01
    loss.backward()
    assert pred.grad.abs().sum() > 0 and torch.isfinite(pred.grad).all()
    assert target.grad is None


def test_empty_boundaries_and_bad_configuration():
    a = torch.ones(2, 3, 8, 8, requires_grad=True)
    loss, _ = structure_statistics_loss(a, a, distances=[1, 2])
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(a.grad).all()
    for cfg in [{'distances': [1, 1]}, {'temperature': 0}, {'max_samples': 0}]:
        with pytest.raises(ValueError):
            validate_statistics_config(cfg)
    with pytest.raises(ValueError):
        boundary_statistics(torch.ones(1, 8, 8), [8])


def test_epsilon_inverse_and_gradient_probe_preserve_optimizer_and_rng():
    z = torch.randn(2, 3, 4, 4)
    e = torch.randn_like(z)
    alpha = torch.tensor([.8, .4]); t = torch.tensor([0, 1])
    noisy = alpha.sqrt()[:, None, None, None] * z + (1-alpha).sqrt()[:, None, None, None] * e
    assert torch.allclose(x0_from_epsilon(noisy, e, t, alpha), z, atol=1e-6)
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    rng = torch.get_rng_state().clone(); before = p.detach().clone()
    values = gradient_measurement(p.square().mean(), (p-3).square().mean(), [p])
    assert p.grad is None and torch.equal(before, p.detach()) and torch.equal(rng, torch.get_rng_state())
    assert values['gradient_cosine'] < 0


def test_pair_config_rejects_unmatched_training():
    from scripts.run_structure_statistics import assert_pair
    pair = {name: {'train_diffusion': {'output_dir': name, 'learning_rate': .001}}
            for name in ('A_mse', 'B_statistics')}
    pair['B_statistics']['train_diffusion']['structure_statistics'] = {'distances': [1]}
    assert_pair(pair)
    pair['B_statistics']['train_diffusion']['learning_rate'] = .01
    with pytest.raises(ValueError):
        assert_pair(pair)


def test_prepare_uses_reviewed_final_and_freezes_inputs(tmp_path, monkeypatch):
    from scripts import run_structure_statistics as pilot
    base = tmp_path/'baseline'
    final = base/'A_mse/final'
    weights = final/'unet_raw/diffusion_pytorch_model.safetensors'
    encoder = final/'condition_encoder/pytorch_model.bin'
    for p in (weights, encoder):
        p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'reviewed final weights')
    config = {'seed': 0, 'data': {'patch_size': 256}, 'diffusion': {'prediction_type': 'epsilon'},
              'train_diffusion': {'gradient_accumulation_steps': 1, 'initialization': {'old': 'source'}}}
    metadata = {'global_step': 10000, 'effective_total_step': 60000, 'latent_scaling': 1.125}
    monkeypatch.setattr(pilot, 'verify_baseline', lambda *a: (base, {}, config, metadata, None))
    def check(p, root, arms):
        assert p['expected_source_step'] == 10000
        assert arms['A_mse']['train_diffusion']['initialization']['raw_unet'] == str(weights)
        return {'initialization': {}, 'file_sha256': {}}
    monkeypatch.setattr(pilot, 'preflight', check)
    protocol = {'output_root': str(tmp_path/'run'), 'expected_baseline_effective_step': 60000,
        'expected_baseline_sha256': {'raw_unet': pilot.file_sha256(weights)},
        'training': {'additional_steps': 5000, 'warmup_steps': 250, 'learning_rate': 3e-5,
                     'seed': 5, 'save_every': 1000, 'log_every': 25}, 'structure_statistics': {}}
    path = tmp_path/'protocol.yaml'; path.write_text(yaml.safe_dump(protocol))
    pilot.prepare(path)
    _, manifest = pilot.load_prepared(path)
    assert manifest['source_effective_step'] == 60000
    with pytest.raises(FileExistsError):
        pilot.prepare(path)
    Path(manifest['configs']['B_statistics']).write_text('tampered')
    with pytest.raises(ValueError, match='changed'):
        pilot.load_prepared(path)


def test_statistics_calibration_and_optimizer_path_cpu(tiny_protocol):
    # Shared real model fixture: small VAE/U-Net, no mock gradients or fake weights.
    from scripts.run_minsnr_ablation import make_arm_configs, PROJECT_ROOT
    _, protocol = tiny_protocol
    _, configs = make_arm_configs(protocol, PROJECT_ROOT)
    cfg = configs['A_mse']
    cfg['train_diffusion']['track_source_effective_step'] = True
    cfg['train_diffusion']['structure_statistics'] = {
        'distances': [1, 2], 'max_t': 9, 'max_samples': 2, 'calibration_batches': 4,
        'calibration_file': str(Path(protocol['output_root']).parent / 'calibration/statistics_calibration.json')}
    path = Path(protocol['output_root']).parent / 'statistics.yaml'
    path.write_text(yaml.safe_dump(cfg))
    calibration_dir = Path(cfg['train_diffusion']['structure_statistics']['calibration_file']).parent
    def run(out, calibrate=False):
        cmd = [sys.executable, str(PROJECT_ROOT/'train_diffusion.py'), '--config', str(path),
               '--exclude-alloy-id', '3', '--output-dir', str(out), '--software-test-cpu']
        if calibrate: cmd.append('--statistics-calibrate')
        import os
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'ACCELERATE_USE_CPU': 'true'}
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    run(calibration_dir, True)
    calibration = json.loads((calibration_dir/'statistics_calibration.json').read_text())
    assert calibration['optimizer_updates'] == 0 and calibration['lambda'] > 0
    assert not (calibration_dir/'final').exists()
    run(Path(protocol['output_root'])/'statistics')
    final = json.loads((Path(protocol['output_root'])/'statistics/final/run_metadata.json').read_text())
    assert final['global_step'] == 2
    assert final['structure_statistics']['lambda'] == calibration['lambda']
    assert final['effective_total_step'] == 50002
