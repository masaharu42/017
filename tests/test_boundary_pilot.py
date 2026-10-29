import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml
import pytest
from safetensors.torch import load_file

from test_minsnr_ablation import tiny_protocol
from scripts.run_minsnr_ablation import make_arm_configs, PROJECT_ROOT


def test_cpu_synthetic_calibration_and_training(tiny_protocol, tmp_path):
    """Synthetic tiny network validates no-update calibration and training integration."""
    _, protocol = tiny_protocol
    _, arms = make_arm_configs(protocol, PROJECT_ROOT)
    config = arms['A_mse']
    config['train_diffusion']['gradient_checkpointing'] = True
    config['train_diffusion']['boundary_dice'] = {
        'max_t': 300, 'max_samples': 8, 'calibration_batches': 4,
        'calibration_file': str(tmp_path / 'calibration/boundary_calibration.json'),
    }
    config_path = tmp_path / 'training.yaml'
    config_path.write_text(yaml.safe_dump(config), 'utf-8')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', ACCELERATE_USE_CPU='true')
    cmd = [sys.executable, str(PROJECT_ROOT / 'train_diffusion.py'), '--config', str(config_path), '--exclude-alloy-id', '3', '--software-test-cpu']
    source = Path(config['train_diffusion']['initialization']['raw_unet'])
    before = source.read_bytes()
    result = subprocess.run(cmd + ['--output-dir', str(tmp_path / 'calibration'), '--boundary-calibrate'],
                            cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    calibration = json.loads((tmp_path / 'calibration/boundary_calibration.json').read_text())
    assert calibration['status'] == 'passed' and calibration['optimizer_updates'] == 0
    assert calibration['lambda'] > 0 and source.read_bytes() == before
    result = subprocess.run(cmd + ['--output-dir', str(tmp_path / 'C')], cwd=PROJECT_ROOT,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    meta = json.loads((tmp_path / 'C/final/run_metadata.json').read_text())
    assert meta['global_step'] == 2 and meta['boundary_dice']['lambda'] == calibration['lambda']
    assert 'boundary_dice_loss' in (tmp_path / 'C/training_metrics.csv').read_text()
    assert any(not torch.equal(a, b) for a, b in zip(load_file(str(source)).values(),
                                                     load_file(str(tmp_path / 'C/final/unet_raw/diffusion_pytorch_model.safetensors')).values()))


def test_prepare_reuses_a_and_guards_modified_sources(tiny_protocol, tmp_path, monkeypatch):
    from scripts import run_boundary_pilot as pilot
    _, original = tiny_protocol
    _, arms = make_arm_configs(original, PROJECT_ROOT)
    config = arms['A_mse']
    config['train_diffusion'].update(batch_size=32, learning_rate=3e-5, warmup_steps=500)
    base = tmp_path / 'old'
    files = ['experiment_manifest.json', 'A_mse/input_fingerprints.csv', 'A_mse/final/run_metadata.json',
             'A_mse/final/unet_raw/diffusion_pytorch_model.safetensors', 'A_mse/final/unet_raw/config.json',
             'A_mse/final/condition_encoder/config.json', 'A_mse/final/condition_encoder/pytorch_model.bin']
    for filename in files:
        p = base / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('baseline test data')
    old = {'preflight': {'initialization': {'source_global_step': 50000},
                         'training_alloy_ids': [5, 7, 20], 'file_sha256': {}}}
    monkeypatch.setattr(pilot, 'verify_baseline', lambda *args: (base, old, config, {'effective_total_step': 60000}, base / files[1]))
    protocol = {'baseline_run': str(base), 'output_root': str(tmp_path / 'run_boundary'),
                'expected_source_step': 50000, 'expected_additional_steps': 10000,
                'evaluation': {'alloy_ids': [5, 7, 20], 'seeds': [20261012, 20261013]}}
    path = tmp_path / 'protocol.yaml'
    path.write_text(yaml.safe_dump(protocol))
    assert pilot.prepare(path)['A_retrained'] is False
    assert pilot.load_prepared(path)[1]['status'] == 'prepared'
    with pytest.raises(FileExistsError):
        pilot.prepare(path)
    (base / files[-1]).write_text('tampered')
    with pytest.raises(ValueError, match='changed or missing'):
        pilot.load_prepared(path)
