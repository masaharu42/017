import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest
import yaml

from test_minsnr_ablation import tiny_protocol
from test_local_descriptors import grid
from scripts import run_descriptor_pilot as local
from scripts import run_minsnr_ablation as old


def test_reuse_control_end_to_end_and_stop_on_tampering(tiny_protocol, tmp_path, monkeypatch):
    """Two optimizer steps of tiny synthetic CPU models; no scientific run/data."""
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setenv('MKL_NUM_THREADS', '1')
    path, protocol = tiny_protocol
    protocol['evaluation']['alloy_ids'] = [1, 2]
    path.write_text(yaml.safe_dump(protocol))
    for i in (1, 2, 3):
        Image.fromarray(grid()).save(tmp_path / 'source' / f'real_{i}.png')
    old.prepare(path)
    old.train(path)
    old.evaluate(path)
    p = {'name': 'synthetic_local5_only', 'baseline_run': protocol['output_root'],
        'output_root': str(tmp_path / 'local_run'), 'expected_source_step': 50000,
        'expected_additional_steps': 2, 'require_cuda': False, 'software_test_only': True,
        'descriptor_hidden_dim': 32, 'uncertainty_inputs': False, 'normalizer_windows_per_alloy': 16,
        'descriptor_settings': {'min_region_area_px': 1, 'boundary_dilation_px': 0,
                                'coarse_diameter_px': 3.},
        'evaluation': {**protocol['evaluation'], 'target_quantiles': [.1,.5,.9]}}
    pp = tmp_path / 'local.yaml'; pp.write_text(yaml.safe_dump(p))
    bad_protocol = tmp_path / 'uncertainty.yaml'
    bad_protocol.write_text(yaml.safe_dump({**p, 'uncertainty_inputs': True}))
    with pytest.raises(ValueError, match='excludes uncertainty'):
        local.read_protocol(bad_protocol)
    manifest = local.prepare(pp)
    assert manifest['status'] == 'prepared_not_trained'
    root = Path(p['output_root'])
    meta = json.loads((root / 'labels/metadata.json').read_text())
    assert meta['training_alloy_ids'] == [1, 2]
    assert {r['alloy_id'] for r in meta['images']} == {1, 2}
    with pytest.raises(FileExistsError):
        local.prepare(pp)
    result = local.train(pp)
    assert result['new_training_arms'] == ['B_local5']
    verified = json.loads((root / 'B_local5/matched_baseline_inputs.json').read_text())
    assert verified['status'] == 'passed' and verified['checked_steps'] == [1, 2]
    normalizer = json.loads((root / 'B_local5/final/condition_encoder/descriptor_normalizer.json').read_text())
    assert normalizer['training_alloy_ids'] == [1, 2]
    report = local.evaluate(pp)
    assert report['new_samples'] == 12 and report['reused_control_images'] == 4
    assert not (root / 'A_mse').exists()
    assert not (root / 'evaluation/alloy_03').exists()
    assert (root / 'evaluation/paired_descriptor_errors.csv').is_file()
    # A changed reference stream must fail BEFORE the first optimizer update,
    # rather than merely warning after a paid server run.
    config = yaml.safe_load(Path(manifest['config']).read_text())
    bad_reference = tmp_path / 'wrong_fingerprints.csv'
    bad_reference.write_text('additional_step,sha256\n1,wrong\n2,wrong\n')
    config['train_diffusion']['output_dir'] = str(tmp_path / 'must_stop')
    config['train_diffusion']['local_descriptors']['reference_fingerprints'] = str(bad_reference)
    bad_config = tmp_path / 'bad_config.yaml'; bad_config.write_text(yaml.safe_dump(config))
    failed = subprocess.run([sys.executable, str(local.ROOT / 'train_diffusion.py'), '--config',
        str(bad_config), '--exclude-alloy-id', '3'], capture_output=True, text=True)
    assert failed.returncode != 0 and 'Baseline input mismatch at step 1' in failed.stderr
    assert not (tmp_path / 'must_stop/final').exists()
    with pytest.raises(ValueError, match='Complete training'):
        local.evaluate(pp)
    metadata = root / 'labels/metadata.json'; metadata.write_text(metadata.read_text() + '\n')
    with pytest.raises(ValueError, match='changed'):
        local.load_prepared(pp)
