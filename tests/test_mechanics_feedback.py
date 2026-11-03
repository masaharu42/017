from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from physics.soft_mechanics_features import SoftMechanicsFeatures, group_mean, build_feature_extractor
from physics.linear_mechanics import LinearMechanics, fit_ridge, nested_oof, SCHEMA, TARGET_NAMES
from ebsd_diffusion.mechanics_groups import AlloyGroupBatchSampler, validate_groups, group_random_inputs
from ebsd_diffusion.mechanics_runtime import target_losses, gradient_probe, protocol_signature
from test_minsnr_ablation import tiny_protocol


def artifact():
    # Synthetic calibrated readout, not an empirical mechanics claim.
    model = {'input_mean': [0.]*4, 'input_scale': [1.]*4, 'coefficient': [0., .3, .2, -.5],
             'target_mean': 0., 'target_scale': 1., 'alpha': 1.}
    return {'schema': SCHEMA, 'composition_names': ['Ni'], 'target_names': list(TARGET_NAMES),
            'feature_extractor': SoftMechanicsFeatures(lag=2).metadata(),
            'models': [model, deepcopy(model)], 'approved_targets': [True, True]}


def test_soft_boundary_is_continuous_not_a_learned_classifier():
    features = SoftMechanicsFeatures(lag=1)
    x = torch.full((1, 3, 4, 4), -.6, requires_grad=True)  # RGB .2 => B .5
    b = features.boundary(x)
    assert torch.allclose(b, torch.full_like(b, .5), atol=1e-6)
    b.mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert not list(features.parameters())
    with pytest.raises(ValueError, match='CNN'):
        build_feature_extractor({'type': 'cnn'})


def test_spacing_and_gradients_without_wrapping():
    extractor = SoftMechanicsFeatures(lag=2)
    x = torch.zeros(2, 3, 8, 8)
    x[0, :, :, [0, 2]] = -1
    x[1, :, :, [0, 1]] = -1
    x.requires_grad_()
    d = extractor(x)
    assert d[0, 0].item() == pytest.approx(d[1, 0].item())
    assert d[0, 2].item() != pytest.approx(d[1, 2].item())
    d.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    with pytest.raises(ValueError): group_mean(d, 3)


def test_frozen_proxy_preserves_full_image_path_and_blocks_composition_shortcut():
    proxy = LinearMechanics(artifact())
    generator = torch.nn.Conv2d(3, 3, 1)
    decoder = torch.nn.Conv2d(3, 3, 1).requires_grad_(False)
    image = decoder(generator(torch.randn(4, 3, 8, 8)))
    c = torch.ones(2, 1, requires_grad=True)
    prediction = proxy(c, group_mean(proxy.features(image), 2))
    loss, _ = target_losses(prediction, torch.ones(2, 2), torch.ones(2, 2, dtype=torch.bool),
                            proxy.target_scale, proxy.active, torch.ones(2))
    loss.backward()
    assert generator.weight.grad.abs().sum() > 0
    assert decoder.weight.grad is None and c.grad is None
    assert not list(proxy.parameters())
    # Linearity means aggregation order agrees.
    per_window = proxy(c.detach().repeat_interleave(2, 0), proxy.features(image.detach()))
    assert torch.allclose(prediction.detach(), per_window.reshape(2, 2, 2).mean(1), atol=1e-6)


def test_equal_target_losses_are_scaled_and_missing_does_not_double_other():
    pred = torch.tensor([[1100., 18.]], requires_grad=True)
    target = torch.tensor([[1200., 20.]])
    args = (torch.tensor([100., 2.]), torch.tensor([True, True]), torch.ones(1))
    total, terms = target_losses(pred, target, torch.tensor([[True, True]]), *args)
    assert total.item() == pytest.approx(.5) and terms[0].item() == terms[1].item()
    target[0, 1] = float('nan')
    missing, _ = target_losses(pred, target, torch.tensor([[True, False]]), *args)
    assert missing.item() == pytest.approx(.25)
    missing.backward()
    assert pred.grad[0, 1] == 0 and torch.isfinite(pred.grad).all()


def test_sampler_and_group_noise_conditions_do_not_mix_alloys():
    class Dataset:
        rows = [{'alloy_id': 1}, {'alloy_id': 2}, {'alloy_id': 4}]
        def __len__(self): return 48
    sampler = AlloyGroupBatchSampler(Dataset(), 8, 4, 9)
    batches = list(sampler)
    assert len(batches) == 6
    assert all(len(set(b[i:i+4])) == 1 for b in batches for i in (0, 4))
    sampler.set_epoch(1)
    assert list(sampler) != batches
    c = torch.ones(8, 2)
    t, jitter, dropped = group_random_inputs(c, 4, 1000, .05, .15)
    for v in (t[:, None], jitter, dropped[:, None]):
        assert torch.equal(v[:4], v[:1].expand_as(v[:4]))
    validate_groups(torch.tensor([1]*4+[2]*4), c, 4)
    with pytest.raises(ValueError): validate_groups(torch.arange(8), c, 4)


def test_gradient_probe_is_zero_update_and_reports_conflict():
    m = torch.nn.Linear(1, 1, bias=False)
    v = m(torch.ones(1, 1)).sum()
    before = m.weight.detach().clone()
    rng = torch.get_rng_state().clone()
    p = gradient_probe(v.square(), [v, -v], v*0, {'generator': m})
    assert p['target_cosine'] == pytest.approx(-1)
    assert p['mechanics_norm'] == 0
    assert m.weight.grad is None and torch.equal(before, m.weight)
    assert torch.equal(rng, torch.get_rng_state())


def test_nested_predictions_do_not_fit_the_heldout_label():
    x = np.arange(7.)[:, None]
    y = x[:, 0]*2
    p = nested_oof(x, y, alphas=(.1, 1.))
    y[0] += 1000
    q = nested_oof(x, y, alphas=(.1, 1.))
    assert p[0] == pytest.approx(q[0])
    model = fit_ridge(np.column_stack([x, np.ones(7)]), y, 1.)
    assert model['input_scale'][1] == 1.


def test_pair_guard_rejects_extra_changes():
    from scripts.run_mechanics_pair import assert_pair
    base = {'seed': 1, 'train_diffusion': {'output_dir': 'C0', 'learning_rate': .001,
                                         'mechanics': {'arm': 'C0'}}}
    other = deepcopy(base); other['train_diffusion'].update(output_dir='C1', mechanics={'arm': 'C1'})
    assert_pair({'C0': base, 'C1': other})
    other['train_diffusion']['learning_rate'] *= 2
    with pytest.raises(ValueError): assert_pair({'C0': base, 'C1': other})


def test_review_gate_does_not_accept_pending_or_wrong_source(tmp_path):
    from ebsd_diffusion.mechanics_runtime import MechanicsRuntime
    model = tmp_path/'proxy.json'; model.write_text(json.dumps(artifact()))
    decision = tmp_path/'decision.json'
    config = {'_project_root': str(tmp_path), 'diffusion': {'num_train_timesteps': 10},
              'train_diffusion': {'mechanics': {'arm': 'C1'}, 'output_dir': str(tmp_path/'unused')}}
    cfg = {'proxy_file': str(model), 'decision_file': str(decision)}
    decision.write_text(json.dumps({'schema': 'mechanics_m0_decision_v1', 'status': 'pending_review'}))
    with pytest.raises(ValueError, match='reviewed M0'):
        MechanicsRuntime(config, cfg, {}, calibrate=True)


def test_calibrate_and_both_continuation_arms_on_synthetic_models(tiny_protocol):
    from ebsd_diffusion.training_control import resolve_initialization, file_sha256
    from scripts.run_minsnr_ablation import make_arm_configs, PROJECT_ROOT
    _, protocol = tiny_protocol
    _, original = make_arm_configs(protocol, PROJECT_ROOT)
    cfg = original['A_mse']
    home = (Path(protocol['output_root']).parent/'mechanics')
    home.mkdir()
    data = pd.DataFrame({'alloy_id': [1, 2], 'peak_stress_MPa': [2., 3.], 'endpoint_strain_pct': [4., 5.]})
    labels = home/'labels.csv'; data.to_csv(labels, index=False)
    images = [Path(protocol['source_sampling_config']).parent/f'real_{i}.png' for i in (1, 2)]
    a = artifact(); a.update(training_alloy_ids=[1, 2], labels_sha256=file_sha256(labels),
                             input_sha256={str(p.resolve()): file_sha256(p) for p in images})
    proxy = home/'proxy.json'; proxy.write_text(json.dumps(a))
    provenance_cfg = deepcopy(cfg); provenance_cfg['_project_root'] = str(PROJECT_ROOT)
    _, source = resolve_initialization(provenance_cfg)
    gates = {}
    for name in ('window_stability', 'perturbations', 'domain', 'timestep_and_trajectory'):
        p = home/f'{name}.json'
        p.write_text(json.dumps({'status': 'passed', 'scope': 'SYNTHETIC SOFTWARE FIXTURE ONLY'}))
        gates[name] = {'status': 'passed', 'path': p.name, 'sha256': file_sha256(p)}
    vae = Path(cfg['train_diffusion']['vae_dir'])
    decision = home/'decision.json'
    decision.write_text(json.dumps({'schema': 'mechanics_m0_decision_v1', 'status': 'passed',
        'proxy_sha256': file_sha256(proxy), 'source_sha256': source['sha256'],
        'vae_sha256': {name: file_sha256(vae/name) for name in ('config.json', 'diffusion_pytorch_model.safetensors')},
        'active_targets': [True, True], 'max_t': 9, 'group_size': 2, 'gates': gates}))
    cfg['diffusion']['condition_dropout'] = 0.
    cfg['train_diffusion'].update(track_source_effective_step=True, mechanics={
        'arm': 'C1', 'proxy_file': str(proxy), 'decision_file': str(decision), 'labels_file': str(labels),
        'group_size': 2, 'max_decoded_windows': 2, 'calibration_batches': 8,
        'calibration_file': str(home/'calibration/mechanics_calibration.json')})
    path = home/'config.yaml'
    def run(name, calibrate=False):
        cfg['train_diffusion']['mechanics']['arm'] = 'C0' if name == 'C0' else 'C1'
        path.write_text(yaml.safe_dump(cfg))
        command = [sys.executable, str(PROJECT_ROOT/'train_diffusion.py'), '--config', str(path),
                   '--exclude-alloy-id', '3', '--software-test-cpu', '--output-dir', str(home/name)]
        if calibrate: command.append('--mechanics-calibrate')
        import os
        result = subprocess.run(command, capture_output=True, text=True, timeout=120,
            env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'ACCELERATE_USE_CPU': 'true'})
        assert result.returncode == 0, result.stdout+result.stderr
    run('calibration', True)
    cal = json.loads((home/'calibration/mechanics_calibration.json').read_text())
    assert cal['optimizer_updates'] == 0 and cal['lambda'] > 0
    assert not (home/'calibration/final').exists()
    run('C0'); run('C1')
    for name in ('C0', 'C1'):
        state = json.loads((home/name/'final/run_metadata.json').read_text())
        assert state['global_step'] == 2 and state['effective_total_step'] == 50002
        assert (state['mechanics']['lambda'] == 0) == (name == 'C0')
    assert (home/'C0/input_fingerprints.csv').read_text() == (home/'C1/input_fingerprints.csv').read_text()
    # Pair preparation is separate from draft and refuses post-prepare changes.
    from scripts.run_mechanics_pair import draft, prepare, load_prepared
    pair = deepcopy(protocol)
    pair.update(output_root=str(home/'pair'), expected_effective_step=50000,
                mechanics={key: value for key, value in cfg['train_diffusion']['mechanics'].items()
                           if key not in ('arm', 'calibration_file')})
    pair_path = home/'pair.yaml'; pair_path.write_text(yaml.safe_dump(pair))
    proposal = draft(pair_path)
    assert (proposal/'C1.yaml').is_file() and not Path(pair['output_root']).exists()
    prepare(pair_path)
    _, _, prepared = load_prepared(pair_path)
    p = Path(prepared['configs']['C0']); p.write_text(p.read_text()+'\n# changed\n')
    with pytest.raises(ValueError, match='changed'): load_prepared(pair_path)


def test_m0_extract_and_fit_never_open_holdout_or_issue_overall_pass(tmp_path):
    from PIL import Image
    from scripts.validate_mechanics_m0 import extract, fit
    ids = [1, 2, 4, 5, 6, 7]
    rows = []
    for i in ids:
        pixels = np.full((20, 20, 3), 160, dtype=np.uint8)
        pixels[:, :i] = 0
        p = tmp_path/f'{i}.png'; Image.fromarray(pixels).save(p)
        rows.append({'alloy_id': i, 'image_path': str(p)})
    rows.append({'alloy_id': 3, 'image_path': '/DO_NOT_OPEN_HELDOUT.png'})
    manifest = tmp_path/'manifest.csv'; pd.DataFrame(rows).to_csv(manifest, index=False)
    c = tmp_path/'compositions.csv'; pd.DataFrame({'alloy_id': ids+[3], 'Ni': ids+[999]}).to_csv(c, index=False)
    y = tmp_path/'labels.csv'
    pd.DataFrame({'alloy_id': ids+[3], 'peak_stress_MPa': ids+[99999],
                  'endpoint_strain_pct': np.asarray(ids+[99999])*2}).to_csv(y, index=False)
    source = tmp_path/'conditioner.json'
    source.write_text(json.dumps({'training_alloy_ids': ids, 'active_features': ['Ni']}))
    cfg = tmp_path/'config.yaml'
    cfg.write_text(yaml.safe_dump({'mechanics_m0': {'manifest': str(manifest), 'compositions': str(c),
        'labels': str(y), 'source_conditioner': str(source), 'exclude_alloy_id': 3,
        'patch_size': 8, 'windows_per_alloy': 8, 'feature_settings': {'lag': 2}}}))
    extract(cfg, tmp_path/'extraction')
    features = pd.read_csv(tmp_path/'extraction/alloy_features.csv')
    assert features.alloy_id.tolist() == ids
    fit(tmp_path/'extraction', tmp_path/'proxy', permutations=0)
    decision = json.loads((tmp_path/'proxy/m0_decision.template.json').read_text())
    proxy = json.loads((tmp_path/'proxy/proxy_model.json').read_text())
    assert decision['status'] == 'pending_review' and not any(proxy['approved_targets'])
    assert all(v['status'] == 'pending' for v in decision['gates'].values())
    with pytest.raises(FileExistsError): extract(cfg, tmp_path/'extraction')
    # Missing strain labels disable only the strain head, not the peak fitting path.
    labels = pd.read_csv(y)
    labels.loc[labels.alloy_id.isin(ids[:3]), 'endpoint_strain_pct'] = np.nan
    labels.to_csv(y, index=False)
    extract(cfg, tmp_path/'extraction_missing')
    fit(tmp_path/'extraction_missing', tmp_path/'proxy_missing', permutations=0)
    missing = json.loads((tmp_path/'proxy_missing/proxy_model.json').read_text())
    assert 'disabled_reason' not in missing['models'][0]
    assert missing['models'][1]['disabled_reason'] == 'insufficient_labels'
    missing['approved_targets'][1] = True
    with pytest.raises(ValueError, match='disabled'):
        LinearMechanics(missing)


def test_input_diagnostic_keeps_trajectory_review_pending(tiny_protocol):
    from PIL import Image
    from scripts.probe_mechanics_inputs import run
    from scripts.run_minsnr_ablation import make_arm_configs, PROJECT_ROOT
    _, protocol = tiny_protocol
    _, arms = make_arm_configs(protocol, PROJECT_ROOT)
    cfg = arms['A_mse']
    base = Path(protocol['source_sampling_config']).parent
    home = base/'diagnostic'; home.mkdir()
    features = SoftMechanicsFeatures(lag=2)
    rows = []
    for i in (1, 2):
        pixels = np.full((12, 12, 3), 150, dtype=np.uint8)
        if i == 1: pixels[:, 0] = 0
        else: pixels[:, 3:5] = 0
        Image.fromarray(pixels).save(base/f'real_{i}.png')
        x = torch.from_numpy(pixels[:8, :8].copy()).permute(2, 0, 1)[None]/127.5-1
        rows.append(dict(zip(features.names, features(x)[0].tolist())))
    pd.DataFrame(rows).to_csv(home/'alloy_features.csv', index=False)
    (home/'extraction.json').write_text(json.dumps({'input_sha256': {}, 'training_alloy_ids': [1, 2],
        'feature_extractor': features.metadata(), 'windows_per_alloy': 1, 'patch_size': 8,
        'positions': {'1': [[0, 0]], '2': [[0, 0]]}}))
    proxy = home/'proxy.json'; proxy.write_text(json.dumps(artifact()))
    cfg['train_diffusion']['mechanics'] = {'proxy_file': str(proxy)}
    cfg['mechanics_diagnostics'] = {'timesteps': [1, 5], 'noise_seeds': [17], 'windows_per_alloy': 1}
    path = home/'config.yaml'; path.write_text(yaml.safe_dump(cfg))
    run(path, home, home/'results', device='cpu')
    result = json.loads((home/'results/timestep_stability.json').read_text())
    assert result['status'] == 'trajectory_review_pending' and result['optimizer_updates'] == 0
    assert len(result['records']) == 4
    assert not (home/'results/final').exists()
