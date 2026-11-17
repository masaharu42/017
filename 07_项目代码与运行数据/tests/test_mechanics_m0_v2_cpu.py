from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import yaml

from ebsd_diffusion import block_ridge as ridge
from ebsd_diffusion.hard_morphology import describe_hard, NAMES, LAYOUT_NAME
from ebsd_diffusion.local_descriptors import describe_rgb, DEFAULTS
from ebsd_diffusion.mechanics_cpu_audit import sha256, trial_audit, compare_tables
from scripts import run_mechanics_m0_v2 as cpu


def test_shared_penalty_matches_direct_ridge_and_contributions_sum():
    rng = np.random.default_rng(8)
    c, d, y = rng.normal(size=(9, 2)), rng.normal(size=(9, 3)), rng.normal(size=9)
    model = ridge.fit_selected(c, d, y, 'shared', (3., 1.))
    raw = np.column_stack([c, d]); z = (raw-raw.mean(0))/raw.std(0)
    coefficient = np.linalg.solve(z.T@z+3*np.eye(5), z.T@(y-y.mean()))
    expected = y.mean()+z@coefficient
    assert np.allclose(model.predict(c, d), expected)
    parts = model.contributions(c, d)
    assert np.allclose(parts['prediction'], parts['intercept']+np.array(parts['composition'])+parts['image'])
    off = ridge.fit_selected(c, d, y, 'block_all', (3., None))
    assert np.allclose(off.predict(c, d), off.predict(c, d*1000))


def test_nested_scaling_pca_and_pruning_never_fit_outer_or_inner_validation(monkeypatch):
    rng = np.random.default_rng(18)
    c, d, y = rng.normal(size=(7, 2)), rng.normal(size=(7, 4)), rng.normal(size=7)
    d[0] *= 10000
    original = ridge.fit_transform
    calls = []
    def observed(cc, dd, mode, **kwargs):
        calls.append((len(cc), mode))
        return original(cc, dd, mode, **kwargs)
    monkeypatch.setattr(ridge, 'fit_transform', observed)
    p, models = ridge.nested_predict(c, d, y, 'block_pc1', [.1, 1.], [1., 3.], return_models=True)
    assert {n for n, _ in calls} == {5, 6}
    assert np.allclose(models[0].transform.d_mean, d[1:].mean(0))
    altered = y.copy(); altered[0] += 5000
    q = ridge.nested_predict(c, d, altered, 'block_pc1', [.1, 1.], [1., 3.])
    assert p[0] == pytest.approx(q[0])
    # Priority, not raw variance or units, chooses among correlated columns.
    dd = np.column_stack([np.arange(7.), np.arange(7.)*1000, rng.normal(size=7)])
    t = original(c, dd, 'hard', priority=[0, 1, 2], correlation_threshold=.9, max_features=3)
    assert 0 in t.retained and 1 not in t.retained
    rank = ridge.rank_diagnostics(dd)
    assert rank['effective_rank'] < 3


def test_hard_region_statistics_preserve_existing_visible_aspect_and_quality():
    rgb = np.full((32, 32, 3), 180, dtype=np.uint8)
    rgb[::8] = 0; rgb[:, ::8] = 0
    settings = {**DEFAULTS, 'boundary_dilation_px': 0, 'min_region_area_px': 2}
    values, valid, quality = describe_hard(rgb, settings, include_layout=True, spatial_block=8)
    old, old_valid, _ = describe_rgb(rgb, settings)
    assert valid and old_valid and quality['retained_region_count'] >= 3
    assert values[3] == pytest.approx(old[2])
    assert values[2] == pytest.approx(old[3])
    assert np.isfinite(values).all() and values.shape == (5,)
    blank = np.full_like(rgb, 180)
    invalid, ok, _ = describe_hard(blank, settings)
    assert not ok and np.isnan(invalid).all()


def test_repeatability_excludes_holdout_and_does_not_diagnose_equal_endpoint():
    trials, labels = [], []
    for i, center in [(1, 10.), (2, 20.), (3, 100000.)]:
        labels.append({'alloy_id': i, 'peak_stress_MPa': center, 'endpoint_strain_pct': center/2})
        for j, e in enumerate([-1., 0., 1.]):
            trials.append({'alloy_id': i, 'ncorr_curve_file': f'{i}_{j}', 'peak_stress_MPa': center+e,
                'endpoint_strain_pct': center/2+e, 'peak_strain_pct': center/2+e,
                'endpoint_stress_MPa': .8*(center+e)})
    report, quality, summary = trial_audit(pd.DataFrame(trials), pd.DataFrame(labels), [1, 2], bootstrap_count=20)
    assert report['development_trial_rows'] == 6 and report['heldout_or_other_rows_excluded'] == 3
    assert set(quality.alloy_id) == {1, 2} and quality.endpoint_equals_peak_flag.all()
    assert set(quality.endpoint_reason) == {'unknown'}
    assert report['targets']['peak_stress_MPa']['pooled_within_trial_variance'] == pytest.approx(1.)
    assert report['targets']['peak_stress_MPa']['max_absolute_aggregation_difference'] == pytest.approx(0.)
    assert 'rmse_floor' not in report['targets']['endpoint_strain_pct']
    assert summary.confirmed_fracture_subset_median.isna().all()


def test_ridge_leverage_and_cook_like_are_labelled_and_query_sensitive():
    rng = np.random.default_rng(11)
    c, d = rng.normal(size=(8, 2)), rng.normal(size=(8, 3))
    y = c[:, 0]*2+rng.normal(size=8)*.1
    m = ridge.fit_selected(c, d, y, 'block_all', (1., 3.))
    result = ridge.influence_diagnostics(m, c, d, y, c[:1]*1000, d[:1]*1000)
    assert len(result['ridge_cook_like']) == 8 and result['effective_df'] < 8
    assert result['query_design_leverage'][0] > 1
    assert 'not an OLS' in result['limits']


@pytest.fixture
def prepared_config(tmp_path):
    ids = [1, 2, 4, 5, 6, 21]
    images, manifest, hashes, rows = {}, [], {}, []
    for i in ids:
        pixels = np.full((40, 40, 3), 160, dtype=np.uint8)
        spacing = 4+i%3
        pixels[::spacing] = 0; pixels[:, ::spacing] = 0
        p = tmp_path/f'{i}.png'; Image.fromarray(pixels).save(p)
        images[i] = p; hashes[str(p)] = sha256(p)
        manifest.append({'alloy_id': i, 'image_path': str(p)})
        rows.append({'alloy_id': i, 'Ni': float(i), 'Al': i%2+.2,
            'soft_boundary_fraction': .1*i, 'soft_boundary_variation': .2*i+.01*(i%2),
            'interior_s2': 1-.1*i, 'peak_stress_MPa': 100.+i*5,
            'endpoint_strain_pct': 10.+i})
    manifest.append({'alloy_id': 3, 'image_path': '/DO_NOT_OPEN_HELDOUT.png'})
    m = tmp_path/'manifest.csv'; pd.DataFrame(manifest).to_csv(m, index=False)
    table = pd.DataFrame(rows)
    c = tmp_path/'compositions.csv'; table[['alloy_id', 'Ni', 'Al']].to_csv(c, index=False)
    y = tmp_path/'labels.csv'; table[['alloy_id', *cpu.TARGETS]].to_csv(y, index=False)
    trials = []
    for row in rows+[{'alloy_id': 3, 'peak_stress_MPa': 99999., 'endpoint_strain_pct': 88888.}]:
        for j, delta in enumerate([-1., 0., 1.]):
            trials.append({'alloy_id': row['alloy_id'], 'ncorr_curve_file': f'{row["alloy_id"]}_{j}',
                'peak_stress_MPa': row['peak_stress_MPa']+delta,
                'endpoint_strain_pct': row['endpoint_strain_pct']+delta,
                'peak_strain_pct': row['endpoint_strain_pct']+delta,
                'endpoint_stress_MPa': .8*(row['peak_stress_MPa']+delta)})
    trial = tmp_path/'trials.csv'; pd.DataFrame(trials).to_csv(trial, index=False)
    extraction = tmp_path/'extraction'; extraction.mkdir()
    table.to_csv(extraction/'alloy_features.csv', index=False)
    np.savez(extraction/'window_features.npz', features=np.ones((6, 4, 3)), alloy_ids=ids)
    meta = {'training_alloy_ids': ids, 'composition_names': ['Ni', 'Al'], 'patch_size': 16,
        'positions': {str(i): [[0, 0], [8, 0], [0, 8], [8, 8]] for i in ids},
        'input_sha256': hashes, 'feature_table_sha256': sha256(extraction/'alloy_features.csv'),
        'window_bank_sha256': sha256(extraction/'window_features.npz')}
    (extraction/'extraction.json').write_text(json.dumps(meta))
    template = yaml.safe_load((cpu.ROOT/'configs/mechanics_m0_v2_cpu.yaml').read_text())
    s = template['mechanics_m0_v2']
    s.update(extraction=str(extraction), output_root=str(tmp_path/'result'), manifest=str(m),
        compositions=str(c), labels=str(y), trials=str(trial), expected_training_alloys=6,
        software_fixture=True, permutations=1, bootstrap_count=20, alphas=[.1, 1.], image_penalty_ratios=[1., 3., 'off'],
        hard_settings={**DEFAULTS, 'boundary_dilation_px': 0, 'min_region_area_px': 2}, spatial_block=8)
    config = tmp_path/'audit.yaml'; config.write_text(yaml.safe_dump(template))
    cpu.prepare(config)
    return config, tmp_path/'result', template


def test_locked_cpu_pipeline_has_no_training_authorization(prepared_config):
    config, root, _ = prepared_config
    cpu.audit(config)
    report = json.loads((root/'audit/repeatability.json').read_text())
    assert report['development_trial_rows'] == 18 and report['heldout_or_other_rows_excluded'] == 3
    diagnostics = json.loads((root/'audit/soft_diagnostics.json').read_text())
    for mode in ['composition', 'shared', 'block_pc1', 'block_all', 'image_only']:
        point = diagnostics['targets']['peak_stress_MPa'][mode]['id21']
        parts = point['decomposition']
        assert point['prediction'] == pytest.approx(parts['intercept']+parts['composition'][0]+parts['image'][0])
    cpu.extract_hard(config)
    hard = pd.read_csv(root/'hard/alloy_hard_features.csv')
    assert 3 not in hard.alloy_id.tolist() and hard.quality_gate_passed.all()
    cpu.validate(config)
    result = cpu.decision(config)
    assert result['training_authorized'] is False and result['cnn_training_authorized'] is False
    assert result['next_step'] == 'software_fixture_no_scientific_decision'
    summary = json.loads((root/'validation/summary.json').read_text())
    assert len(summary['all_reports']) == 10
    assert all(r['permutation_count'] == 1 for r in summary['all_reports'])
    assert all(not r['permutation_gate_passed'] and not r['cpu_proxy_gate_passed'] for r in summary['all_reports'])
    assert all(r['permutation_p_threshold'] == .05 for r in summary['all_reports'])
    assert not (root/'proxy_model.json').exists()
    with pytest.raises(FileExistsError): cpu.extract_hard(config)
    # Reviewed phase outputs are bound, not silently overwritten.
    target = root/'hard/alloy_hard_features.csv'; target.write_text(target.read_text()+'\n')
    with pytest.raises(ValueError, match='changed'): cpu.validate(config)


def test_pre_registration_rejects_changes_and_duplicate_runs(prepared_config):
    config, root, template = prepared_config
    with pytest.raises(FileExistsError): cpu.prepare(config)
    changed = deepcopy(template); changed['mechanics_m0_v2']['alphas'] = [1.]
    config.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match='changed'): cpu.load_plan(config)
    config.write_text(yaml.safe_dump(template))
    # Restore bytes exactly as prepare wrote; hash includes the registered config.
    cpu.load_plan(config)
    plan = json.loads((root/'plan.json').read_text()); plan['training_alloy_ids'].append(3)
    (root/'plan.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='plan changed'): cpu.load_plan(config)


def test_field_comparison_does_not_guess_why_bytes_differ(tmp_path):
    a, b = tmp_path/'a.csv', tmp_path/'b.csv'
    pd.DataFrame({'alloy_id': [1, 2, 3], 'x': [1., 2., 999]}).to_csv(a, index=False)
    pd.DataFrame({'alloy_id': [2, 1, 3], 'x': [2., 1., -999]}).to_csv(b, index=False)
    report = compare_tables(a, b, [1, 2])
    assert not report['byte_identical'] and not report['changed_cells']
    assert compare_tables(a, None, [1, 2])['status'] == 'server_original_not_provided'


def test_no_usable_target_is_reported_without_inventing_a_noise_floor():
    frame = pd.DataFrame({'alloy_id': [1, 1, 2, 2], 'ncorr_curve_file': ['a', 'b', 'c', 'd'],
        'peak_stress_MPa': [10., 11., 20., 21.], 'endpoint_strain_pct': [np.nan]*4,
        'peak_strain_pct': [1., 1., 2., 2.], 'endpoint_stress_MPa': [8., 9., 18., 19.]})
    labels = pd.DataFrame({'alloy_id': [1, 2], 'peak_stress_MPa': [10.5, 20.5],
                           'endpoint_strain_pct': [np.nan, np.nan]})
    report, _, _ = trial_audit(frame, labels, [1, 2], bootstrap_count=10)
    assert report['targets']['endpoint_strain_pct']['status'] == 'no_finite_specimen_labels'
    assert report['targets']['peak_stress_MPa']['max_absolute_aggregation_difference'] == 0


def test_prediction_improvement_cannot_bypass_failed_permutation(monkeypatch):
    # Isolate decision logic: all permuted pairings improve just as much, so p=1.
    def predictions(c, d, y, mode, alphas, ratios, **kwargs):
        result = np.asarray(y)+(1. if mode == 'composition' else 0.)
        return (result, []) if kwargs.get('return_models') else result
    monkeypatch.setattr(ridge, 'nested_predict', predictions)
    report = ridge.validate_representation(np.ones((6, 2)), np.ones((6, 3)),
        np.arange(6.), np.arange(6), 'block_all', alphas=[1.], ratios=[1.], permutations=1000)
    assert report['v2_prediction_gate_passed']
    assert report['permutation_tail_probability'] == 1.
    assert not report['permutation_gate_passed'] and not report['cpu_proxy_gate_passed']
