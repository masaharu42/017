"""Second-round, locked CPU-only mechanics audits. Never authorizes C0/C1.

prepare -> audit -> extract-hard -> validate -> decision. Each phase is explicit;
'all' executes the prepared CPU analysis, with no generator/CNN training.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from PIL import Image

from ebsd_diffusion.block_ridge import (candidates, composition_to_pc1, fit_selected, influence_diagnostics,
    nested_predict, rank_diagnostics, scores, tune, validate_representation)
from ebsd_diffusion.config import load_config, resolve_project_path
from ebsd_diffusion.hard_morphology import describe_hard, NAMES, LAYOUT_NAME, SCHEMA
from ebsd_diffusion.local_descriptors import validate_settings
from ebsd_diffusion.mechanics_cpu_audit import compare_tables, sha256, trial_audit, TARGETS

ROOT = Path(__file__).resolve().parents[1]
SOFT_NAMES = ['soft_boundary_fraction', 'soft_boundary_variation', 'interior_s2']
SOURCE_FILES = ['scripts/run_mechanics_m0_v2.py', 'ebsd_diffusion/block_ridge.py',
                'ebsd_diffusion/hard_morphology.py', 'ebsd_diffusion/mechanics_cpu_audit.py',
                'ebsd_diffusion/local_descriptors.py', 'ebsd_diffusion/config.py']


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')


def unique_frame(path):
    frame = pd.read_csv(path)
    frame['alloy_id'] = frame.alloy_id.astype(int)
    if frame.alloy_id.duplicated().any(): raise ValueError(f'Duplicate alloy ID: {path}')
    return frame.set_index('alloy_id')


def image_path(manifest, value):
    p = Path(value)
    if p.is_absolute(): return p.resolve()
    for candidate in (manifest.parent/p, manifest.parent.parent/p):
        if candidate.is_file(): return candidate.resolve()
    raise FileNotFoundError(f'Cannot resolve image: {value}')


def config_settings(path):
    cfg = load_config(path)
    settings = cfg['mechanics_m0_v2']
    if settings.get('soft_modes') != ['shared', 'block_pc1', 'block_all', 'image_only']:
        raise ValueError('Lock all four soft diagnostics; do not select only the best after seeing results')
    if settings.get('hard_variants') not in (['hard_base4'], ['hard_base4', 'hard_layout5']):
        raise ValueError('Register one primary hard version and at most one fixed layout variant')
    if type(settings['permutations']) is not int or settings['permutations'] < 0:
        raise ValueError('Invalid permutation count')
    if not 0 < settings['permutation_p_threshold'] < 1:
        raise ValueError('Invalid predeclared permutation threshold')
    if not 0 < settings['correlation_threshold'] < 1 or not 1 <= settings['max_hard_features'] <= 3:
        raise ValueError('Invalid hard-feature selection')
    if not 0 < settings['min_valid_window_fraction'] <= 1:
        raise ValueError('Invalid valid-window threshold')
    for mode in ('composition', 'block_pc1'):
        candidates(mode, settings['alphas'], settings['image_penalty_ratios'])
    validate_settings(settings['hard_settings'])
    if settings['bootstrap_count'] < 2: raise ValueError('Need at least two bootstrap repetitions')
    if settings['spatial_block'] < 1: raise ValueError('Invalid spatial block size')
    for key in ('extraction', 'output_root', 'manifest', 'compositions', 'labels', 'trials'):
        settings[key] = str(resolve_project_path(cfg, settings[key]).resolve())
    for key in ('terminal_reasons', 'server_manifest', 'server_compositions'):
        settings[key] = str(resolve_project_path(cfg, settings[key]).resolve()) if settings.get(key) else None
    return settings


def phase_directory(root, name):
    target = root/name
    if target.exists(): raise FileExistsError(f'Phase already exists; use a new registered output root: {target}')
    target.mkdir()
    return target


def prepare(path):
    settings = config_settings(path)
    root = Path(settings['output_root'])
    if root.exists() and any(root.iterdir()): raise FileExistsError('Do not overwrite an existing analysis')
    extraction = Path(settings['extraction'])
    meta = json.loads((extraction/'extraction.json').read_text('utf-8'))
    for name, key in [('alloy_features.csv', 'feature_table_sha256'), ('window_features.npz', 'window_bank_sha256')]:
        if sha256(extraction/name) != meta[key]: raise ValueError(f'Original extraction changed: {name}')
    ids = sorted(meta['training_alloy_ids'])
    if (len(ids) != len(set(ids)) or settings['exclude_alloy_id'] in ids
            or len(ids) != settings['expected_training_alloys']):
        raise ValueError('Wrong development fold')
    if not settings.get('software_fixture') and (len(ids) != 21 or settings['permutations'] < 1000):
        raise ValueError('Scientific run requires all 21 development alloys and >=1000 permutations')
    names = meta['composition_names']
    soft = unique_frame(extraction/'alloy_features.csv').loc[ids]
    composition, labels = unique_frame(settings['compositions']), unique_frame(settings['labels'])
    if not np.allclose(soft[names], composition.loc[ids, names], atol=1e-10, rtol=0):
        raise ValueError('Composition values differ from original feature extraction')
    if not np.allclose(soft[TARGETS], labels.loc[ids, TARGETS], equal_nan=True, atol=1e-10, rtol=0):
        raise ValueError('Labels changed; register and extract a separate QC version first')
    manifest_path = Path(settings['manifest'])
    manifest = unique_frame(manifest_path)
    images, image_checks = {}, []
    for alloy in ids:
        record = str(manifest.loc[alloy, 'image_path'])
        image = image_path(manifest_path, record)
        suffix = '/'+record.replace('\\', '/').lstrip('/')
        expected = [digest for name, digest in meta['input_sha256'].items() if name.endswith(suffix)]
        if len(expected) != 1:
            # Explicit project relocation is permitted, but not a different per-alloy image.
            suffix = '/'+str(Path(record).parent.name)+'/'+Path(record).name
            expected = [digest for name, digest in meta['input_sha256'].items() if name.endswith(suffix)]
        if len(expected) != 1 or sha256(image) != expected[0]:
            raise ValueError(f'Alloy {alloy}: source image differs from original extraction')
        images[str(alloy)] = str(image)
        image_checks.append({'alloy_id': alloy, 'image_sha256': expected[0], 'matched_original': True})
    for priority, names_expected in [('base_priority', NAMES), ('layout_priority', NAMES+[LAYOUT_NAME])]:
        if sorted(settings[priority]) != sorted(names_expected): raise ValueError('Invalid fixed feature priority')
    files = [Path(path).resolve(), extraction/'extraction.json', extraction/'alloy_features.csv',
             extraction/'window_features.npz', *(Path(settings[k]) for k in ('manifest', 'compositions', 'labels', 'trials')),
             *(Path(v) for v in images.values()), *(ROOT/p for p in SOURCE_FILES)]
    files += [Path(settings[k]) for k in ('terminal_reasons', 'server_manifest', 'server_compositions') if settings[k]]
    hashes = {str(p): sha256(p) for p in files}
    root.mkdir(parents=True, exist_ok=True)
    plan = {'schema': 'mechanics_cpu_v2_plan', 'status': 'registered_not_executed',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'development_round': 2,
        'post_v1_development': True, 'settings': settings, 'training_alloy_ids': ids,
        'composition_names': names, 'images': images, 'source_images': image_checks,
        'input_and_code_sha256': hashes, 'config_path': str(Path(path).resolve()),
        'runtime': {'python': sys.version, 'platform': platform.platform(),
                    'packages': {name: version(name) for name in
                                 ('numpy', 'pandas', 'scipy', 'scikit-image', 'Pillow', 'PyYAML')}},
        'limits': 'Pre-registration after v1, not independent confirmatory evidence; no feedback permission'}
    write_json(root/'plan.json', plan)
    (root/'plan.sha256').write_text(sha256(root/'plan.json')+'  plan.json\n', 'utf-8')
    return plan


def load_plan(path):
    settings = config_settings(path)
    root = Path(settings['output_root'])
    plan = json.loads((root/'plan.json').read_text('utf-8'))
    if sha256(root/'plan.json') != (root/'plan.sha256').read_text('utf-8').split()[0]:
        raise ValueError('Registered plan changed')
    if plan['settings'] != settings: raise ValueError('Configuration changed after prepare')
    for p, digest in plan['input_and_code_sha256'].items():
        if sha256(p) != digest: raise ValueError(f'Input/code changed after prepare: {p}')
    return root, plan


def soft_inputs(plan):
    s = plan['settings']
    table = unique_frame(Path(s['extraction'])/'alloy_features.csv').loc[plan['training_alloy_ids']]
    return table, table[plan['composition_names']].to_numpy(float), table[SOFT_NAMES].to_numpy(float)


def audit(path):
    root, plan = load_plan(path); output = phase_directory(root, 'audit')
    s, ids = plan['settings'], plan['training_alloy_ids']
    table, c, d = soft_inputs(plan)
    targets = {}
    for target in TARGETS:
        y = table[target].to_numpy(float)
        if not np.isfinite(y).all():
            targets[target] = {'status': 'missing_label_requires_review'}; continue
        variants = {}
        for mode in ['composition', *s['soft_modes']]:
            pred, models = nested_predict(c, d, y, mode, s['alphas'], s['image_penalty_ratios'], return_models=True)
            pair = tune(c, d, y, mode, s['alphas'], s['image_penalty_ratios'])
            full = fit_selected(c, d, y, mode, pair)
            info = {'scores': scores(y, pred), 'oof': pred.tolist(), 'full_fit': full.metadata(),
                    'full_influence': influence_diagnostics(full, c, d, y), 'id21': None}
            if 21 in ids:
                i = ids.index(21); keep = np.arange(len(ids)) != i
                model = models[i]
                means, stds = c[keep].mean(0), c[keep].std(0)
                dz_mean, dz_std = d[keep].mean(0), d[keep].std(0)
                info['id21'] = {'true': float(y[i]), 'prediction': float(pred[i]),
                    'oof_residual_in_training_target_sd': float((pred[i]-y[i])/model.target_scale),
                    'decomposition': model.contributions(c[i:i+1], d[i:i+1]),
                    'composition_z': dict(zip(plan['composition_names'], ((c[i]-means)/np.where(stds < 1e-12, 1., stds)).tolist())),
                    'image_z': dict(zip(SOFT_NAMES, ((d[i]-dz_mean)/np.where(dz_std < 1e-12, 1., dz_std)).tolist())),
                    'composition_outside_other_alloy_range': dict(zip(plan['composition_names'], ((c[i] < c[keep].min(0)) | (c[i] > c[keep].max(0))).tolist())),
                    'outer_query_design': influence_diagnostics(model, c[keep], d[keep], y[keep], c[i:i+1], d[i:i+1]),
                    'outer_fold_model': model.metadata()}
            variants[mode] = info
        targets[target] = variants
    rank = {'all_training': rank_diagnostics(d),
            'outer_training': {str(i): rank_diagnostics(np.delete(d, index, axis=0)) for index, i in enumerate(ids)},
            'pearson': table[SOFT_NAMES].corr().to_dict(), 'spearman': table[SOFT_NAMES].corr(method='spearman').to_dict(),
            'composition_to_pc1': composition_to_pc1(c, d, s['alphas'])}
    write_json(output/'soft_diagnostics.json', {'status': 'diagnostic_not_admission', 'targets': targets,
        'alloy_ids': ids, 'rank': rank, 'labels_or_outliers_modified': False})
    labels, trials = pd.read_csv(s['labels']), pd.read_csv(s['trials'])
    terminal = pd.read_csv(s['terminal_reasons']) if s['terminal_reasons'] else None
    report, quality, repeated = trial_audit(trials, labels, ids, terminal_reasons=terminal,
                                           bootstrap_count=s['bootstrap_count'], seed=s['seed'])
    write_json(output/'repeatability.json', report)
    quality.to_csv(output/'trial_quality.csv', index=False)
    repeated.to_csv(output/'trial_target_summary.csv', index=False)
    pd.DataFrame([{'item': k, 'status': 'unknown_pending_original_records', 'evidence_path': ''} for k in
        ['ID21_composition_source_and_units', 'ID21_load_area_dic_records', 'ID21_ebsd_step_and_field',
         'ID21_heat_treatment', 'ID21_specimen_view_correspondence']]).to_csv(output/'id21_originals_checklist.csv', index=False)
    table_checks = {}
    meta = json.loads((Path(s['extraction'])/'extraction.json').read_text('utf-8'))
    for key in ('manifest', 'compositions'):
        result = compare_tables(s[key], s['server_'+key], ids)
        expected = [digest for name, digest in meta['input_sha256'].items() if name.endswith('/data/'+key+'.csv')]
        result['recorded_server_sha256'] = expected[0] if len(expected) == 1 else None
        result['matches_recorded_server_bytes'] = sha256(s[key]) == expected[0] if len(expected) == 1 else None
        table_checks[key] = result
    write_json(output/'table_comparison.json', table_checks)
    record_phase(root, 'audit')
    return {'status': 'cpu_diagnostics_complete_feedback_not_authorized'}


def extract_hard(path):
    root, plan = load_plan(path); output = phase_directory(root, 'hard')
    s, ids = plan['settings'], plan['training_alloy_ids']
    meta = json.loads((Path(s['extraction'])/'extraction.json').read_text('utf-8'))
    table, _, _ = soft_inputs(plan)
    layout = 'hard_layout5' in s['hard_variants']
    names = NAMES+([LAYOUT_NAME] if layout else [])
    rows, bank, valid_bank, quality_rows = [], [], [], []
    size = meta['patch_size']
    for alloy in ids:
        with Image.open(plan['images'][str(alloy)]) as im: image = im.convert('RGB')
        features, valid = [], []
        for index, (x, y) in enumerate(meta['positions'][str(alloy)]):
            if not 0 <= x <= image.width-size or not 0 <= y <= image.height-size:
                raise ValueError('Locked crop outside source image')
            pixels = np.asarray(image.crop((x, y, x+size, y+size)), dtype=np.uint8)
            values, ok, q = describe_hard(pixels, s['hard_settings'], spatial_block=s['spatial_block'], include_layout=layout)
            features.append(values); valid.append(ok)
            quality_rows.append({'alloy_id': alloy, 'window_index': index, 'x': x, 'y': y,
                                 'valid': ok, **q})
        features, valid = np.asarray(features), np.asarray(valid)
        enough = float(valid.mean()) >= s['min_valid_window_fraction']
        mean = features[valid].mean(0) if enough and valid.any() else np.full(len(names), np.nan)
        bank.append(features); valid_bank.append(valid)
        rows.append({'alloy_id': alloy, **table.loc[alloy, plan['composition_names']+TARGETS].to_dict(),
                     **dict(zip(names, mean.tolist())), 'valid_window_fraction': float(valid.mean()),
                     'quality_gate_passed': enough})
    pd.DataFrame(rows).to_csv(output/'alloy_hard_features.csv', index=False)
    pd.DataFrame(quality_rows).to_csv(output/'window_quality.csv', index=False)
    np.savez(output/'window_hard_features.npz', features=np.asarray(bank), valid=np.asarray(valid_bank), alloy_ids=ids)
    write_json(output/'hard_metadata.json', {'schema': SCHEMA, 'names': names,
        'settings': s['hard_settings'], 'spatial_block': s['spatial_block'], 'training_alloy_ids': ids,
        'aggregation': 'Mean transformed descriptor over valid locked windows; no alloy deletion',
        'min_valid_window_fraction': s['min_valid_window_fraction'],
        'pixel_units_only': True, 'physical_grain_labels': False})
    record_phase(root, 'hard')
    return {'status': 'hard_features_extracted_not_validated'}


def record_phase(root, phase):
    files = {str(p.relative_to(root)): sha256(p) for p in sorted((root/phase).glob('*')) if p.is_file()}
    write_json(root/(phase+'_phase.json'), {'schema': 'cpu_phase_outputs_v1', 'files': files})


def verify_phase(root, phase):
    record = json.loads((root/(phase+'_phase.json')).read_text('utf-8'))
    for p, digest in record['files'].items():
        if sha256(root/p) != digest: raise ValueError(f'Phase output changed: {p}')


def validate(path):
    root, plan = load_plan(path)
    verify_phase(root, 'audit'); verify_phase(root, 'hard')
    output = phase_directory(root, 'validation')
    s, ids = plan['settings'], plan['training_alloy_ids']
    soft, c, d = soft_inputs(plan)
    hard = unique_frame(root/'hard/alloy_hard_features.csv').loc[ids]
    all_reports = []
    for target in TARGETS:
        y = soft[target].to_numpy(float)
        if not np.isfinite(y).all():
            all_reports.append({'target': target, 'status': 'missing_labels_full_fold_required'}); continue
        for mode in s['soft_modes']:
            print(f'CPU nested validation: {target} / {mode}', flush=True)
            report = validate_representation(c, d, y, ids, mode, alphas=s['alphas'], ratios=s['image_penalty_ratios'],
                permutations=s['permutations'], seed=s['seed'], non_degradation=s['additional_non_degradation'],
                permutation_p_threshold=s['permutation_p_threshold'])
            report.update(target=target, representation=mode, status='validated_not_feedback_authorized')
            filename = target+'__'+mode+'.json'; write_json(output/filename, report)
            all_reports.append({k: v for k, v in report.items() if k != 'outer_fold_models'})
        for variant in s['hard_variants']:
            names = NAMES+([LAYOUT_NAME] if variant == 'hard_layout5' else [])
            values = hard[names].to_numpy(float)
            if not np.isfinite(values).all() or not hard.quality_gate_passed.all():
                all_reports.append({'target': target, 'representation': variant,
                                    'status': 'hard_window_quality_failed_no_alloy_dropped',
                                    'v2_prediction_gate_passed': False}); continue
            priority = s['layout_priority'] if variant == 'hard_layout5' else s['base_priority']
            transform = {'priority': [names.index(name) for name in priority],
                         'correlation_threshold': s['correlation_threshold'], 'max_features': s['max_hard_features']}
            print(f'CPU nested validation: {target} / {variant}', flush=True)
            report = validate_representation(c, values, y, ids, 'hard', alphas=s['alphas'], ratios=s['image_penalty_ratios'],
                transform_settings=transform, permutations=s['permutations'], seed=s['seed'],
                non_degradation=s['additional_non_degradation'], permutation_p_threshold=s['permutation_p_threshold'])
            report.update(target=target, representation=variant, feature_names=names, status='validated_not_feedback_authorized')
            write_json(output/(target+'__'+variant+'.json'), report)
            all_reports.append({k: v for k, v in report.items() if k != 'outer_fold_models'})
    write_json(output/'summary.json', {'status': 'prediction_checked_not_m0_passed',
        'development_round': 2, 'all_reports': all_reports,
        'selection_limits': 'All registered candidates reported; repeated use of this fold is exploratory',
        'generator_updates': 0, 'gpu_used': False})
    record_phase(root, 'validation')
    return {'status': 'cpu_validation_complete_feedback_not_authorized'}


def decision(path):
    root, plan = load_plan(path)
    for phase in ('audit', 'hard', 'validation'): verify_phase(root, phase)
    destination = root/'cpu_decision.json'
    if destination.exists(): raise FileExistsError('Do not overwrite an existing decision')
    summary = json.loads((root/'validation/summary.json').read_text('utf-8'))
    s = plan['settings']
    qualified = [r for r in summary['all_reports'] if r.get('cpu_proxy_gate_passed')]
    repeat = json.loads((root/'audit/repeatability.json').read_text('utf-8'))
    data_issues = [name for name, t in repeat['targets'].items()
                   if t['max_absolute_aggregation_difference'] is None
                   or t['max_absolute_aggregation_difference'] > 1e-6
                   or t['development_alloy_count'] != len(plan['training_alloy_ids'])
                   or t.get('stored_target_missing_ids')]
    hard = [r for r in qualified if r.get('representation', '').startswith('hard_') and r['target'] not in data_issues]
    soft = [r for r in qualified if r.get('representation') in ('block_pc1', 'block_all') and r['target'] not in data_issues]
    quality = unique_frame(root/'hard/alloy_hard_features.csv')
    if s.get('software_fixture') or s['permutations'] < 1000:
        next_step = 'software_fixture_no_scientific_decision'
    elif not quality.quality_gate_passed.all(): next_step = 'need_data_or_segmentation_review'
    elif hard: next_step = 'consider_descriptor_distillation_after_review'
    elif soft: next_step = 'review_regularized_soft_proxy_before_remaining_m0'
    elif data_issues: next_step = 'need_data_or_segmentation_review'
    else: next_step = 'stop_registered_feedback_route_analysis_only'
    value = {'schema': 'mechanics_cpu_v2_decision_not_training_permission',
        'status': 'cpu_analysis_complete_feedback_not_authorized', 'next_step': next_step,
        'qualified_candidates': [{'target': r['target'], 'representation': r['representation']} for r in qualified],
        'target_data_issues': data_issues,
        'training_authorized': False, 'cnn_training_authorized': False,
        'no_id21_deletion': True, 'heldout_excluded': s['exclude_alloy_id'],
        'plan_sha256': sha256(root/'plan.json'), 'summary_sha256': sha256(root/'validation/summary.json'),
        'limits': 'A prediction gate is not domain/gradient/physics validation. Hard RGB descriptors are non-differentiable. '
                  'No C0/C1-compatible proxy artifact is exported. New-fold confirmation remains necessary.'}
    write_json(destination, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/mechanics_m0_v2_cpu.yaml')
    parser.add_argument('--phase', required=True, choices=['prepare', 'audit', 'extract-hard', 'validate', 'decision', 'all'])
    args = parser.parse_args()
    functions = {'prepare': prepare, 'audit': audit, 'extract-hard': extract_hard, 'validate': validate, 'decision': decision}
    phases = ['audit', 'extract-hard', 'validate', 'decision'] if args.phase == 'all' else [args.phase]
    for phase in phases:
        value = functions[phase](args.config)
        print(json.dumps(clean({'phase': phase, 'result': value if phase != 'prepare' else
                         {'status': value['status'], 'plan_sha256': sha256(Path(value['settings']['output_root'])/'plan.json')}}),
                         ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__': main()
