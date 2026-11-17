"""CPU feature extraction/nested ridge checks and generated-domain diagnostics.

These phases never update a generator and never create an overall M0 pass.
Timestep/trajectory and perturbation evidence must be obtained and reviewed
separately before writing a locked applicability decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch

from ebsd_diffusion.config import load_config, resolve_project_path
from ebsd_diffusion.data import resolve_manifest_image_path
from ebsd_diffusion.training_control import file_sha256
from physics.linear_mechanics import (SCHEMA, TARGET_NAMES, compare_proxy, nested_oof)
from physics.soft_mechanics_features import SoftMechanicsFeatures


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')


def nonoverwriting_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Refusing to overwrite: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract(config, output):
    cfg = load_config(config)
    m = cfg['mechanics_m0']
    root = nonoverwriting_directory(output)
    paths = {key: resolve_project_path(cfg, m[key]).resolve()
             for key in ('manifest', 'compositions', 'labels', 'source_conditioner')}
    conditioner = json.loads(paths['source_conditioner'].read_text('utf-8'))
    ids = sorted(conditioner['training_alloy_ids'])
    if m['exclude_alloy_id'] in ids:
        raise ValueError('Held-out alloy occurs in source conditioner')
    names = conditioner['active_features']
    manifest = pd.read_csv(paths['manifest']).set_index('alloy_id')
    composition = pd.read_csv(paths['compositions']).set_index('alloy_id')
    labels = pd.read_csv(paths['labels']).set_index('alloy_id')
    if not all(t.index.is_unique for t in (manifest, composition, labels)):
        raise ValueError('Duplicate alloy IDs')
    extractor = SoftMechanicsFeatures(**m.get('feature_settings', {}))
    size, count = int(m.get('patch_size', 256)), int(m.get('windows_per_alloy', 32))
    if count < 8:
        raise ValueError('Use at least 8 fixed window positions for K4/K8 diagnostics')
    rng = np.random.default_rng(m.get('seed', 20261102))
    rows, banks, boxes, nonoverlap, hashes = [], [], {}, [], {str(p): file_sha256(p) for p in paths.values()}
    for i in ids:
        path = resolve_manifest_image_path(paths['manifest'], manifest.loc[i, 'image_path'])
        hashes[str(path)] = file_sha256(path)
        with Image.open(path) as source:
            image = source.convert('RGB')
        if min(image.size) < size:
            raise ValueError('Source view is smaller than patch size')
        coords = [(int(rng.integers(image.width-size+1)), int(rng.integers(image.height-size+1)))
                  for _ in range(count)]
        values = []
        corners = [(0, 0), (image.width-size, image.height-size)]
        if image.width < 2*size and image.height < 2*size:
            raise ValueError('This sensitivity protocol requires two nonoverlapping windows')
        for x, y in coords + corners:
            array = np.asarray(image.crop((x, y, x+size, y+size)), dtype=np.float32).copy()
            tensor = torch.from_numpy(array).permute(2, 0, 1)[None]/127.5-1
            with torch.no_grad():
                values.append(extractor(tensor).numpy()[0])
        all_values = np.asarray(values)
        values = all_values[:count]
        nonoverlap.append(all_values[count:])
        banks.append(values)
        boxes[str(i)] = coords
        row = {'alloy_id': i, **{name: float(composition.loc[i, name]) for name in names},
               **dict(zip(extractor.names, values.mean(0).tolist()))}
        for name in TARGET_NAMES:
            value = labels.loc[i, name]
            valid = pd.notna(value) and np.isfinite(float(value))
            if name+'_valid' in labels:
                valid &= str(labels.loc[i, name+'_valid']).lower() in ('true', '1', 'yes')
            row[name] = float(value) if valid else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(root/'alloy_features.csv', index=False)
    np.savez(root/'window_features.npz', features=np.asarray(banks), alloy_ids=np.asarray(ids),
             nonoverlap_features=np.asarray(nonoverlap))
    write_json(root/'extraction.json', {'schema': 'mechanics_extraction_v1', 'training_alloy_ids': ids,
        'composition_names': names, 'feature_extractor': extractor.metadata(), 'patch_size': size,
        'windows_per_alloy': count, 'seed': m.get('seed', 20261102), 'positions': boxes,
        'input_sha256': hashes, 'labels_sha256': file_sha256(paths['labels']),
        'feature_table_sha256': file_sha256(root/'alloy_features.csv'),
        'window_bank_sha256': file_sha256(root/'window_features.npz'),
        'nonoverlap_rule': 'top-left and bottom-right; either horizontal or vertical separation verified',
        'endpoint_semantics': 'DIC endpoint proxy, not confirmed physical fracture'})
    return {'status': 'extracted_not_fitted', 'output': str(root)}


def fit(extraction, output, permutations=1000, seed=20261102):
    root = nonoverwriting_directory(output)
    source = Path(extraction)
    meta = json.loads((source/'extraction.json').read_text('utf-8'))
    if file_sha256(source/'alloy_features.csv') != meta['feature_table_sha256']:
        raise ValueError('Extracted feature table changed')
    for p, digest in meta['input_sha256'].items():
        if file_sha256(Path(p)) != digest:
            raise ValueError(f'Extraction input changed: {p}')
    frame = pd.read_csv(source/'alloy_features.csv')
    c = frame[meta['composition_names']].to_numpy(float)
    d = frame[meta['feature_extractor']['names']].to_numpy(float)
    reports, models = [], []
    rng = np.random.default_rng(seed)
    for name in TARGET_NAMES:
        valid = np.isfinite(frame[name].to_numpy(float))
        y, cc, dd = frame.loc[valid, name].to_numpy(float), c[valid], d[valid]
        if len(y) < 5:
            reports.append({'target': name, 'alloy_ids': frame.loc[valid, 'alloy_id'].tolist(),
                            'prediction_gate_passed': False, 'reason': 'fewer_than_five_usable_alloys',
                            'permutation_count': 0})
            width = c.shape[1]+d.shape[1]
            # Inactive head only, never an invented mechanics label or an activatable predictor.
            models.append({'input_mean': [0.]*width, 'input_scale': [1.]*width,
                           'coefficient': [0.]*width, 'target_mean': 0., 'target_scale': 1.,
                           'alpha': 1., 'disabled_reason': 'insufficient_labels'})
            continue
        report, model = compare_proxy(cc, dd, y)
        base_mse = report['rmse_composition']**2
        observed = base_mse-report['rmse_joint']**2
        null = []
        for _ in range(permutations):
            x = np.column_stack([cc, dd[rng.permutation(len(dd))]])
            prediction = nested_oof(x, y)
            null.append(float(base_mse-np.mean((prediction-y)**2)))
        report.update(target=name, alloy_ids=frame.loc[valid, 'alloy_id'].tolist(),
                      permutation_count=permutations,
                      permutation_tail_probability=(1+sum(v >= observed for v in null))/(permutations+1),
                      permutation_interpretation='unconditional pairing sensitivity, not conditional independence')
        # Prediction gate alone is not sufficient to enter feedback.
        reports.append(report)
        models.append(model)
    artifact = {'schema': SCHEMA, 'target_names': list(TARGET_NAMES),
        'composition_names': meta['composition_names'], 'feature_extractor': meta['feature_extractor'],
        'training_alloy_ids': meta['training_alloy_ids'], 'models': models,
        'approved_targets': [r['prediction_gate_passed'] and permutations >= 1000 for r in reports],
        'labels_sha256': meta['labels_sha256'],
        'input_sha256': {**meta['input_sha256'], str((source/'alloy_features.csv').resolve()): meta['feature_table_sha256']}}
    write_json(root/'proxy_model.json', artifact)
    write_json(root/'proxy_validation.json', {'status': 'prediction_checked_applicability_pending',
        'reports': reports, 'optimizer_updates': 0, 'proxy_sha256': file_sha256(root/'proxy_model.json')})
    write_json(root/'m0_decision.template.json', {'schema': 'mechanics_m0_decision_v1',
        'status': 'pending_review', 'proxy_sha256': file_sha256(root/'proxy_model.json'),
        'source_sha256': {}, 'vae_sha256': {}, 'active_targets': [False, False], 'group_size': 4,
        'max_t': None, 'gates': {key: {'status': 'pending', 'path': '', 'sha256': ''}
             for key in ('window_stability', 'perturbations', 'domain', 'timestep_and_trajectory')}})
    return {'status': 'proxy_fitted_m0_not_passed', 'output': str(root)}


def domain_distances(reference, queries):
    mean, scale = reference.mean(0), reference.std(0)
    scale = np.where(scale < 1e-10, 1., scale)
    r, q = (reference-mean)/scale, (queries-mean)/scale
    covariance = np.atleast_2d(np.cov(r, rowvar=False))
    covariance = .9*covariance + .1*np.eye(r.shape[1])
    inv = np.linalg.inv(covariance)
    nn = np.sqrt(((q[:, None, :]-r[None, :, :])**2).sum(-1)).min(1)
    mahal = np.sqrt(np.maximum(0, np.einsum('bi,ij,bj->b', q, inv, q)))
    return np.column_stack([nn, mahal])


def domain(extraction, generated_index, output, k=4, seed=20261102):
    root = nonoverwriting_directory(output)
    source = Path(extraction)
    meta = json.loads((source/'extraction.json').read_text('utf-8'))
    if file_sha256(source/'window_features.npz') != meta['window_bank_sha256']:
        raise ValueError('Real feature bank changed')
    bank = np.load(source/'window_features.npz', allow_pickle=False)
    ids, real = bank['alloy_ids'].tolist(), bank['features']
    center = real.mean(1)
    cutoffs = np.max(np.concatenate([domain_distances(np.delete(center, i, 0), center[i:i+1])
                                    for i in range(len(ids))]), axis=0)
    frame = pd.read_csv(generated_index)
    if not {'alloy_id', 'image_path', 'seed'} <= set(frame):
        raise ValueError('Generated index requires alloy_id,image_path,seed')
    frame = frame[frame.alloy_id.isin(ids)]  # held-out images are never opened
    extractor = SoftMechanicsFeatures(**meta['feature_extractor']['settings'])
    rng, rows, stability, hashes = np.random.default_rng(seed), [], [], {}
    for index, i in enumerate(ids):
        selected = frame[frame.alloy_id == i]
        if selected.seed.duplicated().any() or len(selected) < max(20, k):
            raise ValueError(f'Need at least 20 unique fixed seeds for alloy {i}')
        values = []
        for record in selected.to_dict('records'):
            path = Path(record['image_path'])
            if not path.is_absolute(): path = Path(generated_index).resolve().parent/path
            hashes[str(path.resolve())] = file_sha256(path)
            with Image.open(path) as im: array = np.asarray(im.convert('RGB'), dtype=np.float32).copy()
            if array.shape[:2] != (meta['patch_size'], meta['patch_size']):
                raise ValueError('Generated patch size differs from real extraction')
            with torch.no_grad():
                values.append(extractor(torch.from_numpy(array).permute(2, 0, 1)[None]/127.5-1).numpy()[0])
        values = np.asarray(values)
        def groups(v):
            return np.asarray([v[rng.choice(len(v), k, replace=False)].mean(0) for _ in range(32)])
        generated_groups, real_groups = groups(values), groups(real[index])
        distances = domain_distances(center, generated_groups)
        fraction = float(np.mean((distances <= cutoffs).all(1)))
        rows.append({'alloy_id': i, 'in_domain_fraction': fraction, 'passed': fraction >= .9,
                     'mean_distance_nn_mahal': distances.mean(0).tolist()})
        denominator = center.var(0)
        rr = np.where(denominator > 1e-12, real_groups.var(0)/np.maximum(denominator, 1e-12), np.inf)
        gr = np.where(denominator > 1e-12, generated_groups.var(0)/np.maximum(denominator, 1e-12), np.inf)
        nonoverlap_delta = ((bank['nonoverlap_features'][index].mean(0)-center[index]) /
                            np.where(center.std(0) < 1e-12, 1., center.std(0)))
        stability.append({'alloy_id': i, 'real_ratio': [float(v) if np.isfinite(v) else None for v in rr],
                          'generated_ratio': [float(v) if np.isfinite(v) else None for v in gr],
                          'nonoverlap_pair_mean_standardized_delta': nonoverlap_delta.tolist(),
                          'passed': bool((rr <= .25).all() and (gr <= .25).all())})
    common = {'training_alloy_ids': ids, 'group_size': k, 'seed': seed,
              'generated_input_sha256': hashes,
              'limits': 'Repeated groups overlap and are not independent samples; no physics validation'}
    write_json(root/'domain.json', {**common, 'status': 'passed' if all(r['passed'] for r in rows) else 'failed',
                                  'rows': rows, 'distance_cutoffs': cutoffs.tolist()})
    write_json(root/'window_stability.json', {**common,
        'status': 'nonoverlap_review_pending' if all(r['passed'] for r in stability) else 'failed', 'rows': stability,
        'nonoverlap_sensitivity': 'Two nonoverlapping windows are a sensitivity check, not two independent specimens'})
    return {'status': 'domain_checked_not_overall_m0_passed', 'output': str(root)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    p = sub.add_parser('extract'); p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('fit'); p.add_argument('--extraction', required=True); p.add_argument('--output', required=True)
    p.add_argument('--permutations', type=int, default=1000)
    p = sub.add_parser('domain'); p.add_argument('--extraction', required=True)
    p.add_argument('--generated-index', required=True); p.add_argument('--output', required=True)
    p.add_argument('--k', type=int, choices=[4, 8], default=4)
    args = vars(parser.parse_args()); phase = args.pop('phase')
    if args.get('permutations', 0) < 0: parser.error('permutations must be nonnegative')
    print(json.dumps(globals()[phase](**args), ensure_ascii=False))


if __name__ == '__main__':
    main()
