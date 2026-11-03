"""Read-only specimen reliability and tabular provenance audits (no GPU)."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

TARGETS = ['peak_stress_MPa', 'endpoint_strain_pct']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def compare_tables(local_path, server_path, ids):
    if server_path is None:
        return {'status': 'server_original_not_provided',
                'local_sha256': sha256(local_path), 'semantic_equality_not_inferred_from_hash': True}
    a, b = (pd.read_csv(p).set_index('alloy_id') for p in (local_path, server_path))
    if not a.index.is_unique or not b.index.is_unique: raise ValueError('Duplicate table IDs')
    missing_a, missing_b = sorted(set(ids)-set(a.index)), sorted(set(ids)-set(b.index))
    common = sorted(set(a.columns)&set(b.columns))
    changes = []
    for i in sorted(set(ids)&set(a.index)&set(b.index)):
        for column in common:
            x, y = a.loc[i, column], b.loc[i, column]
            if pd.isna(x) and pd.isna(y): continue
            numeric = isinstance(x, (float, int, np.number)) and isinstance(y, (float, int, np.number))
            equal = bool(np.isclose(x, y, rtol=0, atol=1e-10)) if numeric else str(x) == str(y)
            if not equal: changes.append({'alloy_id': int(i), 'column': column, 'local': x, 'server': y})
    return {'status': 'compared', 'byte_identical': sha256(local_path) == sha256(server_path),
            'local_sha256': sha256(local_path), 'server_sha256': sha256(server_path),
            'missing_local_ids': missing_a, 'missing_server_ids': missing_b,
            'local_only_columns': sorted(set(a.columns)-set(b.columns)),
            'server_only_columns': sorted(set(b.columns)-set(a.columns)), 'changed_cells': changes,
            'scope': 'Development alloys only; held-out values not printed'}


def trial_audit(trials, alloy_labels, ids, *, terminal_reasons=None, bootstrap_count=1000, seed=20261102):
    required = {'alloy_id', 'ncorr_curve_file', *TARGETS, 'peak_strain_pct', 'endpoint_stress_MPa'}
    if not required <= set(trials): raise ValueError(f'Missing trial columns: {sorted(required-set(trials))}')
    if trials.duplicated(['alloy_id', 'ncorr_curve_file']).any(): raise ValueError('Duplicate specimen IDs')
    selected = trials[trials.alloy_id.isin(ids)].copy()
    if set(selected.alloy_id) != set(ids): raise ValueError('Missing development specimens')
    reasons = {}
    if terminal_reasons is not None:
        required = {'alloy_id', 'ncorr_curve_file', 'endpoint_reason', 'evidence'}
        if not required <= set(terminal_reasons): raise ValueError('Incomplete termination-reason table')
        if terminal_reasons.duplicated(['alloy_id', 'ncorr_curve_file']).any():
            raise ValueError('Duplicate termination-reason records')
        for row in terminal_reasons.to_dict('records'):
            if row['endpoint_reason'] not in ('confirmed_fracture', 'confirmed_correlation_loss', 'unknown'):
                raise ValueError('Unknown termination reason code')
            if row['endpoint_reason'] != 'unknown' and (pd.isna(row['evidence']) or not str(row['evidence']).strip()):
                raise ValueError('Confirmed termination reasons require evidence')
            reasons[(int(row['alloy_id']), row['ncorr_curve_file'])] = row
    quality = []
    for row in selected.to_dict('records'):
        peak, endpoint, peak_strain = row['peak_stress_MPa'], row['endpoint_strain_pct'], row['peak_strain_pct']
        reason = reasons.get((int(row['alloy_id']), row['ncorr_curve_file']), {})
        delta = float(endpoint-peak_strain) if np.isfinite(endpoint) and np.isfinite(peak_strain) else None
        quality.append({'alloy_id': int(row['alloy_id']), 'ncorr_curve_file': row['ncorr_curve_file'],
            'endpoint_reason': reason.get('endpoint_reason', 'unknown'), 'evidence': reason.get('evidence', ''),
            'endpoint_minus_peak_strain_pct_points': delta,
            'endpoint_equals_peak_flag': bool(abs(delta) < 1e-9) if delta is not None else None,
            'endpoint_before_peak_flag': bool(delta < -1e-9) if delta is not None else None,
            'peak_stress_nonpositive_or_missing': not (np.isfinite(peak) and peak > 0),
            'endpoint_segment_stress_over_peak': float(row['endpoint_stress_MPa']/peak) if np.isfinite(peak) and peak > 0 else None,
            'limits': 'Equality alone does not diagnose truncation; endpoint stress is a tail-window statistic'})
    rng, reports, summary_rows = np.random.default_rng(seed), {}, []
    labels = alloy_labels.set_index('alloy_id')
    if not labels.index.is_unique: raise ValueError('Duplicate alloy labels')
    for target in TARGETS:
        groups = {i: g[target].to_numpy(float) for i, g in selected.groupby('alloy_id', sort=True)}
        groups = {i: v[np.isfinite(v)] for i, v in groups.items()}
        groups = {i: v for i, v in groups.items() if len(v)}
        k, n = len(groups), sum(len(v) for v in groups.values())
        if not groups:
            reports[target] = {'status': 'no_finite_specimen_labels', 'development_trial_count': 0,
                               'development_alloy_count': 0, 'max_absolute_aggregation_difference': None,
                               'stored_target_missing_ids': list(ids)}
            continue
        medians = np.asarray([np.median(v) for v in groups.values()])
        between = float(np.var(medians, ddof=1)) if k > 1 else None
        within_ss = sum(float(((v-v.mean())**2).sum()) for v in groups.values())
        within_var = within_ss/(n-k) if n > k else None
        global_mean = sum(float(v.sum()) for v in groups.values())/n
        msb = sum(len(v)*(float(v.mean())-global_mean)**2 for v in groups.values())/(k-1) if k > 1 else None
        n0 = (n-sum(len(v)**2 for v in groups.values())/n)/(k-1) if k > 1 else None
        icc = ((msb-within_var)/(msb+(n0-1)*within_var)
               if msb is not None and within_var is not None and msb+(n0-1)*within_var > 0 else None)
        aggregate_differences, confirmed_medians = {}, {}
        for alloy, values in groups.items():
            median = float(np.median(values))
            stored = float(labels.loc[alloy, target])
            aggregate_differences[str(alloy)] = median-stored if np.isfinite(stored) else None
            samples = rng.choice(values, size=(bootstrap_count, len(values)), replace=True)
            median_sd = float(np.median(samples, axis=1).std(ddof=1)) if bootstrap_count > 1 else None
            subset = selected[selected.alloy_id == alloy]
            confirmed = [float(r[target]) for r in subset.to_dict('records')
                         if reasons.get((int(alloy), r['ncorr_curve_file']), {}).get('endpoint_reason') == 'confirmed_fracture'
                         and np.isfinite(r[target])]
            confirmed_medians[str(alloy)] = float(np.median(confirmed)) if confirmed else None
            summary_rows.append({'alloy_id': int(alloy), 'target': target, 'trial_count': len(values),
                'trial_mean': float(values.mean()), 'trial_median': median,
                'trial_std_ddof1': float(values.std(ddof=1)) if len(values) > 1 else None,
                'median_bootstrap_sd_conditional_on_observed_trials': median_sd,
                'confirmed_fracture_subset_median': confirmed_medians[str(alloy)]})
        reports[target] = {'development_trial_count': n, 'development_alloy_count': k,
            'pooled_within_trial_variance': within_var, 'between_alloy_median_variance': between,
            'within_to_between_variance_ratio': within_var/between if within_var is not None and between and between > 0 else None,
            'icc_1_1_unequal_group_approximation': icc,
            'aggregate_minus_stored_target': aggregate_differences,
            'max_absolute_aggregation_difference': max((abs(v) for v in aggregate_differences.values() if v is not None), default=None),
            'stored_target_missing_ids': [int(i) for i in ids if not np.isfinite(float(labels.loc[i, target]))],
            'confirmed_subset_medians': confirmed_medians,
            'limits': 'Repeatability combines specimen and measurement variation, is not an irreducible RMSE floor. '
                      'ICC assumes a one-way random-effects model; bootstrap misses unobserved/systematic DIC error.'}
    return {'status': 'descriptive_quality_audit_only', 'input_trial_rows': len(trials),
            'development_trial_rows': len(selected), 'heldout_or_other_rows_excluded': len(trials)-len(selected),
            'targets': reports, 'labels_modified': False}, pd.DataFrame(quality), pd.DataFrame(summary_rows)
