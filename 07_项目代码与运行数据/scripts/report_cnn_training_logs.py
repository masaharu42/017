"""Collect a read-only, sampled log audit; no model weights or datasets copied."""
import csv
import json
import math
import statistics
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/mechanics_cnn_log_review_v1'
ARMS = {'C0': ('mechanics_cnn_500_low_v1', 'C0'),
        'C1_0.01': ('mechanics_cnn_500_low_v1', 'C1'),
        'C1_0.03': ('mechanics_cnn_500_high_v1', 'C1')}

def readcsv(p):
    if not p.is_file(): return []
    with p.open(newline='', encoding='utf-8') as f: return list(csv.DictReader(f))

def number(value):
    try: x = float(value)
    except (ValueError, TypeError): return None
    return x if math.isfinite(x) else None

def summary(values):
    v = [x for x in values if x is not None]
    return {'count':len(v), 'min':min(v), 'median':statistics.median(v),
            'max':max(v), 'mean':statistics.mean(v)} if v else None

def main():
    if OUT.exists(): raise FileExistsError(f'Output already exists: {OUT}')
    OUT.mkdir(parents=True)
    report = {'scope': 'Logged batches only; no optimizer or model changes.',
              'limitations': ['training_metrics.csv does not contain mechanics loss in this training code',
                  'mechanics_gradients.csv is sampled at log intervals, not every update',
                  'training losses are stochastic batches, not held-out evaluation'], 'arms': {}}
    fingerprints = {}
    for label, (experiment, arm) in ARMS.items():
        base=ROOT/'output'/experiment/arm
        dst=OUT/label; dst.mkdir()
        paths=['training_metrics.csv','mechanics_gradients.csv','input_fingerprints.csv',
               'training_diagnostics.json','run_progress.json','initialization_metadata.json',
               'final/run_metadata.json']
        missing=[]
        for rel in paths:
            src=base/rel
            if src.is_file():
                target=dst/rel; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src,target)
            else: missing.append(rel)
        cal=base.parent/'calibration/mechanics_calibration.json'
        if cal.is_file(): shutil.copy2(cal,dst/'mechanics_calibration.json')
        metrics=readcsv(base/'training_metrics.csv'); grads=readcsv(base/'mechanics_gradients.csv')
        fp=readcsv(base/'input_fingerprints.csv')
        fingerprints[label]={r['additional_step']: r['sha256'] for r in fp}
        active=[r for r in grads if (number(r.get('groups')) or 0)>0]
        weight={'C0':0.,'C1_0.01':.01,'C1_0.03':.03}[label]
        ratios=[]
        for row in active:
            mse=number(row.get('mse_norm')); mech=number(row.get('mechanics_norm'))
            if mse is not None and mse>0 and mech is not None: ratios.append(weight*mech/mse)
        meta_path=base/'final/run_metadata.json'
        meta=json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        report['arms'][label]={'missing_files':missing,'final_global_step':meta.get('global_step'),
            'metadata_lambda':(meta.get('mechanics') or {}).get('lambda'),
            'metrics_logged_rows':len(metrics), 'gradient_logged_rows':len(grads),
            'active_logged_rows':len(active), 'active_groups_in_logged_rows':sum(number(r.get('groups')) or 0 for r in grads),
            'noise_loss_logged':summary([number(r.get('noise_loss')) for r in metrics]),
            'noise_loss_first_five':summary([number(r.get('noise_loss')) for r in metrics[:5]]),
            'noise_loss_last_five':summary([number(r.get('noise_loss')) for r in metrics[-5:]]),
            'weighted_gradient_ratio_active_logged':summary(ratios),
            'mechanics_mse_cosine_active_logged':summary([number(r.get('mechanics_mse_cosine')) for r in active]),
            'out_of_range_fraction_active_logged':summary([number(r.get('decoded_outside_fraction')) for r in active]),
            'mechanics_loss_available_in_training_metrics': bool(metrics and 'mechanics_loss' in metrics[0])}
    common=set.intersection(*(set(x) for x in fingerprints.values())) if all(fingerprints.values()) else set()
    mismatches=[s for s in sorted(common,key=int) if len({fingerprints[k][s] for k in ARMS})!=1]
    report['fingerprints']={'common_recorded_steps':len(common), 'mismatched_steps':mismatches,
                            'step_sets_identical':len({tuple(sorted(x)) for x in fingerprints.values()})==1 and bool(common)}
    (OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    print('Review files saved to:',OUT)

if __name__=='__main__': main()
