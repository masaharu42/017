"""Real-image grouped CNN training, nested alloy validation, reviewed refit.

This command never loads or updates a diffusion generator. CV does not approve
feedback. All normalization/checkpoint selection is inside the outer fold.
"""
from pathlib import Path
import argparse
import json
import random
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F

from ebsd_diffusion.config import load_config, resolve_project_path
from physics.cnn_mechanics import SingleScaleCNN, SCHEMA, TARGETS, sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['scripts/train_mechanics_cnn.py', 'physics/cnn_mechanics.py', 'ebsd_diffusion/config.py']
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def settings(path):
    cfg = load_config(path); s = cfg['mechanics_cnn']
    for key in ('extraction', 'manifest', 'labels', 'output_root'):
        s[key] = str(resolve_project_path(cfg, s[key]).resolve())
    for k in ('patch_size', 'inner_folds', 'max_epochs', 'groups_per_epoch', 'batch_groups', 'group_size'):
        if type(s[k]) is not int or s[k] < 1:
            raise ValueError('Invalid '+k)
    if s['batch_groups'] < 2 or s['groups_per_epoch'] % s['batch_groups']:
        raise ValueError('Balanced batches require >=2 alloy groups and divisible group budget')
    if (not s['seeds'] or len(set(s['seeds'])) != len(s['seeds']) or s['learning_rate'] <= 0):
        raise ValueError('Invalid fixed seeds or learning rate')
    screen=s['prediction_screen']
    if not 0 <= screen['minimum_ensemble_r2'] < 1 or screen['minimum_improved_alloys'] < 1:
        raise ValueError('Invalid registered prediction screen')
    return s


def prepare(path):
    s = settings(path); root = Path(s['output_root'])
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Choose a new output root')
    extraction = Path(s['extraction']); meta = json.loads((extraction/'extraction.json').read_text())
    ids = sorted(meta['training_alloy_ids'])
    if len(ids) != len(set(ids)) or s['exclude_alloy_id'] in ids or len(ids) != s['expected_alloys']:
        raise ValueError('Wrong development fold')
    if not s['software_fixture'] and (len(ids) != 21 or s['exclude_alloy_id'] != 3):
        raise ValueError('Scientific development fold must keep all 21 alloys, excluding ID03')
    if s['inner_folds'] < 2 or s['inner_folds'] > len(ids)-2:
        raise ValueError('Invalid grouped inner folds')
    if meta['patch_size'] != s['patch_size']:
        raise ValueError('Reuse native locked crop size')
    old = pd.read_csv(extraction/'alloy_features.csv').set_index('alloy_id')
    labels = pd.read_csv(s['labels']).set_index('alloy_id')
    manifest = pd.read_csv(s['manifest']).set_index('alloy_id')
    if not labels.index.is_unique or not manifest.index.is_unique or not old.index.is_unique:
        raise ValueError('Duplicate alloy IDs')
    if sha256(extraction/'alloy_features.csv') != meta['feature_table_sha256']:
        raise ValueError('Original feature table changed')
    y = labels.loc[ids, TARGETS].to_numpy(float)
    if not np.isfinite(y).all() or not np.allclose(y, old.loc[ids, TARGETS], atol=1e-10, rtol=0):
        raise ValueError('Missing/changed labels: register label QC separately')
    records, hashes = {}, {}
    for i in ids:
        p = Path(str(manifest.loc[i, 'image_path']))
        if not p.is_absolute():
            choices = [Path(s['manifest']).parent/p, Path(s['manifest']).parent.parent/p]
            p = next((v for v in choices if v.is_file()), choices[0])
        p = p.resolve(); digest = sha256(p)
        suffix = '/'+p.parent.name+'/'+p.name
        expected = [v for k,v in meta['input_sha256'].items() if k.endswith(suffix)]
        if len(expected) != 1 or digest != expected[0]:
            raise ValueError('Source image differs from original extraction')
        with Image.open(p) as im:
            if min(im.size) < s['patch_size']:
                raise ValueError('Image smaller than native crop')
            for x, yy in meta['positions'][str(i)]:
                if not (0 <= x <= im.width-s['patch_size'] and 0 <= yy <= im.height-s['patch_size']):
                    raise ValueError('Evaluation crop out of bounds')
        records[str(i)] = dict(image=str(p), target=labels.loc[i, TARGETS].tolist(),
                               positions=meta['positions'][str(i)])
        hashes[str(p)] = digest
    for p in [Path(path).resolve(), Path(s['manifest']), Path(s['labels']),
              extraction/'extraction.json', extraction/'alloy_features.csv', *(ROOT/f for f in SOURCES)]:
        hashes[str(p)] = sha256(p)
    root.mkdir(parents=True, exist_ok=True)
    plan = dict(schema='image_cnn_training_plan_v1', settings=s, alloy_ids=ids, records=records,
                hashes=hashes, post_previous_results_development_round=3,
                pixel_transform='native_rgb_minus_one_one', augmentation='random_native_crop_only',
                runtime=dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda),
                training_authorized=False, scope='Predictor training only; no diffusion authorization')
    dump(root/'plan.json', plan); (root/'plan.sha256').write_text(sha256(root/'plan.json'))
    return plan


def load(path):
    s = settings(path); root = Path(s['output_root'])
    if sha256(root/'plan.json') != (root/'plan.sha256').read_text().strip():
        raise ValueError('Registered plan changed')
    plan = json.loads((root/'plan.json').read_text())
    if plan['settings'] != s:
        raise ValueError('Config changed after prepare')
    for p, digest in plan['hashes'].items():
        if sha256(p) != digest:
            raise ValueError('Input/code changed: '+p)
    return root, plan


class Windows:
    def __init__(self, plan):
        self.plan = plan; self.images = {}
        for i, r in plan['records'].items():
            with Image.open(r['image']) as im:
                self.images[int(i)] = np.asarray(im.convert('RGB')).copy()

    def tensor(self, i, positions):
        size = self.plan['settings']['patch_size']; image = self.images[i]
        crops = np.stack([image[y:y+size, x:x+size] for x, y in positions])
        return torch.from_numpy(crops).permute(0, 3, 1, 2).float()/127.5-1

    def random(self, i, k, rng):
        h, w = self.images[i].shape[:2]; s = self.plan['settings']['patch_size']
        return self.tensor(i, [(int(rng.integers(w-s+1)), int(rng.integers(h-s+1))) for _ in range(k)])

    def labels(self, ids):
        return np.asarray([self.plan['records'][str(i)]['target'] for i in ids], dtype=np.float32)


@torch.no_grad()
def predict(model, windows, ids, mean, scale, device):
    model.eval(); result = []
    for i in ids:
        xx = windows.tensor(i, windows.plan['records'][str(i)]['positions'])
        pp = torch.cat([model(x.to(device)).cpu() for x in xx.split(8)])
        result.append((pp.mean(0).numpy()*scale+mean).tolist())
    return np.asarray(result)


def fit(windows, train_ids, seed, epochs, validation_ids=None):
    s = windows.plan['settings']; device = torch.device(s['device'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    y = windows.labels(train_ids); mean = y.mean(0); scale = y.std(0)
    scale = np.where(scale < 1e-8, 1., scale)
    model = SingleScaleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=s['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=s['max_epochs'])
    trace = []
    for epoch in range(epochs):
        model.train()
        if len(train_ids) < s['batch_groups']:
            raise ValueError('Need enough training alloys for distinct-alloy batches')
        total = 0.
        for start in range(0, s['groups_per_epoch'], s['batch_groups']):
            ids = rng.choice(train_ids, size=s['batch_groups'], replace=False).tolist()
            x = torch.cat([windows.random(i, s['group_size'], rng) for i in ids]).to(device)
            target = torch.tensor((windows.labels(ids)-mean)/scale, device=device)
            output = model(x).reshape(len(ids), s['group_size'], 2).mean(1)
            loss = F.mse_loss(output, target)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite CNN loss')
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.detach())
        scheduler.step()
        item = dict(epoch=epoch+1, training_loss=total/(s['groups_per_epoch']/s['batch_groups']))
        if validation_ids:
            pp = predict(model, windows, validation_ids, mean, scale, device)
            item['validation_loss'] = float(np.mean(((pp-windows.labels(validation_ids))/scale)**2))
        trace.append(item)
    return model.cpu().eval(), mean, scale, trace


def train_outer(root, plan, windows, alloy, seed):
    s = plan['settings']; destination = root/'cv'/f'alloy_{alloy:02d}_seed_{seed}'
    if (destination/'result.json').exists():
        result = json.loads((destination/'result.json').read_text())
        if result['plan_sha256'] != sha256(root/'plan.json') or sha256(destination/'model.pt') != result['weights_sha256']:
            raise ValueError('Completed fold changed')
        return
    if destination.exists(): raise FileExistsError('Incomplete fold preserved; use a new output root')
    destination.mkdir(parents=True)
    train_ids = [i for i in plan['alloy_ids'] if i != alloy]
    partitions = np.array_split(np.random.default_rng(seed).permutation(train_ids), s['inner_folds'])
    epochs, inner = [], []
    for j, part in enumerate(partitions):
        val = part.tolist(); train = [i for i in train_ids if i not in val]
        _, _, _, trace = fit(windows, train, seed+j+1, s['max_epochs'], val)
        best = min(trace, key=lambda r:r['validation_loss'])['epoch']; epochs.append(best)
        inner.append(dict(train_ids=train, validation_ids=val, selected_epoch=best, trace=trace))
    epoch = max(1, int(np.median(epochs)))
    model, mean, scale, trace = fit(windows, train_ids, seed, epoch)
    prediction = predict(model, windows, [alloy], mean, scale, torch.device('cpu'))[0]
    torch.save(model.state_dict(), destination/'model.pt')
    dump(destination/'result.json', dict(alloy_id=alloy, seed=seed, train_ids=train_ids,
        target=windows.labels([alloy])[0].tolist(), prediction=prediction.tolist(),
        mean_baseline=mean.tolist(), target_mean=mean.tolist(), target_scale=scale.tolist(),
        selected_epoch=epoch, inner=inner, refit_trace=trace,
        plan_sha256=sha256(root/'plan.json'), weights_sha256=sha256(destination/'model.pt')))


def metric(y, p):
    errors = p-y; den = ((y-y.mean(0))**2).sum(0)
    return dict(rmse=np.sqrt(np.mean(errors**2, 0)).tolist(), mae=np.abs(errors).mean(0).tolist(),
                r2=[float(1-v/d) if d > 1e-12 else None for v,d in zip((errors**2).sum(0), den)])


def collect(root, plan):
    if (root/'summary.json').exists(): raise FileExistsError('Do not overwrite summary')
    rows, hashes = [], {}
    for seed in plan['settings']['seeds']:
        for alloy in plan['alloy_ids']:
            p = root/'cv'/f'alloy_{alloy:02d}_seed_{seed}'/'result.json'
            row = json.loads(p.read_text())
            if row['plan_sha256'] != sha256(root/'plan.json') or sha256(p.parent/'model.pt') != row['weights_sha256']:
                raise ValueError('Fold changed')
            rows.append(row); hashes[str(p.relative_to(root))] = sha256(p)
    n = len(plan['alloy_ids']); pred = np.asarray([r['prediction'] for r in rows]).reshape(-1,n,2)
    y = np.asarray([r['target'] for r in rows[:n]])
    baseline = np.asarray([r['mean_baseline'] for r in rows[:n]])
    ensemble=metric(y,pred.mean(0));per_seed=[metric(y,p) for p in pred]
    improved=(np.abs(pred.mean(0)-y)<np.abs(baseline-y)).sum(0).tolist()
    screen=plan['settings']['prediction_screen']
    passed=[bool(ensemble['r2'][j] is not None and ensemble['r2'][j]>=screen['minimum_ensemble_r2']
        and improved[j]>=screen['minimum_improved_alloys']
        and (not screen['require_each_seed_positive_r2'] or all(m['r2'][j] is not None and m['r2'][j]>0 for m in per_seed))) for j in range(2)]
    dump(root/'summary.json', dict(schema='image_cnn_oof_summary_v1', target_names=TARGETS,
        alloy_ids=plan['alloy_ids'], seeds=plan['settings']['seeds'],
        ensemble_oof=pred.mean(0).tolist(), targets=y.tolist(),
        ensemble_metrics=ensemble, per_seed_metrics=per_seed, improved_alloys_vs_mean=improved,
        prediction_screen=screen,prediction_screen_passed=passed,
        mean_baseline_metrics=metric(y,baseline), fold_result_sha256=hashes,
        plan_sha256=sha256(root/'plan.json'), approved_targets=[False,False],
        training_authorized=False, cnn_refit_authorized=False,
        limits='Exploratory third-round grouped validation; requires scientific review, domain and gradient checks'))


def refit(root, plan, windows, review_path):
    if review_path is None: raise ValueError('Refit requires an explicit reviewed prediction decision')
    review = json.loads(Path(review_path).read_text()); summary = json.loads((root/'summary.json').read_text())
    active = review.get('active_targets')
    if (review.get('schema') != 'image_cnn_prediction_review_v1' or review.get('status') != 'passed'
            or review.get('summary_sha256') != sha256(root/'summary.json')
            or not isinstance(active,list) or len(active)!=2 or not all(type(v) is bool for v in active) or not any(active)
            or not review.get('rationale') or summary['plan_sha256'] != sha256(root/'plan.json')
            or any(a and not b for a,b in zip(active or [],summary['prediction_screen_passed']))
            or plan['settings']['software_fixture']):
        raise ValueError('Missing or mismatched scientific prediction review')
    for name, digest in summary['fold_result_sha256'].items():
        p = root/name; row = json.loads(p.read_text())
        if sha256(p) != digest or sha256(p.parent/'model.pt') != row['weights_sha256']:
            raise ValueError('CV evidence changed')
    output = root/'refit'
    if output.exists(): raise FileExistsError('Do not overwrite refit')
    output.mkdir(); members=[]
    for seed in plan['settings']['seeds']:
        counts=[json.loads((root/'cv'/f'alloy_{i:02d}_seed_{seed}'/'result.json').read_text())['selected_epoch'] for i in plan['alloy_ids']]
        epochs=max(1,int(np.median(counts)))
        model,mean,scale,trace=fit(windows,plan['alloy_ids'],seed,epochs)
        p=output/f'seed_{seed}.pt';torch.save(model.state_dict(),p)
        members.append(dict(seed=seed,weights_file=p.name,weights_sha256=sha256(p),target_mean=mean.tolist(),target_scale=scale.tolist(),epochs=epochs))
        dump(output/f'seed_{seed}_trace.json',trace)
    dump(output/'cnn_proxy.json',dict(schema=SCHEMA,target_names=TARGETS,members=members,
        patch_size=plan['settings']['patch_size'],pixel_transform='native_rgb_minus_one_one',
        target_scale=scale.tolist(),approved_targets=active,training_alloy_ids=plan['alloy_ids'],
        labels_sha256=sha256(plan['settings']['labels']),input_sha256=plan['hashes'],
        prediction_review_sha256=sha256(review_path),summary_sha256=sha256(root/'summary.json'),
        diffusion_training_authorized=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/mechanics_cnn.local.yaml')
    p.add_argument('--phase',required=True,choices=['prepare','cv','fold','collect','refit'])
    p.add_argument('--alloy-id',type=int);p.add_argument('--seed',type=int);p.add_argument('--review')
    args=p.parse_args()
    if args.phase=='prepare': prepare(args.config);print('Predictor plan locked; no training run');return
    root,plan=load(args.config)
    if args.phase=='collect': collect(root,plan);return
    windows=Windows(plan)
    if args.phase=='refit': refit(root,plan,windows,args.review);return
    alloys=plan['alloy_ids'] if args.phase=='cv' else [args.alloy_id]
    seeds=plan['settings']['seeds'] if args.phase=='cv' else [args.seed]
    if any(i not in plan['alloy_ids'] for i in alloys) or any(s not in plan['settings']['seeds'] for s in seeds):
        raise ValueError('Fold/seed outside registered pool')
    for seed in seeds:
        for alloy in alloys:
            print(f'CNN nested validation: alloy={alloy}, seed={seed}',flush=True)
            train_outer(root,plan,windows,alloy,seed)
    if args.phase=='cv': collect(root,plan)


if __name__=='__main__': main()
