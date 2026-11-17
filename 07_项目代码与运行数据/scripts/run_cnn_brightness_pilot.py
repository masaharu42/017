"""Exploratory paired CNN brightness pilot. Never modifies the registered run."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts import train_mechanics_cnn as base
from physics.cnn_mechanics import SingleScaleCNN, sha256

SEED = 20261103
# Fixed exploratory subset: deliberately includes previously observed difficult cases.
ALLOYS = [1, 7, 14, 21]
BRIGHTNESS = 0.03


def fit(windows, train_ids, seed, epochs, validation_ids, progress, label):
    s = windows.plan['settings']
    device = torch.device(s['device'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    # Separate RNG ensures original crop/alloy sequence is unchanged.
    aug_rng = np.random.default_rng(seed + 9000001)
    targets = windows.labels(train_ids)
    mean = targets.mean(0); scale = targets.std(0)
    scale = np.where(scale < 1e-8, 1., scale)
    model = SingleScaleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=s['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=s['max_epochs'])
    trace = []
    for epoch in range(epochs):
        model.train(); total = 0.
        for _ in range(0, s['groups_per_epoch'], s['batch_groups']):
            ids = rng.choice(train_ids, size=s['batch_groups'], replace=False).tolist()
            x = torch.cat([windows.random(i, s['group_size'], rng) for i in ids]).to(device)
            # Shared multiplicative brightness per crop; same factor for RGB channels.
            factors = torch.as_tensor(aug_rng.uniform(1-BRIGHTNESS, 1+BRIGHTNESS, len(x)),
                                      device=device, dtype=x.dtype).view(-1, 1, 1, 1)
            x = (((x + 1) / 2) * factors).clamp(0, 1) * 2 - 1
            target = torch.tensor((windows.labels(ids)-mean)/scale, device=device)
            output = model(x).reshape(len(ids), s['group_size'], 2).mean(1)
            loss = F.mse_loss(output, target)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.detach())
        scheduler.step()
        item = dict(epoch=epoch+1, training_loss=total/(s['groups_per_epoch']/s['batch_groups']))
        if validation_ids:
            prediction = base.predict(model, windows, validation_ids, mean, scale, device)
            item['validation_loss'] = float(np.mean(((prediction-windows.labels(validation_ids))/scale)**2))
        trace.append(item); progress['done'] += 1
        if epoch == 0 or (epoch+1) % 10 == 0 or epoch+1 == epochs:
            elapsed = time.monotonic()-progress['start']
            eta = elapsed / progress['done'] * max(0, progress['estimate']-progress['done'])
            val = f"{item['validation_loss']:.5f}" if validation_ids else 'n/a'
            print(f"{label} epoch={epoch+1}/{epochs} train_loss={item['training_loss']:.5f} "
                  f"val_loss={val} elapsed_min={elapsed/60:.1f} estimated_remaining_min={eta/60:.1f}", flush=True)
    return model.cpu().eval(), mean, scale, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', default='output/mechanics_cnn_brightness_pilot_v1')
    args = parser.parse_args()
    root, plan = base.load(args.config)
    s = plan['settings']
    if SEED not in s['seeds'] or not set(ALLOYS) <= set(plan['alloy_ids']):
        raise ValueError('Original plan does not match pilot')
    if s['inner_folds'] != 4 or s['max_epochs'] != 200:
        raise ValueError('Pilot requires original 4-fold, 200-epoch protocol')
    originals = {}
    for alloy in ALLOYS:
        folder = root/'cv'/f'alloy_{alloy:02d}_seed_{SEED}'
        result = json.loads((folder/'result.json').read_text())
        if result['plan_sha256'] != sha256(root/'plan.json') or result['weights_sha256'] != sha256(folder/'model.pt'):
            raise ValueError(f'Baseline hash mismatch: {alloy}')
        originals[alloy] = result
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    base.dump(out/'pilot_plan.json', dict(original_plan_sha256=sha256(root/'plan.json'),
        pilot_code_sha256=sha256(Path(__file__)), seed=SEED, alloys=ALLOYS,
        augmentation='independent per-crop RGB brightness factor uniform [0.97, 1.03], clip [0,1]',
        reason='exploratory selected folds; includes previously observed large errors; not representative',
        evaluation='original fixed 32 windows; no augmentation', no_diffusion=True))
    windows = base.Windows(plan)
    # Refit length unknown in advance: use baseline selected epochs as a rough time estimate.
    progress = dict(start=time.monotonic(), done=0,
                    estimate=sum(4*s['max_epochs']+originals[i]['selected_epoch'] for i in ALLOYS))
    comparisons = []
    for alloy in ALLOYS:
        folder = out/f'alloy_{alloy:02d}'; folder.mkdir()
        train_ids = [i for i in plan['alloy_ids'] if i != alloy]
        partitions = np.array_split(np.random.default_rng(SEED).permutation(train_ids), s['inner_folds'])
        inner = []; selected = []
        for j, part in enumerate(partitions):
            val = part.tolist(); train = [i for i in train_ids if i not in val]
            _, _, _, trace = fit(windows, train, SEED+j+1, s['max_epochs'], val, progress, f'alloy={alloy} inner={j+1}/4')
            best = min(trace, key=lambda t:t['validation_loss'])['epoch']
            selected.append(best); inner.append(dict(train_ids=train, validation_ids=val, selected_epoch=best, trace=trace))
        epoch = max(1, int(np.median(selected)))
        progress['estimate'] += epoch-originals[alloy]['selected_epoch']
        model, mean, scale, trace = fit(windows, train_ids, SEED, epoch, None, progress, f'alloy={alloy} refit')
        pred = base.predict(model, windows, [alloy], mean, scale, torch.device('cpu'))[0].tolist()
        torch.save(model.state_dict(), folder/'model.pt')
        baseline = originals[alloy]
        result = dict(alloy_id=alloy, seed=SEED, target=baseline['target'], baseline_prediction=baseline['prediction'],
                      augmented_prediction=pred, selected_epoch=epoch, inner=inner, refit_trace=trace,
                      target_mean=mean.tolist(), target_scale=scale.tolist(), weights_sha256=sha256(folder/'model.pt'))
        base.dump(folder/'result.json', result); comparisons.append(result)
        print(f"DONE alloy={alloy} target={result['target']} A={baseline['prediction']} B={pred}",flush=True)
    base.dump(out/'comparison.json', comparisons)
    print(f'COMPLETE results={out.resolve()} elapsed_min={(time.monotonic()-progress["start"])/60:.1f}', flush=True)

if __name__ == '__main__': main()
