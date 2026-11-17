"""One split, two-target exploratory CNN; no generator training or ID03 access."""
import argparse
import math
import copy
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scripts import train_mechanics_cnn as base
from physics.cnn_mechanics import SingleScaleCNN, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--labels', required=True, help='One row per alloy, with alloy_id')
    p.add_argument('--strength-column', required=True)
    p.add_argument('--strain-column', required=True)
    p.add_argument('--validation-ids', required=True, help='Exactly 3 comma-separated development alloy IDs')
    p.add_argument('--epochs', required=True, type=int)
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    ids = [int(x.strip()) for x in a.validation_ids.split(',')]
    if len(ids) != 3 or len(set(ids)) != 3 or 3 in ids: raise ValueError('Provide three unique validation IDs, excluding ID03')
    if a.epochs < 1 or a.epochs > 200: raise ValueError('epochs must be 1..200')
    if a.strength_column == a.strain_column: raise ValueError('Targets must be distinct')
    root, original = base.load(a.config)
    if len(original['alloy_ids']) != 21 or 3 in original['alloy_ids'] or not set(ids) <= set(original['alloy_ids']):
        raise ValueError('Expected 21 development alloys; validation IDs must be among them')
    label_path = Path(a.labels).resolve()
    with label_path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'alloy_id', a.strength_column, a.strain_column} <= set(reader.fieldnames or []):
            raise ValueError('Missing target columns')
        labels = {}
        for row in reader:
            i = int(row['alloy_id'])
            if i not in original['alloy_ids']: continue
            if i in labels: raise ValueError(f'Duplicate alloy {i}')
            y = np.array([float(row[a.strength_column]), float(row[a.strain_column])])
            if not np.isfinite(y).all(): raise ValueError(f'Invalid labels: {i}')
            labels[i] = y
    if set(labels) != set(original['alloy_ids']): raise ValueError('Missing development alloy labels')
    plan = copy.deepcopy(original)
    for i in original['alloy_ids']: plan['records'][str(i)]['target'] = labels[i].tolist()
    train = [i for i in plan['alloy_ids'] if i not in ids]
    if len(train) != 18: raise ValueError('Expected 18 training alloys')
    s = plan['settings']; device = torch.device(s['device'])
    if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    windows = base.Windows(plan)
    y = windows.labels(train); mean = y.mean(0); scale = y.std(0)
    if not np.isfinite(scale).all() or (scale < 1e-8).any(): raise ValueError('Target scale invalid')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    manifest = dict(type='exploratory_single_split', source_plan_sha256=sha256(root/'plan.json'),
        script_sha256=sha256(Path(__file__)), labels=str(label_path), labels_sha256=sha256(label_path),
        targets=[a.strength_column,a.strain_column], train_ids=train, validation_ids=ids, excluded_id=3,
        seed=a.seed, epochs=a.epochs, augmentation='horizontal/vertical flips p=0.5 each; rotation +/-10 degrees bilinear reflection; brightness contrast saturation +/-3 percent; train only; color range assumed', group_size=s['group_size'], batch_groups=s['batch_groups'],
        groups_per_epoch=s['groups_per_epoch'], target_mean=mean.tolist(), target_scale=scale.tolist(),
        gradient_scope='shared CNN encoder, first training batch of each epoch', gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu')
    base.dump(out/'plan.json', manifest)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(a.seed)
    torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(a.seed); aug_rng=np.random.default_rng(a.seed+9000001); model=SingleScaleCNN().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=s['learning_rate'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    start=time.monotonic(); best=float('inf'); best_epoch=0
    trace_path=out/'epochs.jsonl'
    for epoch in range(a.epochs):
        model.train(); totals=np.zeros(2); grad_info=None
        for batch_idx in range(s['groups_per_epoch']//s['batch_groups']):
            batch_ids=rng.choice(train,size=s['batch_groups'],replace=False).tolist()
            x=torch.cat([windows.random(i,s['group_size'],rng) for i in batch_ids]).to(device)
            # Independent flip per crop, then rotation; reflection padding avoids black corners.
            n = len(x)
            horizontal = torch.as_tensor(aug_rng.random(n) < .5, device=device).view(-1, 1, 1, 1)
            vertical = torch.as_tensor(aug_rng.random(n) < .5, device=device).view(-1, 1, 1, 1)
            x = torch.where(horizontal, x.flip(-1), x)
            x = torch.where(vertical, x.flip(-2), x)
            angles = torch.as_tensor(aug_rng.uniform(-10, 10, n) * math.pi / 180, device=device, dtype=x.dtype)
            theta = torch.zeros((n, 2, 3), device=device, dtype=x.dtype)
            theta[:, 0, 0] = angles.cos(); theta[:, 0, 1] = -angles.sin()
            theta[:, 1, 0] = angles.sin(); theta[:, 1, 1] = angles.cos()
            grid = F.affine_grid(theta, x.shape, align_corners=False)
            x = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
            x = (x + 1) / 2
            factors = aug_rng.uniform(1-0.03, 1+0.03, (3, n))
            factors = torch.as_tensor(factors, device=device, dtype=x.dtype).view(3, n, 1, 1, 1)
            x = x * factors[0]
            center = x.mean(dim=(1, 2, 3), keepdim=True)
            x = (x-center)*factors[1]+center
            gray = (x[:, 0:1]*.299 + x[:, 1:2]*.587 + x[:, 2:3]*.114)
            x = ((x-gray)*factors[2]+gray).clamp(0, 1)*2-1
            target=torch.tensor((windows.labels(batch_ids)-mean)/scale,device=device,dtype=torch.float32)
            pred=model(x).reshape(len(batch_ids),s['group_size'],2).mean(1)
            terms=[F.mse_loss(pred[:,j],target[:,j]) for j in range(2)]
            if batch_idx==0:
                params=[v for v in model.encoder.parameters() if v.requires_grad]
                grads=[torch.autograd.grad(t,params,retain_graph=True,allow_unused=True) for t in terms]
                dot=sum((u.float()*v.float()).sum() for u,v in zip(*grads) if u is not None and v is not None)
                norms=[torch.sqrt(sum(g.float().square().sum() for g in gs if g is not None)) for gs in grads]
                n=[float(x.detach()) for x in norms]; denom=n[0]*n[1]
                grad_info=dict(strength_norm=n[0],strain_norm=n[1],cosine=float(dot.detach())/denom if denom>1e-20 else None)
            loss=(terms[0]+terms[1])/2
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            totals += [float(t.detach()) for t in terms]
        scheduler.step()
        vp=base.predict(model,windows,ids,mean,scale,device)
        truth=windows.labels(ids); val=((vp-truth)/scale)**2
        val_terms=val.mean(0); val_loss=float(val_terms.mean())
        if val_loss < best:
            best=val_loss;best_epoch=epoch+1
            torch.save({'model':model.state_dict(),'mean':mean.tolist(),'scale':scale.tolist(),'epoch':best_epoch},out/'best.pt')
            base.dump(out/'best_predictions.json',dict(ids=ids,truth=truth.tolist(),prediction=vp.tolist(),epoch=best_epoch))
        elapsed=time.monotonic()-start; eta=elapsed/(epoch+1)*(a.epochs-epoch-1)
        item=dict(epoch=epoch+1,train_strength=float(totals[0]/(s['groups_per_epoch']//s['batch_groups'])),
            train_strain=float(totals[1]/(s['groups_per_epoch']//s['batch_groups'])),
            val_strength=float(val_terms[0]),val_strain=float(val_terms[1]),val_loss=val_loss,
            gradient=grad_info,elapsed_seconds=elapsed,remaining_estimate_seconds=eta,best_epoch=best_epoch)
        with trace_path.open('a',encoding='utf-8') as f: f.write(json.dumps(item,allow_nan=False)+'\n')
        print(f"epoch={epoch+1}/{a.epochs} train_strength={item['train_strength']:.4f} train_strain={item['train_strain']:.4f} val_strength={item['val_strength']:.4f} val_strain={item['val_strain']:.4f} cosine={grad_info['cosine']} best_epoch={best_epoch} elapsed_min={elapsed/60:.1f} remaining_min={eta/60:.1f}",flush=True)
    base.dump(out/'summary.json',dict(best_epoch=best_epoch,best_validation_loss=best,best_weights_sha256=sha256(out/'best.pt'),
              limitation='Single split; validation used for model selection; ID03 not evaluated; exploratory only'))
    print(f'COMPLETE output={out.resolve()}',flush=True)

if __name__=='__main__': main()
