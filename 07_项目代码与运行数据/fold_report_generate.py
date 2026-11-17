#!/usr/bin/env python3
"""Generate and compare completed strict leave-one-alloy-out folds."""
import argparse, csv, hashlib, json, subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw

P = argparse.ArgumentParser()
P.add_argument('--output', required=True, type=Path)
P.add_argument('--folds', nargs='+', type=int, default=[3, 14, 18, 19, 21])
P.add_argument('--seeds', nargs='+', type=int, default=[20261012, 20261013, 20261014, 20261015, 20261016])
P.add_argument('--smoke', action='store_true', help='One held-out and one training ID, first seed')
P.add_argument('--allow-heldout-extrapolation', action='store_true')
args = P.parse_args()
root = Path.cwd()
roots = [root/'08_训练与生成结果/05_留一验证', root/'08_训练与生成结果/06_其余18折正式留一验证_4080S']
if len(set(args.folds)) != len(args.folds) or len(set(args.seeds)) != len(args.seeds): raise SystemExit('Duplicate fold or seed')
if not (root/'sample.py').is_file(): raise SystemExit('Run from the project root')
with (root/'data/manifest.csv').open(encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
manifest = {}
for row in rows:
    if row.get('use_for_diffusion', '').lower() not in ('true','1','yes'): continue
    i = int(row['alloy_id'])
    if i in manifest: raise SystemExit(f'ID{i}: multiple diffusion source images, inspect manifest')
    path = Path(row['image_path'].replace('\\','/'))
    candidates = [path, root/path, root/'data'/path]
    path = next((p.resolve() for p in candidates if p.is_file()), None)
    if path is None: raise SystemExit(f'ID{i}: source image missing: {row["image_path"]}')
    with Image.open(path) as im:
        if min(im.size) < 256: raise SystemExit(f'ID{i}: source smaller than 256: {path}')
    manifest[i] = path

def valid_weight(p):
    if not p.is_file(): raise SystemExit(f'Missing weight: {p}')
    with p.open('rb') as f:
        if f.read(43).startswith(b'version https://git-lfs.github.com/spec'): raise SystemExit(f'Git LFS pointer: {p}')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

plans = []
for fold in args.folds:
    matches = [r/f'fold_{fold:02d}' for r in roots if (r/f'fold_{fold:02d}'/'fold_summary.json').is_file()]
    if len(matches) != 1: raise SystemExit(f'Fold {fold}: expected one completed fold, found: {matches}')
    fd = matches[0]; summary = json.loads((fd/'fold_summary.json').read_text())
    if summary.get('status') != 'completed' or int(summary.get('held_out_alloy_id',-1)) != fold: raise SystemExit(f'Fold {fold}: incomplete or inconsistent summary')
    ck = fd/'models_strict/diffusion/final'; vae = fd/'models_strict/vae/final'
    meta = json.loads((ck/'run_metadata.json').read_text())
    vmeta = json.loads((vae/'training_metadata.json').read_text())
    if int(meta.get('exclude_alloy_id',-1)) != fold or int(vmeta.get('exclude_alloy_id',-1)) != fold: raise SystemExit(f'Fold {fold}: weight metadata mismatch')
    ids = sorted(int(x) for x in meta['training_alloy_ids'])
    if fold in ids or set(ids) != set(map(int, vmeta['training_alloy_ids'])): raise SystemExit(f'Fold {fold}: invalid train IDs')
    if not set(ids+[fold]).issubset(manifest): raise SystemExit(f'Fold {fold}: missing manifest images')
    weights = [ck/'unet/diffusion_pytorch_model.safetensors',ck/'condition_encoder/pytorch_model.bin',vae/'diffusion_pytorch_model.safetensors']
    for p in weights: valid_weight(p)
    plans.append((fold, fd, ck, vae, ids, meta, weights))

args.output.mkdir(parents=True,exist_ok=True)
provenance = {'folds':args.folds,'seeds':args.seeds[:1] if args.smoke else args.seeds,'size':[256,256],'ddim_steps':50,'cfg':1.8,'smoke':args.smoke,'allow_heldout_extrapolation':args.allow_heldout_extrapolation,'runs':[]}
for fold, fd, ck, vae, ids, meta, weights in plans:
    selected = [fold,ids[0]] if args.smoke else [fold]+ids
    seeds = args.seeds[:1] if args.smoke else args.seeds
    provenance['runs'].append({'fold':fold,'checkpoint':str(ck),'vae':str(vae),'training_ids':ids,'global_step':meta.get('global_step'),'weights_sha256':{str(p):sha(p) for p in weights}})
    (args.output/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    for aid in selected:
        group = 'held_out' if aid == fold else 'train'
        out = args.output/f'fold_{fold:02d}'/group/f'ID{aid:02d}'
        out.mkdir(parents=True,exist_ok=True)
        generated = []
        for seed in seeds:
            sd = out/f'seed_{seed}'
            img = sd/f'alloy_{aid:02d}_seed{seed}_000.png'
            ext = sd/f'alloy_{aid:02d}_seed{seed}_000_EXTRAPOLATION.png'
            report = sd/f'alloy_{aid:02d}_generation_report.json'
            if report.exists() and (img.exists() or ext.exists()):
                d=json.loads(report.read_text())
                if d.get('seed') != seed or d.get('generated_width') != 256 or d.get('generated_height') != 256 or str(Path(d.get('checkpoint','')).resolve()) != str(ck.resolve()): raise SystemExit(f'Existing report mismatch: {report}')
            else:
                cmd=[sys.executable,'-u','sample.py','--checkpoint',str(ck),'--vae-dir',str(vae),'--alloy-id',str(aid),'--count','1','--batch-size','1','--height','256','--width','256','--crop-height','256','--crop-width','256','--steps','50','--guidance-scale','1.8','--seed',str(seed),'--output-dir',str(sd)]
                if group=='held_out' and args.allow_heldout_extrapolation: cmd.append('--allow-extrapolation')
                print(f'Fold {fold:02d} / {group} / ID {aid:02d} / seed {seed}',flush=True)
                subprocess.run(cmd,check=True)
            chosen = ext if ext.exists() else img
            if not chosen.is_file(): raise SystemExit(f'Missing generated image: {chosen}')
            generated.append((seed,chosen))
        with Image.open(manifest[aid]) as original:
            original=original.convert('RGB'); w,h=original.size
            positions=[(0,0),(w-256,0),((w-256)//2,(h-256)//2),(0,h-256),(w-256,h-256)]
            windows=[original.crop((x,y,x+256,y+256)) for x,y in positions]
        n=max(5,len(generated)); grid=Image.new('RGB',(n*256,2*288),'white'); draw=ImageDraw.Draw(grid)
        for j,im in enumerate(windows): grid.paste(im,(j*256,32))
        for j,(seed,path) in enumerate(generated):
            with Image.open(path) as im: grid.paste(im.convert('RGB'),(j*256,320))
            draw.text((j*256+3,294),f'seed {seed}'+(' EXTRAPOLATION' if 'EXTRAPOLATION' in path.name else ''),fill='black')
        draw.text((3,5),'Real 256 crops: TL / TR / Center / BL / BR (not paired)',fill='black')
        grid.save(out/'comparison.png')
        print('Saved:',out/'comparison.png',flush=True)
print('Finished:',args.output.resolve())
