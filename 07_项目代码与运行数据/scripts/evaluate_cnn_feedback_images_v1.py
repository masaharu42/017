"""Exploratory image and CNN comparison; no model updates."""
import csv, json, random, statistics, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from ebsd_diffusion.metrics import describe_image
from physics.cnn_mechanics import CNNMechanics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/mechanics_cnn_image_evaluation_v1'
NAMES=('S0','C0','C1_0.01','C1_0.03')
IDS=(3,5,7,14,15,19,20)
SEEDS=(20261012,20261013)
FIELDS=('gb_fraction','grain_count','grain_area_median_px','edge_density')
LABELS=('peak_stress_MPa','endpoint_strain_pct')

def main():
    if OUT.exists(): raise FileExistsError(f'Output exists: {OUT}')
    manifest=pd.read_csv(ROOT/'data/manifest.csv').set_index('alloy_id')
    labels=pd.read_csv(ROOT/'09_力学试验数据_只读整理_20261029/tables_from_windows/alloy_median_9_mechanics_targets.csv').set_index('alloy_id')
    proxy=ROOT/'output/mechanics_cnn_feedback_review/cnn_proxy.json'
    model=CNNMechanics.load(proxy).to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    device=next(model.parameters()).device
    inputs={}; rows=[]; summary=[]
    for alloy in IDS:
        root=ROOT/('output/mechanics_cnn_weight_comparison_ID03_v1' if alloy==3 else 'output/mechanics_cnn_weight_comparison_v1')/f'ID{alloy:02d}'
        src=ROOT/str(manifest.loc[alloy,'image_path']); inputs[str(alloy)]=str(src)
        with Image.open(src) as im: real=im.convert('RGB')
        if real.width<256 or real.height<256: raise ValueError(f'Real image too small: {src}')
        rng=random.Random(20261105+alloy)
        positions=[(rng.randrange(real.width-256+1),rng.randrange(real.height-256+1)) for _ in range(16)]
        print('alloy',alloy,'real windows',len(positions),flush=True)
        with tempfile.TemporaryDirectory() as td:
            def process(img, kind, seed, pos=None):
                p=Path(td)/'window.png'; img.save(p)
                d=describe_image(p,gb_threshold=.18)
                a=np.asarray(img,dtype=np.float32)/127.5-1
                x=torch.from_numpy(a.copy()).permute(2,0,1).unsqueeze(0).to(device)
                with torch.no_grad(): pred=model(x)[0].cpu().tolist()
                row={'alloy_id':alloy,'held_out':alloy==3,'group':kind,'seed':seed,
                     'crop_x':pos[0] if pos else '', 'crop_y':pos[1] if pos else '',
                     **{f:getattr(d,f) for f in FIELDS},
                     'pred_peak_stress_MPa':pred[0], 'pred_endpoint_strain_pct':pred[1]}
                for lab in LABELS:
                    val=labels.loc[alloy,lab] if alloy in labels.index else float('nan')
                    row['target_'+lab]=float(val) if pd.notna(val) else ''
                rows.append(row)
            for x,y in positions: process(real.crop((x,y,x+256,y+256)),'real_window','',(x,y))
            for name in NAMES:
                for seed in SEEDS:
                    path=root/f'{name}_seed_{seed}.png'
                    with Image.open(path) as im:
                        img=im.convert('RGB')
                        if img.size!=(256,256): raise ValueError(f'Unexpected image size: {path}')
                        process(img,name,seed)
        actual=[r for r in rows if r['alloy_id']==alloy and r['group']=='real_window']
        for name in NAMES:
            subset=[r for r in rows if r['alloy_id']==alloy and r['group']==name]
            out={'alloy_id':alloy,'held_out':alloy==3,'group':name,'generated_count':len(subset),'real_window_count':len(actual)}
            for key in (*FIELDS,'pred_peak_stress_MPa','pred_endpoint_strain_pct'):
                out['generated_mean_'+key]=statistics.mean(r[key] for r in subset)
                out['real_window_mean_'+key]=statistics.mean(r[key] for r in actual)
                out['difference_from_real_window_mean_'+key]=out['generated_mean_'+key]-out['real_window_mean_'+key]
            for lab in LABELS:
                out['target_'+lab]=subset[0]['target_'+lab]
                if out['target_'+lab]!='': out['prediction_minus_target_'+lab]=out['generated_mean_pred_'+lab]-out['target_'+lab]
            summary.append(out)
    OUT.mkdir(parents=True,exist_ok=False)
    def write(name,values):
        keys=list(dict.fromkeys(k for row in values for k in row))
        with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(values)
    write('per_image.csv',rows);write('per_alloy_group.csv',summary)
    (OUT/'notes.json').write_text(json.dumps({'ids':IDS,'seeds':SEEDS,'real_crops_per_alloy':16,'crop_seed':'20261105 + alloy_id',
        'threshold':.18,'cnn_proxy':str(proxy),'real_images':inputs,'warnings':[
        'Two generated samples per group; real windows overlap and are not independent alloys.',
        'CNN is the same model used for feedback, not independent validation; ID03 predictor generalization is unproven.',
        'CNN sees saved clipped RGB images; training feedback receives unclipped decoded images.',
        'Threshold-derived connected regions are approximate image descriptors, not verified physical grains.',
        'Endpoint strain is a DIC endpoint proxy; target validity and biological replicates require review.'
        ]},ensure_ascii=False,indent=2))
    print('Finished:',OUT)
if __name__=='__main__': main()
