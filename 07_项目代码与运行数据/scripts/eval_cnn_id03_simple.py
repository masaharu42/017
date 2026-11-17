"""Evaluate the 18-alloy exploratory CNN on fixed ID03 image crops; no training."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from physics.cnn_mechanics import SingleScaleCNN,sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--manifest',default='data/manifest.csv');p.add_argument('--labels',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 model_dir=Path(a.model_dir); plan=json.loads((model_dir/'plan.json').read_text()); summary=json.loads((model_dir/'summary.json').read_text())
 if 3 in plan['train_ids'] or 3 in plan['validation_ids']:raise ValueError('ID03 was included in training or selection')
 if plan['targets'] != ['peak_stress_MPa','endpoint_strain_pct']:raise ValueError('Wrong targets')
 if sha256(model_dir/'best.pt')!=summary['best_weights_sha256']:raise ValueError('Weights hash mismatch')
 with open(a.manifest,encoding='utf-8-sig',newline='') as f: rows=[r for r in csv.DictReader(f) if int(r['alloy_id'])==3]
 if len(rows)!=1:raise ValueError('Expected one ID03 image')
 image_path=Path(rows[0]['image_path']); image_path=image_path if image_path.is_absolute() else (Path(a.manifest).resolve().parent.parent/image_path).resolve()
 with Image.open(image_path) as im: arr=np.asarray(im.convert('RGB')).copy()
 h,w=arr.shape[:2]; size=256
 if min(h,w)<size:raise ValueError('Image too small')
 # Fixed full-field 8x4 grid, no selection using prediction; crops may overlap.
 xs=np.rint(np.linspace(0,w-size,8)).astype(int);ys=np.rint(np.linspace(0,h-size,4)).astype(int)
 positions=[(int(x),int(y)) for y in ys for x in xs]
 x=torch.from_numpy(np.stack([arr[y:y+size,x:x+size] for x,y in positions])).permute(0,3,1,2).float()/127.5-1
 ckpt=torch.load(model_dir/'best.pt',map_location='cpu',weights_only=True);m=SingleScaleCNN();m.load_state_dict(ckpt['model']);m.eval();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');m.to(device)
 with torch.no_grad(): pred=torch.cat([m(batch.to(device)).cpu() for batch in x.split(8)]).numpy()*np.array(ckpt['scale'])+np.array(ckpt['mean'])
 with open(a.labels,encoding='utf-8-sig',newline='') as f: labels=[r for r in csv.DictReader(f) if int(r['alloy_id'])==3]
 if len(labels)!=1:raise ValueError('Expected one ID03 label')
 truth=np.array([float(labels[0]['peak_stress_MPa']),float(labels[0]['endpoint_strain_pct'])]); avg=pred.mean(0)
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'scope':'exploratory ID03; no training; fixed 32 potentially overlapping windows','image':str(image_path),'image_sha256':sha256(image_path),'labels_sha256':sha256(a.labels),'weights_sha256':sha256(model_dir/'best.pt'),'model_plan_sha256':sha256(model_dir/'plan.json'),'target_names':plan['targets'],'target':truth.tolist(),'prediction_mean':avg.tolist(),'error':(avg-truth).tolist(),'window_std_population':pred.std(0).tolist(),'window_min':pred.min(0).tolist(),'window_max':pred.max(0).tolist(),'positions':positions,'window_predictions':pred.tolist()}
 (out/'id03_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
 print(json.dumps({k:v for k,v in result.items() if k not in ('positions','window_predictions')},ensure_ascii=False,indent=2));print('COMPLETE:',out.resolve())
if __name__=='__main__':main()
