"""Exploratory zero-update gradient connectivity probe; not an authorization to train."""
import json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, UNet2DModel, DDIMScheduler
from ebsd_diffusion.composition import CompositionConditioner,load_compositions
from ebsd_diffusion.models import ContinuousConditionEmbedding
from ebsd_diffusion.structure_statistics import x0_from_epsilon
from ebsd_diffusion.mechanics_runtime import gradient_probe,feedback_gradient_modules
from physics.cnn_mechanics import SingleScaleCNN,sha256

ROOT=Path('.'); SRC=ROOT/'08_训练与生成结果/14_Fold03_MSE_结构统计_5k_运行/A_mse/final'
VAE=ROOT/'01_基线留一验证_同一配置/00_基线模型/VAE模型/final'
CNN=ROOT/'output/mechanics_cnn_simple_paper_aug_v1'; OUT=ROOT/'output/mechanics_cnn_zero_update_probe_v1'
EXPECTED_U='872ed5cb88d0648a72118b1bf6da432732f6167610619ecb4c1140168b0fdd7a'
EXPECTED_C='0483a621f1214d2df8bea83e67019d194000bc5533b7d5cd8c413599a9327cad'
def digest(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):
  h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 start=time.monotonic()
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 if sha256(SRC/'unet_raw/diffusion_pytorch_model.safetensors')!=EXPECTED_U:raise ValueError('S0 UNet hash mismatch')
 if sha256(SRC/'condition_encoder/pytorch_model.bin')!=EXPECTED_C:raise ValueError('S0 condition encoder hash mismatch')
 summary=json.loads((CNN/'summary.json').read_text());plan=json.loads((CNN/'plan.json').read_text())
 if sha256(CNN/'best.pt')!=summary['best_weights_sha256']:raise ValueError('CNN hash mismatch')
 if 1 not in plan['train_ids'] or plan['targets']!=['peak_stress_MPa','endpoint_strain_pct']:raise ValueError('CNN plan mismatch')
 meta=json.loads((SRC/'run_metadata.json').read_text());assert meta['effective_total_step']==65000 and meta['exclude_alloy_id']==3
 device=torch.device('cuda');torch.manual_seed(20261104);torch.cuda.manual_seed_all(20261104)
 vae=AutoencoderKL.from_pretrained(VAE).to(device).eval().requires_grad_(False)
 cnn=SingleScaleCNN().to(device);ckpt=torch.load(CNN/'best.pt',map_location=device,weights_only=True);cnn.load_state_dict(ckpt['model']);cnn.eval().requires_grad_(False)
 unet=UNet2DModel.from_pretrained(SRC/'unet_raw').to(device).eval()
 encoder=ContinuousConditionEmbedding.from_pretrained(SRC/'condition_encoder').to(device).eval()
 scheduler=DDIMScheduler.from_pretrained(SRC/'scheduler')
 if scheduler.config.prediction_type!='epsilon':raise ValueError('Expected epsilon scheduler')
 before={name:digest(m) for name,m in [('vae',vae),('cnn',cnn),('unet',unet),('condition',encoder)]}
 original=json.loads((ROOT/'output/mechanics_cnn_v1/plan.json').read_text()); rec=original['records']['1'];path=Path(rec['image'])
 if sha256(path)!=original['hashes'][str(path)]:raise ValueError('Source image hash mismatch')
 with Image.open(path) as im:arr=np.array(im.convert('RGB'))
 pos=rec['positions'][:4];x=torch.from_numpy(np.stack([arr[y:y+256,x:x+256] for x,y in pos])).permute(0,3,1,2).float().to(device)/127.5-1
 conditioner=CompositionConditioner.load(SRC/'composition_conditioner.json'); frame=load_compositions(ROOT/'data/compositions.csv')
 condition=torch.as_tensor(conditioner.transform_by_alloy_id(frame,1),dtype=torch.float32,device=device).unsqueeze(0).repeat(4,1)
 with torch.no_grad(): z=vae.encode(x).latent_dist.mode()*float(meta['latent_scaling'])
 target=torch.tensor(rec['target'],device=device,dtype=torch.float32); mean=torch.tensor(ckpt['mean'],device=device);scale=torch.tensor(ckpt['scale'],device=device)
 results=[]
 for t in [50,150]:
  noise=torch.randn_like(z);ts=torch.full((4,),t,device=device,dtype=torch.long);noisy=scheduler.add_noise(z,noise,ts)
  pred=unet(noisy,ts,class_labels=encoder(condition)).sample; x0=x0_from_epsilon(noisy,pred,ts,scheduler.alphas_cumprod)
  decoded=vae.decode(x0/float(meta['latent_scaling'])).sample
  values=(cnn(decoded)*scale+mean).mean(0); errors=(values-target)/scale
  terms=[torch.nn.functional.huber_loss(errors[i],torch.zeros_like(errors[i]),delta=1.) for i in range(2)]
  mse=torch.nn.functional.mse_loss(pred.float(),noise.float()); mechanics=(terms[0]+terms[1])/2
  recg=gradient_probe(mse,terms,mechanics,feedback_gradient_modules(unet,encoder))
  recg.update(timestep=t,predicted=values.detach().cpu().tolist(),target=target.cpu().tolist(),mse=float(mse.detach()),mechanics_loss=float(mechanics.detach()))
  results.append(recg);print('t=',t,'noise_grad=',recg['mse_norm'],'mechanics_grad=',recg['mechanics_norm'],flush=True)
  del pred,x0,decoded,values,errors,terms,mse,mechanics
 after={name:digest(m) for name,m in [('vae',vae),('cnn',cnn),('unet',unet),('condition',encoder)]}
 if before!=after:raise RuntimeError('Model state changed')
 OUT.mkdir(parents=True,exist_ok=False)
 report={'purpose':'exploratory gradient connectivity only, no optimization, not scientific feedback approval','alloy_id':1,'timesteps':[50,150],'windows':pos,'seed':20261104,'source_unet_sha256':EXPECTED_U,'source_condition_sha256':EXPECTED_C,'cnn_sha256':summary['best_weights_sha256'],'vae_directory':str(VAE.resolve()),'model_state_before':before,'model_state_after':after,'results':results,'seconds':time.monotonic()-start}
 (OUT/'probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False));print('COMPLETE',OUT.resolve(),flush=True)
if __name__=='__main__':main()
