"""Sample S0, C0, and two exploratory CNN-weight arms using paired seeds."""
import csv
import json
from pathlib import Path
import torch
import yaml
from PIL import Image, ImageDraw
from diffusers import AutoencoderKL, UNet2DModel, DDIMScheduler
from ebsd_diffusion.composition import CompositionConditioner, load_compositions
from ebsd_diffusion.models import ContinuousConditionEmbedding
from scripts.sample_fold03_ab2c_256 import sample_image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/mechanics_cnn_weight_comparison_2000_v1'
LOW = ROOT / 'output/mechanics_cnn_2000_low_v1'
HIGH = ROOT / 'output/mechanics_cnn_2000_high_v1'

def fingerprints(path):
    with path.open() as f:
        return {row['additional_step']: row['sha256'] for row in csv.DictReader(f)}

def main():
    if OUT.exists():
        raise FileExistsError(f'Output already exists: {OUT}')
    if not torch.cuda.is_available():
        raise RuntimeError('This comparison requires CUDA')
    cfg = yaml.safe_load((ROOT/'configs/mechanics_cnn_2000_low_v1.yaml').read_text())
    source = yaml.safe_load((ROOT/cfg['source_sampling_config']).read_text())
    src = (ROOT/source['raw_unet']).parent.parent
    dirs = {'S0': src, 'C0': LOW/'C0/final', 'C1_0.01': LOW/'C1/final', 'C1_0.03': HIGH/'C1/final'}
    refs = fingerprints(LOW/'C0/input_fingerprints.csv')
    for label, base in [('C1_0.01', LOW/'C1'), ('C1_0.03', HIGH/'C1')]:
        if fingerprints(base/'input_fingerprints.csv') != refs:
            raise ValueError(f'Input fingerprints do not match C0 for {label}')
    for label, path in dirs.items():
        meta = json.loads((path/'run_metadata.json').read_text())
        if label != 'S0':
            if meta.get('global_step') != 2000:
                raise ValueError(f'{label} is not a completed 2000-step run')
            expected = {'C0': 0., 'C1_0.01': .01, 'C1_0.03': .03}[label]
            if abs(float(meta['mechanics']['lambda']) - expected) > 1e-9:
                raise ValueError(f'{label} has the wrong weight')
    device = torch.device('cuda'); dtype = torch.float16
    vae = AutoencoderKL.from_pretrained(ROOT/source['vae_dir']).to(device=device, dtype=dtype).eval()
    frame = load_compositions(ROOT/'data/compositions.csv')
    ids, seeds = cfg['evaluation']['alloy_ids'] + cfg['evaluation']['heldout_alloy_ids'], cfg['evaluation']['seeds']
    OUT.mkdir(parents=True, exist_ok=False)
    files = {}
    with torch.no_grad():
        for label, path in dirs.items():
            meta = json.loads((path/'run_metadata.json').read_text())
            conditioner = CompositionConditioner.load(path/'composition_conditioner.json')
            unet = UNet2DModel.from_pretrained(path/'unet_raw').to(device=device, dtype=dtype).eval()
            encoder = ContinuousConditionEmbedding.from_pretrained(path/'condition_encoder').to(device=device, dtype=dtype).eval()
            scheduler = DDIMScheduler.from_pretrained(path/'scheduler')
            for alloy in ids:
                condition = torch.as_tensor(conditioner.transform_by_alloy_id(frame, alloy), device=device, dtype=dtype).unsqueeze(0)
                for seed in seeds:
                    img = sample_image(unet, vae, encoder, scheduler, condition, float(meta['latent_scaling']),
                                       seed, 50, 1.8, 256, 256, 256, 256)
                    target = OUT / f'ID{alloy:02d}' / f'{label}_seed_{seed}.png'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    img.save(target)
                    files[(label, alloy, seed)] = target
            del unet, encoder, scheduler
        for alloy in ids:
            for seed in seeds:
                canvas = Image.new('RGB', (256*4, 280), 'white'); draw = ImageDraw.Draw(canvas)
                for i, label in enumerate(dirs):
                    with Image.open(files[(label, alloy, seed)]) as img: canvas.paste(img.convert('RGB'), (i*256, 24))
                    draw.text((i*256+5, 5), label, fill='black')
                canvas.save(OUT/f'ID{alloy:02d}'/f'comparison_seed_{seed}.png')
    (OUT/'manifest.json').write_text(json.dumps({'models': {k:str(v) for k,v in dirs.items()}, 'alloy_ids':ids,
        'seeds':seeds, 'steps':50, 'guidance_scale':1.8, 'matched_fingerprint_steps':list(refs),
        'scope':'exploratory; matching fingerprints cover recorded steps only'}, indent=2))
    print('Saved paired images:', OUT)

if __name__ == '__main__': main()
