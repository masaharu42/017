"""Fixed real-window perturbation/reconstruction/timestep diagnostics, zero updates.

This script deliberately leaves the trajectory-coverage/physical interpretation
gate pending. It cannot determine where a layout defect formed from one-step
denoising alone. Use the existing trajectory diagnostic and review both results.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F
from diffusers import AutoencoderKL

from ebsd_diffusion.composition import load_compositions
from ebsd_diffusion.config import load_config, resolve_project_path
from ebsd_diffusion.data import resolve_manifest_image_path
from ebsd_diffusion.structure_statistics import x0_from_epsilon
from ebsd_diffusion.training_control import (file_sha256, resolve_initialization,
                                            load_frozen_conditioner, load_initialized_models)
from physics.linear_mechanics import LinearMechanics
from scripts.validate_mechanics_m0 import nonoverwriting_directory, write_json


def perturbations(image):
    rgb = (image+1)/2
    boundary = (rgb.amax(1, keepdim=True) < .2).float()
    expanded = F.max_pool2d(boundary, 3, 1, 1)
    eroded = 1-F.max_pool2d(1-boundary, 3, 1, 1)
    local = F.avg_pool2d(rgb*(1-boundary), 3, 1, 1)
    count = F.avg_pool2d(1-boundary, 3, 1, 1)
    # Colour replacement is an approximation; report it, do not label it physical truth.
    restored = torch.where(count > 0, local/count.clamp_min(1e-8), rgb)
    narrow = torch.where(eroded.bool(), torch.zeros_like(rgb),
                         torch.where(boundary.bool(), restored, rgb))
    return {'brightness_plus_0p05': ((rgb+.05).clamp(0, 1))*2-1,
            'blur_3px': F.avg_pool2d(F.pad(image, (1, 1, 1, 1), mode='reflect'), 3, 1),
            'boundary_expand_1px': torch.where(expanded.bool(), torch.zeros_like(rgb), rgb)*2-1,
            'boundary_erode_1px': narrow*2-1}


def run(config_path, extraction, output, device='cuda'):
    config = load_config(config_path)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; explicit --device cpu is required to change device')
    target_device = torch.device(device)
    precision = config['train_diffusion']['mixed_precision']
    autocast_enabled = target_device.type == 'cuda' and precision in ('fp16', 'bf16')
    autocast_dtype = torch.float16 if precision == 'fp16' else torch.bfloat16
    root = nonoverwriting_directory(output)
    source = Path(extraction)
    meta = json.loads((source/'extraction.json').read_text('utf-8'))
    for p, digest in meta['input_sha256'].items():
        if file_sha256(Path(p)) != digest: raise ValueError('Extraction inputs changed')
    paths, provenance = resolve_initialization(config)
    if not paths: raise ValueError('Explicit S0 weights are required')
    ids = meta['training_alloy_ids']
    compositions = load_compositions(resolve_project_path(config, config['data']['compositions']))
    conditioner = load_frozen_conditioner(paths, set(ids), compositions)
    unet, encoder, scheduler = load_initialized_models(paths, config['diffusion'], conditioner.dimension)
    if scheduler.config.prediction_type != 'epsilon': raise ValueError('Only epsilon is supported')
    vae_path = resolve_project_path(config, config['train_diffusion']['vae_dir'])
    vae = AutoencoderKL.from_pretrained(vae_path).to(target_device).float().eval().requires_grad_(False)
    unet.to(target_device).float().eval().requires_grad_(False)
    encoder.to(target_device).float().eval().requires_grad_(False)
    proxy = LinearMechanics.load(resolve_project_path(config, config['train_diffusion']['mechanics']['proxy_file']))
    proxy.to(target_device)
    if proxy.composition_names != conditioner.active_features or proxy.artifact['feature_extractor'] != meta['feature_extractor']:
        raise ValueError('Proxy/feature/conditioner versions differ')
    diag = config.get('mechanics_diagnostics', {})
    times = diag.get('timesteps', [100, 200, 300, 450, 600])
    seeds = diag.get('noise_seeds', [20261102, 20261103, 20261104])
    windows = int(diag.get('windows_per_alloy', 4))
    if (times != sorted(set(times)) or not times or any(type(t) is not int or not 0 < t < scheduler.config.num_train_timesteps for t in times)
            or not seeds or len(seeds) != len(set(seeds)) or not 0 < windows <= meta['windows_per_alloy']):
        raise ValueError('Invalid fixed diagnostic grid')
    frame = pd.read_csv(source/'alloy_features.csv')
    scale = frame[list(proxy.features.names)].to_numpy(float).std(0)
    if (scale < 1e-12).any(): raise ValueError('Unidentifiable real feature scale')
    feature_scale = torch.tensor(scale, dtype=torch.float32, device=target_device)
    manifest = pd.read_csv(resolve_project_path(config, config['data']['manifest'])).set_index('alloy_id')
    comp = compositions.set_index('alloy_id')
    results, perturb, clean_grads, reconstruction = [], [], [], []
    alpha = scheduler.alphas_cumprod.to(target_device)
    for alloy in ids:
        path = resolve_manifest_image_path(resolve_project_path(config, config['data']['manifest']),
                                           manifest.loc[alloy, 'image_path'])
        with Image.open(path) as im: image = im.convert('RGB')
        c = torch.tensor(comp.loc[alloy, proxy.composition_names].to_numpy(float),
                         dtype=torch.float32, device=target_device)[None]
        cond = torch.tensor(conditioner.transform_by_alloy_id(compositions, alloy),
                            device=target_device, dtype=torch.float32)[None]
        for w, (x, y) in enumerate(meta['positions'][str(alloy)][:windows]):
            size = meta['patch_size']
            arr = np.asarray(image.crop((x, y, x+size, y+size)), dtype=np.float32).copy()
            clean = torch.from_numpy(arr).permute(2, 0, 1)[None].to(target_device)/127.5-1
            with torch.no_grad():
                d = proxy.features(clean); prediction = proxy(c, d)
                z = vae.encode(clean).latent_dist.mean * provenance['latent_scaling']
                rec = vae.decode(z/provenance['latent_scaling']).sample
                rec_delta = ((proxy.features(rec)-d)/feature_scale)[0].cpu().tolist()
                reconstruction.append({'alloy_id': alloy, 'window': w, 'feature_delta': rec_delta})
                for name, corrupted in perturbations(clean).items():
                    delta = ((proxy(c, proxy.features(corrupted))-prediction)/proxy.target_scale)[0].cpu().tolist()
                    perturb.append({'alloy_id': alloy, 'window': w, 'perturbation': name, 'prediction_delta': delta})
                with torch.autocast(device_type=target_device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                    emb = encoder(cond, drop_mask=torch.zeros(1, dtype=torch.bool, device=target_device))
                for seed in seeds:
                    rng = torch.Generator(device=target_device).manual_seed(seed+alloy*1009+w)
                    noise = torch.randn(z.shape, generator=rng, device=target_device)
                    for time in times:
                        t = torch.tensor([time], device=target_device)
                        noisy = scheduler.add_noise(z, noise, t)
                        with torch.autocast(device_type=target_device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                            eps = unet(noisy, t, class_labels=emb).sample
                        decoded = vae.decode(x0_from_epsilon(noisy, eps, t, alpha)/provenance['latent_scaling']).sample
                        dd = proxy.features(decoded)
                        results.append({'alloy_id': alloy, 'window': w, 'seed': seed, 't': time,
                            'feature_delta': ((dd-d)/feature_scale)[0].cpu().tolist(),
                            'prediction_delta': ((proxy(c, dd)-prediction)/proxy.target_scale)[0].cpu().tolist(),
                            'decoded_outside_fraction': float((decoded.abs()>1).float().mean())})
            leaf = clean.detach().requires_grad_()
            value = proxy(c, proxy.features(leaf))
            norms = []
            for j in range(2):
                g = torch.autograd.grad(value[0, j]/proxy.target_scale[j], leaf, retain_graph=True)[0]
                norms.append(float(g.norm()))
            clean_grads.append({'alloy_id': alloy, 'window': w,
                                'pixel_readout_gradient_norms': norms,
                                'scope': 'readout sensitivity, not target error or U-Net gradient'})
    summary, lower_passed, max_t = [], True, None
    for time in times:
        selected = [r for r in results if r['t'] == time]
        f = np.asarray([r['feature_delta'] for r in selected])
        p = np.asarray([r['prediction_delta'] for r in selected])
        feature_ok = float(np.mean((np.abs(f) <= .25).all(1))) >= .9
        output_ok = [float(np.mean(np.abs(p[:, j]) <= .25)) >= .9 for j in range(2)]
        passed = feature_ok and all(ok or not active for ok, active in zip(output_ok, proxy.active.tolist()))
        lower_passed &= passed
        if lower_passed: max_t = time
        summary.append({'t': time, 'combination_count': len(selected), 'feature_passed': feature_ok,
                        'target_passed': output_ok, 'signed_feature_bias': f.mean(0).tolist(),
                        'signed_prediction_bias': p.mean(0).tolist()})
    identity = {'source_sha256': provenance['sha256'], 'proxy_sha256': file_sha256(resolve_project_path(
        config, config['train_diffusion']['mechanics']['proxy_file'])),
        'vae_sha256': {name: file_sha256(vae_path/name) for name in ('config.json', 'diffusion_pytorch_model.safetensors')},
        'training_alloy_ids': ids, 'prediction_gate_targets': proxy.active.tolist(),
        'device': str(target_device), 'readout_dtype': 'float32',
        'unet_autocast': precision if autocast_enabled else 'float32',
        'latent_encoding': 'posterior mean to isolate forward-noise sensitivity', 'optimizer_updates': 0}
    write_json(root/'timestep_stability.json', {**identity, 'status': 'trajectory_review_pending',
        'candidate_max_t': max_t, 'summary': summary, 'records': results, 'reconstruction': reconstruction,
        'limits': 'Combination counts are not independent samples; real trajectory coverage still required'})
    write_json(root/'perturbations.json', {**identity, 'status': 'hard_feature_review_pending', 'records': perturb,
        'limits': 'Synthetic colour/width perturbations need review against hard morphology; no automatic pass'})
    write_json(root/'clean_input_gradients.json', {**identity, 'status': 'diagnostic_only', 'records': clean_grads})
    return {'status': 'diagnosed_not_overall_m0_passed', 'output': str(root)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--extraction', required=True)
    p.add_argument('--output', required=True); p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = p.parse_args()
    print(json.dumps(run(args.config, args.extraction, args.output, args.device), ensure_ascii=False))


if __name__ == '__main__': main()
