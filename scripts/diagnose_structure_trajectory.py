"""Read-only DDIM trajectory inspection and separate exploratory ID03 sampling.

Adds no training/config changes to the frozen structure-statistics experiment.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml
from diffusers import AutoencoderKL, DDIMScheduler
from PIL import Image
from skimage import measure, morphology

from ebsd_diffusion.composition import load_compositions
from ebsd_diffusion.data import EbsdPatchDataset
from ebsd_diffusion.metrics import describe_image, descriptors_as_dict
from ebsd_diffusion.training_control import (
    file_sha256, resolve_initialization, load_frozen_conditioner,
    load_initialized_models, tensor_fingerprint,
)
from ebsd_diffusion.utils import save_json, tensor_to_pil
from scripts.run_minsnr_ablation import reference_boxes
from scripts.run_structure_statistics import load_prepared, completed
from scripts.sample_fold03_ab2c_256 import sample_image, save_png, build_montage

ROOT = Path(__file__).resolve().parents[1]
TRACE_IDS = (14, 15, 19, 20)
TRACE_SEEDS = (20261012, 20261013)
CAPTURE_T = (980, 800, 600, 400, 300, 200, 100, 0)
HELDOUT_SEEDS = tuple(range(20261012, 20261017))


def metrics(path):
    values = descriptors_as_dict(describe_image(path, .18))
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
    mask = morphology.binary_dilation(rgb.max(2) <= .18, morphology.disk(1))
    labels = measure.label(~mask, connectivity=1)
    areas = [r.area for r in measure.regionprops(labels) if r.area >= 9]
    values['largest_region_area_fraction'] = max(areas, default=0) / labels.size
    values['valid_regions'] = bool(areas)
    return {k: v for k, v in values.items() if not isinstance(v, (list, dict))}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader(); writer.writerows(rows)


@torch.inference_mode()
def trace_sample(unet, vae, encoder, scheduler, condition, scaling, seed,
                 capture_timesteps, capture, *, steps=50, guidance=1.8, size=256):
    """Same DDIM/CFG/RNG path as sample_image; capture must not consume RNG."""
    device, dtype = condition.device, condition.dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler.set_timesteps(steps, device=device)
    actual = [int(t) for t in scheduler.timesteps.cpu().tolist()]
    if not set(capture_timesteps) <= set(actual):
        raise ValueError('Requested timesteps do not belong to the actual DDIM grid')
    factor = 2 ** (len(vae.config.block_out_channels) - 1)
    if size % factor:
        raise ValueError('Image size does not fit the VAE scale')
    latent = torch.randn((1, int(vae.config.latent_channels), size//factor, size//factor),
                         generator=generator, device=device, dtype=dtype) * scheduler.init_noise_sigma
    audit = {'initial_latent_sha256': tensor_fingerprint(initial_latents=latent),
             'latent_shape': list(latent.shape), 'latent_dtype': str(latent.dtype),
             'timesteps': actual, 'eta': 0.0}
    cond = encoder(condition)
    null = encoder.unconditional(1, device=device, dtype=dtype)
    for t in scheduler.timesteps:
        pred = unet(torch.cat([latent, latent]), t, class_labels=torch.cat([null, cond])).sample
        uncond, conditional = pred.chunk(2)
        pred = uncond + guidance * (conditional - uncond)
        result = scheduler.step(pred, t, latent, generator=generator)
        if int(t) in capture_timesteps:
            x0 = result.pred_original_sample
            decoded = vae.decode(x0 / float(scaling)).sample
            # Explicitly guard against a diagnostic callback changing sampling RNG.
            before = generator.get_state().clone()
            capture(int(t), decoded, {'x0_abs_max': float(x0.float().abs().max()),
                'decoded_outside_pm1_fraction': float((decoded.float().abs() > 1).float().mean()),
                'decoded_nonfinite_fraction': float((~torch.isfinite(decoded)).float().mean())})
            if not torch.equal(before, generator.get_state()):
                raise RuntimeError('Capture changed the sampler RNG state')
        latent = result.prev_sample
    final = tensor_to_pil(vae.decode(latent / float(scaling)).sample[0])
    return final, audit


def plot_curves(rows, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for aid in TRACE_IDS:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
        fields = [('grain_count', 'RGB region count'),
                  ('largest_region_area_fraction', 'Largest region / window'),
                  ('gb_fraction', 'Boundary fraction')]
        for ax, (key, title) in zip(axes, fields):
            for method, color in [('A_mse', '#2563eb'), ('B_statistics', '#e76f26')]:
                for seed in TRACE_SEEDS:
                    rr = [r for r in rows if r['alloy_id']==aid and r['method']==method
                          and r['seed']==seed and r['stage']=='predicted_x0']
                    rr.sort(key=lambda r: -r['t'])
                    ax.plot([r['t'] for r in rr], [r[key] for r in rr], marker='o', color=color,
                            linestyle='-' if seed==TRACE_SEEDS[0] else '--',
                            label=f'{method} / {seed}')
            ax.axvspan(0, 300, color='gray', alpha=.12)
            ax.set_xlim(1000, 0); ax.set_title(title); ax.set_xlabel('DDIM timestep (generation direction →)')
        axes[0].legend(fontsize=7)
        fig.suptitle(f'ID{aid:02d}: high-noise x0 estimates are provisional; shaded t<=300')
        fig.tight_layout(); fig.savefig(folder/f'alloy_{aid:02d}/trajectory_curves.png', dpi=150)
        plt.close(fig)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Run on the server CUDA environment; no CPU fallback')
    if args.mode in ('all', 'trace'):
        import matplotlib  # Fail before generation if the plotting dependency is missing.
    experiment = args.experiment.resolve()
    raw_manifest = json.loads((experiment/'experiment_manifest.json').read_text('utf-8'))
    checked_output, manifest = load_prepared(Path(raw_manifest['protocol_path']))
    if checked_output.resolve() != experiment or manifest['status'] != 'evaluated_pending_review':
        raise ValueError('Use the completed and reviewed structure-statistics experiment')
    frozen = dict(manifest['preflight']['file_sha256'])
    frozen.update(manifest['final_sha256'])
    for path, expected in frozen.items():
        if file_sha256(Path(path)) != expected:
            raise ValueError(f'Frozen input changed: {path}')
    for method in ('A_mse', 'B_statistics'):
        completed(experiment, manifest, method)
    if set(manifest['preflight']['training_alloy_ids']) != set(range(1, 23)) - {3}:
        raise ValueError('Expected Fold03 training IDs with ID03 excluded')
    monitor = experiment/'evaluation/run_metadata.json'
    monitor_state = json.loads(monitor.read_text('utf-8'))
    for path, expected in monitor_state.get('artifact_sha256', {}).items():
        if file_sha256(experiment/'evaluation'/path) != expected:
            raise ValueError(f'Monitor artifact changed: {path}')
    if args.mode in ('all', 'heldout') and not args.review_note.read_text('utf-8').strip():
        raise ValueError('A saved monitor conclusion is required for the separate ID03 report')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory; preserving {output}')
    cfg = yaml.safe_load(Path(manifest['configs']['A_mse']).read_text('utf-8'))
    cfg['_project_root'] = str(ROOT)
    paths, source = resolve_initialization(cfg)
    compositions = load_compositions(cfg['data']['compositions'])
    conditioner = load_frozen_conditioner(paths, set(manifest['preflight']['training_alloy_ids']), compositions)
    if cfg['diffusion']['prediction_type'] != 'epsilon':
        raise ValueError('This diagnostic reproduces the epsilon/DDIM pilot')
    output.mkdir(parents=True)
    if args.mode in ('all', 'heldout'):
        shutil.copy2(args.review_note, output/'monitor_review.md')
    vae = AutoencoderKL.from_pretrained(cfg['train_diffusion']['vae_dir']).to('cuda', dtype=torch.float16).eval()
    trace_rows, heldout_rows, audits, models = [], [], {}, {}
    methods = ('start_A', 'A_mse', 'B_statistics') if args.mode in ('all', 'heldout') else ('A_mse', 'B_statistics')
    heldout_grids = []
    if args.mode in ('all', 'heldout'):
        dataset = EbsdPatchDataset(cfg['data']['manifest'], 256, 1, 'evaluation', crop_margins=tuple(cfg['data']['crop_margins']))
        real = next(r for r in dataset.rows if int(r['alloy_id'])==3)
        refs = []
        with Image.open(real['image_path']) as im:
            im = im.convert('RGB')
            for j, box in enumerate(reference_boxes(im.width, im.height, 256, cfg['data']['crop_margins'])):
                path = output/f'heldout_ID03/real/window_{j}.png'; save_png(im.crop(box), path)
                heldout_rows.append({'alloy_id':3, 'method':'real', 'window':j, 'box':json.dumps(box), **metrics(path)})
                refs.append((f'real {j}', path))
        heldout_grids.append(('real windows (not seed-paired targets)', refs))
    for method in methods:
        model = dict(paths)
        if method != 'start_A':
            final = experiment/method/'final'
            model.update(raw_unet=final/'unet_raw/diffusion_pytorch_model.safetensors', unet_config=final/'unet_raw',
                condition_encoder_dir=final/'condition_encoder', condition_encoder_weights=final/'condition_encoder/pytorch_model.bin')
        models[method] = {k: {'path':str(model[k]), 'sha256':file_sha256(model[k])}
                          for k in ('raw_unet','condition_encoder_weights')}
        unet, encoder, _ = load_initialized_models(model, cfg['diffusion'], conditioner.dimension)
        unet.to('cuda', dtype=torch.float16).eval(); encoder.to('cuda', dtype=torch.float16).eval()
        def scheduler():
            s = DDIMScheduler.from_pretrained(paths['scheduler_dir'])
            if s.config.clip_sample:
                raise ValueError('Unexpected sample clipping; keep the original sampler configuration')
            return s
        if args.mode in ('all', 'trace') and method != 'start_A':
            for aid in TRACE_IDS:
                condition = torch.tensor(conditioner.transform_by_alloy_id(compositions, aid), device='cuda', dtype=torch.float16)[None]
                for seed in TRACE_SEEDS:
                    folder = output/f'trajectories/alloy_{aid:02d}/{method}/seed_{seed}'
                    tiles = []
                    def capture(t, decoded, diagnostics):
                        if diagnostics['decoded_nonfinite_fraction']:
                            raise RuntimeError(f'Nonfinite x0 decode: {method} {aid} {seed} {t}')
                        path = folder/f'x0_t{t:04d}.png'; save_png(tensor_to_pil(decoded[0]), path)
                        trace_rows.append({'alloy_id':aid, 'method':method, 'seed':seed, 't':t,
                                           'stage':'predicted_x0', **diagnostics, **metrics(path)})
                        tiles.append((f't={t}', path))
                    image, audit = trace_sample(unet, vae, encoder, scheduler(), condition, source['latent_scaling'], seed, CAPTURE_T, capture)
                    # Match the already reviewed endpoint, catching implementation/runtime drift.
                    endpoint = experiment/f'evaluation/alloy_{aid:02d}/{method}/seed_{seed}.png'
                    equal = np.array_equal(np.asarray(image), np.asarray(Image.open(endpoint).convert('RGB')))
                    if not equal:
                        raise ValueError(f'Trace endpoint differs from reviewed image: {endpoint}; inspect runtime/precision')
                    path = folder/'final.png'; save_png(image, path); tiles.append(('final', path))
                    build_montage(folder/'trajectory_grid.png', [(method, tiles)], (256,256))
                    trace_rows.append({'alloy_id':aid, 'method':method, 'seed':seed, 't':0, 'stage':'final', **metrics(path)})
                    key = f'trace_{aid}_{seed}'
                    if key in audits and audits[key]!=audit: raise ValueError('Trace initial inputs differ')
                    audits[key] = audit
                    print(f'TRACE {method} ID{aid:02d} seed={seed} endpoint_match=True', flush=True)
        if args.mode in ('all', 'heldout'):
            condition = torch.tensor(conditioner.transform_by_alloy_id(compositions, 3), device='cuda', dtype=torch.float16)[None]
            tiles = []
            for seed in HELDOUT_SEEDS:
                audit = {}
                image = sample_image(unet, vae, encoder, scheduler(), condition, source['latent_scaling'], seed,
                                     50, 1.8, 256,256,256,256, sampling_audit=audit)
                key = f'ID03_{seed}'
                if key in audits and audits[key]!=audit: raise ValueError('ID03 sampling inputs differ')
                audits[key] = audit
                path = output/f'heldout_ID03/{method}/seed_{seed}.png'; save_png(image, path)
                heldout_rows.append({'alloy_id':3, 'method':method, 'seed':seed, **metrics(path)})
                tiles.append((f'seed {seed}', path)); print(f'ID03 {method} seed={seed}', flush=True)
            heldout_grids.append((method, tiles))
        del unet, encoder
        torch.cuda.empty_cache()
    if trace_rows:
        write_csv(output/'trajectories/trajectory_metrics.csv', trace_rows)
        plot_curves(trace_rows, output/'trajectories')
        for aid in TRACE_IDS:
            grid = []
            for method in ('A_mse','B_statistics'):
                for seed in TRACE_SEEDS:
                    folder = output/f'trajectories/alloy_{aid:02d}/{method}/seed_{seed}'
                    grid.append((f'{method} seed={seed}', [(f't={t}',folder/f'x0_t{t:04d}.png') for t in CAPTURE_T]+[('final',folder/'final.png')]))
            build_montage(output/f'trajectories/alloy_{aid:02d}/comparison_grid.png', grid, (256,256))
    if heldout_rows:
        write_csv(output/'heldout_ID03/per_image_metrics.csv', heldout_rows)
        build_montage(output/'heldout_ID03/comparison_grid.png', heldout_grids, (256,256))
    save_json(audits, output/'sampling_inputs.json')
    for path, expected in frozen.items():
        if file_sha256(Path(path)) != expected: raise ValueError(f'Input changed during diagnostic: {path}')
    save_json({'status':'completed', 'mode':args.mode, 'training_updates':0,
        'trace_count':16 if trace_rows else 0, 'heldout_generated_count':15 if heldout_rows else 0,
        'ids_trace':list(TRACE_IDS), 'trace_seeds':list(TRACE_SEEDS), 'heldout_id':3,
        'capture_timesteps':list(CAPTURE_T), 'steps':50, 'cfg':1.8, 'dtype':'float16',
        'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__, 'cuda':torch.version.cuda,
        'experiment':str(experiment), 'models':models, 'script_sha256':file_sha256(Path(__file__)),
        'limits':['High-noise x0 snapshots are provisional, not final physical structures.',
                  'RGB connected regions are not CTF grain counts.', 'ID03 is exploratory, previously inspected, not a blind test.'],
        'artifact_sha256':{str(p.relative_to(output)):file_sha256(p) for p in output.rglob('*') if p.is_file()}},
        output/'run_metadata.json')
    print(f'COMPLETED: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=Path('08_训练与生成结果/14_Fold03_MSE_结构统计_5k_运行'))
    parser.add_argument('--mode',choices=('all','trace','heldout'),default='all')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--review-note',type=Path,default=ROOT/'docs/structure_statistics_monitor_review_20261101.md')
    run(parser.parse_args())
