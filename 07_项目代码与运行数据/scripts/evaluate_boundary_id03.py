"""Separate exploratory ID03 A/C comparison; does not alter the frozen pilot."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import torch
import yaml
from diffusers import AutoencoderKL, DDIMScheduler
from PIL import Image
from ebsd_diffusion.composition import load_compositions
from ebsd_diffusion.data import EbsdPatchDataset
from ebsd_diffusion.metrics import describe_image, descriptors_as_dict
from ebsd_diffusion.training_control import file_sha256, load_frozen_conditioner, load_initialized_models, resolve_initialization
from ebsd_diffusion.utils import save_json
from scripts.run_minsnr_ablation import reference_boxes
from scripts.sample_fold03_ab2c_256 import build_montage, sample_image, save_png
from scripts.run_boundary_pilot import ROOT, load_prepared


def evaluate(path):
    output, manifest = load_prepared(path)
    if manifest['status'] not in {'trained_pending_evaluation', 'evaluated_pending_review'}:
        raise ValueError('Complete C training before evaluation')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for evaluation')
    for name, digest in manifest['final_sha256'].items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f'Trained artifact changed: {name}')
    folder = output / 'evaluation_id03'
    if folder.exists():
        raise FileExistsError(f'Evaluation already exists: {folder}')
    config = yaml.safe_load(Path(manifest['config_path']).read_text('utf-8'))
    config['_project_root'] = str(ROOT)
    paths, provenance = resolve_initialization(config)
    if provenance != manifest['baseline_initialization']:
        raise ValueError('Source initialization changed')
    settings = {**manifest['protocol']['evaluation'], 'alloy_ids': [3]}
    device = torch.device('cuda'); dtype = torch.float16
    vae = AutoencoderKL.from_pretrained(config['train_diffusion']['vae_dir']).to(device=device, dtype=dtype).eval()
    compositions = load_compositions(config['data']['compositions'])
    conditioner = load_frozen_conditioner(paths, set(manifest['training_alloy_ids']), compositions)
    dataset = EbsdPatchDataset(config['data']['manifest'], 256, 1, 'evaluation', crop_margins=tuple(config['data']['crop_margins']))
    rows, grids, audits = [], {aid: [] for aid in settings['alloy_ids']}, {}
    for aid in settings['alloy_ids']:
        real = next(r for r in dataset.rows if int(r['alloy_id']) == aid)
        with Image.open(real['image_path']) as im:
            im = im.convert('RGB')
            boxes = reference_boxes(im.width, im.height, 256, config['data']['crop_margins'])
            real_images = []
            for i, box in enumerate(boxes):
                target = folder / f'alloy_{aid:02d}' / 'real' / f'window_{i}.png'
                save_png(im.crop(box), target)
                rows.append({'alloy_id': aid, 'method': 'real', 'box': list(box), 'path': str(target.relative_to(folder)),
                             'metrics': descriptors_as_dict(describe_image(target, 0.18))})
                real_images.append((f'real {i}', target))
            build_montage(folder / f'alloy_{aid:02d}' / 'real_windows.png', [('real windows', real_images)], (256, 256))
    for method, final in [('A_mse', Path(manifest['baseline_root']) / 'A_mse/final'), ('C_boundary', output / 'C_boundary/final')]:
        model = dict(paths)
        model.update({'raw_unet': final / 'unet_raw/diffusion_pytorch_model.safetensors', 'unet_config': final / 'unet_raw',
                      'condition_encoder_dir': final / 'condition_encoder', 'condition_encoder_weights': final / 'condition_encoder/pytorch_model.bin'})
        unet, encoder, _ = load_initialized_models(model, config['diffusion'], conditioner.dimension)
        unet.to(device=device, dtype=dtype).eval(); encoder.to(device=device, dtype=dtype).eval()
        scheduler = DDIMScheduler.from_pretrained(paths['scheduler_dir'])
        for aid in settings['alloy_ids']:
            condition = torch.tensor(conditioner.transform_by_alloy_id(compositions, aid), device=device, dtype=dtype).unsqueeze(0)
            images = []
            for seed in settings['seeds']:
                audit = {}
                image = sample_image(unet, vae, encoder, scheduler, condition, provenance['latent_scaling'], seed, 50, 1.8, 256, 256, 256, 256, sampling_audit=audit)
                key = f'{aid}_{seed}'
                if key in audits and audits[key] != audit:
                    raise ValueError(f'Input mismatch for {key}')
                audits[key] = audit
                target = folder / f'alloy_{aid:02d}' / method / f'seed_{seed}.png'
                save_png(image, target)
                rows.append({'alloy_id': aid, 'method': method, 'seed': seed, 'path': str(target.relative_to(folder)),
                             'sha256': file_sha256(target), 'metrics': descriptors_as_dict(describe_image(target, 0.18))})
                images.append((f'seed {seed}', target))
                print(f'Sampled {method} ID{aid:02d} seed {seed}', flush=True)
            grids[aid].append((method, images))
        del unet, encoder
        torch.cuda.empty_cache()
    for aid, grid in grids.items():
        build_montage(folder / f'alloy_{aid:02d}' / 'comparison_grid.png', grid, (256, 256))
    save_json(rows, folder / 'per_image_metrics.json')
    scalar_rows = [{**{k: v for k, v in row.items() if k != 'metrics'},
                    **{k: v for k, v in row['metrics'].items() if isinstance(v, (int, float))}}
                   for row in rows]
    fields = sorted({key for row in scalar_rows for key in row})
    with (folder / 'per_image_metrics.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar_rows)
    save_json(audits, folder / 'sampling_inputs.json')
    report = {'status': 'screening_only', 'generated_count': 4, 'seeds': settings['seeds'],
              'scope': 'ID03 exploratory held-out report; inspected during previous development, not a blind test; cross-device training screening', 'settings': '256 DDIM50 CFG1.8 raw',
              'artifact_sha256': {p.relative_to(folder).as_posix(): file_sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()}}
    save_json(report, folder / 'run_metadata.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.config.resolve()), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
