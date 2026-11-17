"""Fold03 boundary Dice experiment: reuse A, calibrate without updates, train only C."""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
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
from scripts.run_descriptor_pilot import verify_baseline
from scripts.run_minsnr_ablation import reference_boxes
from scripts.sample_fold03_ab2c_256 import build_montage, sample_image, save_png

ROOT = Path(__file__).resolve().parents[1]


def abs_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_json(path):
    return json.loads(Path(path).read_text('utf-8'))


def load_protocol(path):
    p = yaml.safe_load(Path(path).read_text('utf-8'))
    if p['expected_source_step'] != 50000 or p['expected_additional_steps'] != 10000:
        raise ValueError('Only raw50k plus 10k pilot is supported')
    if p['evaluation']['alloy_ids'] != [5, 7, 20] or p['evaluation']['seeds'] != [20261012, 20261013]:
        raise ValueError('Monitor must use fixed ID05/07/20 and two existing seeds')
    return p


def prepare(path):
    protocol = load_protocol(path)
    output = abs_path(protocol['output_root'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output exists: {output}; select a new output root')
    old_root, old, config, metadata, fingerprints = verify_baseline(protocol, ROOT)
    train = config['train_diffusion']
    if int(train['batch_size']) != 32 or float(train['learning_rate']) != 3e-5 or int(train['warmup_steps']) != 500:
        raise ValueError('Baseline settings differ from approved protocol')
    if metadata.get('effective_total_step') != 60000:
        raise ValueError('A baseline is not completed raw50k+10k')
    tracked = [ROOT / 'train_diffusion.py', *sorted((ROOT / 'ebsd_diffusion').glob('*.py')),
               ROOT / 'scripts/run_boundary_pilot.py', ROOT / 'scripts/run_descriptor_pilot.py',
               ROOT / 'scripts/run_minsnr_ablation.py', ROOT / 'scripts/sample_fold03_ab2c_256.py',
               Path(path).resolve(), fingerprints,
               old_root / 'experiment_manifest.json', old_root / 'A_mse/final/run_metadata.json',
               old_root / 'A_mse/final/unet_raw/diffusion_pytorch_model.safetensors',
               old_root / 'A_mse/final/unet_raw/config.json',
               old_root / 'A_mse/final/condition_encoder/pytorch_model.bin',
               old_root / 'A_mse/final/condition_encoder/config.json']
    frozen = {str(p): file_sha256(p) for p in tracked}
    # Baseline audit includes source data/weights; archived old code was validated by verify_baseline.
    for name, digest in old['preflight']['file_sha256'].items():
        source = Path(name)
        if source.suffix != '.py':
            frozen[str(source)] = digest
    for key in list(config):
        if key.startswith('_'):
            del config[key]
    train['output_dir'] = str(output / 'C_boundary')
    train['boundary_dice'] = {'max_t': 300, 'max_samples': 8, 'calibration_batches': 8,
                              'calibration_file': str(output / 'calibration/boundary_calibration.json')}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'configs').mkdir()
    config_path = output / 'configs/C_boundary.yaml'
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), 'utf-8')
    frozen[str(config_path)] = file_sha256(config_path)
    manifest = {'status': 'prepared', 'protocol': protocol, 'protocol_path': str(Path(path).resolve()),
                'config_path': str(config_path), 'baseline_root': str(old_root), 'baseline_initialization': old['preflight']['initialization'],
                'training_alloy_ids': old['preflight']['training_alloy_ids'], 'frozen_sha256': frozen,
                'comparison_scope': 'Existing A; cross-device training screening; no exhaustive matching claim.'}
    save_json(manifest, output / 'experiment_manifest.json')
    return {'status': 'prepared', 'output': str(output), 'baseline': str(old_root), 'A_retrained': False}


def load_prepared(path):
    protocol = load_protocol(path)
    output = abs_path(protocol['output_root'])
    manifest = read_json(output / 'experiment_manifest.json')
    if manifest['protocol'] != protocol or manifest['protocol_path'] != str(Path(path).resolve()):
        raise ValueError('Protocol/path changed after prepare')
    for name, expected in manifest['frozen_sha256'].items():
        p = Path(name)
        if not p.is_file() or file_sha256(p) != expected:
            raise ValueError(f'Prepared input changed or missing: {p}')
    return output, manifest


def execute(path, phase):
    output, manifest = load_prepared(path)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for calibration/training; no CPU fallback.')
    expected = 'prepared' if phase == 'calibrate' else 'calibrated'
    if manifest['status'] != expected:
        raise ValueError(f'Expected status {expected}, got {manifest["status"]}; no automatic restart')
    if phase == 'train' and file_sha256(output / 'calibration/boundary_calibration.json') != manifest['calibration_sha256']:
        raise ValueError('Calibration changed after measurement; refusing to train')
    folder = output / ('calibration' if phase == 'calibrate' else 'C_boundary')
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f'Partial or existing output preserved: {folder}')
    cmd = [sys.executable, '-u', str(ROOT / 'train_diffusion.py'), '--config', manifest['config_path'],
           '--exclude-alloy-id', '3', '--output-dir', str(folder)]
    if phase == 'calibrate':
        cmd.append('--boundary-calibrate')
    (output / 'console_logs').mkdir(exist_ok=True)
    with (output / 'console_logs' / (phase + '.log')).open('x', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        if proc.wait() != 0:
            raise RuntimeError(f'{phase} failed; inspect {log.name}; no automatic retry')
    if phase == 'calibrate':
        record = read_json(folder / 'boundary_calibration.json')
        if record['status'] != 'passed' or record['optimizer_updates'] != 0:
            raise ValueError('Calibration did not pass without updates')
        manifest['calibration_sha256'] = file_sha256(folder / 'boundary_calibration.json')
        manifest['status'] = 'calibrated'
    else:
        metadata = read_json(folder / 'final/run_metadata.json')
        if metadata['global_step'] != 10000 or metadata['source_global_step'] != 50000 or metadata['initialization'] != manifest['baseline_initialization']:
            raise ValueError('Training completion or source identity mismatch')
        manifest['status'] = 'trained_pending_evaluation'
        manifest['final_sha256'] = {str(p): file_sha256(p) for p in [folder / 'final/run_metadata.json',
            folder / 'final/unet_raw/diffusion_pytorch_model.safetensors', folder / 'final/unet_raw/config.json',
            folder / 'final/condition_encoder/pytorch_model.bin', folder / 'final/condition_encoder/config.json']}
    save_json(manifest, output / 'experiment_manifest.json')
    return {'status': manifest['status'], 'output': str(output), 'A_retrained': False}


def evaluate(path):
    output, manifest = load_prepared(path)
    if manifest['status'] != 'trained_pending_evaluation':
        raise ValueError('Complete C training before evaluation')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for evaluation')
    for name, digest in manifest['final_sha256'].items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f'Trained artifact changed: {name}')
    folder = output / 'evaluation'
    if folder.exists():
        raise FileExistsError(f'Evaluation already exists: {folder}')
    config = yaml.safe_load(Path(manifest['config_path']).read_text('utf-8'))
    config['_project_root'] = str(ROOT)
    paths, provenance = resolve_initialization(config)
    if provenance != manifest['baseline_initialization']:
        raise ValueError('Source initialization changed')
    settings = manifest['protocol']['evaluation']
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
    report = {'status': 'screening_only', 'generated_count': 12, 'seeds': settings['seeds'],
              'scope': manifest['comparison_scope'], 'settings': '256 DDIM50 CFG1.8 raw',
              'artifact_sha256': {p.relative_to(folder).as_posix(): file_sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()}}
    save_json(report, folder / 'run_metadata.json')
    manifest['status'] = 'evaluated_pending_review'
    save_json(manifest, output / 'experiment_manifest.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare', 'calibrate', 'train', 'evaluate'], required=True)
    args = parser.parse_args()
    result = prepare(args.config.resolve()) if args.phase == 'prepare' else evaluate(args.config.resolve()) if args.phase == 'evaluate' else execute(args.config.resolve(), args.phase)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
