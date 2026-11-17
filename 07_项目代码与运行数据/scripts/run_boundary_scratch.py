"""Fold03 scratch boundary pilot: prepare, zero-update calibration, benchmark, train."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from diffusers import DDPMScheduler
from ebsd_diffusion.training_control import file_sha256
from ebsd_diffusion.utils import save_json
from scripts.run_descriptor_pilot import verify_baseline

ROOT = Path(__file__).resolve().parents[1]

def path_of(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()

def read_json(path):
    return json.loads(Path(path).read_text('utf-8'))

def protocol_at(path):
    p = yaml.safe_load(Path(path).read_text('utf-8'))
    if p['expected_source_step'] != 50000 or p['expected_additional_steps'] != 10000:
        raise ValueError('Need the completed Fold03 baseline for provenance')
    if p['evaluation']['alloy_ids'] != [5, 7, 20] or p['evaluation']['seeds'] != [20261012, 20261013]:
        raise ValueError('Unexpected baseline monitor settings')
    return p

def prepare(path):
    protocol = protocol_at(path)
    output = path_of(protocol['output_root'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'New output directory required: {output}')
    baseline, old, cfg, meta, fingerprints = verify_baseline(protocol, ROOT)
    if meta['effective_total_step'] != 60000:
        raise ValueError('Existing baseline is not raw50k plus 10k')
    source = old['preflight']['initialization']
    scheduler_path = Path(source['paths']['scheduler_dir'])
    old_scheduler = DDPMScheduler.from_pretrained(scheduler_path)
    model = cfg['diffusion']
    fresh_scheduler = DDPMScheduler(num_train_timesteps=int(model['num_train_timesteps']),
        beta_schedule=str(model['beta_schedule']), prediction_type=str(model['prediction_type']), clip_sample=False)
    if not torch.equal(old_scheduler.alphas_cumprod, fresh_scheduler.alphas_cumprod):
        raise ValueError('Fresh noise schedule differs from the old raw50k; stop and inspect')
    cfg = deepcopy(cfg)
    for key in list(cfg):
        if key.startswith('_'): del cfg[key]
    train = cfg['train_diffusion']
    train.pop('initialization', None)
    train.update({'max_steps': 50000, 'batch_size': 32, 'gradient_accumulation_steps': 1,
                  'learning_rate': 1e-4, 'warmup_steps': 1000, 'save_every': 10000,
                  'gradient_checkpointing': False, 'export_raw': True,
                  'fixed_latent_scaling': float(source['latent_scaling']),
                  'output_dir': str(output / 'scratch_boundary'),
                  'boundary_dice': {'max_t': 300, 'max_samples': 8, 'calibration_batches': 8,
                                    'calibration_file': str(output / 'calibration/boundary_calibration.json')}})
    output.mkdir(parents=True, exist_ok=True)
    (output / 'configs').mkdir()
    config_path = output / 'configs/scratch_boundary.yaml'
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), 'utf-8')
    files = [Path(path).resolve(), config_path, ROOT / 'train_diffusion_scratch.py',
             *sorted((ROOT / 'ebsd_diffusion').glob('*.py')), ROOT / 'scripts/run_boundary_scratch.py',
             ROOT / 'scripts/evaluate_boundary_scratch.py',
             ROOT / 'scripts/run_descriptor_pilot.py', ROOT / 'scripts/run_minsnr_ablation.py',
             ROOT / 'scripts/sample_fold03_ab2c_256.py', baseline / 'experiment_manifest.json',
             baseline / 'A_mse/final/run_metadata.json', fingerprints]
    frozen = {str(x): file_sha256(x) for x in files}
    for name, digest in old['preflight']['file_sha256'].items():
        if Path(name).suffix != '.py': frozen[name] = digest
    manifest = {'status':'prepared', 'protocol':protocol, 'protocol_path':str(Path(path).resolve()),
        'config_path':str(config_path), 'frozen_sha256':frozen, 'baseline_root':str(baseline),
        'baseline_source':source, 'training_alloy_ids':old['preflight']['training_alloy_ids'],
        'scientific_scope':'Fold03 scratch network, frozen VAE and baseline latent scaling; compare raw50k, not a matched continuation.'}
    save_json(manifest, output / 'experiment_manifest.json')
    return {'status':'prepared','output':str(output),'initialization':'random_unet_and_encoder','batch_size':32,'gradient_checkpointing':False}

def load_prepared(path):
    protocol = protocol_at(path); output = path_of(protocol['output_root'])
    m = read_json(output / 'experiment_manifest.json')
    if m['protocol'] != protocol or m['protocol_path'] != str(Path(path).resolve()): raise ValueError('Protocol changed')
    for name, digest in m['frozen_sha256'].items():
        if not Path(name).is_file() or file_sha256(Path(name)) != digest: raise ValueError(f'Frozen input changed: {name}')
    return output, m

def execute(path, phase):
    output, m = load_prepared(path)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required; no CPU training fallback')
    expected = {'calibrate':'prepared', 'benchmark':'calibrated', 'train':'benchmarked'}[phase]
    if m['status'] != expected: raise ValueError(f'Expected {expected}; got {m["status"]}')
    if phase != 'calibrate' and file_sha256(output / 'calibration/boundary_calibration.json') != m['calibration_sha256']:
        raise ValueError('Calibration changed')
    if phase == 'train':
        if file_sha256(output / 'benchmark/benchmark.json') != m['benchmark_sha256']: raise ValueError('Benchmark changed')
        bench = read_json(output / 'benchmark/benchmark.json')
        if bench['gpu_name'] != torch.cuda.get_device_name(): raise ValueError('GPU changed; use a new run and benchmark again')
        free, _ = torch.cuda.mem_get_info()
        if free < bench['peak_reserved_bytes'] + 1024**3: raise RuntimeError('Insufficient current free VRAM relative to benchmark plus 1 GiB')
        if shutil.disk_usage(output).free < 12 * 1024**3: raise RuntimeError('Less than 12 GiB free; inspect storage before training')
    folder = output / {'calibrate':'calibration','benchmark':'benchmark','train':'scratch_boundary'}[phase]
    if folder.exists() and any(folder.iterdir()): raise FileExistsError(f'Existing partial output: {folder}; no automatic retry')
    cmd = [sys.executable, '-u', str(ROOT / 'train_diffusion_scratch.py'), '--config', m['config_path'], '--exclude-alloy-id','3', '--output-dir',str(folder)]
    if phase == 'calibrate': cmd.append('--boundary-calibrate')
    if phase == 'benchmark': cmd += ['--benchmark-steps','30','--benchmark-warmup','5']
    (output / 'console_logs').mkdir(exist_ok=True)
    with (output / 'console_logs' / f'{phase}.log').open('x', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end='', flush=True); log.write(line); log.flush()
        if proc.wait(): raise RuntimeError(f'{phase} failed; inspect log; no retry')
    if phase == 'calibrate':
        report = read_json(folder / 'boundary_calibration.json')
        if report['status'] != 'passed' or report['optimizer_updates'] != 0: raise ValueError('Calibration failed')
        m['calibration_sha256'] = file_sha256(folder / 'boundary_calibration.json'); m['status'] = 'calibrated'
    elif phase == 'benchmark':
        report = read_json(folder / 'benchmark.json')
        m['benchmark_sha256'] = file_sha256(folder / 'benchmark.json'); m['status'] = 'benchmarked'
    else:
        report = read_json(folder / 'final/run_metadata.json')
        if report['global_step'] != 50000 or report['source_global_step'] != 0 or report['initialization'] is not None:
            raise ValueError('Expected 50000 fresh optimizer steps')
        m['status'] = 'trained_pending_evaluation'
        m['final_sha256'] = {str(p):file_sha256(p) for p in [folder / 'final/run_metadata.json',
             folder / 'final/unet_raw/diffusion_pytorch_model.safetensors', folder / 'final/unet_raw/config.json',
             folder / 'final/condition_encoder/pytorch_model.bin', folder / 'final/condition_encoder/config.json',
             folder / 'final/composition_conditioner.json']}
    save_json(m, output / 'experiment_manifest.json')
    return {'status':m['status'], 'output':str(output), 'report':report}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','calibrate','benchmark','train'],required=True)
    args=parser.parse_args(); path=args.config.resolve()
    result = prepare(path) if args.phase == 'prepare' else execute(path,args.phase)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__ == '__main__': main()
