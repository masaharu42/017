"""Prepare a reviewed C0/C1 pair; run each phase explicitly, never auto-train."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import yaml

from ebsd_diffusion.mechanics_runtime import MechanicsRuntime, protocol_signature, validate_mechanics_config
from ebsd_diffusion.training_control import file_sha256, resolve_initialization
from scripts.run_minsnr_ablation import make_arm_configs, preflight, absolute

ROOT = Path(__file__).resolve().parents[1]


def assert_pair(configs):
    if protocol_signature(configs['C0']) != protocol_signature(configs['C1']):
        raise ValueError('C0/C1 differ beyond arm/output_dir')
    if [configs[n]['train_diffusion']['mechanics']['arm'] for n in ('C0', 'C1')] != ['C0', 'C1']:
        raise ValueError('Incorrect mechanics arm assignment')


def build_configs(protocol):
    _, old = make_arm_configs(protocol, ROOT)
    base = old['A_mse']
    base['train_diffusion'].update(track_source_effective_step=True, snr_gamma=None)
    output = absolute(ROOT, protocol['output_root'])
    shared = dict(protocol['mechanics'])
    for key in ('proxy_file', 'decision_file', 'labels_file'):
        shared[key] = str(absolute(ROOT, shared[key]))
    shared['calibration_file'] = str(output/'calibration/mechanics_calibration.json')
    configs = {}
    for name in ('C0', 'C1'):
        cfg = deepcopy(base)
        cfg['train_diffusion'].update(output_dir=str(output/name), mechanics={**shared, 'arm': name})
        validate_mechanics_config(cfg['train_diffusion']['mechanics'], cfg['train_diffusion'], cfg['diffusion'])
        configs[name] = cfg
    assert_pair(configs)
    return output, configs, shared


def draft(path):
    protocol = yaml.safe_load(Path(path).read_text('utf-8'))
    output, configs, _ = build_configs(protocol)
    destination = output.with_name(output.name+'_draft')/'configs'
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Draft configurations already exist')
    destination.mkdir(parents=True, exist_ok=True)
    for name, cfg in configs.items():
        (destination/f'{name}.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), 'utf-8')
    return destination


def prepare(path):
    protocol = yaml.safe_load(Path(path).read_text('utf-8'))
    output, configs, shared = build_configs(protocol)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new output root')
    # Existing preflight validates source weights, images, VAE and fold isolation.
    report = preflight(protocol, ROOT, {'A_mse': configs['C0']})
    source_meta = json.loads(Path(configs['C0']['train_diffusion']['initialization']['run_metadata']).read_text())
    if source_meta.get('effective_total_step', source_meta['global_step']) != protocol['expected_effective_step']:
        raise ValueError('Wrong effective S0 step')
    for name, cfg in configs.items():
        cfg['_project_root'] = str(ROOT)
        _, initialization = resolve_initialization(cfg)
        runtime = MechanicsRuntime(cfg, cfg['train_diffusion']['mechanics'], initialization, calibrate=name=='C1') if name=='C1' else None
        if runtime:
            runtime.bind_vae(Path(cfg['train_diffusion']['vae_dir']))
        cfg.pop('_project_root')
    for p in [Path(path).resolve(), *sorted((ROOT/'physics').glob('*.py')),
              ROOT/'scripts/run_mechanics_pair.py', ROOT/'scripts/validate_mechanics_m0.py',
              ROOT/'scripts/probe_mechanics_gradients.py',
              ROOT/'scripts/probe_mechanics_inputs.py',
              *(Path(shared[key]) for key in ('proxy_file', 'decision_file', 'labels_file'))]:
        report['file_sha256'][str(p)] = file_sha256(p)
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, cfg in configs.items():
        p = output/'configs'/f'{name}.yaml'; p.parent.mkdir(exist_ok=True)
        p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), 'utf-8')
        paths[name] = str(p)
        report['file_sha256'][str(p)] = file_sha256(p)
    manifest = {'status': 'prepared_not_trained', 'protocol': protocol,
                'configs': paths, 'preflight': report}
    (output/'experiment_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), 'utf-8')
    return manifest


def load_prepared(path):
    protocol = yaml.safe_load(Path(path).read_text('utf-8'))
    output = absolute(ROOT, protocol['output_root'])
    m = json.loads((output/'experiment_manifest.json').read_text('utf-8'))
    if m['protocol'] != protocol:
        raise ValueError('Protocol changed after prepare')
    for p, digest in m['preflight']['file_sha256'].items():
        if file_sha256(Path(p)) != digest:
            raise ValueError(f'File changed after prepare: {p}')
    return protocol, output, m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--phase', choices=['draft', 'prepare', 'calibrate', 'train-c0', 'train-c1'], required=True)
    args = p.parse_args()
    if args.phase == 'draft':
        print(f'Diagnostic configs only, M0 not approved: {draft(args.config)}')
        return
    if args.phase == 'prepare':
        prepare(args.config)
        print('Prepared only; no updates or model calibration performed.')
        return
    protocol, output, manifest = load_prepared(args.config)
    arm = 'C0' if args.phase == 'train-c0' else 'C1'
    command = [sys.executable, str(ROOT/'train_diffusion.py'), '--config', manifest['configs'][arm],
               '--exclude-alloy-id', str(protocol['exclude_alloy_id'])]
    if args.phase == 'calibrate':
        command += ['--mechanics-calibrate', '--output-dir', str(output/'calibration')]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
