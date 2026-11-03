"""Opt-in mechanics continuation, reviewed evidence gates, and zero-update probes."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from physics.linear_mechanics import LinearMechanics, TARGET_NAMES
from physics.cnn_mechanics import CNNMechanics, SCHEMA as CNN_SCHEMA
from physics.soft_mechanics_features import group_mean
from .config import resolve_project_path
from .structure_statistics import x0_from_epsilon
from .training_control import file_sha256


def protocol_signature(config):
    value = deepcopy(config)
    value = {k: v for k, v in value.items() if not k.startswith('_')}
    value['train_diffusion'].pop('output_dir', None)
    value['train_diffusion']['mechanics'].pop('arm', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_mechanics_config(cfg, train, model, *, calibrate=False):
    c = dict(cfg)
    if c.get('arm') not in ('C0', 'C1'):
        raise ValueError('Mechanics arm must be C0 or C1')
    for key, default in [('group_size', 4), ('max_decoded_windows', 8), ('calibration_batches', 200)]:
        c.setdefault(key, default)
        if isinstance(c[key], bool) or not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f'Invalid mechanics {key}')
    if train['batch_size'] % c['group_size'] or c['max_decoded_windows'] < c['group_size']:
        raise ValueError('Mechanics needs complete groups in batch and decode budget')
    if (not train.get('controlled_experiment') or train['gradient_accumulation_steps'] != 1
            or model['prediction_type'] != 'epsilon' or train.get('snr_gamma') is not None
            or any(train.get(k) for k in ('boundary_dice', 'structure_statistics', 'local_descriptors',
                                        'lambda_gb', 'lambda_edge'))):
        raise ValueError('Mechanics requires controlled epsilon/MSE without other auxiliary losses')
    if calibrate and c['arm'] != 'C1':
        raise ValueError('Calibrate the C1 configuration, not C0')
    c.setdefault('gradient_ratio', .15)
    if not math.isfinite(c['gradient_ratio']) or not 0 < c['gradient_ratio'] <= .2:
        raise ValueError('Invalid mechanics gradient ratio')
    return c


class MechanicsDataset(torch.utils.data.Dataset):
    """Add physical composition values and masked alloy-level labels to existing crops."""
    def __init__(self, dataset, compositions, labels_path, composition_names, active):
        self.dataset = dataset
        self.rows = dataset.rows
        self.alloy_ids = dataset.alloy_ids
        c = compositions.set_index('alloy_id')
        y = pd.read_csv(labels_path).set_index('alloy_id')
        if not c.index.is_unique or not y.index.is_unique:
            raise ValueError('Duplicate alloy IDs in mechanics inputs')
        # Optional analysed labels already recomputed after specimen QC.
        self.values = {}
        for i in self.alloy_ids:
            raw = c.loc[i, composition_names].to_numpy(dtype=np.float32)
            target = y.loc[i, list(TARGET_NAMES)].to_numpy(dtype=np.float32)
            mask = np.isfinite(target) & np.asarray(active, dtype=bool)
            for j, name in enumerate(TARGET_NAMES):
                column = name + '_valid'
                if column in y.columns:
                    mask[j] &= str(y.loc[i, column]).lower() in ('true', '1', 'yes')
            if not np.isfinite(raw).all():
                raise ValueError('Non-finite physical composition')
            self.values[i] = raw, np.nan_to_num(target), mask

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        raw, target, mask = self.values[int(item['alloy_id'])]
        item.update(mechanics_composition=torch.tensor(raw), mechanics_target=torch.tensor(target),
                    mechanics_valid=torch.tensor(mask))
        return item


def target_losses(prediction, target, valid, scales, active, weights):
    """Equal target coefficients; missing targets do not double the other target."""
    valid = valid.bool() & active[None, :] & torch.isfinite(target)
    clean = torch.where(valid, target.detach(), prediction.detach())
    errors = F.huber_loss((prediction-clean) / scales, torch.zeros_like(prediction),
                          reduction='none', delta=1.)
    terms = [(errors[:, j] * weights * valid[:, j]).sum() / valid[:, j].sum().clamp_min(1)
             for j in range(2)]
    total = sum(terms[j] * active[j] for j in range(2)) / active.sum().clamp_min(1)
    return total, terms


def gradient_probe(mse, terms, total, modules):
    params, spans = [], {}
    for name, module in modules.items():
        start = len(params)
        params.extend(p for p in module.parameters() if p.requires_grad)
        spans[name] = (start, len(params))
    gradients = [torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
                 for v in (mse, *terms, total)]
    def dot(a, b, span=None):
        lo, hi = span or (0, len(params))
        return sum(float((x.detach().float()*y.detach().float()).sum())
                   for x, y in zip(a[lo:hi], b[lo:hi]) if x is not None and y is not None)
    norms = [math.sqrt(max(0., dot(g, g))) for g in gradients]
    if not all(math.isfinite(v) for v in norms):
        raise RuntimeError('Non-finite mechanics probe')
    def cosine(a, b):
        return dot(gradients[a], gradients[b]) / (norms[a]*norms[b]) if norms[a]*norms[b] > 0 else None
    return {'mse_norm': norms[0], 'peak_norm': norms[1], 'strain_norm': norms[2],
            'mechanics_norm': norms[3], 'target_cosine': cosine(1, 2),
            'peak_mse_cosine': cosine(0, 1), 'strain_mse_cosine': cosine(0, 2),
            'mechanics_mse_cosine': cosine(0, 3),
            'module_norms': {name: [math.sqrt(max(0., dot(g, g, span))) for g in gradients]
                             for name, span in spans.items()}}


def feedback_gradient_modules(unet, condition_encoder):
    """Disjoint parameter blocks, so a global norm does not double-count tensors."""
    modules = {}
    for name, child in unet.named_children():
        if name in ('down_blocks', 'up_blocks'):
            modules.update({f'unet.{name}.{i}': block for i, block in enumerate(child)})
        elif any(p.requires_grad for p in child.parameters()):
            modules['unet.'+name] = child
    modules['condition_encoder'] = condition_encoder
    seen = [id(p) for m in modules.values() for p in m.parameters() if p.requires_grad]
    expected = [id(p) for m in (unet, condition_encoder) for p in m.parameters() if p.requires_grad]
    if len(seen) != len(set(seen)) or set(seen) != set(expected):
        raise ValueError('Gradient block inventory must cover trainable parameters exactly once')
    return modules


class MechanicsRuntime:
    def __init__(self, config, cfg, initialization_report, *, calibrate=False):
        self.cfg, self.signature, self.calibrate = cfg, protocol_signature(config), calibrate
        self.proxy_path = resolve_project_path(config, cfg['proxy_file'])
        artifact = json.loads(self.proxy_path.read_text('utf-8'))
        self.is_cnn = artifact.get('schema') == CNN_SCHEMA
        self.proxy = CNNMechanics.load(self.proxy_path) if self.is_cnn else LinearMechanics.load(self.proxy_path)
        decision_path = resolve_project_path(config, cfg['decision_file'])
        decision = json.loads(decision_path.read_text('utf-8'))
        expected_schema = 'image_cnn_feedback_decision_v1' if self.is_cnn else 'mechanics_m0_decision_v1'
        if decision.get('schema') != expected_schema or decision.get('status') != 'passed':
            raise ValueError('A reviewed M0 applicability decision is required (template is not a pass)')
        if decision.get('proxy_sha256') != file_sha256(self.proxy_path):
            raise ValueError('M0 decision belongs to a different proxy')
        if not initialization_report or decision.get('source_sha256') != initialization_report['sha256']:
            raise ValueError('M0 decision belongs to different source weights/configuration')
        # Evidence is supplied by actual diagnostics + scientific review, never manufactured by fitting.
        root = decision_path.parent
        for gate in ('window_stability', 'perturbations', 'domain', 'timestep_and_trajectory'):
            evidence = decision.get('gates', {}).get(gate, {})
            p = root / evidence.get('path', '__missing__')
            if evidence.get('status') != 'passed' or not p.is_file() or file_sha256(p) != evidence.get('sha256'):
                raise ValueError(f'Missing/changed M0 evidence: {gate}')
            payload = json.loads(p.read_text('utf-8'))
            if payload.get('status') != 'passed':
                raise ValueError(f'M0 evidence itself has not passed: {gate}')
        self.active = decision.get('active_targets')
        if (not isinstance(self.active, list) or len(self.active) != 2
                or not all(isinstance(v, bool) for v in self.active) or not any(self.active)
                or any(a and not b for a, b in zip(self.active, self.proxy.artifact['approved_targets']))):
            raise ValueError('Unvalidated target activated')
        self.proxy.active.copy_(torch.tensor(self.active))
        self.max_t = decision.get('max_t')
        if (type(self.max_t) is not int or not 0 < self.max_t < config['diffusion']['num_train_timesteps']
                or decision.get('group_size') != cfg['group_size']):
            raise ValueError('Invalid locked timestep/group size')
        self.decision_sha = file_sha256(decision_path)
        self.decision = decision
        for p, digest in self.proxy.artifact.get('input_sha256', {}).items():
            if file_sha256(Path(p)) != digest:
                raise ValueError(f'Proxy input changed: {p}')
        self.weight = 0.
        self.records = []
        self.calibration_path = resolve_project_path(config, cfg['calibration_file'])
        if not calibrate:
            calibration = json.loads(self.calibration_path.read_text('utf-8'))
            expected = {'protocol_sha256': self.signature, 'decision_sha256': self.decision_sha,
                        'proxy_sha256': file_sha256(self.proxy_path), 'gpu_name': self.gpu_name()}
            if (calibration.get('status') != 'passed' or calibration.get('optimizer_updates') != 0
                    or any(calibration.get(k) != v for k, v in expected.items())):
                raise ValueError('A matching zero-update mechanics calibration is required')
            self.weight = float(calibration['lambda']) if cfg['arm'] == 'C1' else 0.
            if cfg['arm'] == 'C1' and (not math.isfinite(self.weight) or self.weight <= 0):
                raise ValueError('Invalid calibrated mechanics weight')

    @staticmethod
    def gpu_name():
        return torch.cuda.get_device_name() if torch.cuda.is_available() else None

    def bind_data(self, dataset, labels_path):
        if sorted(dataset.alloy_ids) != sorted(self.proxy.artifact['training_alloy_ids']):
            raise ValueError('Proxy and generator folds differ')
        expected = self.proxy.artifact['labels_sha256']
        if file_sha256(Path(labels_path)) != expected:
            raise ValueError('Feedback labels differ from proxy labels')
        hashes = self.proxy.artifact.get('input_sha256', {})
        for row in dataset.rows:
            path = Path(row['image_path']).resolve()
            if hashes.get(str(path)) != file_sha256(path):
                raise ValueError('Generator images differ from proxy extraction; re-extract and review')

    def bind_vae(self, directory):
        actual = {name: file_sha256(directory/name) for name in
                  ('config.json', 'diffusion_pytorch_model.safetensors')}
        if actual != self.decision.get('vae_sha256'):
            raise ValueError('VAE differs from reviewed M0 VAE')

    def loss(self, vae, noisy, prediction, timesteps, dropped, batch, alpha, latent_scale):
        k = self.cfg['group_size']
        info = {'groups': 0, 'decoded_outside_fraction': None, 'peak_prediction': None,
                'strain_prediction': None, 'image_gradient_norm': None,
                'clean_latent_gradient_norm': None, 'noise_prediction_gradient_norm': None,
                'boundary_saturated_fraction': None, 'image_gradient_boundary_energy_fraction': None}
        group_valid = batch['mechanics_valid'].reshape(-1, k, 2).all(1)
        selected = torch.nonzero((timesteps[::k] <= self.max_t) & ~dropped[::k]
                                 & group_valid.any(1), as_tuple=True)[0]
        selected = selected[:self.cfg['max_decoded_windows']//k]
        if not len(selected):
            zero = prediction.float().sum()*0
            return zero, [zero, zero], info
        indices = (selected[:, None]*k + torch.arange(k, device=selected.device)).flatten()
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            z = x0_from_epsilon(noisy[indices], prediction[indices], timesteps[indices], alpha)
            decoded = vae.decode(z.float()/latent_scale).sample
            target = batch['mechanics_target'][indices[::k]]
            if self.is_cnn:
                values = group_mean(self.proxy(decoded.float()), k)
            else:
                features = group_mean(self.proxy.features(decoded.float()), k)
                raw = batch['mechanics_composition'][indices[::k]]
                values = self.proxy(raw, features)
            total, terms = target_losses(values, target, group_valid[selected], self.proxy.target_scale,
                                         self.proxy.active, alpha.to(prediction.device)[timesteps[indices[::k]]].float())
        info.update(groups=len(selected), decoded_outside_fraction=float((decoded.detach().abs()>1).float().mean()),
                    peak_prediction=float(values[:, 0].detach().mean()) if self.active[0] else None,
                    strain_prediction=float(values[:, 1].detach().mean()) if self.active[1] else None)
        boundary = None
        if not self.is_cnn:
            with torch.no_grad():
                boundary = self.proxy.features.boundary(decoded.detach())
                info['boundary_saturated_fraction'] = float(((boundary < .001) | (boundary > .999)).float().mean())
        if self.calibrate:
            gradients = torch.autograd.grad(total, [decoded, z, prediction], retain_graph=True)
            for name, g in zip(('image_gradient_norm', 'clean_latent_gradient_norm',
                                'noise_prediction_gradient_norm'), gradients):
                if not bool(torch.isfinite(g).all()):
                    raise RuntimeError('Non-finite gradient in mechanics input path')
                info[name] = float(g.detach().float().norm())
            energy = gradients[0].detach().float().square().sum(1)
            if boundary is not None and float(energy.sum()) > 0:
                info['image_gradient_boundary_energy_fraction'] = float(
                    (energy*(boundary > .5)).sum()/energy.sum())
        return total, terms, info

    def calibration_result(self):
        valid = [r for r in self.records if r['mse_norm'] > 0 and r['mechanics_norm'] > 0]
        if len(valid) < max(4, self.cfg['calibration_batches']//10):
            raise RuntimeError('Too few active mechanics probe batches; no calibration exported')
        for r in valid:
            summed = (r['peak_norm'] + r['strain_norm'])/sum(self.active)
            if summed > 0 and r['mechanics_norm']/summed < 1e-4:
                raise RuntimeError('Target gradient cancellation; do not amplify with a large lambda')
        weight = self.cfg['gradient_ratio']*statistics.median(r['mse_norm']/r['mechanics_norm'] for r in valid)
        return {'status': 'passed', 'scope': 'software gradient calibration; not a physical validation',
                'optimizer_updates': 0, 'lambda': weight, 'protocol_sha256': self.signature,
                'decision_sha256': self.decision_sha, 'proxy_sha256': file_sha256(self.proxy_path),
                'gpu_name': self.gpu_name(), 'records': self.records}
