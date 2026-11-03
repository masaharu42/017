import numpy as np
import pytest
import torch

from test_minsnr_ablation import tiny_protocol
from scripts.run_minsnr_ablation import make_arm_configs, PROJECT_ROOT
from scripts.diagnose_structure_trajectory import trace_sample
from scripts.sample_fold03_ab2c_256 import sample_image
from ebsd_diffusion.training_control import resolve_initialization, load_initialized_models
from diffusers import AutoencoderKL, DDIMScheduler


def test_trajectory_capture_preserves_original_sampler_endpoint(tiny_protocol):
    _, protocol = tiny_protocol
    _, arms = make_arm_configs(protocol, PROJECT_ROOT)
    cfg = arms['A_mse']; cfg['_project_root'] = str(PROJECT_ROOT)
    paths, source = resolve_initialization(cfg)
    unet, encoder, _ = load_initialized_models(paths, cfg['diffusion'], 1)
    unet.eval(); encoder.eval()
    vae = AutoencoderKL.from_pretrained(cfg['train_diffusion']['vae_dir']).eval()
    condition = torch.tensor([[.25]])
    def scheduler(): return DDIMScheduler.from_pretrained(paths['scheduler_dir'])
    original_audit = {}
    previous_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        original = sample_image(unet, vae, encoder, scheduler(), condition, source['latent_scaling'],
                                101, 2, 1.8, 8,8,8,8, sampling_audit=original_audit)
        records = []
        def capture(t, decoded, metadata):
            records.append((t, decoded.clone(), metadata))
        traced, audit = trace_sample(unet, vae, encoder, scheduler(), condition, source['latent_scaling'],
                                      101, [5,0], capture, steps=2, size=8)
        assert np.array_equal(np.asarray(original), np.asarray(traced))
        assert audit == original_audit
        assert [r[0] for r in records] == [5,0]
        assert all(r[1].shape == (1,3,8,8) and r[2]['decoded_nonfinite_fraction']==0 for r in records)
        assert all(p.grad is None for p in unet.parameters())
        with pytest.raises(ValueError, match='actual DDIM grid'):
            trace_sample(unet, vae, encoder, scheduler(), condition, source['latent_scaling'],
                         101, [9], capture, steps=2, size=8)
    finally:
        torch.set_num_threads(previous_threads)
