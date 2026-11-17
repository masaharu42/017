"""Single-scale RGB predictor and frozen, differentiable ensemble readout."""
from pathlib import Path
import hashlib
import json
import torch
from torch import nn

SCHEMA = 'image_cnn_mechanics_v1'
TARGETS = ['peak_stress_MPa', 'endpoint_strain_pct']


class SingleScaleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        for i, (a, b) in enumerate(((3, 16), (16, 32), (32, 64))):
            blocks += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU()]
            if i < 2:
                blocks += [nn.MaxPool2d(2)]
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, 20), nn.ReLU(), nn.Dropout(.3),
                                  nn.Linear(20, 12), nn.ReLU(), nn.Dropout(.3), nn.Linear(12, 2))

    def forward(self, images):
        return self.head(self.encoder(images))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CNNMechanics(nn.Module):
    """Input [-1,1] RGB -> physical units. Frozen weights retain input autograd."""
    def __init__(self, artifact, directory):
        super().__init__()
        if artifact.get('schema') != SCHEMA or artifact.get('target_names') != TARGETS:
            raise ValueError('Unsupported CNN artifact')
        if artifact.get('pixel_transform') != 'native_rgb_minus_one_one':
            raise ValueError('Unsupported pixel transform')
        self.artifact, self.composition_names = artifact, []
        self.models = nn.ModuleList()
        means, scales = [], []
        for member in artifact['members']:
            p = Path(directory)/member['weights_file']
            if sha256(p) != member['weights_sha256']:
                raise ValueError('CNN weights changed')
            model = SingleScaleCNN()
            model.load_state_dict(torch.load(p, map_location='cpu', weights_only=True))
            self.models.append(model)
            means.append(member['target_mean']); scales.append(member['target_scale'])
        if not self.models:
            raise ValueError('Empty CNN ensemble')
        self.register_buffer('means', torch.tensor(means, dtype=torch.float32))
        self.register_buffer('scales', torch.tensor(scales, dtype=torch.float32))
        self.register_buffer('target_scale', torch.tensor(artifact['target_scale'], dtype=torch.float32))
        self.register_buffer('active', torch.tensor(artifact['approved_targets'], dtype=torch.bool))
        if (self.means.shape != (len(self.models), 2) or self.scales.shape != self.means.shape
                or self.target_scale.shape != (2,) or self.active.shape != (2,)
                or not torch.isfinite(self.means).all() or not torch.isfinite(self.scales).all()
                or not torch.isfinite(self.target_scale).all()
                or (self.scales <= 0).any() or (self.target_scale <= 0).any()):
            raise ValueError('Invalid CNN target normalization')
        self.requires_grad_(False).eval()

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()), Path(path).parent)

    def train(self, mode=True):
        # The exported readout is always frozen, including BN buffers and dropout.
        return super().train(False)

    def forward(self, images):
        size = self.artifact['patch_size']
        if images.ndim != 4 or images.shape[1:] != (3, size, size):
            raise ValueError('CNN input shape differs from registered native crop')
        return torch.stack([m(images)*self.scales[i]+self.means[i]
                            for i, m in enumerate(self.models)]).mean(0)
