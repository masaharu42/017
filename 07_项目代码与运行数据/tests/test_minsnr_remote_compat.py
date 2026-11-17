from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import train_diffusion


def test_existing_structure_descriptors_and_gradients_are_preserved():
    images = torch.linspace(-1, 1, 96).reshape(2, 3, 4, 4).requires_grad_()
    gb, edge = train_diffusion.soft_structure_descriptors(images)
    rgb = (images + 1.0) * 0.5
    expected_gb = torch.sigmoid((0.22 - rgb.amax(dim=1, keepdim=True)) / 0.035).mean((1, 2, 3))
    kernel = images.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    gray = rgb.mean(1, keepdim=True)
    expected_edge = torch.sqrt(
        F.conv2d(gray, kernel, padding=1).square()
        + F.conv2d(gray, kernel.transpose(2, 3), padding=1).square() + 1e-8
    ).mean((1, 2, 3))
    assert torch.equal(gb, expected_gb)
    assert torch.equal(edge, expected_edge)
    (gb.sum() + edge.sum()).backward()
    assert torch.isfinite(images.grad).all()


def test_existing_structure_cli_is_preserved(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_diffusion.py", "--lambda-gb", "2", "--lambda-edge", "3", "--structure-max-t", "150"])
    args = train_diffusion.parse_args()
    assert (args.lambda_gb, args.lambda_edge, args.structure_max_t) == (2, 3, 150)


@pytest.mark.parametrize("field", ["lambda_gb", "lambda_edge"])
def test_controlled_pilot_rejects_nonzero_structure_loss(monkeypatch, field):
    args = SimpleNamespace(config=None, resume_from=None, max_steps=None,
                           lambda_gb=None, lambda_edge=None, structure_max_t=None)
    config = {"seed": 1, "data": {}, "composition": {}, "diffusion": {"prediction_type": "epsilon"},
              "train_diffusion": {"controlled_experiment": True, field: 1.0}}
    monkeypatch.setattr(train_diffusion, "parse_args", lambda: args)
    monkeypatch.setattr(train_diffusion, "load_config", lambda _: config)
    with pytest.raises(ValueError, match="structure loss weights to be zero"):
        train_diffusion.main()
