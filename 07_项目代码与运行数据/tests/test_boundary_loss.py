import pytest
import torch

from ebsd_diffusion.boundary_loss import boundary_dice_loss


def test_loss_rewards_boundary_alignment_and_backpropagates():
    target = torch.ones(1, 3, 16, 16)
    target[:, :, :, 3:5] = -1
    aligned = target.clone().requires_grad_(True)
    shifted = torch.ones_like(target)
    shifted[:, :, :, 10:12] = -1
    loss = boundary_dice_loss(aligned, target)
    assert loss < boundary_dice_loss(shifted, target)
    loss.backward()
    assert aligned.grad is not None and torch.isfinite(aligned.grad).all()
    assert aligned.grad.abs().sum() > 0


def test_validation_and_empty_masks_are_finite():
    x = torch.ones(1, 3, 8, 8, requires_grad=True)
    loss = boundary_dice_loss(x, x)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    with pytest.raises(ValueError):
        boundary_dice_loss(x, x[:, :1])
