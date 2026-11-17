import pytest
from scripts import evaluate_boundary_id03 as heldout


def test_heldout_requires_cuda_and_preserves_existing_output(tmp_path, monkeypatch):
    manifest = {'status': 'trained_pending_evaluation', 'final_sha256': {}}
    monkeypatch.setattr(heldout, 'load_prepared', lambda path: (tmp_path, manifest))
    monkeypatch.setattr(heldout.torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA'):
        heldout.evaluate(tmp_path / 'config.yaml')
    monkeypatch.setattr(heldout.torch.cuda, 'is_available', lambda: True)
    (tmp_path / 'evaluation_id03').mkdir()
    with pytest.raises(FileExistsError, match='already exists'):
        heldout.evaluate(tmp_path / 'config.yaml')
    assert manifest['status'] == 'trained_pending_evaluation'


def test_incomplete_training_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(heldout, 'load_prepared', lambda path: (tmp_path, {'status': 'calibrated'}))
    with pytest.raises(ValueError, match='Complete C training'):
        heldout.evaluate(tmp_path / 'config.yaml')
