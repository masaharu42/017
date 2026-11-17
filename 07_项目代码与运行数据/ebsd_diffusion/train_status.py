"""Stdout lines for a training step. They do not change the loss."""
from __future__ import annotations

from pathlib import Path

import torch


def require_fresh_output_dir(path: Path, resume_from: Path | None = None) -> None:
    if resume_from is not None:
        return
    if path.exists() and any(child.name != ".DS_Store" for child in path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. Choose a new output_root; this run will not overwrite it."
        )


def cuda_memory_gb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        float(torch.cuda.memory_allocated() / 1024**3),
        float(torch.cuda.max_memory_allocated() / 1024**3),
    )


def format_step_line(step: int, total: int, loss: float, timing: dict | None, extras: list[str] | None = None) -> str:
    allocated, peak = cuda_memory_gb()
    elapsed = timing["elapsed"] if timing is not None else "--:--:--"
    remaining = timing["estimated_remaining"] if timing is not None else "--:--:--"
    extra = " ".join(extras or [])
    gap = " " if extra else ""
    return (
        f"step {step}/{total} loss {loss:.4f}{gap}{extra} "
        f"elapsed {elapsed} remaining {remaining} "
        f"gpu {allocated:.2f}GB peak {peak:.2f}GB"
    )
