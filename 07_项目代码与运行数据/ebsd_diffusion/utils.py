from __future__ import annotations

import csv
import json
import math
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(value, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_csv_row(path: str | Path, row: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_content = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not has_content:
            writer.writeheader()
        writer.writerow(row)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    total_seconds = int(round(seconds))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class RunProgressTracker:
    """Persist stage, elapsed time, and ETA without coupling to a UI logger."""

    def __init__(
        self,
        output_dir: str | Path,
        run_name: str,
        total_units: int,
        unit_name: str,
        *,
        initial_units: int = 0,
        previous_elapsed_seconds: float = 0.0,
        session_started_monotonic: float | None = None,
        file_prefix: str = "run",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = str(run_name)
        self.total_units = max(int(total_units), 0)
        self.unit_name = str(unit_name)
        self.initial_units = max(int(initial_units), 0)
        self.completed_units = self.initial_units
        self.previous_elapsed_seconds = max(float(previous_elapsed_seconds), 0.0)
        self.session_started_monotonic = (
            float(session_started_monotonic)
            if session_started_monotonic is not None
            else time.perf_counter()
        )
        self.started_at_local = datetime.now().astimezone()
        self.stage = "initializing"
        self.latest_path = self.output_dir / f"{file_prefix}_progress.json"
        self.history_path = self.output_dir / f"{file_prefix}_progress_history.jsonl"
        self.update(self.completed_units, status="running", write_history=True)

    @property
    def elapsed_seconds(self) -> float:
        session_elapsed = max(time.perf_counter() - self.session_started_monotonic, 0.0)
        return self.previous_elapsed_seconds + session_elapsed

    def set_stage(
        self,
        stage: str,
        completed_units: int | None = None,
        details: dict | None = None,
    ) -> dict:
        self.stage = str(stage)
        return self.update(
            self.completed_units if completed_units is None else completed_units,
            details=details,
            status="running",
            write_history=True,
        )

    def update(
        self,
        completed_units: int,
        *,
        details: dict | None = None,
        status: str = "running",
        write_history: bool = True,
    ) -> dict:
        self.completed_units = min(max(int(completed_units), 0), self.total_units)
        elapsed = self.elapsed_seconds
        session_completed = max(self.completed_units - self.initial_units, 0)
        session_elapsed = max(time.perf_counter() - self.session_started_monotonic, 0.0)

        if self.previous_elapsed_seconds > 0 and self.completed_units > 0:
            units_per_second = self.completed_units / max(elapsed, 1.0e-9)
        elif session_completed > 0:
            units_per_second = session_completed / max(session_elapsed, 1.0e-9)
        else:
            units_per_second = None

        remaining_units = max(self.total_units - self.completed_units, 0)
        remaining_seconds = (
            remaining_units / units_per_second
            if units_per_second is not None and units_per_second > 0
            else None
        )
        now = datetime.now().astimezone()
        eta_at_local = now + timedelta(seconds=remaining_seconds) if remaining_seconds is not None else None
        percent = (
            100.0 * self.completed_units / self.total_units
            if self.total_units > 0
            else 100.0
        )
        payload = {
            "run_name": self.run_name,
            "status": str(status),
            "flow_stage": self.stage,
            "unit_name": self.unit_name,
            "completed_units": self.completed_units,
            "total_units": self.total_units,
            "percent_complete": round(percent, 3),
            "started_at_local": self.started_at_local.isoformat(timespec="seconds"),
            "last_update_local": now.isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 3),
            "elapsed": format_duration(elapsed),
            "estimated_remaining_seconds": (
                round(remaining_seconds, 3) if remaining_seconds is not None else None
            ),
            "estimated_remaining": format_duration(remaining_seconds),
            "estimated_finish_local": (
                eta_at_local.isoformat(timespec="seconds") if eta_at_local is not None else None
            ),
            "average_seconds_per_unit": (
                round(1.0 / units_per_second, 6)
                if units_per_second is not None and units_per_second > 0
                else None
            ),
            "details": details or {},
        }
        save_json(payload, self.latest_path)
        if write_history:
            with self.history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def finish(self, completed_units: int | None = None, details: dict | None = None) -> dict:
        self.stage = "completed"
        return self.update(
            self.total_units if completed_units is None else completed_units,
            details=details,
            status="completed",
            write_history=True,
        )


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    value = tensor.detach().float().cpu().clamp(-1, 1)
    value = ((value + 1.0) * 127.5).round().to(torch.uint8)
    value = value.permute(1, 2, 0).numpy()
    return Image.fromarray(value, mode="RGB")


def center_crop_pil(image: Image.Image, height: int, width: int) -> Image.Image:
    if height <= 0 or width <= 0:
        return image
    if image.height < height or image.width < width:
        raise ValueError(f"Cannot crop {width}x{height} from image size {image.size}.")
    left = (image.width - width) // 2
    top = (image.height - height) // 2
    return image.crop((left, top, left + width, top + height))

