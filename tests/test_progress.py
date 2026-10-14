from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ebsd_diffusion.utils import RunProgressTracker, format_duration


class ProgressTrackerTests(unittest.TestCase):
    def test_format_duration(self) -> None:
        self.assertEqual(format_duration(0), "00:00:00")
        self.assertEqual(format_duration(3661), "01:01:01")
        self.assertEqual(format_duration(90061), "1d 01:01:01")
        self.assertEqual(format_duration(None), "--:--:--")

    def test_tracker_writes_latest_and_history(self) -> None:
        with TemporaryDirectory() as directory:
            tracker = RunProgressTracker(
                output_dir=directory,
                run_name="test run",
                total_units=4,
                unit_name="step",
            )
            tracker.set_stage("training")
            current = tracker.update(2, details={"loss": 0.5})
            final = tracker.finish(4)

            self.assertEqual(current["flow_stage"], "training")
            self.assertEqual(current["percent_complete"], 50.0)
            self.assertEqual(final["status"], "completed")
            self.assertEqual(final["estimated_remaining_seconds"], 0.0)

            latest_path = Path(directory) / "run_progress.json"
            history_path = Path(directory) / "run_progress_history.jsonl"
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            history = history_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(latest["status"], "completed")
            self.assertGreaterEqual(len(history), 4)


if __name__ == "__main__":
    unittest.main()

