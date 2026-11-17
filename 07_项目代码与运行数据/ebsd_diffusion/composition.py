from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass
class CompositionConditionerState:
    candidate_features: list[str]
    active_features: list[str]
    dropped_features: dict[str, str]
    closure_drop_feature: str | None
    means: list[float]
    scales: list[float]
    raw_minimums: list[float]
    raw_maximums: list[float]
    normalized_training_conditions: list[list[float]]
    training_alloy_ids: list[int]
    min_nonzero_samples: int


class CompositionConditioner:
    def __init__(self, state: CompositionConditionerState):
        self.state = state

    @property
    def active_features(self) -> list[str]:
        return self.state.active_features

    @property
    def dimension(self) -> int:
        return len(self.state.active_features)

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        candidate_features: Iterable[str],
        alloy_ids: Iterable[int],
        closure_drop_feature: str | None = "Fe",
        min_nonzero_samples: int = 2,
        min_standard_deviation: float = 1.0e-8,
    ) -> "CompositionConditioner":
        ids = sorted({int(value) for value in alloy_ids})
        train = frame.loc[frame["alloy_id"].astype(int).isin(ids)].copy()
        if train.empty:
            raise ValueError("No composition rows match the requested training alloy IDs.")

        ready = train["conditional_ready"].map(_as_bool)
        train = train.loc[ready]
        if train.empty:
            raise ValueError("No condition-ready composition rows remain after filtering.")

        active: list[str] = []
        dropped: dict[str, str] = {}
        candidates = list(candidate_features)

        for feature in candidates:
            if feature not in train.columns:
                dropped[feature] = "missing_column"
                continue
            if closure_drop_feature and feature == closure_drop_feature:
                dropped[feature] = "closure_balance_removed"
                continue

            values = pd.to_numeric(train[feature], errors="coerce")
            if values.isna().any():
                dropped[feature] = "missing_numeric_values"
                continue
            if float(values.std(ddof=0)) <= min_standard_deviation:
                dropped[feature] = "constant_or_near_constant"
                continue

            nonzero_count = int((values.abs() > min_standard_deviation).sum())
            if 0 < nonzero_count < min_nonzero_samples:
                dropped[feature] = f"nonzero_in_only_{nonzero_count}_sample(s)"
                continue
            active.append(feature)

        if not active:
            raise ValueError("All composition features were removed; check the source table.")

        matrix = train[active].astype(float).to_numpy()
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        scales = np.where(scales <= min_standard_deviation, 1.0, scales)
        normalized = (matrix - means) / scales

        state = CompositionConditionerState(
            candidate_features=candidates,
            active_features=active,
            dropped_features=dropped,
            closure_drop_feature=closure_drop_feature,
            means=means.tolist(),
            scales=scales.tolist(),
            raw_minimums=matrix.min(axis=0).tolist(),
            raw_maximums=matrix.max(axis=0).tolist(),
            normalized_training_conditions=normalized.tolist(),
            training_alloy_ids=train["alloy_id"].astype(int).tolist(),
            min_nonzero_samples=min_nonzero_samples,
        )
        return cls(state)

    def transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.active_features if name not in frame.columns]
        if missing:
            raise ValueError(f"Missing composition features: {missing}")
        matrix = frame[self.active_features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if np.isnan(matrix).any():
            raise ValueError("Composition contains missing/non-numeric active features.")
        return self._normalize(matrix)

    def transform_mapping(self, composition: Mapping[str, float]) -> np.ndarray:
        try:
            raw = np.asarray([[float(composition[name]) for name in self.active_features]])
        except KeyError as exc:
            raise ValueError(f"Custom composition is missing active feature {exc.args[0]!r}.") from exc
        return self._normalize(raw)[0]

    def transform_by_alloy_id(self, frame: pd.DataFrame, alloy_id: int) -> np.ndarray:
        row = frame.loc[frame["alloy_id"].astype(int) == int(alloy_id)]
        if len(row) != 1:
            raise ValueError(f"Expected one composition row for alloy {alloy_id}, got {len(row)}.")
        if not _as_bool(row.iloc[0]["conditional_ready"]):
            raise ValueError(f"Alloy {alloy_id} has no exact condition-ready composition.")
        return self.transform_frame(row)[0]

    def nearest_training_distance(self, normalized_condition: np.ndarray) -> float:
        train = np.asarray(self.state.normalized_training_conditions, dtype=float)
        value = np.asarray(normalized_condition, dtype=float).reshape(1, -1)
        return float(np.linalg.norm(train - value, axis=1).min())

    def raw_range_violations(self, composition: Mapping[str, float]) -> dict[str, float]:
        violations: dict[str, float] = {}
        for index, feature in enumerate(self.active_features):
            value = float(composition[feature])
            low = self.state.raw_minimums[index]
            high = self.state.raw_maximums[index]
            if value < low:
                violations[feature] = value - low
            elif value > high:
                violations[feature] = value - high
        return violations

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self.state), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CompositionConditioner":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(CompositionConditionerState(**data))

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        means = np.asarray(self.state.means, dtype=float)
        scales = np.asarray(self.state.scales, dtype=float)
        return (matrix - means) / scales


def load_compositions(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"alloy_id", "conditional_ready"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Composition CSV is missing columns: {sorted(missing)}")
    frame["alloy_id"] = frame["alloy_id"].astype(int)
    if frame["alloy_id"].duplicated().any():
        duplicates = frame.loc[frame["alloy_id"].duplicated(), "alloy_id"].tolist()
        raise ValueError(f"Duplicate alloy IDs in composition CSV: {duplicates}")
    return frame


def parse_composition_argument(value: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Expected element=value, got {item!r}.")
        name, raw = item.split("=", 1)
        result[name.strip()] = float(raw.strip())
    return result


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

