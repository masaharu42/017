from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path)
    config["_project_root"] = str(path.parent.parent.resolve())
    return config


def resolve_project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def with_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    result = deepcopy(config)
    for dotted_key, value in overrides.items():
        target = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return result


def flatten_config_for_logging(config: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), nested)
        elif isinstance(value, (list, tuple)):
            flattened[prefix] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flattened[prefix] = "" if value is None else value
        else:
            flattened[prefix] = str(value)

    visit("", config)
    return flattened
