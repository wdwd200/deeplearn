from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_yaml(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            from uav_relay_env.config import _parse_simple_yaml

            data = _parse_simple_yaml(text)
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a mapping")
        return data
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result.append(float(np.mean(values[start : index + 1])))
    return result


def sample_random_action(max_action: float, action_dim: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(100):
        action = rng.uniform(-max_action, max_action, size=action_dim).astype(np.float32)
        if float(np.linalg.norm(action)) <= max_action:
            return action
    return np.zeros(action_dim, dtype=np.float32)
