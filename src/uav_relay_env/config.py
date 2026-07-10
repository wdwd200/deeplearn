from __future__ import annotations

import ast
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

Vector3 = Tuple[float, float, float]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "comm_env_default.yaml"


def _as_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _as_vector3(value: Sequence[Any], name: str) -> Vector3:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must be a sequence of three numbers")
    return (
        _as_float(value[0], f"{name}[0]"),
        _as_float(value[1], f"{name}[1]"),
        _as_float(value[2], f"{name}[2]"),
    )


def _optional_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _merge_mappings(*mappings: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        merged.update(mapping)
    return merged


def _range_pair(value: Sequence[Any], name: str) -> Tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-number range")
    low = _as_float(value[0], f"{name}[0]")
    high = _as_float(value[1], f"{name}[1]")
    if low >= high:
        raise ValueError(f"{name} lower bound must be smaller than upper bound")
    return low, high


@dataclass(frozen=True)
class Bounds3D:
    x_min: float = 0.0
    x_max: float = 1000.0
    y_min: float = 0.0
    y_max: float = 1000.0
    z_min: float = 50.0
    z_max: float = 500.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "Bounds3D":
        if not data:
            return cls()
        x_min, x_max = _range_pair(data.get("x", [cls.x_min, cls.x_max]), "bounds_m.x")
        y_min, y_max = _range_pair(data.get("y", [cls.y_min, cls.y_max]), "bounds_m.y")
        z_min, z_max = _range_pair(data.get("z", [cls.z_min, cls.z_max]), "bounds_m.z")
        return cls(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max)

    def contains(self, position: Vector3) -> bool:
        x, y, z = position
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max and self.z_min <= z <= self.z_max

    def clamp(self, position: Vector3) -> Vector3:
        x, y, z = position
        return (
            min(max(x, self.x_min), self.x_max),
            min(max(y, self.y_min), self.y_max),
            min(max(z, self.z_min), self.z_max),
        )


@dataclass(frozen=True)
class ScenarioConfig:
    q_H_m: Vector3 = (500.0, 500.0, 1000.0)
    q_R_initial_m: Vector3 = (500.0, 500.0, 200.0)
    q_L_m: Vector3 = (800.0, 500.0, 100.0)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ScenarioConfig":
        if not data:
            return cls()
        default = cls()
        return cls(
            q_H_m=_as_vector3(data.get("q_H_m", default.q_H_m), "scenario.q_H_m"),
            q_R_initial_m=_as_vector3(
                data.get("q_R_initial_m", default.q_R_initial_m),
                "scenario.q_R_initial_m",
            ),
            q_L_m=_as_vector3(data.get("q_L_m", default.q_L_m), "scenario.q_L_m"),
        )


@dataclass(frozen=True)
class MobilityConfig:
    delta_t_s: float = 1.0
    max_speed_mps: float = 30.0
    max_steps: int = 100
    bounds_m: Bounds3D = field(default_factory=Bounds3D)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "MobilityConfig":
        if not data:
            return cls()
        default = cls()
        max_steps = int(data.get("max_steps", default.max_steps))
        return cls(
            delta_t_s=_as_float(data.get("delta_t_s", default.delta_t_s), "mobility.delta_t_s"),
            max_speed_mps=_as_float(data.get("max_speed_mps", default.max_speed_mps), "mobility.max_speed_mps"),
            max_steps=max_steps,
            bounds_m=Bounds3D.from_mapping(data.get("bounds_m")),
        )


@dataclass(frozen=True)
class ChannelConfig:
    beta0: float = 1e-3
    path_loss_exponent: float = 2.2
    antenna_model: str = "dipole"
    g_max: float = 1.0
    g_min: float = 0.05
    min_distance_m: float = 1.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ChannelConfig":
        if not data:
            return cls()
        default = cls()
        return cls(
            beta0=_as_float(data.get("beta0", default.beta0), "channel.beta0"),
            path_loss_exponent=_as_float(
                data.get("path_loss_exponent", default.path_loss_exponent),
                "channel.path_loss_exponent",
            ),
            antenna_model=str(data.get("antenna_model", default.antenna_model)),
            g_max=_as_float(data.get("g_max", default.g_max), "channel.g_max"),
            g_min=_as_float(data.get("g_min", default.g_min), "channel.g_min"),
            min_distance_m=_as_float(data.get("min_distance_m", default.min_distance_m), "channel.min_distance_m"),
        )


@dataclass(frozen=True)
class RateConfig:
    bandwidth_hz: float = 1_000_000.0
    noise_power_w: float = 1e-13
    power_HR_w: float = 1.0
    power_RL_w: float = 1.0
    half_duplex: bool = True
    r_min_bps: float = 1_000_000.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RateConfig":
        if not data:
            return cls()
        default = cls()
        return cls(
            bandwidth_hz=_as_float(data.get("bandwidth_hz", default.bandwidth_hz), "rate.bandwidth_hz"),
            noise_power_w=_as_float(data.get("noise_power_w", default.noise_power_w), "rate.noise_power_w"),
            power_HR_w=_as_float(data.get("power_HR_w", default.power_HR_w), "rate.power_HR_w"),
            power_RL_w=_as_float(data.get("power_RL_w", default.power_RL_w), "rate.power_RL_w"),
            half_duplex=bool(data.get("half_duplex", default.half_duplex)),
            r_min_bps=_as_float(data.get("r_min_bps", default.r_min_bps), "rate.r_min_bps"),
        )


@dataclass(frozen=True)
class RewardConfig:
    omega_R: float = 1e-6
    omega_E: float = 0.01
    omega_O: float = 1.0
    omega_B: float = 0.5
    omega_C: float = 1.0
    kappa: float = 0.001
    epsilon: float = 1e-9

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RewardConfig":
        if not data:
            return cls()
        default = cls()
        return cls(
            omega_R=_as_float(data.get("omega_R", default.omega_R), "reward.omega_R"),
            omega_E=_as_float(data.get("omega_E", default.omega_E), "reward.omega_E"),
            omega_O=_as_float(data.get("omega_O", default.omega_O), "reward.omega_O"),
            omega_B=_as_float(data.get("omega_B", default.omega_B), "reward.omega_B"),
            omega_C=_as_float(data.get("omega_C", default.omega_C), "reward.omega_C"),
            kappa=_as_float(data.get("kappa", default.kappa), "reward.kappa"),
            epsilon=_as_float(data.get("epsilon", default.epsilon), "reward.epsilon"),
        )


@dataclass(frozen=True)
class EnvConfig:
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    mobility: MobilityConfig = field(default_factory=MobilityConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    rate: RateConfig = field(default_factory=RateConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "EnvConfig":
        if not data:
            config = cls()
        else:
            scenario_data = (
                _optional_mapping(data, "scenario")
                if "scenario" in data
                else _optional_mapping(data, "uavs")
            )
            mobility_data = (
                _optional_mapping(data, "mobility")
                if "mobility" in data
                else _merge_mappings(
                    _optional_mapping(data, "simulation"),
                    _optional_mapping(data, "environment"),
                    {"bounds_m": _optional_mapping(data, "area").get("bounds_m", {})},
                )
            )
            if "channel" in data:
                channel_data = _optional_mapping(data, "channel")
            else:
                communication_data = _optional_mapping(data, "communication")
                antenna_data = _optional_mapping(data, "antenna")
                channel_data = _merge_mappings(
                    communication_data,
                    {
                        "antenna_model": antenna_data.get("model", antenna_data.get("antenna_model", "dipole")),
                        "g_max": antenna_data.get("g_max", ChannelConfig.g_max),
                        "g_min": antenna_data.get("g_min", ChannelConfig.g_min),
                    },
                )
            rate_data = (
                _optional_mapping(data, "rate")
                if "rate" in data
                else _optional_mapping(data, "communication")
            )
            config = cls(
                scenario=ScenarioConfig.from_mapping(scenario_data),
                mobility=MobilityConfig.from_mapping(mobility_data),
                channel=ChannelConfig.from_mapping(channel_data),
                rate=RateConfig.from_mapping(rate_data),
                reward=RewardConfig.from_mapping(_optional_mapping(data, "reward")),
            )
        config.validate()
        return config

    def validate(self) -> None:
        if self.mobility.delta_t_s <= 0.0:
            raise ValueError("mobility.delta_t_s must be positive")
        if self.mobility.max_speed_mps <= 0.0:
            raise ValueError("mobility.max_speed_mps must be positive")
        if self.mobility.max_steps <= 0:
            raise ValueError("mobility.max_steps must be positive")
        if self.channel.beta0 <= 0.0:
            raise ValueError("channel.beta0 must be positive")
        if self.channel.path_loss_exponent <= 0.0:
            raise ValueError("channel.path_loss_exponent must be positive")
        if self.channel.g_max < self.channel.g_min:
            raise ValueError("channel.g_max must be greater than or equal to channel.g_min")
        if self.channel.g_min < 0.0:
            raise ValueError("channel.g_min must be non-negative")
        if self.channel.min_distance_m <= 0.0:
            raise ValueError("channel.min_distance_m must be positive")
        if self.rate.bandwidth_hz <= 0.0:
            raise ValueError("rate.bandwidth_hz must be positive")
        if self.rate.noise_power_w <= 0.0:
            raise ValueError("rate.noise_power_w must be positive")
        if self.rate.power_HR_w < 0.0 or self.rate.power_RL_w < 0.0:
            raise ValueError("transmit powers must be non-negative")
        if self.rate.r_min_bps < 0.0:
            raise ValueError("rate.r_min_bps must be non-negative")
        if self.reward.epsilon <= 0.0:
            raise ValueError("reward.epsilon must be positive")


def _read_config_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = _parse_simple_yaml(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _parse_simple_yaml(text: str) -> Mapping[str, Any]:
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        without_comment = raw_line.split("#", 1)[0].rstrip()
        if not without_comment.strip():
            continue
        indent = len(without_comment) - len(without_comment.lstrip(" "))
        lines.append((indent, without_comment.strip()))

    if not lines:
        return {}
    parsed, index = _parse_yaml_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise ValueError("could not parse full YAML config")
    if not isinstance(parsed, Mapping):
        raise ValueError("YAML config must contain a mapping")
    return parsed


def _parse_yaml_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    current_indent, current_text = lines[index]
    if current_indent != indent:
        raise ValueError("inconsistent YAML indentation")
    if current_text.startswith("- "):
        return _parse_yaml_list(lines, index, indent)
    return _parse_yaml_mapping(lines, index, indent)


def _parse_yaml_mapping(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        line_indent, text = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError("unexpected nested YAML mapping")
        if text.startswith("- "):
            break
        key, separator, raw_value = text.partition(":")
        if not separator:
            raise ValueError(f"invalid YAML mapping entry: {text}")
        key = key.strip()
        raw_value = raw_value.strip()
        index += 1
        if raw_value:
            result[key] = _parse_yaml_scalar(raw_value)
        elif index < len(lines) and lines[index][0] > indent:
            result[key], index = _parse_yaml_block(lines, index, lines[index][0])
        else:
            result[key] = {}
    return result, index


def _parse_yaml_list(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[list[Any], int]:
    result: list[Any] = []
    while index < len(lines):
        line_indent, text = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError("unexpected nested YAML list")
        if not text.startswith("- "):
            break
        raw_value = text[2:].strip()
        index += 1
        if raw_value:
            result.append(_parse_yaml_scalar(raw_value))
        elif index < len(lines) and lines[index][0] > indent:
            value, index = _parse_yaml_block(lines, index, lines[index][0])
            result.append(value)
        else:
            result.append(None)
    return result, index


def _parse_yaml_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value)
    try:
        if any(marker in value for marker in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config(path: str | Path | None = None) -> EnvConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if config_path.exists():
        return EnvConfig.from_mapping(_read_config_mapping(config_path))
    if path is not None:
        raise FileNotFoundError(config_path)
    return EnvConfig()
