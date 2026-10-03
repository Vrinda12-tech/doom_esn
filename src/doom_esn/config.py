"""Typed, strictly-validated experiment configuration (YAML in, frozen dataclasses out)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, cast, get_origin, get_type_hints

import yaml

from .reservoir import ReservoirConfig

T = TypeVar("T")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class EnvConfig:
    env_id: str = "VizdoomBasic-v1"
    frame_skip: int = 4
    calib_steps: int = 1500


@dataclass(frozen=True)
class EncoderConfig:
    height: int = 24
    width: int = 32


@dataclass(frozen=True)
class CalibrationConfig:
    enabled: bool = True
    spectral_radii: tuple[float, ...] = (0.8, 0.95, 1.1)
    input_scalings: tuple[float, ...] = (0.3, 1.0, 3.0)
    taus: tuple[float, ...] = (4.0, 8.0, 16.0)
    max_lyapunov: float = -0.02
    min_horizon: float = 2.0
    max_horizon: float = 60.0
    burn: int = 100


@dataclass(frozen=True)
class ESConfig:
    popsize: int = 16
    generations: int = 20
    episodes: int = 3
    sigma0: float = 0.05
    seed: int = 0
    eval_episodes: int = 30
    max_steps: int = 2000
    max_seconds: float | None = None
    track_health: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "basic"
    env: EnvConfig = field(default_factory=EnvConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    reservoir: ReservoirConfig = field(default_factory=ReservoirConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    es: ESConfig = field(default_factory=ESConfig)


def build(cls: type[T], data: dict[str, Any]) -> T:
    """Recursively build a dataclass from a dict; unknown keys are an error."""
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for key, val in data.items():
        t = hints[key]
        if is_dataclass(t) and isinstance(val, dict):
            kwargs[key] = build(cast(type[Any], t), val)
        elif get_origin(t) is tuple:
            kwargs[key] = tuple(val)
        else:
            kwargs[key] = val
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError("top-level YAML must be a mapping")
    return build(ExperimentConfig, data)
