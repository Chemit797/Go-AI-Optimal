"""Configuration loading with paths resolved relative to the YAML file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    metadata_train_val: Path
    proteome_train_val: Path
    metadata_test: Path
    missing_rate_threshold: float


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int
    dropout: float
    learning_rate: float
    epochs: int
    seed: int
    device: str


@dataclass(frozen=True)
class FeatureConfig:
    chemical_hash_dim: int


@dataclass(frozen=True)
class RuntimeConfig:
    runs_dir: Path


@dataclass(frozen=True)
class BaselineConfig:
    path: Path
    data: DataConfig
    model: ModelConfig
    features: FeatureConfig
    runtime: RuntimeConfig


def _resolve(config_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_dir / path).resolve()


def _section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping")
    return value


def load_config(path: str | Path) -> BaselineConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a mapping")

    config_dir = config_path.parent
    data = _section(payload, "data")
    model = _section(payload, "model")
    features = _section(payload, "features")
    runtime = _section(payload, "runtime")
    threshold = float(data["missing_rate_threshold"])
    if not 0.0 < threshold <= 1.0:
        raise ValueError("data.missing_rate_threshold must be in (0, 1]")
    dropout = float(model["dropout"])
    if not 0.0 <= dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")

    return BaselineConfig(
        path=config_path,
        data=DataConfig(
            metadata_train_val=_resolve(config_dir, str(data["metadata_train_val"])),
            proteome_train_val=_resolve(config_dir, str(data["proteome_train_val"])),
            metadata_test=_resolve(config_dir, str(data["metadata_test"])),
            missing_rate_threshold=threshold,
        ),
        model=ModelConfig(
            hidden_dim=int(model["hidden_dim"]),
            dropout=dropout,
            learning_rate=float(model["learning_rate"]),
            epochs=int(model["epochs"]),
            seed=int(model["seed"]),
            device=str(model["device"]),
        ),
        features=FeatureConfig(chemical_hash_dim=int(features["chemical_hash_dim"])),
        runtime=RuntimeConfig(runs_dir=_resolve(config_dir, str(runtime["runs_dir"]))),
    )
