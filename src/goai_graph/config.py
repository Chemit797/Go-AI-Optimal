"""Configuration for the isolated PPI graph experiment track."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GraphResourceConfig:
    sgd_features: Path
    string_physical_links: Path
    artifact: Path
    manifest: Path
    protein_mapping: Path
    min_score: int
    rewire_swaps_per_edge: int
    sgd_url: str
    string_url: str


@dataclass(frozen=True)
class GraphModelConfig:
    hidden_dim: int
    condition_hidden_dim: int
    graph_layers: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int
    target_scale_floor: float
    seed: int
    device: str


@dataclass(frozen=True)
class GraphExperimentConfig:
    path: Path
    baseline_config: Path
    graph: GraphResourceConfig
    model: GraphModelConfig
    runs_dir: Path


def _section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping")
    return value


def _resolve(directory: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (directory / path).resolve()


def load_graph_config(path: str | Path) -> GraphExperimentConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a mapping")
    graph = _section(payload, "graph")
    model = _section(payload, "model")
    runtime = _section(payload, "runtime")
    directory = config_path.parent

    hidden_dim = int(model["hidden_dim"])
    graph_layers = int(model["graph_layers"])
    dropout = float(model["dropout"])
    if hidden_dim <= 0 or graph_layers <= 0:
        raise ValueError("hidden_dim and graph_layers must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")

    return GraphExperimentConfig(
        path=config_path,
        baseline_config=_resolve(directory, payload["baseline_config"]),
        graph=GraphResourceConfig(
            sgd_features=_resolve(directory, graph["sgd_features"]),
            string_physical_links=_resolve(directory, graph["string_physical_links"]),
            artifact=_resolve(directory, graph["artifact"]),
            manifest=_resolve(directory, graph["manifest"]),
            protein_mapping=_resolve(directory, graph["protein_mapping"]),
            min_score=int(graph["min_score"]),
            rewire_swaps_per_edge=int(graph["rewire_swaps_per_edge"]),
            sgd_url=str(graph["sgd_url"]),
            string_url=str(graph["string_url"]),
        ),
        model=GraphModelConfig(
            hidden_dim=hidden_dim,
            condition_hidden_dim=int(model["condition_hidden_dim"]),
            graph_layers=graph_layers,
            dropout=dropout,
            learning_rate=float(model["learning_rate"]),
            weight_decay=float(model["weight_decay"]),
            epochs=int(model["epochs"]),
            batch_size=int(model["batch_size"]),
            target_scale_floor=float(model["target_scale_floor"]),
            seed=int(model["seed"]),
            device=str(model["device"]),
        ),
        runs_dir=_resolve(directory, runtime["runs_dir"]),
    )
