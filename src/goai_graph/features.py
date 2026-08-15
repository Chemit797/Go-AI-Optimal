"""Target-free condition features for graph ablation experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from goai_baseline.schema import CHEMICAL, MEDIUM, STRAIN, TEMPERATURE, TIME


CATEGORICAL_FIELDS = (STRAIN, CHEMICAL, MEDIUM, TEMPERATURE)


def _categories(values: pd.Series) -> list[str]:
    return sorted(values.astype(str).unique().tolist())


def _one_hot(values: pd.Series, categories: list[str]) -> np.ndarray:
    result = np.zeros((len(values), len(categories)), dtype=np.float32)
    lookup = {value: index for index, value in enumerate(categories)}
    positions = values.astype(str).map(lookup)
    valid = positions.notna().to_numpy()
    if valid.any():
        rows = np.flatnonzero(valid)
        result[rows, positions.iloc[rows].astype(int).to_numpy()] = 1.0
    return result


@dataclass
class ConditionFeatureBuilder:
    categories: dict[str, list[str]] = field(default_factory=dict)
    max_train_time: float | None = None
    time_centers: np.ndarray | None = None
    rbf_width: float | None = None

    def fit(self, metadata: pd.DataFrame, train_ids: pd.Index) -> "ConditionFeatureBuilder":
        train = metadata.loc[train_ids]
        self.categories = {field: _categories(train[field]) for field in CATEGORICAL_FIELDS}
        times = pd.to_numeric(train[TIME], errors="raise").to_numpy(dtype=np.float32)
        if np.any(times < 0):
            raise ValueError("Perturbation time must be non-negative")
        self.max_train_time = max(float(times.max()), 1.0)
        self.time_centers = np.unique(times / self.max_train_time).astype(np.float32)
        if len(self.time_centers) > 1:
            gaps = np.diff(self.time_centers)
            self.rbf_width = max(float(np.median(gaps)), 0.05)
        else:
            self.rbf_width = 0.25
        return self

    def transform(self, metadata: pd.DataFrame) -> np.ndarray:
        if not self.categories or self.max_train_time is None or self.time_centers is None or self.rbf_width is None:
            raise RuntimeError("ConditionFeatureBuilder must be fit before transform")
        blocks = [_one_hot(metadata[field], self.categories[field]) for field in CATEGORICAL_FIELDS]
        raw_time = pd.to_numeric(metadata[TIME], errors="raise").to_numpy(dtype=np.float32)
        scaled = raw_time / self.max_train_time
        log_scaled = np.log1p(raw_time) / np.log1p(self.max_train_time)
        rbf = np.exp(-0.5 * ((scaled[:, None] - self.time_centers[None, :]) / self.rbf_width) ** 2)
        blocks.extend([scaled[:, None], log_scaled[:, None], rbf.astype(np.float32)])
        return np.concatenate(blocks, axis=1, dtype=np.float32)

    def fit_transform(self, metadata: pd.DataFrame, train_ids: pd.Index) -> np.ndarray:
        return self.fit(metadata, train_ids).transform(metadata.loc[train_ids])

    @property
    def output_dim(self) -> int:
        if self.time_centers is None:
            raise RuntimeError("ConditionFeatureBuilder must be fit before reading output_dim")
        return sum(len(values) for values in self.categories.values()) + 2 + len(self.time_centers)

    def state_dict(self) -> dict[str, object]:
        return {
            "categories": self.categories,
            "max_train_time": self.max_train_time,
            "time_centers": self.time_centers,
            "rbf_width": self.rbf_width,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ConditionFeatureBuilder":
        builder = cls()
        builder.categories = {str(key): list(value) for key, value in dict(state["categories"]).items()}
        builder.max_train_time = float(state["max_train_time"])
        builder.time_centers = np.asarray(state["time_centers"], dtype=np.float32)
        builder.rbf_width = float(state["rbf_width"])
        return builder

    def summary(self) -> dict[str, object]:
        return {
            "categorical_fields": list(CATEGORICAL_FIELDS),
            "category_dimensions": {field: len(values) for field, values in self.categories.items()},
            "time_encoding": "scaled + log1p + train-time RBF",
            "time_centers": [] if self.time_centers is None else self.time_centers.tolist(),
            "output_dim": self.output_dim,
            "uses_target_statistics": False,
        }
