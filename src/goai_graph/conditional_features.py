"""Condition features for the evolutionary, sample-specific graph model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from goai_baseline.schema import CHEMICAL, MEDIUM, STRAIN, TEMPERATURE, TIME

from .chemistry import ChemicalFeatures

BASE_FIELDS = (STRAIN, MEDIUM, TEMPERATURE)


def _categories(values: pd.Series) -> list[str]:
    return sorted(values.astype(str).drop_duplicates().tolist())


def _one_hot(values: pd.Series, categories: list[str]) -> np.ndarray:
    result = np.zeros((len(values), len(categories)), dtype=np.float32)
    lookup = {value: index for index, value in enumerate(categories)}
    for row, value in enumerate(values.astype(str)):
        index = lookup.get(value)
        if index is not None:
            result[row, index] = 1.0
    return result


@dataclass
class ConditionalFeatureBuilder:
    chemical_features: ChemicalFeatures
    categories: dict[str, list[str]] = field(default_factory=dict)
    chemical_lookup: dict[str, int] = field(default_factory=dict)
    structure_mean: np.ndarray | None = None
    structure_scale: np.ndarray | None = None
    max_train_time: float | None = None
    time_centers: np.ndarray | None = None
    rbf_width: float | None = None
    chemical_id_slice: tuple[int, int] | None = None

    def fit(self, metadata: pd.DataFrame, train_ids: pd.Index) -> "ConditionalFeatureBuilder":
        train = metadata.loc[train_ids]
        self.categories = {field: _categories(train[field]) for field in BASE_FIELDS}
        self.chemical_lookup = {name: index for index, name in enumerate(self.chemical_features.names)}
        structure = self._structure_block(train[CHEMICAL])
        self.structure_mean = structure.mean(axis=0)
        self.structure_scale = np.maximum(structure.std(axis=0), 1e-6)
        times = pd.to_numeric(train[TIME], errors="raise").to_numpy(dtype=np.float32)
        if np.any(times < 0):
            raise ValueError("Perturbation time must be non-negative")
        self.max_train_time = max(float(times.max()), 1.0)
        self.time_centers = np.unique(times / self.max_train_time).astype(np.float32)
        gaps = np.diff(self.time_centers)
        self.rbf_width = max(float(np.median(gaps)) if len(gaps) else 0.25, 0.05)
        start = sum(len(values) for values in self.categories.values())
        self.chemical_id_slice = (start, start + len(self.chemical_features.names))
        return self

    def _structure_block(self, values: pd.Series) -> np.ndarray:
        if self.chemical_features.matrix.ndim != 2:
            raise ValueError("chemical feature matrix must be two-dimensional")
        positions = values.astype(str).map(self.chemical_lookup)
        result = np.zeros((len(values), self.chemical_features.matrix.shape[1]), dtype=np.float32)
        valid = positions.notna().to_numpy()
        if valid.any():
            result[np.flatnonzero(valid)] = self.chemical_features.matrix[
                positions[valid].astype(int).to_numpy()
            ]
        return result

    def transform(self, metadata: pd.DataFrame) -> np.ndarray:
        if self.structure_mean is None or self.structure_scale is None or self.max_train_time is None:
            raise RuntimeError("ConditionalFeatureBuilder must be fit before transform")
        blocks = [_one_hot(metadata[field], self.categories[field]) for field in BASE_FIELDS]
        chemical_ids = _one_hot(metadata[CHEMICAL], self.chemical_features.names)
        structure = self._structure_block(metadata[CHEMICAL])
        structure = (structure - self.structure_mean[None, :]) / self.structure_scale[None, :]
        raw_time = pd.to_numeric(metadata[TIME], errors="raise").to_numpy(dtype=np.float32)
        scaled = raw_time / self.max_train_time
        log_scaled = np.log1p(raw_time) / np.log1p(self.max_train_time)
        centers = self.time_centers
        rbf = np.exp(-0.5 * ((scaled[:, None] - centers[None, :]) / self.rbf_width) ** 2)
        blocks.extend([chemical_ids, structure.astype(np.float32), scaled[:, None], log_scaled[:, None], rbf.astype(np.float32)])
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def fit_transform(self, metadata: pd.DataFrame, train_ids: pd.Index) -> np.ndarray:
        return self.fit(metadata, train_ids).transform(metadata.loc[train_ids])

    @property
    def output_dim(self) -> int:
        if self.chemical_id_slice is None:
            raise RuntimeError("builder has not been fit")
        return sum(len(values) for values in self.categories.values()) + len(self.chemical_features.names) + self.chemical_features.matrix.shape[1] + 2 + len(self.time_centers)

    def state_dict(self) -> dict[str, object]:
        return {
            "categories": self.categories,
            "chemical_names": self.chemical_features.names,
            "chemical_id_slice": self.chemical_id_slice,
            "structure_mean": self.structure_mean,
            "structure_scale": self.structure_scale,
            "max_train_time": self.max_train_time,
            "time_centers": self.time_centers,
            "rbf_width": self.rbf_width,
        }

    def summary(self) -> dict[str, object]:
        return {
            "base_fields": list(BASE_FIELDS),
            "chemical_feature_count": int(self.chemical_features.matrix.shape[1]),
            "chemical_id_slice": list(self.chemical_id_slice or (0, 0)),
            "output_dim": self.output_dim,
            "structure_encoding": "Morgan radius 2, 512 bits + 7 descriptors + resolved/control flags",
            "time_encoding": "scaled + log1p + train-time RBF",
            "uses_target_statistics": False,
        }


def build_node_signal(
    metadata: pd.DataFrame,
    proteins: list[str],
    target_table: pd.DataFrame,
    strain_table: pd.DataFrame | None = None,
    protein_mapping: pd.DataFrame | None = None,
) -> np.ndarray:
    """Return [samples, proteins, 3] signed condition-specific node inputs."""
    systematic_to_index = {name: index for index, name in enumerate(proteins)}
    if protein_mapping is not None:
        systematic_to_index = {
            str(row.systematic_name): int(row.protein_index)
            for row in protein_mapping.itertuples(index=False)
            if str(row.systematic_name) and int(row.protein_index) < len(proteins)
        }
    chemical_targets: dict[str, list[tuple[int, float]]] = {}
    for row in target_table.itertuples(index=False):
        index = systematic_to_index.get(str(row.systematic_name))
        if index is None:
            continue
        sign = -1.0 if str(row.action).lower() in {"inhibition", "degradation", "antagonism"} else 1.0
        chemical_targets.setdefault(str(row.chemical_raw_name), []).append((index, sign * float(row.weight)))

    strain_targets: dict[str, list[tuple[int, float]]] = {}
    if strain_table is not None and not strain_table.empty:
        for row in strain_table.itertuples(index=False):
            index = systematic_to_index.get(str(row.systematic_name))
            if index is not None:
                strain_targets.setdefault(str(row.strain_code), []).append((index, float(row.mutation_value)))

    signal = np.zeros((len(metadata), len(proteins), 3), dtype=np.float32)
    for row_index, row in enumerate(metadata.itertuples(index=False)):
        chemical = str(getattr(row, CHEMICAL))
        strain = str(getattr(row, STRAIN))
        for protein_index, value in chemical_targets.get(chemical, []):
            signal[row_index, protein_index, 0] += value
            signal[row_index, protein_index, 1] = max(signal[row_index, protein_index, 1], abs(value))
            signal[row_index, protein_index, 2] = 1.0
        for protein_index, value in strain_targets.get(strain, []):
            signal[row_index, protein_index, 0] += value
            signal[row_index, protein_index, 1] = max(signal[row_index, protein_index, 1], abs(value))
            signal[row_index, protein_index, 2] = 1.0
    return signal
