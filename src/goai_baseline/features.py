"""Training-only condition features matching the document baseline ladder."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .controls import exact_control_predictions
from .schema import CHEMICAL, MEDIUM, STRAIN, TEMPERATURE, TIME, treatment_mask


VALID_VARIANTS = ("p0_onehot", "p1_priors", "p2_crosses", "p3_time", "p4_hash", "p1_oof_priors")
BASIC_CATEGORIES = (STRAIN, CHEMICAL, MEDIUM, TEMPERATURE, TIME)
CROSS_CATEGORIES = ("strain_medium", "chemical_temperature")


def validate_variant(variant: str) -> None:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Unknown MLP variant '{variant}'. Expected one of {VALID_VARIANTS}")


def _variant_at_least(variant: str, marker: str) -> bool:
    base_variant = "p1_priors" if variant == "p1_oof_priors" else variant
    ladder = VALID_VARIANTS[:5]
    return ladder.index(base_variant) >= ladder.index(marker)


def _cross_values(metadata: pd.DataFrame, name: str) -> pd.Series:
    if name == "strain_medium":
        return metadata[STRAIN].astype(str) + "__" + metadata[MEDIUM].astype(str)
    if name == "chemical_temperature":
        return metadata[CHEMICAL].astype(str) + "__" + metadata[TEMPERATURE].astype(str)
    raise ValueError(f"Unknown cross feature '{name}'")


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


def _hash_matrix(values: pd.Series, dimension: int) -> np.ndarray:
    if dimension <= 0:
        raise ValueError("chemical_hash_dim must be positive")
    result = np.empty((len(values), dimension), dtype=np.float32)
    for row, value in enumerate(values.astype(str)):
        material = bytearray()
        counter = 0
        while len(material) < dimension:
            material.extend(hashlib.md5(f"{value}|{counter}".encode("utf-8")).digest())
            counter += 1
        result[row] = np.frombuffer(bytes(material[:dimension]), dtype=np.uint8) / 255.0
    return result


@dataclass
class FeatureBuilder:
    variant: str
    chemical_hash_dim: int = 32
    categories: dict[str, list[str]] = field(default_factory=dict)
    max_train_time: float | None = None
    strain_prior: pd.DataFrame | None = None
    chemical_delta_prior: pd.DataFrame | None = None
    global_mean: np.ndarray | None = None
    global_delta: np.ndarray | None = None

    def fit(self, metadata: pd.DataFrame, y_log2: pd.DataFrame, train_ids: pd.Index) -> "FeatureBuilder":
        validate_variant(self.variant)
        train_metadata = metadata.loc[train_ids]
        for name in BASIC_CATEGORIES:
            self.categories[name] = _categories(train_metadata[name])
        if _variant_at_least(self.variant, "p2_crosses"):
            for name in CROSS_CATEGORIES:
                self.categories[name] = _categories(_cross_values(train_metadata, name))
        if _variant_at_least(self.variant, "p3_time"):
            times = pd.to_numeric(train_metadata[TIME], errors="raise")
            self.max_train_time = float(times.max())
            if self.max_train_time <= 0:
                raise ValueError("Maximum training time must be positive")
        if _variant_at_least(self.variant, "p1_priors"):
            self._fit_priors(metadata, y_log2, train_ids)
        return self

    def _fit_priors(self, metadata: pd.DataFrame, y_log2: pd.DataFrame, train_ids: pd.Index) -> None:
        train_metadata = metadata.loc[train_ids]
        train_targets = y_log2.loc[train_ids]
        self.global_mean = train_targets.mean(axis=0).fillna(0.0).to_numpy(dtype=np.float32)
        self.strain_prior = train_targets.groupby(train_metadata[STRAIN].astype(str).to_numpy(), sort=True).mean()

        train_treatment_ids = train_ids[treatment_mask(train_metadata).to_numpy()]
        matched = exact_control_predictions(metadata, y_log2, train_treatment_ids, train_ids)
        valid_ids = train_treatment_ids[matched.has_exact_match.to_numpy()]
        if len(valid_ids) == 0:
            self.chemical_delta_prior = pd.DataFrame(columns=y_log2.columns, dtype=np.float32)
            self.global_delta = np.zeros(y_log2.shape[1], dtype=np.float32)
            return
        deltas = y_log2.loc[valid_ids] - matched.predictions.loc[valid_ids]
        self.global_delta = deltas.mean(axis=0).fillna(0.0).to_numpy(dtype=np.float32)
        self.chemical_delta_prior = deltas.groupby(
            metadata.loc[valid_ids, CHEMICAL].astype(str).to_numpy(), sort=True
        ).mean()

    @staticmethod
    def _lookup_prior(
        labels: pd.Series,
        table: pd.DataFrame,
        fallback: np.ndarray,
    ) -> np.ndarray:
        values = table.reindex(labels.astype(str).to_numpy()).to_numpy(dtype=np.float32)
        return np.where(np.isnan(values), fallback[None, :], values)

    def transform(self, metadata: pd.DataFrame) -> np.ndarray:
        if not self.categories:
            raise RuntimeError("FeatureBuilder must be fit before transform")
        blocks = [_one_hot(metadata[name], self.categories[name]) for name in BASIC_CATEGORIES]
        if _variant_at_least(self.variant, "p1_priors"):
            if any(value is None for value in (self.strain_prior, self.chemical_delta_prior, self.global_mean, self.global_delta)):
                raise RuntimeError("Statistical priors were not fit")
            blocks.extend(
                [
                    self._lookup_prior(metadata[STRAIN], self.strain_prior, self.global_mean),
                    self._lookup_prior(metadata[CHEMICAL], self.chemical_delta_prior, self.global_delta),
                ]
            )
        if _variant_at_least(self.variant, "p2_crosses"):
            blocks.extend(
                [_one_hot(_cross_values(metadata, name), self.categories[name]) for name in CROSS_CATEGORIES]
            )
        if _variant_at_least(self.variant, "p3_time"):
            if self.max_train_time is None:
                raise RuntimeError("Time encoding was not fit")
            theta = 2.0 * np.pi * pd.to_numeric(metadata[TIME], errors="raise").to_numpy(dtype=np.float32)
            theta = theta / self.max_train_time
            blocks.append(np.stack((np.sin(theta), np.cos(theta)), axis=1).astype(np.float32))
        if _variant_at_least(self.variant, "p4_hash"):
            blocks.append(_hash_matrix(metadata[CHEMICAL], self.chemical_hash_dim))
        return np.concatenate(blocks, axis=1, dtype=np.float32)

    def fit_transform(self, metadata: pd.DataFrame, y_log2: pd.DataFrame, train_ids: pd.Index) -> np.ndarray:
        self.fit(metadata, y_log2, train_ids)
        if self.variant == "p1_oof_priors":
            return self._transform_training_oof(metadata, y_log2, train_ids)
        return self.transform(metadata.loc[train_ids])

    def _transform_training_oof(self, metadata: pd.DataFrame, y_log2: pd.DataFrame, train_ids: pd.Index) -> np.ndarray:
        """Use leave-one-row-out target statistics only for training feature rows."""
        train_metadata = metadata.loc[train_ids]
        targets = y_log2.loc[train_ids]
        observed = targets.notna().astype(np.float32)
        global_sum = targets.sum(axis=0, skipna=True)
        global_count = targets.count(axis=0)
        global_fallback = pd.DataFrame(np.tile(self.global_mean, (len(train_ids), 1)), index=train_ids, columns=targets.columns)
        global_without_self = (global_sum - targets.fillna(0.0)).div(global_count - observed).where((global_count - observed) > 0, global_fallback)
        group_sum = targets.groupby(train_metadata[STRAIN].astype(str).to_numpy(), sort=False).transform("sum")
        group_count = targets.notna().groupby(train_metadata[STRAIN].astype(str).to_numpy(), sort=False).transform("sum")
        strain_oof = (group_sum - targets.fillna(0.0)).div(group_count - observed).where((group_count - observed) > 0, global_without_self)

        chemical_oof = pd.DataFrame(np.tile(self.global_delta, (len(train_ids), 1)), index=train_ids, columns=targets.columns)
        treatment_ids = train_ids[treatment_mask(train_metadata).to_numpy()]
        matched = exact_control_predictions(metadata, y_log2, treatment_ids, train_ids)
        valid_ids = treatment_ids[matched.has_exact_match.to_numpy()]
        if len(valid_ids):
            deltas = y_log2.loc[valid_ids] - matched.predictions.loc[valid_ids]
            delta_observed = deltas.notna().astype(np.float32)
            delta_sum = deltas.sum(axis=0, skipna=True)
            delta_count = deltas.count(axis=0)
            delta_fallback = pd.DataFrame(np.tile(self.global_delta, (len(valid_ids), 1)), index=valid_ids, columns=targets.columns)
            global_delta_oof = (delta_sum - deltas.fillna(0.0)).div(delta_count - delta_observed).where((delta_count - delta_observed) > 0, delta_fallback)
            chemical_sum = deltas.groupby(metadata.loc[valid_ids, CHEMICAL].astype(str).to_numpy(), sort=False).transform("sum")
            chemical_count = deltas.notna().groupby(metadata.loc[valid_ids, CHEMICAL].astype(str).to_numpy(), sort=False).transform("sum")
            chemical_oof.loc[valid_ids, :] = (
                (chemical_sum - deltas.fillna(0.0))
                .div(chemical_count - delta_observed)
                .where((chemical_count - delta_observed) > 0, global_delta_oof)
                .to_numpy(dtype=np.float32)
            )
        blocks = [_one_hot(train_metadata[name], self.categories[name]) for name in BASIC_CATEGORIES]
        blocks.extend([strain_oof.to_numpy(dtype=np.float32), chemical_oof.to_numpy(dtype=np.float32)])
        return np.concatenate(blocks, axis=1, dtype=np.float32)

    def state_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "chemical_hash_dim": self.chemical_hash_dim,
            "categories": self.categories,
            "max_train_time": self.max_train_time,
            "strain_prior": None if self.strain_prior is None else self.strain_prior.to_dict(orient="split"),
            "chemical_delta_prior": None if self.chemical_delta_prior is None else self.chemical_delta_prior.to_dict(orient="split"),
            "global_mean": self.global_mean,
            "global_delta": self.global_delta,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "FeatureBuilder":
        builder = cls(str(state["variant"]), int(state["chemical_hash_dim"]))
        builder.categories = {str(key): list(value) for key, value in dict(state["categories"]).items()}
        builder.max_train_time = state["max_train_time"] if state["max_train_time"] is None else float(state["max_train_time"])
        for name in ("strain_prior", "chemical_delta_prior"):
            payload = state[name]
            if payload is not None:
                frame = pd.DataFrame(payload["data"], columns=payload["columns"], index=payload["index"])
                setattr(builder, name, frame.astype(np.float32))
        for name in ("global_mean", "global_delta"):
            payload = state[name]
            if payload is not None:
                setattr(builder, name, np.asarray(payload, dtype=np.float32))
        return builder

    def summary(self) -> dict[str, object]:
        dimensions = {name: len(values) for name, values in self.categories.items()}
        prior_dimensions = 0 if self.global_mean is None else 2 * len(self.global_mean)
        return {
            "variant": self.variant,
            "category_dimensions": dimensions,
            "total_input_dim": int(
                sum(dimensions.values())
                + prior_dimensions
                + (2 if _variant_at_least(self.variant, "p3_time") else 0)
                + (self.chemical_hash_dim if _variant_at_least(self.variant, "p4_hash") else 0)
            ),
        }
