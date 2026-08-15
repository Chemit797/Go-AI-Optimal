"""Leakage-safe entity-level OOF validation for the response MVP."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from goai_baseline.audit import audit_inputs, sha256
from goai_baseline.metrics import evaluate_predictions
from goai_baseline.official_metrics import evaluate_prediction_set
from goai_baseline.preprocess import PreprocessedData, prepare_data
from goai_baseline.schema import (
    CHEMICAL,
    MEDIUM,
    PLATE,
    SAMPLE_ID,
    STRAIN,
    TEMPERATURE,
    TIME,
    TIME_UNIT,
    control_mask,
    treatment_mask,
)

from .config import ResponseConfig, load_response_config
from .entities import (
    build_support_manifest,
    load_json_with_hash,
    load_registry,
    manifest_sha256,
    normalize_entity_key,
    write_json_with_hash,
)
from .nested_scale import (
    ALLOWED_SCALES as NESTED_ALLOWED_SCALES,
    METRICS as NESTED_SCALE_METRICS,
    PROTOCOL as NESTED_SCALE_PROTOCOL,
    active_expert_axes,
    compose_scaled_prediction,
    ids_sha256,
    predict_fit_components,
    scale_grid,
    select_nested_scales,
    validate_receipt as validate_nested_scale_receipt,
    write_receipt as write_nested_scale_receipt,
)
from .train import _predict, _predict_core_components, fit_response_model


SCENARIOS = ("S1", "S2", "S3", "time", "time_forward")
# Rxy denotes whether the validation strain (x) and chemical (y) remain
# represented in the corresponding fold's fit support.  RT is the
# condition-time analogue: both entities and their pair remain represented,
# while the exact condition-time group is withheld.
REGIME_SCENARIOS = ("R00", "R10", "R01", "R11", "RT")
DIAGNOSTIC_SCENARIOS = ("plate",)
ALL_SCENARIOS = SCENARIOS + REGIME_SCENARIOS + DIAGNOSTIC_SCENARIOS
TIME_GROUP_FIELDS = (STRAIN, CHEMICAL, MEDIUM, TEMPERATURE, TIME, TIME_UNIT)
TIME_CONTEXT_FIELDS = (STRAIN, CHEMICAL, MEDIUM, TEMPERATURE, TIME_UNIT)
PAIR_FIELDS = (STRAIN, CHEMICAL)

# Producer behaviour depends on these local source trees plus the matrix
# launcher that materialises each effective config.  Artifact builders are not
# included here because their outputs are independently content-hashed in the
# run contract.  Keep paths project-relative so the same checkout hashes
# identically when mounted at a different absolute location.
SOURCE_FINGERPRINT_DIRS = (
    Path("src/goai_response"),
    Path("src/goai_baseline"),
    Path("src/goai_graph"),
)
SOURCE_FINGERPRINT_FILES = (Path("scripts/nightly/run_matrix.py"),)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FoldSlice:
    scenario: str
    fold: int
    train_ids: pd.Index
    validation_ids: pd.Index
    heldout_strains: tuple[str, ...] = ()
    heldout_chemicals: tuple[str, ...] = ()
    heldout_time_groups: tuple[str, ...] = ()
    heldout_plates: tuple[str, ...] = ()


def _balanced_group_folds(groups: pd.Series, n_folds: int, seed: int) -> dict[str, int]:
    """Assign whole groups to deterministic folds, greedily balancing rows."""
    counts = groups.astype(str).value_counts(sort=False)
    if len(counts) < n_folds:
        raise ValueError(f"Need at least {n_folds} groups, found {len(counts)}")
    rng = np.random.default_rng(seed)
    tie_order = {name: rank for rank, name in enumerate(rng.permutation(sorted(counts.index)))}
    ordered = sorted(counts.index, key=lambda name: (-int(counts[name]), tie_order[name], name))
    loads = np.zeros(n_folds, dtype=np.int64)
    mapping: dict[str, int] = {}
    for name in ordered:
        candidates = np.flatnonzero(loads == loads.min())
        fold = int(candidates[int(rng.integers(len(candidates)))])
        mapping[name] = fold
        loads[fold] += int(counts[name])
    return mapping


def _balanced_nested_group_folds(
    groups: pd.Series,
    contexts: pd.Series,
    n_folds: int,
    seed: int,
) -> dict[str, int]:
    """Balance exact groups while spreading each context across folds.

    RT needs at least one alternate condition-time group for the same context
    to remain in fit support.  A global greedy assignment can accidentally put
    every time of one context in the same fold; this context-aware assignment
    prevents that whenever a context has two or more groups.
    """

    frame = pd.DataFrame({"group": groups.astype(str), "context": contexts.astype(str)})
    counts = frame.groupby(["context", "group"], sort=False).size()
    rng = np.random.default_rng(seed)
    loads = np.zeros(n_folds, dtype=np.int64)
    mapping: dict[str, int] = {}
    for context in sorted(frame["context"].unique()):
        context_counts = counts.loc[context]
        names = list(context_counts.index.astype(str))
        if len(names) < 2:
            continue
        tie_order = {name: rank for rank, name in enumerate(rng.permutation(sorted(names)))}
        ordered = sorted(names, key=lambda name: (-int(context_counts[name]), tie_order[name], name))
        available: set[int] = set(range(n_folds))
        for name in ordered:
            if not available:
                available = set(range(n_folds))
            candidate_load = min(loads[fold] for fold in available)
            choices = sorted(fold for fold in available if loads[fold] == candidate_load)
            fold = choices[int(rng.integers(len(choices)))]
            mapping[name] = fold
            loads[fold] += int(context_counts[name])
            available.remove(fold)
    return mapping


def _time_group_keys(metadata: pd.DataFrame) -> pd.Series:
    frame = metadata.loc[:, TIME_GROUP_FIELDS].astype(str).copy()
    frame[STRAIN] = frame[STRAIN].map(normalize_entity_key)
    frame[CHEMICAL] = frame[CHEMICAL].map(normalize_entity_key)
    return frame.agg("\x1f".join, axis=1)


def _time_context_keys(metadata: pd.DataFrame) -> pd.Series:
    frame = metadata.loc[:, TIME_CONTEXT_FIELDS].astype(str).copy()
    frame[STRAIN] = frame[STRAIN].map(normalize_entity_key)
    frame[CHEMICAL] = frame[CHEMICAL].map(normalize_entity_key)
    return frame.agg("\x1f".join, axis=1)


def _pair_keys(metadata: pd.DataFrame) -> pd.Series:
    frame = metadata.loc[:, PAIR_FIELDS].astype(str).copy()
    frame[STRAIN] = frame[STRAIN].map(normalize_entity_key)
    frame[CHEMICAL] = frame[CHEMICAL].map(normalize_entity_key)
    return frame.agg("\x1f".join, axis=1)


def _entity_keys(values: pd.Series) -> pd.Series:
    return values.map(normalize_entity_key)


def _fold_identity_keys(
    metadata: pd.DataFrame,
    config: ResponseConfig | None,
) -> tuple[pd.Series, pd.Series]:
    """Resolve the grouping identity used by both expert support and OOF.

    Registry aliases that resolve to the same canonical support key must never
    be split across train and validation under different raw labels.  When no
    authoritative registries are configured this retains the historical,
    conservative normalized-raw behaviour for legacy configs/checkpoints.
    """

    if config is None or (
        config.entity.chemical_registry is None
        and config.entity.strain_registry is None
    ):
        return _entity_keys(metadata[STRAIN]), _entity_keys(metadata[CHEMICAL])
    if (
        config.entity.chemical_registry is None
        or config.entity.strain_registry is None
    ):
        raise ValueError(
            "OOF canonical identity requires both chemical and strain registries"
        )
    manifest = build_support_manifest(
        metadata,
        {
            "chemical": load_registry(config.entity.chemical_registry, "chemical"),
            "strain": load_registry(config.entity.strain_registry, "strain"),
        },
    )
    from .entities import support_flags

    flags = support_flags(metadata, manifest)
    return (
        flags["strain_support_key"].astype(str),
        flags["chemical_support_key"].astype(str),
    )


def _identity_pair_keys(
    metadata: pd.DataFrame,
    strain_keys: pd.Series,
    chemical_keys: pd.Series,
) -> pd.Series:
    if not strain_keys.index.equals(metadata.index) or not chemical_keys.index.equals(
        metadata.index
    ):
        raise ValueError("Canonical entity keys do not align with metadata")
    return strain_keys.astype(str) + "\x1f" + chemical_keys.astype(str)


def _identity_time_keys(
    metadata: pd.DataFrame,
    strain_keys: pd.Series,
    chemical_keys: pd.Series,
    *,
    include_time: bool,
) -> pd.Series:
    fields = [MEDIUM, TEMPERATURE, *([TIME] if include_time else []), TIME_UNIT]
    frame = metadata.loc[:, fields].astype(str).copy()
    frame.insert(0, CHEMICAL, chemical_keys.astype(str).to_numpy())
    frame.insert(0, STRAIN, strain_keys.astype(str).to_numpy())
    return frame.agg("\x1f".join, axis=1)


def make_fold_slices(
    metadata: pd.DataFrame,
    train_ids: pd.Index,
    n_folds: int = 4,
    seed: int = 42,
    scenarios: tuple[str, ...] = SCENARIOS,
    config: ResponseConfig | None = None,
) -> tuple[list[FoldSlice], pd.DataFrame]:
    """Build persistent folds that emulate the four frozen outer scenarios."""
    unknown = sorted(set(scenarios) - set(ALL_SCENARIOS))
    if unknown:
        raise ValueError(f"Unknown OOF scenarios: {unknown}")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    ids = pd.Index(train_ids)
    train = metadata.loc[ids]
    treatment_ids = ids[treatment_mask(train).to_numpy()]
    treatment = metadata.loc[treatment_ids]
    all_strain_keys, all_chemical_keys = _fold_identity_keys(metadata, config)
    train_strain_keys = all_strain_keys.loc[ids]
    train_chemical_keys = all_chemical_keys.loc[ids]
    treatment_strain_keys = all_strain_keys.loc[treatment_ids]
    treatment_chemical_keys = all_chemical_keys.loc[treatment_ids]
    entity_regimes = {"R00", "R10", "R01", "R11"}
    chemical_fold = (
        _balanced_group_folds(treatment_chemical_keys, n_folds, seed + 101)
        if ({"S1", "S3"} | entity_regimes) & set(scenarios)
        else {}
    )
    strain_fold = (
        _balanced_group_folds(treatment_strain_keys, n_folds, seed + 211)
        if ({"S2", "S3"} | entity_regimes) & set(scenarios)
        else {}
    )
    all_time_keys = _identity_time_keys(
        metadata, all_strain_keys, all_chemical_keys, include_time=True
    )
    all_context_keys = _identity_time_keys(
        metadata, all_strain_keys, all_chemical_keys, include_time=False
    )
    all_pair_keys = _identity_pair_keys(metadata, all_strain_keys, all_chemical_keys)
    time_keys = all_time_keys.loc[treatment_ids]
    time_fold = (
        _balanced_group_folds(time_keys, n_folds, seed + 307)
        if "time" in scenarios
        else {}
    )
    rt_fold = (
        _balanced_nested_group_folds(
            time_keys,
            all_context_keys.loc[treatment_ids],
            n_folds,
            seed + 353,
        )
        if "RT" in scenarios
        else {}
    )
    forward_context = all_context_keys.loc[treatment_ids]
    forward_time = pd.to_numeric(treatment[TIME], errors="coerce")
    if "time_forward" in scenarios and forward_time.isna().any():
        raise ValueError("time_forward requires numeric perturbation times")
    forward_counts = forward_time.groupby(forward_context).nunique()
    forward_contexts = forward_counts.index[forward_counts >= 2]
    forward_max = forward_time.groupby(forward_context).max()
    forward_candidates = treatment_ids[forward_context.isin(forward_contexts).to_numpy()]
    forward_fold = (
        _balanced_group_folds(forward_context.loc[forward_candidates], n_folds, seed + 401)
        if "time_forward" in scenarios
        else {}
    )
    plate_fold = _balanced_group_folds(train[PLATE], n_folds, seed + 503) if "plate" in scenarios else {}

    slices: list[FoldSlice] = []
    assignments: list[dict[str, object]] = []
    for scenario in scenarios:
        fold_specs: list[tuple[int, int, int]]
        if scenario in {"S3", "R00", "R11"}:
            fold_specs = [
                (strain_part * n_folds + chemical_part, strain_part, chemical_part)
                for strain_part in range(n_folds)
                for chemical_part in range(n_folds)
            ]
        elif scenario == "R10":
            # The fit set depends only on the held-out chemical partition.  A
            # strain cross-product would retrain the same support n_folds times
            # merely to split its predictions; score all represented strains in
            # the corresponding chemical fold with one fitted model instead.
            fold_specs = [
                (chemical_part, -1, chemical_part)
                for chemical_part in range(n_folds)
            ]
        elif scenario == "R01":
            # Symmetric to R10: one fit per held-out strain partition predicts
            # every validation chemical that remains represented in that fit.
            fold_specs = [
                (strain_part, strain_part, -1)
                for strain_part in range(n_folds)
            ]
        else:
            fold_specs = [(fold, fold, fold) for fold in range(n_folds)]
        for fold, strain_part, chemical_part in fold_specs:
            heldout_strains: tuple[str, ...] = ()
            heldout_chemicals: tuple[str, ...] = ()
            heldout_time_groups: tuple[str, ...] = ()
            heldout_plates: tuple[str, ...] = ()
            if scenario == "S1":
                heldout_chemicals = tuple(sorted(name for name, value in chemical_fold.items() if value == fold))
                train_mask = ~train_chemical_keys.isin(heldout_chemicals)
                fold_train = ids[train_mask.to_numpy()]
                candidates = treatment_ids[treatment_chemical_keys.isin(heldout_chemicals).to_numpy()]
                represented = train_strain_keys.loc[fold_train].unique()
                fold_valid = candidates[
                    all_strain_keys.loc[candidates].isin(represented).to_numpy()
                ]
            elif scenario == "S2":
                heldout_strains = tuple(sorted(name for name, value in strain_fold.items() if value == fold))
                train_mask = ~train_strain_keys.isin(heldout_strains)
                fold_train = ids[train_mask.to_numpy()]
                candidates = treatment_ids[treatment_strain_keys.isin(heldout_strains).to_numpy()]
                represented = train_chemical_keys.loc[fold_train].unique()
                fold_valid = candidates[
                    all_chemical_keys.loc[candidates].isin(represented).to_numpy()
                ]
            elif scenario == "S3":
                heldout_strains = tuple(sorted(name for name, value in strain_fold.items() if value == strain_part))
                heldout_chemicals = tuple(sorted(name for name, value in chemical_fold.items() if value == chemical_part))
                train_mask = ~(
                    train_strain_keys.isin(heldout_strains)
                    | train_chemical_keys.isin(heldout_chemicals)
                )
                fold_train = ids[train_mask.to_numpy()]
                valid_mask = treatment_strain_keys.isin(heldout_strains) & treatment_chemical_keys.isin(heldout_chemicals)
                candidates = treatment_ids[valid_mask.to_numpy()]
                fold_valid = candidates
            elif scenario in entity_regimes:
                if scenario != "R10":
                    heldout_strains = tuple(
                        sorted(
                            name
                            for name, value in strain_fold.items()
                            if value == strain_part
                        )
                    )
                if scenario != "R01":
                    heldout_chemicals = tuple(
                        sorted(
                            name
                            for name, value in chemical_fold.items()
                            if value == chemical_part
                        )
                    )
                strain_match = train_strain_keys.isin(heldout_strains)
                chemical_match = train_chemical_keys.isin(heldout_chemicals)
                if scenario == "R00":
                    valid_mask = (
                        treatment_strain_keys.isin(heldout_strains)
                        & treatment_chemical_keys.isin(heldout_chemicals)
                    )
                    train_mask = ~(strain_match | chemical_match)
                elif scenario == "R10":
                    valid_mask = treatment_chemical_keys.isin(heldout_chemicals)
                    train_mask = ~chemical_match
                elif scenario == "R01":
                    valid_mask = treatment_strain_keys.isin(heldout_strains)
                    train_mask = ~strain_match
                else:  # R11: retain each entity, withhold only their pair block.
                    valid_mask = (
                        treatment_strain_keys.isin(heldout_strains)
                        & treatment_chemical_keys.isin(heldout_chemicals)
                    )
                    train_mask = ~(strain_match & chemical_match)
                candidates = treatment_ids[valid_mask.to_numpy()]
                fold_train = ids[train_mask.to_numpy()]
                remaining = metadata.loc[fold_train]
                candidate_metadata = metadata.loc[candidates]
                if scenario == "R10":
                    represented = all_strain_keys.loc[candidates].isin(
                        all_strain_keys.loc[fold_train].unique()
                    )
                elif scenario == "R01":
                    represented = all_chemical_keys.loc[candidates].isin(
                        all_chemical_keys.loc[fold_train].unique()
                    )
                elif scenario == "R11":
                    represented = (
                        all_strain_keys.loc[candidates].isin(
                            all_strain_keys.loc[fold_train].unique()
                        )
                        & all_chemical_keys.loc[candidates].isin(
                            all_chemical_keys.loc[fold_train].unique()
                        )
                    )
                else:
                    represented = pd.Series(True, index=candidates)
                fold_valid = candidates[represented.to_numpy()]
            elif scenario in {"time", "RT"}:
                current_time_fold = rt_fold if scenario == "RT" else time_fold
                heldout_time_groups = tuple(sorted(name for name, value in current_time_fold.items() if value == fold))
                all_keys = all_time_keys.loc[ids]
                valid_mask = time_keys.isin(heldout_time_groups)
                candidates = treatment_ids[valid_mask.to_numpy()]
                train_mask = ~all_keys.isin(heldout_time_groups)
                fold_train = ids[train_mask.to_numpy()]
                remaining = metadata.loc[fold_train]
                represented = (
                    all_context_keys.loc[candidates].isin(
                        all_context_keys.loc[fold_train].unique()
                    )
                    & metadata.loc[candidates, TIME].astype(str).isin(remaining[TIME].astype(str).unique())
                )
                if scenario == "RT":
                    represented &= all_pair_keys.loc[candidates].isin(
                        all_pair_keys.loc[fold_train].unique()
                    )
                fold_valid = candidates[represented.to_numpy()]
            elif scenario == "time_forward":
                heldout_contexts = tuple(
                    sorted(name for name, value in forward_fold.items() if value == fold)
                )
                context_mask = forward_context.isin(heldout_contexts)
                candidates = treatment_ids[context_mask.to_numpy()]
                candidate_times = forward_time.loc[candidates]
                candidate_contexts = forward_context.loc[candidates]
                is_latest = np.isclose(
                    candidate_times.to_numpy(dtype=float),
                    candidate_contexts.map(forward_max).to_numpy(dtype=float),
                )
                fold_valid = candidates[is_latest]
                fold_train = ids.difference(fold_valid)
                heldout_time_groups = tuple(
                    sorted(all_time_keys.loc[fold_valid].unique().tolist())
                )
            else:
                heldout_plates = tuple(sorted(name for name, value in plate_fold.items() if value == fold))
                plate_mask = train[PLATE].astype(str).isin(heldout_plates)
                candidates = ids[plate_mask.to_numpy()]
                fold_valid = candidates
                fold_train = ids[~plate_mask.to_numpy()]

            if len(fold_train) == 0:
                raise ValueError(f"{scenario} fold {fold} has no training rows")
            fold_slice = FoldSlice(
                scenario=scenario,
                fold=fold,
                train_ids=fold_train,
                validation_ids=fold_valid,
                heldout_strains=heldout_strains,
                heldout_chemicals=heldout_chemicals,
                heldout_time_groups=heldout_time_groups,
                heldout_plates=heldout_plates,
            )
            assert_fold_isolation(
                metadata,
                fold_slice,
                config,
                (all_strain_keys, all_chemical_keys),
            )
            slices.append(fold_slice)
            eligible_set = set(fold_valid)
            supported_strains = set(all_strain_keys.loc[fold_train])
            supported_chemicals = set(all_chemical_keys.loc[fold_train])
            supported_pairs = set(all_pair_keys.loc[fold_train])
            supported_time_groups = set(all_time_keys.loc[fold_train])
            for sample_id in candidates:
                row = metadata.loc[sample_id]
                row_pair = all_pair_keys.loc[sample_id]
                row_time_group = all_time_keys.loc[sample_id]
                assignments.append(
                    {
                        SAMPLE_ID: sample_id,
                        "scenario": scenario,
                        "fold": fold,
                        STRAIN: row[STRAIN],
                        CHEMICAL: row[CHEMICAL],
                        TIME: row[TIME],
                        "time_group": row_time_group,
                        "strain_support_key": all_strain_keys.loc[sample_id],
                        "chemical_support_key": all_chemical_keys.loc[sample_id],
                        "strain_seen_in_fold": all_strain_keys.loc[sample_id] in supported_strains,
                        "chemical_seen_in_fold": all_chemical_keys.loc[sample_id] in supported_chemicals,
                        "pair_seen_in_fold": row_pair in supported_pairs,
                        "time_group_seen_in_fold": row_time_group in supported_time_groups,
                        "eligible": sample_id in eligible_set,
                        "exclusion_reason": ""
                        if sample_id in eligible_set
                        else (
                            "not_context_max_time"
                            if scenario == "time_forward"
                            else "would_mix_additional_OOD_axis"
                        ),
                    }
                )
    frame = pd.DataFrame(assignments)
    if not frame.empty and frame.duplicated([SAMPLE_ID, "scenario"]).any():
        raise AssertionError("A sample was assigned to multiple folds within a scenario")
    return slices, frame


def assert_fold_isolation(
    metadata: pd.DataFrame,
    fold: FoldSlice,
    config: ResponseConfig | None = None,
    identity_keys: tuple[pd.Series, pd.Series] | None = None,
) -> None:
    """Fail fast if a validation entity or condition leaks into fold training."""
    train = metadata.loc[fold.train_ids]
    valid = metadata.loc[fold.validation_ids]
    if not train.index.intersection(valid.index).empty:
        raise AssertionError("Fold train and validation IDs overlap")
    all_strains, all_chemicals = (
        identity_keys
        if identity_keys is not None
        else _fold_identity_keys(metadata, config)
    )
    all_pairs = _identity_pair_keys(metadata, all_strains, all_chemicals)
    all_times = _identity_time_keys(
        metadata, all_strains, all_chemicals, include_time=True
    )
    all_contexts = _identity_time_keys(
        metadata, all_strains, all_chemicals, include_time=False
    )
    train_strains = set(all_strains.loc[fold.train_ids])
    valid_strains = set(all_strains.loc[fold.validation_ids])
    train_chemicals = set(all_chemicals.loc[fold.train_ids])
    valid_chemicals = set(all_chemicals.loc[fold.validation_ids])
    train_pairs = set(all_pairs.loc[fold.train_ids])
    valid_pairs = set(all_pairs.loc[fold.validation_ids])
    if fold.scenario in {"S1", "S3"} and valid_chemicals & train_chemicals:
        raise AssertionError(f"{fold.scenario} validation chemical is present in fold training")
    if fold.scenario in {"S2", "S3"} and valid_strains & train_strains:
        raise AssertionError(f"{fold.scenario} validation strain is present in fold training")
    if fold.scenario == "R00":
        if valid_strains & train_strains or valid_chemicals & train_chemicals:
            raise AssertionError("R00 validation entities must both be absent from fold training")
    elif fold.scenario == "R10":
        if not valid_strains <= train_strains or valid_chemicals & train_chemicals:
            raise AssertionError("R10 requires seen strains and unseen chemicals")
    elif fold.scenario == "R01":
        if valid_strains & train_strains or not valid_chemicals <= train_chemicals:
            raise AssertionError("R01 requires unseen strains and seen chemicals")
    elif fold.scenario == "R11":
        if not valid_strains <= train_strains or not valid_chemicals <= train_chemicals:
            raise AssertionError("R11 requires both validation entities in fold training")
        if valid_pairs & train_pairs:
            raise AssertionError("R11 validation pairs must be absent from fold training")
    if fold.scenario in {"time", "RT"} and set(
        all_times.loc[fold.validation_ids]
    ) & set(all_times.loc[fold.train_ids]):
        raise AssertionError(f"{fold.scenario} validation condition-time group is present in fold training")
    if fold.scenario == "RT":
        if not valid_strains <= train_strains or not valid_chemicals <= train_chemicals:
            raise AssertionError("RT requires both validation entities in fold training")
        if not valid_pairs <= train_pairs:
            raise AssertionError("RT requires validation strain-chemical pairs in fold training")
    if fold.scenario == "time_forward":
        valid_context = all_contexts.loc[fold.validation_ids]
        train_context = all_contexts.loc[fold.train_ids]
        for context in valid_context.unique():
            valid_times = pd.to_numeric(valid.loc[valid_context.eq(context), TIME])
            train_times = pd.to_numeric(train.loc[train_context.eq(context), TIME])
            if train_times.empty or not float(train_times.max()) < float(valid_times.min()):
                raise AssertionError(
                    "time_forward validation must be later than every training row in its context"
                )
    if fold.scenario == "plate" and set(valid[PLATE].astype(str)) & set(train[PLATE].astype(str)):
        raise AssertionError("plate validation group is present in fold training")


def _json_fold(fold: FoldSlice) -> dict[str, object]:
    def digest(ids: pd.Index) -> str:
        value = "\n".join(ids.astype(str)).encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    return {
        "scenario": fold.scenario,
        "fold": fold.fold,
        "train_count": int(len(fold.train_ids)),
        "validation_count": int(len(fold.validation_ids)),
        "train_ids_sha256": digest(fold.train_ids),
        "validation_ids_sha256": digest(fold.validation_ids),
        "heldout_strains": list(fold.heldout_strains),
        "heldout_chemicals": list(fold.heldout_chemicals),
        "heldout_time_group_count": len(fold.heldout_time_groups),
        "heldout_plates": list(fold.heldout_plates),
    }


def _normalise_contract(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalise_contract(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_contract(item) for item in value]
    return value


def _optional_artifact(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    return {"path": str(path), "sha256": sha256(path)}


def _source_fingerprint(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Hash every producer source file using stable relative paths and bytes.

    The per-file manifest makes a resume rejection inspectable while the
    aggregate SHA-256 is compact enough to embed in every producer contract.
    Only regular ``.py`` files are admitted from package trees; explicit
    launcher files are included regardless of extension.
    """

    root = project_root.resolve()
    paths: set[Path] = set()
    for relative_dir in SOURCE_FINGERPRINT_DIRS:
        directory = root / relative_dir
        if not directory.is_dir():
            raise FileNotFoundError(f"Source fingerprint directory is missing: {directory}")
        paths.update(path for path in directory.rglob("*.py") if path.is_file())
    for relative_file in SOURCE_FINGERPRINT_FILES:
        path = root / relative_file
        if not path.is_file():
            raise FileNotFoundError(f"Source fingerprint file is missing: {path}")
        paths.add(path)

    manifest: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        file_sha256 = hashlib.sha256(content).hexdigest()
        relative_bytes = relative.encode("utf-8")
        # Length prefixes make concatenation unambiguous without depending on
        # JSON formatting or platform path separators.
        aggregate.update(len(relative_bytes).to_bytes(8, "big"))
        aggregate.update(relative_bytes)
        aggregate.update(len(content).to_bytes(8, "big"))
        aggregate.update(content)
        manifest.append(
            {
                "path": relative,
                "size_bytes": len(content),
                "sha256": file_sha256,
            }
        )
    return {
        "algorithm": "sha256(relative_path_length+relative_path+content_length+content)",
        "sha256": aggregate.hexdigest(),
        "file_count": len(manifest),
        "files": manifest,
    }


def _run_contract(
    config: ResponseConfig,
    input_audit: dict[str, object],
    n_folds: int,
    seed: int,
    model_seed: int,
    scenarios: tuple[str, ...],
    save_components: bool,
) -> dict[str, object]:
    effective = {
        "model": asdict(config.model),
        "entity": asdict(config.entity),
        "graph": asdict(config.graph),
        "baseline_config": str(config.baseline.path),
    }
    fingerprint_payload = {
        "response_config_sha256": sha256(config.path),
        "effective_config": _normalise_contract(effective),
        "input_hashes": {
            key: value for key, value in input_audit.items() if str(key).endswith("_sha256")
        },
        "n_folds": n_folds,
        "seed": seed,
        "model_seed": model_seed,
        "scenarios": list(scenarios),
        "save_components": bool(save_components),
        "source_fingerprint": _source_fingerprint(),
        "external_artifacts": {
            "chemical_map": _optional_artifact(config.entity.chemical_map),
            "chemical_features": _optional_artifact(config.entity.chemical_features),
            "strain_features": _optional_artifact(config.entity.strain_features),
            "chemical_registry": _optional_artifact(config.entity.chemical_registry),
            "strain_registry": _optional_artifact(config.entity.strain_registry),
            "chemical_parent_views": _optional_artifact(config.entity.chemical_parent_views),
            "chemical_identity_risks": _optional_artifact(config.entity.chemical_identity_risks),
            "graph": _optional_artifact(config.graph.artifact),
        },
    }
    material = json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "protocol": "support_regime_oof_run_contract_v3"
        if set(scenarios) & set(REGIME_SCENARIOS)
        else "entity_oof_run_contract_v2",
        "fingerprint_sha256": hashlib.sha256(material).hexdigest(),
        **fingerprint_payload,
        "input_audit": input_audit,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }


def _write_or_validate_run_contract(path: Path, contract: dict[str, object], resume: bool) -> None:
    if resume:
        if not path.is_file():
            raise ValueError(f"Cannot resume without run contract: {path}")
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("fingerprint_sha256") != contract["fingerprint_sha256"]:
            raise ValueError(
                "Existing OOF run contract does not match source, config, inputs, folds, seed, or scenarios"
            )
        return
    with path.open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, ensure_ascii=False, indent=2)


def _metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    numeric = [column for column in metrics.select_dtypes(include=np.number).columns if column != "fold"]
    rows: list[dict[str, object]] = []
    for scenario, group in metrics.groupby("scenario", sort=False):
        row: dict[str, object] = {"scenario": scenario, "n_scored_folds": int(len(group))}
        for column in numeric:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_fold_result(
    fold_dir: Path,
    prediction: pd.DataFrame,
    metrics: dict[str, object],
    official_metrics: dict[str, object],
    history: list[dict[str, float | int]],
    manifest: dict[str, object],
    support_manifest: dict[str, object] | None,
    components: dict[str, np.ndarray] | None = None,
) -> None:
    """Persist a complete fold before publishing its completion marker."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    with (fold_dir / "prediction.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            values=prediction.to_numpy(dtype=np.float32),
            sample_ids=np.asarray(prediction.index.astype(str).tolist(), dtype=np.str_),
        )
    if components is not None:
        expected_shape = prediction.shape
        required = {
            "background_plus_calibration",
            "response",
            "final",
            "is_treatment",
        }
        if set(components) != required:
            raise ValueError(
                "Core component export must contain exactly "
                f"{sorted(required)}, found {sorted(components)}"
            )
        for name, values in components.items():
            if name == "is_treatment":
                if np.asarray(values).shape != (len(prediction), 1):
                    raise ValueError(
                        "is_treatment component must have shape "
                        f"({len(prediction)}, 1)"
                    )
                continue
            if np.asarray(values).shape != expected_shape:
                raise ValueError(
                    f"Component {name} has shape {np.asarray(values).shape}, "
                    f"expected {expected_shape}"
                )
        reconstruction = components["background_plus_calibration"] + (
            components["is_treatment"] * components["response"]
        )
        reconstruction_error = float(
            np.max(np.abs(reconstruction - components["final"]))
        )
        if reconstruction_error > 5e-5:
            raise ValueError(
                "Treatment OOF component reconstruction failed: "
                f"max_abs_error={reconstruction_error}"
            )
        with (fold_dir / "components.npz").open("wb") as handle:
            np.savez_compressed(
                handle,
                sample_ids=np.asarray(
                    prediction.index.astype(str).tolist(), dtype=np.str_
                ),
                protein_ids=np.asarray(prediction.columns.astype(str), dtype=np.str_),
                background_plus_calibration=np.asarray(
                    components["background_plus_calibration"], dtype=np.float32
                ),
                response=np.asarray(components["response"], dtype=np.float32),
                final=np.asarray(components["final"], dtype=np.float32),
                is_treatment=np.asarray(components["is_treatment"], dtype=np.float32),
                reconstruction_max_abs_error=np.asarray(
                    [reconstruction_error], dtype=np.float64
                ),
            )
    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)
    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with (fold_dir / "official_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(official_metrics, handle, ensure_ascii=False, indent=2)
    if support_manifest is not None:
        file_hash = write_json_with_hash(
            fold_dir / "support_manifest.json", support_manifest
        )
        manifest["support_manifest_file_sha256"] = file_hash
    with (fold_dir / "completed.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def _load_fold_result(
    fold_dir: Path,
    proteins: list[str],
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object], dict[str, object]]:
    required = ("prediction.npz", "metrics.json", "official_metrics.json", "completed.json")
    missing = [name for name in required if not (fold_dir / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete resumed fold {fold_dir}: missing {missing}")
    with np.load(fold_dir / "prediction.npz", allow_pickle=False) as payload:
        values = np.asarray(payload["values"], dtype=np.float32)
        sample_ids = pd.Index(payload["sample_ids"].astype(str), name=SAMPLE_ID)
    prediction = pd.DataFrame(values, index=sample_ids, columns=proteins)
    records: list[dict[str, object]] = []
    for name in ("metrics.json", "official_metrics.json", "completed.json"):
        with (fold_dir / name).open("r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    completion = records[2]
    expected_support_hash = str(completion.get("support_manifest_sha256", ""))
    if expected_support_hash:
        support_path = fold_dir / "support_manifest.json"
        if not support_path.is_file():
            raise ValueError(f"Resumed fold is missing support manifest: {fold_dir}")
        support = load_json_with_hash(
            support_path,
            str(completion.get("support_manifest_file_sha256", "")) or None,
        )
        if manifest_sha256(support) != expected_support_hash:
            raise ValueError(f"Resumed fold support manifest hash mismatch: {fold_dir}")
    return prediction, records[0], records[1], records[2]


def _nested_scale_seed(
    outer_seed: int,
    model_seed: int,
    scenario: str,
    outer_fold: int,
) -> int:
    """Derive stable inner fold/model seeds without Python's salted hash()."""

    scenario_offset = REGIME_SCENARIOS.index(scenario) * 10_000
    return int(1_000_000 + outer_seed * 101 + model_seed * 17 + scenario_offset + outer_fold)


def _select_outer_fold_expert_scales(
    config: ResponseConfig,
    data: PreprocessedData,
    outer: FoldSlice,
    fold_output: Path,
    *,
    outer_seed: int,
    model_seed: int,
    source_contract_fingerprint_sha256: str,
) -> tuple[dict[str, float], dict[str, object], str]:
    """Select scales solely from nested OOF predictions inside outer.train_ids."""

    axes = active_expert_axes(config.model, outer.scenario)
    inner_n_folds = int(config.model.nested_expert_scale_inner_folds)
    nested_dir = fold_output / "nested_expert_scale"
    canonical = {"strain": 1.0, "chemical": 1.0, "pair": 1.0}
    if nested_dir.exists():
        receipt = validate_nested_scale_receipt(
            nested_dir,
            expected_scenario=outer.scenario,
            expected_fold=outer.fold,
            expected_train_ids_sha256=ids_sha256(outer.train_ids),
            expected_validation_ids_sha256=ids_sha256(outer.validation_ids),
            expected_source_contract_sha256=source_contract_fingerprint_sha256,
        )
        selected = receipt.get("selected_scales")
        if not isinstance(selected, dict):
            raise ValueError("Nested scale resume receipt lacks selected scales")
        return (
            {axis: float(selected[axis]) for axis in ("strain", "chemical", "pair")},
            receipt,
            sha256(nested_dir / "receipt.json"),
        )
    common_receipt: dict[str, object] = {
        "protocol": NESTED_SCALE_PROTOCOL,
        "status": "selected" if axes else "not_applicable",
        "scenario": outer.scenario,
        "outer_fold": int(outer.fold),
        "outer_train_ids_sha256": ids_sha256(outer.train_ids),
        "outer_validation_ids_sha256": ids_sha256(outer.validation_ids),
        "source_contract_fingerprint_sha256": source_contract_fingerprint_sha256,
        "global_scale_used": False,
        "outer_validation_labels_used": False,
        "canonical_training_scales": canonical,
        "active_axes": list(axes),
        "inner_n_folds": inner_n_folds,
        "allowed_scales": list(NESTED_ALLOWED_SCALES),
        "selection_objective": "maximize mean inner-OOF fc_pcc",
        "guardrails": {
            "relevant_residual_delta_vs_all_zero_min": 0.0,
            "high_effect_pcc_delta_vs_all_zero_min": -0.005,
            "high_effect_f1_delta_vs_all_zero_min": -0.005,
            "tie_break": "lower scales in strain,chemical,pair order",
        },
    }
    if not axes:
        selected = {"strain": 0.0, "chemical": 0.0, "pair": 0.0}
        candidate = pd.DataFrame(
            [
                {
                    "scenario": outer.scenario,
                    "strain_scale": 0.0,
                    "chemical_scale": 0.0,
                    "pair_scale": 0.0,
                    "status": "not_applicable",
                }
            ]
        )
        receipt, receipt_hash = write_nested_scale_receipt(
            nested_dir,
            payload={**common_receipt, "selected_scales": selected},
            assignments=pd.DataFrame(
                columns=["scenario", "fold", SAMPLE_ID, "eligible"]
            ),
            candidates=candidate,
            fit_receipts=[],
        )
        return selected, receipt, receipt_hash

    inner_seed = _nested_scale_seed(
        outer_seed, model_seed, outer.scenario, outer.fold
    )
    inner_slices, assignments = make_fold_slices(
        data.metadata,
        outer.train_ids,
        n_folds=inner_n_folds,
        seed=inner_seed,
        scenarios=(outer.scenario,),
        config=config,
    )
    outer_train_set = set(outer.train_ids.astype(str))
    if any(
        not set(current.train_ids.astype(str)).issubset(outer_train_set)
        or not set(current.validation_ids.astype(str)).issubset(outer_train_set)
        for current in inner_slices
    ):
        raise RuntimeError("Nested scale fold escaped the outer-training boundary")
    grid = scale_grid(axes)
    per_candidate: dict[tuple[float, float, float], list[dict[str, object]]] = {
        (item["strain"], item["chemical"], item["pair"]): [] for item in grid
    }
    fit_receipts: list[dict[str, object]] = []
    outer_train_controls = outer.train_ids[
        control_mask(data.metadata.loc[outer.train_ids]).to_numpy()
    ]
    scored_inner_folds = 0
    for ordinal, inner in enumerate(inner_slices):
        if len(inner.validation_ids) == 0:
            continue
        scored_inner_folds += 1
        inner_model_seed = inner_seed + 1000 + ordinal
        fit = fit_response_model(
            config,
            replace(data, train_ids=inner.train_ids),
            inner.train_ids,
            seed=inner_model_seed,
        )
        components = predict_fit_components(
            fit,
            data.metadata,
            inner.validation_ids,
            config.model.batch_size,
        )
        is_treatment = treatment_mask(
            data.metadata.loc[inner.validation_ids]
        ).to_numpy(dtype=np.float32)
        inner_data = replace(data, train_ids=inner.train_ids)
        control_pool = inner.train_ids.union(outer_train_controls)
        for scales in grid:
            values = compose_scaled_prediction(components, is_treatment, scales)
            prediction = pd.DataFrame(
                values,
                index=inner.validation_ids,
                columns=data.proteins,
            )
            metrics = evaluate_prediction_set(
                inner_data,
                prediction,
                inner.validation_ids,
                inner.scenario,
                control_pool_ids=control_pool,
            )
            key = (scales["strain"], scales["chemical"], scales["pair"])
            per_candidate[key].append(
                {
                    "inner_fold": int(inner.fold),
                    "inner_fit_ordinal": int(ordinal),
                    **{name: float(metrics[name]) for name in NESTED_SCALE_METRICS},
                }
            )
        fit_receipts.append(
            {
                "inner_fold": int(inner.fold),
                "inner_fit_ordinal": int(ordinal),
                "model_seed": int(inner_model_seed),
                "train_count": int(len(inner.train_ids)),
                "validation_count": int(len(inner.validation_ids)),
                "train_ids_sha256": ids_sha256(inner.train_ids),
                "validation_ids_sha256": ids_sha256(inner.validation_ids),
                "support_manifest_sha256": fit.support_manifest_sha256,
                "artifact_chain_sha256": fit.artifact_chain_sha256,
                "source_contract_fingerprint_sha256": source_contract_fingerprint_sha256,
                "canonical_training_scales": canonical,
            }
        )
    if scored_inner_folds < inner_n_folds:
        raise ValueError(
            f"Nested scale selection scored only {scored_inner_folds} folds; "
            f"at least {inner_n_folds} are required"
        )
    candidate_rows: list[dict[str, object]] = []
    for (strain, chemical, pair), fold_rows in per_candidate.items():
        if len(fold_rows) != scored_inner_folds:
            raise RuntimeError("Nested scale candidate grid is incomplete")
        row: dict[str, object] = {
            "scenario": outer.scenario,
            "strain_scale": strain,
            "chemical_scale": chemical,
            "pair_scale": pair,
            "n_scored_inner_fits": len(fold_rows),
            "per_inner_fold_metrics_json": json.dumps(
                fold_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        }
        for name in NESTED_SCALE_METRICS:
            row[name] = float(np.mean([float(item[name]) for item in fold_rows]))
        candidate_rows.append(row)
    selected, candidate_frame = select_nested_scales(
        pd.DataFrame(candidate_rows), outer.scenario, axes
    )
    receipt, receipt_hash = write_nested_scale_receipt(
        nested_dir,
        payload={
            **common_receipt,
            "inner_seed": int(inner_seed),
            "scored_inner_fits": int(scored_inner_folds),
            "selected_scales": selected,
        },
        assignments=assignments,
        candidates=candidate_frame,
        fit_receipts=fit_receipts,
    )
    return selected, receipt, receipt_hash


def run_entity_oof(
    config: ResponseConfig,
    run_dir: str | Path | None = None,
    n_folds: int = 4,
    seed: int | None = None,
    model_seed: int | None = None,
    scenarios: tuple[str, ...] = SCENARIOS,
    audit_only: bool = False,
    resume: bool = False,
    write_csv: bool = True,
    save_components: bool = False,
) -> Path:
    """Generate folds and optionally train one unchanged response model per fold."""
    input_audit = audit_inputs(config.baseline)
    data = prepare_data(config.baseline)
    oof_seed = config.model.seed if seed is None else seed
    effective_model_seed = oof_seed if model_seed is None else model_seed
    output = Path(run_dir) if run_dir else config.runs_dir / f"entity-oof-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output.mkdir(parents=True, exist_ok=resume)
    contract = _run_contract(
        config,
        input_audit,
        n_folds,
        oof_seed,
        effective_model_seed,
        scenarios,
        save_components,
    )
    _write_or_validate_run_contract(output / "run_contract.json", contract, resume)
    slices, assignments = make_fold_slices(
        data.metadata,
        data.train_ids,
        n_folds,
        oof_seed,
        scenarios,
        config,
    )
    assignment_path = output / "fold_assignments.csv"
    if resume and assignment_path.is_file():
        existing = pd.read_csv(assignment_path, keep_default_na=False)
        expected = assignments.astype({SAMPLE_ID: str})
        try:
            pd.testing.assert_frame_equal(existing, expected, check_dtype=False)
        except AssertionError as error:
            raise ValueError("Existing fold assignments do not match this config/seed") from error
    else:
        assignments.to_csv(assignment_path, index=False)

    metrics_rows: list[dict[str, object]] = []
    official_rows: list[dict[str, object]] = []
    predictions: dict[str, list[pd.DataFrame]] = {scenario: [] for scenario in scenarios}
    fold_manifest: list[dict[str, object]] = []
    if not audit_only:
        history_dir = output / "training_histories"
        history_dir.mkdir(exist_ok=resume)
        result_dir = output / "folds"
        result_dir.mkdir(exist_ok=resume)
        # Matched controls are observed labels used by the published FC scorer,
        # not model inputs.  Inner S2/S3 folds may therefore expose held-out
        # strain controls to the scorer while every fitted state remains based
        # solely on fold_slice.train_ids.
        outer_train_controls = data.train_ids[
            control_mask(data.metadata.loc[data.train_ids]).to_numpy()
        ]
        for fold_slice in slices:
            if len(fold_slice.validation_ids) == 0:
                fold_manifest.append({**_json_fold(fold_slice), "status": "no_eligible_validation_rows"})
                continue
            fold_output = result_dir / f"{fold_slice.scenario}_fold_{fold_slice.fold}"
            completion = fold_output / "completed.json"
            if resume and completion.is_file():
                prediction, standard_record, official_record, fold_record = _load_fold_result(
                    fold_output, data.proteins
                )
                expected_ids = fold_slice.validation_ids.astype(str)
                if not prediction.index.equals(pd.Index(expected_ids, name=SAMPLE_ID)):
                    raise ValueError(f"Resumed prediction IDs do not match {fold_output}")
                if config.model.nested_expert_scale_selection:
                    nested_hash = str(
                        fold_record.get("nested_expert_scale_receipt_sha256", "")
                    )
                    if not nested_hash:
                        raise ValueError(
                            f"Resumed formal fold lacks nested scale receipt: {fold_output}"
                        )
                    validate_nested_scale_receipt(
                        fold_output / "nested_expert_scale",
                        expected_sha256=nested_hash,
                        expected_scenario=fold_slice.scenario,
                        expected_fold=fold_slice.fold,
                        expected_train_ids_sha256=ids_sha256(
                            fold_slice.train_ids
                        ),
                        expected_validation_ids_sha256=ids_sha256(
                            fold_slice.validation_ids
                        ),
                        expected_source_contract_sha256=str(
                            contract["fingerprint_sha256"]
                        ),
                    )
                predictions[fold_slice.scenario].append(prediction)
                metrics_rows.append(standard_record)
                official_rows.append(official_record)
                fold_manifest.append(fold_record)
                print(f"resume: reused {fold_slice.scenario} fold {fold_slice.fold}")
                continue
            nested_scales: dict[str, float] | None = None
            nested_receipt: dict[str, object] | None = None
            nested_receipt_hash = ""
            if config.model.nested_expert_scale_selection:
                nested_scales, nested_receipt, nested_receipt_hash = (
                    _select_outer_fold_expert_scales(
                        config,
                        data,
                        fold_slice,
                        fold_output,
                        outer_seed=oof_seed,
                        model_seed=effective_model_seed,
                        source_contract_fingerprint_sha256=str(
                            contract["fingerprint_sha256"]
                        ),
                    )
                )
            fold_data: PreprocessedData = replace(data, train_ids=fold_slice.train_ids)
            fit = fit_response_model(
                config,
                fold_data,
                fold_slice.train_ids,
                seed=effective_model_seed + fold_slice.fold,
            )
            pd.DataFrame(fit.history).to_csv(
                history_dir / f"{fold_slice.scenario}_fold_{fold_slice.fold}.csv", index=False
            )
            if nested_scales is None:
                prediction = _predict(
                    fit.model,
                    fit.builder,
                    data.metadata,
                    fold_slice.validation_ids,
                    data.proteins,
                    fit.target_mean,
                    fit.target_scale,
                    fit.device,
                    config.model.batch_size,
                )
                components = (
                    _predict_core_components(
                        fit.model,
                        fit.builder,
                        data.metadata,
                        fold_slice.validation_ids,
                        fit.target_mean,
                        fit.target_scale,
                        fit.device,
                        config.model.batch_size,
                    )
                    if save_components
                    else None
                )
            else:
                components = predict_fit_components(
                    fit,
                    data.metadata,
                    fold_slice.validation_ids,
                    config.model.batch_size,
                )
                values = compose_scaled_prediction(
                    components,
                    treatment_mask(
                        data.metadata.loc[fold_slice.validation_ids]
                    ).to_numpy(dtype=np.float32),
                    nested_scales,
                )
                prediction = pd.DataFrame(
                    values,
                    index=fold_slice.validation_ids,
                    columns=data.proteins,
                )
                if save_components:
                    components = {
                        "background_plus_calibration": (
                            components["B_U"]
                            + nested_scales["strain"] * components["B_s"]
                            + components["C_obs"]
                        ),
                        "response": (
                            components["R_U"]
                            + nested_scales["strain"] * components["R_s"]
                            + nested_scales["chemical"] * components["R_c"]
                            + nested_scales["pair"] * components["R_sc"]
                        ),
                        "final": values,
                        "is_treatment": treatment_mask(
                            data.metadata.loc[fold_slice.validation_ids]
                        ).to_numpy(dtype=np.float32).reshape(-1, 1),
                    }
                else:
                    components = None
            predictions[fold_slice.scenario].append(prediction)
            standard, _ = evaluate_predictions(prediction, data.y_log2.loc[fold_slice.validation_ids])
            standard_record = {"scenario": fold_slice.scenario, "fold": fold_slice.fold, **standard}
            metrics_rows.append(standard_record)
            official = evaluate_prediction_set(
                fold_data,
                prediction,
                fold_slice.validation_ids,
                fold_slice.scenario,
                control_pool_ids=fold_slice.train_ids.union(outer_train_controls),
            )
            official_record = {"scenario": fold_slice.scenario, "fold": fold_slice.fold, **official}
            official_rows.append(official_record)
            fold_record = {
                **_json_fold(fold_slice),
                "status": "trained",
                "feature_summary": fit.builder.summary(),
                "response_basis": fit.basis_summary,
                "support_manifest_sha256": fit.support_manifest_sha256,
                "artifact_hashes": fit.artifact_hashes,
                "artifact_chain_sha256": fit.artifact_chain_sha256,
                "training_receipt": fit.training_receipt,
                "fc_training_pairs": fit.fc_pair_count,
                "scorer_control_count": int(len(outer_train_controls)),
                "scorer_only_control_count": int(
                    len(outer_train_controls.difference(fold_slice.train_ids))
                ),
                "final_training_loss": fit.history[-1]["loss"],
                "nested_expert_scale_protocol": (
                    NESTED_SCALE_PROTOCOL
                    if config.model.nested_expert_scale_selection
                    else None
                ),
                "nested_expert_scale_receipt_sha256": nested_receipt_hash,
                "nested_expert_scale_status": (
                    nested_receipt.get("status")
                    if nested_receipt is not None
                    else "disabled"
                ),
                "nested_expert_scales": nested_scales,
            }
            fold_manifest.append(fold_record)
            _write_fold_result(
                fold_output,
                prediction,
                standard_record,
                official_record,
                fit.history,
                fold_record,
                fit.support_manifest,
                components,
            )
        prediction_dir = output / "oof_predictions"
        prediction_dir.mkdir(exist_ok=resume)
        for scenario, frames in predictions.items():
            if frames:
                result = pd.concat(frames).sort_index()
                result.index.name = SAMPLE_ID
                with (prediction_dir / f"{scenario}.npz").open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        values=result.to_numpy(dtype=np.float32),
                        sample_ids=np.asarray(result.index.astype(str).tolist(), dtype=np.str_),
                        protein_ids=np.asarray(data.proteins, dtype=np.str_),
                    )
                if write_csv:
                    result.to_csv(prediction_dir / f"{scenario}.csv")

    metrics = pd.DataFrame(metrics_rows)
    official_metrics = pd.DataFrame(official_rows)
    metrics.to_csv(output / "oof_metrics.csv", index=False)
    official_metrics.to_csv(output / "oof_official_proxy_metrics.csv", index=False)
    if metrics.empty or official_metrics.empty:
        combined_metrics = metrics
    else:
        combined_metrics = metrics.merge(
            official_metrics.drop(columns=["split"], errors="ignore"),
            on=["scenario", "fold"],
            how="outer",
            validate="one_to_one",
        )
    _metric_summary(combined_metrics).to_csv(output / "oof_summary.csv", index=False)

    treatment_count = int(treatment_mask(data.metadata.loc[data.train_ids]).sum())
    coverage = {
        scenario: {
            "eligible_samples": int((assignments["scenario"].eq(scenario) & assignments["eligible"]).sum()) if not assignments.empty else 0,
            "excluded_samples": int((assignments["scenario"].eq(scenario) & ~assignments["eligible"]).sum()) if not assignments.empty else 0,
            "training_treatment_samples": treatment_count,
        }
        for scenario in scenarios
    }
    manifest = {
        "protocol": "support_regime_oof_v2"
        if set(scenarios) & set(REGIME_SCENARIOS)
        else "entity_oof_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "response_config": str(config.path),
        "seed": oof_seed,
        "model_seed": effective_model_seed,
        "n_folds": n_folds,
        "scenarios": list(scenarios),
        "audit_only": audit_only,
        "oof_prediction_formats": ["npz", *( ["csv"] if write_csv else [])],
        "core_components_saved": bool(save_components),
        "outer_validation_splits": ["val_chem_only", "val_strain_only", "val_both", "val_time"],
        "outer_validation_used_for_fitting_or_tuning": False,
        "scorer_only_controls_used_for_training": False,
        "scorer_control_policy": "all observed controls in official outer-train; held-out controls used only for matched-control subtraction",
        "fold_local_state": ["feature vocabularies", "authoritative entity support manifest", "chemical scaling", "strain scaling", "target mean/scale", "FC controls", "response SVD basis", "official-proxy references", *( ["nested inner-OOF expert scales"] if config.model.nested_expert_scale_selection else [])],
        "nested_expert_scale_selection": {
            "enabled": bool(config.model.nested_expert_scale_selection),
            "protocol": (
                NESTED_SCALE_PROTOCOL
                if config.model.nested_expert_scale_selection
                else None
            ),
            "inner_n_folds": int(config.model.nested_expert_scale_inner_folds),
            "global_scale_used": False,
            "outer_validation_labels_used": False,
        },
        "support_regime_definition": {
            "R00": "validation strain unseen; validation chemical unseen",
            "R10": "validation strain seen; validation chemical unseen",
            "R01": "validation strain unseen; validation chemical seen",
            "R11": "both entities seen; exact strain-chemical pair unseen",
            "RT": "both entities and pair seen; exact condition-time group unseen",
        }
        if set(scenarios) & set(REGIME_SCENARIOS)
        else None,
        "coverage": coverage,
        "folds": fold_manifest if fold_manifest else [_json_fold(fold) for fold in slices],
        "run_contract_fingerprint_sha256": contract["fingerprint_sha256"],
    }
    with (output / "oof_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe entity-level OOF validation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--scenarios", nargs="+", choices=ALL_SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--npz-only", action="store_true")
    parser.add_argument("--save-components", action="store_true")
    args = parser.parse_args()
    output = run_entity_oof(
        load_response_config(args.config),
        args.run_dir,
        args.n_folds,
        args.seed,
        args.model_seed,
        tuple(args.scenarios),
        args.audit_only,
        args.resume,
        not args.npz_only,
        args.save_components,
    )
    print(f"Wrote entity OOF run: {output.resolve()}")


if __name__ == "__main__":
    main()
