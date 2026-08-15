"""Leakage-safe S1 OOF scoring for the independent RNA-transfer lab.

This module deliberately consumes truth and matched controls from the private
GOAI cache instead of copying them into every model artifact.  It implements
the local published-score proxies used by the parent GOAI project, but remains
a local proxy: no executable organizer scorer is available in the workspace.

Prediction CLI contract
-----------------------
Each ``--prediction`` is ``LABEL=KIND:PATH`` where KIND is ``delta``,
``absolute``, or ``auto``.  Repeat the same label to average aligned seeds.
A directory may contain four disjoint fold NPZ files; they are concatenated.

Independent fold NPZ files should contain::

    sample_ids, proteins, pred_delta

and may additionally contain ``chemicals`` and ``fold`` for contract checks.
Legacy GOAI OOF files contain ``sample_ids``, ``protein_ids``, and ``values``;
score those with ``KIND=absolute``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


MATCH_CONTROL_FIELDS = (
    "data_source",
    "instrument",
    "Yeast_cell_plate",
    "Strains",
    "Medium",
    "Temperature",
    "pert_time",
    "pert_time_unit",
)

SUMMARY_METRICS = (
    "fc_pcc",
    "context_residual_pcc",
    "high_effect_pcc",
    "high_effect_f1",
    "absolute_sample_r2_median",
)

BOOTSTRAP_METRICS = (
    "fc_pcc",
    "context_residual_pcc",
    "high_effect_pcc",
    "high_effect_f1",
)


@dataclass
class S1Cache:
    sample_ids: np.ndarray
    chemicals: np.ndarray
    clusters: np.ndarray
    proteins: np.ndarray
    folds: np.ndarray
    true_delta: np.ndarray
    delta_mask: np.ndarray
    matched_control: np.ndarray
    matched_control_mask: np.ndarray
    truth_absolute: np.ndarray
    truth_absolute_mask: np.ndarray
    context_keys: np.ndarray
    cluster_source: str


@dataclass
class PredictionRequest:
    label: str
    kind: str
    paths: List[Path]


@dataclass
class PredictionPayload:
    sample_ids: np.ndarray
    proteins: np.ndarray
    values: np.ndarray
    kind: str
    source_files: List[Path]


@dataclass
class BootstrapSufficientStatistics:
    """Additive held-out-cluster statistics for exact metric recomputation.

    ``pearson`` has dimensions cluster x fold x metric x statistic, where the
    last dimension is n, sum(x), sum(y), sum(x^2), sum(y^2), sum(x*y) and the
    metric axis is raw FC, context-residual FC, and high-effect FC.  ``f1`` is
    cluster x fold x (same-direction true-positive, predicted-high, true-high).
    """

    model: str
    prediction_kind: str
    clusters: np.ndarray
    pearson: np.ndarray
    f1: np.ndarray


def _load_yaml(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ValueError("Configuration must be a mapping")
    return loaded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _string_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).astype(str)


def _normalize_context_keys(values: np.ndarray) -> np.ndarray:
    array = _string_array(values)
    if array.ndim == 1:
        return array
    if array.ndim != 2:
        raise ValueError("context_keys must be one- or two-dimensional")
    return np.asarray(["\x1f".join(row.tolist()) for row in array], dtype=str)


def _context_keys_from_metadata(
    sample_ids: np.ndarray,
    metadata_path: Path,
) -> np.ndarray:
    metadata = pd.read_csv(metadata_path, keep_default_na=False)
    missing = [field for field in ("sample_ID",) + MATCH_CONTROL_FIELDS if field not in metadata]
    if missing:
        raise ValueError("Metadata cannot reconstruct context keys; missing: %s" % missing)
    metadata = metadata.set_index("sample_ID", verify_integrity=True)
    selected = metadata.reindex(sample_ids.astype(str))
    if selected[list(MATCH_CONTROL_FIELDS)].isna().any(axis=None):
        raise ValueError("Metadata is missing cached S1 sample IDs")
    return np.asarray(
        ["\x1f".join(row) for row in selected[list(MATCH_CONTROL_FIELDS)].astype(str).to_numpy()],
        dtype=str,
    )


def load_s1_cache(
    cache_path: Path,
    metadata_path: Optional[Path] = None,
) -> S1Cache:
    """Load the private cache and preserve missingness masks exactly."""
    with np.load(cache_path, allow_pickle=False) as payload:
        required = {
            "sample_ids",
            "chemicals",
            "proteins",
            "folds",
            "delta",
            "mask",
            "matched_control",
            "matched_control_mask",
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError("Private S1 cache is missing arrays: %s" % missing)
        sample_ids = _string_array(payload["sample_ids"])
        chemicals = _string_array(payload["chemicals"])
        proteins = _string_array(payload["proteins"])
        folds = np.asarray(payload["folds"], dtype=np.int64)
        true_delta = np.asarray(payload["delta"], dtype=np.float32)
        delta_mask = np.asarray(payload["mask"], dtype=bool)
        matched = np.asarray(payload["matched_control"], dtype=np.float32)
        matched_mask = np.asarray(payload["matched_control_mask"], dtype=bool)

        cluster_key = next(
            (
                key
                for key in ("chemical_clusters", "chemical_cluster", "cluster_ids")
                if key in payload.files
            ),
            None,
        )
        clusters = chemicals.copy() if cluster_key is None else _string_array(payload[cluster_key])
        cluster_source = "chemicals" if cluster_key is None else str(cluster_key)

        context_keys = (
            _normalize_context_keys(payload["context_keys"])
            if "context_keys" in payload.files
            else None
        )

        truth_key = next(
            (key for key in ("treatment_truth", "truth_absolute", "truth") if key in payload.files),
            None,
        )
        truth_mask_key = next(
            (
                key
                for key in ("treatment_truth_mask", "truth_absolute_mask", "truth_mask")
                if key in payload.files
            ),
            None,
        )
        if truth_key is None:
            truth_absolute = matched + true_delta
            truth_absolute_mask = matched_mask & delta_mask
        else:
            truth_absolute = np.asarray(payload[truth_key], dtype=np.float32)
            truth_absolute_mask = (
                np.asarray(payload[truth_mask_key], dtype=bool)
                if truth_mask_key is not None
                else np.isfinite(truth_absolute)
            )

    n_samples = len(sample_ids)
    n_proteins = len(proteins)
    if len(set(sample_ids.tolist())) != n_samples:
        raise ValueError("Private S1 cache has duplicate sample IDs")
    if len(set(proteins.tolist())) != n_proteins:
        raise ValueError("Private S1 cache has duplicate protein IDs")
    for label, array in (
        ("chemicals", chemicals),
        ("clusters", clusters),
        ("folds", folds),
    ):
        if array.shape != (n_samples,):
            raise ValueError("%s must have one value per cached sample" % label)
    expected_shape = (n_samples, n_proteins)
    for label, array in (
        ("delta", true_delta),
        ("mask", delta_mask),
        ("matched_control", matched),
        ("matched_control_mask", matched_mask),
        ("truth_absolute", truth_absolute),
        ("truth_absolute_mask", truth_absolute_mask),
    ):
        if array.shape != expected_shape:
            raise ValueError("%s has shape %s, expected %s" % (label, array.shape, expected_shape))
    if context_keys is None:
        if metadata_path is None:
            raise ValueError("Cache has no context_keys; metadata_path is required")
        context_keys = _context_keys_from_metadata(sample_ids, metadata_path)
    if context_keys.shape != (n_samples,):
        raise ValueError("context_keys must have one key per cached sample")
    if not np.isin(folds, np.asarray([0, 1, 2, 3], dtype=np.int64)).all():
        raise ValueError("S1 cache must use the frozen four folds numbered 0..3")

    # Filled cache values are only meaningful under their explicit masks.
    true_delta = np.where(delta_mask, true_delta, np.nan).astype(np.float32)
    matched = np.where(matched_mask, matched, np.nan).astype(np.float32)
    truth_absolute = np.where(truth_absolute_mask, truth_absolute, np.nan).astype(np.float32)
    return S1Cache(
        sample_ids=sample_ids,
        chemicals=chemicals,
        clusters=clusters,
        proteins=proteins,
        folds=folds,
        true_delta=true_delta,
        delta_mask=delta_mask,
        matched_control=matched,
        matched_control_mask=matched_mask,
        truth_absolute=truth_absolute,
        truth_absolute_mask=truth_absolute_mask,
        context_keys=context_keys,
        cluster_source=cluster_source,
    )


def build_fold_train_context_reference(cache: S1Cache) -> np.ndarray:
    """Return per-validation-row context means fitted on the other folds only."""
    result = np.full(cache.true_delta.shape, np.nan, dtype=np.float32)
    for fold in range(4):
        train_rows = cache.folds != fold
        validation_rows = cache.folds == fold
        train_keys = cache.context_keys[train_rows]
        train_delta = cache.true_delta[train_rows]
        train_mask = cache.delta_mask[train_rows]
        references: Dict[str, np.ndarray] = {}
        for key in np.unique(train_keys):
            rows = train_keys == key
            counts = train_mask[rows].sum(axis=0, dtype=np.int64)
            sums = np.where(train_mask[rows], train_delta[rows], 0.0).sum(axis=0, dtype=np.float64)
            mean = np.full(len(cache.proteins), np.nan, dtype=np.float32)
            observed = counts > 0
            mean[observed] = (sums[observed] / counts[observed]).astype(np.float32)
            references[str(key)] = mean
        validation_indices = np.flatnonzero(validation_rows)
        for key in np.unique(cache.context_keys[validation_rows]):
            selected = validation_indices[cache.context_keys[validation_rows] == key]
            if str(key) in references:
                result[selected] = references[str(key)]
    return result


def _prediction_key(files: Iterable[str], requested_kind: str) -> Tuple[str, str]:
    available = set(files)
    if requested_kind == "delta":
        if "pred_delta" in available:
            return "pred_delta", "delta"
        if "values" in available:
            return "values", "delta"
        raise ValueError("Delta NPZ must contain pred_delta (or explicit delta values)")
    if requested_kind == "absolute":
        for key in ("pred_absolute", "prediction", "values"):
            if key in available:
                return key, "absolute"
        raise ValueError("Absolute NPZ must contain pred_absolute, prediction, or values")
    if requested_kind != "auto":
        raise ValueError("Prediction kind must be delta, absolute, or auto")
    if "pred_delta" in available:
        return "pred_delta", "delta"
    for key in ("pred_absolute", "prediction", "values"):
        if key in available:
            return key, "absolute"
    raise ValueError("Could not infer prediction kind from NPZ fields")


def _looks_like_prediction(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as payload:
            return "sample_ids" in payload.files and bool(
                {"pred_delta", "pred_absolute", "prediction", "values"}.intersection(payload.files)
            )
    except (OSError, ValueError):
        return False


def _discover_prediction_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError("Prediction path does not exist: %s" % path)
    canonical = (path / "S1.npz", path / "oof_predictions" / "S1.npz")
    existing = [candidate for candidate in canonical if candidate.is_file()]
    if len(existing) > 1:
        raise ValueError("Ambiguous canonical S1 files under %s" % path)
    if existing:
        return existing
    candidates = [candidate for candidate in sorted(path.rglob("*.npz")) if _looks_like_prediction(candidate)]
    if not candidates:
        raise ValueError("No prediction NPZ files found under %s" % path)
    return candidates


def _load_one_prediction(path: Path, requested_kind: str, cache: S1Cache) -> PredictionPayload:
    with np.load(path, allow_pickle=False) as payload:
        if "sample_ids" not in payload.files:
            raise ValueError("Prediction NPZ has no sample_ids: %s" % path)
        value_key, actual_kind = _prediction_key(payload.files, requested_kind)
        sample_ids = _string_array(payload["sample_ids"])
        protein_key = "proteins" if "proteins" in payload.files else "protein_ids" if "protein_ids" in payload.files else None
        proteins = cache.proteins.copy() if protein_key is None else _string_array(payload[protein_key])
        values = np.asarray(payload[value_key], dtype=np.float32)
        chemicals = _string_array(payload["chemicals"]) if "chemicals" in payload.files else None
        declared_fold = np.asarray(payload["fold"]) if "fold" in payload.files else None

    if sample_ids.ndim != 1 or len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError("Prediction sample_ids must be unique and one-dimensional: %s" % path)
    if values.shape != (len(sample_ids), len(proteins)):
        raise ValueError("Prediction matrix shape does not match IDs in %s" % path)
    cache_rows = pd.Index(cache.sample_ids).get_indexer(sample_ids)
    if (cache_rows < 0).any():
        raise ValueError("Prediction contains sample IDs absent from the private S1 cache")
    if chemicals is not None and not np.array_equal(chemicals, cache.chemicals[cache_rows]):
        raise ValueError("Prediction chemical labels disagree with the private S1 cache")
    if declared_fold is not None:
        if declared_fold.ndim == 0 or declared_fold.size == 1:
            declared = np.full(len(sample_ids), int(declared_fold.reshape(-1)[0]), dtype=np.int64)
        else:
            declared = declared_fold.astype(np.int64).reshape(-1)
        if declared.shape != (len(sample_ids),) or not np.array_equal(declared, cache.folds[cache_rows]):
            raise ValueError("Prediction fold labels disagree with the frozen S1 assignments")

    protein_rows = pd.Index(proteins).get_indexer(cache.proteins)
    if (protein_rows < 0).any() or len(proteins) != len(cache.proteins):
        raise ValueError("Prediction protein set does not equal the frozen cache protein set")
    values = values[:, protein_rows]
    return PredictionPayload(sample_ids, cache.proteins.copy(), values, actual_kind, [path])


def _load_path_entry(path: Path, requested_kind: str, cache: S1Cache) -> PredictionPayload:
    files = _discover_prediction_files(path)
    pieces = [_load_one_prediction(file_path, requested_kind, cache) for file_path in files]
    kinds = {piece.kind for piece in pieces}
    if len(kinds) != 1:
        raise ValueError("One prediction path mixes delta and absolute files: %s" % path)
    sample_ids = np.concatenate([piece.sample_ids for piece in pieces])
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError("A prediction directory contains duplicate fold sample IDs: %s" % path)
    values = np.concatenate([piece.values for piece in pieces], axis=0)
    return PredictionPayload(
        sample_ids=sample_ids,
        proteins=cache.proteins.copy(),
        values=values,
        kind=pieces[0].kind,
        source_files=[source for piece in pieces for source in piece.source_files],
    )


def load_aligned_prediction(request: PredictionRequest, cache: S1Cache) -> PredictionPayload:
    """Concatenate disjoint folds and average overlapping complete seed entries."""
    entries = [_load_path_entry(path, request.kind, cache) for path in request.paths]
    kinds = {entry.kind for entry in entries}
    if len(kinds) != 1:
        raise ValueError("Repeated label %s mixes prediction kinds" % request.label)
    totals = np.zeros(cache.true_delta.shape, dtype=np.float64)
    counts = np.zeros(len(cache.sample_ids), dtype=np.int32)
    cache_index = pd.Index(cache.sample_ids)
    sources: List[Path] = []
    if len(entries) > 1:
        # Repeated labels mean seed ensembling, not a way to stitch incomplete
        # seeds together.  Every path entry must independently be a complete
        # 5,078-row OOF producer; a single directory may still contain its four
        # disjoint fold files because _load_path_entry concatenates them first.
        incomplete = [
            str(request.paths[index])
            for index, entry in enumerate(entries)
            if len(entry.sample_ids) != len(cache.sample_ids)
            or set(entry.sample_ids.tolist()) != set(cache.sample_ids.tolist())
        ]
        if incomplete:
            raise ValueError(
                "Repeated label %s requires every path entry to cover all S1 rows; incomplete: %s"
                % (request.label, incomplete)
            )
    for entry in entries:
        rows = cache_index.get_indexer(entry.sample_ids)
        totals[rows] += entry.values
        counts[rows] += 1
        sources.extend(entry.source_files)
    if (counts == 0).any():
        missing = cache.sample_ids[counts == 0][:10].tolist()
        raise ValueError("Model %s does not cover all S1 OOF rows; first missing: %s" % (request.label, missing))
    values = (totals / counts[:, None]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Model %s contains non-finite prediction values" % request.label)
    return PredictionPayload(
        sample_ids=cache.sample_ids.copy(),
        proteins=cache.proteins.copy(),
        values=values,
        kind=entries[0].kind,
        source_files=sources,
    )


def _pearson(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    observed = np.asarray(mask, dtype=bool)
    if int(observed.sum()) < 2:
        return float("nan")
    x = np.asarray(prediction[observed], dtype=np.float64)
    y = np.asarray(truth[observed], dtype=np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denominator) if denominator > 0.0 else float("nan")


def _row_metric_median(
    prediction: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    metric: str,
) -> float:
    values: List[float] = []
    for row in range(len(prediction)):
        observed = mask[row]
        if int(observed.sum()) < 2:
            continue
        x = np.asarray(prediction[row, observed], dtype=np.float64)
        y = np.asarray(truth[row, observed], dtype=np.float64)
        if metric == "pcc":
            x -= x.mean()
            y -= y.mean()
            denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
            if denominator > 0.0:
                values.append(float(np.dot(x, y) / denominator))
        elif metric == "r2":
            total = float(np.sum((y - y.mean()) ** 2))
            if total > 0.0:
                values.append(float(1.0 - np.sum((x - y) ** 2) / total))
        else:
            raise ValueError("Unknown row metric: %s" % metric)
    return float(np.median(values)) if values else float("nan")


def _score_rows(
    rows: np.ndarray,
    cache: S1Cache,
    predicted_delta: np.ndarray,
    predicted_absolute: np.ndarray,
    context_reference: np.ndarray,
) -> Dict[str, object]:
    actual = cache.true_delta[rows]
    predicted = predicted_delta[rows]
    response_mask = cache.delta_mask[rows] & np.isfinite(predicted)
    context = context_reference[rows]
    context_mask = response_mask & np.isfinite(context)

    high_true = response_mask & (np.abs(actual) > 1.0)
    high_pred = response_mask & (np.abs(predicted) > 1.0)
    true_positive = high_true & high_pred & (np.sign(predicted) == np.sign(actual))
    precision = float(true_positive.sum() / high_pred.sum()) if high_pred.any() else float("nan")
    recall = float(true_positive.sum() / high_true.sum()) if high_true.any() else float("nan")
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if np.isfinite(precision + recall) and (precision + recall) > 0.0
        else float("nan")
    )

    absolute_truth = cache.truth_absolute[rows]
    absolute_prediction = predicted_absolute[rows]
    absolute_mask = cache.truth_absolute_mask[rows] & np.isfinite(absolute_prediction)
    return {
        "n_samples": int(len(rows)),
        "response_n_samples": int(np.any(response_mask, axis=1).sum()),
        "response_n_observed_values": int(response_mask.sum()),
        "fc_pcc": _pearson(predicted, actual, response_mask),
        "context_n_observed_values": int(context_mask.sum()),
        "context_residual_pcc": _pearson(predicted - context, actual - context, context_mask),
        "high_effect_n_true": int(high_true.sum()),
        "high_effect_n_pred": int(high_pred.sum()),
        "high_effect_direction_accuracy": (
            float(np.mean(np.sign(predicted[high_true]) == np.sign(actual[high_true])))
            if high_true.any()
            else float("nan")
        ),
        "high_effect_pcc": _pearson(predicted, actual, high_true),
        "high_effect_precision": precision,
        "high_effect_recall": recall,
        "high_effect_f1": f1,
        "absolute_n_observed_values": int(absolute_mask.sum()),
        "absolute_sample_pcc_median": _row_metric_median(
            absolute_prediction, absolute_truth, absolute_mask, "pcc"
        ),
        "absolute_sample_r2_median": _row_metric_median(
            absolute_prediction, absolute_truth, absolute_mask, "r2"
        ),
    }


def _prediction_views(
    payload: PredictionPayload,
    cache: S1Cache,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return response and absolute views without inventing an absolute baseline."""
    if payload.kind == "delta":
        predicted_delta = payload.values
        # A response-only model does not predict the absolute background.
        # Adding the observed validation control would be an oracle baseline and
        # cannot be compared with genuine absolute-output models, so absolute
        # fidelity is deliberately undefined for delta predictions.
        predicted_absolute = np.full_like(predicted_delta, np.nan, dtype=np.float32)
    elif payload.kind == "absolute":
        predicted_absolute = payload.values
        predicted_delta = np.where(
            cache.matched_control_mask,
            predicted_absolute - cache.matched_control,
            np.nan,
        ).astype(np.float32)
    else:
        raise ValueError("Unsupported aligned prediction kind: %s" % payload.kind)
    return predicted_delta, predicted_absolute


def evaluate_prediction(
    label: str,
    payload: PredictionPayload,
    cache: S1Cache,
    context_reference: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one aligned prediction and return fold and cluster tables."""
    predicted_delta, predicted_absolute = _prediction_views(payload, cache)

    fold_rows: List[Dict[str, object]] = []
    for fold in range(4):
        rows = np.flatnonzero(cache.folds == fold)
        fold_rows.append({"model": label, "prediction_kind": payload.kind, "fold": fold, **_score_rows(
            rows, cache, predicted_delta, predicted_absolute, context_reference
        )})

    cluster_rows: List[Dict[str, object]] = []
    for cluster in np.unique(cache.clusters):
        rows = np.flatnonzero(cache.clusters == cluster)
        chemical_names = sorted(set(cache.chemicals[rows].tolist()))
        cluster_folds = sorted(set(cache.folds[rows].tolist()))
        cluster_rows.append(
            {
                "model": label,
                "prediction_kind": payload.kind,
                "chemical_cluster": str(cluster),
                "chemicals": "|".join(chemical_names),
                "fold": cluster_folds[0] if len(cluster_folds) == 1 else -1,
                **_score_rows(rows, cache, predicted_delta, predicted_absolute, context_reference),
            }
        )
    return pd.DataFrame(fold_rows), pd.DataFrame(cluster_rows)


def summarize_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Four-fold macro summary; no sample-size weighting is applied."""
    rows: List[Dict[str, object]] = []
    for model, group in fold_metrics.groupby("model", sort=False):
        if sorted(group["fold"].astype(int).tolist()) != [0, 1, 2, 3]:
            raise ValueError("Model %s does not have exactly four S1 fold rows" % model)
        record: Dict[str, object] = {
            "model": model,
            "prediction_kind": str(group["prediction_kind"].iloc[0]),
            "n_scored_folds": 4,
            "n_samples_total": int(group["n_samples"].sum()),
        }
        for metric in SUMMARY_METRICS:
            # Match the parent scorer's pandas aggregation: a metric that is
            # undefined in one fold does not poison otherwise defined folds.
            values = group[metric].astype(float)
            record[metric + "_mean"] = float(values.mean())
            record[metric + "_std"] = float(values.std(ddof=0))
        rows.append(record)
    return pd.DataFrame(rows)


def _pearson_sufficient_statistics(
    prediction: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    observed = np.asarray(mask, dtype=bool)
    if not observed.any():
        return np.zeros(6, dtype=np.float64)
    x = np.asarray(prediction[observed], dtype=np.float64)
    y = np.asarray(truth[observed], dtype=np.float64)
    return np.asarray(
        [
            len(x),
            x.sum(dtype=np.float64),
            y.sum(dtype=np.float64),
            np.sum(x * x, dtype=np.float64),
            np.sum(y * y, dtype=np.float64),
            np.sum(x * y, dtype=np.float64),
        ],
        dtype=np.float64,
    )


def build_bootstrap_sufficient_statistics(
    label: str,
    payload: PredictionPayload,
    cache: S1Cache,
    context_reference: np.ndarray,
) -> BootstrapSufficientStatistics:
    """Pre-aggregate every model by held-out cluster and fold.

    Pearson correlation and high-effect F1 are nonlinear, but both can be
    recomputed exactly after resampling from additive moments/counts.  This
    reduces the 5,078 x 4,422 response matrix to a few kilobytes per model.
    """
    predicted_delta, _ = _prediction_views(payload, cache)
    clusters = np.unique(cache.clusters)
    pearson = np.zeros((len(clusters), 4, 3, 6), dtype=np.float64)
    f1 = np.zeros((len(clusters), 4, 3), dtype=np.float64)
    for cluster_index, cluster in enumerate(clusters):
        cluster_rows = cache.clusters == cluster
        for fold in range(4):
            rows = np.flatnonzero(cluster_rows & (cache.folds == fold))
            if not len(rows):
                continue
            actual = cache.true_delta[rows]
            predicted = predicted_delta[rows]
            response_mask = cache.delta_mask[rows] & np.isfinite(predicted)
            context = context_reference[rows]
            context_mask = response_mask & np.isfinite(context)
            high_true = response_mask & (np.abs(actual) > 1.0)
            high_pred = response_mask & (np.abs(predicted) > 1.0)
            true_positive = high_true & high_pred & (
                np.sign(predicted) == np.sign(actual)
            )
            pearson[cluster_index, fold, 0] = _pearson_sufficient_statistics(
                predicted, actual, response_mask
            )
            pearson[cluster_index, fold, 1] = _pearson_sufficient_statistics(
                predicted - context, actual - context, context_mask
            )
            pearson[cluster_index, fold, 2] = _pearson_sufficient_statistics(
                predicted, actual, high_true
            )
            f1[cluster_index, fold] = np.asarray(
                [true_positive.sum(), high_pred.sum(), high_true.sum()],
                dtype=np.float64,
            )
    return BootstrapSufficientStatistics(
        model=label,
        prediction_kind=payload.kind,
        clusters=clusters,
        pearson=pearson,
        f1=f1,
    )


def _pearson_from_sufficient_statistics(statistics: np.ndarray) -> np.ndarray:
    n = statistics[..., 0]
    sum_x = statistics[..., 1]
    sum_y = statistics[..., 2]
    sum_x2 = statistics[..., 3]
    sum_y2 = statistics[..., 4]
    sum_xy = statistics[..., 5]
    with np.errstate(divide="ignore", invalid="ignore"):
        covariance = sum_xy - (sum_x * sum_y / n)
        variance_x = sum_x2 - (sum_x * sum_x / n)
        variance_y = sum_y2 - (sum_y * sum_y / n)
        denominator = np.sqrt(variance_x * variance_y)
        result = covariance / denominator
    valid = (n >= 2.0) & (variance_x > 0.0) & (variance_y > 0.0) & np.isfinite(result)
    return np.where(valid, result, np.nan)


def _f1_from_sufficient_statistics(statistics: np.ndarray) -> np.ndarray:
    true_positive = statistics[..., 0]
    predicted_high = statistics[..., 1]
    true_high = statistics[..., 2]
    denominator = predicted_high + true_high
    # Match _score_rows exactly: when TP == 0 both precision and recall are
    # zero, and its current published-proxy convention leaves F1 undefined.
    valid = (
        (true_positive > 0.0)
        & (predicted_high > 0.0)
        & (true_high > 0.0)
        & (denominator > 0.0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        result = 2.0 * true_positive / denominator
    return np.where(valid, result, np.nan)


def _macro_four_folds(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    sums = np.where(finite, values, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = sums / counts
    return np.where(counts > 0, result, np.nan)


def _metrics_from_cluster_weights(
    statistics: BootstrapSufficientStatistics,
    weights: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Recompute fold metrics and four-fold macro means for each bootstrap draw."""
    if weights.ndim != 2 or weights.shape[1] != len(statistics.clusters):
        raise ValueError("Bootstrap cluster weights do not match sufficient statistics")
    pearson_moments = np.einsum(
        "dc,cfms->dfms", weights, statistics.pearson, optimize=True
    )
    pearson = _pearson_from_sufficient_statistics(pearson_moments)
    f1_counts = np.einsum("dc,cfs->dfs", weights, statistics.f1, optimize=True)
    f1 = _f1_from_sufficient_statistics(f1_counts)
    return {
        "fc_pcc": _macro_four_folds(pearson[:, :, 0]),
        "context_residual_pcc": _macro_four_folds(pearson[:, :, 1]),
        "high_effect_pcc": _macro_four_folds(pearson[:, :, 2]),
        "high_effect_f1": _macro_four_folds(f1),
    }


def _bootstrap_result(
    values: np.ndarray,
    observed_delta: float,
    n_clusters: int,
) -> Dict[str, object]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "n_clusters": n_clusters,
            "mean_delta": float(observed_delta),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_delta_gt_zero": float("nan"),
        }
    return {
        "n_clusters": n_clusters,
        # Preserve the historical CSV name, but report the observed full-OOF
        # four-fold macro delta rather than the mean of bootstrap replicates.
        "mean_delta": float(observed_delta),
        "ci_low": float(np.quantile(finite, 0.025)),
        "ci_high": float(np.quantile(finite, 0.975)),
        "p_delta_gt_zero": float(np.mean(finite > 0.0)),
    }


def paired_cluster_bootstrap(
    statistics_by_model: Mapping[str, BootstrapSufficientStatistics],
    comparisons: Sequence[Tuple[str, str]],
    draws: int = 10000,
    seed: int = 4200,
) -> pd.DataFrame:
    """Resample chemical clusters and exactly recompute fold-macro metrics.

    Every draw samples ``n_clusters`` held-out clusters with replacement.  A
    sampled cluster contributes all of its rows and proteins.  Each response
    metric is reconstructed inside each fold, the four folds are averaged
    without sample weighting, and only then is candidate minus control taken.
    Candidate and control always share the same cluster multiplicities.
    """
    if draws <= 0:
        raise ValueError("Bootstrap draws must be positive")
    rows: List[Dict[str, object]] = []
    for pair_index, (candidate, control) in enumerate(comparisons):
        if candidate not in statistics_by_model or control not in statistics_by_model:
            raise ValueError("Bootstrap comparison references an unknown model")
        candidate_statistics = statistics_by_model[candidate]
        control_statistics = statistics_by_model[control]
        if not np.array_equal(candidate_statistics.clusters, control_statistics.clusters):
            raise ValueError("Candidate/control chemical cluster sets do not align")
        n_clusters = len(candidate_statistics.clusters)
        point_weights = np.ones((1, n_clusters), dtype=np.float64)
        candidate_point = _metrics_from_cluster_weights(
            candidate_statistics, point_weights
        )
        control_point = _metrics_from_cluster_weights(control_statistics, point_weights)

        distributions = {
            metric: np.empty(draws, dtype=np.float64) for metric in BOOTSTRAP_METRICS
        }
        rng = np.random.default_rng(seed + pair_index)
        probabilities = np.full(n_clusters, 1.0 / n_clusters, dtype=np.float64)
        chunk_size = 1000
        for start in range(0, draws, chunk_size):
            stop = min(start + chunk_size, draws)
            weights = rng.multinomial(
                n_clusters, probabilities, size=stop - start
            ).astype(np.float64)
            candidate_draws = _metrics_from_cluster_weights(
                candidate_statistics, weights
            )
            control_draws = _metrics_from_cluster_weights(control_statistics, weights)
            for metric in BOOTSTRAP_METRICS:
                distributions[metric][start:stop] = (
                    candidate_draws[metric] - control_draws[metric]
                )

        for metric in BOOTSTRAP_METRICS:
            observed_delta = float(
                candidate_point[metric][0] - control_point[metric][0]
            )
            rows.append(
                {
                    "candidate": candidate,
                    "control": control,
                    "metric": metric,
                    **_bootstrap_result(
                        distributions[metric], observed_delta, n_clusters
                    ),
                }
            )
    return pd.DataFrame(rows)


def _parse_prediction_specs(specs: Sequence[str]) -> List[PredictionRequest]:
    ordered: List[str] = []
    grouped: Dict[str, PredictionRequest] = {}
    for spec in specs:
        if "=" not in spec or ":" not in spec.split("=", 1)[1]:
            raise ValueError("Prediction must be LABEL=KIND:PATH: %s" % spec)
        label, remainder = spec.split("=", 1)
        kind, raw_path = remainder.split(":", 1)
        label = label.strip()
        kind = kind.strip().casefold()
        if not label or kind not in {"delta", "absolute", "auto"} or not raw_path:
            raise ValueError("Invalid prediction specification: %s" % spec)
        if label in grouped and grouped[label].kind != kind:
            raise ValueError("Repeated prediction label must keep the same kind")
        if label not in grouped:
            ordered.append(label)
            grouped[label] = PredictionRequest(label, kind, [])
        grouped[label].paths.append(Path(raw_path).expanduser().resolve())
    return [grouped[label] for label in ordered]


def _parse_comparisons(
    raw: Sequence[str],
    labels: Sequence[str],
) -> List[Tuple[str, str]]:
    if raw:
        result: List[Tuple[str, str]] = []
        for value in raw:
            if ":" not in value:
                raise ValueError("Comparison must be CANDIDATE:CONTROL")
            candidate, control = value.split(":", 1)
            if candidate not in labels or control not in labels or candidate == control:
                raise ValueError("Invalid comparison: %s" % value)
            result.append((candidate, control))
        return result
    # Default ordering is explicit in output: each later model minus each
    # earlier model.  Use --compare when the desired direction matters.
    return [(candidate, control) for control, candidate in itertools.combinations(labels, 2)]


def evaluate_suite(
    config_path: Path,
    requests: Sequence[PredictionRequest],
    output_dir: Path,
    cache_path: Optional[Path] = None,
    comparisons: Optional[Sequence[Tuple[str, str]]] = None,
    bootstrap_draws: int = 10000,
    bootstrap_seed: int = 4200,
) -> Path:
    config = _load_yaml(config_path)
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("Configuration has no paths mapping")
    effective_cache = (
        cache_path.resolve()
        if cache_path is not None
        else Path(str(paths["private_cache"])).resolve() / "goai_s1_delta.npz"
    )
    metadata_path = Path(str(paths["goai_metadata"])).resolve()
    cache = load_s1_cache(effective_cache, metadata_path)
    context_reference = build_fold_train_context_reference(cache)

    fold_frames: List[pd.DataFrame] = []
    cluster_frames: List[pd.DataFrame] = []
    bootstrap_statistics: Dict[str, BootstrapSufficientStatistics] = {}
    source_contract: Dict[str, object] = {}
    for request in requests:
        payload = load_aligned_prediction(request, cache)
        folds, clusters = evaluate_prediction(request.label, payload, cache, context_reference)
        fold_frames.append(folds)
        cluster_frames.append(clusters)
        bootstrap_statistics[request.label] = build_bootstrap_sufficient_statistics(
            request.label, payload, cache, context_reference
        )
        source_contract[request.label] = {
            "prediction_kind": payload.kind,
            "files": [
                {"path": str(path), "sha256": _sha256(path)} for path in payload.source_files
            ],
            "prediction_path_entries": len(request.paths),
        }
    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    cluster_metrics = pd.concat(cluster_frames, ignore_index=True)
    summary = summarize_folds(fold_metrics)
    effective_comparisons = list(comparisons or [])
    if not effective_comparisons and len(requests) > 1:
        effective_comparisons = _parse_comparisons([], [request.label for request in requests])
    bootstrap = paired_cluster_bootstrap(
        bootstrap_statistics,
        effective_comparisons,
        draws=bootstrap_draws,
        seed=bootstrap_seed,
    ) if effective_comparisons else pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    cluster_metrics.to_csv(output_dir / "per_chemical_cluster_metrics.csv", index=False)
    bootstrap.to_csv(output_dir / "paired_bootstrap.csv", index=False)
    manifest = {
        "protocol": "independent_rna_transfer_s1_oof_proxy_v2",
        "score_status": "local_proxy_not_official",
        "fold_aggregation": "unweighted arithmetic mean of four frozen S1 folds",
        "context_reference": "per-protein truth delta mean by exact context, fit on the other three folds only",
        "high_effect_threshold_abs_log2_fc": 1.0,
        "bootstrap_unit": "held-out chemical cluster; all rows and proteins retained",
        "bootstrap_aggregation": "resample clusters globally with replacement, recompute metrics within each fold, unweighted four-fold macro mean, then paired candidate-minus-control",
        "bootstrap_metrics": list(BOOTSTRAP_METRICS),
        "seed_ensemble_contract": "every repeated-label path entry independently covers all cached S1 rows; partial seeds cannot complement one another",
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": bootstrap_seed,
        "cluster_source": cache.cluster_source,
        "cache": {"path": str(effective_cache), "sha256": _sha256(effective_cache)},
        "config": {"path": str(config_path.resolve()), "sha256": _sha256(config_path)},
        "n_samples": int(len(cache.sample_ids)),
        "n_proteins": int(len(cache.proteins)),
        "fold_counts": {
            str(fold): int(np.sum(cache.folds == fold)) for fold in range(4)
        },
        "predictions": source_contract,
        "comparisons": [
            {"candidate": candidate, "control": control}
            for candidate, control in effective_comparisons
        ],
        "limitations": [
            "This reproduces published local components, not an organizer-provided final scorer.",
            "Absolute fidelity is deliberately undefined for delta-only models; observed validation controls are not treated as model outputs.",
            "Cluster bootstrap is reported for response metrics only; absolute sample R2 is not bootstrapped.",
            "Model promotion still requires multiple model seeds and held-out-entity uncertainty checks.",
        ],
    }
    with (output_dir / "evaluation_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="LABEL=KIND:PATH; repeat a label to average aligned seeds",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="CANDIDATE:CONTROL; repeat for selected paired cluster bootstraps",
    )
    parser.add_argument("--cache", default=None)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=4200)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    requests = _parse_prediction_specs(args.prediction)
    labels = [request.label for request in requests]
    comparisons = _parse_comparisons(args.compare, labels)
    output = evaluate_suite(
        config_path=Path(args.config).expanduser().resolve(),
        requests=requests,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        cache_path=Path(args.cache).expanduser().resolve() if args.cache else None,
        comparisons=comparisons,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary = pd.read_csv(output / "summary.csv")
    print(summary.to_string(index=False))
    print("Wrote S1 OOF evaluation: %s" % output)


if __name__ == "__main__":
    main()
