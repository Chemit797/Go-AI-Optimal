"""Local proxy metrics derived from the published virtual-cell scoring outline.

The organizer has not supplied an executable scorer.  These metrics therefore
support model selection only: they expose the published components and freeze
reference statistics on training rows, but do not claim to reproduce final
leaderboard aggregation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .controls import exact_control_predictions
from .metrics import protein_r2
from .preprocess import PreprocessedData
from .schema import CHEMICAL, MATCH_CONTROL_FIELDS, SPLIT, treatment_mask


VALIDATION_SPLITS = ("val_chem_only", "val_strain_only", "val_both", "val_time")


def _pearson(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    observed = mask.astype(bool)
    if observed.sum() < 2:
        return float("nan")
    x = prediction[observed]
    y = truth[observed]
    if np.isclose(x.std(), 0.0) or np.isclose(y.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _r2(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    observed = mask.astype(bool)
    if observed.sum() < 2:
        return float("nan")
    target = truth[observed]
    total = np.sum((target - target.mean()) ** 2)
    if np.isclose(total, 0.0):
        return float("nan")
    return float(1.0 - np.sum((prediction[observed] - target) ** 2) / total)


def _axis_values(
    prediction: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    axis: int,
) -> np.ndarray:
    if axis == 0:
        prediction, truth, mask = prediction.T, truth.T, mask.T
    return np.asarray([metric(prediction[row], truth[row], mask[row]) for row in range(prediction.shape[0])])


def _median_or_nan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else float("nan")


def absolute_fidelity(prediction: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float | int]:
    """Return both sample-axis and protein-axis absolute-fidelity diagnostics."""
    if not prediction.index.equals(truth.index) or not prediction.columns.equals(truth.columns):
        raise ValueError("Prediction and truth must have identical index and columns")
    predicted = prediction.to_numpy(dtype=np.float64)
    actual = truth.to_numpy(dtype=np.float64)
    mask = np.isfinite(predicted) & np.isfinite(actual)
    sample_pcc = _axis_values(predicted, actual, mask, _pearson, axis=1)
    sample_r2 = _axis_values(predicted, actual, mask, _r2, axis=1)
    protein_pcc = _axis_values(predicted, actual, mask, _pearson, axis=0)
    protein_scores = protein_r2(predicted, actual, mask)
    return {
        "absolute_n_samples": int(len(truth)),
        "absolute_n_observed_values": int(mask.sum()),
        "absolute_sample_pcc_median": _median_or_nan(sample_pcc),
        "absolute_sample_r2_median": _median_or_nan(sample_r2),
        "absolute_protein_pcc_median": _median_or_nan(protein_pcc),
        "absolute_protein_r2_median": _median_or_nan(protein_scores),
    }


def _control_deltas(
    metadata: pd.DataFrame,
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    target_ids: pd.Index,
    control_pool_ids: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build treatment-control deltas using observed matched controls.

    The published outline writes ``delta_pred = y_hat_treat - y_control``.
    We follow that expression for this local proxy and only score positions
    where the treatment and matched-control intensities are observed.
    """
    targets = pd.Index(target_ids)
    treatments = targets[treatment_mask(metadata.loc[targets]).to_numpy()]
    matched = exact_control_predictions(metadata, truth, treatments, control_pool_ids)
    usable = treatments[matched.has_exact_match.to_numpy()]
    if len(usable) == 0:
        empty = pd.DataFrame(index=usable, columns=truth.columns, dtype=float)
        return empty, empty, metadata.loc[usable]
    control = matched.predictions.loc[usable]
    truth_delta = truth.loc[usable] - control
    prediction_delta = prediction.loc[usable] - control
    usable_mask = truth_delta.notna() & prediction_delta.notna()
    return prediction_delta.where(usable_mask), truth_delta.where(usable_mask), metadata.loc[usable]


def _frozen_delta_references(data: PreprocessedData) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_metadata = data.metadata.loc[data.train_ids]
    treatments = data.train_ids[treatment_mask(train_metadata).to_numpy()]
    # A finite placeholder is required because _control_deltas applies the
    # common prediction/truth observation mask. The predicted values are not
    # used here; only the matched training truth deltas define references.
    baseline = data.y_log2.loc[treatments].fillna(0.0)
    prediction_delta, truth_delta, matched_metadata = _control_deltas(
        data.metadata,
        data.y_log2,
        baseline,
        treatments,
        data.train_ids,
    )
    # prediction_delta is intentionally ignored. Matched truth deltas supply frozen references.
    if truth_delta.empty:
        empty = pd.DataFrame(columns=data.proteins, dtype=float)
        return empty, empty
    context = truth_delta.groupby(
        [matched_metadata[field].astype(str).to_numpy() for field in MATCH_CONTROL_FIELDS], sort=False
    ).mean()
    drug = truth_delta.groupby(matched_metadata[CHEMICAL].astype(str).to_numpy(), sort=False).mean()
    return context, drug


def _reference_for_rows(
    reference: pd.DataFrame,
    metadata: pd.DataFrame,
    fields: tuple[str, ...],
) -> pd.DataFrame:
    if reference.empty:
        return pd.DataFrame(index=metadata.index, columns=reference.columns, dtype=float)
    if len(fields) == 1:
        selected = reference.reindex(metadata[fields[0]].astype(str).to_numpy())
    else:
        keys = pd.MultiIndex.from_frame(metadata.loc[:, fields].astype(str))
        selected = reference.reindex(keys)
    selected.index = metadata.index
    return selected


def response_metrics(
    data: PreprocessedData,
    prediction: pd.DataFrame,
    split_ids: pd.Index,
    context_reference: pd.DataFrame,
    drug_reference: pd.DataFrame,
    control_pool_ids: pd.Index | None = None,
) -> dict[str, float | int]:
    """Score FC, residual FC, and high-effect detection for one frozen split."""
    predicted_delta, truth_delta, metadata = _control_deltas(
        data.metadata,
        data.y_log2,
        prediction,
        split_ids,
        data.metadata.index if control_pool_ids is None else control_pool_ids,
    )
    if truth_delta.empty:
        return {
            "response_n_samples": 0,
            "response_n_observed_values": 0,
            "fc_pcc": float("nan"),
            "context_residual_pcc": float("nan"),
            "drug_residual_pcc": float("nan"),
            "high_effect_direction_accuracy": float("nan"),
            "high_effect_pcc": float("nan"),
            "high_effect_precision": float("nan"),
            "high_effect_recall": float("nan"),
            "high_effect_f1": float("nan"),
        }

    predicted = predicted_delta.to_numpy(dtype=np.float64)
    actual = truth_delta.to_numpy(dtype=np.float64)
    mask = np.isfinite(predicted) & np.isfinite(actual)
    result: dict[str, float | int] = {
        "response_n_samples": int(len(truth_delta)),
        "response_n_observed_values": int(mask.sum()),
        "fc_pcc": _pearson(predicted, actual, mask),
    }

    context = _reference_for_rows(context_reference, metadata, MATCH_CONTROL_FIELDS).to_numpy(dtype=np.float64)
    context_mask = mask & np.isfinite(context)
    result["context_residual_pcc"] = _pearson(predicted - context, actual - context, context_mask)
    drug = _reference_for_rows(drug_reference, metadata, (CHEMICAL,)).to_numpy(dtype=np.float64)
    drug_mask = mask & np.isfinite(drug)
    result["drug_residual_pcc"] = _pearson(predicted - drug, actual - drug, drug_mask)

    high_true = mask & (np.abs(actual) > 1.0)
    high_pred = mask & (np.abs(predicted) > 1.0)
    true_positive = high_true & high_pred & (np.sign(predicted) == np.sign(actual))
    result["high_effect_direction_accuracy"] = (
        float(np.mean(np.sign(predicted[high_true]) == np.sign(actual[high_true]))) if high_true.any() else float("nan")
    )
    result["high_effect_pcc"] = _pearson(predicted, actual, high_true)
    result["high_effect_precision"] = float(true_positive.sum() / high_pred.sum()) if high_pred.any() else float("nan")
    result["high_effect_recall"] = float(true_positive.sum() / high_true.sum()) if high_true.any() else float("nan")
    precision = float(result["high_effect_precision"])
    recall = float(result["high_effect_recall"])
    result["high_effect_f1"] = 2.0 * precision * recall / (precision + recall) if np.isfinite(precision + recall) and precision + recall else float("nan")
    return result


def evaluate_prediction_set(
    data: PreprocessedData,
    prediction: pd.DataFrame,
    validation_ids: pd.Index,
    split_label: str,
    control_pool_ids: pd.Index | None = None,
) -> dict[str, float | int | str]:
    """Evaluate explicit IDs with references frozen to ``data.train_ids``.

    OOF callers pass a fold-specific ``PreprocessedData`` view and its training
    IDs as the control pool.  This keeps target-derived references and matched
    controls on the training side of the fold boundary.
    """
    ids = pd.Index(validation_ids)
    if not prediction.index.equals(ids):
        prediction = prediction.reindex(ids)
    truth = data.y_log2.loc[ids]
    context_reference, drug_reference = _frozen_delta_references(data)
    absolute = absolute_fidelity(prediction, truth)
    response = response_metrics(
        data,
        prediction,
        ids,
        context_reference,
        drug_reference,
        data.train_ids if control_pool_ids is None else control_pool_ids,
    )
    return {"split": split_label, **absolute, **response}


def evaluate_official_proxy(
    data: PreprocessedData,
    predictor: Callable[[pd.Index], pd.DataFrame],
) -> pd.DataFrame:
    """Evaluate a predictor against each published internal generalization split."""
    context_reference, drug_reference = _frozen_delta_references(data)
    rows: list[dict[str, float | int | str]] = []
    for split in VALIDATION_SPLITS:
        split_ids = data.metadata.index[data.metadata[SPLIT].eq(split)]
        if len(split_ids) == 0:
            continue
        prediction = predictor(split_ids)
        absolute = absolute_fidelity(prediction, data.y_log2.loc[split_ids])
        response = response_metrics(data, prediction, split_ids, context_reference, drug_reference)
        rows.append({"split": split, **absolute, **response})
    return pd.DataFrame(rows)
