"""Mask-aware diagnostics used by the baseline reproduction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_shapes(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> None:
    if prediction.shape != truth.shape or truth.shape != mask.shape:
        raise ValueError("prediction, truth, and mask must have identical shapes")


def masked_rmse(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    _validate_shapes(prediction, truth, mask)
    observed = mask.astype(bool)
    if not observed.any():
        return float("nan")
    return float(np.sqrt(np.mean((prediction[observed] - truth[observed]) ** 2)))


def masked_global_r2(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    _validate_shapes(prediction, truth, mask)
    observed = mask.astype(bool)
    if not observed.any():
        return float("nan")
    observed_truth = truth[observed]
    total = np.sum((observed_truth - observed_truth.mean()) ** 2)
    if total == 0:
        return float("nan")
    residual = np.sum((prediction[observed] - observed_truth) ** 2)
    return float(1.0 - residual / total)


def protein_r2(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    _validate_shapes(prediction, truth, mask)
    values = np.full(truth.shape[1], np.nan, dtype=np.float64)
    for column in range(truth.shape[1]):
        observed = mask[:, column].astype(bool)
        if observed.sum() < 2:
            continue
        target = truth[observed, column]
        total = np.sum((target - target.mean()) ** 2)
        if total == 0:
            continue
        residual = np.sum((prediction[observed, column] - target) ** 2)
        values[column] = 1.0 - residual / total
    return values


def evaluate_predictions(prediction: pd.DataFrame, truth: pd.DataFrame) -> tuple[dict[str, float], pd.Series]:
    if not prediction.index.equals(truth.index) or not prediction.columns.equals(truth.columns):
        raise ValueError("Prediction and truth must have identical index and columns")
    predicted = prediction.to_numpy(dtype=np.float64)
    actual = truth.to_numpy(dtype=np.float64)
    mask = ~np.isnan(actual)
    per_protein = pd.Series(protein_r2(predicted, actual, mask), index=truth.columns, name="protein_r2")
    report = {
        "n_samples": int(len(truth)),
        "n_observed_values": int(mask.sum()),
        "log2_rmse": masked_rmse(predicted, actual, mask),
        "global_r2": masked_global_r2(predicted, actual, mask),
        "protein_r2_median": float(per_protein.median(skipna=True)),
        "n_evaluable_proteins": int(per_protein.notna().sum()),
    }
    return report, per_protein
