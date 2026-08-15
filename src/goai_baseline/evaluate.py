"""Frozen-split evaluation for statistical and neural baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .audit import audit_inputs
from .config import BaselineConfig, load_config
from .controls import exact_control_predictions
from .metrics import evaluate_predictions
from .preprocess import PreprocessedData, prepare_data
from .schema import SPLIT, treatment_mask


VALIDATION_SPLITS = ("val_chem_only", "val_strain_only", "val_both", "val_time")


def mean_prediction(data: PreprocessedData, sample_ids: pd.Index) -> pd.DataFrame:
    mean = data.y_log2.loc[data.train_ids].mean(axis=0).fillna(0.0)
    values = np.repeat(mean.to_numpy(dtype=np.float32)[None, :], len(sample_ids), axis=0)
    return pd.DataFrame(values, index=sample_ids, columns=data.y_log2.columns)


def _record(
    rows: list[dict[str, object]],
    method: str,
    split: str,
    subset: str,
    prediction: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    report, per_protein = evaluate_predictions(prediction, truth)
    rows.append({"method": method, "split": split, "subset": subset, **report})
    return per_protein.rename(f"{method}__{split}__{subset}").to_frame()


def _matched_subset(
    data: PreprocessedData,
    split_ids: pd.Index,
) -> tuple[pd.Index, pd.DataFrame, pd.DataFrame]:
    """Return treatment truth masked to positions observed in its exact control."""
    treatment_ids = split_ids[treatment_mask(data.metadata.loc[split_ids]).to_numpy()]
    matched = exact_control_predictions(data.metadata, data.y_log2, treatment_ids)
    candidate_ids = treatment_ids[matched.has_exact_match.to_numpy()]
    control_prediction = matched.predictions.loc[candidate_ids]
    truth = data.y_log2.loc[candidate_ids].where(control_prediction.notna())
    usable_ids = candidate_ids[truth.notna().any(axis=1).to_numpy()]
    return usable_ids, control_prediction.loc[usable_ids], truth.loc[usable_ids]


def evaluate_predictor(
    data: PreprocessedData,
    predictor: Callable[[pd.Index], pd.DataFrame],
    method: str,
    include_control_subset: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one predictor on all frozen splits and comparable control subsets."""
    rows: list[dict[str, object]] = []
    protein_reports: list[pd.DataFrame] = []
    for split in VALIDATION_SPLITS:
        split_ids = data.metadata.index[data.metadata[SPLIT].eq(split)]
        if len(split_ids) == 0:
            continue
        prediction = predictor(split_ids)
        protein_reports.append(_record(rows, method, split, "all_rows", prediction, data.y_log2.loc[split_ids]))

        if include_control_subset:
            valid_ids, _, comparable_truth = _matched_subset(data, split_ids)
            if len(valid_ids):
                comparable = prediction.reindex(valid_ids)
                protein_reports.append(
                    _record(rows, method, split, "exact_control_subset", comparable, comparable_truth)
                )
    report_frame = pd.DataFrame(rows)
    proteins = pd.concat(protein_reports, axis=1) if protein_reports else pd.DataFrame(index=data.y_log2.columns)
    return report_frame, proteins


def evaluate_mean_baseline(data: PreprocessedData) -> tuple[pd.DataFrame, pd.DataFrame]:
    return evaluate_predictor(data, lambda ids: mean_prediction(data, ids), "protein_mean")


def evaluate_matched_control(data: PreprocessedData) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    protein_reports: list[pd.DataFrame] = []
    for split in VALIDATION_SPLITS:
        split_ids = data.metadata.index[data.metadata[SPLIT].eq(split)]
        valid_ids, control_prediction, truth = _matched_subset(data, split_ids)
        if len(valid_ids) == 0:
            continue
        protein_reports.append(
            _record(
                rows,
                "matched_control",
                split,
                "exact_control_subset",
                control_prediction,
                truth,
            )
        )
    report_frame = pd.DataFrame(rows)
    proteins = pd.concat(protein_reports, axis=1) if protein_reports else pd.DataFrame(index=data.y_log2.columns)
    return report_frame, proteins


def write_evaluation(output_dir: str | Path, report: pd.DataFrame, protein_report: pd.DataFrame) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report.to_csv(directory / "metrics.csv", index=False)
    protein_report.to_csv(directory / "protein_r2.csv", index_label="protein")
    with (directory / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(orient="records"), handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GOAI statistical baselines")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=("b0_mean", "b1_matched_control"), required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config: BaselineConfig = load_config(args.config)
    audit_inputs(config)
    data = prepare_data(config)
    if args.variant == "b0_mean":
        report, proteins = evaluate_mean_baseline(data)
    else:
        report, proteins = evaluate_matched_control(data)
    output = Path(args.output_dir) if args.output_dir else config.runtime.runs_dir / args.variant
    write_evaluation(output, report, proteins)
    print(report.to_string(index=False))
    print(f"Wrote evaluation: {output.resolve()}")


if __name__ == "__main__":
    main()
