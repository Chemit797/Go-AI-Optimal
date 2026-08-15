"""Select a constrained MSE/Huber blend from strict S1 OOF predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from goai_baseline.official_metrics import (
    _axis_values,
    _frozen_delta_references,
    _median_or_nan,
    _pearson,
    _r2,
    response_metrics,
)
from goai_baseline.preprocess import prepare_data
from goai_baseline.schema import CHEMICAL, SAMPLE_ID, control_mask
from goai_response.config import load_response_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_prediction(run: Path, proteins: list[str]) -> pd.DataFrame:
    prediction = pd.read_csv(run / "oof_predictions" / "S1.csv").set_index(
        SAMPLE_ID, verify_integrity=True
    )
    return prediction.loc[:, proteins].sort_index()


def select_blend(
    config_path: Path,
    run_a: Path,
    run_b: Path,
    output_dir: Path,
    weights: np.ndarray,
) -> Path:
    config = load_response_config(config_path)
    data = prepare_data(config.baseline)
    assignment_a = run_a / "fold_assignments.csv"
    assignment_b = run_b / "fold_assignments.csv"
    if _sha256(assignment_a) != _sha256(assignment_b):
        raise ValueError("Blend candidates do not share identical S1 fold assignments")
    assignments = pd.read_csv(assignment_a, keep_default_na=False)
    assignments = assignments.loc[
        assignments["scenario"].eq("S1") & assignments["eligible"]
    ].copy()
    assignments[SAMPLE_ID] = assignments[SAMPLE_ID].astype(str)
    prediction_a = _load_prediction(run_a, data.proteins)
    prediction_b = _load_prediction(run_b, data.proteins)
    if not prediction_a.index.equals(prediction_b.index):
        raise ValueError("Blend candidates have different OOF prediction IDs")
    outer_controls = data.train_ids[
        control_mask(data.metadata.loc[data.train_ids]).to_numpy()
    ]

    fold_states: list[tuple[int, object, pd.Index, pd.DataFrame, pd.DataFrame, pd.Index]] = []
    for fold, assignment in assignments.groupby("fold", sort=True):
        heldout = sorted(assignment[CHEMICAL].astype(str).unique().tolist())
        keep = ~data.metadata.loc[data.train_ids, CHEMICAL].astype(str).isin(heldout)
        fold_train = data.train_ids[keep.to_numpy()]
        ids = pd.Index(assignment[SAMPLE_ID].astype(str), name=SAMPLE_ID)
        fold_data = replace(data, train_ids=fold_train)
        context_reference, drug_reference = _frozen_delta_references(fold_data)
        fold_states.append(
            (
                int(fold),
                fold_data,
                ids,
                context_reference,
                drug_reference,
                fold_train.union(outer_controls),
            )
        )

    rows: list[dict[str, float | int | str]] = []
    blended_by_weight: dict[float, pd.DataFrame] = {}
    for weight in weights:
        weight = float(weight)
        blended = (1.0 - weight) * prediction_a + weight * prediction_b
        blended_by_weight[weight] = blended
        fold_rows: list[dict[str, float | int | str]] = []
        for fold, fold_data, ids, context_reference, drug_reference, scorer_controls in fold_states:
            fold_prediction = blended.reindex(ids)
            predicted = fold_prediction.to_numpy(dtype=np.float64)
            truth = data.y_log2.loc[ids].to_numpy(dtype=np.float64)
            mask = np.isfinite(predicted) & np.isfinite(truth)
            sample_pcc = _axis_values(predicted, truth, mask, _pearson, axis=1)
            sample_r2 = _axis_values(predicted, truth, mask, _r2, axis=1)
            metrics = {
                "absolute_sample_pcc_median": _median_or_nan(sample_pcc),
                "absolute_sample_r2_median": _median_or_nan(sample_r2),
                **response_metrics(
                    fold_data,
                    fold_prediction,
                    ids,
                    context_reference,
                    drug_reference,
                    control_pool_ids=scorer_controls,
                ),
            }
            fold_rows.append({"weight_b": weight, "fold": int(fold), **metrics})
        fold_frame = pd.DataFrame(fold_rows)
        numeric = [
            column
            for column in fold_frame.select_dtypes(include=np.number).columns
            if column not in {"weight_b", "fold"}
        ]
        row: dict[str, float | int | str] = {
            "scenario": "S1",
            "n_scored_folds": int(len(fold_frame)),
            "weight_b": weight,
        }
        row.update({f"{column}_mean": float(fold_frame[column].mean()) for column in numeric})
        rows.append(row)
    grid = pd.DataFrame(rows).sort_values("weight_b").reset_index(drop=True)
    control = grid.loc[np.isclose(grid["weight_b"], 0.0)].iloc[0]
    grid["fc_context_gain_index"] = (
        0.25 * (grid["fc_pcc_mean"] - control["fc_pcc_mean"])
        + 0.20
        * (grid["context_residual_pcc_mean"] - control["context_residual_pcc_mean"])
    )
    grid["passes_high_effect_guardrail"] = (
        (grid["high_effect_pcc_mean"] >= control["high_effect_pcc_mean"] - 0.002)
        & (grid["high_effect_f1_mean"] >= control["high_effect_f1_mean"] - 0.002)
        & (
            grid["absolute_sample_r2_median_mean"]
            >= control["absolute_sample_r2_median_mean"] - 0.0005
        )
    )
    eligible = grid.loc[grid["passes_high_effect_guardrail"]]
    selected = eligible.sort_values(
        ["fc_context_gain_index", "weight_b"], ascending=[False, True]
    ).iloc[0]
    selected_weight = float(selected["weight_b"])

    output_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output_dir / "blend_grid.csv", index=False)
    shutil.copyfile(assignment_a, output_dir / "fold_assignments.csv")
    prediction_dir = output_dir / "oof_predictions"
    prediction_dir.mkdir(exist_ok=True)
    selected_prediction = blended_by_weight[selected_weight]
    selected_prediction.index.name = SAMPLE_ID
    selected_prediction.to_csv(prediction_dir / "S1.csv")
    selected.to_frame().T.to_csv(output_dir / "oof_summary.csv", index=False)
    manifest = {
        "protocol": "constrained_s1_oof_blend_v1",
        "candidate_a": str(run_a.resolve()),
        "candidate_b": str(run_b.resolve()),
        "weight_definition": "prediction=(1-weight_b)*candidate_a+weight_b*candidate_b",
        "weights_tested": [float(value) for value in weights],
        "selected_weight_b": selected_weight,
        "selection_index": "0.25*delta_fc_pcc + 0.20*delta_context_residual_pcc",
        "guardrails": {
            "high_effect_pcc_max_drop": 0.002,
            "high_effect_f1_max_drop": 0.002,
            "absolute_sample_r2_max_drop": 0.0005,
        },
        "not_an_official_score": True,
        "fold_assignments_sha256": _sha256(assignment_a),
        "outer_validation_used_for_selection": False,
    }
    with (output_dir / "blend_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(grid[[
        "weight_b",
        "fc_pcc_mean",
        "context_residual_pcc_mean",
        "high_effect_pcc_mean",
        "high_effect_f1_mean",
        "fc_context_gain_index",
        "passes_high_effect_guardrail",
    ]].to_string(index=False))
    print(f"selected_weight_b={selected_weight:.3f}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a constrained GOAI S1 OOF blend")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()
    weights = np.arange(0.0, 1.0 + args.step / 2.0, args.step)
    select_blend(
        Path(args.config),
        Path(args.run_a),
        Path(args.run_b),
        Path(args.output_dir),
        weights,
    )


if __name__ == "__main__":
    main()
