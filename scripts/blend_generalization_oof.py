"""Select scenario-specific constrained blends for GOAI entity OOF runs."""

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
from goai_baseline.schema import SAMPLE_ID, control_mask
from goai_response.config import load_response_config
from goai_response.oof import make_fold_slices


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_prediction(run: Path, scenario: str, proteins: list[str]) -> pd.DataFrame:
    prediction = pd.read_csv(run / "oof_predictions" / f"{scenario}.csv").set_index(
        SAMPLE_ID, verify_integrity=True
    )
    return prediction.loc[:, proteins].sort_index()


def _selection_gain(grid: pd.DataFrame, control: pd.Series, scenario: str) -> pd.Series:
    if scenario == "S2":
        return 0.25 * (grid["fc_pcc_mean"] - control["fc_pcc_mean"]) + 0.20 * (
            grid["drug_residual_pcc_mean"] - control["drug_residual_pcc_mean"]
        )
    if scenario in {"time", "time_forward"}:
        return (
            0.05 * (grid["fc_pcc_mean"] - control["fc_pcc_mean"])
            + 0.025
            * (
                grid["absolute_sample_pcc_median_mean"]
                - control["absolute_sample_pcc_median_mean"]
            )
            + 0.025
            * (
                grid["absolute_sample_r2_median_mean"]
                - control["absolute_sample_r2_median_mean"]
            )
        )
    return (
        0.05 * (grid["fc_pcc_mean"] - control["fc_pcc_mean"])
        + 0.025
        * (
            grid["absolute_sample_pcc_median_mean"]
            - control["absolute_sample_pcc_median_mean"]
        )
        + 0.025
        * (
            grid["absolute_sample_r2_median_mean"]
            - control["absolute_sample_r2_median_mean"]
        )
    )


def select_blends(
    config_path: Path,
    run_a: Path,
    run_b: Path,
    output_dir: Path,
    scenarios: tuple[str, ...],
    n_folds: int,
    fold_seed: int,
    weights: np.ndarray,
) -> Path:
    config = load_response_config(config_path)
    data = prepare_data(config.baseline)
    assignment_a = run_a / "fold_assignments.csv"
    assignment_b = run_b / "fold_assignments.csv"
    if _sha256(assignment_a) != _sha256(assignment_b):
        raise ValueError("Blend candidates do not share identical fold assignments")
    slices, regenerated = make_fold_slices(
        data.metadata, data.train_ids, n_folds, fold_seed, scenarios, config
    )
    existing = pd.read_csv(assignment_a, keep_default_na=False)
    pd.testing.assert_frame_equal(existing, regenerated, check_dtype=False)
    outer_controls = data.train_ids[
        control_mask(data.metadata.loc[data.train_ids]).to_numpy()
    ]

    all_grid: list[pd.DataFrame] = []
    selected_rows: list[pd.Series] = []
    selected_fold_rows: list[dict[str, float | int | str]] = []
    selected_predictions: dict[str, pd.DataFrame] = {}
    selected_weights: dict[str, float] = {}
    for scenario in scenarios:
        prediction_a = _load_prediction(run_a, scenario, data.proteins)
        prediction_b = _load_prediction(run_b, scenario, data.proteins)
        if not prediction_a.index.equals(prediction_b.index):
            raise ValueError(f"{scenario} candidates have different OOF IDs")
        states = []
        for fold in (item for item in slices if item.scenario == scenario and len(item.validation_ids)):
            fold_data = replace(data, train_ids=fold.train_ids)
            context_reference, drug_reference = _frozen_delta_references(fold_data)
            states.append(
                (
                    fold.fold,
                    fold_data,
                    fold.validation_ids,
                    context_reference,
                    drug_reference,
                    fold.train_ids.union(outer_controls),
                )
            )
        scenario_rows: list[dict[str, float | int | str]] = []
        fold_by_weight: dict[float, list[dict[str, float | int | str]]] = {}
        blend_by_weight: dict[float, pd.DataFrame] = {}
        for value in weights:
            weight = float(value)
            blended = (1.0 - weight) * prediction_a + weight * prediction_b
            blend_by_weight[weight] = blended
            fold_rows: list[dict[str, float | int | str]] = []
            for fold_number, fold_data, ids, context_ref, drug_ref, controls in states:
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
                        context_ref,
                        drug_ref,
                        control_pool_ids=controls,
                    ),
                }
                fold_rows.append(
                    {"scenario": scenario, "fold": fold_number, "weight_b": weight, **metrics}
                )
            fold_by_weight[weight] = fold_rows
            frame = pd.DataFrame(fold_rows)
            numeric = [
                column
                for column in frame.select_dtypes(include=np.number).columns
                if column not in {"fold", "weight_b"}
            ]
            row: dict[str, float | int | str] = {
                "scenario": scenario,
                "n_scored_folds": int(len(frame)),
                "weight_b": weight,
            }
            row.update({f"{column}_mean": float(frame[column].mean()) for column in numeric})
            scenario_rows.append(row)
        grid = pd.DataFrame(scenario_rows).sort_values("weight_b").reset_index(drop=True)
        control = grid.loc[np.isclose(grid["weight_b"], 0.0)].iloc[0]
        grid["selection_gain_index"] = _selection_gain(grid, control, scenario)
        grid["passes_high_effect_guardrail"] = (
            (grid["high_effect_pcc_mean"] >= control["high_effect_pcc_mean"] - 0.002)
            & (grid["high_effect_f1_mean"] >= control["high_effect_f1_mean"] - 0.002)
            & (
                grid["absolute_sample_r2_median_mean"]
                >= control["absolute_sample_r2_median_mean"] - 0.0005
            )
        )
        selected = grid.loc[grid["passes_high_effect_guardrail"]].sort_values(
            ["selection_gain_index", "weight_b"], ascending=[False, True]
        ).iloc[0]
        selected_weight = float(selected["weight_b"])
        selected_weights[scenario] = selected_weight
        selected_rows.append(selected)
        selected_fold_rows.extend(fold_by_weight[selected_weight])
        selected_predictions[scenario] = blend_by_weight[selected_weight]
        all_grid.append(grid)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(all_grid, ignore_index=True).to_csv(output_dir / "blend_grid.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(output_dir / "oof_summary.csv", index=False)
    pd.DataFrame(selected_fold_rows).to_csv(
        output_dir / "oof_official_proxy_metrics.csv", index=False
    )
    shutil.copyfile(assignment_a, output_dir / "fold_assignments.csv")
    prediction_dir = output_dir / "oof_predictions"
    prediction_dir.mkdir(exist_ok=True)
    for scenario, prediction in selected_predictions.items():
        prediction.index.name = SAMPLE_ID
        prediction.to_csv(prediction_dir / f"{scenario}.csv")
    manifest = {
        "protocol": "scenario_constrained_oof_blend_v1",
        "candidate_a": str(run_a.resolve()),
        "candidate_b": str(run_b.resolve()),
        "selected_weight_b": selected_weights,
        "weight_definition": "prediction=(1-weight_b)*candidate_a+weight_b*candidate_b",
        "weights_tested": [float(value) for value in weights],
        "not_an_official_score": True,
        "guardrails": {
            "high_effect_pcc_max_drop": 0.002,
            "high_effect_f1_max_drop": 0.002,
            "absolute_sample_r2_max_drop": 0.0005,
        },
        "outer_validation_used_for_selection": False,
        "fold_assignments_sha256": _sha256(assignment_a),
    }
    with (output_dir / "blend_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(pd.DataFrame(selected_rows)[[
        "scenario",
        "weight_b",
        "fc_pcc_mean",
        "context_residual_pcc_mean",
        "drug_residual_pcc_mean",
        "high_effect_pcc_mean",
        "high_effect_f1_mean",
        "selection_gain_index",
    ]].to_string(index=False))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Select scenario-specific GOAI OOF blends")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--step", type=float, default=0.10)
    args = parser.parse_args()
    select_blends(
        Path(args.config),
        Path(args.run_a),
        Path(args.run_b),
        Path(args.output_dir),
        tuple(args.scenarios),
        args.n_folds,
        args.fold_seed,
        np.arange(0.0, 1.0 + args.step / 2.0, args.step),
    )


if __name__ == "__main__":
    main()
