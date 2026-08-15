"""Evaluate aligned OOF seed averages and two-family blends without refitting.

This is a consumer only: every input must already be a complete OOF producer
using identical fold assignments.  It reconstructs the fold-local official
proxy references before scoring, so seed averaging does not weaken the
leakage boundary used by the original producer runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from goai_baseline.metrics import evaluate_predictions
from goai_baseline.official_metrics import evaluate_prediction_set
from goai_baseline.preprocess import PreprocessedData, prepare_data
from goai_baseline.schema import SAMPLE_ID, control_mask
from goai_response.config import load_response_config
from goai_response.oof import _metric_summary, make_fold_slices


def _assignment_hash(run: Path, scenario: str) -> str:
    frame = pd.read_csv(run / "fold_assignments.csv", keep_default_na=False)
    # Older producers do not carry the newer support-audit columns and use
    # raw-case diagnostic time keys.  The leakage boundary is the actual
    # sample-to-fold assignment, so hash only that stable contract.
    frame = frame.loc[
        frame["scenario"].eq(scenario), ["fold", SAMPLE_ID]
    ].sort_values(["fold", SAMPLE_ID])
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def _load_prediction(run: Path, scenario: str) -> pd.DataFrame:
    path = run / "oof_predictions" / f"{scenario}.npz"
    if not (run / "oof_summary.csv").is_file() or not path.is_file():
        raise ValueError(f"Incomplete OOF producer: {run}")
    with np.load(path, allow_pickle=False) as payload:
        values = np.asarray(payload["values"], dtype=np.float64)
        sample_ids = pd.Index(payload["sample_ids"].astype(str), name=SAMPLE_ID)
        proteins = payload["protein_ids"].astype(str).tolist()
    if sample_ids.has_duplicates:
        raise ValueError(f"Duplicate OOF sample IDs: {run}")
    return pd.DataFrame(values, index=sample_ids, columns=proteins).sort_index()


def _seed_average(runs: list[Path], scenario: str) -> tuple[pd.DataFrame, str]:
    if not runs:
        raise ValueError("A seed ensemble needs at least one producer")
    assignment_hashes = {_assignment_hash(run, scenario) for run in runs}
    if len(assignment_hashes) != 1:
        raise ValueError("Seed producers use different fold assignments")
    predictions = [_load_prediction(run, scenario) for run in runs]
    reference = predictions[0]
    for current in predictions[1:]:
        if not current.index.equals(reference.index) or not current.columns.equals(reference.columns):
            raise ValueError("Seed producer prediction contracts do not align")
    values = np.mean([frame.to_numpy(dtype=np.float64) for frame in predictions], axis=0)
    return pd.DataFrame(values, index=reference.index, columns=reference.columns), assignment_hashes.pop()


def _score(
    data: PreprocessedData,
    prediction: pd.DataFrame,
    scenario: str,
    n_folds: int,
    fold_seed: int,
    config=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    slices, _ = make_fold_slices(
        data.metadata,
        data.train_ids,
        n_folds,
        fold_seed,
        (scenario,),
        config,
    )
    outer_controls = data.train_ids[
        control_mask(data.metadata.loc[data.train_ids]).to_numpy()
    ]
    standard_rows: list[dict[str, object]] = []
    official_rows: list[dict[str, object]] = []
    for fold in slices:
        expected = pd.Index(fold.validation_ids.astype(str), name=SAMPLE_ID)
        current = prediction.reindex(expected)
        if current.isna().all(axis=None):
            raise ValueError(f"No predictions for {scenario} fold {fold.fold}")
        standard, _ = evaluate_predictions(current, data.y_log2.loc[expected])
        standard_rows.append({"scenario": scenario, "fold": fold.fold, **standard})
        fold_data = replace(data, train_ids=fold.train_ids)
        official = evaluate_prediction_set(
            fold_data,
            current,
            expected,
            scenario,
            control_pool_ids=fold.train_ids.union(outer_controls),
        )
        official_rows.append({"scenario": scenario, "fold": fold.fold, **official})
    standard_frame = pd.DataFrame(standard_rows)
    official_frame = pd.DataFrame(official_rows)
    combined = standard_frame.merge(
        official_frame.drop(columns=["split"], errors="ignore"),
        on=["scenario", "fold"],
        validate="one_to_one",
    )
    return combined, _metric_summary(combined)


def _bootstrap(delta: np.ndarray, seed: int, draws: int = 20000) -> dict[str, float | int]:
    values = delta[np.isfinite(delta)]
    if not len(values):
        return {"n_units": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_positive": float("nan")}
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "n_units": int(len(values)),
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "p_positive": float((sampled > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--control-scenario")
    parser.add_argument("--candidate-scenario")
    parser.add_argument("--control-runs", nargs="+", required=True)
    parser.add_argument("--candidate-runs", nargs="+", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 1.0])
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if any(alpha < 0.0 or alpha > 1.0 for alpha in args.alphas):
        raise ValueError("Every candidate alpha must be in [0, 1]")
    control_runs = [Path(value).resolve() for value in args.control_runs]
    candidate_runs = [Path(value).resolve() for value in args.candidate_runs]
    control_scenario = args.control_scenario or args.scenario
    candidate_scenario = args.candidate_scenario or args.scenario
    control, control_hash = _seed_average(control_runs, control_scenario)
    candidate, candidate_hash = _seed_average(candidate_runs, candidate_scenario)
    if control_hash != candidate_hash:
        raise ValueError("Control and candidate fold assignments differ")
    if not control.index.equals(candidate.index) or not control.columns.equals(candidate.columns):
        raise ValueError("Control and candidate OOF contracts differ")

    config = load_response_config(args.config)
    data = prepare_data(config.baseline)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fold_tables: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    for alpha in args.alphas:
        prediction = (1.0 - alpha) * control + alpha * candidate
        folds, summary = _score(
            data,
            prediction,
            args.scenario,
            args.n_folds,
            args.fold_seed,
            config,
        )
        label = f"candidate_alpha_{alpha:g}"
        folds.insert(0, "ensemble", label)
        summary.insert(0, "ensemble", label)
        fold_tables.append(folds)
        summaries.append(summary)
    fold_frame = pd.concat(fold_tables, ignore_index=True)
    summary_frame = pd.concat(summaries, ignore_index=True)
    fold_frame.to_csv(output / "fold_metrics.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)

    baseline = fold_frame.loc[fold_frame["ensemble"].eq("candidate_alpha_0")]
    comparisons: list[dict[str, object]] = []
    for label, group in fold_frame.groupby("ensemble", sort=False):
        if label == "candidate_alpha_0":
            continue
        paired = baseline.merge(group, on=["scenario", "fold"], suffixes=("_control", "_candidate"), validate="one_to_one")
        for index, metric in enumerate(("fc_pcc", "context_residual_pcc", "drug_residual_pcc", "high_effect_pcc", "high_effect_f1", "absolute_sample_r2_median")):
            left, right = f"{metric}_control", f"{metric}_candidate"
            if left not in paired or right not in paired:
                continue
            comparisons.append({
                "ensemble": label,
                "metric": metric,
                **_bootstrap(paired[right].to_numpy(float) - paired[left].to_numpy(float), 20260812 + index),
            })
    pd.DataFrame(comparisons).to_csv(output / "paired_bootstrap.csv", index=False)
    contract = {
        "protocol": "aligned_oof_seed_ensemble_v1",
        "scenario": args.scenario,
        "control_scenario": control_scenario,
        "candidate_scenario": candidate_scenario,
        "fold_assignment_sha256": control_hash,
        "control_runs": [str(path) for path in control_runs],
        "candidate_runs": [str(path) for path in candidate_runs],
        "alphas": args.alphas,
        "selection_warning": "Alpha screening on these OOF predictions is model selection evidence, not independent confirmation or an official score.",
    }
    with (output / "contract.json").open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, ensure_ascii=False, indent=2)
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
