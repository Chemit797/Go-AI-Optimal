"""Generic paired S1 OOF comparison at the held-out-chemical level."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from compare_chemistry_s1 import _bootstrap_paired, _sha256
from goai_baseline.official_metrics import (
    _axis_values,
    _frozen_delta_references,
    _median_or_nan,
    _pearson,
    _r2,
    response_metrics,
)
from goai_baseline.preprocess import prepare_data
from goai_baseline.schema import CHEMICAL, SAMPLE_ID
from goai_response.config import load_response_config


METRICS = (
    "fc_pcc",
    "context_residual_pcc",
    "absolute_sample_pcc_median",
    "absolute_sample_r2_median",
    "high_effect_direction_accuracy",
    "high_effect_pcc",
    "high_effect_f1",
)


def _per_chemical_fast(label: str, run: Path, data) -> pd.DataFrame:
    """Compute only entity-level metrics that are defined for sparse chemicals.

    Most chemicals have one sample, so protein-axis correlation/R2 is undefined
    at this level and would wastefully loop over all 4,422 proteins.  Those
    metrics remain present in the ordinary four-fold run summary.
    """
    assignments = pd.read_csv(run / "fold_assignments.csv", keep_default_na=False)
    assignments = assignments.loc[assignments["scenario"].eq("S1") & assignments["eligible"]].copy()
    assignments[SAMPLE_ID] = assignments[SAMPLE_ID].astype(str)
    csv_path = run / "oof_predictions" / "S1.csv"
    npz_path = run / "oof_predictions" / "S1.npz"
    if csv_path.is_file():
        prediction = pd.read_csv(csv_path).set_index(SAMPLE_ID, verify_integrity=True)
        prediction = prediction.loc[:, data.proteins]
    elif npz_path.is_file():
        with np.load(npz_path, allow_pickle=False) as payload:
            prediction = pd.DataFrame(
                np.asarray(payload["values"], dtype=np.float32),
                index=pd.Index(payload["sample_ids"].astype(str), name=SAMPLE_ID),
                columns=payload["protein_ids"].astype(str),
            )
        prediction = prediction.loc[:, data.proteins]
    elif list((run / "folds").glob("S1_fold_*/prediction.npz")):
        frames = []
        for fold_path in sorted((run / "folds").glob("S1_fold_*/prediction.npz")):
            with np.load(fold_path, allow_pickle=False) as payload:
                frames.append(
                    pd.DataFrame(
                        np.asarray(payload["values"], dtype=np.float32),
                        index=pd.Index(payload["sample_ids"].astype(str), name=SAMPLE_ID),
                        columns=data.proteins,
                    )
                )
        prediction = pd.concat(frames).sort_index()
        if prediction.index.has_duplicates:
            raise ValueError(f"Duplicate fold prediction IDs in {run}")
    else:
        raise ValueError(f"No S1 OOF prediction found in {run}")
    rows: list[dict[str, object]] = []
    for fold, fold_rows in assignments.groupby("fold", sort=True):
        heldout = sorted(fold_rows[CHEMICAL].astype(str).unique().tolist())
        train_mask = ~data.metadata.loc[data.train_ids, CHEMICAL].astype(str).isin(heldout)
        fold_train = data.train_ids[train_mask.to_numpy()]
        fold_data = replace(data, train_ids=fold_train)
        context_reference, drug_reference = _frozen_delta_references(fold_data)
        for chemical, chemical_rows in fold_rows.groupby(CHEMICAL, sort=True):
            ids = pd.Index(chemical_rows[SAMPLE_ID].astype(str), name=SAMPLE_ID)
            chemical_prediction = prediction.reindex(ids)
            predicted = chemical_prediction.to_numpy(dtype=np.float64)
            truth = data.y_log2.loc[ids].to_numpy(dtype=np.float64)
            mask = np.isfinite(predicted) & np.isfinite(truth)
            sample_pcc = _axis_values(predicted, truth, mask, _pearson, axis=1)
            sample_r2 = _axis_values(predicted, truth, mask, _r2, axis=1)
            metrics = {
                "absolute_sample_pcc_median": _median_or_nan(sample_pcc),
                "absolute_sample_r2_median": _median_or_nan(sample_r2),
                **response_metrics(
                    fold_data,
                    chemical_prediction,
                    ids,
                    context_reference,
                    drug_reference,
                    control_pool_ids=fold_train,
                ),
            }
            rows.append(
                {
                    "model": label,
                    "fold": int(fold),
                    "chemical": str(chemical),
                    "n_samples": int(len(ids)),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def compare(
    configs: list[Path],
    runs: list[Path],
    labels: list[str],
    control: str,
    output_dir: Path,
) -> Path:
    if not (len(configs) == len(runs) == len(labels)) or len(labels) < 2:
        raise ValueError("configs, runs, and labels must have equal length >= 2")
    if control not in labels:
        raise ValueError("control label must be one of labels")
    loaded = [load_response_config(path) for path in configs]
    if len({item.baseline.path for item in loaded}) != 1:
        raise ValueError("All candidates must use the same baseline config")
    assignment_hashes = set()
    for run in runs:
        frame = pd.read_csv(run / "fold_assignments.csv", keep_default_na=False)
        frame = frame.loc[frame["scenario"].eq("S1")].sort_values(["fold", SAMPLE_ID])
        assignment_hashes.add(hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest())
    if len(assignment_hashes) != 1:
        raise ValueError("Candidates do not share identical fold assignments")
    data = prepare_data(loaded[0].baseline)

    summaries: list[dict[str, object]] = []
    entity_frames: list[pd.DataFrame] = []
    for config, run, label, parsed in zip(configs, runs, labels, loaded):
        summary_path = run / "oof_summary.csv"
        row = (
            pd.read_csv(summary_path).query("scenario == 'S1'").iloc[0].to_dict()
            if summary_path.is_file()
            else {"scenario": "S1"}
        )
        summaries.append(
            {
                "model": label,
                "config": str(config.resolve()),
                "run": str(run.resolve()),
                "response_basis": parsed.model.response_basis,
                "response_rank": parsed.model.response_rank,
                "absolute_loss": parsed.model.absolute_loss,
                "fc_loss": parsed.model.fc_loss,
                **row,
            }
        )
        entity_frames.append(_per_chemical_fast(label, run, data))
    summary = pd.DataFrame(summaries)
    entities = pd.concat(entity_frames, ignore_index=True)

    control_frame = entities.loc[entities["model"].eq(control)].set_index("chemical")
    paired_rows: list[dict[str, object]] = []
    for label in labels:
        if label == control:
            continue
        candidate = entities.loc[entities["model"].eq(label)].set_index("chemical").reindex(control_frame.index)
        for metric_index, metric in enumerate(METRICS):
            delta = candidate[metric].to_numpy(dtype=float) - control_frame[metric].to_numpy(dtype=float)
            paired_rows.append(
                {
                    "candidate": label,
                    "control": control,
                    "metric": metric,
                    **_bootstrap_paired(delta, seed=4200 + metric_index),
                }
            )
    paired = pd.DataFrame(paired_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "run_summary.csv", index=False)
    entities.to_csv(output_dir / "per_chemical_metrics.csv", index=False)
    paired.to_csv(output_dir / "paired_bootstrap.csv", index=False)
    manifest = {
        "protocol": "generic_s1_entity_ablation_v1",
        "fold_assignments_sha256": next(iter(assignment_hashes)),
        "outer_validation_used_for_selection": False,
        "comparison_unit": "held-out chemical",
        "bootstrap_draws": 10000,
        "control": control,
        "configs": {
            label: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for label, path in zip(labels, configs)
        },
        "runs": {label: str(path.resolve()) for label, path in zip(labels, runs)},
        "limitations": [
            "One OOF partition seed; repeat model seeds before promotion.",
            "The organizer has not released an executable final scorer.",
            "Protein retention was frozen from the official outer training split.",
        ],
    }
    with (output_dir / "comparison_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    headline = [
        "model",
        "fc_pcc_mean",
        "context_residual_pcc_mean",
        "high_effect_pcc_mean",
        "high_effect_f1_mean",
    ]
    print(summary[[column for column in headline if column in summary]].to_string(index=False))
    print(paired.to_string(index=False))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GOAI S1 OOF response models")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    compare(
        [Path(value) for value in args.configs],
        [Path(value) for value in args.runs],
        args.labels,
        args.control,
        Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
