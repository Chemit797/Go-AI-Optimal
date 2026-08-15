"""Compare S1 OOF chemistry ablations with paired held-out-chemical metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from goai_baseline.official_metrics import (
    _frozen_delta_references,
    absolute_fidelity,
    response_metrics,
)
from goai_baseline.preprocess import prepare_data
from goai_baseline.schema import CHEMICAL, SAMPLE_ID
from goai_response.config import load_response_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _per_chemical(label: str, run: Path, data) -> pd.DataFrame:
    assignments = pd.read_csv(run / "fold_assignments.csv", keep_default_na=False)
    assignments = assignments.loc[assignments["scenario"].eq("S1") & assignments["eligible"]].copy()
    assignments[SAMPLE_ID] = assignments[SAMPLE_ID].astype(str)
    prediction = pd.read_csv(run / "oof_predictions" / "S1.csv").set_index(SAMPLE_ID, verify_integrity=True)
    prediction = prediction.loc[:, data.proteins]
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
            metrics = {
                **absolute_fidelity(chemical_prediction, data.y_log2.loc[ids]),
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


def _bootstrap_paired(values: np.ndarray, seed: int, draws: int = 10000) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n_entities": 0, "mean_delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_delta_gt_zero": float("nan")}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "n_entities": int(len(values)),
        "mean_delta": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_delta_gt_zero": float(np.mean(means > 0)),
    }


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
    baseline_paths = {item.baseline.path for item in loaded}
    if len(baseline_paths) != 1:
        raise ValueError("All candidates must use the same baseline config")
    data = prepare_data(loaded[0].baseline)
    assignment_hashes = {_sha256(run / "fold_assignments.csv") for run in runs}
    if len(assignment_hashes) != 1:
        raise ValueError("Candidates do not share identical fold assignments")

    summaries = []
    entity_frames = []
    for config, run, label, parsed in zip(configs, runs, labels, loaded):
        summary = pd.read_csv(run / "oof_summary.csv")
        row = summary.loc[summary["scenario"].eq("S1")].iloc[0].to_dict()
        summaries.append(
            {
                "model": label,
                "config": str(config.resolve()),
                "run": str(run.resolve()),
                "chemical_bits": parsed.entity.chemical_bits,
                "uses_chemical_map": parsed.entity.chemical_map is not None,
                **row,
            }
        )
        entity_frames.append(_per_chemical(label, run, data))
    summary_frame = pd.DataFrame(summaries)
    entity_frame = pd.concat(entity_frames, ignore_index=True)

    metrics = [
        "fc_pcc",
        "context_residual_pcc",
        "absolute_sample_pcc_median",
        "absolute_sample_r2_median",
        "absolute_protein_r2_median",
        "high_effect_direction_accuracy",
        "high_effect_pcc",
        "high_effect_f1",
    ]
    paired_rows: list[dict[str, object]] = []
    control_frame = entity_frame.loc[entity_frame["model"].eq(control)].set_index("chemical")
    for label in labels:
        if label == control:
            continue
        candidate = entity_frame.loc[entity_frame["model"].eq(label)].set_index("chemical")
        if not candidate.index.equals(control_frame.index):
            candidate = candidate.reindex(control_frame.index)
        for metric_index, metric in enumerate(metrics):
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
    summary_frame.to_csv(output_dir / "run_summary.csv", index=False)
    entity_frame.to_csv(output_dir / "per_chemical_metrics.csv", index=False)
    paired.to_csv(output_dir / "paired_bootstrap.csv", index=False)
    chemical_maps = [item.entity.chemical_map for item in loaded if item.entity.chemical_map is not None]
    if not chemical_maps or len(set(chemical_maps)) != 1:
        raise ValueError("Chemistry candidates must share exactly one declared chemical map")
    chemical_map = chemical_maps[0]
    assert chemical_map is not None
    manifest = {
        "protocol": "chemistry_s1_ablation_v1",
        "fold_assignments_sha256": next(iter(assignment_hashes)),
        "outer_validation_used_for_selection": False,
        "comparison_unit": "held-out chemical",
        "bootstrap_draws": 10000,
        "control": control,
        "configs": {label: {"path": str(path.resolve()), "sha256": _sha256(path)} for label, path in zip(labels, configs)},
        "runs": {label: str(path.resolve()) for label, path in zip(labels, runs)},
        "chemical_map": {
            "path": str(chemical_map),
            "sha256": _sha256(chemical_map),
        },
        "limitations": [
            "One OOF partition seed; model-seed replication is still required.",
            "The organizer has not released an executable final scorer.",
            "Protein retention was frozen from the official outer training split, not recomputed per inner fold.",
        ],
    }
    with (output_dir / "comparison_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(summary_frame[["model", "fc_pcc_mean", "context_residual_pcc_mean", "absolute_sample_r2_median_mean", "high_effect_f1_mean"]].to_string(index=False))
    print(paired.to_string(index=False))
    print(f"Wrote comparison: {output_dir.resolve()}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GOAI S1 chemistry OOF runs")
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
