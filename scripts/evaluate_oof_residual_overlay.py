"""Evaluate a fold-aligned residual expert on top of an existing OOF model.

The overlay is ``base + alpha * (expert - general)``.  All three inputs must
cover the same held-out rows, proteins, and fold assignments.  This keeps the
expert contribution identifiable while allowing the base to be a different
model family, such as the M6/M9 response fusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from goai_baseline.preprocess import prepare_data
from goai_baseline.schema import SAMPLE_ID
from goai_response.config import load_response_config
from scripts.evaluate_oof_seed_ensemble import (
    _bootstrap,
    _score,
    _seed_average,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_base(path: Path) -> tuple[pd.DataFrame, str]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"sample_ids", "proteins", "folds", "pred_absolute"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"Base OOF artifact lacks arrays: {sorted(missing)}")
        sample_ids = pd.Index(payload["sample_ids"].astype(str), name=SAMPLE_ID)
        proteins = payload["proteins"].astype(str).tolist()
        folds = np.asarray(payload["folds"], dtype=np.int64)
        values = np.asarray(payload["pred_absolute"], dtype=np.float64)
    if sample_ids.has_duplicates:
        raise ValueError("Base OOF artifact has duplicate sample IDs")
    if values.shape != (len(sample_ids), len(proteins)) or folds.shape != (len(sample_ids),):
        raise ValueError("Base OOF artifact has inconsistent array shapes")
    if not np.isfinite(values).all():
        raise ValueError("Base OOF prediction contains NaN or infinity")
    assignment = pd.DataFrame({"fold": folds, SAMPLE_ID: sample_ids.astype(str)})
    assignment = assignment.sort_values(["fold", SAMPLE_ID])
    assignment_hash = hashlib.sha256(
        assignment.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    prediction = pd.DataFrame(values, index=sample_ids, columns=proteins).sort_index()
    return prediction, assignment_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--base-npz", required=True)
    parser.add_argument("--general-runs", nargs="+", required=True)
    parser.add_argument("--expert-runs", nargs="+", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if any(not np.isfinite(alpha) or alpha < 0.0 for alpha in args.alphas):
        raise ValueError("Every residual alpha must be finite and non-negative")
    base_path = Path(args.base_npz).resolve()
    general_runs = [Path(value).resolve() for value in args.general_runs]
    expert_runs = [Path(value).resolve() for value in args.expert_runs]
    base, base_hash = _load_base(base_path)
    general, general_hash = _seed_average(general_runs, args.scenario)
    expert, expert_hash = _seed_average(expert_runs, args.scenario)
    if len({base_hash, general_hash, expert_hash}) != 1:
        raise ValueError("Base, general, and expert inputs use different folds")
    for label, current in (("general", general), ("expert", expert)):
        if not current.index.equals(base.index) or not current.columns.equals(base.columns):
            raise ValueError(f"{label} OOF prediction contract differs from base")

    config = load_response_config(args.config)
    data = prepare_data(config.baseline)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    residual = expert - general
    fold_tables: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    for alpha in args.alphas:
        prediction = base + alpha * residual
        folds, summary = _score(
            data,
            prediction,
            args.scenario,
            args.n_folds,
            args.fold_seed,
            config,
        )
        label = f"expert_residual_alpha_{alpha:g}"
        folds.insert(0, "ensemble", label)
        summary.insert(0, "ensemble", label)
        fold_tables.append(folds)
        summaries.append(summary)

    fold_frame = pd.concat(fold_tables, ignore_index=True)
    summary_frame = pd.concat(summaries, ignore_index=True)
    fold_frame.to_csv(output / "fold_metrics.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)
    baseline_label = f"expert_residual_alpha_{args.alphas[0]:g}"
    if args.alphas[0] != 0.0:
        raise ValueError("The first alpha must be 0 so paired comparisons have a base")
    baseline = fold_frame.loc[fold_frame["ensemble"].eq(baseline_label)]
    comparisons: list[dict[str, object]] = []
    metrics = (
        "fc_pcc",
        "context_residual_pcc",
        "drug_residual_pcc",
        "high_effect_pcc",
        "high_effect_f1",
        "absolute_sample_r2_median",
    )
    for label, group in fold_frame.groupby("ensemble", sort=False):
        if label == baseline_label:
            continue
        paired = baseline.merge(
            group,
            on=["scenario", "fold"],
            suffixes=("_base", "_candidate"),
            validate="one_to_one",
        )
        for index, metric in enumerate(metrics):
            left, right = f"{metric}_base", f"{metric}_candidate"
            if left not in paired or right not in paired:
                continue
            comparisons.append(
                {
                    "ensemble": label,
                    "metric": metric,
                    **_bootstrap(
                        paired[right].to_numpy(float) - paired[left].to_numpy(float),
                        20260815 + index,
                    ),
                }
            )
    pd.DataFrame(comparisons).to_csv(output / "paired_bootstrap.csv", index=False)
    contract = {
        "protocol": "aligned_oof_residual_overlay_v1",
        "formula": "base + alpha * (expert - general)",
        "scenario": args.scenario,
        "fold_assignment_sha256": base_hash,
        "base_npz": str(base_path),
        "base_npz_sha256": _sha256(base_path),
        "general_runs": [str(path) for path in general_runs],
        "expert_runs": [str(path) for path in expert_runs],
        "general_oof_sha256": [
            _sha256(path / "oof_predictions" / f"{args.scenario}.npz")
            for path in general_runs
        ],
        "expert_oof_sha256": [
            _sha256(path / "oof_predictions" / f"{args.scenario}.npz")
            for path in expert_runs
        ],
        "alphas": args.alphas,
        "selection_warning": (
            "Alpha screening is model-selection evidence on aligned strict OOF, "
            "not an independent confirmation or official score."
        ),
    }
    with (output / "contract.json").open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, ensure_ascii=False, indent=2)
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
