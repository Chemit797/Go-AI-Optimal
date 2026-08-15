"""Compare two GOAI OOF runs across S2/S3/time scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_chemistry_s1 import _bootstrap_paired


METRICS = (
    "absolute_sample_pcc_median",
    "absolute_sample_r2_median",
    "absolute_protein_r2_median",
    "fc_pcc",
    "context_residual_pcc",
    "drug_residual_pcc",
    "high_effect_direction_accuracy",
    "high_effect_pcc",
    "high_effect_f1",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(
    control_run: Path,
    candidate_run: Path,
    control_label: str,
    candidate_label: str,
    output_dir: Path,
) -> Path:
    assignment_hash = _sha256(control_run / "fold_assignments.csv")
    if assignment_hash != _sha256(candidate_run / "fold_assignments.csv"):
        raise ValueError("Runs do not share identical fold assignments")
    summaries = []
    for label, run in ((control_label, control_run), (candidate_label, candidate_run)):
        frame = pd.read_csv(run / "oof_summary.csv")
        frame.insert(0, "model", label)
        summaries.append(frame)
    summary = pd.concat(summaries, ignore_index=True)

    control = pd.read_csv(control_run / "oof_official_proxy_metrics.csv")
    candidate = pd.read_csv(candidate_run / "oof_official_proxy_metrics.csv")
    merged = control.merge(
        candidate,
        on=["scenario", "fold"],
        suffixes=("_control", "_candidate"),
        validate="one_to_one",
    )
    paired_rows: list[dict[str, float | int | str]] = []
    for scenario, group in merged.groupby("scenario", sort=False):
        for metric_index, metric in enumerate(METRICS):
            control_column = f"{metric}_control"
            candidate_column = f"{metric}_candidate"
            if control_column not in group or candidate_column not in group:
                continue
            delta = (
                group[candidate_column].to_numpy(dtype=float)
                - group[control_column].to_numpy(dtype=float)
            )
            paired_rows.append(
                {
                    "scenario": scenario,
                    "candidate": candidate_label,
                    "control": control_label,
                    "metric": metric,
                    **_bootstrap_paired(delta, seed=7300 + metric_index),
                }
            )
    paired = pd.DataFrame(paired_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "run_summary.csv", index=False)
    merged.to_csv(output_dir / "paired_fold_metrics.csv", index=False)
    paired.to_csv(output_dir / "paired_bootstrap.csv", index=False)
    manifest = {
        "protocol": "generalization_fold_paired_comparison_v1",
        "control": {"label": control_label, "run": str(control_run.resolve())},
        "candidate": {"label": candidate_label, "run": str(candidate_run.resolve())},
        "fold_assignments_sha256": assignment_hash,
        "outer_validation_used_for_selection": False,
        "comparison_unit": "held-out fold; S3 folds are strain-by-chemical cross-blocks",
        "bootstrap_draws": 10000,
        "limitations": [
            "Four-fold S2/time confidence intervals are diagnostic and low-powered.",
            "No executable organizer scorer is available; metrics remain separate proxies.",
        ],
    }
    with (output_dir / "comparison_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    columns = [
        "model",
        "scenario",
        "fc_pcc_mean",
        "context_residual_pcc_mean",
        "drug_residual_pcc_mean",
        "high_effect_pcc_mean",
        "high_effect_f1_mean",
    ]
    print(summary[[column for column in columns if column in summary]].to_string(index=False))
    print(paired.to_string(index=False))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GOAI generalization OOF runs")
    parser.add_argument("--control-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    compare(
        Path(args.control_run),
        Path(args.candidate_run),
        args.control_label,
        args.candidate_label,
        Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
