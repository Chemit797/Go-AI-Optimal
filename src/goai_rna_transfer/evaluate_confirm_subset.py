"""Evaluate a predeclared subset of S1 folds after using fold 0 for design."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate_s1 import (
    _parse_comparisons,
    _parse_prediction_specs,
    build_bootstrap_sufficient_statistics,
    build_fold_train_context_reference,
    evaluate_prediction,
    load_aligned_prediction,
    load_s1_cache,
    paired_cluster_bootstrap,
)
from .common import load_config, write_json


METRICS = ("fc_pcc", "context_residual_pcc", "high_effect_pcc", "high_effect_f1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prediction", action="append", required=True)
    parser.add_argument("--compare", action="append", default=[])
    parser.add_argument("--fold", action="append", type=int, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=4242)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    selected_folds = sorted(set(args.fold))
    if not selected_folds or not set(selected_folds).issubset({0, 1, 2, 3}):
        raise ValueError("Selected folds must be a nonempty subset of 0..3")
    config = load_config(args.config)
    cache_path = Path(config["paths"]["private_cache"]) / "goai_s1_delta.npz"
    cache = load_s1_cache(cache_path, Path(config["paths"]["goai_metadata"]))
    context = build_fold_train_context_reference(cache)
    requests = _parse_prediction_specs(args.prediction)
    comparisons = _parse_comparisons(args.compare, [request.label for request in requests])
    fold_frames = []
    statistics = {}
    excluded = sorted({0, 1, 2, 3}.difference(selected_folds))
    for request in requests:
        payload = load_aligned_prediction(request, cache)
        folds, _ = evaluate_prediction(request.label, payload, cache, context)
        fold_frames.append(folds.loc[folds["fold"].isin(selected_folds)])
        value = build_bootstrap_sufficient_statistics(request.label, payload, cache, context)
        keep_clusters = value.pearson[:, selected_folds, 0, 0].sum(axis=1) > 0
        pearson = value.pearson[keep_clusters].copy()
        f1 = value.f1[keep_clusters].copy()
        pearson[:, excluded] = 0.0
        f1[:, excluded] = 0.0
        statistics[request.label] = replace(
            value,
            clusters=value.clusters[keep_clusters],
            pearson=pearson,
            f1=f1,
        )
    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    summary_rows = []
    for model, group in fold_metrics.groupby("model", sort=False):
        row = {
            "model": model,
            "prediction_kind": group["prediction_kind"].iloc[0],
            "selected_folds": "|".join(map(str, selected_folds)),
            "n_samples_total": int(group["n_samples"].sum()),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    bootstrap = paired_cluster_bootstrap(
        statistics,
        comparisons,
        draws=args.bootstrap_draws,
        seed=args.bootstrap_seed,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    bootstrap.to_csv(output / "paired_bootstrap.csv", index=False)
    write_json(
        output / "manifest.json",
        {
            "status": "complete",
            "protocol": "post_design_confirmation_on_predeclared_s1_fold_subset",
            "selected_folds": selected_folds,
            "excluded_design_folds": excluded,
            "bootstrap_unit": "held-out chemical cluster; metrics recomputed within selected folds",
            "bootstrap_draws": args.bootstrap_draws,
            "bootstrap_seed": args.bootstrap_seed,
            "comparisons": [list(pair) for pair in comparisons],
        },
    )
    print(summary.to_string(index=False))
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
