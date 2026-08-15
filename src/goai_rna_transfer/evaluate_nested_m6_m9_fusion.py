"""Nested selection of M6/M9 response fusion without reusing outer-fold scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd

from .common import load_config, sha256, write_json
from .evaluate_s1 import (
    PredictionPayload,
    _pearson,
    build_bootstrap_sufficient_statistics,
    build_fold_train_context_reference,
    evaluate_prediction,
    load_aligned_prediction,
    load_s1_cache,
    paired_cluster_bootstrap,
    summarize_folds,
    PredictionRequest,
)
from .fuse_m6_m9_s1 import _average_component_seeds


METRICS = ("fc_pcc", "context_residual_pcc", "high_effect_pcc", "high_effect_f1")


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _response(
    family: str,
    weight: float,
    m6_response: np.ndarray,
    m9_response: np.ndarray,
    m9_base_response: np.ndarray,
    threshold: float | None = None,
    takeover: float | None = None,
) -> np.ndarray:
    if family == "blend":
        return ((1.0 - weight) * m6_response + weight * m9_response).astype(np.float32)
    if family == "op3_residual":
        return (m6_response + weight * (m9_response - m9_base_response)).astype(np.float32)
    if family == "high_specialist":
        if threshold is None or takeover is None:
            raise ValueError("high_specialist requires threshold and takeover")
        base = (1.0 - weight) * m6_response + weight * m9_response
        # The gate is prediction-only. Validation targets never decide which
        # protein positions are handed back to the M6 high-effect specialist.
        gate = np.abs(m6_response) >= threshold
        return (base + takeover * gate * (m6_response - base)).astype(np.float32)
    raise ValueError(f"Unknown fusion family: {family}")


def _candidate_specs(args: argparse.Namespace) -> list[dict[str, float | str]]:
    specs: list[dict[str, float | str]] = []
    for weight in np.arange(0.0, args.blend_max + 1e-9, args.step):
        specs.append({"family": "blend", "weight": float(np.round(weight, 8))})
    for weight in np.arange(0.0, args.residual_max + 1e-9, args.step):
        specs.append({"family": "op3_residual", "weight": float(np.round(weight, 8))})
    for weight in args.specialist_base_weights:
        for threshold in args.specialist_thresholds:
            for takeover in args.specialist_takeovers:
                specs.append(
                    {
                        "family": "high_specialist",
                        "weight": float(weight),
                        "threshold": float(threshold),
                        "takeover": float(takeover),
                    }
                )
    return specs


def _label(spec: Mapping[str, float | str]) -> str:
    label = f"{spec['family']}_w{float(spec['weight']):g}"
    if spec["family"] == "high_specialist":
        label += f"_t{float(spec['threshold']):g}_g{float(spec['takeover']):g}"
    return label


def _fold_metrics(
    predicted_delta: np.ndarray,
    cache,
    context_reference: np.ndarray,
) -> list[dict[str, float]]:
    rows = []
    for fold in range(4):
        selected = cache.folds == fold
        actual = cache.true_delta[selected]
        predicted = predicted_delta[selected]
        mask = cache.delta_mask[selected] & np.isfinite(predicted)
        context = context_reference[selected]
        context_mask = mask & np.isfinite(context)
        high_true = mask & (np.abs(actual) > 1.0)
        high_pred = mask & (np.abs(predicted) > 1.0)
        true_positive = high_true & high_pred & (np.sign(predicted) == np.sign(actual))
        denominator = float(high_pred.sum() + high_true.sum())
        rows.append(
            {
                "fold": fold,
                "fc_pcc": _pearson(predicted, actual, mask),
                "context_residual_pcc": _pearson(
                    predicted - context, actual - context, context_mask
                ),
                "high_effect_pcc": _pearson(predicted, actual, high_true),
                "high_effect_f1": (
                    float(2.0 * true_positive.sum() / denominator)
                    if denominator > 0.0
                    else float("nan")
                ),
            }
        )
    return rows


def _select(
    grid: pd.DataFrame,
    outer_fold: int,
    selector: str,
    family: str | None = None,
) -> pd.Series:
    inner_all = grid.loc[grid["fold"] != outer_fold]
    baseline = (
        inner_all.loc[inner_all["candidate"] == "blend_w0"]
        .groupby("candidate", as_index=False)[list(METRICS)]
        .mean()
        .iloc[0]
    )
    inner = inner_all
    if family is not None:
        inner = inner.loc[inner["family"] == family]
    summary = inner.groupby("candidate", as_index=False)[list(METRICS)].mean()
    if selector == "guarded":
        allowed = summary.loc[
            summary["high_effect_pcc"].ge(baseline["high_effect_pcc"] - 0.005)
            & summary["high_effect_f1"].ge(baseline["high_effect_f1"] - 0.005)
        ]
        if not allowed.empty:
            summary = allowed
    winner = summary.sort_values(
        ["fc_pcc", "context_residual_pcc", "high_effect_pcc", "candidate"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return winner


def run(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    cache_path = Path(config["paths"]["private_cache"]) / "goai_s1_delta.npz"
    cache = load_s1_cache(cache_path, Path(config["paths"]["goai_metadata"]))
    context_reference = build_fold_train_context_reference(cache)
    roots = [Path(value).expanduser().resolve() for value in args.m6_components]
    m6 = _average_component_seeds(roots, cache)
    m9 = load_aligned_prediction(
        PredictionRequest("m9_op3", "delta", [Path(args.m9_op3).resolve()]), cache
    ).values
    m9_base = load_aligned_prediction(
        PredictionRequest("m9_base", "delta", [Path(args.m9_base).resolve()]), cache
    ).values
    base_delta = np.where(
        cache.matched_control_mask,
        m6["background_plus_calibration"] - cache.matched_control,
        np.nan,
    ).astype(np.float32)

    specs = _candidate_specs(args)
    spec_by_label = {_label(spec): spec for spec in specs}
    if len(spec_by_label) != len(specs):
        raise ValueError("Candidate labels are not unique")
    grid_rows = []
    for label, spec in spec_by_label.items():
        response = _response(
            str(spec["family"]),
            float(spec["weight"]),
            m6["response"],
            m9,
            m9_base,
            None if "threshold" not in spec else float(spec["threshold"]),
            None if "takeover" not in spec else float(spec["takeover"]),
        )
        for row in _fold_metrics(base_delta + response, cache, context_reference):
            grid_rows.append({"candidate": label, **spec, **row})
    grid = pd.DataFrame(grid_rows)

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    grid.to_csv(output / "inner_grid_fold_metrics.csv", index=False)
    finalist_frames = []
    selection_rows = []
    finalist_arrays: Dict[str, np.ndarray] = {"M6_core": m6["final"]}
    selector_specs = (
        ("fc", "fc", None),
        ("guarded", "guarded", None),
        ("high_specialist_fc", "fc", "high_specialist"),
        ("high_specialist_guarded", "guarded", "high_specialist"),
    )
    for output_label, criterion, family in selector_specs:
        nested = np.empty_like(m6["final"])
        for outer_fold in range(4):
            winner = _select(grid, outer_fold, criterion, family)
            label = str(winner["candidate"])
            spec = spec_by_label[label]
            response = _response(
                str(spec["family"]),
                float(spec["weight"]),
                m6["response"],
                m9,
                m9_base,
                None if "threshold" not in spec else float(spec["threshold"]),
                None if "takeover" not in spec else float(spec["takeover"]),
            )
            selected = cache.folds == outer_fold
            nested[selected] = (
                m6["background_plus_calibration"][selected] + response[selected]
            )
            selection_rows.append(
                {
                    "selector": output_label,
                    "outer_fold": outer_fold,
                    "selected_candidate": label,
                    **{f"inner_{metric}": float(winner[metric]) for metric in METRICS},
                }
            )
        finalist_arrays[f"nested_{output_label}"] = nested

    statistics = {}
    for label, absolute in finalist_arrays.items():
        payload = PredictionPayload(
            sample_ids=cache.sample_ids.copy(),
            proteins=cache.proteins.copy(),
            values=absolute.astype(np.float32),
            kind="absolute",
            source_files=[],
        )
        folds, _ = evaluate_prediction(label, payload, cache, context_reference)
        finalist_frames.append(folds)
        statistics[label] = build_bootstrap_sufficient_statistics(
            label, payload, cache, context_reference
        )
    fold_metrics = pd.concat(finalist_frames, ignore_index=True)
    summary = summarize_folds(fold_metrics)
    bootstrap = paired_cluster_bootstrap(
        statistics,
        [(label, "M6_core") for label in finalist_arrays if label != "M6_core"],
        draws=args.bootstrap_draws,
        seed=args.bootstrap_seed,
    )
    selections = pd.DataFrame(selection_rows)
    selections.to_csv(output / "outer_selections.csv", index=False)
    fold_metrics.to_csv(output / "outer_fold_metrics.csv", index=False)
    summary.to_csv(output / "outer_summary.csv", index=False)
    bootstrap.to_csv(output / "paired_bootstrap.csv", index=False)
    for label, absolute in finalist_arrays.items():
        np.savez_compressed(
            output / f"{label}.npz",
            sample_ids=cache.sample_ids,
            proteins=cache.proteins,
            folds=cache.folds,
            pred_absolute=absolute.astype(np.float32),
        )
    manifest = {
        "status": "complete",
        "protocol": "m6_m9_nested_leave_one_fold_out_fusion_v1",
        "score_status": "local_strict_oof_not_official",
        "selection": "For each outer fold, candidate chosen on the other three folds only",
        "candidate_count": len(specs),
        "formulae": {
            "blend": "B6+C6+(1-w)*R6+w*R9.6",
            "op3_residual": "B6+C6+R6+g*(R9.6-R9.0)",
            "high_specialist": "blend + takeover*I(abs(R6)>=threshold)*(R6-blend)",
        },
        "source_code_sha256": _source_sha256(),
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "m9_op3": {"path": str(Path(args.m9_op3).resolve()), "sha256": sha256(Path(args.m9_op3))},
        "m9_base": {"path": str(Path(args.m9_base).resolve()), "sha256": sha256(Path(args.m9_base))},
        "m6_component_roots": [str(path) for path in roots],
        "limitations": [
            "Only four outer folds are available; selected weights may vary by fold.",
            "This is a local strict OOF proxy, not an organizer score.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    print(summary.to_string(index=False))
    print(json.dumps({"output": str(output), "candidate_count": len(specs)}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--m6-components", action="append", required=True)
    parser.add_argument("--m9-op3", required=True)
    parser.add_argument("--m9-base", required=True)
    parser.add_argument("--step", type=float, default=0.10)
    parser.add_argument("--blend-max", type=float, default=1.5)
    parser.add_argument("--residual-max", type=float, default=2.0)
    parser.add_argument("--specialist-base-weights", nargs="+", type=float, default=[0.8, 0.9, 1.0, 1.1, 1.2])
    parser.add_argument("--specialist-thresholds", nargs="+", type=float, default=[0.75, 1.0, 1.25])
    parser.add_argument("--specialist-takeovers", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=150815)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
