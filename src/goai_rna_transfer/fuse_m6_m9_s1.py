"""Search complete-model M6/M9 response fusions on the frozen S1 OOF surface."""

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
    PredictionRequest,
    _pearson,
    build_bootstrap_sufficient_statistics,
    build_fold_train_context_reference,
    evaluate_prediction,
    load_aligned_prediction,
    load_s1_cache,
    paired_cluster_bootstrap,
    summarize_folds,
)


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _load_component_seed(root: Path, cache) -> Dict[str, np.ndarray]:
    files = sorted((root / "folds").glob("S1_fold_*/components.npz"))
    if len(files) != 4:
        raise ValueError(f"Expected four S1 component folds under {root}, found {len(files)}")
    result = {
        "background_plus_calibration": np.empty(cache.true_delta.shape, dtype=np.float32),
        "response": np.empty(cache.true_delta.shape, dtype=np.float32),
        "final": np.empty(cache.true_delta.shape, dtype=np.float32),
    }
    counts = np.zeros(len(cache.sample_ids), dtype=np.int8)
    cache_rows = pd.Index(cache.sample_ids)
    for path in files:
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "sample_ids",
                "protein_ids",
                "background_plus_calibration",
                "response",
                "final",
                "is_treatment",
            }
            missing = sorted(required.difference(payload.files))
            if missing:
                raise ValueError(f"{path} lacks component arrays: {missing}")
            sample_ids = payload["sample_ids"].astype(str)
            proteins = payload["protein_ids"].astype(str)
            rows = cache_rows.get_indexer(sample_ids)
            if (rows < 0).any() or len(set(sample_ids.tolist())) != len(sample_ids):
                raise ValueError(f"{path} has invalid or duplicate S1 sample IDs")
            protein_rows = pd.Index(proteins).get_indexer(cache.proteins)
            if (protein_rows < 0).any() or len(proteins) != len(cache.proteins):
                raise ValueError(f"{path} protein contract differs from the frozen S1 cache")
            treatment = np.asarray(payload["is_treatment"], dtype=np.float32)
            if treatment.shape != (len(rows), 1) or not np.all(treatment == 1.0):
                raise ValueError(f"S1 component fold contains a non-treatment row: {path}")
            for name in result:
                values = np.asarray(payload[name], dtype=np.float32)[:, protein_rows]
                if values.shape != (len(rows), len(cache.proteins)):
                    raise ValueError(f"{path}:{name} has an invalid shape")
                result[name][rows] = values
            counts[rows] += 1
    if not np.all(counts == 1):
        raise ValueError(f"S1 component folds under {root} are not disjoint and complete")
    error = float(
        np.max(
            np.abs(
                result["background_plus_calibration"]
                + result["response"]
                - result["final"]
            )
        )
    )
    if error > 5e-5:
        raise ValueError(f"M6 component reconstruction failed for {root}: {error}")
    return result


def _average_component_seeds(roots: Iterable[Path], cache) -> Dict[str, np.ndarray]:
    roots = list(roots)
    if not roots:
        raise ValueError("At least one M6 component root is required")
    total = {
        name: np.zeros(cache.true_delta.shape, dtype=np.float64)
        for name in ("background_plus_calibration", "response", "final")
    }
    for root in roots:
        current = _load_component_seed(root, cache)
        for name in total:
            total[name] += current[name]
    return {name: (values / len(roots)).astype(np.float32) for name, values in total.items()}


def _fast_fold_metrics(
    absolute: np.ndarray,
    cache,
    context_reference: np.ndarray,
) -> Mapping[str, float]:
    rows = []
    predicted_delta = np.where(
        cache.matched_control_mask,
        absolute - cache.matched_control,
        np.nan,
    ).astype(np.float32)
    for fold in range(4):
        selected = cache.folds == fold
        actual = cache.true_delta[selected]
        predicted = predicted_delta[selected]
        mask = cache.delta_mask[selected] & np.isfinite(predicted)
        context = context_reference[selected]
        context_mask = mask & np.isfinite(context)
        high_true = mask & (np.abs(actual) > 1.0)
        high_pred = mask & (np.abs(predicted) > 1.0)
        true_positive = high_true & high_pred & (
            np.sign(predicted) == np.sign(actual)
        )
        denominator = float(high_pred.sum() + high_true.sum())
        rows.append(
            {
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
    return {
        name: float(np.nanmean([row[name] for row in rows])) for name in rows[0]
    }


def _payload(label: str, values: np.ndarray, cache) -> PredictionPayload:
    if values.shape != cache.true_delta.shape or not np.isfinite(values).all():
        raise ValueError(f"Candidate {label} has an invalid absolute prediction matrix")
    return PredictionPayload(
        sample_ids=cache.sample_ids.copy(),
        proteins=cache.proteins.copy(),
        values=values.astype(np.float32, copy=False),
        kind="absolute",
        source_files=[],
    )


def _candidate_absolute(
    family: str,
    weight: float,
    base: np.ndarray,
    m6_response: np.ndarray,
    m9_response: np.ndarray,
    m9_base_response: np.ndarray,
) -> np.ndarray:
    if family == "blend":
        response = (1.0 - weight) * m6_response + weight * m9_response
    elif family == "op3_residual":
        response = m6_response + weight * (m9_response - m9_base_response)
    else:
        raise ValueError(f"Unknown fusion family: {family}")
    return (base + response).astype(np.float32)


def run(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    cache_path = Path(config["paths"]["private_cache"]) / "goai_s1_delta.npz"
    cache = load_s1_cache(cache_path, Path(config["paths"]["goai_metadata"]))
    context_reference = build_fold_train_context_reference(cache)
    component_roots = [Path(path).expanduser().resolve() for path in args.m6_components]
    m6 = _average_component_seeds(component_roots, cache)
    m9 = load_aligned_prediction(
        PredictionRequest("m9_op3", "delta", [Path(args.m9_op3).resolve()]), cache
    ).values
    m9_base = load_aligned_prediction(
        PredictionRequest("m9_base", "delta", [Path(args.m9_base).resolve()]), cache
    ).values

    grids = {
        "blend": np.round(np.arange(0.0, args.blend_max + 1e-9, args.step), 8),
        "op3_residual": np.round(
            np.arange(0.0, args.residual_max + 1e-9, args.step), 8
        ),
    }
    grid_rows = []
    for family, weights in grids.items():
        for weight in weights:
            absolute = _candidate_absolute(
                family,
                float(weight),
                m6["background_plus_calibration"],
                m6["response"],
                m9,
                m9_base,
            )
            grid_rows.append(
                {
                    "family": family,
                    "weight": float(weight),
                    **_fast_fold_metrics(absolute, cache, context_reference),
                }
            )
    grid = pd.DataFrame(grid_rows)
    winners = {
        family: group.sort_values(
            ["fc_pcc", "context_residual_pcc", "high_effect_pcc"],
            ascending=False,
        ).iloc[0]
        for family, group in grid.groupby("family", sort=False)
    }

    candidates: Dict[str, np.ndarray] = {
        "M6_core": m6["final"],
        "M9_replace": _candidate_absolute(
            "blend",
            1.0,
            m6["background_plus_calibration"],
            m6["response"],
            m9,
            m9_base,
        ),
    }
    for family, winner in winners.items():
        weight = float(winner["weight"])
        candidates[f"{family}_w{weight:g}"] = _candidate_absolute(
            family,
            weight,
            m6["background_plus_calibration"],
            m6["response"],
            m9,
            m9_base,
        )

    fold_frames = []
    statistics = {}
    for label, absolute in candidates.items():
        payload = _payload(label, absolute, cache)
        fold_frame, _ = evaluate_prediction(label, payload, cache, context_reference)
        fold_frames.append(fold_frame)
        statistics[label] = build_bootstrap_sufficient_statistics(
            label, payload, cache, context_reference
        )
    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    summary = summarize_folds(fold_metrics)
    selected_label = str(
        summary.sort_values(
            ["fc_pcc_mean", "context_residual_pcc_mean", "high_effect_pcc_mean"],
            ascending=False,
        ).iloc[0]["model"]
    )
    comparisons = [
        (label, "M6_core") for label in candidates if label != "M6_core"
    ]
    bootstrap = paired_cluster_bootstrap(
        statistics, comparisons, draws=args.bootstrap_draws, seed=args.bootstrap_seed
    )

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    grid.to_csv(output / "weight_grid_metrics.csv", index=False)
    fold_metrics.to_csv(output / "finalist_fold_metrics.csv", index=False)
    summary.to_csv(output / "finalist_summary.csv", index=False)
    bootstrap.to_csv(output / "paired_bootstrap.csv", index=False)
    selected_path = output / "selected_s1_absolute.npz"
    np.savez_compressed(
        selected_path,
        sample_ids=cache.sample_ids,
        proteins=cache.proteins,
        folds=cache.folds,
        pred_absolute=candidates[selected_label].astype(np.float32),
        selected_label=np.asarray([selected_label]),
    )
    manifest = {
        "status": "complete",
        "protocol": "m6_m9_complete_response_fusion_s1_v1",
        "score_status": "local_strict_oof_not_official",
        "selection_metric": "four-fold macro fc_pcc; context then high-effect PCC tie-break",
        "selected_label": selected_label,
        "selected_prediction": str(selected_path),
        "selected_prediction_sha256": sha256(selected_path),
        "source_code_sha256": _source_sha256(),
        "m6_component_roots": [str(path) for path in component_roots],
        "m6_component_sha256": [
            {
                path.parent.name: sha256(path)
                for path in sorted((root / "folds").glob("S1_fold_*/components.npz"))
            }
            for root in component_roots
        ],
        "m9_op3": {"path": str(Path(args.m9_op3).resolve()), "sha256": sha256(Path(args.m9_op3))},
        "m9_base": {"path": str(Path(args.m9_base).resolve()), "sha256": sha256(Path(args.m9_base))},
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "weight_grid": {
            "step": args.step,
            "blend_max": args.blend_max,
            "residual_max": args.residual_max,
        },
        "formulae": {
            "M6_core": "B6+C6+R6",
            "blend": "B6+C6+(1-w)*R6+w*R9.6",
            "op3_residual": "B6+C6+R6+g*(R9.6-R9.0)",
        },
        "limitations": [
            "Weight selection and reporting use the same frozen OOF surface during this score sprint.",
            "This is a local proxy, not an organizer PSS or leaderboard score.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    print(summary.to_string(index=False))
    print(json.dumps({"selected_label": selected_label, "output": str(output)}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--m6-components", action="append", required=True)
    parser.add_argument("--m9-op3", required=True)
    parser.add_argument("--m9-base", required=True)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--blend-max", type=float, default=1.5)
    parser.add_argument("--residual-max", type=float, default=2.0)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=140815)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
