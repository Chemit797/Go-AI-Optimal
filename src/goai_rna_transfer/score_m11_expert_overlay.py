"""Select a frozen entity-expert residual scale on the M11 S1 OOF surface."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import load_config, sha256, write_json
from .evaluate_s1 import (
    PredictionPayload,
    build_bootstrap_sufficient_statistics,
    build_fold_train_context_reference,
    evaluate_prediction,
    load_s1_cache,
    paired_cluster_bootstrap,
    summarize_folds,
)
from .fuse_m6_m9_s1 import _fast_fold_metrics


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _load_absolute(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return (
            payload["sample_ids"].astype(str),
            payload["proteins"].astype(str),
            np.asarray(payload["pred_absolute"], dtype=np.float32),
        )


def _load_response_oof(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return (
            payload["sample_ids"].astype(str),
            payload["protein_ids"].astype(str),
            np.asarray(payload["values"], dtype=np.float32),
        )


def _align(
    sample_ids: np.ndarray,
    proteins: np.ndarray,
    values: np.ndarray,
    target_ids: np.ndarray,
    target_proteins: np.ndarray,
    label: str,
) -> np.ndarray:
    rows = pd.Index(sample_ids).get_indexer(target_ids)
    columns = pd.Index(proteins).get_indexer(target_proteins)
    if (rows < 0).any() or (columns < 0).any():
        raise ValueError(f"{label} does not cover the frozen S1 contract")
    aligned = values[rows][:, columns]
    if aligned.shape != (len(target_ids), len(target_proteins)):
        raise ValueError(f"{label} has an invalid aligned shape")
    if not np.isfinite(aligned).all():
        raise ValueError(f"{label} contains NaN or infinity")
    return aligned


def _payload(label: str, values: np.ndarray, cache) -> PredictionPayload:
    return PredictionPayload(
        sample_ids=cache.sample_ids.copy(),
        proteins=cache.proteins.copy(),
        values=values.astype(np.float32, copy=False),
        kind="absolute",
        source_files=[],
    )


def run(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    cache_path = Path(config["paths"]["private_cache"]) / "goai_s1_delta.npz"
    cache = load_s1_cache(cache_path, Path(config["paths"]["goai_metadata"]))
    context_reference = build_fold_train_context_reference(cache)

    base_path = Path(args.base_absolute).resolve()
    general_path = Path(args.general_oof).resolve()
    expert_path = Path(args.expert_oof).resolve()
    base_ids, base_proteins, base = _load_absolute(base_path)
    if not np.array_equal(base_ids, cache.sample_ids) or not np.array_equal(
        base_proteins, cache.proteins
    ):
        raise ValueError("Base M11 absolute prediction differs from the S1 cache contract")
    general_ids, general_proteins, general = _load_response_oof(general_path)
    expert_ids, expert_proteins, expert = _load_response_oof(expert_path)
    general = _align(
        general_ids,
        general_proteins,
        general,
        cache.sample_ids,
        cache.proteins,
        "general OOF",
    )
    expert = _align(
        expert_ids,
        expert_proteins,
        expert,
        cache.sample_ids,
        cache.proteins,
        "expert OOF",
    )
    residual = expert - general

    scales = [float(value) for value in args.scales.split(",")]
    if not scales or any(not np.isfinite(value) or value < 0.0 for value in scales):
        raise ValueError("Expert scales must be non-negative finite numbers")
    candidates = {
        f"expert_scale_{scale:g}": (base + scale * residual).astype(np.float32)
        for scale in scales
    }
    grid = pd.DataFrame(
        [
            {
                "model": label,
                "scale": scale,
                **_fast_fold_metrics(values, cache, context_reference),
            }
            for (label, values), scale in zip(candidates.items(), scales)
        ]
    )
    winner = grid.sort_values(
        ["fc_pcc", "context_residual_pcc", "high_effect_pcc"], ascending=False
    ).iloc[0]
    selected_label = str(winner["model"])
    selected_scale = float(winner["scale"])

    fold_frames = []
    statistics = {}
    finalist_labels = ["expert_scale_0", selected_label]
    finalist_labels = list(dict.fromkeys(finalist_labels))
    for label in finalist_labels:
        payload = _payload(label, candidates[label], cache)
        fold_frame, _ = evaluate_prediction(label, payload, cache, context_reference)
        fold_frames.append(fold_frame)
        statistics[label] = build_bootstrap_sufficient_statistics(
            label, payload, cache, context_reference
        )
    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    summary = summarize_folds(fold_metrics)
    if selected_label == "expert_scale_0":
        bootstrap = pd.DataFrame()
    else:
        bootstrap = paired_cluster_bootstrap(
            statistics,
            [(selected_label, "expert_scale_0")],
            draws=args.bootstrap_draws,
            seed=args.bootstrap_seed,
        )

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    grid.to_csv(output / "expert_scale_grid.csv", index=False)
    fold_metrics.to_csv(output / "finalist_fold_metrics.csv", index=False)
    summary.to_csv(output / "finalist_summary.csv", index=False)
    bootstrap.to_csv(output / "paired_bootstrap.csv", index=False)
    selected_path = output / "selected_s1_absolute.npz"
    np.savez_compressed(
        selected_path,
        sample_ids=cache.sample_ids,
        proteins=cache.proteins,
        folds=cache.folds,
        pred_absolute=candidates[selected_label],
        selected_label=np.asarray([selected_label]),
        selected_scale=np.asarray([selected_scale], dtype=np.float32),
    )
    manifest = {
        "status": "complete",
        "protocol": "m11_frozen_entity_expert_overlay_s1_v1",
        "score_status": "local_strict_oof_not_official",
        "selected_label": selected_label,
        "selected_scale": selected_scale,
        "source_code_sha256": _source_sha256(),
        "base_absolute": {"path": str(base_path), "sha256": sha256(base_path)},
        "general_oof": {"path": str(general_path), "sha256": sha256(general_path)},
        "expert_oof": {"path": str(expert_path), "sha256": sha256(expert_path)},
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "selected_prediction": {
            "path": str(selected_path),
            "sha256": sha256(selected_path),
        },
        "limitations": [
            "The expert residual uses model seed 42 while the M6/M9 base is a three-seed ensemble.",
            "Scale selection and reporting use the same frozen OOF surface during this score sprint.",
            "This is a local proxy, not an organizer PSS or leaderboard score.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    print(grid.to_string(index=False))
    print(json.dumps(manifest, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-absolute", required=True)
    parser.add_argument("--general-oof", required=True)
    parser.add_argument("--expert-oof", required=True)
    parser.add_argument("--scales", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=140815)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
