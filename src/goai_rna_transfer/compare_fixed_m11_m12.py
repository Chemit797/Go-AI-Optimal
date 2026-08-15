"""Compare fixed M11 and M12 fusion formulas on the same strict S1 OOF rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import load_config, sha256, write_json
from .evaluate_nested_m6_m9_fusion import _response
from .evaluate_s1 import (
    PredictionPayload,
    PredictionRequest,
    build_bootstrap_sufficient_statistics,
    build_fold_train_context_reference,
    evaluate_prediction,
    load_aligned_prediction,
    load_s1_cache,
    paired_cluster_bootstrap,
    summarize_folds,
)
from .fuse_m6_m9_s1 import _average_component_seeds


def run(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    cache_path = Path(config["paths"]["private_cache"]) / "goai_s1_delta.npz"
    cache = load_s1_cache(cache_path, Path(config["paths"]["goai_metadata"]))
    context = build_fold_train_context_reference(cache)
    roots = [Path(value).expanduser().resolve() for value in args.m6_components]
    m6 = _average_component_seeds(roots, cache)
    m9 = load_aligned_prediction(
        PredictionRequest("m9_op3", "delta", [Path(args.m9_op3).resolve()]), cache
    ).values
    m9_base = load_aligned_prediction(
        PredictionRequest("m9_base", "delta", [Path(args.m9_base).resolve()]), cache
    ).values
    responses = {
        "M11_blend_w1.05": _response(
            "blend", 1.05, m6["response"], m9, m9_base
        ),
        "M12_high_w1.075_t0.5_g0.15": _response(
            "high_specialist",
            1.075,
            m6["response"],
            m9,
            m9_base,
            threshold=0.5,
            takeover=0.15,
        ),
    }
    fold_frames = []
    statistics = {}
    predictions = {}
    for label, response in responses.items():
        absolute = (m6["background_plus_calibration"] + response).astype(np.float32)
        predictions[label] = absolute
        payload = PredictionPayload(
            sample_ids=cache.sample_ids.copy(),
            proteins=cache.proteins.copy(),
            values=absolute,
            kind="absolute",
            source_files=[],
        )
        folds, _ = evaluate_prediction(label, payload, cache, context)
        fold_frames.append(folds)
        statistics[label] = build_bootstrap_sufficient_statistics(
            label, payload, cache, context
        )
    folds = pd.concat(fold_frames, ignore_index=True)
    summary = summarize_folds(folds)
    bootstrap = paired_cluster_bootstrap(
        statistics,
        [("M12_high_w1.075_t0.5_g0.15", "M11_blend_w1.05")],
        draws=args.bootstrap_draws,
        seed=args.bootstrap_seed,
    )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    folds.to_csv(output / "fold_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    bootstrap.to_csv(output / "paired_bootstrap.csv", index=False)
    for label, values in predictions.items():
        np.savez_compressed(
            output / f"{label}.npz",
            sample_ids=cache.sample_ids,
            proteins=cache.proteins,
            folds=cache.folds,
            pred_absolute=values,
        )
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {
        "status": "complete",
        "protocol": "fixed_m11_m12_same_oof_comparison_v1",
        "score_status": "local_strict_oof_not_official",
        "selection": "Both formulas fixed before this paired comparison",
        "formulae": {
            "M11_blend_w1.05": "B6+C6-0.05*R6+1.05*R9.6",
            "M12_high_w1.075_t0.5_g0.15": (
                "B6+C6+blend(1.075)+0.15*I(abs(R6)>=0.5)*(R6-blend)"
            ),
        },
        "source_code_sha256": source,
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "m9_op3": {"path": str(Path(args.m9_op3).resolve()), "sha256": sha256(Path(args.m9_op3))},
        "m9_base": {"path": str(Path(args.m9_base).resolve()), "sha256": sha256(Path(args.m9_base))},
        "m6_component_roots": [str(path) for path in roots],
    }
    write_json(output / "manifest.json", manifest)
    print(summary.to_string(index=False))
    print(bootstrap.to_string(index=False))
    print(json.dumps({"output": str(output)}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--m6-components", action="append", required=True)
    parser.add_argument("--m9-op3", required=True)
    parser.add_argument("--m9-base", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=150816)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
