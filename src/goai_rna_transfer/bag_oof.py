"""Persist an aligned equal-weight multi-seed delta OOF artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import load_config, sha256, write_json
from .evaluate_s1 import PredictionRequest, load_aligned_prediction, load_s1_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    cache_path = Path(config["paths"]["private_cache"]) / "goai_s1_delta.npz"
    cache = load_s1_cache(cache_path, Path(config["paths"]["goai_metadata"]))
    root = Path(config["paths"]["output_root"])
    paths = [root / "oof" / args.arm / f"seed_{seed}" for seed in args.seed]
    payload = load_aligned_prediction(
        PredictionRequest(args.arm, "delta", paths), cache
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "S1.npz"
    np.savez_compressed(
        prediction_path,
        sample_ids=cache.sample_ids,
        chemicals=cache.chemicals,
        proteins=cache.proteins,
        folds=cache.folds,
        pred_delta=payload.values.astype(np.float32),
        seeds=np.asarray(args.seed, dtype=np.int64),
    )
    write_json(
        output / "manifest.json",
        {
            "status": "complete",
            "prediction_kind": "delta",
            "aggregation": "equal-weight arithmetic mean of aligned complete OOF seeds",
            "arm": args.arm,
            "seeds": args.seed,
            "source_directories": [str(path) for path in paths],
            "source_prediction_sha256": [
                {
                    f"fold_{fold}": sha256(path / f"fold_{fold}" / "predictions.npz")
                    for fold in range(4)
                }
                for path in paths
            ],
            "n_samples": len(cache.sample_ids),
            "n_proteins": len(cache.proteins),
            "output": str(prediction_path),
            "output_sha256": sha256(prediction_path),
        },
    )
    print(json.dumps({"status": "complete", "output": str(prediction_path)}, indent=2))


if __name__ == "__main__":
    main()
