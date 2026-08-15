"""Evaluate one frozen scenario-specific blend on the four outer splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from goai_baseline.official_metrics import evaluate_prediction_set
from goai_baseline.preprocess import prepare_data
from goai_baseline.schema import SAMPLE_ID, SPLIT
from goai_response.config import load_response_config
from goai_response.predict import load_response_checkpoint
from goai_response.train import _predict


DEFAULT_WEIGHTS = {
    "val_chem_only": 0.15,
    "val_strain_only": 0.0,
    "val_both": 0.0,
    "val_time": 0.30,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(
    config_a_path: Path,
    config_b_path: Path,
    run_a: Path,
    run_b: Path,
    output_dir: Path,
    weights: dict[str, float],
) -> Path:
    config_a = load_response_config(config_a_path)
    config_b = load_response_config(config_b_path)
    if config_a.baseline.path != config_b.baseline.path:
        raise ValueError("Outer blend configs must share a baseline")
    data = prepare_data(config_a.baseline)
    device_a = torch.device(config_a.model.device)
    device_b = torch.device(config_b.model.device)
    checkpoint_a = run_a / "checkpoint.pt"
    checkpoint_b = run_b / "checkpoint.pt"
    model_a, builder_a, proteins_a, mean_a, scale_a = load_response_checkpoint(checkpoint_a, device_a)
    model_b, builder_b, proteins_b, mean_b, scale_b = load_response_checkpoint(checkpoint_b, device_b)
    if proteins_a != proteins_b or proteins_a != data.proteins:
        raise ValueError("Outer blend checkpoint protein contracts differ")

    rows: list[dict[str, float | int | str]] = []
    for split, weight_b in weights.items():
        ids = data.metadata.index[data.metadata[SPLIT].eq(split)]
        prediction_a = _predict(
            model_a, builder_a, data.metadata, ids, proteins_a, mean_a, scale_a,
            device_a, config_a.model.batch_size,
        )
        prediction_b = _predict(
            model_b, builder_b, data.metadata, ids, proteins_b, mean_b, scale_b,
            device_b, config_b.model.batch_size,
        )
        prediction = (1.0 - weight_b) * prediction_a + weight_b * prediction_b
        rows.append(
            {
                "split": split,
                "weight_b": weight_b,
                **evaluate_prediction_set(
                    data,
                    prediction,
                    ids,
                    split,
                    control_pool_ids=data.metadata.index,
                ),
            }
        )
    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "official_proxy_metrics.csv", index=False)
    manifest = {
        "protocol": "frozen_outer_confirmation_blend_v1",
        "candidate_a": {"run": str(run_a.resolve()), "checkpoint_sha256": _sha256(checkpoint_a)},
        "candidate_b": {"run": str(run_b.resolve()), "checkpoint_sha256": _sha256(checkpoint_b)},
        "weight_b_by_split": weights,
        "weights_selected_from_inner_oof_only": True,
        "outer_used_to_change_weights": False,
        "scorer_control_policy": "all released observed controls; controls are never model inputs or reference-fitting rows",
        "not_an_official_score": True,
    }
    with (output_dir / "confirmation_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(result.to_string(index=False))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen GOAI outer blend")
    parser.add_argument("--config-a", required=True)
    parser.add_argument("--config-b", required=True)
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    evaluate(
        Path(args.config_a), Path(args.config_b), Path(args.run_a), Path(args.run_b),
        Path(args.output_dir), DEFAULT_WEIGHTS,
    )


if __name__ == "__main__":
    main()
