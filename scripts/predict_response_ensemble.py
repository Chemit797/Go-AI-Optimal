"""Generate and validate the frozen scenario-specific GOAI test ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from goai_baseline.schema import SAMPLE_ID, SPLIT, require_metadata_columns, require_unique_sample_ids
from goai_baseline.submission import verify_submission
from goai_response.config import load_response_config
from goai_response.predict import load_response_checkpoint
from goai_response.train import _predict


DEFAULT_WEIGHTS = {
    "test_chem_only": 0.15,
    "test_strain_only": 0.0,
    "test_both": 0.0,
    "test_time": 0.30,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predict(
    config_a_path: Path,
    config_b_path: Path,
    run_a: Path,
    run_b: Path,
    output_csv: Path,
    weights: dict[str, float],
) -> Path:
    config_a = load_response_config(config_a_path)
    config_b = load_response_config(config_b_path)
    if config_a.baseline.path != config_b.baseline.path:
        raise ValueError("Ensemble configs must share a baseline")
    metadata_path = config_a.baseline.data.metadata_test
    metadata = pd.read_csv(metadata_path, low_memory=False)
    require_metadata_columns(metadata)
    require_unique_sample_ids(metadata, "test metadata")
    unknown = sorted(set(metadata[SPLIT].astype(str)) - set(weights))
    if unknown:
        raise ValueError(f"No frozen ensemble weight for test splits: {unknown}")
    metadata = metadata.set_index(SAMPLE_ID, verify_integrity=True)

    checkpoint_a = run_a / "checkpoint.pt"
    checkpoint_b = run_b / "checkpoint.pt"
    device_a = torch.device(config_a.model.device)
    device_b = torch.device(config_b.model.device)
    model_a, builder_a, proteins_a, mean_a, scale_a = load_response_checkpoint(checkpoint_a, device_a)
    model_b, builder_b, proteins_b, mean_b, scale_b = load_response_checkpoint(checkpoint_b, device_b)
    if proteins_a != proteins_b:
        raise ValueError("Ensemble checkpoint protein contracts differ")
    prediction_a = _predict(
        model_a, builder_a, metadata, metadata.index, proteins_a, mean_a, scale_a,
        device_a, config_a.model.batch_size,
    )
    prediction_b = _predict(
        model_b, builder_b, metadata, metadata.index, proteins_b, mean_b, scale_b,
        device_b, config_b.model.batch_size,
    )
    row_weights = metadata[SPLIT].astype(str).map(weights).to_numpy(dtype=np.float32)[:, None]
    values = (1.0 - row_weights) * prediction_a.to_numpy(dtype=np.float32) + row_weights * prediction_b.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Ensemble produced non-finite predictions")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(values, index=metadata.index, columns=proteins_a)
    submission.index.name = SAMPLE_ID
    submission.to_csv(output_csv)
    report = verify_submission(output_csv, metadata_path, proteins_a)
    report.update(
        {
            "model": "scenario_specific_mse_huber_ensemble",
            "weight_b_by_split": weights,
            "candidate_a_checkpoint_sha256": _sha256(checkpoint_a),
            "candidate_b_checkpoint_sha256": _sha256(checkpoint_b),
            "candidate_a_fit_scope": torch.load(checkpoint_a, map_location="cpu", weights_only=False).get("fit_scope", "unknown"),
            "candidate_b_fit_scope": torch.load(checkpoint_b, map_location="cpu", weights_only=False).get("fit_scope", "unknown"),
            "official_submission_not_performed": True,
        }
    )
    with (output_csv.parent / "prediction_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen GOAI ensemble predictions")
    parser.add_argument("--config-a", required=True)
    parser.add_argument("--config-b", required=True)
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    predict(
        Path(args.config_a), Path(args.config_b), Path(args.run_a), Path(args.run_b),
        Path(args.output_csv), DEFAULT_WEIGHTS,
    )


if __name__ == "__main__":
    main()
