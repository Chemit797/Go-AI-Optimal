"""Average multiple seeds within each loss family, then apply frozen scenario weights."""

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


def _seed_average(
    config,
    runs: list[Path],
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[dict[str, object]]]:
    if not runs:
        raise ValueError("At least one run is required per loss family")
    device = torch.device(config.model.device)
    total: np.ndarray | None = None
    proteins: list[str] | None = None
    records: list[dict[str, object]] = []
    for run in runs:
        checkpoint_path = run / "checkpoint.pt"
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("fit_scope") != "all_released_labeled_rows":
            raise ValueError(f"Final seed component is not an all-labeled refit: {checkpoint_path}")
        model, builder, current_proteins, mean, scale = load_response_checkpoint(checkpoint_path, device)
        if proteins is None:
            proteins = current_proteins
        elif current_proteins != proteins:
            raise ValueError("Seed components have different protein contracts")
        prediction = _predict(
            model,
            builder,
            metadata,
            metadata.index,
            current_proteins,
            mean,
            scale,
            device,
            config.model.batch_size,
        ).to_numpy(dtype=np.float64)
        total = prediction if total is None else total + prediction
        records.append(
            {
                "run": str(run.resolve()),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "fit_sample_count": int(payload.get("fit_sample_count", 0)),
            }
        )
        del model, prediction
    assert total is not None and proteins is not None
    return total / len(runs), proteins, records


def predict(
    config_a_path: Path,
    config_b_path: Path,
    runs_a: list[Path],
    runs_b: list[Path],
    output_csv: Path,
    weights: dict[str, float],
) -> Path:
    config_a = load_response_config(config_a_path)
    config_b = load_response_config(config_b_path)
    if config_a.baseline.path != config_b.baseline.path:
        raise ValueError("Seed ensemble configs must share a baseline")
    metadata_path = config_a.baseline.data.metadata_test
    metadata = pd.read_csv(metadata_path, low_memory=False)
    require_metadata_columns(metadata)
    require_unique_sample_ids(metadata, "test metadata")
    unknown = sorted(set(metadata[SPLIT].astype(str)) - set(weights))
    if unknown:
        raise ValueError(f"No frozen weight for test splits: {unknown}")
    metadata = metadata.set_index(SAMPLE_ID, verify_integrity=True)

    average_a, proteins_a, records_a = _seed_average(config_a, runs_a, metadata)
    average_b, proteins_b, records_b = _seed_average(config_b, runs_b, metadata)
    if proteins_a != proteins_b:
        raise ValueError("Loss-family protein contracts differ")
    row_weights = metadata[SPLIT].astype(str).map(weights).to_numpy(dtype=np.float64)[:, None]
    values = (1.0 - row_weights) * average_a + row_weights * average_b
    if not np.isfinite(values).all():
        raise ValueError("Seed ensemble produced non-finite predictions")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(values.astype(np.float32), index=metadata.index, columns=proteins_a)
    submission.index.name = SAMPLE_ID
    submission.to_csv(output_csv)
    report = verify_submission(output_csv, metadata_path, proteins_a)
    report.update(
        {
            "model": "three_seed_average_then_scenario_specific_mse_huber_blend",
            "seeds_per_loss_family": len(runs_a),
            "weight_b_by_split": weights,
            "family_a": records_a,
            "family_b": records_b,
            "prediction_sha256": _sha256(output_csv),
            "official_submission_not_performed": True,
        }
    )
    with (output_csv.parent / "prediction_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-seed GOAI test ensemble")
    parser.add_argument("--config-a", required=True)
    parser.add_argument("--config-b", required=True)
    parser.add_argument("--runs-a", nargs="+", required=True)
    parser.add_argument("--runs-b", nargs="+", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    if len(args.runs_a) != len(args.runs_b):
        raise ValueError("Loss families must use the same number of seeds")
    predict(
        Path(args.config_a),
        Path(args.config_b),
        [Path(value) for value in args.runs_a],
        [Path(value) for value in args.runs_b],
        Path(args.output_csv),
        DEFAULT_WEIGHTS,
    )


if __name__ == "__main__":
    main()
