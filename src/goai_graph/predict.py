"""Inference and submission-contract checks for PPI graph checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from goai_baseline.audit import assert_allowed_inputs
from goai_baseline.config import load_config
from goai_baseline.schema import SAMPLE_ID, require_metadata_columns, require_unique_sample_ids
from goai_baseline.submission import verify_submission
from goai_baseline.train import resolve_device

from .config import load_graph_config
from .features import ConditionFeatureBuilder
from .graph import load_graph_bundle, sha256_file
from .model import StaticProteinGraphRegressor
from .train import predict_batches, select_adjacency


def load_graph_checkpoint(
    checkpoint_path: str | Path,
    graph_artifact: str | Path,
    device: torch.device,
) -> tuple[
    StaticProteinGraphRegressor,
    ConditionFeatureBuilder,
    torch.Tensor,
    list[str],
    np.ndarray,
    np.ndarray,
]:
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    artifact = Path(graph_artifact)
    expected_hash = str(payload["graph_artifact_sha256"])
    if sha256_file(artifact) != expected_hash:
        raise ValueError("Graph artifact hash does not match the checkpoint")
    bundle = load_graph_bundle(artifact)
    proteins = list(payload["proteins"])
    if proteins != bundle.proteins:
        raise ValueError("Checkpoint and graph artifact protein orders differ")
    model = StaticProteinGraphRegressor(**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    builder = ConditionFeatureBuilder.from_state_dict(payload["feature_state"])
    adjacency = select_adjacency(str(payload["variant"]), bundle, len(proteins), device)
    target_mean = np.asarray(payload["target_mean"], dtype=np.float32)
    target_scale = np.asarray(payload["target_scale"], dtype=np.float32)
    return model, builder, adjacency, proteins, target_mean, target_scale


def predict_test(
    graph_config_path: str | Path,
    run_dir: str | Path,
    output_csv: str | Path | None = None,
) -> Path:
    graph_config = load_graph_config(graph_config_path)
    baseline_config = load_config(graph_config.baseline_config)
    assert_allowed_inputs(baseline_config)
    run = Path(run_dir)
    checkpoint_path = run / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = resolve_device(graph_config.model.device)
    model, builder, adjacency, proteins, target_mean, target_scale = load_graph_checkpoint(
        checkpoint_path,
        graph_config.graph.artifact,
        device,
    )
    metadata = pd.read_csv(baseline_config.data.metadata_test, low_memory=False)
    require_metadata_columns(metadata)
    require_unique_sample_ids(metadata, "test metadata")
    metadata = metadata.set_index(SAMPLE_ID, verify_integrity=True)
    prediction = predict_batches(
        model,
        builder,
        adjacency,
        metadata,
        metadata.index,
        proteins,
        target_mean,
        target_scale,
        graph_config.model.batch_size,
        device,
    )
    if not np.isfinite(prediction.to_numpy()).all():
        raise ValueError("Model produced non-finite predictions")
    output = Path(output_csv) if output_csv is not None else run / "prediction.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction.index.name = SAMPLE_ID
    prediction.to_csv(output)
    report = verify_submission(output, baseline_config.data.metadata_test, proteins)
    with (output.parent / "prediction_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote submission: {output.resolve()}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions from a PPI graph checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()
    predict_test(args.config, args.run_dir, args.output_csv)


if __name__ == "__main__":
    main()
