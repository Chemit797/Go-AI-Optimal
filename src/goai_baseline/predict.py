"""Inference on official submission conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .audit import assert_allowed_inputs
from .config import load_config
from .features import FeatureBuilder
from .model import ConditionMLP
from .schema import SAMPLE_ID, require_metadata_columns, require_unique_sample_ids
from .submission import verify_submission
from .train import resolve_device


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[ConditionMLP, FeatureBuilder, list[str]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = ConditionMLP(**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    builder = FeatureBuilder.from_state_dict(payload["feature_state"])
    proteins = list(payload["proteins"])
    if payload.get("target_scale") != "log2":
        raise ValueError("Checkpoint target scale is not log2")
    return model, builder, proteins


def predict_test(config_path: str | Path, run_dir: str | Path, output_csv: str | Path | None = None) -> Path:
    config = load_config(config_path)
    assert_allowed_inputs(config)
    run = Path(run_dir)
    checkpoint_path = run / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = resolve_device(config.model.device)
    model, builder, proteins = load_checkpoint(checkpoint_path, device)
    metadata = pd.read_csv(config.data.metadata_test, low_memory=False)
    require_metadata_columns(metadata)
    require_unique_sample_ids(metadata, "test metadata")
    metadata = metadata.set_index(SAMPLE_ID, verify_integrity=True)
    inputs = torch.from_numpy(builder.transform(metadata)).to(device)
    with torch.no_grad():
        prediction = model(inputs).detach().cpu().numpy()
    if not np.isfinite(prediction).all():
        raise ValueError("Model produced non-finite predictions")
    output = Path(output_csv) if output_csv is not None else run / "prediction.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(prediction, index=metadata.index, columns=proteins)
    submission.index.name = SAMPLE_ID
    submission.to_csv(output)
    report = verify_submission(output, config.data.metadata_test, proteins)
    with (output.parent / "prediction_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote submission: {output.resolve()}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GOAI submission predictions")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()
    predict_test(args.config, args.run_dir, args.output_csv)


if __name__ == "__main__":
    main()
