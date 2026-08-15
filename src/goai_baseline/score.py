"""Score a baseline or saved MLP run with the published scoring proxy."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .audit import audit_inputs
from .config import load_config
from .controls import exact_control_predictions
from .evaluate import mean_prediction
from .official_metrics import evaluate_official_proxy
from .preprocess import prepare_data
from .schema import treatment_mask
from .train import predict_with_model, resolve_device
from .predict import load_checkpoint


def matched_control_prediction(data, ids: pd.Index) -> pd.DataFrame:
    prediction = pd.DataFrame(index=ids, columns=data.proteins, dtype=float)
    treatments = ids[treatment_mask(data.metadata.loc[ids]).to_numpy()]
    matched = exact_control_predictions(data.metadata, data.y_log2, treatments)
    prediction.loc[treatments] = matched.predictions
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Run published-scoring proxy on local validation splits")
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--baseline", choices=("b0_mean", "b1_matched_control"))
    source.add_argument("--run-dir")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    audit_inputs(config)
    data = prepare_data(config)
    if args.baseline == "b0_mean":
        predictor = lambda ids: mean_prediction(data, ids)
        label = "b0_mean"
    elif args.baseline == "b1_matched_control":
        predictor = lambda ids: matched_control_prediction(data, ids)
        label = "b1_matched_control"
    else:
        device = resolve_device(config.model.device)
        model, builder, proteins = load_checkpoint(Path(args.run_dir) / "checkpoint.pt", device)
        if proteins != data.proteins:
            raise ValueError("Checkpoint proteins do not match current feature contract")
        predictor = lambda ids: predict_with_model(model, builder, data.metadata, ids, proteins, device)
        label = Path(args.run_dir).name

    report = evaluate_official_proxy(data, predictor)
    output = Path(args.output) if args.output else config.runtime.runs_dir / label / "official_proxy_metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    print(report.to_string(index=False))
    print(f"Wrote official scoring proxy: {output.resolve()}")


if __name__ == "__main__":
    main()
