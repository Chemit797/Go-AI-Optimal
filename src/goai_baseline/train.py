"""Full-batch training for the fixed document MLP."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .audit import audit_inputs
from .config import BaselineConfig, load_config
from .evaluate import evaluate_predictor, write_evaluation
from .features import FeatureBuilder, VALID_VARIANTS, validate_variant
from .loss import masked_mse
from .manifest import write_manifest
from .model import ConditionMLP
from .preprocess import feature_contract, prepare_data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def default_run_dir(config: BaselineConfig, variant: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return config.runtime.runs_dir / f"{variant}-{timestamp}"


def predict_with_model(
    model: ConditionMLP,
    builder: FeatureBuilder,
    metadata: pd.DataFrame,
    sample_ids: pd.Index,
    proteins: list[str],
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    inputs = torch.from_numpy(builder.transform(metadata.loc[sample_ids])).to(device)
    with torch.no_grad():
        values = model(inputs).detach().cpu().numpy()
    return pd.DataFrame(values, index=sample_ids, columns=proteins)


def train_variant(config: BaselineConfig, variant: str, run_dir: str | Path | None = None) -> Path:
    validate_variant(variant)
    audit_inputs(config)
    set_seed(config.model.seed)
    device = resolve_device(config.model.device)
    data = prepare_data(config)
    output = Path(run_dir) if run_dir is not None else default_run_dir(config, variant)
    output.mkdir(parents=True, exist_ok=False)

    builder = FeatureBuilder(variant=variant, chemical_hash_dim=config.features.chemical_hash_dim)
    x_train = builder.fit_transform(data.metadata, data.y_log2, data.train_ids)
    y_train = data.y_log2.loc[data.train_ids].fillna(0.0).to_numpy(dtype=np.float32)
    mask_train = data.mask.loc[data.train_ids].to_numpy(dtype=np.float32)
    inputs = torch.from_numpy(x_train).to(device)
    targets = torch.from_numpy(y_train).to(device)
    masks = torch.from_numpy(mask_train).to(device)

    model = ConditionMLP(
        input_dim=x_train.shape[1],
        output_dim=len(data.proteins),
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.model.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = masked_mse(model(inputs), targets, masks)
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch, "masked_mse": float(loss.detach().cpu())})
        if epoch == 1 or epoch % 10 == 0 or epoch == config.model.epochs:
            print(f"epoch={epoch:03d} masked_mse={history[-1]['masked_mse']:.6f}")

    checkpoint = {
        "variant": variant,
        "model_state_dict": model.state_dict(),
        "model_kwargs": {
            "input_dim": x_train.shape[1],
            "output_dim": len(data.proteins),
            "hidden_dim": config.model.hidden_dim,
            "dropout": config.model.dropout,
        },
        "feature_state": builder.state_dict(),
        "proteins": data.proteins,
        "target_scale": "log2",
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    with (output / "feature_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_contract(data, config), handle, ensure_ascii=False, indent=2)
    with (output / "feature_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(builder.summary(), handle, ensure_ascii=False, indent=2)

    def predictor(ids: pd.Index) -> pd.DataFrame:
        return predict_with_model(model, builder, data.metadata, ids, data.proteins, device)

    report, protein_report = evaluate_predictor(data, predictor, variant)
    write_evaluation(output, report, protein_report)
    write_manifest(
        output / "manifest.json",
        config,
        {
            "variant": variant,
            "device": str(device),
            "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "feature_input_dim": int(x_train.shape[1]),
            "target_scale": "log2",
        },
    )
    print(report.to_string(index=False))
    print(f"Wrote run: {output.resolve()}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GOAI document MLP baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=VALID_VARIANTS, required=True)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    train_variant(load_config(args.config), args.variant, args.run_dir)


if __name__ == "__main__":
    main()
