"""Train and evaluate no-graph, physical-PPI, and rewired-PPI models."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from goai_baseline.audit import audit_inputs
from goai_baseline.config import load_config
from goai_baseline.evaluate import evaluate_predictor, write_evaluation
from goai_baseline.loss import masked_mse
from goai_baseline.manifest import write_manifest
from goai_baseline.official_metrics import evaluate_official_proxy
from goai_baseline.preprocess import feature_contract, prepare_data
from goai_baseline.train import resolve_device

from .build_graph import build_graph
from .config import GraphExperimentConfig, load_graph_config
from .features import ConditionFeatureBuilder
from .graph import (
    GraphBundle,
    identity_adjacency,
    load_graph_bundle,
    normalized_adjacency,
    sha256_file,
)
from .model import StaticProteinGraphRegressor


GRAPH_VARIANTS = ("no_graph", "real_ppi", "rewired_ppi")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_adjacency(
    variant: str,
    bundle: GraphBundle,
    n_proteins: int,
    device: torch.device,
) -> torch.Tensor:
    if variant == "no_graph":
        return identity_adjacency(n_proteins, device)
    if variant == "real_ppi":
        return normalized_adjacency(n_proteins, bundle.edge_index, bundle.edge_weight, device)
    if variant == "rewired_ppi":
        return normalized_adjacency(
            n_proteins,
            bundle.rewired_edge_index,
            bundle.rewired_edge_weight,
            device,
        )
    raise ValueError(f"Unknown graph variant '{variant}'")


def target_statistics(y_train: pd.DataFrame, scale_floor: float) -> tuple[np.ndarray, np.ndarray]:
    if scale_floor <= 0:
        raise ValueError("target_scale_floor must be positive")
    values = y_train.to_numpy(dtype=np.float32)
    mean = np.nanmean(values, axis=0).astype(np.float32)
    scale = np.nanstd(values, axis=0).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Every retained protein must have finite training mean and scale")
    return mean, np.maximum(scale, scale_floor).astype(np.float32)


def predict_batches(
    model: StaticProteinGraphRegressor,
    builder: ConditionFeatureBuilder,
    adjacency: torch.Tensor,
    metadata: pd.DataFrame,
    sample_ids: pd.Index,
    proteins: list[str],
    target_mean: np.ndarray,
    target_scale: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    features = torch.from_numpy(builder.transform(metadata.loc[sample_ids]))
    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (conditions,) in loader:
            standardized = model(conditions.to(device), adjacency).cpu().numpy()
            outputs.append(target_mean[None, :] + target_scale[None, :] * standardized)
    values = np.concatenate(outputs, axis=0) if outputs else np.empty((0, len(proteins)), dtype=np.float32)
    return pd.DataFrame(values, index=sample_ids, columns=proteins)


def _default_run_dir(config: GraphExperimentConfig, variant: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return config.runs_dir / f"{variant}-{timestamp}"


def train_graph_variant(
    config_path: str | Path,
    variant: str,
    run_dir: str | Path | None = None,
) -> Path:
    if variant not in GRAPH_VARIANTS:
        raise ValueError(f"variant must be one of {GRAPH_VARIANTS}")
    graph_config = load_graph_config(config_path)
    baseline_config = load_config(graph_config.baseline_config)
    audit_inputs(baseline_config)
    set_seed(graph_config.model.seed)
    device = resolve_device(graph_config.model.device)
    data = prepare_data(baseline_config)
    if not graph_config.graph.artifact.exists():
        build_graph(graph_config.path)
    bundle = load_graph_bundle(graph_config.graph.artifact)
    if bundle.proteins != data.proteins:
        raise ValueError("Graph artifact protein order does not match the current feature contract")
    adjacency = select_adjacency(variant, bundle, len(data.proteins), device)

    builder = ConditionFeatureBuilder()
    x_train = builder.fit_transform(data.metadata, data.train_ids)
    target_mean, target_scale = target_statistics(
        data.y_log2.loc[data.train_ids],
        graph_config.model.target_scale_floor,
    )
    raw_targets = data.y_log2.loc[data.train_ids].to_numpy(dtype=np.float32)
    target_mask = np.isfinite(raw_targets).astype(np.float32)
    standardized = (raw_targets - target_mean[None, :]) / target_scale[None, :]
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(standardized),
        torch.from_numpy(target_mask),
    )
    generator = torch.Generator().manual_seed(graph_config.model.seed)
    loader = DataLoader(
        dataset,
        batch_size=graph_config.model.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    model = StaticProteinGraphRegressor(
        condition_dim=x_train.shape[1],
        n_proteins=len(data.proteins),
        hidden_dim=graph_config.model.hidden_dim,
        condition_hidden_dim=graph_config.model.condition_hidden_dim,
        graph_layers=graph_config.model.graph_layers,
        dropout=graph_config.model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=graph_config.model.learning_rate,
        weight_decay=graph_config.model.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, graph_config.model.epochs + 1):
        model.train()
        weighted_loss = 0.0
        observed_count = 0.0
        for conditions, targets, masks in loader:
            conditions = conditions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(conditions, adjacency), targets, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            count = float(masks.sum().detach().cpu())
            weighted_loss += float(loss.detach().cpu()) * count
            observed_count += count
        epoch_loss = weighted_loss / observed_count
        scheduler.step(epoch_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "standardized_masked_mse": epoch_loss, "learning_rate": learning_rate})
        if epoch == 1 or epoch % 5 == 0 or epoch == graph_config.model.epochs:
            print(f"variant={variant} epoch={epoch:03d} loss={epoch_loss:.6f} lr={learning_rate:.2e}")

    output = Path(run_dir) if run_dir is not None else _default_run_dir(graph_config, variant)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = {
        "variant": variant,
        "model_state_dict": model.state_dict(),
        "model_kwargs": {
            "condition_dim": int(x_train.shape[1]),
            "n_proteins": len(data.proteins),
            "hidden_dim": graph_config.model.hidden_dim,
            "condition_hidden_dim": graph_config.model.condition_hidden_dim,
            "graph_layers": graph_config.model.graph_layers,
            "dropout": graph_config.model.dropout,
        },
        "feature_state": builder.state_dict(),
        "proteins": data.proteins,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "target_scale_name": "per-protein standardized log2 residual",
        "graph_artifact": str(graph_config.graph.artifact),
        "graph_artifact_sha256": sha256_file(graph_config.graph.artifact),
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    with (output / "feature_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_contract(data, baseline_config), handle, ensure_ascii=False, indent=2)
    with (output / "condition_feature_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(builder.summary(), handle, ensure_ascii=False, indent=2)

    def predictor(ids: pd.Index) -> pd.DataFrame:
        return predict_batches(
            model,
            builder,
            adjacency,
            data.metadata,
            ids,
            data.proteins,
            target_mean,
            target_scale,
            graph_config.model.batch_size,
            device,
        )

    standard_report, protein_report = evaluate_predictor(data, predictor, f"ppi_{variant}")
    write_evaluation(output, standard_report, protein_report)
    official_report = evaluate_official_proxy(data, predictor)
    official_report.to_csv(output / "official_proxy_metrics.csv", index=False)
    with graph_config.path.open("r", encoding="utf-8") as handle:
        graph_config_payload = yaml.safe_load(handle)
    graph_manifest = {}
    if graph_config.graph.manifest.exists():
        with graph_config.graph.manifest.open("r", encoding="utf-8") as handle:
            graph_manifest = json.load(handle)
    write_manifest(
        output / "manifest.json",
        baseline_config,
        {
            "experiment": "ppi_graph_mvp_v1",
            "variant": variant,
            "graph_config": graph_config_payload,
            "graph_manifest": graph_manifest,
            "device": str(device),
            "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "condition_input_dim": int(x_train.shape[1]),
            "target_training_scale": "per-protein standardized log2 residual",
        },
    )
    print(official_report.to_string(index=False))
    print(f"Wrote run: {output.resolve()}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PPI graph MVP and topology controls")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=GRAPH_VARIANTS, required=True)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    train_graph_variant(args.config, args.variant, args.run_dir)


if __name__ == "__main__":
    main()
