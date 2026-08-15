"""Train the structure-aware and condition-injected graph variants."""

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
from .chemistry import load_chemical_features
from .conditional_features import ConditionalFeatureBuilder, build_node_signal
from .conditional_model import ConditionalProteinGraphRegressor
from .config import load_graph_config
from .graph import GraphBundle, identity_adjacency, load_graph_bundle, normalized_adjacency, sha256_file

GRAPH_VARIANTS = ("no_graph", "real_ppi", "rewired_ppi")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_adjacency(variant: str, bundle: GraphBundle, n_proteins: int, device: torch.device) -> torch.Tensor:
    if variant == "no_graph":
        return identity_adjacency(n_proteins, device)
    if variant == "real_ppi":
        return normalized_adjacency(n_proteins, bundle.edge_index, bundle.edge_weight, device)
    if variant == "rewired_ppi":
        return normalized_adjacency(n_proteins, bundle.rewired_edge_index, bundle.rewired_edge_weight, device)
    raise ValueError(f"Unknown graph variant '{variant}'")


def target_statistics(y_train: pd.DataFrame, scale_floor: float) -> tuple[np.ndarray, np.ndarray]:
    values = y_train.to_numpy(dtype=np.float32)
    mean = np.nanmean(values, axis=0).astype(np.float32)
    scale = np.nanstd(values, axis=0).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Every retained protein must have finite training mean and scale")
    return mean, np.maximum(scale, scale_floor).astype(np.float32)


def _load_resource_table(path: Path, sep: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path, sep=sep, keep_default_na=False)
    # The empty strain template is intentionally harmless even when checked
    # out by tools that preserve the two-character ``\\t`` marker literally.
    if len(table.columns) == 1 and "\\t" in str(table.columns[0]):
        table = pd.read_csv(path, sep=r"\\t", engine="python", keep_default_na=False)
    return table


def _node_banks(metadata: pd.DataFrame, proteins: list[str], target_table: pd.DataFrame, strain_table: pd.DataFrame, protein_mapping: pd.DataFrame):
    chemical_names = sorted(metadata["perturbation_no_concentration"].astype(str).unique())
    strain_names = sorted(metadata["Strains"].astype(str).unique())
    chemical_meta = pd.DataFrame({"perturbation_no_concentration": chemical_names, "Strains": ["" for _ in chemical_names]})
    strain_meta = pd.DataFrame({"perturbation_no_concentration": ["" for _ in strain_names], "Strains": strain_names})
    chemical_bank = build_node_signal(chemical_meta, proteins, target_table, None, protein_mapping)
    strain_bank = build_node_signal(strain_meta, proteins, pd.DataFrame(columns=target_table.columns), strain_table, protein_mapping)
    chemical_lookup = {name: index for index, name in enumerate(chemical_names)}
    strain_lookup = {name: index for index, name in enumerate(strain_names)}
    return chemical_bank, strain_bank, chemical_lookup, strain_lookup


def _signals_for(metadata: pd.DataFrame, chemical_bank, strain_bank, chemical_lookup, strain_lookup) -> np.ndarray:
    chemical_indices = [chemical_lookup[str(value)] for value in metadata["perturbation_no_concentration"]]
    strain_indices = [strain_lookup[str(value)] for value in metadata["Strains"]]
    return chemical_bank[np.asarray(chemical_indices)] + strain_bank[np.asarray(strain_indices)]


def predict_batches(
    model: ConditionalProteinGraphRegressor,
    builder: ConditionalFeatureBuilder,
    adjacency: torch.Tensor,
    metadata: pd.DataFrame,
    sample_ids: pd.Index,
    proteins: list[str],
    target_mean: np.ndarray,
    target_scale: np.ndarray,
    chemical_bank,
    strain_bank,
    chemical_lookup,
    strain_lookup,
    batch_size: int,
    device: torch.device,
    node_injection: bool,
) -> pd.DataFrame:
    conditions = torch.from_numpy(builder.transform(metadata.loc[sample_ids]))
    node_signal = _signals_for(metadata.loc[sample_ids], chemical_bank, strain_bank, chemical_lookup, strain_lookup)
    if not node_injection:
        node_signal.fill(0.0)
    loader = DataLoader(TensorDataset(conditions, torch.from_numpy(node_signal)), batch_size=batch_size, shuffle=False)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_conditions, batch_nodes in loader:
            standardized = model(batch_conditions.to(device), batch_nodes.to(device), adjacency).cpu().numpy()
            outputs.append(target_mean[None, :] + target_scale[None, :] * standardized)
    values = np.concatenate(outputs, axis=0) if outputs else np.empty((0, len(proteins)), dtype=np.float32)
    return pd.DataFrame(values, index=sample_ids, columns=proteins)


def train_conditional_variant(config_path: str | Path, variant: str, node_injection: bool, run_dir: str | Path | None = None) -> Path:
    if variant not in GRAPH_VARIANTS:
        raise ValueError(f"variant must be one of {GRAPH_VARIANTS}")
    graph_config = load_graph_config(config_path)
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    baseline_config = load_config(graph_config.baseline_config)
    chemistry_payload = payload["chemistry"]
    entity_map = Path(config_path).parent.joinpath(chemistry_payload["entity_map"]).resolve()
    target_seed = Path(config_path).parent.joinpath(chemistry_payload["target_seed"]).resolve()
    strain_map = Path(config_path).parent.joinpath(chemistry_payload["strain_map"]).resolve()
    model_payload = payload["model"]
    audit_inputs(baseline_config)
    set_seed(graph_config.model.seed)
    device = resolve_device(graph_config.model.device)
    data = prepare_data(baseline_config)
    if not graph_config.graph.artifact.exists():
        build_graph(graph_config.path)
    bundle = load_graph_bundle(graph_config.graph.artifact)
    if bundle.proteins != data.proteins:
        raise ValueError("Graph artifact protein order does not match current protein contract")
    adjacency = select_adjacency(variant, bundle, len(data.proteins), device)

    chemical_features = load_chemical_features(entity_map)
    builder = ConditionalFeatureBuilder(chemical_features).fit(data.metadata, data.train_ids)
    x_train = builder.transform(data.metadata.loc[data.train_ids])
    target_table = _load_resource_table(target_seed, "|")
    strain_table = _load_resource_table(strain_map, "\t")
    protein_mapping = pd.read_csv(graph_config.graph.protein_mapping, sep="\t", keep_default_na=False)
    if target_table.empty:
        target_table = pd.DataFrame(columns=["chemical_raw_name", "systematic_name", "action", "weight", "evidence_tier"])
    chemical_bank, strain_bank, chemical_lookup, strain_lookup = _node_banks(data.metadata, data.proteins, target_table, strain_table, protein_mapping)
    train_nodes = _signals_for(data.metadata.loc[data.train_ids], chemical_bank, strain_bank, chemical_lookup, strain_lookup)
    if not node_injection:
        train_nodes.fill(0.0)

    target_mean, target_scale = target_statistics(data.y_log2.loc[data.train_ids], graph_config.model.target_scale_floor)
    raw_targets = data.y_log2.loc[data.train_ids].to_numpy(dtype=np.float32)
    masks = np.isfinite(raw_targets).astype(np.float32)
    standardized = np.nan_to_num((raw_targets - target_mean[None, :]) / target_scale[None, :], nan=0.0).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(train_nodes), torch.from_numpy(standardized), torch.from_numpy(masks))
    loader = DataLoader(dataset, batch_size=graph_config.model.batch_size, shuffle=True, generator=torch.Generator().manual_seed(graph_config.model.seed), pin_memory=device.type == "cuda")
    id_start, id_end = builder.chemical_id_slice or (0, 0)
    model = ConditionalProteinGraphRegressor(
        condition_dim=x_train.shape[1], n_proteins=len(data.proteins), hidden_dim=graph_config.model.hidden_dim,
        condition_hidden_dim=graph_config.model.condition_hidden_dim, graph_layers=graph_config.model.graph_layers,
        dropout=graph_config.model.dropout, condition_id_dropout=float(model_payload.get("condition_id_dropout", 0.0)),
        chemical_id_start=id_start, chemical_id_end=id_end,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=graph_config.model.learning_rate, weight_decay=graph_config.model.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, graph_config.model.epochs + 1):
        model.train(); weighted_loss = 0.0; observed_count = 0.0
        for conditions, nodes, targets, batch_masks in loader:
            conditions, nodes, targets, batch_masks = [value.to(device, non_blocking=True) for value in (conditions, nodes, targets, batch_masks)]
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(conditions, nodes, adjacency), targets, batch_masks)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0); optimizer.step()
            count = float(batch_masks.sum().detach().cpu()); weighted_loss += float(loss.detach().cpu()) * count; observed_count += count
        epoch_loss = weighted_loss / observed_count; scheduler.step(epoch_loss)
        history.append({"epoch": epoch, "standardized_masked_mse": epoch_loss, "learning_rate": float(optimizer.param_groups[0]["lr"])})
        if epoch == 1 or epoch % 5 == 0 or epoch == graph_config.model.epochs:
            print(f"variant={variant} node_injection={node_injection} epoch={epoch:03d} loss={epoch_loss:.6f}")

    output = Path(run_dir) if run_dir is not None else graph_config.runs_dir / f"{variant}-node{int(node_injection)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)
    torch.save({"variant": variant, "node_injection": node_injection, "model_state_dict": model.state_dict(), "model_kwargs": {"condition_dim": int(x_train.shape[1]), "n_proteins": len(data.proteins), "hidden_dim": graph_config.model.hidden_dim, "condition_hidden_dim": graph_config.model.condition_hidden_dim, "graph_layers": graph_config.model.graph_layers, "dropout": graph_config.model.dropout, "condition_id_dropout": float(model_payload.get("condition_id_dropout", 0.0)), "chemical_id_start": id_start, "chemical_id_end": id_end}, "feature_state": builder.state_dict(), "proteins": data.proteins, "target_mean": target_mean, "target_scale": target_scale, "graph_artifact": str(graph_config.graph.artifact), "graph_artifact_sha256": sha256_file(graph_config.graph.artifact)}, output / "checkpoint.pt")
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    with (output / "feature_contract.json").open("w", encoding="utf-8") as handle: json.dump(feature_contract(data, baseline_config), handle, ensure_ascii=False, indent=2)
    with (output / "condition_feature_summary.json").open("w", encoding="utf-8") as handle: json.dump(builder.summary(), handle, ensure_ascii=False, indent=2)
    with (output / "node_signal_summary.json").open("w", encoding="utf-8") as handle: json.dump({"node_injection": node_injection, "target_rows": int(len(target_table)), "target_chemicals": sorted(target_table["chemical_raw_name"].astype(str).unique().tolist()) if not target_table.empty else [], "strain_rows": int(len(strain_table))}, handle, ensure_ascii=False, indent=2)

    def predictor(ids: pd.Index) -> pd.DataFrame:
        return predict_batches(model, builder, adjacency, data.metadata, ids, data.proteins, target_mean, target_scale, chemical_bank, strain_bank, chemical_lookup, strain_lookup, graph_config.model.batch_size, device, node_injection)

    standard_report, protein_report = evaluate_predictor(data, predictor, f"conditional_{variant}_node{int(node_injection)}")
    write_evaluation(output, standard_report, protein_report)
    official_report = evaluate_official_proxy(data, predictor)
    official_report.to_csv(output / "official_proxy_metrics.csv", index=False)
    graph_manifest = {}
    if graph_config.graph.manifest.exists():
        with graph_config.graph.manifest.open("r", encoding="utf-8") as handle: graph_manifest = json.load(handle)
    write_manifest(output / "manifest.json", baseline_config, {"experiment": "ppi_graph_evolution_v2", "variant": variant, "node_injection": node_injection, "graph_manifest": graph_manifest, "chemical_map": str(entity_map), "chemical_map_resolved": int(sum(chemical_features.resolved.values())), "target_seed": str(target_seed), "strain_map": str(strain_map), "device": str(device), "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters()))})
    print(official_report.to_string(index=False)); print(f"Wrote run: {output.resolve()}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train structure-aware conditional PPI variants")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=GRAPH_VARIANTS, required=True)
    parser.add_argument("--node-injection", action="store_true")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    train_conditional_variant(args.config, args.variant, args.node_injection, args.run_dir)


if __name__ == "__main__":
    main()
