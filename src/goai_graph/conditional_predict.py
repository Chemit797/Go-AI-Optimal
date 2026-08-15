"""Generate official-format predictions from a conditional graph checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from goai_baseline.config import load_config
from goai_baseline.preprocess import prepare_data

from .chemistry import load_chemical_features
from .conditional_features import ConditionalFeatureBuilder
from .conditional_model import ConditionalProteinGraphRegressor
from .conditional_train import _load_resource_table, _node_banks, predict_batches, select_adjacency
from .config import load_graph_config
from .graph import load_graph_bundle


def predict(config_path: str | Path, checkpoint_path: str | Path, output_path: str | Path) -> Path:
    graph_config = load_graph_config(config_path)
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    baseline_config = load_config(graph_config.baseline_config)
    data = prepare_data(baseline_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bundle = load_graph_bundle(graph_config.graph.artifact)
    device = torch.device("cuda" if graph_config.model.device == "auto" and torch.cuda.is_available() else "cpu")
    adjacency = select_adjacency(checkpoint["variant"], bundle, len(data.proteins), device)
    chemistry_payload = payload["chemistry"]
    entity_map = Path(config_path).parent.joinpath(chemistry_payload["entity_map"]).resolve()
    target_seed = Path(config_path).parent.joinpath(chemistry_payload["target_seed"]).resolve()
    strain_map = Path(config_path).parent.joinpath(chemistry_payload["strain_map"]).resolve()
    chemical_features = load_chemical_features(entity_map)
    builder = ConditionalFeatureBuilder(chemical_features).fit(data.metadata, data.train_ids)
    model = ConditionalProteinGraphRegressor(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    target_table = _load_resource_table(target_seed, "|")
    strain_table = _load_resource_table(strain_map, "\t")
    protein_mapping = pd.read_csv(graph_config.graph.protein_mapping, sep="\t", keep_default_na=False)
    test_metadata = pd.read_csv(baseline_config.data.metadata_test, low_memory=False).set_index("sample_ID")
    bank_metadata = pd.concat([data.metadata.reset_index(), test_metadata.reset_index()], ignore_index=True)
    chemical_bank, strain_bank, chemical_lookup, strain_lookup = _node_banks(bank_metadata, data.proteins, target_table, strain_table, protein_mapping)
    predictions = predict_batches(
        model, builder, adjacency, test_metadata, test_metadata.index, data.proteins,
        checkpoint["target_mean"], checkpoint["target_scale"], chemical_bank, strain_bank,
        chemical_lookup, strain_lookup, graph_config.model.batch_size, device,
        bool(checkpoint.get("node_injection", False)),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index_label="sample_ID", float_format="%.6g")
    with output.with_suffix(".contract.json").open("w", encoding="utf-8") as handle:
        json.dump({"rows": len(predictions), "proteins": len(data.proteins), "variant": checkpoint["variant"], "node_injection": bool(checkpoint.get("node_injection", False)), "sample_id_order": "metadata_test", "protein_order": "training_contract"}, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {len(predictions):,} rows x {len(data.proteins):,} proteins to {output.resolve()}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate conditional graph predictions")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predict(args.config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
