"""Build a task-specific STRING physical PPI graph and topology control."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from goai_baseline.config import load_config
from goai_baseline.preprocess import prepare_data

from .config import load_graph_config
from .graph import (
    degree_preserving_rewire,
    download_if_missing,
    graph_statistics,
    map_proteins_to_sgd,
    read_string_edges,
    save_graph_bundle,
    sha256_file,
    write_graph_manifest,
)


def build_graph(config_path: str | Path) -> Path:
    config = load_graph_config(config_path)
    baseline_config = load_config(config.baseline_config)
    data = prepare_data(baseline_config)
    sgd_path = download_if_missing(config.graph.sgd_url, config.graph.sgd_features)
    string_path = download_if_missing(config.graph.string_url, config.graph.string_physical_links)

    mapping = map_proteins_to_sgd(data.proteins, sgd_path)
    config.graph.protein_mapping.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(config.graph.protein_mapping, sep="\t", index=False)
    systematic_to_index = {
        row.systematic_name: int(row.protein_index)
        for row in mapping.itertuples()
        if row.mapping_status == "mapped"
    }
    edge_index, edge_weight = read_string_edges(
        string_path,
        systematic_to_index,
        config.graph.min_score,
    )
    rewired_edge_index = degree_preserving_rewire(
        edge_index,
        seed=config.model.seed,
        swaps_per_edge=config.graph.rewire_swaps_per_edge,
    )
    rng = np.random.default_rng(config.model.seed)
    rewired_edge_weight = edge_weight[rng.permutation(len(edge_weight))]
    save_graph_bundle(
        config.graph.artifact,
        data.proteins,
        mapping,
        edge_index,
        edge_weight,
        rewired_edge_index,
        rewired_edge_weight,
        config.graph.min_score,
    )
    resources = {
        "sgd_features": {
            "url": config.graph.sgd_url,
            "path": str(sgd_path),
            "sha256": sha256_file(sgd_path),
            "license": "SGD data are distributed under CC BY 4.0",
        },
        "string_physical_links": {
            "url": config.graph.string_url,
            "path": str(string_path),
            "sha256": sha256_file(string_path),
            "license": "STRING downloadable files are distributed under Creative Commons terms",
        },
    }
    real_stats = graph_statistics(len(data.proteins), edge_index)
    rewired_stats = graph_statistics(len(data.proteins), rewired_edge_index)
    write_graph_manifest(
        config.graph.manifest,
        resources=resources,
        mapping=mapping,
        min_score=config.graph.min_score,
        rewire_seed=config.model.seed,
        rewire_swaps_per_edge=config.graph.rewire_swaps_per_edge,
        real_stats=real_stats,
        rewired_stats=rewired_stats,
    )
    print(f"Mapped proteins: {mapping['mapping_status'].eq('mapped').sum():,}/{len(mapping):,}")
    print(f"Physical PPI edges: {edge_index.shape[1]:,}")
    print(f"Covered proteins: {real_stats['covered_nodes']:,}/{len(data.proteins):,}")
    print(f"Wrote graph artifact: {config.graph.artifact}")
    return config.graph.artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the task-specific physical PPI graph")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    build_graph(args.config)


if __name__ == "__main__":
    main()
