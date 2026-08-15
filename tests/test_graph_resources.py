from __future__ import annotations

import gzip
from collections import Counter

import numpy as np

from goai_graph.graph import (
    degree_preserving_rewire,
    graph_statistics,
    map_proteins_to_sgd,
    read_string_edges,
)


def test_sgd_mapping_and_string_graph_build(tmp_path):
    sgd = tmp_path / "SGD_features.tab"
    sgd.write_text(
        "S1\tORF\tVerified\tYAA001W\tGENE1\tALIAS1\tchromosome 1\n"
        "S2\tORF\tVerified\tYAA002W\tGENE2\t\tchromosome 1\n"
        "S3\tORF\tVerified\tYAA003W\tGENE3\t\tchromosome 1\n",
        encoding="utf-8",
    )
    mapping = map_proteins_to_sgd(["GENE1", "YAA002W", "ALIAS1", "UNKNOWN"], sgd)
    assert mapping["mapping_status"].tolist() == ["mapped", "mapped", "mapped", "unresolved"]
    assert mapping["mapping_method"].tolist() == ["standard", "systematic", "alias", "unresolved"]

    links = tmp_path / "physical.txt.gz"
    with gzip.open(links, "wt", encoding="utf-8") as handle:
        handle.write("protein1 protein2 combined_score\n")
        handle.write("4932.YAA001W 4932.YAA002W 800\n")
        handle.write("4932.YAA001W 4932.YAA003W 300\n")
    edge_index, weights = read_string_edges(
        links,
        {"YAA001W": 0, "YAA002W": 1, "YAA003W": 2},
        min_score=400,
    )
    assert edge_index.tolist() == [[0], [1]]
    assert np.allclose(weights, [0.8])
    assert graph_statistics(4, edge_index)["covered_nodes"] == 2


def _degree_sequence(edge_index: np.ndarray) -> Counter[int]:
    return Counter(int(node) for node in edge_index.reshape(-1))


def test_rewiring_preserves_degree_and_simple_edges():
    edge_index = np.asarray(
        [[0, 0, 1, 1, 2, 2, 3, 4], [1, 2, 3, 4, 3, 5, 4, 5]],
        dtype=np.int64,
    )
    rewired = degree_preserving_rewire(edge_index, seed=7, swaps_per_edge=2)
    edges = [tuple(edge) for edge in rewired.T.tolist()]
    assert rewired.shape == edge_index.shape
    assert _degree_sequence(rewired) == _degree_sequence(edge_index)
    assert all(a < b for a, b in edges)
    assert len(edges) == len(set(edges))
