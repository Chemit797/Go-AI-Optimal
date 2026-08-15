from __future__ import annotations

import numpy as np
import torch

from goai_graph.graph import identity_adjacency, normalized_adjacency
from goai_graph.model import StaticProteinGraphRegressor


def test_graph_model_forward_and_parameter_matched_controls():
    device = torch.device("cpu")
    edge_index = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    edge_weight = np.asarray([0.8, 0.7], dtype=np.float32)
    graph = normalized_adjacency(4, edge_index, edge_weight, device)
    identity = identity_adjacency(4, device)
    model = StaticProteinGraphRegressor(
        condition_dim=5,
        n_proteins=4,
        hidden_dim=8,
        condition_hidden_dim=12,
        graph_layers=2,
        dropout=0.0,
    )
    inputs = torch.randn(3, 5)
    real_output = model(inputs, graph)
    no_graph_output = model(inputs, identity)

    assert real_output.shape == (3, 4)
    assert no_graph_output.shape == (3, 4)
    assert torch.isfinite(real_output).all()
    assert not torch.allclose(real_output, no_graph_output)
