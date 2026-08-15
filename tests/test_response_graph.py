from __future__ import annotations

import torch

from goai_response.train import _ppi_smoothness


def test_ppi_smoothness_is_zero_without_graph_and_positive_with_edges():
    response = torch.tensor([[0.0, 2.0, 1.0]])
    assert _ppi_smoothness(response, None).item() == 0.0
    edges = torch.tensor([[0, 1], [1, 2]])
    weights = torch.ones(2)
    assert _ppi_smoothness(response, (edges, weights)).item() > 0
