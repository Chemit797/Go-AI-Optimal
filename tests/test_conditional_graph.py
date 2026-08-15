from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from goai_graph.conditional_features import build_node_signal
from goai_graph.conditional_model import sparse_propagate_batch
from goai_graph.graph import identity_adjacency


def test_target_signal_uses_sgd_to_output_column_mapping():
    metadata = pd.DataFrame(
        {"perturbation_no_concentration": ["Rapamycin", "Water"], "Strains": ["BAH", "BAH"]}
    )
    proteins = ["TOR1", "TOR2", "FPR1", "OTHER"]
    protein_mapping = pd.DataFrame(
        {
            "protein_index": [0, 1, 2, 3],
            "raw_name": proteins,
            "systematic_name": ["YJR066W", "YKL203C", "YNL135C", ""],
        }
    )
    targets = pd.DataFrame(
        {
            "chemical_raw_name": ["Rapamycin", "Rapamycin", "Rapamycin"],
            "systematic_name": ["YJR066W", "YKL203C", "YNL135C"],
            "action": ["inhibition", "inhibition", "binding"],
            "weight": [1.0, 1.0, 0.5],
        }
    )
    signal = build_node_signal(metadata, proteins, targets, protein_mapping=protein_mapping)
    assert signal.shape == (2, 4, 3)
    assert np.all(signal[0, :3, 2] == 1.0)
    assert np.all(signal[0, :2, 0] < 0.0)
    assert np.allclose(signal[1], 0.0)


def test_identity_graph_preserves_batched_states():
    states = torch.randn(3, 4, 5)
    adjacency = identity_adjacency(4, torch.device("cpu"))
    assert torch.allclose(sparse_propagate_batch(adjacency, states), states)
