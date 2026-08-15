"""Parameter-matched static PPI and graph-control regressors."""

from __future__ import annotations

import math

import torch
from torch import nn


def sparse_propagate(adjacency: torch.Tensor, node_states: torch.Tensor) -> torch.Tensor:
    if not adjacency.is_sparse:
        raise ValueError("adjacency must be a sparse COO tensor")
    return torch.sparse.mm(adjacency, node_states)


class ProteinGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        message = sparse_propagate(adjacency, states)
        update = self.activation(self.self_linear(states) + self.neighbor_linear(message))
        return self.norm(states + self.dropout(update))


class StaticProteinGraphRegressor(nn.Module):
    """Encode a shared protein graph and decode sample conditions bilinearly.

    The graph is encoded once per optimizer step, independent of batch size.
    Passing an identity adjacency produces the parameter-matched no-graph model.
    """

    def __init__(
        self,
        condition_dim: int,
        n_proteins: int,
        hidden_dim: int = 64,
        condition_hidden_dim: int = 128,
        graph_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_proteins = n_proteins
        self.protein_embedding = nn.Parameter(torch.randn(n_proteins, hidden_dim) * 0.02)
        self.protein_bias = nn.Parameter(torch.zeros(n_proteins))
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, condition_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(condition_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.graph_encoder = nn.ModuleList(
            [ProteinGraphLayer(hidden_dim, dropout) for _ in range(graph_layers)]
        )

    def encode_proteins(self, adjacency: torch.Tensor) -> torch.Tensor:
        states = self.protein_embedding
        for layer in self.graph_encoder:
            states = layer(states, adjacency)
        return states

    def forward(self, conditions: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        condition_states = self.condition_encoder(conditions)
        protein_states = self.encode_proteins(adjacency)
        residual = condition_states @ protein_states.transpose(0, 1)
        return residual / math.sqrt(self.hidden_dim) + self.protein_bias
