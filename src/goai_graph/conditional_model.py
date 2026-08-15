"""Condition-injected protein graph decoder.

Unlike the static MVP, this model creates a protein state for every sample
before propagation.  The graph therefore has a mechanism through which a
verified perturbation target can affect neighbouring proteins.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .model import sparse_propagate


def sparse_propagate_batch(adjacency: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    if states.ndim != 3:
        raise ValueError("states must have shape [batch, nodes, hidden]")
    batch, nodes, hidden = states.shape
    # The no-graph control uses an exact identity COO matrix.  Avoid routing
    # that control through sparse kernels, which are needlessly expensive for
    # a batched identity multiplication on CPU and some CUDA builds.
    coalesced = adjacency.coalesce()
    if coalesced._nnz() == nodes:
        indices = coalesced.indices()
        diagonal = torch.arange(nodes, device=indices.device)
        if torch.equal(indices[0], diagonal) and torch.equal(indices[1], diagonal) and torch.allclose(
            coalesced.values(), torch.ones(nodes, device=coalesced.values().device, dtype=coalesced.values().dtype)
        ):
            return states
    flattened = states.transpose(0, 1).reshape(nodes, batch * hidden)
    propagated = sparse_propagate(coalesced, flattened)
    return propagated.reshape(nodes, batch, hidden).transpose(0, 1)


class ConditionalGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, condition_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(condition_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor, condition: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        message = sparse_propagate_batch(adjacency, states)
        update = self.activation(self.self_linear(states) + self.neighbor_linear(message))
        gate = torch.sigmoid(self.gate(condition)).unsqueeze(1)
        return self.norm(states + self.dropout(update * gate))


class ConditionalProteinGraphRegressor(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        n_proteins: int,
        node_feature_dim: int = 3,
        hidden_dim: int = 64,
        condition_hidden_dim: int = 128,
        graph_layers: int = 2,
        dropout: float = 0.1,
        condition_id_dropout: float = 0.0,
        chemical_id_start: int = 0,
        chemical_id_end: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_proteins = n_proteins
        self.condition_id_dropout = condition_id_dropout
        self.chemical_id_start = chemical_id_start
        self.chemical_id_end = chemical_id_end
        self.protein_embedding = nn.Parameter(torch.randn(n_proteins, hidden_dim) * 0.02)
        self.protein_bias = nn.Parameter(torch.zeros(n_proteins))
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, condition_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(condition_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.film = nn.Linear(hidden_dim, hidden_dim * 2)
        self.node_projection = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.graph_encoder = nn.ModuleList(
            [ConditionalGraphLayer(hidden_dim, hidden_dim, dropout) for _ in range(graph_layers)]
        )
        self.decoder = nn.Linear(hidden_dim, hidden_dim)

    def _drop_condition_ids(self, conditions: torch.Tensor) -> torch.Tensor:
        if not self.training or self.condition_id_dropout <= 0:
            return conditions
        if self.chemical_id_end <= self.chemical_id_start:
            return conditions
        keep = torch.rand(conditions.shape[0], device=conditions.device) >= self.condition_id_dropout
        result = conditions.clone()
        result[~keep, self.chemical_id_start : self.chemical_id_end] = 0.0
        return result

    def encode_proteins(
        self,
        conditions: torch.Tensor,
        node_signal: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition_states = self.condition_encoder(self._drop_condition_ids(conditions))
        gamma, beta = self.film(condition_states).chunk(2, dim=-1)
        states = self.protein_embedding.unsqueeze(0) * (1.0 + 0.1 * torch.tanh(gamma).unsqueeze(1))
        states = states + beta.unsqueeze(1) + self.node_projection(node_signal)
        for layer in self.graph_encoder:
            states = layer(states, condition_states, adjacency)
        return condition_states, states

    def forward(
        self,
        conditions: torch.Tensor,
        node_signal: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        condition_states, protein_states = self.encode_proteins(conditions, node_signal, adjacency)
        decoded_condition = self.decoder(condition_states)
        residual = torch.einsum("bh,bnh->bn", decoded_condition, protein_states)
        return residual / math.sqrt(self.hidden_dim) + self.protein_bias.unsqueeze(0)
