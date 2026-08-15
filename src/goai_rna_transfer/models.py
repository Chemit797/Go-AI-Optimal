from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ChemicalEncoder(nn.Module):
    def __init__(self, n_bits: int, hidden: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_bits, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, fingerprint: torch.Tensor) -> torch.Tensor:
        return self.network(fingerprint)


class RNAResponseModel(nn.Module):
    def __init__(
        self,
        n_bits: int,
        encoder_hidden: int,
        encoder_dim: int,
        n_cells: int,
        cell_dim: int,
        fusion_hidden: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.chemical_encoder = ChemicalEncoder(n_bits, encoder_hidden, encoder_dim, dropout)
        self.cell_embedding = nn.Embedding(n_cells, cell_dim)
        self.cell_projector = nn.Sequential(nn.Linear(cell_dim, encoder_dim), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(encoder_dim * 3, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, output_dim),
        )

    def forward(self, fingerprint: torch.Tensor, cell_index: torch.Tensor) -> torch.Tensor:
        chemical = self.chemical_encoder(fingerprint)
        cell = self.cell_projector(self.cell_embedding(cell_index))
        return self.head(torch.cat([chemical, cell, chemical * cell], dim=1))


class ContextEncoder(nn.Module):
    def __init__(self, cardinalities: Sequence[int], output_dim: int) -> None:
        super().__init__()
        dimensions = [min(16, max(3, int(round(cardinality**0.5)) * 2)) for cardinality in cardinalities]
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, dimension) for cardinality, dimension in zip(cardinalities, dimensions)
        )
        self.projector = nn.Sequential(
            nn.Linear(sum(dimensions), output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        pieces = [embedding(values[:, index]) for index, embedding in enumerate(self.embeddings)]
        return self.projector(torch.cat(pieces, dim=1))


class ProteinDeltaModel(nn.Module):
    def __init__(
        self,
        chemical_encoder: ChemicalEncoder,
        cardinalities: Sequence[int],
        chemical_dim: int,
        context_dim: int,
        fusion_hidden: int,
        n_proteins: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.chemical_encoder = chemical_encoder
        self.context_encoder = ContextEncoder(cardinalities, context_dim)
        interaction_dim = min(chemical_dim, context_dim)
        self.interaction_dim = interaction_dim
        self.chemical_interaction = nn.Linear(chemical_dim, interaction_dim)
        self.context_interaction = nn.Linear(context_dim, interaction_dim)
        self.fusion = nn.Sequential(
            nn.Linear(chemical_dim + context_dim + interaction_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.Linear(fusion_hidden, n_proteins)

    def forward(self, fingerprint: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        chemical = self.chemical_encoder(fingerprint)
        encoded_context = self.context_encoder(context)
        interaction = self.chemical_interaction(chemical) * self.context_interaction(encoded_context)
        hidden = self.fusion(torch.cat([chemical, encoded_context, interaction], dim=1))
        return self.decoder(hidden)
