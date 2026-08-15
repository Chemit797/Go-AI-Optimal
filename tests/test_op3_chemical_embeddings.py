from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from scripts.build_op3_chemical_embeddings import (
    encode,
    morgan_fingerprints,
)
from src.models import ChemicalEncoder


def test_morgan_fingerprints_zero_unresolved_rows():
    mapping = pd.DataFrame(
        {
            "raw_name": ["ethanol", "missing"],
            "status": ["resolved", "unresolved"],
            "canonical_smiles": ["CCO", ""],
        }
    )
    values, resolved = morgan_fingerprints(
        mapping, smiles_column="canonical_smiles", radius=2, n_bits=64
    )
    assert values.shape == (2, 64)
    assert resolved.tolist() == [True, False]
    assert values[0].sum() > 0
    assert np.array_equal(values[1], np.zeros(64, dtype=np.float32))


def test_encode_is_deterministic_in_eval_mode():
    torch.manual_seed(3)
    encoder = ChemicalEncoder(16, 8, 4, dropout=0.5)
    encoder.eval()
    values = np.arange(32, dtype=np.float32).reshape(2, 16)
    first = encode(encoder, values)
    second = encode(encoder, values)
    assert first.shape == (2, 4)
    assert np.array_equal(first, second)
