from __future__ import annotations

import numpy as np
import pandas as pd

from src.rna_pretrain import permute_drug_fingerprints
from src.train_s1 import (
    fold_context_reference,
    masked_location_scale,
    permute_validation_fingerprints,
)


def test_drug_shuffle_is_whole_drug_derangement() -> None:
    observation = pd.DataFrame(
        {
            "sm_lincs_id": ["a", "a", "b", "b", "c", "c"],
            "cell_type": ["x", "y", "x", "y", "x", "y"],
        }
    )
    fingerprints = np.repeat(np.arange(3, dtype=np.float32), 2)[:, None]
    selected = np.ones(len(observation), dtype=bool)
    shuffled = permute_drug_fingerprints(observation, fingerprints, selected, seed=7)
    assert shuffled[0, 0] == shuffled[1, 0]
    assert shuffled[2, 0] == shuffled[3, 0]
    assert shuffled[4, 0] == shuffled[5, 0]
    assert np.all(shuffled[:, 0] != fingerprints[:, 0])


def test_masked_scale_ignores_unobserved_fill_values() -> None:
    values = np.asarray([[1.0, 999.0], [3.0, 5.0]], dtype=np.float32)
    mask = np.asarray([[True, False], [True, True]])
    mean, scale = masked_location_scale(values, mask)
    np.testing.assert_allclose(mean, [2.0, 5.0])
    np.testing.assert_allclose(scale, [1.0, 1.0])


def test_context_reference_uses_only_fold_training_rows() -> None:
    delta = np.asarray([[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]], dtype=np.float32)
    mask = np.ones_like(delta, dtype=bool)
    keys = np.asarray(["same", "same", "same"])
    train = np.asarray([True, True, False])
    valid = ~train
    reference, reference_mask = fold_context_reference(delta, mask, keys, train, valid)
    np.testing.assert_allclose(reference, [[2.0, 3.0]])
    assert reference_mask.all()


def test_validation_permutation_is_whole_chemical_derangement() -> None:
    fingerprints = np.asarray([[1.0], [1.0], [2.0], [2.0], [9.0]], dtype=np.float32)
    chemicals = np.asarray(["a", "a", "b", "b", "train"])
    valid = np.asarray([True, True, True, True, False])
    permuted = permute_validation_fingerprints(fingerprints, chemicals, valid)
    np.testing.assert_array_equal(permuted[:, 0], [2.0, 2.0, 1.0, 1.0])
