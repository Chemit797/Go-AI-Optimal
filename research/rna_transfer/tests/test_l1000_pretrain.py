from __future__ import annotations

import numpy as np
import pandas as pd

from src.l1000_pretrain import (
    apply_cell_residualizer,
    deterministic_subsample_drugs,
    fit_cell_residualizer,
    fit_context,
    permute_drug_fingerprints,
    transform_context,
)


def test_drug_shuffle_is_whole_drug_and_deranged() -> None:
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])
    fingerprints = np.repeat(np.eye(3, dtype=np.float32), 2, axis=0)
    shuffled = permute_drug_fingerprints(
        groups, fingerprints, np.ones(len(groups), dtype=bool), seed=42
    )
    for group in np.unique(groups):
        rows = shuffled[groups == group]
        assert np.array_equal(rows[0], rows[1])
        assert not np.array_equal(rows[0], fingerprints[np.flatnonzero(groups == group)[0]])


def test_cell_residualizer_is_fit_on_training_cells_only() -> None:
    target = np.asarray([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]], dtype=np.float32)
    residualizer = fit_cell_residualizer(target[:2], ["A", "A"])
    residual = apply_cell_residualizer(target[2:], ["unseen"], residualizer)
    np.testing.assert_allclose(residual, [[8.0, 17.0]])


def test_context_uses_unknown_cell_and_train_only_scaling() -> None:
    train = pd.DataFrame(
        {"cell_id": ["A", "B"], "pert_time": [6.0, 24.0], "pert_dose": [0.0, 9.0]}
    )
    valid = pd.DataFrame({"cell_id": ["C"], "pert_time": [15.0], "pert_dose": [3.0]})
    fitted = fit_context(train)
    cell, numeric = transform_context(valid, fitted)
    assert cell.tolist() == [0]
    assert numeric.shape == (1, 2)
    assert np.isfinite(numeric).all()


def test_smoke_subsample_keeps_parent_aliases_together() -> None:
    table = pd.DataFrame(
        {
            "pert_id": ["alias-1", "alias-2", "other"],
            "parent_key": ["same-parent", "same-parent", "other-parent"],
        }
    )
    selected = deterministic_subsample_drugs(table, max_drugs=1, seed=1)
    if selected["parent_key"].iloc[0] == "same-parent":
        assert set(selected["pert_id"]) == {"alias-1", "alias-2"}
    assert selected["parent_key"].nunique() == 1


def test_group_shuffle_treats_parent_aliases_as_one_drug() -> None:
    parents = np.asarray(["same", "same", "other", "other"])
    fingerprints = np.asarray([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32)
    shuffled = permute_drug_fingerprints(
        parents, fingerprints, np.ones(len(parents), dtype=bool), seed=7
    )
    # Both aliases/rows for one standardized parent receive one deliberately
    # wrong parent input, regardless of their original pert_id spelling.
    np.testing.assert_array_equal(shuffled[0], shuffled[1])
    np.testing.assert_array_equal(shuffled[2], shuffled[3])
    np.testing.assert_array_equal(shuffled[0], fingerprints[2])
    np.testing.assert_array_equal(shuffled[2], fingerprints[0])
