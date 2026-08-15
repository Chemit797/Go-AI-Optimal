from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from goai_baseline.schema import SAMPLE_ID
from scripts.evaluate_oof_residual_overlay import _load_base


def test_load_base_sorts_predictions_and_hashes_fold_contract(tmp_path) -> None:
    path = tmp_path / "base.npz"
    np.savez_compressed(
        path,
        sample_ids=np.asarray(["B", "A"]),
        proteins=np.asarray(["P1", "P2"]),
        folds=np.asarray([1, 0]),
        pred_absolute=np.asarray([[3.0, 4.0], [1.0, 2.0]], dtype=np.float32),
    )
    prediction, actual_hash = _load_base(path)
    assert prediction.index.tolist() == ["A", "B"]
    np.testing.assert_allclose(prediction.to_numpy(), [[1.0, 2.0], [3.0, 4.0]])
    assignments = pd.DataFrame({"fold": [0, 1], SAMPLE_ID: ["A", "B"]})
    expected_hash = hashlib.sha256(
        assignments.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    assert actual_hash == expected_hash


def test_load_base_rejects_nonfinite_predictions(tmp_path) -> None:
    path = tmp_path / "base.npz"
    np.savez_compressed(
        path,
        sample_ids=np.asarray(["A"]),
        proteins=np.asarray(["P1"]),
        folds=np.asarray([0]),
        pred_absolute=np.asarray([[np.nan]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="NaN or infinity"):
        _load_base(path)
