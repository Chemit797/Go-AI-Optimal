from __future__ import annotations

from goai_baseline.config import load_config
from goai_baseline.controls import exact_control_predictions
from goai_baseline.preprocess import prepare_data

from .conftest import write_config


def test_exact_control_uses_document_match_keys(tmp_path):
    data = prepare_data(load_config(write_config(tmp_path)))
    matched = exact_control_predictions(data.metadata, data.y_log2, ["tr_a"], data.train_ids)
    assert matched.has_exact_match.loc["tr_a"]
    assert matched.predictions.loc["tr_a", "P1"] == 3.0

    unmatched = exact_control_predictions(data.metadata, data.y_log2, ["tr_b"], data.train_ids)
    assert not unmatched.has_exact_match.loc["tr_b"]
