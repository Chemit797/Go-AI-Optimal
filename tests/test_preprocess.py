from __future__ import annotations

from goai_baseline.config import load_config
from goai_baseline.preprocess import prepare_data

from .conftest import write_config


def test_train_only_missing_filter_and_log2_mask(tmp_path):
    config = load_config(write_config(tmp_path))
    data = prepare_data(config)
    assert data.proteins == ["P1", "P2"]
    assert data.mask.loc["tr_ctrl", "P2"] == False
    assert data.y_log2.loc["tr_ctrl", "P2"] != data.y_log2.loc["tr_ctrl", "P2"]
    assert data.y_log2.loc["tr_ctrl", "P1"] == 3.0
