from __future__ import annotations

import numpy as np

from goai_baseline.config import load_config
from goai_baseline.evaluate import evaluate_matched_control
from goai_baseline.preprocess import prepare_data

from .conftest import write_config


def test_matched_control_metrics_exclude_control_missing_positions(tmp_path):
    data = prepare_data(load_config(write_config(tmp_path)))
    report, _ = evaluate_matched_control(data)
    assert not report.empty
    assert np.isfinite(report["log2_rmse"]).all()
    assert (report["n_observed_values"] > 0).all()
