from __future__ import annotations

import pandas as pd

from goai_baseline.config import load_config
from goai_baseline.predict import predict_test
from goai_baseline.train import train_variant

from .conftest import write_config


def test_p0_training_and_metadata_only_prediction(tmp_path):
    config_path = write_config(tmp_path, epochs=2)
    run_dir = tmp_path / "run"
    train_variant(load_config(config_path), "p0_onehot", run_dir)
    output = predict_test(config_path, run_dir)
    submission = pd.read_csv(output)
    assert submission.columns.tolist() == ["sample_ID", "P1", "P2"]
    assert submission["sample_ID"].tolist() == ["test_1", "test_2"]
