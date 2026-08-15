from __future__ import annotations

import numpy as np

from goai_baseline.config import load_config
from goai_baseline.features import FeatureBuilder
from goai_baseline.preprocess import prepare_data

from .conftest import write_config


def test_p4_features_accept_unseen_entities_with_training_only_priors(tmp_path):
    data = prepare_data(load_config(write_config(tmp_path)))
    builder = FeatureBuilder("p4_hash", chemical_hash_dim=8)
    train_x = builder.fit_transform(data.metadata, data.y_log2, data.train_ids)
    test_x = builder.transform(data.metadata.loc[["val_both"]])
    assert train_x.dtype == np.float32
    assert test_x.shape[1] == train_x.shape[1]
    assert np.isfinite(test_x).all()
    assert builder.summary()["total_input_dim"] == train_x.shape[1]


def test_oof_priors_do_not_reuse_a_training_row_target(tmp_path):
    data = prepare_data(load_config(write_config(tmp_path)))
    regular = FeatureBuilder("p1_priors").fit_transform(data.metadata, data.y_log2, data.train_ids)
    oof = FeatureBuilder("p1_oof_priors").fit_transform(data.metadata, data.y_log2, data.train_ids)
    assert regular.shape == oof.shape
    assert not np.allclose(regular, oof)
