from __future__ import annotations

import numpy as np
import pandas as pd

from goai_graph.features import CATEGORICAL_FIELDS, ConditionFeatureBuilder


def test_condition_features_are_target_free_and_unknown_safe():
    metadata = pd.DataFrame(
        {
            "Strains": ["S1", "S2", "S9"],
            "perturbation_no_concentration": ["DrugA", "DrugB", "DrugZ"],
            "Medium": ["M1", "M1", "M9"],
            "Temperature": [30, 30, 37],
            "pert_time": [15, 30, 60],
            "split_final": ["train", "train", "val_both"],
        },
        index=["a", "b", "c"],
    )
    train_ids = pd.Index(["a", "b"])
    builder = ConditionFeatureBuilder().fit(metadata, train_ids)
    transformed = builder.transform(metadata.loc[["c"]])
    categorical_dim = sum(len(builder.categories[field]) for field in CATEGORICAL_FIELDS)

    assert transformed.shape == (1, builder.output_dim)
    assert np.allclose(transformed[0, :categorical_dim], 0.0)
    assert np.isfinite(transformed).all()
    assert builder.summary()["uses_target_statistics"] is False


def test_condition_feature_state_round_trip():
    metadata = pd.DataFrame(
        {
            "Strains": ["S1", "S2"],
            "perturbation_no_concentration": ["DrugA", "DrugB"],
            "Medium": ["M1", "M2"],
            "Temperature": [30, 37],
            "pert_time": [15, 60],
        },
        index=["a", "b"],
    )
    builder = ConditionFeatureBuilder().fit(metadata, metadata.index)
    restored = ConditionFeatureBuilder.from_state_dict(builder.state_dict())
    assert np.allclose(builder.transform(metadata), restored.transform(metadata))
