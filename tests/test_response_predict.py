from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import yaml

from goai_response.features import ResponseFeatureBuilder
from goai_response.model import ResponseDecompositionRegressor
from goai_response.predict import predict_response_test

from .conftest import METADATA_COLUMNS, metadata_row


def test_response_checkpoint_predicts_submission_without_training_targets(tmp_path):
    train_metadata = pd.DataFrame(
        [metadata_row("train_control", "Water", "train"), metadata_row("train_drug", "DrugA", "train")],
        columns=METADATA_COLUMNS,
    ).set_index("sample_ID")
    test_metadata = pd.DataFrame(
        [metadata_row("test_1", "DrugZ", "test_chem_only"), metadata_row("test_2", "DrugA", "test_strain_only", strain="S9")],
        columns=METADATA_COLUMNS,
    )
    test_path = tmp_path / "WAYB_WAYC_metadata_test.csv"
    test_metadata.to_csv(test_path, index=False)

    builder = ResponseFeatureBuilder().fit(train_metadata, train_metadata.index)
    features = builder.transform(train_metadata)
    proteins = ["P1", "P2", "P3"]
    model = ResponseDecompositionRegressor(
        features.response.shape[1], features.background.shape[1], features.observation.shape[1],
        len(proteins), hidden_dim=8, response_rank=2, calibration_rank=2, dropout=0.0,
    )
    run = tmp_path / "run"
    run.mkdir()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": {
                "response_input_dim": features.response.shape[1],
                "background_input_dim": features.background.shape[1],
                "observation_input_dim": features.observation.shape[1],
                "n_proteins": len(proteins),
                "hidden_dim": 8,
                "response_rank": 2,
                "calibration_rank": 2,
                "dropout": 0.0,
                "calibration_enabled": True,
            },
            "feature_state": builder.state_dict(),
            "proteins": proteins,
            "target_mean": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            "target_scale": np.ones(3, dtype=np.float32),
        },
        run / "checkpoint.pt",
    )

    baseline = {
        "data": {
            "metadata_train_val": "unused_metadata.csv",
            "proteome_train_val": "unused_proteome.csv",
            "metadata_test": test_path.name,
            "missing_rate_threshold": 0.8,
        },
        "model": {"hidden_dim": 8, "dropout": 0.0, "learning_rate": 0.001, "epochs": 1, "seed": 42, "device": "cpu"},
        "features": {"chemical_hash_dim": 8},
        "runtime": {"runs_dir": "runs"},
    }
    with (tmp_path / "baseline.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(baseline, handle)
    response = {
        "baseline_config": "baseline.yaml",
        "model": {
            "hidden_dim": 8, "response_rank": 2, "calibration_rank": 2, "dropout": 0.0,
            "learning_rate": 0.001, "weight_decay": 0.0, "epochs": 1, "batch_size": 1,
            "seed": 42, "device": "cpu", "absolute_weight": 0.25, "background_weight": 1.0,
            "fc_weight": 1.0, "target_scale_floor": 0.1, "calibration_enabled": True,
        },
        "entity": {"chemical_map": None, "strain_features": None, "chemical_bits": 16},
        "graph": {"variant": "none", "artifact": None, "weight": 0.0},
        "runtime": {"runs_dir": "runs"},
    }
    with (tmp_path / "response.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(response, handle)

    output = predict_response_test(tmp_path / "response.yaml", run, tmp_path / "prediction.csv")
    prediction = pd.read_csv(output)
    assert prediction.columns.tolist() == ["sample_ID", *proteins]
    assert prediction["sample_ID"].tolist() == ["test_1", "test_2"]
    assert np.isfinite(prediction.loc[:, proteins].to_numpy()).all()
    with (tmp_path / "prediction_contract.json").open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    assert contract["rows"] == 2
    assert contract["proteins"] == 3
