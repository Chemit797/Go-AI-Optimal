from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from goai_response.artifacts import validate_chemical_feature_artifact
from goai_response.config import load_response_config
from goai_response.features import ResponseFeatureBuilder
from goai_response.model import ResponseDecompositionRegressor
from goai_response.predict import load_response_checkpoint
from goai_response.train import _artifact_hash_chain

from .conftest import METADATA_COLUMNS, metadata_row


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(
    root: Path, names: tuple[str, ...] = ("DrugA", "DrugB")
) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    source = root / "chemical_map.tsv"
    real = root / "chemberta_real.tsv"
    shuffled = root / "chemberta_shuffled.tsv"
    pd.DataFrame(
        {
            "raw_name": list(names),
            "status": ["resolved"] * len(names),
            "isomeric_smiles": [f"{'C' * (index + 1)}N" for index in range(len(names))],
        }
    ).to_csv(source, sep="\t", index=False)
    pd.DataFrame(
        {
            "raw_name": list(names),
            "f0": [float(index + 1) for index in range(len(names))],
            "f1": [float(index + 11) for index in range(len(names))],
        }
    ).to_csv(real, sep="\t", index=False)
    pd.DataFrame(
        {
            "raw_name": list(names),
            "f0": [float(index + 1) for index in reversed(range(len(names)))],
            "f1": [float(index + 11) for index in reversed(range(len(names)))],
        }
    ).to_csv(shuffled, sep="\t", index=False)
    manifest = {
        "model": "example/frozen-model",
        "model_revision": "0123456789abcdef",
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "smiles_column": "isomeric_smiles",
        "rows": len(names),
        "resolved_rows": len(names),
        "embedding_dim": 2,
        "real_path": str(real.resolve()),
        "real_sha256": _sha256(real),
        "shuffled_path": str(shuffled.resolve()),
        "shuffled_sha256": _sha256(shuffled),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source, real, manifest_path


def test_formal_chemical_feature_manifest_validates_and_enters_checkpoint_chain(tmp_path):
    source, real, manifest = _write_artifact(tmp_path / "features")
    receipt = validate_chemical_feature_artifact(real, manifest_required=True)
    assert receipt is not None
    assert receipt["status"] == "verified_manifest"
    assert receipt["manifest_sha256"] == _sha256(manifest)
    assert receipt["source_sha256"] == _sha256(source)
    assert receipt["selected_kind"] == "real"
    assert receipt["rows"] == 2
    assert receipt["embedding_dim"] == 2

    base = load_response_config(ROOT / "configs/experiments/m7_general_only.yaml")
    config = replace(
        base,
        entity=replace(
            base.entity,
            chemical_features=real,
            chemical_features_manifest_required=True,
        ),
    )
    artifacts, chain = _artifact_hash_chain(config)
    assert artifacts["chemical_features_manifest"] == receipt
    assert isinstance(chain, str) and len(chain) == 64


def test_checkpoint_prediction_rechecks_mapping_to_embedding_chain(tmp_path):
    source, real, _ = _write_artifact(tmp_path / "features", ("Water", "DrugA"))
    base = load_response_config(ROOT / "configs/experiments/m7_general_only.yaml")
    config = replace(
        base,
        entity=replace(
            base.entity,
            chemical_map=None,
            chemical_features=real,
            chemical_features_manifest_required=True,
            strain_features=None,
            chemical_registry=None,
            strain_registry=None,
            chemical_parent_views=None,
        ),
    )
    metadata = pd.DataFrame(
        [
            metadata_row("control", "Water", "train"),
            metadata_row("treated", "DrugA", "train"),
        ],
        columns=METADATA_COLUMNS,
    ).set_index("sample_ID")
    builder = ResponseFeatureBuilder(chemical_features_path=real).fit(
        metadata, metadata.index
    )
    features = builder.transform(metadata)
    model = ResponseDecompositionRegressor(
        features.response.shape[1],
        features.background.shape[1],
        features.observation.shape[1],
        2,
        hidden_dim=4,
        response_rank=2,
        calibration_rank=2,
        dropout=0.0,
    )
    artifacts, chain = _artifact_hash_chain(config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": {
                "response_input_dim": features.response.shape[1],
                "background_input_dim": features.background.shape[1],
                "observation_input_dim": features.observation.shape[1],
                "n_proteins": 2,
                "hidden_dim": 4,
                "response_rank": 2,
                "calibration_rank": 2,
                "dropout": 0.0,
                "calibration_enabled": True,
            },
            "feature_state": builder.state_dict(),
            "proteins": ["P1", "P2"],
            "target_mean": np.zeros(2, dtype=np.float32),
            "target_scale": np.ones(2, dtype=np.float32),
            "artifact_hashes": artifacts,
            "artifact_chain_sha256": chain,
        },
        checkpoint,
    )
    load_response_checkpoint(checkpoint, torch.device("cpu"), config)

    source_table = pd.read_csv(source, sep="\t")
    source_table.loc[0, "isomeric_smiles"] = "CCCC"
    source_table.to_csv(source, sep="\t", index=False)
    with pytest.raises(ValueError, match="mapping hash drift"):
        load_response_checkpoint(checkpoint, torch.device("cpu"), config)


def test_formal_chemical_feature_manifest_rejects_mapping_drift(tmp_path):
    source, real, _ = _write_artifact(tmp_path / "features")
    table = pd.read_csv(source, sep="\t")
    table.loc[0, "isomeric_smiles"] = "CCC"
    table.to_csv(source, sep="\t", index=False)
    with pytest.raises(ValueError, match="mapping hash drift"):
        validate_chemical_feature_artifact(real, manifest_required=True)


def test_formal_chemical_feature_manifest_rejects_feature_tampering(tmp_path):
    _, real, _ = _write_artifact(tmp_path / "features")
    table = pd.read_csv(real, sep="\t")
    table.loc[0, "f0"] = 99.0
    table.to_csv(real, sep="\t", index=False)
    with pytest.raises(ValueError, match="feature hash mismatch"):
        validate_chemical_feature_artifact(real, manifest_required=True)


def test_identity_risk_review_tampering_changes_checkpoint_artifact_chain(tmp_path):
    base = load_response_config(ROOT / "configs/experiments/m7_general_only.yaml")
    risk_path = tmp_path / "chemical_identity_risk_review.tsv"
    risk_path.write_text(
        "raw_name\trisk_class\tzero_risky\tevidence_path\n"
        "G418\tstereochemistry_conflict\ttrue\tevidence.json\n",
        encoding="utf-8",
    )
    config = replace(
        base,
        entity=replace(base.entity, chemical_identity_risks=risk_path),
    )
    artifacts_before, chain_before = _artifact_hash_chain(config)
    assert artifacts_before["chemical_identity_risks"]["sha256"] == _sha256(risk_path)

    risk_path.write_text(
        risk_path.read_text(encoding="utf-8").replace(
            "stereochemistry_conflict", "silently_changed_risk"
        ),
        encoding="utf-8",
    )
    artifacts_after, chain_after = _artifact_hash_chain(config)
    assert artifacts_after["chemical_identity_risks"]["sha256"] == _sha256(risk_path)
    assert chain_after != chain_before


def test_formal_chemical_feature_manifest_rejects_entity_order_even_with_new_hash(tmp_path):
    _, real, manifest_path = _write_artifact(tmp_path / "features")
    table = pd.read_csv(real, sep="\t").iloc[::-1].reset_index(drop=True)
    table.to_csv(real, sep="\t", index=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["real_sha256"] = _sha256(real)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="entity order"):
        validate_chemical_feature_artifact(real, manifest_required=True)


def test_manifestless_historical_feature_is_explicitly_legacy(tmp_path):
    feature = tmp_path / "legacy.tsv"
    pd.DataFrame({"raw_name": ["DrugA"], "f0": [1.0]}).to_csv(
        feature, sep="\t", index=False
    )
    with pytest.warns(RuntimeWarning, match="legacy_unverified"):
        assert validate_chemical_feature_artifact(feature) == {
            "status": "legacy_unverified"
        }
    with pytest.raises(FileNotFoundError, match="requires"):
        validate_chemical_feature_artifact(feature, manifest_required=True)
