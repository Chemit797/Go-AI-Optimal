from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from goai_response.artifacts import (
    validate_chemical_structure_artifact,
    validate_strain_feature_artifact,
)
from goai_response.config import load_response_config
from goai_response.features import ResponseFeatureBuilder
from goai_response.train import _artifact_hash_chain
from scripts.nightly.build_m7_m8_confirm import (
    PROTOCOL_LABEL,
    materialize_confirm_matrix,
)


ROOT = Path(__file__).resolve().parents[1]


def _registry(path: Path, entity_type: str, rows: list[tuple[str, str]]) -> None:
    key = "raw_name" if entity_type == "chemical" else "strain_code"
    pd.DataFrame(
        [
            {
                key: name,
                "canonical_id": f"test:{name}",
                "canonical_name": name,
                "mapping_status": status,
                "evidence_tier": (
                    "A_verified"
                    if status == "verified"
                    else "B_primary_candidate"
                    if status == "high_confidence_candidate"
                    else "D_proxy_assumption"
                    if status == "proxy"
                    else "E_unresolved"
                ),
                "proxy_target": "test:parent" if status == "proxy" else "",
                "is_control": False,
                "is_quality_control": False,
            }
            for name, status in rows
        ]
    ).to_csv(path, sep="\t", index=False)


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Strains": ["VerifiedStrain", "CandidateStrain", "MissingStrain"],
            "perturbation_no_concentration": [
                "VerifiedDrug",
                "CandidateDrug",
                "MissingDrug",
            ],
            "Medium": ["M"] * 3,
            "Temperature": ["30"] * 3,
            "pert_time": [15] * 3,
            "data_source": ["S"] * 3,
            "pert_id": ["1", "2", "3"],
            "instrument": ["I"] * 3,
            "Yeast_cell_plate": ["P"] * 3,
        },
        index=["verified", "candidate", "missing"],
    )


def test_verified_only_zeros_candidates_but_keeps_identity_and_support_gates(tmp_path: Path) -> None:
    chemical_registry = tmp_path / "chemical.tsv"
    strain_registry = tmp_path / "strain.tsv"
    _registry(
        chemical_registry,
        "chemical",
        [
            ("VerifiedDrug", "verified"),
            ("CandidateDrug", "high_confidence_candidate"),
            ("MissingDrug", "unresolved"),
        ],
    )
    _registry(
        strain_registry,
        "strain",
        [
            ("VerifiedStrain", "verified"),
            ("CandidateStrain", "high_confidence_candidate"),
            ("MissingStrain", "unresolved"),
        ],
    )
    chemical_map = tmp_path / "map.tsv"
    pd.DataFrame(
        {
            "raw_name": ["VerifiedDrug", "CandidateDrug", "MissingDrug"],
            "status": ["resolved", "resolved", "unresolved"],
            "is_control": [False] * 3,
            "isomeric_smiles": ["CC", "c1ccccc1", ""],
        }
    ).to_csv(chemical_map, sep="\t", index=False)
    strain_features = tmp_path / "strain_features.tsv"
    pd.DataFrame(
        {
            "strain_code": ["VerifiedStrain", "CandidateStrain", "MissingStrain"],
            "axis": [1.0, 9.0, 0.0],
        }
    ).to_csv(strain_features, sep="\t", index=False)
    metadata = _metadata()

    verified = ResponseFeatureBuilder(
        chemical_map=chemical_map,
        chemical_bits=16,
        strain_features_path=strain_features,
        chemical_registry_path=chemical_registry,
        strain_registry_path=strain_registry,
        semantic_identity_policy="verified_only",
    ).fit(metadata, metadata.index)
    verified_features = verified.transform(metadata)
    chemical = verified_features.general_perturbation
    strain = verified_features.general_cell[:, -(1 + 4) :]
    # With one admissible verified identity, fold-fit centering can turn its
    # transformed vector into zero.  The scaler mean proves it was admitted.
    assert verified.chemical_mean is not None
    assert verified.chemical_matrix is not None
    assert np.allclose(verified.chemical_mean, verified.chemical_matrix[0])
    assert np.allclose(chemical[1:, :-4], 0.0)
    assert np.allclose(chemical[:, -4:], np.eye(4, dtype=np.float32)[[0, 1, 3]])
    assert np.allclose(strain[1, 0], 0.0)
    assert np.allclose(strain[:, -4:], np.eye(4, dtype=np.float32)[[0, 1, 3]])
    # Expert support is a fold-fit fact; semantic policy must not turn a seen
    # candidate into an unseen entity.
    assert verified_features.strain_seen.ravel().tolist() == [1.0, 1.0, 1.0]
    assert verified_features.chemical_seen.ravel().tolist() == [1.0, 1.0, 1.0]

    research = ResponseFeatureBuilder(
        chemical_map=chemical_map,
        chemical_bits=16,
        strain_features_path=strain_features,
        chemical_registry_path=chemical_registry,
        strain_registry_path=strain_registry,
        semantic_identity_policy="research_allow_candidate",
    ).fit(metadata, metadata.index)
    research_features = research.transform(metadata)
    assert np.any(np.abs(research_features.general_perturbation[1, :-4]) > 0)
    assert not np.allclose(research_features.general_cell[1, -(1 + 4)], 0.0)


def test_formal_m8_configs_require_manifests_and_verified_identity_policy() -> None:
    for name in ("m8_chemical_morgan.yaml", "m8_strain_mds.yaml", "m8_dual_semantics.yaml"):
        config = load_response_config(ROOT / "configs/experiments" / name)
        assert config.entity.semantic_identity_policy == "verified_only"
        assert config.entity.semantic_training_coverage_required is True
    assert load_response_config(
        ROOT / "configs/experiments/m8_chemical_morgan.yaml"
    ).entity.chemical_structure_manifest_required
    assert load_response_config(
        ROOT / "configs/experiments/m8_strain_mds.yaml"
    ).entity.strain_features_manifest_required


def test_fold_fit_semantic_coverage_blocks_all_zero_formal_axes(
    tmp_path: Path,
) -> None:
    chemical_registry = tmp_path / "chemical.tsv"
    strain_registry = tmp_path / "strain.tsv"
    _registry(
        chemical_registry,
        "chemical",
        [("CandidateDrug", "high_confidence_candidate")],
    )
    _registry(
        strain_registry,
        "strain",
        [("CandidateStrain", "high_confidence_candidate")],
    )
    chemical_map = tmp_path / "map.tsv"
    pd.DataFrame(
        {
            "raw_name": ["CandidateDrug"],
            "status": ["resolved"],
            "is_control": [False],
            "isomeric_smiles": ["CCO"],
        }
    ).to_csv(chemical_map, sep="\t", index=False)
    strain_features = tmp_path / "strain.tsv.features"
    pd.DataFrame(
        {"strain_code": ["CandidateStrain"], "axis": [3.0]}
    ).to_csv(strain_features, sep="\t", index=False)
    metadata = _metadata().loc[["candidate"]]

    formal = ResponseFeatureBuilder(
        chemical_map=chemical_map,
        chemical_bits=16,
        strain_features_path=strain_features,
        chemical_registry_path=chemical_registry,
        strain_registry_path=strain_registry,
        semantic_identity_policy="verified_only",
        semantic_training_coverage_required=True,
    )
    with pytest.raises(ValueError, match="semantic training coverage is empty"):
        formal.fit(metadata, metadata.index)

    research = ResponseFeatureBuilder(
        chemical_map=chemical_map,
        chemical_bits=16,
        strain_features_path=strain_features,
        chemical_registry_path=chemical_registry,
        strain_registry_path=strain_registry,
        semantic_identity_policy="research_allow_candidate",
        semantic_training_coverage_required=True,
    ).fit(metadata, metadata.index)
    assert research.semantic_training_coverage["chemical"][
        "admitted_unique_entities"
    ] == 1
    assert research.semantic_training_coverage["strain"][
        "admitted_unique_entities"
    ] == 1


def test_proxy_semantics_are_research_only() -> None:
    with pytest.raises(ValueError, match="research-only"):
        # Exercise parsing rather than dataclass construction so the machine
        # gate cannot be bypassed by a permissive formal YAML.
        payload = (ROOT / "configs/experiments/m8_chemical_morgan.yaml").read_text()
        payload = payload.replace(
            "allow_proxy_semantics: false", "allow_proxy_semantics: true"
        ).replace(
            "chemical_structure_view: exact", "chemical_structure_view: parent"
        ).replace(
            "chemical_registry:",
            "chemical_parent_views: ../../data/processed/entities/chemical_parent_normalized_views.tsv\n  chemical_registry:",
        )
        path = ROOT / "configs/experiments/.test_invalid_proxy.yaml"
        try:
            path.write_text(payload, encoding="utf-8")
            load_response_config(path)
        finally:
            path.unlink(missing_ok=True)


def test_formal_m8_materialization_blocks_zero_verified_fit_coverage() -> None:
    matrix_dir = ROOT / "configs/nightly/20260813-m7-m8"
    template = yaml.safe_load(
        (matrix_dir / "confirm_candidates.yaml").read_text(encoding="utf-8")
    )
    selection = {
        "selection_status": "selected",
        "protocol_label": PROTOCOL_LABEL,
        "selected": {
            "strain": {"scale": 0.5},
            "chemical": {"scale": 0.5},
            "pair": {"scale": 0.5},
        },
    }
    registry_artifacts = {}
    for axis in ("chemical", "strain"):
        path = ROOT / f"data/processed/entities/{axis}_registry.tsv"
        registry_artifacts[axis] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    # Even a synthetically ready identity receipt cannot turn the current
    # candidate-only train identities into a meaningful verified-only M8.
    evidence = {
        "schema": "goai.m7_m8.preconfirmation_evidence.v1",
        "protocol_label": PROTOCOL_LABEL,
        "status": "valid",
        "identity": {
            "semantic_promotion_status": "ready",
            "registry_artifacts": registry_artifacts,
        },
        "calibration": {
            "selected_experiment_id": "test-calibration",
            "sha256": "synthetic",
            "selected_config": {
                "calibration_rank": 8,
                "calibration_use_plate": True,
                "calibration_plate_dropout": 0.3,
                "calibration_plate_shuffle": False,
            },
        },
    }
    materialized = materialize_confirm_matrix(
        template,
        selection,
        absolute_base_config=str(
            (matrix_dir / template["base_config"]).resolve()
        ),
        preconfirmation_evidence=evidence,
    )
    assert any(
        str(item["model_id"]).startswith("M7")
        for item in materialized["experiments"]
    )
    assert not any(
        str(item["model_id"]).startswith("M8")
        for item in materialized["experiments"]
    )
    assert {
        item["reason"] for item in materialized["blocked_confirm_candidates"]
    } == {"semantic_training_coverage_not_ready"}


def test_strain_manifest_tampering_is_rejected_and_receipted(tmp_path: Path) -> None:
    real = ROOT / "data/processed/entities/strain_semantics_numeric.tsv"
    manifest = json.loads(
        (real.parent / "strain_semantics_manifest.json").read_text(encoding="utf-8")
    )
    required_external = []
    for field in ("source_table", "distance_matrix", "identity_evidence_manifest"):
        path = Path(str(manifest[field]))
        required_external.append(
            path if path.is_absolute() else (real.parent / path).resolve()
        )
    if not all(path.is_file() for path in required_external):
        pytest.skip("requires the optional public 1,011-genome evidence bundle")

    receipt = validate_strain_feature_artifact(real, manifest_required=True)
    assert receipt and receipt["selected_kind"] == "real"
    base = load_response_config(ROOT / "configs/experiments/m8_strain_mds.yaml")
    artifacts, _ = _artifact_hash_chain(base)
    assert artifacts["strain_features_manifest"] == receipt

    source_dir = real.parent
    for name in ("strain_semantics_numeric.tsv", "strain_semantics_shuffled.tsv", "strain_semantics_manifest.json"):
        (tmp_path / name).write_bytes((source_dir / name).read_bytes())
    manifest_path = tmp_path / "strain_semantics_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, filename in (
        ("real_path", "strain_semantics_numeric.tsv"),
        ("shuffled_path", "strain_semantics_shuffled.tsv"),
    ):
        manifest[key] = str((tmp_path / filename).resolve())
    # Keep public evidence paths real; only the selected feature is tampered.
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    table = pd.read_csv(tmp_path / "strain_semantics_numeric.tsv", sep="\t")
    table.loc[0, "snp_mds_001"] += 1.0
    table.to_csv(tmp_path / "strain_semantics_numeric.tsv", sep="\t", index=False)
    with pytest.raises(ValueError, match="feature hash mismatch"):
        validate_strain_feature_artifact(
            tmp_path / "strain_semantics_numeric.tsv", manifest_required=True
        )


def test_morgan_real_and_shuffled_share_a_verified_structure_chain() -> None:
    views = ROOT / "data/processed/chemical_views"
    real = validate_chemical_structure_artifact(
        views / "chemical_entity_map_exact.tsv", manifest_required=True
    )
    shuffled = validate_chemical_structure_artifact(
        views / "chemical_entity_map_exact_shuffled.tsv", manifest_required=True
    )
    assert real and shuffled
    assert real["manifest_sha256"] == shuffled["manifest_sha256"]
    assert real["paired_real_sha256"] == shuffled["paired_real_sha256"]
    assert real["paired_shuffled_sha256"] == shuffled["paired_shuffled_sha256"]
    assert real["selected_kind"] == "exact"
    assert shuffled["selected_kind"] == "exact_shuffled"
    assert int(real["permuted_rows"]) > 1

    formal = load_response_config(
        ROOT / "configs/experiments/m8_chemical_morgan.yaml"
    )
    mislabeled = replace(
        formal,
        entity=replace(
            formal.entity,
            chemical_map=views / "chemical_entity_map_exact_shuffled.tsv",
            chemical_structure_view="exact",
        ),
    )
    with pytest.raises(ValueError, match="view label"):
        _artifact_hash_chain(mislabeled)
