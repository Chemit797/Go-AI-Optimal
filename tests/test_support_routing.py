from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd
import pytest
import torch

from goai_baseline.schema import CHEMICAL, STRAIN
from goai_response import routing
from goai_response.entities import (
    build_support_manifest,
    registry_from_frame,
    manifest_sha256,
    write_json_with_hash,
)

from .conftest import METADATA_COLUMNS, metadata_row


ROOT = Path(__file__).resolve().parents[1]


def _registries(metadata: pd.DataFrame):
    chemical_rows = []
    for name in sorted(metadata[CHEMICAL].astype(str).unique()):
        chemical_rows.append(
            {
                "raw_name": name,
                "canonical_id": f"chemical:{name}",
                "canonical_name": name,
                "mapping_status": "verified",
                "evidence_tier": "A_verified",
                "proxy_target": "",
                "is_control": name.casefold() in {"water", "dmso"},
                "is_quality_control": name.casefold() == "quality control",
            }
        )
    strain_rows = [
        {
            "strain_code": name,
            "canonical_id": f"strain:{name}",
            "canonical_name": name,
            "mapping_status": "verified",
            "evidence_tier": "A_verified",
            "proxy_target": "",
            "is_control": False,
            "is_quality_control": False,
        }
        for name in sorted(metadata[STRAIN].astype(str).unique())
    ]
    return {
        "chemical": registry_from_frame(pd.DataFrame(chemical_rows), "chemical"),
        "strain": registry_from_frame(pd.DataFrame(strain_rows), "strain"),
    }


def test_full_refit_support_routes_match_released_test_structure():
    train_path = ROOT / "WAYB_WAYC_metadata_train_val.csv"
    test_path = ROOT / "WAYB_WAYC_metadata_test.csv"
    if not train_path.is_file() or not test_path.is_file():
        pytest.skip("requires the official GOAI train/test metadata files")
    fit = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False).set_index(
        "sample_ID", verify_integrity=True
    )
    manifest = build_support_manifest(
        fit,
        _registries(pd.concat([fit, test.reset_index()], ignore_index=True)),
    )

    audit = routing.support_route_audit(test, manifest)
    assert audit["support_regime"].value_counts().to_dict() == {
        "R10": 2072,
        "R01": 1594,
        "R00": 425,
        "R11": 135,
        "control": 228,
    }
    both = audit.loc[audit["split_final"].eq("test_both") & audit["is_treatment"]]
    assert both["support_regime"].value_counts().to_dict() == {
        "R10": 432,
        "R01": 272,
        "R00": 425,
    }
    masks = routing.support_route_masks(audit)
    assert sum(int(mask.sum()) for mask in masks.values()) == len(test)


def test_support_route_ignores_role_labels_and_preserves_canonical_ids():
    metadata = pd.DataFrame(
        {
            "sample_ID": ["fit", "a", "b"],
            "data_source": ["D", "D", "D"],
            "pert_id": ["#1", "#2", "#3"],
            "split_final": ["train", "test_both", "test_both"],
            "strain_role": ["train", "test", "train"],
            "chemical_role": ["train", "test", "train"],
            "Strains": ["S-seen", "S-seen", "S-new"],
            "perturbation_no_concentration": ["Drug seen", "Drug seen", "Drug new"],
        }
    )
    registries = _registries(metadata)
    manifest = build_support_manifest(metadata.iloc[[0]], registries)

    audit = routing.support_route_audit(metadata.iloc[1:].set_index("sample_ID"), manifest)
    assert audit["support_regime"].tolist() == ["R11", "R00"]
    assert audit["strain_canonical_id"].tolist() == ["strain:S-seen", "strain:S-new"]
    assert audit["chemical_canonical_id"].tolist() == [
        "chemical:Drug seen",
        "chemical:Drug new",
    ]


def test_support_routed_predictor_writes_row_audit_and_keeps_legacy_base(monkeypatch, tmp_path):
    import scripts.predict_response_routed_ensemble as predictor

    fit = pd.DataFrame(
        [
            metadata_row("fit_tx", "DrugA", "train", strain="S1"),
            metadata_row("fit_ctrl", "Water", "train", strain="S1"),
        ],
        columns=METADATA_COLUMNS,
    )
    test = pd.DataFrame(
        [
            metadata_row("r11", "DrugA", "test_both", strain="S1"),
            metadata_row("r10", "DrugB", "test_both", strain="S1"),
            metadata_row("r01", "DrugA", "test_both", strain="S2"),
            metadata_row("r00", "DrugB", "test_both", strain="S2"),
            metadata_row("ctrl", "Water", "test_strain_only", strain="S2"),
        ],
        columns=METADATA_COLUMNS,
    )
    manifest = build_support_manifest(
        fit,
        _registries(pd.concat([fit, test], ignore_index=True)),
    )
    run = tmp_path / "run"
    run.mkdir()
    write_json_with_hash(run / "support_manifest.json", manifest)
    torch.save(
        {
            "support_manifest": manifest,
            "support_manifest_sha256": manifest_sha256(manifest),
            "artifact_hashes": {},
            "artifact_chain_sha256": "unit-nonempty-chain",
        },
        run / "checkpoint.pt",
    )
    config = tmp_path / "config.yaml"
    config.write_text("unit: true\n", encoding="utf-8")
    metadata_path = tmp_path / "metadata.csv"
    test.to_csv(metadata_path, index=False)
    base_path = tmp_path / "base.csv"
    pd.DataFrame({"sample_ID": test["sample_ID"], "P1": 1.0}).to_csv(base_path, index=False)
    route_path = tmp_path / "routes.json"
    route_path.write_text(
        json.dumps(
            {
                "model": "unit",
                "routes": {
                    "R10": {
                        "config": str(config),
                        "runs": [str(run)],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "prediction.csv"

    def fake_route_average(config_path, runs, metadata):
        return pd.DataFrame(9.0, index=metadata.index, columns=["P1"]), [{"unit": True}]

    monkeypatch.setattr(predictor, "_route_average", fake_route_average)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict_response_routed_ensemble.py",
            "--base-prediction",
            str(base_path),
            "--metadata",
            str(metadata_path),
            "--route-manifest",
            str(route_path),
            "--output-csv",
            str(output),
        ],
    )
    predictor.main()

    prediction = pd.read_csv(output).set_index("sample_ID")
    assert prediction.loc["r10", "P1"] == 9.0
    assert (prediction.drop(index="r10")["P1"] == 1.0).all()
    audit = pd.read_csv(tmp_path / "route_audit.csv").set_index("sample_ID")
    assert audit["support_regime"].to_dict() == {
        "r11": "R11",
        "r10": "R10",
        "r01": "R01",
        "r00": "R00",
        "ctrl": "control",
    }
    assert audit.loc["r10", "selected_route"] == "R10"
    assert (audit.drop(index="r10")["selected_route"] == "base").all()
    assert {
        "strain_seen",
        "chemical_seen",
        "pair_seen",
        "context_time_seen",
        "strain_expert_reason",
        "chemical_expert_reason",
        "pair_expert_reason",
    } <= set(audit.columns)
    assert bool(audit.loc["r11", "pair_seen"])
    assert audit.loc["r10", "chemical_expert_reason"] == (
        "canonical_support_key_unseen"
    )
    assert audit.loc["ctrl", "chemical_expert_reason"] == (
        "non_treatment_response_disabled"
    )


def test_routed_runs_with_different_fit_support_manifests_fail_hard(tmp_path):
    import scripts.predict_response_routed_ensemble as predictor

    runs = [tmp_path / "run-a", tmp_path / "run-b"]
    for run, marker in zip(runs, ("A", "B")):
        run.mkdir()
        write_json_with_hash(
            run / "support_manifest.json",
            {"schema_version": "unit", "fit_support": marker},
        )
    routes = {
        "R10": {
            "config": str(tmp_path / "config.yaml"),
            "runs": [str(run) for run in runs],
        }
    }
    with pytest.raises(ValueError, match="different fit support manifests"):
        predictor._fit_support_manifest(
            {"routes": routes},
            tmp_path / "route_manifest.json",
            routes,
        )
