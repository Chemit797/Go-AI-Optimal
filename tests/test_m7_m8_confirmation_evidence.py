from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import yaml

from scripts.nightly.build_m7_m8_confirm import materialize_confirm_matrix
from scripts.nightly.confirmation_evidence import (
    PROTOCOL_LABEL,
    validate_calibration_audit,
    validate_fair_expert_audit,
    validate_identity_audit,
    validate_preconfirmation_evidence,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _identity(tmp_path: Path) -> dict:
    registry_artifacts = {}
    for entity_type in ("chemical", "strain"):
        registry = tmp_path / f"{entity_type}.tsv"
        registry.write_text(f"entity\n{entity_type}\n", encoding="utf-8")
        registry_artifacts[entity_type] = {
            "path": str(registry),
            "sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
        }
    return {
        "schema_version": "goai.entity-registry-audit.v1",
        "ok": False,
        "strict_semantic": True,
        "entity_counts": {"chemical": 57, "strain": 6},
        "role_counts": {
            "chemical": {"train": 40, "val": 6, "test": 11},
            "strain": {"train": 4, "val": 1, "test": 1},
        },
        "test_only_structure_failures": [],
        "verified_evidence_failures": [],
        "promotion_blockers": [{"raw_name": "research-only"}],
        "issues": [
            {
                "severity": "error",
                "code": "test_only_promotion_blocked",
                "message": "semantic promotion intentionally blocked",
            }
        ],
        "registry_artifacts": registry_artifacts,
    }


def _calibration() -> dict:
    return {
        "schema": "goai.m7_m8.calibration_audit_receipt.v1",
        "protocol_label": PROTOCOL_LABEL,
        "status": "approved",
        "audit_complete": True,
        "decision_reason": "rank/dropout selected after all locked checks",
        "selection": {
            "selected_experiment_id": "CAL-M7.0-NO-PLATE",
            "selected_config": {
                "calibration_rank": 16,
                "calibration_use_plate": False,
                "calibration_plate_dropout": 0.0,
                "calibration_plate_shuffle": False,
            },
            "selected_profile": {"five_regime_macro_fc_pcc": 0.3},
            "selected_leave_one_plate_out_fc_pcc": 0.3,
            "selected_leave_one_plate_out_fc_delta_vs_no_plate": 0.0,
            "no_plate_macro_fc": 0.3,
            "plate_shuffle_macro_fc": 0.29,
            "guardrail_max_drop": 0.005,
        },
        "checks": {
            "leave_one_plate_out": True,
            "plate_label_shuffle": True,
            "rank_4_8_16": True,
            "plate_dropout": True,
            "no_plate_control": True,
            "observation_metadata_only": True,
        },
    }


def _fair() -> dict:
    return {
        "schema": "fair_expert_receipt_audit_v1",
        "status": "valid",
        "source_fingerprints_are_identical": True,
        "source_fingerprint_sha256": "source",
        "invalid_receipt_rows": 0,
        "task_fit_counts": {f"task-{index}": 14 for index in range(7)},
    }


def _selection() -> dict:
    return {
        "selection_status": "selected",
        "protocol_label": PROTOCOL_LABEL,
        "selection_rule_id": "goai.expert_scale_selection.v2",
        "pair_selection_rule": {"regimes": ["R11", "RT"]},
        "selected": {
            "strain": {"scale": 0.5},
            "chemical": {"scale": 0.75},
            "pair": {
                "scale": 0.25,
                "scenario": "R11+RT",
                "guardrail_regimes": ["R11", "RT"],
            },
        },
    }


def test_preconfirmation_bundle_requires_all_four_receipts(tmp_path: Path) -> None:
    identity = _write(tmp_path / "identity.json", _identity(tmp_path))
    calibration = _write(tmp_path / "calibration.json", _calibration())
    fair = _write(tmp_path / "fair.json", _fair())
    selection_path = _write(tmp_path / "selection.json", _selection())
    receipt = validate_preconfirmation_evidence(
        identity_audit=identity,
        calibration_audit=calibration,
        fair_expert_audit=fair,
        scale_selection=_selection(),
        scale_selection_path=selection_path,
    )
    assert receipt["status"] == "valid"
    assert receipt["identity"]["core_registry_integrity"] == "valid"
    assert receipt["identity"]["semantic_promotion_status"] == "blocked"
    assert receipt["calibration"]["status"] == "approved"
    assert receipt["fair_experts"]["status"] == "valid"
    assert receipt["expert_scales"]["selected"]["pair"] == 0.25


def test_calibration_receipt_fails_closed_when_one_audit_is_missing(tmp_path: Path) -> None:
    payload = _calibration()
    payload["checks"]["plate_label_shuffle"] = False
    path = _write(tmp_path / "calibration.json", payload)
    with pytest.raises(ValueError, match="plate_label_shuffle"):
        validate_calibration_audit(path)


def test_fair_receipt_requires_all_seven_complete_discovery_producers(tmp_path: Path) -> None:
    payload = _fair()
    payload["task_fit_counts"].pop("task-6")
    path = _write(tmp_path / "fair.json", payload)
    with pytest.raises(ValueError, match="seven"):
        validate_fair_expert_audit(path)


def test_identity_core_errors_block_even_if_semantics_are_research_only(tmp_path: Path) -> None:
    payload = _identity(tmp_path)
    payload["issues"].append(
        {"severity": "error", "code": "normalized_collision", "message": "bad"}
    )
    path = _write(tmp_path / "identity.json", payload)
    with pytest.raises(ValueError, match="normalized_collision"):
        validate_identity_audit(path)


def test_hashed_calibration_receipt_tampering_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "calibration.json", _calibration())
    (tmp_path / "calibration.json.sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selection"]["selected_config"]["calibration_use_plate"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar hash mismatch"):
        validate_calibration_audit(path)


def test_selected_calibration_is_injected_into_every_materialized_candidate(
    tmp_path: Path,
) -> None:
    identity = _write(tmp_path / "identity.json", _identity(tmp_path))
    calibration = _write(tmp_path / "calibration.json", _calibration())
    fair = _write(tmp_path / "fair.json", _fair())
    selection_path = _write(tmp_path / "selection.json", _selection())
    evidence = validate_preconfirmation_evidence(
        identity_audit=identity,
        calibration_audit=calibration,
        fair_expert_audit=fair,
        scale_selection=_selection(),
        scale_selection_path=selection_path,
    )
    project = Path(__file__).resolve().parents[1]
    template = yaml.safe_load(
        (
            project / "configs/nightly/20260813-m7-m8/confirm_candidates.yaml"
        ).read_text(encoding="utf-8")
    )
    matrix = materialize_confirm_matrix(
        template, _selection(), preconfirmation_evidence=evidence
    )
    expected = _calibration()["selection"]["selected_config"]
    assert matrix["fixed_calibration_selection"]["selected_config"] == expected
    assert matrix["experiments"]
    for item in matrix["experiments"]:
        assert item["fixed_calibration"] == expected
        for key, value in expected.items():
            assert item["overrides"]["model"][key] == value


def test_identity_promotion_is_bound_to_untampered_registry_artifacts(
    tmp_path: Path,
) -> None:
    registries = {}
    for entity_type in ("chemical", "strain"):
        registry = tmp_path / f"{entity_type}.tsv"
        registry.write_text(f"entity\n{entity_type}\n", encoding="utf-8")
        registries[entity_type] = {
            "path": str(registry),
            "sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
        }
    payload = _identity(tmp_path)
    payload.update(
        {
            "ok": True,
            "promotion_blockers": [],
            "issues": [],
            "registry_artifacts": registries,
        }
    )
    receipt_path = _write(tmp_path / "identity.json", payload)
    receipt = validate_identity_audit(receipt_path)
    assert receipt["semantic_promotion_status"] == "ready"
    assert receipt["registry_artifact_chain_status"] == "valid"

    Path(registries["chemical"]["path"]).write_text(
        "entity\ntampered\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="registry hash mismatch"):
        validate_identity_audit(receipt_path)


def test_identity_audit_without_registry_hash_chain_is_rejected(
    tmp_path: Path,
) -> None:
    payload = _identity(tmp_path)
    payload.pop("registry_artifacts")
    receipt_path = _write(tmp_path / "identity.json", payload)
    with pytest.raises(ValueError, match="registry artifact hash chain"):
        validate_identity_audit(receipt_path)
