"""Validate immutable pre-confirmation evidence receipts for M7/M8."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_LABEL = "LOCAL_STRICT_OOF_NOT_OFFICIAL"
SCALE_SELECTION_RULE_ID = "goai.expert_scale_selection.v2"
EXPECTED_ENTITY_COUNTS = {"chemical": 57, "strain": 6}
EXPECTED_ROLE_COUNTS = {
    "chemical": {"train": 40, "val": 6, "test": 11},
    "strain": {"train": 4, "val": 1, "test": 1},
}
REQUIRED_CALIBRATION_CHECKS = {
    "leave_one_plate_out",
    "plate_label_shuffle",
    "rank_4_8_16",
    "plate_dropout",
    "no_plate_control",
    "observation_metadata_only",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_receipt(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"Evidence receipt is unavailable: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Evidence receipt root must be a mapping: {resolved}")
    sidecar = Path(str(resolved) + ".sha256")
    if sidecar.is_file():
        expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
        if expected != sha256_file(resolved):
            raise ValueError(f"Evidence receipt sidecar hash mismatch: {resolved}")
    return resolved, payload


def validate_identity_audit(path: str | Path) -> dict[str, Any]:
    resolved, payload = load_receipt(path)
    if payload.get("schema_version") != "goai.entity-registry-audit.v1":
        raise ValueError("Identity audit has an unsupported schema")
    if payload.get("entity_counts") != EXPECTED_ENTITY_COUNTS:
        raise ValueError("Identity audit entity counts do not match the locked competition set")
    if payload.get("role_counts") != EXPECTED_ROLE_COUNTS:
        raise ValueError("Identity audit role partitions do not match the locked competition set")
    if payload.get("test_only_structure_failures"):
        raise ValueError("Identity audit reports unresolved test-only structures")
    if payload.get("verified_evidence_failures"):
        raise ValueError("Identity audit reports invalid verified-evidence snapshots")
    issues = payload.get("issues", [])
    if not isinstance(issues, list):
        raise ValueError("Identity audit issues must be a list")
    core_error_codes = {
        str(item.get("code", ""))
        for item in issues
        if isinstance(item, dict)
        and item.get("severity") == "error"
        and str(item.get("code", ""))
        not in {
            "test_only_promotion_blocked",
            "chemical_proxy",
            "strain_proxy",
            "chemical_unresolved",
            "strain_unresolved",
        }
    }
    if core_error_codes:
        raise ValueError(f"Identity audit has core registry errors: {sorted(core_error_codes)}")
    promotion_blockers = payload.get("promotion_blockers", [])
    declared_registries = payload.get("registry_artifacts")
    registry_artifacts: dict[str, dict[str, str]] = {}
    registry_chain_valid = isinstance(declared_registries, dict)
    if registry_chain_valid:
        for entity_type in ("chemical", "strain"):
            record = declared_registries.get(entity_type)
            if not isinstance(record, dict) or not record.get("path") or not record.get(
                "sha256"
            ):
                registry_chain_valid = False
                break
            registry_path = Path(str(record["path"])).resolve()
            if not registry_path.is_file():
                raise ValueError(
                    f"Identity-audited {entity_type} registry is unavailable: "
                    f"{registry_path}"
                )
            actual = sha256_file(registry_path)
            expected = str(record["sha256"])
            if actual != expected:
                raise ValueError(
                    f"Identity-audited {entity_type} registry hash mismatch"
                )
            registry_artifacts[entity_type] = {
                "path": str(registry_path),
                "sha256": actual,
            }
    if not registry_chain_valid:
        raise ValueError(
            "Identity audit lacks the required chemical/strain registry artifact hash chain"
        )
    semantic_ready = bool(
        payload.get("strict_semantic") is True
        and payload.get("ok") is True
        and not promotion_blockers
        and registry_chain_valid
    )
    return {
        "kind": "identity_audit",
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "schema": payload["schema_version"],
        "core_registry_integrity": "valid",
        "semantic_promotion_status": "ready" if semantic_ready else "blocked",
        "promotion_blocker_count": len(promotion_blockers),
        "registry_artifacts": registry_artifacts,
        "registry_artifact_chain_status": "valid",
    }


def validate_calibration_audit(path: str | Path) -> dict[str, Any]:
    resolved, payload = load_receipt(path)
    if payload.get("schema") != "goai.m7_m8.calibration_audit_receipt.v1":
        raise ValueError("Calibration audit has an unsupported schema")
    if payload.get("protocol_label") != PROTOCOL_LABEL:
        raise ValueError("Calibration audit lacks the locked local-OOF label")
    if payload.get("status") != "approved" or payload.get("audit_complete") is not True:
        raise ValueError("Calibration audit is incomplete or has not been explicitly approved")
    checks = payload.get("checks", {})
    if not isinstance(checks, dict):
        raise ValueError("Calibration audit checks must be a mapping")
    failed = sorted(
        name for name in REQUIRED_CALIBRATION_CHECKS if checks.get(name) is not True
    )
    if failed:
        raise ValueError(f"Calibration audit lacks required passing checks: {failed}")
    if not str(payload.get("decision_reason", "")).strip():
        raise ValueError("Calibration approval must record a decision reason")
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Calibration audit lacks deterministic model selection evidence")
    selected_config = selection.get("selected_config")
    required_config = {
        "calibration_rank",
        "calibration_use_plate",
        "calibration_plate_dropout",
        "calibration_plate_shuffle",
    }
    if not isinstance(selected_config, dict) or set(selected_config) != required_config:
        raise ValueError("Calibration audit selected_config is incomplete")
    if bool(selected_config["calibration_plate_shuffle"]):
        raise ValueError("A shuffled plate calibration can never enter confirmation")
    if not str(selection.get("selected_experiment_id", "")).strip():
        raise ValueError("Calibration audit lacks selected_experiment_id")
    for metric in (
        "selected_profile",
        "selected_leave_one_plate_out_fc_pcc",
        "selected_leave_one_plate_out_fc_delta_vs_no_plate",
        "no_plate_macro_fc",
        "plate_shuffle_macro_fc",
        "guardrail_max_drop",
    ):
        if metric not in selection:
            raise ValueError(f"Calibration selection lacks metric evidence: {metric}")
    return {
        "kind": "calibration_audit",
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "schema": payload["schema"],
        "status": "approved",
        "selected_experiment_id": selection["selected_experiment_id"],
        "selected_config": selected_config,
        "selection_evidence": selection,
    }


def validate_fair_expert_audit(path: str | Path) -> dict[str, Any]:
    resolved, payload = load_receipt(path)
    if payload.get("schema") != "fair_expert_receipt_audit_v1":
        raise ValueError("Fair-expert audit has an unsupported schema")
    if payload.get("status") != "valid":
        raise ValueError("Fair-expert receipt audit is not valid")
    if payload.get("source_fingerprints_are_identical") is not True:
        raise ValueError("Fair-expert producers do not share one source fingerprint")
    if int(payload.get("invalid_receipt_rows", -1)) != 0:
        raise ValueError("Fair-expert audit contains invalid fold receipts")
    task_counts = payload.get("task_fit_counts", {})
    if not isinstance(task_counts, dict) or len(task_counts) != 7:
        raise ValueError("Fair-expert audit does not cover all seven locked producers")
    if set(int(value) for value in task_counts.values()) != {14}:
        raise ValueError("Every fair-expert producer must contain exactly 14 discovery fits")
    return {
        "kind": "fair_expert_audit",
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "schema": payload["schema"],
        "status": "valid",
        "source_fingerprint_sha256": payload.get("source_fingerprint_sha256", ""),
    }


def validate_scale_selection(selection: dict[str, Any], path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if selection.get("selection_status") != "selected":
        raise ValueError("Expert-scale selection is incomplete")
    if selection.get("protocol_label") != PROTOCOL_LABEL:
        raise ValueError("Expert-scale selection lacks the locked local-OOF label")
    if selection.get("selection_rule_id") != SCALE_SELECTION_RULE_ID:
        raise ValueError(
            "Expert-scale selection does not use the locked R11+RT pair rule"
        )
    pair_rule = selection.get("pair_selection_rule", {})
    if not isinstance(pair_rule, dict) or pair_rule.get("regimes") != ["R11", "RT"]:
        raise ValueError("Pair scale receipt must precombine R11 and RT")
    selected = selection.get("selected", {})
    if not isinstance(selected, dict) or set(selected) != {"strain", "chemical", "pair"}:
        raise ValueError("Scale nomination must cover strain, chemical, and pair axes")
    for axis in ("strain", "chemical", "pair"):
        if not isinstance(selected[axis], dict) or "scale" not in selected[axis]:
            raise ValueError(f"Scale nomination lacks selected {axis} value")
    pair = selected["pair"]
    if pair.get("scenario") != "R11+RT" or pair.get("guardrail_regimes") != [
        "R11",
        "RT",
    ]:
        raise ValueError(
            "Selected pair scale lacks the locked R11+RT macro/guardrail receipt"
        )
    return {
        "kind": "discovery_expert_scale_nomination",
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "status": "selected",
        "selection_rule_id": SCALE_SELECTION_RULE_ID,
        "binding_to_formal_predictions": False,
        "formal_scale_source": "per_outer_fold_nested_inner_oof",
        "selected": {
            axis: float(selected[axis]["scale"])
            for axis in ("strain", "chemical", "pair")
        },
    }


def validate_preconfirmation_evidence(
    *,
    identity_audit: str | Path,
    calibration_audit: str | Path,
    fair_expert_audit: str | Path,
    scale_selection: dict[str, Any],
    scale_selection_path: str | Path,
) -> dict[str, Any]:
    """Return an immutable evidence bundle or fail closed."""
    return {
        "schema": "goai.m7_m8.preconfirmation_evidence.v1",
        "protocol_label": PROTOCOL_LABEL,
        "status": "valid",
        "identity": validate_identity_audit(identity_audit),
        "calibration": validate_calibration_audit(calibration_audit),
        "fair_experts": validate_fair_expert_audit(fair_expert_audit),
        "expert_scales": validate_scale_selection(
            scale_selection, scale_selection_path
        ),
    }
