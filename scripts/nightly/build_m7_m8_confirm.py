"""Materialize a runnable, evidence-gated 4-fold M7/M8 confirm matrix.

The candidate template is intentionally not runnable.  This command refuses
to materialize confirmation jobs until the discovery nomination, registry
identity, calibration, and fair-expert receipts pass their locked contracts.
Discovery-wide expert scales are never injected: every formal outer fold
chooses its scales from nested inner OOF on that outer fold's training rows.
"""

from __future__ import annotations

import argparse
import hashlib
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    # Module execution (`python -m scripts.nightly.build_m7_m8_confirm`).
    from scripts.nightly.confirmation_evidence import validate_preconfirmation_evidence
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    # `python scripts/nightly/build_m7_m8_confirm.py` places scripts/nightly on
    # sys.path rather than the project root.
    from confirmation_evidence import validate_preconfirmation_evidence


ALLOWED_SCALES = {0.0, 0.25, 0.5, 0.75, 1.0}
PROTOCOL_LABEL = "LOCAL_STRICT_OOF_NOT_OFFICIAL"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chosen_scale(selection: dict[str, Any], axis: str) -> float:
    selected = selection.get("selected", {}).get(axis)
    if not isinstance(selected, dict) or "scale" not in selected:
        raise ValueError(f"Missing selected {axis} expert scale")
    value = float(selected["scale"])
    if value not in ALLOWED_SCALES:
        raise ValueError(f"Selected {axis} scale {value} is outside the locked grid")
    return value


def _formal_semantic_coverage(
    item: dict[str, Any],
    identity_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed when a formal M8 semantic axis has no admitted fit entity."""

    entity = item.get("overrides", {}).get("entity", {})
    if not isinstance(entity, dict):
        return {"required_axes": [], "status": "not_applicable"}
    required: list[tuple[str, str]] = []
    if entity.get("chemical_map") or entity.get("chemical_features"):
        required.append(("chemical", "chemical_registry"))
    if entity.get("strain_features"):
        required.append(("strain", "strain_registry"))
    if not required:
        return {"required_axes": [], "status": "not_applicable"}
    policy = str(entity.get("semantic_identity_policy", "verified_only"))
    allowed = (
        {"verified"}
        if policy == "verified_only"
        else {"verified", "high_confidence_candidate"}
    )
    coverage: dict[str, Any] = {
        "required_axes": [axis for axis, _ in required],
        "semantic_identity_policy": policy,
        "axes": {},
    }
    audited_registries = (
        identity_receipt.get("registry_artifacts", {})
        if isinstance(identity_receipt, dict)
        else {}
    )
    for axis, registry_key in required:
        registry = entity.get(registry_key)
        if not registry:
            raise ValueError(f"{item['id']} requires {axis} semantics without a registry")
        path = Path(str(registry))
        path = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        if not path.is_file():
            raise ValueError(f"{item['id']} semantic registry is unavailable: {path}")
        actual_registry_sha = _sha256(path)
        if identity_receipt is not None:
            audit_record = (
                audited_registries.get(axis)
                if isinstance(audited_registries, dict)
                else None
            )
            if not isinstance(audit_record, dict) or str(
                audit_record.get("sha256", "")
            ) != actual_registry_sha:
                raise ValueError(
                    f"{item['id']} {axis} registry is not the identity-audited artifact"
                )
        table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        fit_rows = table.loc[table["role"].astype(str).eq("train")]
        if axis == "chemical":
            for flag in ("is_control", "is_quality_control"):
                if flag in fit_rows:
                    is_flagged = fit_rows[flag].astype(str).str.lower().isin(
                        {"1", "true", "yes"}
                    )
                    fit_rows = fit_rows.loc[~is_flagged]
        admitted = fit_rows["mapping_status"].astype(str).isin(allowed)
        count = int(admitted.sum())
        coverage["axes"][axis] = {
            "fit_entities": int(len(fit_rows)),
            "admitted_fit_entities": count,
            "registry": str(path),
            "registry_sha256": actual_registry_sha,
            "allowed_mapping_status": sorted(allowed),
        }
        if count == 0:
            raise ValueError(
                f"{item['id']} has zero admitted {axis} semantics in fit roles "
                f"under {policy}; formal confirmation is blocked"
            )
    coverage["status"] = "ready"
    return coverage


def materialize_confirm_matrix(
    template: dict[str, Any],
    selection: dict[str, Any],
    *,
    absolute_base_config: str | None = None,
    preconfirmation_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if selection.get("selection_status") != "selected":
        raise ValueError("Expert-scale selection is incomplete; confirmation is blocked")
    if selection.get("protocol_label") != PROTOCOL_LABEL:
        raise ValueError("Expert-scale selection lacks the locked local-OOF protocol label")
    # Retain the quick/global result only as an immutable, non-binding
    # nomination.  Reading these values here is safe because they are never
    # copied into a formal model config or its predictions.
    nominated_scales = {
        axis: _chosen_scale(selection, axis)
        for axis in ("strain", "chemical", "pair")
    }
    candidates = template.get("confirm_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("confirm template must contain confirm_candidates")
    output = {
        key: deepcopy(value)
        for key, value in template.items()
        if key not in {
            "confirm_candidates",
            "requires_expert_scale_selection",
            "confirm_scenarios",
            "confirm_seeds",
            "full_general",
            "registry_gate",
            "materialize_joint_confirm_variants",
            "joint_confirm_epochs",
        }
    }
    if absolute_base_config is not None:
        output["base_config"] = absolute_base_config
    output["protocol_label"] = PROTOCOL_LABEL
    if preconfirmation_evidence is not None:
        if (
            preconfirmation_evidence.get("schema")
            != "goai.m7_m8.preconfirmation_evidence.v1"
            or preconfirmation_evidence.get("status") != "valid"
            or preconfirmation_evidence.get("protocol_label") != PROTOCOL_LABEL
        ):
            raise ValueError("Invalid pre-confirmation evidence bundle")
        output["preconfirmation_evidence"] = deepcopy(preconfirmation_evidence)
    output["non_binding_expert_scale_nomination"] = {
        "strain_expert_scale": nominated_scales["strain"],
        "chemical_expert_scale": nominated_scales["chemical"],
        "pair_expert_scale": nominated_scales["pair"],
        "method": selection.get("method", ""),
        "binding_to_formal_predictions": False,
        "formal_policy": "per_outer_fold_nested_inner_oof",
    }
    expanded_candidates: list[dict[str, Any]] = []
    blocked_candidates: list[dict[str, Any]] = []
    identity_receipt = (
        preconfirmation_evidence.get("identity", {})
        if isinstance(preconfirmation_evidence, dict)
        else {}
    )
    identity_semantics_ready = bool(
        isinstance(identity_receipt, dict)
        and identity_receipt.get("semantic_promotion_status") == "ready"
    )
    fixed_calibration: dict[str, Any] | None = None
    if preconfirmation_evidence is not None:
        calibration_receipt = preconfirmation_evidence.get("calibration", {})
        candidate_calibration = (
            calibration_receipt.get("selected_config", {})
            if isinstance(calibration_receipt, dict)
            else {}
        )
        required_calibration = {
            "calibration_rank",
            "calibration_use_plate",
            "calibration_plate_dropout",
            "calibration_plate_shuffle",
        }
        if not isinstance(candidate_calibration, dict) or set(candidate_calibration) != required_calibration:
            raise ValueError(
                "Pre-confirmation Calibration receipt lacks a complete selected_config"
            )
        if bool(candidate_calibration["calibration_plate_shuffle"]):
            raise ValueError("Shuffled plate calibration cannot be injected into confirmation")
        fixed_calibration = deepcopy(candidate_calibration)
        output["fixed_calibration_selection"] = {
            "selected_experiment_id": calibration_receipt.get(
                "selected_experiment_id", ""
            ),
            "selected_config": deepcopy(fixed_calibration),
            "receipt_sha256": calibration_receipt.get("sha256", ""),
        }
    emit_joint = bool(template.get("materialize_joint_confirm_variants", False))
    joint_epochs = int(template.get("joint_confirm_epochs", 0))
    if emit_joint and joint_epochs <= 0:
        raise ValueError("joint_confirm_epochs must be positive when joint variants are enabled")
    for raw in candidates:
        frozen = deepcopy(raw)
        joint_primary_control = frozen.pop("joint_primary_control", None)
        is_formal_m8 = (
            frozen.get("kind") == "model_confirm"
            and str(frozen.get("model_id", "")).startswith("M8")
        )
        coverage_receipt: dict[str, Any] | None = None
        if is_formal_m8 and not identity_semantics_ready:
            blocked_candidates.append(
                {
                    "id": str(frozen.get("id", "")),
                    "model_id": str(frozen.get("model_id", "")),
                    "reason": (
                        "identity_semantic_promotion_blocked"
                        if preconfirmation_evidence is not None
                        else "identity_semantic_evidence_missing"
                    ),
                    "identity_semantic_promotion_status": str(
                        identity_receipt.get("semantic_promotion_status", "missing")
                    ),
                }
            )
            continue
        if is_formal_m8:
            # A strict identity receipt alone is insufficient: an M8 axis is
            # untrainable when verified-only admits no fit-role entity.  This
            # registry-level preflight is followed by a per-fold builder gate.
            try:
                coverage_receipt = _formal_semantic_coverage(
                    frozen, identity_receipt
                )
            except (ValueError, KeyError) as error:
                blocked_candidates.append(
                    {
                        "id": str(frozen.get("id", "")),
                        "model_id": str(frozen.get("model_id", "")),
                        "reason": "semantic_training_coverage_not_ready",
                        "detail": str(error),
                        "identity_semantic_promotion_status": "ready",
                    }
                )
                continue
        if bool(frozen.get("materialization_blocked", False)):
            if is_formal_m8 and coverage_receipt is not None:
                # The template records the current blocked snapshot.  A fresh,
                # strict identity receipt plus non-empty verified fit coverage
                # is the only machine-authorized way to clear that snapshot.
                frozen.pop("materialization_blocked", None)
                frozen.pop("materialization_block_reason", None)
            else:
                blocked_candidates.append(
                    {
                        "id": str(frozen.get("id", "")),
                        "model_id": str(frozen.get("model_id", "")),
                        "reason": str(
                            frozen.get(
                                "materialization_block_reason",
                                "formal_preflight_blocked",
                            )
                        ),
                    }
                )
                continue
        if coverage_receipt is not None:
            frozen["semantic_coverage_preflight"] = coverage_receipt
        expanded_candidates.append(frozen)
        model = frozen.get("overrides", {}).get("model", {})
        expert_epochs = sum(
            int(model.get(name, 0))
            for name in (
                "strain_expert_epochs",
                "chemical_expert_epochs",
                "pair_expert_epochs",
            )
        )
        if emit_joint and expert_epochs > 0:
            joint = deepcopy(frozen)
            joint["id"] = f"{joint['id']}-JOINT"
            joint["model_id"] = f"{joint.get('model_id', joint['id'])}-joint"
            joint["confirmation_training_variant"] = "joint_finetune"
            joint_model = joint.setdefault("overrides", {}).setdefault("model", {})
            staged = sum(
                int(joint_model.get(name, 0))
                for name in (
                    "universal_epochs",
                    "strain_expert_epochs",
                    "chemical_expert_epochs",
                    "pair_expert_epochs",
                )
            )
            joint_model["epochs"] = staged + joint_epochs
            controls = joint.get("required_negative_controls")
            if isinstance(controls, list):
                joint["required_negative_controls"] = [
                    f"{value}-JOINT" for value in controls
                ]
            if isinstance(joint_primary_control, str) and joint_primary_control:
                joint["primary_control"] = joint_primary_control
            else:
                primary = joint.get("primary_control")
                if isinstance(primary, str) and primary:
                    joint["primary_control"] = f"{primary}-JOINT"
            expanded_candidates.append(joint)

    experiments: list[dict[str, Any]] = []
    ids: set[str] = set()
    for raw in expanded_candidates:
        item = deepcopy(raw)
        experiment_id = str(item["id"])
        if experiment_id in ids:
            raise ValueError(f"Duplicate confirm experiment id: {experiment_id}")
        ids.add(experiment_id)
        policy = str(item.pop("scale_policy", "none"))
        if policy not in {"none", "strain", "chemical", "both", "all"}:
            raise ValueError(f"Unknown scale policy {policy!r} for {experiment_id}")
        model = item.setdefault("overrides", {}).setdefault("model", {})
        if fixed_calibration is not None:
            model.update(deepcopy(fixed_calibration))
            item["fixed_calibration"] = deepcopy(fixed_calibration)
        # Canonical training scale is always one.  The OOF producer recomposes
        # named components with independently selected per-outer-fold scales.
        model.update(
            {
                "strain_expert_scale": 1.0,
                "chemical_expert_scale": 1.0,
                "pair_expert_scale": 1.0,
                "nested_expert_scale_selection": True,
                "nested_expert_scale_inner_folds": 2,
            }
        )
        item["nested_scale_policy"] = policy
        item["formal_expert_scale_selection"] = {
            "method": "per_outer_fold_nested_inner_oof",
            "inner_n_folds": 2,
            "canonical_training_scales": {
                "strain": 1.0,
                "chemical": 1.0,
                "pair": 1.0,
            },
            "global_nomination_binding": False,
        }
        if str(model.get("response_prior_mode", "none")) != "none":
            raise ValueError(
                f"Target-stat prototype {experiment_id} is research-only and cannot "
                "enter formal model confirmation"
            )
        if item.get("kind") == "model_confirm":
            if str(item.get("model_id", "")).startswith("M8"):
                entity = item.setdefault("overrides", {}).setdefault("entity", {})
                if str(entity.get("semantic_identity_policy", "")) != "verified_only":
                    raise ValueError(
                        f"Formal M8 confirmation must use verified_only identity policy: "
                        f"{experiment_id}"
                    )
                if not bool(
                    entity.get("semantic_training_coverage_required", False)
                ):
                    raise ValueError(
                        "Formal M8 confirmation must enforce fold-fit semantic "
                        f"training coverage: {experiment_id}"
                    )
                if bool(entity.get("allow_proxy_semantics", False)) or str(
                    entity.get("chemical_structure_view", "exact")
                ) == "parent":
                    raise ValueError(
                        f"Parent/proxy semantics are research-only and cannot be "
                        f"materialized for formal confirmation: {experiment_id}"
                    )
                if entity.get("chemical_map") and not bool(
                    entity.get("chemical_structure_manifest_required", False)
                ):
                    raise ValueError(
                        f"Formal Morgan input lacks its source/shuffled manifest: "
                        f"{experiment_id}"
                    )
                if entity.get("chemical_features") and not bool(
                    entity.get("chemical_features_manifest_required", False)
                ):
                    raise ValueError(
                        f"Formal chemical embedding lacks its source manifest: "
                        f"{experiment_id}"
                    )
                if entity.get("strain_features") and not bool(
                    entity.get("strain_features_manifest_required", False)
                ):
                    raise ValueError(
                        f"Formal strain semantics lack their evidence manifest: "
                        f"{experiment_id}"
                    )
            if not bool(model.get("fold_matched_universal_warm_start", False)):
                raise ValueError(
                    f"Formal confirmation lacks fold-matched warm start: {experiment_id}"
                )
            staged = sum(
                int(model.get(name, 0))
                for name in (
                    "universal_epochs",
                    "strain_expert_epochs",
                    "chemical_expert_epochs",
                    "pair_expert_epochs",
                )
            )
            if str(item.get("confirmation_training_variant", "")) in {
                "frozen_residual_only",
                "frozen_residual_only_negative_control",
            } and staged != int(model.get("epochs", -1)):
                raise ValueError(
                    f"Frozen confirmation unexpectedly includes joint epochs: {experiment_id}"
                )
            item["semantic_coverage_preflight"] = _formal_semantic_coverage(
                item, identity_receipt
            )
        experiments.append(item)
    formal = {
        str(item["id"]): item
        for item in experiments
        if item.get("kind") == "model_confirm"
    }
    parent_controls = [
        item
        for item in formal.values()
        if item.get("confirmation_training_variant") == "universal_parent_control"
    ]
    if len(parent_controls) != 1:
        raise ValueError("Formal confirmation requires exactly one universal parent control")
    base_universal_epochs = int(
        parent_controls[0]["overrides"]["model"]["universal_epochs"]
    )

    def universal_update_epochs(item: dict[str, Any]) -> int:
        model = item["overrides"]["model"]
        staged = sum(
            int(model.get(name, 0))
            for name in (
                "universal_epochs",
                "strain_expert_epochs",
                "chemical_expert_epochs",
                "pair_expert_epochs",
            )
        )
        return int(model.get("universal_epochs", 0)) + max(
            0, int(model.get("epochs", 0)) - staged
        )

    for item in formal.values():
        model = item["overrides"]["model"]
        variant = str(item.get("confirmation_training_variant", ""))
        universal_epochs = int(model.get("universal_epochs", 0))
        update_epochs = universal_update_epochs(item)
        item["universal_update_budget"] = {
            "fold_matched_universal_phase_epochs": universal_epochs,
            "total_universal_update_epochs": update_epochs,
        }
        if variant in {
            "frozen_residual_only",
            "frozen_residual_only_negative_control",
            "joint_finetune",
        } and universal_epochs != base_universal_epochs:
            raise ValueError(
                f"Expert confirmation {item['id']} does not share the locked "
                f"U{base_universal_epochs} fold-matched warm start"
            )
        if variant == "joint_finetune" and update_epochs != (
            base_universal_epochs + joint_epochs
        ):
            raise ValueError(
                f"Joint confirmation {item['id']} has {update_epochs} universal "
                f"updates; expected {base_universal_epochs + joint_epochs}"
            )
        if variant == "same_universal_update_control":
            expert_epochs = sum(
                int(model.get(name, 0))
                for name in (
                    "strain_expert_epochs",
                    "chemical_expert_epochs",
                    "pair_expert_epochs",
                )
            )
            if expert_epochs != 0 or update_epochs != (
                base_universal_epochs + joint_epochs
            ):
                raise ValueError(
                    f"Same-update control {item['id']} must be a pure "
                    f"U{base_universal_epochs + joint_epochs} model"
                )

    for item in formal.values():
        if item.get("promotion_eligible") is False:
            continue
        control_id = str(item.get("primary_control", ""))
        if not control_id or control_id not in formal:
            raise ValueError(
                f"Promotion candidate {item['id']} lacks a materialized primary control"
            )
        candidate_updates = universal_update_epochs(item)
        control_updates = universal_update_epochs(formal[control_id])
        if candidate_updates != control_updates:
            raise ValueError(
                f"Candidate/control universal-update mismatch for {item['id']}: "
                f"{candidate_updates} vs {control_updates} ({control_id})"
            )
        negative_controls = item.get("required_negative_controls", [])
        if not isinstance(negative_controls, list):
            raise ValueError(
                f"Confirmation {item['id']} required_negative_controls must be a list"
            )
        for negative_id in map(str, negative_controls):
            if negative_id not in formal:
                raise ValueError(
                    f"Confirmation {item['id']} references missing negative control "
                    f"{negative_id}"
                )
            if formal[negative_id].get("promotion_eligible") is not False:
                raise ValueError(
                    f"Negative control {negative_id} must be promotion_eligible=false"
                )
            negative_updates = universal_update_epochs(formal[negative_id])
            if negative_updates != candidate_updates:
                raise ValueError(
                    f"Candidate/negative-control universal-update mismatch for "
                    f"{item['id']}: {candidate_updates} vs {negative_updates} "
                    f"({negative_id})"
                )
    output["experiments"] = experiments
    output["blocked_confirm_candidates"] = blocked_candidates
    return output


def _write_blocked_candidates(path: Path) -> Path:
    # Pair scale is a formal requirement only when a selected template asks
    # for the `all` policy.  Listing it here makes the missing receipt visible
    # before materialization without changing historical two-axis screens.
    rows = [
        {
            "axis": axis,
            "candidate_scale": value,
            "status": "SCREEN_REQUIRED_NOT_SELECTED",
            "required_scenario": {
                "strain": "R10",
                "chemical": "R01",
                "pair": "R11+RT",
            }[axis],
            "protocol_label": PROTOCOL_LABEL,
        }
        for axis in ("strain", "chemical", "pair")
        for value in sorted(ALLOWED_SCALES)
    ]
    candidate_path = path.with_suffix(".scale_candidates.csv")
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(candidate_path, index=False)
    # Replace any stale runnable matrix with a deliberately non-runnable
    # receipt. run_matrix requires an `experiments` list and will reject this.
    blocked = {
        "status": "BLOCKED_SCALE_SCREEN_INCOMPLETE",
        "protocol_label": PROTOCOL_LABEL,
        "candidate_table": str(candidate_path),
        "note": "No confirmation experiments were materialized.",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(blocked, sort_keys=False), encoding="utf-8")
    temporary.replace(path)
    return candidate_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument(
        "--identity-audit",
        required=True,
        help="JSON receipt emitted by scripts/audit_entity_registry.py",
    )
    parser.add_argument(
        "--calibration-audit",
        required=True,
        help="Approved complete calibration-audit receipt",
    )
    parser.add_argument(
        "--fair-expert-audit",
        required=True,
        help="receipt_audit.json emitted by audit_fair_expert_receipts.py",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    template_path = Path(args.template).resolve()
    selection_path = Path(args.selection).resolve()
    output_path = Path(args.output).resolve()
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not selection_path.is_file():
        candidate_path = _write_blocked_candidates(output_path)
        raise SystemExit(
            "Confirmation not materialized: no expert-scale selection receipt. "
            f"Candidate table: {candidate_path}"
        )
    selection = yaml.safe_load(selection_path.read_text(encoding="utf-8"))
    if not isinstance(template, dict) or not isinstance(selection, dict):
        raise ValueError("Template and selection roots must be mappings")
    if selection.get("selection_status") != "selected":
        candidate_path = _write_blocked_candidates(output_path)
        raise SystemExit(
            "Confirmation not materialized: scale screen is incomplete. "
            f"Candidate table: {candidate_path}"
        )

    base = Path(str(template["base_config"]))
    base = base if base.is_absolute() else (template_path.parent / base).resolve()
    evidence = validate_preconfirmation_evidence(
        identity_audit=args.identity_audit,
        calibration_audit=args.calibration_audit,
        fair_expert_audit=args.fair_expert_audit,
        scale_selection=selection,
        scale_selection_path=selection_path,
    )
    matrix = materialize_confirm_matrix(
        template,
        selection,
        absolute_base_config=str(base),
        preconfirmation_evidence=evidence,
    )
    matrix["selection_receipt"] = {
        "path": str(selection_path),
        "sha256": _sha256(selection_path),
        "materialized_at": datetime.now().isoformat(timespec="seconds"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(matrix, sort_keys=False, allow_unicode=True), encoding="utf-8")
    temporary.replace(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
