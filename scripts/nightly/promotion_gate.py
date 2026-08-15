#!/usr/bin/env python3
"""Issue a strict M7/M8 promotion receipt from completed confirmation OOF.

This is deliberately a post-confirmation consumer.  Discovery/quick screens
can nominate candidates, but cannot satisfy this contract.  Confidence
intervals resample held-out biological entities, never folds or individual
protein observations.

All reported scores are ``LOCAL_STRICT_OOF_NOT_OFFICIAL``.  This module does
not call, approximate, or claim an official GOAI leaderboard score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from goai_baseline.official_metrics import (
    _control_deltas,
    _frozen_delta_references,
    _reference_for_rows,
)
from goai_baseline.preprocess import prepare_data
from goai_baseline.schema import (
    CHEMICAL,
    MATCH_CONTROL_FIELDS,
    SAMPLE_ID,
    STRAIN,
    control_mask,
)
from goai_response.config import load_response_config
from goai_response.entities import normalize_entity_key
from goai_response.nested_scale import (
    PROTOCOL as NESTED_SCALE_PROTOCOL,
    validate_receipt as validate_nested_scale_receipt,
)
from goai_response.oof import make_fold_slices


PROTOCOL_LABEL = "LOCAL_STRICT_OOF_NOT_OFFICIAL"
CONFIRM_SEEDS = (42, 52, 62)
CONFIRM_SCENARIOS = ("R00", "R10", "R01", "R11", "RT")
PRIMARY_FC_MIN_DELTA = 0.01
NEGATIVE_CONTROL_FC_MIN_DELTA = 0.0
HIGH_EFFECT_MAX_DROP = 0.005
BOOTSTRAP_DRAWS = 20_000

SCENARIO_CLUSTER_AXES: dict[str, tuple[tuple[str, ...], ...]] = {
    # In double-unseen OOF both entity axes must generalize.  Requiring both
    # axis-level CIs avoids treating the strain x chemical grid as independent.
    "R00": ((STRAIN,), (CHEMICAL,)),
    "R10": ((CHEMICAL,),),
    "R01": ((STRAIN,),),
    "R11": ((STRAIN, CHEMICAL),),
    "RT": (("time_group",),),
}
SCENARIO_RESIDUALS: dict[str, tuple[str, ...]] = {
    # R00 references are intentionally unavailable: both target-derived entity
    # references are held out.  A candidate requiring R00 must also name at
    # least one other regime with an identifiable residual guardrail.
    "R00": (),
    "R10": ("context_residual_pcc",),
    "R01": ("drug_residual_pcc",),
    "R11": ("context_residual_pcc", "drug_residual_pcc"),
    "RT": ("context_residual_pcc", "drug_residual_pcc"),
}
GATE_METRICS = (
    "fc_pcc",
    "context_residual_pcc",
    "drug_residual_pcc",
    "high_effect_pcc",
    "high_effect_f1",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _load_prediction(run: Path, scenario: str) -> pd.DataFrame:
    path = run / "oof_predictions" / f"{scenario}.npz"
    if not path.is_file():
        raise ValueError(f"Missing OOF predictions: {path}")
    with np.load(path, allow_pickle=False) as payload:
        values = np.asarray(payload["values"], dtype=np.float64)
        sample_ids = pd.Index(payload["sample_ids"].astype(str), name=SAMPLE_ID)
        proteins = payload["protein_ids"].astype(str).tolist()
    if sample_ids.has_duplicates:
        raise ValueError(f"Duplicate OOF sample IDs: {path}")
    return pd.DataFrame(values, index=sample_ids, columns=proteins).sort_index()


def _matrix_metadata(root: Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    environment_path = root / "environment.json"
    if not environment_path.is_file():
        raise ValueError("Confirmation root lacks environment.json; discovery roots cannot promote")
    environment = _load_json(environment_path)
    matrix_value = str(environment.get("matrix", "")).strip()
    if not matrix_value:
        raise ValueError("Confirmation environment does not identify its matrix")
    matrix_path = Path(matrix_value).resolve()
    if not matrix_path.is_file():
        raise ValueError(f"Confirmation matrix is unavailable: {matrix_path}")
    environment_matrix_hash = str(environment.get("matrix_sha256", ""))
    if environment_matrix_hash and environment_matrix_hash != _sha256(matrix_path):
        raise ValueError("Confirmation matrix changed after the run environment was written")
    payload = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Confirmation matrix root must be a mapping")
    if payload.get("protocol_label") != PROTOCOL_LABEL:
        raise ValueError("Confirmation matrix lacks the locked local-OOF label")
    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("Confirmation matrix must contain runnable experiments")
    metadata = {
        str(item["id"]): dict(item)
        for item in experiments
        if isinstance(item, dict) and "id" in item
    }
    return matrix_path, metadata


def _assignment_hash(run: Path, scenario: str) -> str:
    path = run / "fold_assignments.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    frame = frame.loc[frame["scenario"].eq(scenario)].copy()
    columns = [
        column
        for column in (
            "scenario",
            "fold",
            SAMPLE_ID,
            STRAIN,
            CHEMICAL,
            "time_group",
            "eligible",
            "exclusion_reason",
        )
        if column in frame
    ]
    frame = frame.loc[:, columns].sort_values(["fold", SAMPLE_ID])
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def _validate_confirmation_runs(
    root: Path,
    producer_ids: Iterable[str],
    required_scenarios: tuple[str, ...],
) -> dict[str, Any]:
    """Fail closed unless every producer is a complete locked confirmation."""
    matrix_path, metadata = _matrix_metadata(root)
    matrix_payload = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    nomination_contract = (
        matrix_payload.get("non_binding_expert_scale_nomination", {})
        if isinstance(matrix_payload, dict)
        else {}
    )
    nested_scale_required = "non_binding_expert_scale_nomination" in matrix_payload
    if nested_scale_required and not (
        isinstance(nomination_contract, dict)
        and nomination_contract.get("binding_to_formal_predictions") is False
        and nomination_contract.get("formal_policy")
        == "per_outer_fold_nested_inner_oof"
    ):
        raise ValueError(
            "Formal scale nomination is not explicitly non-binding/nested-inner-OOF"
        )
    checks: list[dict[str, Any]] = []
    source_hashes: set[str] = set()
    input_hashes: set[str] = set()
    assignment_hashes: dict[tuple[int, str], set[str]] = defaultdict(set)
    run_contracts: list[dict[str, Any]] = []
    for producer_id in producer_ids:
        item = metadata.get(producer_id)
        if item is None or item.get("kind") != "model_confirm":
            raise ValueError(f"{producer_id} is not declared as kind=model_confirm")
        expected_model = item.get("overrides", {}).get("model", {})
        if nested_scale_required:
            formal_scale = item.get("formal_expert_scale_selection", {})
            if (
                expected_model.get("nested_expert_scale_selection") is not True
                or int(expected_model.get("nested_expert_scale_inner_folds", 0)) < 2
                or any(
                    float(expected_model.get(name, float("nan"))) != 1.0
                    for name in (
                        "strain_expert_scale",
                        "chemical_expert_scale",
                        "pair_expert_scale",
                    )
                )
                or not isinstance(formal_scale, dict)
                or formal_scale.get("method")
                != "per_outer_fold_nested_inner_oof"
                or formal_scale.get("global_nomination_binding") is not False
            ):
                raise ValueError(
                    f"{producer_id} does not declare locked per-outer-fold nested scales"
                )
        producer = root / "producers" / producer_id
        actual_seeds = {
            int(path.name[1:])
            for path in producer.glob("S*")
            if path.is_dir() and path.name[1:].isdigit()
        }
        if actual_seeds != set(CONFIRM_SEEDS):
            raise ValueError(
                f"{producer_id} seed set is {sorted(actual_seeds)}; "
                f"required={list(CONFIRM_SEEDS)}"
            )
        for seed in CONFIRM_SEEDS:
            run = producer / f"S{seed}"
            required_files = (
                "run_contract.json",
                "oof_manifest.json",
                "oof_summary.csv",
                "fold_assignments.csv",
            )
            missing = [name for name in required_files if not (run / name).is_file()]
            if missing:
                raise ValueError(f"Incomplete confirmation {producer_id}/S{seed}: {missing}")
            contract = _load_json(run / "run_contract.json")
            manifest = _load_json(run / "oof_manifest.json")
            scenarios = tuple(contract.get("scenarios", ()))
            model = contract.get("effective_config", {}).get("model", {})
            expected_epochs = int(expected_model.get("epochs", -1))
            nested_manifest = manifest.get("nested_expert_scale_selection", {})
            nested_contract_valid = bool(
                not nested_scale_required
                or (
                    model.get("nested_expert_scale_selection") is True
                    and int(model.get("nested_expert_scale_inner_folds", 0)) >= 2
                    and all(
                        float(model.get(name, float("nan"))) == 1.0
                        for name in (
                            "strain_expert_scale",
                            "chemical_expert_scale",
                            "pair_expert_scale",
                        )
                    )
                    and isinstance(nested_manifest, dict)
                    and nested_manifest.get("enabled") is True
                    and nested_manifest.get("protocol") == NESTED_SCALE_PROTOCOL
                    and nested_manifest.get("global_scale_used") is False
                    and nested_manifest.get("outer_validation_labels_used") is False
                )
            )
            valid = bool(
                contract.get("protocol") == "support_regime_oof_run_contract_v3"
                and manifest.get("protocol") == "support_regime_oof_v2"
                and int(contract.get("n_folds", -1)) == 4
                and int(manifest.get("n_folds", -1)) == 4
                and int(contract.get("seed", -1)) == 42
                and int(contract.get("model_seed", -1)) == seed
                and int(manifest.get("model_seed", -1)) == seed
                and expected_epochs >= 64
                and int(model.get("epochs", -1)) == expected_epochs
                and set(CONFIRM_SCENARIOS).issubset(scenarios)
                and set(required_scenarios).issubset(scenarios)
                and manifest.get("audit_only") is False
                and nested_contract_valid
            )
            if not valid:
                raise ValueError(
                    f"{producer_id}/S{seed} violates the locked 4-fold, "
                    "fold-seed-42 confirmation contract or its declared epoch count"
                )
            completed = list((run / "folds").glob("*/completed.json"))
            if len(completed) != 44:
                raise ValueError(
                    f"{producer_id}/S{seed} has {len(completed)} completed fits; expected 44"
                )
            training_receipts: list[dict[str, Any]] = []
            nested_receipts: list[dict[str, Any]] = []
            for completed_path in completed:
                fold_completion = _load_json(completed_path)
                training = fold_completion.get("training_receipt")
                if not isinstance(training, dict):
                    raise ValueError(f"Missing training receipt: {completed_path}")
                if training.get("enabled") is not True:
                    raise ValueError(f"Fold-matched warm start is disabled: {completed_path}")
                if not all(
                    str(training.get(name, ""))
                    for name in (
                        "universal_state_sha256",
                        "copied_universal_state_sha256",
                        "post_frozen_expert_universal_state_sha256",
                        "final_universal_state_sha256",
                    )
                ):
                    raise ValueError(f"Training state hashes are incomplete: {completed_path}")
                if training.get("common_state_unchanged_during_frozen_experts") is not True:
                    raise ValueError(f"Frozen expert stage changed universal state: {completed_path}")
                training_receipts.append(training)
                if nested_scale_required:
                    nested_hash = str(
                        fold_completion.get("nested_expert_scale_receipt_sha256", "")
                    )
                    if (
                        fold_completion.get("nested_expert_scale_protocol")
                        != NESTED_SCALE_PROTOCOL
                        or not nested_hash
                    ):
                        raise ValueError(
                            f"Formal fold lacks a nested scale receipt: {completed_path}"
                        )
                    nested_receipts.append(
                        validate_nested_scale_receipt(
                            completed_path.parent / "nested_expert_scale",
                            expected_sha256=nested_hash,
                            expected_scenario=str(fold_completion.get("scenario", "")),
                            expected_fold=int(fold_completion.get("fold", -1)),
                            expected_train_ids_sha256=str(
                                fold_completion.get("train_ids_sha256", "")
                            ),
                            expected_validation_ids_sha256=str(
                                fold_completion.get("validation_ids_sha256", "")
                            ),
                            expected_source_contract_sha256=str(
                                contract.get("fingerprint_sha256", "")
                            ),
                        )
                    )
            for scenario in required_scenarios:
                if not (run / "oof_predictions" / f"{scenario}.npz").is_file():
                    raise ValueError(f"Missing {producer_id}/S{seed} {scenario} OOF predictions")
                assignment_hashes[(seed, scenario)].add(_assignment_hash(run, scenario))
            source = str(contract.get("source_fingerprint", {}).get("sha256", ""))
            if not source:
                raise ValueError(f"Missing source fingerprint: {run}")
            source_hashes.add(source)
            input_hashes.add(
                hashlib.sha256(
                    json.dumps(
                        contract.get("input_hashes", {}), sort_keys=True
                    ).encode("utf-8")
                ).hexdigest()
            )
            run_contracts.append(
                {
                    "producer": producer_id,
                    "seed": seed,
                    "path": str(run / "run_contract.json"),
                    "sha256": _sha256(run / "run_contract.json"),
                    "fingerprint_sha256": contract.get("fingerprint_sha256", ""),
                }
            )
            checks.append(
                {
                    "producer": producer_id,
                    "seed": seed,
                    "n_folds": 4,
                    "epochs": int(model["epochs"]),
                    "completed_fits": len(completed),
                    "fold_matched_training_receipts": len(training_receipts),
                    "nested_scale_receipts": len(nested_receipts),
                    "nested_scale_required": nested_scale_required,
                    "status": "valid",
                }
            )
    if len(source_hashes) != 1:
        raise ValueError("Confirmation producers were executed from different source trees")
    if len(input_hashes) != 1:
        raise ValueError("Confirmation producers do not share identical source data hashes")
    mismatched = [key for key, values in assignment_hashes.items() if len(values) != 1]
    if mismatched:
        raise ValueError(f"Candidate/control fold assignments differ: {mismatched}")
    return {
        "matrix": str(matrix_path),
        "matrix_sha256": _sha256(matrix_path),
        "checks": checks,
        "source_fingerprint_sha256": next(iter(source_hashes)),
        "input_hash_bundle_sha256": next(iter(input_hashes)),
        "assignment_hashes": {
            f"S{seed}/{scenario}": next(iter(values))
            for (seed, scenario), values in sorted(assignment_hashes.items())
        },
        "run_contracts": run_contracts,
        "metadata": metadata,
    }


def _representation_entities(path_value: object, entity_type: str) -> tuple[set[str], set[str]]:
    """Return represented and visibly non-zero entity labels from one artifact."""
    if path_value in (None, "", False):
        return set(), set()
    path = Path(str(path_value)).resolve()
    if not path.is_file():
        raise ValueError(f"Semantic representation artifact is unavailable: {path}")
    table = pd.read_csv(path, sep="\t", keep_default_na=False)
    key = "raw_name" if entity_type == "chemical" else "strain_code"
    if key not in table:
        raise ValueError(f"Semantic artifact {path} lacks {key}")
    represented = set(table[key].map(normalize_entity_key))
    if entity_type == "chemical" and "isomeric_smiles" in table:
        usable = table["isomeric_smiles"].astype(str).str.strip().ne("")
        if "status" in table:
            usable &= table["status"].astype(str).eq("resolved")
    else:
        numeric_columns = [column for column in table.columns if column != key]
        if numeric_columns:
            numeric = table.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
            usable = numeric.abs().sum(axis=1).gt(0)
        else:
            usable = pd.Series(False, index=table.index)
    nonzero = set(table.loc[usable, key].map(normalize_entity_key))
    return represented, nonzero


def _registry_statuses(path_value: object, entity_type: str) -> dict[str, str]:
    if path_value in (None, "", False):
        raise ValueError(f"Configured {entity_type} semantics require an identity registry")
    path = Path(str(path_value)).resolve()
    table = pd.read_csv(path, sep="\t", keep_default_na=False)
    key = "raw_name" if entity_type == "chemical" else "strain_code"
    if key not in table or "mapping_status" not in table:
        raise ValueError(f"Identity registry lacks {key}/mapping_status: {path}")
    return dict(
        zip(
            table[key].map(normalize_entity_key),
            table["mapping_status"].astype(str),
        )
    )


def semantic_coverage_receipt(
    root: str | Path,
    candidate_id: str,
    required_scenarios: tuple[str, ...],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Prove that configured semantic blocks are non-empty in OOD folds."""
    run_root = Path(root).resolve()
    run = run_root / "producers" / candidate_id / "S42"
    contract = _load_json(run / "run_contract.json")
    effective = contract.get("effective_config", {})
    entity = effective.get("entity", {})
    manifest = _load_json(run / "oof_manifest.json")
    config_path = Path(str(manifest.get("response_config", "")))
    config = load_response_config(config_path)
    data = prepare_data(config.baseline)
    slices, _ = make_fold_slices(
        data.metadata,
        data.train_ids,
        4,
        int(contract["seed"]),
        required_scenarios,
        config,
    )
    policy = str(entity.get("semantic_identity_policy", ""))
    allow_proxy = bool(entity.get("allow_proxy_semantics", False))
    axes: dict[str, dict[str, Any]] = {}
    configuration = {
        "chemical": {
            "configured": bool(entity.get("chemical_map") or entity.get("chemical_features")),
            "registry": entity.get("chemical_registry"),
            "representations": [entity.get("chemical_map"), entity.get("chemical_features")],
            "column": CHEMICAL,
            "ood_scenarios": {"R00", "R10"},
        },
        "strain": {
            "configured": bool(entity.get("strain_features")),
            "registry": entity.get("strain_registry"),
            "representations": [entity.get("strain_features")],
            "column": STRAIN,
            "ood_scenarios": {"R00", "R01"},
        },
    }
    for axis, definition in configuration.items():
        if not definition["configured"]:
            continue
        statuses = _registry_statuses(definition["registry"], axis)
        represented: set[str] = set()
        nonzero: set[str] = set()
        for representation in definition["representations"]:
            if representation in (None, "", False):
                continue
            current_represented, current_nonzero = _representation_entities(
                representation, axis
            )
            represented.update(current_represented)
            nonzero.update(current_nonzero)
        admitted_statuses = (
            {"verified"}
            if policy == "verified_only"
            else {"verified", "high_confidence_candidate"}
        )
        if allow_proxy:
            admitted_statuses.add("proxy")
        admitted = {
            entity_key
            for entity_key in represented
            if statuses.get(entity_key) in admitted_statuses
        }
        nonzero_admitted = admitted & nonzero
        relevant_slices = [
            fold
            for fold in slices
            if fold.scenario in set(required_scenarios) & definition["ood_scenarios"]
            and len(fold.validation_ids)
        ]
        training_counts: list[int] = []
        validation_entities: set[str] = set()
        for fold in relevant_slices:
            training = {
                normalize_entity_key(value)
                for value in data.metadata.loc[fold.train_ids, definition["column"]]
            }
            validation = {
                normalize_entity_key(value)
                for value in data.metadata.loc[
                    fold.validation_ids, definition["column"]
                ]
            }
            training_counts.append(len(training & nonzero_admitted))
            validation_entities.update(validation & nonzero_admitted)
        required = bool(relevant_slices)
        passed = bool(
            (not required)
            or (
                len(nonzero_admitted) >= 2
                and training_counts
                and min(training_counts) >= 2
                and len(validation_entities) >= 2
            )
        )
        axes[axis] = {
            "configured": True,
            "policy": policy,
            "required_for_regimes": sorted(
                set(required_scenarios) & definition["ood_scenarios"]
            ),
            "represented_unique_entities": len(represented),
            "identity_admitted_unique_entities": len(admitted),
            "nonzero_admitted_unique_entities": len(nonzero_admitted),
            "min_fold_train_nonzero_admitted_entities": (
                min(training_counts) if training_counts else 0
            ),
            "validation_nonzero_admitted_entities": len(validation_entities),
            "nonzero_admitted_entities_sha256": hashlib.sha256(
                "\n".join(sorted(nonzero_admitted)).encode("utf-8")
            ).hexdigest(),
            "passed": passed,
        }
    semantic_candidate = str(metadata[candidate_id].get("model_id", "")).startswith("M8")
    passed = semantic_coverage_passes(axes, semantic_candidate=semantic_candidate)
    receipt = {
        "schema": "goai.m7_m8.semantic_coverage_receipt.v1",
        "protocol_label": PROTOCOL_LABEL,
        "candidate": candidate_id,
        "semantic_candidate": semantic_candidate,
        "semantic_identity_policy": policy,
        "axes": axes,
        "status": "valid" if passed else "blocked",
        "passed": passed,
    }
    if not passed:
        failed = [axis for axis, item in axes.items() if not item["passed"]]
        raise ValueError(
            f"{candidate_id} has insufficient non-zero fold semantic coverage: {failed}"
        )
    return receipt


def semantic_coverage_passes(
    axes: dict[str, dict[str, Any]], *, semantic_candidate: bool
) -> bool:
    """Fail closed for M8 when no semantic axis is actually non-zero."""
    return bool(
        (not semantic_candidate)
        or (axes and all(bool(item.get("passed")) for item in axes.values()))
    )


def validate_joint_primary_contract(
    candidate_metadata: dict[str, Any],
    primary_metadata: dict[str, Any],
) -> None:
    """Require a fair primary with the same number of universal updates."""
    candidate_variant = str(candidate_metadata.get("confirmation_training_variant", ""))
    if candidate_variant != "joint_finetune":
        return
    primary_variant = str(primary_metadata.get("confirmation_training_variant", ""))
    if primary_variant not in {"joint_finetune", "same_universal_update_control"}:
        raise ValueError(
            "Joint candidate requires joint_finetune or same_universal_update_control primary"
        )
    candidate_updates = int(
        candidate_metadata.get("universal_update_budget", {}).get(
            "total_universal_update_epochs", -1
        )
    )
    primary_updates = int(
        primary_metadata.get("universal_update_budget", {}).get(
            "total_universal_update_epochs", -1
        )
    )
    if candidate_updates < 0 or candidate_updates != primary_updates:
        raise ValueError(
            "Joint candidate and primary control have different universal-update budgets: "
            f"{candidate_updates} vs {primary_updates}"
        )


class _StreamingMetrics:
    """Sufficient statistics for one held-out entity, without row expansion."""

    def __init__(self) -> None:
        self._pcc = {
            name: np.zeros(6, dtype=np.float64)
            for name in (
                "fc_pcc",
                "context_residual_pcc",
                "drug_residual_pcc",
                "high_effect_pcc",
            )
        }
        self.high_true = 0
        self.high_pred = 0
        self.high_true_positive = 0

    def _update_pcc(self, name: str, x: np.ndarray, y: np.ndarray) -> None:
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            return
        left = x[mask].astype(np.float64, copy=False)
        right = y[mask].astype(np.float64, copy=False)
        self._pcc[name] += np.asarray(
            [
                len(left),
                left.sum(),
                right.sum(),
                np.dot(left, left),
                np.dot(right, right),
                np.dot(left, right),
            ],
            dtype=np.float64,
        )

    def update(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        context: np.ndarray,
        drug: np.ndarray,
    ) -> None:
        observed = np.isfinite(predicted) & np.isfinite(actual)
        p = np.where(observed, predicted, np.nan)
        y = np.where(observed, actual, np.nan)
        self._update_pcc("fc_pcc", p, y)
        self._update_pcc("context_residual_pcc", p - context, y - context)
        self._update_pcc("drug_residual_pcc", p - drug, y - drug)
        high_true = observed & (np.abs(actual) > 1.0)
        high_pred = observed & (np.abs(predicted) > 1.0)
        self._update_pcc(
            "high_effect_pcc",
            np.where(high_true, predicted, np.nan),
            np.where(high_true, actual, np.nan),
        )
        true_positive = high_true & high_pred & (np.sign(predicted) == np.sign(actual))
        self.high_true += int(high_true.sum())
        self.high_pred += int(high_pred.sum())
        self.high_true_positive += int(true_positive.sum())

    @staticmethod
    def _pcc_value(stats: np.ndarray) -> float:
        n, sum_x, sum_y, sum_xx, sum_yy, sum_xy = stats
        if n < 2:
            return float("nan")
        covariance = sum_xy - (sum_x * sum_y / n)
        variance_x = sum_xx - (sum_x * sum_x / n)
        variance_y = sum_yy - (sum_y * sum_y / n)
        denominator = np.sqrt(max(variance_x, 0.0) * max(variance_y, 0.0))
        return float(covariance / denominator) if denominator > 0 else float("nan")

    def values(self) -> dict[str, float]:
        result = {name: self._pcc_value(stats) for name, stats in self._pcc.items()}
        precision = (
            self.high_true_positive / self.high_pred if self.high_pred else float("nan")
        )
        recall = (
            self.high_true_positive / self.high_true if self.high_true else float("nan")
        )
        result["high_effect_f1"] = (
            float(2.0 * precision * recall / (precision + recall))
            if np.isfinite(precision + recall) and precision + recall > 0
            else float("nan")
        )
        return result


def _cluster_key(metadata: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    missing = [column for column in columns if column not in metadata]
    if missing:
        raise ValueError(f"Fold assignments/metadata lack cluster fields: {missing}")
    return metadata.loc[:, list(columns)].astype(str).agg("\x1f".join, axis=1)


def _entity_metric_units(
    root: Path,
    candidate_id: str,
    control_id: str,
    scenarios: tuple[str, ...],
) -> pd.DataFrame:
    """Re-score each held-out entity from sample-level OOF predictions."""
    first_run = root / "producers" / control_id / "S42"
    manifest = _load_json(first_run / "oof_manifest.json")
    config_path = Path(str(manifest.get("response_config", "")))
    if not config_path.is_file():
        raise ValueError(f"Confirmation response config is unavailable: {config_path}")
    config = load_response_config(config_path)
    data = prepare_data(config.baseline)
    outer_controls = data.train_ids[
        control_mask(data.metadata.loc[data.train_ids]).to_numpy()
    ]
    rows: list[dict[str, Any]] = []
    for seed in CONFIRM_SEEDS:
        candidate_run = root / "producers" / candidate_id / f"S{seed}"
        control_run = root / "producers" / control_id / f"S{seed}"
        fold_seed = int(_load_json(candidate_run / "oof_manifest.json")["seed"])
        slices, _ = make_fold_slices(
            data.metadata, data.train_ids, 4, fold_seed, scenarios, config
        )
        by_scenario = defaultdict(list)
        for fold in slices:
            by_scenario[fold.scenario].append(fold)
        for scenario in scenarios:
            candidate_prediction = _load_prediction(candidate_run, scenario)
            control_prediction = _load_prediction(control_run, scenario)
            if not candidate_prediction.index.equals(control_prediction.index):
                raise ValueError(f"Prediction IDs differ for S{seed}/{scenario}")
            if not candidate_prediction.columns.equals(control_prediction.columns):
                raise ValueError(f"Prediction proteins differ for S{seed}/{scenario}")
            accumulators: dict[
                tuple[str, str], tuple[_StreamingMetrics, _StreamingMetrics]
            ] = {}
            for fold in by_scenario[scenario]:
                expected = pd.Index(fold.validation_ids.astype(str), name=SAMPLE_ID)
                candidate_current = candidate_prediction.reindex(expected)
                control_current = control_prediction.reindex(expected)
                if candidate_current.isna().all(axis=None) or control_current.isna().all(axis=None):
                    raise ValueError(f"Missing prediction rows for S{seed}/{scenario}/F{fold.fold}")
                fold_data = replace(data, train_ids=fold.train_ids)
                control_pool = fold.train_ids.union(outer_controls)
                candidate_delta, truth_delta, usable_metadata = _control_deltas(
                    data.metadata,
                    data.y_log2,
                    candidate_current,
                    expected,
                    control_pool,
                )
                control_delta, control_truth, control_metadata = _control_deltas(
                    data.metadata,
                    data.y_log2,
                    control_current,
                    expected,
                    control_pool,
                )
                if not candidate_delta.index.equals(control_delta.index):
                    raise ValueError("Candidate/control matched-control rows differ")
                if not truth_delta.equals(control_truth) or not usable_metadata.equals(control_metadata):
                    raise ValueError("Candidate/control truth contracts differ")
                context_reference, drug_reference = _frozen_delta_references(fold_data)
                context = _reference_for_rows(
                    context_reference, usable_metadata, MATCH_CONTROL_FIELDS
                ).to_numpy(dtype=np.float64)
                drug = _reference_for_rows(
                    drug_reference, usable_metadata, (CHEMICAL,)
                ).to_numpy(dtype=np.float64)
                candidate_values = candidate_delta.to_numpy(dtype=np.float64)
                control_values = control_delta.to_numpy(dtype=np.float64)
                truth_values = truth_delta.to_numpy(dtype=np.float64)
                for axis_columns in SCENARIO_CLUSTER_AXES[scenario]:
                    axis = "+".join(axis_columns)
                    keys = _cluster_key(usable_metadata, axis_columns)
                    for entity, positions in keys.groupby(keys, sort=False).groups.items():
                        # pandas group positions are sample IDs; convert them to
                        # integer positions against the aligned delta matrix.
                        integer = candidate_delta.index.get_indexer(pd.Index(positions))
                        key = (axis, str(entity))
                        if key not in accumulators:
                            accumulators[key] = (_StreamingMetrics(), _StreamingMetrics())
                        candidate_accumulator, control_accumulator = accumulators[key]
                        candidate_accumulator.update(
                            candidate_values[integer], truth_values[integer], context[integer], drug[integer]
                        )
                        control_accumulator.update(
                            control_values[integer], truth_values[integer], context[integer], drug[integer]
                        )
            for (axis, entity), (candidate_accumulator, control_accumulator) in sorted(accumulators.items()):
                candidate_values = candidate_accumulator.values()
                control_values = control_accumulator.values()
                for metric in GATE_METRICS:
                    candidate_value = float(candidate_values[metric])
                    control_value = float(control_values[metric])
                    rows.append(
                        {
                            "protocol_label": PROTOCOL_LABEL,
                            "candidate": candidate_id,
                            "control": control_id,
                            "model_seed": seed,
                            "scenario": scenario,
                            "cluster_axis": axis,
                            "heldout_entity": entity,
                            "metric": metric,
                            "control_value": control_value,
                            "candidate_value": candidate_value,
                            "delta": candidate_value - control_value,
                        }
                    )
    return pd.DataFrame(rows)


def _cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    seed: int,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Bootstrap unique entities after averaging repeated model seeds."""
    pivot = frame.pivot_table(
        index="heldout_entity",
        columns="model_seed",
        values="delta",
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=list(CONFIRM_SEEDS))
    complete = pivot.dropna(how="any")
    if complete.empty:
        return {
            "n_heldout_entities": 0,
            "mean_delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "seed_means": {},
            "all_seeds_positive": False,
        }
    entity_means = complete.mean(axis=1).to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = entity_means[
        rng.integers(0, len(entity_means), size=(draws, len(entity_means)))
    ].mean(axis=1)
    seed_means = {
        str(model_seed): float(complete[model_seed].mean())
        for model_seed in CONFIRM_SEEDS
    }
    return {
        "n_heldout_entities": int(len(complete)),
        "mean_delta": float(entity_means.mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "seed_means": seed_means,
        "all_seeds_positive": bool(all(value > 0.0 for value in seed_means.values())),
    }


def decide_promotion(
    units: pd.DataFrame,
    required_scenarios: tuple[str, ...],
    *,
    primary_comparison: bool = True,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the locked thresholds to per-held-out-entity metric deltas."""
    required_columns = {
        "model_seed",
        "scenario",
        "cluster_axis",
        "heldout_entity",
        "metric",
        "delta",
    }
    missing = sorted(required_columns.difference(units.columns))
    if missing:
        raise ValueError(f"Entity metric table lacks columns: {missing}")
    if set(required_scenarios).difference(SCENARIO_CLUSTER_AXES):
        raise ValueError(f"Unknown promotion scenarios: {required_scenarios}")
    fc_threshold = (
        PRIMARY_FC_MIN_DELTA if primary_comparison else NEGATIVE_CONTROL_FC_MIN_DELTA
    )
    rows: list[dict[str, Any]] = []
    for scenario in required_scenarios:
        scenario_rows = units.loc[units["scenario"].eq(scenario)]
        expected_axes = {"+".join(columns) for columns in SCENARIO_CLUSTER_AXES[scenario]}
        actual_axes = set(scenario_rows["cluster_axis"])
        if expected_axes.difference(actual_axes):
            raise ValueError(
                f"{scenario} lacks held-out entity axes: {sorted(expected_axes - actual_axes)}"
            )
        metrics = (
            "fc_pcc",
            *SCENARIO_RESIDUALS[scenario],
            "high_effect_pcc",
            "high_effect_f1",
        )
        for axis in sorted(expected_axes):
            for metric_index, metric in enumerate(metrics):
                current = scenario_rows.loc[
                    scenario_rows["cluster_axis"].eq(axis)
                    & scenario_rows["metric"].eq(metric)
                ]
                stats = _cluster_bootstrap(
                    current,
                    seed=20260813 + metric_index + 101 * len(rows),
                    draws=bootstrap_draws,
                )
                seed_values = list(stats["seed_means"].values())
                if metric == "fc_pcc":
                    passed = bool(
                        stats["n_heldout_entities"] >= 2
                        and stats["mean_delta"] >= fc_threshold
                        and stats["ci_low"] > 0.0
                        and stats["all_seeds_positive"]
                    )
                    rule = (
                        f"mean_delta>={fc_threshold:.3f}; entity-bootstrap ci_low>0; "
                        "all seeds positive"
                    )
                elif metric in {"context_residual_pcc", "drug_residual_pcc"}:
                    passed = bool(
                        stats["n_heldout_entities"] >= 2
                        and stats["mean_delta"] > 0.0
                        and stats["ci_low"] > 0.0
                        and stats["all_seeds_positive"]
                    )
                    rule = "mean_delta>0; entity-bootstrap ci_low>0; all seeds positive"
                else:
                    passed = bool(
                        stats["n_heldout_entities"] >= 2
                        and stats["mean_delta"] >= -HIGH_EFFECT_MAX_DROP
                        and len(seed_values) == len(CONFIRM_SEEDS)
                        and min(seed_values) >= -HIGH_EFFECT_MAX_DROP
                    )
                    rule = "overall and every seed delta>=-0.005"
                rows.append(
                    {
                        "protocol_label": PROTOCOL_LABEL,
                        "scenario": scenario,
                        "cluster_axis": axis,
                        "metric": metric,
                        **stats,
                        "rule": rule,
                        "passed": passed,
                    }
                )
    checks = pd.DataFrame(rows)
    has_identifiable_residual = any(SCENARIO_RESIDUALS[item] for item in required_scenarios)
    reasons: list[str] = []
    if not has_identifiable_residual:
        reasons.append(
            "No required regime has an identifiable residual metric; R00 alone cannot promote"
        )
    if checks.empty or not bool(checks["passed"].all()):
        for row in checks.loc[~checks["passed"]].to_dict(orient="records"):
            reasons.append(
                f"{row['scenario']}/{row['cluster_axis']}/{row['metric']} failed: "
                f"mean={row['mean_delta']:+.6f}, ci_low={row['ci_low']:+.6f}"
            )
    promoted = bool(has_identifiable_residual and not checks.empty and checks["passed"].all())
    return checks, {
        "status": "promoted" if promoted else "blocked",
        "promoted": promoted,
        "reasons": reasons,
        "required_scenarios": list(required_scenarios),
        "thresholds": {
            "fc_mean_delta_min": fc_threshold,
            "residual_mean_delta_min_exclusive": 0.0,
            "entity_bootstrap_ci_low_min_exclusive": 0.0,
            "high_effect_pcc_max_drop": HIGH_EFFECT_MAX_DROP,
            "high_effect_f1_max_drop": HIGH_EFFECT_MAX_DROP,
            "required_model_seeds": list(CONFIRM_SEEDS),
        },
    }


def run_promotion_gate(
    root: str | Path,
    candidate_id: str,
    control_id: str,
    required_scenarios: tuple[str, ...],
    output_dir: str | Path,
    *,
    negative_control_ids: tuple[str, ...] = (),
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Validate confirmation, score entities, and atomically issue a receipt."""
    run_root = Path(root).resolve()
    output = Path(output_dir).resolve()
    if not required_scenarios:
        raise ValueError("At least one required promotion scenario is necessary")
    producer_ids = (candidate_id, control_id, *negative_control_ids)
    contract = _validate_confirmation_runs(run_root, producer_ids, required_scenarios)
    metadata = contract["metadata"]
    candidate_metadata = metadata[candidate_id]
    if candidate_metadata.get("promotion_eligible") is False:
        raise ValueError(f"{candidate_id} is explicitly promotion_eligible=false")
    if candidate_metadata.get("confirmation_training_variant") not in {
        "frozen_residual_only",
        "promotion_confirmation",
        "joint_finetune",
    }:
        raise ValueError(
            f"{candidate_id} is not a promotion-eligible frozen confirmation variant"
        )
    candidate_epochs = int(
        candidate_metadata.get("overrides", {}).get("model", {}).get("epochs", -1)
    )
    if candidate_epochs < 80:
        raise ValueError(f"{candidate_id} has only {candidate_epochs} declared epochs; 80 required")
    candidate_variant = str(candidate_metadata.get("confirmation_training_variant", ""))
    primary_metadata = metadata[control_id]
    validate_joint_primary_contract(candidate_metadata, primary_metadata)
    candidate_update_epochs = int(
        candidate_metadata.get("universal_update_budget", {}).get(
            "total_universal_update_epochs", -1
        )
    )
    primary_update_epochs = int(
        primary_metadata.get("universal_update_budget", {}).get(
            "total_universal_update_epochs", -1
        )
    )
    if candidate_update_epochs < 0 or candidate_update_epochs != primary_update_epochs:
        raise ValueError(
            "Candidate and primary control have different universal-update budgets: "
            f"{candidate_update_epochs} vs {primary_update_epochs}"
        )
    declared_negative_controls = tuple(
        str(item) for item in candidate_metadata.get("required_negative_controls", [])
    )
    if set(negative_control_ids) != set(declared_negative_controls):
        raise ValueError(
            f"{candidate_id} requires negative controls "
            f"{list(declared_negative_controls)}, received {list(negative_control_ids)}"
        )
    for negative_control_id in negative_control_ids:
        if metadata[negative_control_id].get("promotion_eligible") is not False:
            raise ValueError(
                f"Negative control {negative_control_id} must declare promotion_eligible=false"
            )
        declared_target = metadata[negative_control_id].get("negative_control_for")
        if declared_target not in (None, "", candidate_id) and negative_control_id != control_id:
            raise ValueError(
                f"Negative control {negative_control_id} is declared for {declared_target}, "
                f"not {candidate_id}"
            )
        negative_update_epochs = int(
            metadata[negative_control_id]
            .get("universal_update_budget", {})
            .get("total_universal_update_epochs", -1)
        )
        if negative_update_epochs != candidate_update_epochs:
            raise ValueError(
                f"Candidate and negative control {negative_control_id} have different "
                f"universal-update budgets: {candidate_update_epochs} vs "
                f"{negative_update_epochs}"
            )
    semantic_coverage = semantic_coverage_receipt(
        run_root,
        candidate_id,
        required_scenarios,
        metadata,
    )
    all_units: list[pd.DataFrame] = []
    all_checks: list[pd.DataFrame] = []
    decisions: list[dict[str, Any]] = []
    comparisons = ((control_id, True), *((item, False) for item in negative_control_ids))
    for comparison_id, primary in comparisons:
        units = _entity_metric_units(
            run_root, candidate_id, comparison_id, required_scenarios
        )
        checks, decision = decide_promotion(
            units,
            required_scenarios,
            primary_comparison=primary,
            bootstrap_draws=bootstrap_draws,
        )
        units.insert(2, "comparison_type", "primary" if primary else "negative_control")
        checks.insert(1, "candidate", candidate_id)
        checks.insert(2, "control", comparison_id)
        checks.insert(3, "comparison_type", "primary" if primary else "negative_control")
        decision.update(
            {
                "candidate": candidate_id,
                "control": comparison_id,
                "comparison_type": "primary" if primary else "negative_control",
            }
        )
        all_units.append(units)
        all_checks.append(checks)
        decisions.append(decision)
    units_frame = pd.concat(all_units, ignore_index=True)
    checks_frame = pd.concat(all_checks, ignore_index=True)
    promoted = bool(decisions and all(item["promoted"] for item in decisions))
    receipt: dict[str, Any] = {
        "schema": "goai.m7_m8.promotion_receipt.v1",
        "protocol_label": PROTOCOL_LABEL,
        "official_score_status": "NOT_OFFICIAL",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "candidate": candidate_id,
        "primary_control": control_id,
        "negative_controls": list(negative_control_ids),
        "required_scenarios": list(required_scenarios),
        "confirmation_contract": {key: value for key, value in contract.items() if key != "metadata"},
        "semantic_coverage": semantic_coverage,
        "comparison_decisions": decisions,
        "status": "promoted" if promoted else "blocked",
        "promoted": promoted,
        "discovery_results_can_promote": False,
        "bootstrap_unit": "heldout_entity_after_model_seed_averaging",
    }
    output.mkdir(parents=True, exist_ok=True)
    units_frame.to_csv(output / "heldout_entity_metric_units.csv", index=False)
    checks_frame.to_csv(output / "promotion_gate_checks.csv", index=False)
    receipt_path = output / "promotion_receipt.json"
    _atomic_json(receipt_path, receipt)
    receipt_hash = _sha256(receipt_path)
    (output / "promotion_receipt.json.sha256").write_text(
        receipt_hash + "\n", encoding="utf-8"
    )
    return receipt


def _parse_scenarios(values: list[str] | None, metadata: dict[str, Any]) -> tuple[str, ...]:
    selected = values or metadata.get("promotion_regimes")
    if not isinstance(selected, (list, tuple)) or not selected:
        raise ValueError(
            "Required regimes are missing; pass --required-regime or declare promotion_regimes"
        )
    scenarios = tuple(str(item) for item in selected)
    unknown = sorted(set(scenarios).difference(SCENARIO_CLUSTER_AXES))
    if unknown:
        raise ValueError(f"Unknown promotion regimes: {unknown}")
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--control",
        default=None,
        help="Override primary control; otherwise candidate primary_control or M7.0 is used",
    )
    parser.add_argument("--negative-control", action="append", default=[])
    parser.add_argument("--required-regime", action="append", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    _, metadata = _matrix_metadata(root)
    if args.candidate not in metadata:
        raise ValueError(f"Candidate is absent from confirmation matrix: {args.candidate}")
    candidate_metadata = metadata[args.candidate]
    scenarios = _parse_scenarios(args.required_regime, candidate_metadata)
    declared_primary_control = str(
        candidate_metadata.get("primary_control", "CONF-M7.0-GENERAL")
    )
    if args.control is not None and str(args.control) != declared_primary_control:
        raise ValueError(
            f"Explicit --control {args.control!r} conflicts with declared primary_control "
            f"{declared_primary_control!r}"
        )
    primary_control = declared_primary_control
    declared_negative_controls = candidate_metadata.get("required_negative_controls", [])
    if declared_negative_controls is None:
        declared_negative_controls = []
    if not isinstance(declared_negative_controls, list):
        raise ValueError("required_negative_controls must be a list")
    declared_negative_controls = [str(item) for item in declared_negative_controls]
    supplied_negative_controls = [str(item) for item in args.negative_control]
    if supplied_negative_controls:
        if set(supplied_negative_controls) != set(declared_negative_controls):
            raise ValueError(
                "Explicit --negative-control values must exactly match the candidate's "
                "required_negative_controls"
            )
        negative_controls = tuple(supplied_negative_controls)
    else:
        negative_controls = tuple(declared_negative_controls)
    receipt = run_promotion_gate(
        root,
        args.candidate,
        primary_control,
        scenarios,
        args.output_dir,
        negative_control_ids=negative_controls,
        bootstrap_draws=args.bootstrap_draws,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if not receipt["promoted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
