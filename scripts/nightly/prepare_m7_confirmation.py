#!/usr/bin/env python3
"""Select and materialize the automatic closed-data M7 confirmation wave.

Fair/quick/research results are discovery evidence only.  They may nominate a
candidate for the expensive locked confirmation, but can never promote it.
This command also materializes the four pre-confirmation evidence receipts,
filters the matrix to nominated M7 candidates plus their exact controls, and
issues an explicit M8-blocked receipt.  No training is launched here.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

try:
    from scripts.nightly.build_m7_m8_confirm import materialize_confirm_matrix
    from scripts.nightly.confirmation_evidence import (
        PROTOCOL_LABEL,
        sha256_file,
        validate_preconfirmation_evidence,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from build_m7_m8_confirm import materialize_confirm_matrix  # type: ignore
    from confirmation_evidence import (  # type: ignore
        PROTOCOL_LABEL,
        sha256_file,
        validate_preconfirmation_evidence,
    )


PROJECT = Path(__file__).resolve().parents[2]
FAIR_CONTROL = "FAIR-M7.0-U9"
NOMINATION_FC_ABSOLUTE = 0.005
NOMINATION_FC_WITH_POSITIVE_FOLD_CI = 0.002
HIGH_EFFECT_MAX_DROP = 0.005

FAIR_NOMINATIONS: dict[str, dict[str, Any]] = {
    "CONF-M7.1-STRAIN": {
        "source": "FAIR-M7.1-U9-S2-FROZEN",
        "regimes": {"R10": "context_residual_pcc"},
    },
    "CONF-M7.1-STRAIN-JOINT": {
        "source": "FAIR-M7.1-U9-S2-J3",
        "regimes": {"R10": "context_residual_pcc"},
    },
    "CONF-M7.2-CHEMICAL": {
        "source": "FAIR-M7.2-U9-C2-FROZEN",
        "regimes": {"R01": "drug_residual_pcc"},
    },
    "CONF-M7.2-CHEMICAL-JOINT": {
        "source": "FAIR-M7.2-U9-C2-J3",
        "regimes": {"R01": "drug_residual_pcc"},
    },
    "CONF-M7.3-ENTITIES": {
        "source": "FAIR-M7.3-U9-S2-C2-FROZEN",
        "regimes": {
            "R10": "context_residual_pcc",
            "R01": "drug_residual_pcc",
        },
    },
    "CONF-M7.3-ENTITIES-JOINT": {
        "source": "FAIR-M7.3-U9-S2-C2-J3",
        "regimes": {
            "R10": "context_residual_pcc",
            "R01": "drug_residual_pcc",
        },
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_hashed_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_json(path, payload)
    Path(str(path) + ".sha256").write_text(
        sha256_file(path) + "\n", encoding="utf-8"
    )


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def validate_scale_candidate_grid(
    candidates_path: str | Path,
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Bind the scale receipt to all 15 completed discovery grid rows."""
    path = Path(candidates_path).resolve()
    if not path.is_file():
        raise ValueError(f"Scale-candidate evidence is unavailable: {path}")
    table = pd.read_csv(path)
    required = {"axis", "scale", "experiment_id", "complete", "metrics_finite"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Scale-candidate table lacks columns: {missing}")
    expected_scales = {0.0, 0.25, 0.5, 0.75, 1.0}
    selected = selection.get("selected", {})
    evidence: dict[str, Any] = {}
    for axis in ("strain", "chemical", "pair"):
        rows = table.loc[table["axis"].astype(str).eq(axis)].copy()
        observed = set(pd.to_numeric(rows["scale"], errors="coerce"))
        if len(rows) != 5 or observed != expected_scales:
            raise ValueError(f"Scale grid for {axis} is not the locked five-point grid")
        if not rows["complete"].map(_as_bool).all() or not rows[
            "metrics_finite"
        ].map(_as_bool).all():
            raise ValueError(f"Scale grid for {axis} is incomplete or non-finite")
        decision = selected.get(axis)
        if not isinstance(decision, dict):
            raise ValueError(f"Scale selection lacks {axis} decision")
        chosen = rows.loc[
            rows["experiment_id"].astype(str).eq(str(decision.get("experiment_id", "")))
            & pd.to_numeric(rows["scale"], errors="coerce").eq(
                float(decision.get("scale", float("nan")))
            )
        ]
        if len(chosen) != 1:
            raise ValueError(f"Selected {axis} scale does not match its grid evidence")
        evidence[axis] = {
            "selected_experiment_id": str(decision["experiment_id"]),
            "selected_scale": float(decision["scale"]),
            "complete_grid_rows": int(len(rows)),
        }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "axes": evidence,
    }


def _one_metric(
    paired: pd.DataFrame,
    *,
    candidate: str,
    scenario: str,
    metric: str,
) -> dict[str, Any]:
    rows = paired.loc[
        paired["candidate"].astype(str).eq(candidate)
        & paired["control"].astype(str).eq(FAIR_CONTROL)
        & pd.to_numeric(paired["model_seed"], errors="coerce").eq(42)
        & paired["scenario"].astype(str).eq(scenario)
        & paired["metric"].astype(str).eq(metric)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Fair discovery evidence is not one-to-one for "
            f"{candidate}/{scenario}/{metric}"
        )
    row = rows.iloc[0]
    values = {
        "n_paired_folds": int(row["n_paired_folds"]),
        "mean_delta": float(row["mean_delta"]),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row["ci_high"]),
    }
    if values["n_paired_folds"] != 2:
        raise ValueError(
            f"Fair discovery evidence does not contain two folds for "
            f"{candidate}/{scenario}/{metric}"
        )
    values["finite"] = bool(
        np.isfinite(
            [values["mean_delta"], values["ci_low"], values["ci_high"]]
        ).all()
    )
    return values


def _fc_nomination_pass(metric: dict[str, Any]) -> bool:
    if metric.get("finite") is not True:
        return False
    mean = float(metric["mean_delta"])
    return bool(
        mean >= NOMINATION_FC_ABSOLUTE
        or (
            mean >= NOMINATION_FC_WITH_POSITIVE_FOLD_CI
            and float(metric["ci_low"]) > 0.0
        )
    )


def nominate_m7_candidates(
    paired_summary: pd.DataFrame,
    scale_selection: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Apply a predeclared discovery-only rule; never issue promotion."""
    decisions: list[dict[str, Any]] = []
    nominated: list[str] = []
    for formal_id, rule in FAIR_NOMINATIONS.items():
        regime_checks: dict[str, Any] = {}
        passed = True
        for scenario, residual in rule["regimes"].items():
            metrics = {
                metric: _one_metric(
                    paired_summary,
                    candidate=str(rule["source"]),
                    scenario=scenario,
                    metric=metric,
                )
                for metric in (
                    "fc_pcc",
                    residual,
                    "high_effect_pcc",
                    "high_effect_f1",
                )
            }
            checks = {
                "fc_nomination": _fc_nomination_pass(metrics["fc_pcc"]),
                "residual_nonnegative": (
                    metrics[residual].get("finite") is True
                    and metrics[residual]["mean_delta"] >= 0.0
                ),
                "high_effect_pcc_guardrail": (
                    metrics["high_effect_pcc"].get("finite") is True
                    and metrics["high_effect_pcc"]["mean_delta"]
                    >= -HIGH_EFFECT_MAX_DROP
                ),
                "high_effect_f1_guardrail": (
                    metrics["high_effect_f1"].get("finite") is True
                    and metrics["high_effect_f1"]["mean_delta"]
                    >= -HIGH_EFFECT_MAX_DROP
                ),
            }
            scenario_pass = bool(all(checks.values()))
            regime_checks[scenario] = {
                "relevant_residual": residual,
                "metrics": metrics,
                "checks": checks,
                "passed": scenario_pass,
            }
            passed &= scenario_pass
        decision = {
            "candidate": formal_id,
            "source_discovery_candidate": rule["source"],
            "source_control": FAIR_CONTROL,
            "promotion_eligible_from_discovery": False,
            "nominate_for_confirmation": bool(passed),
            "decision": "confirm" if passed else "screen_only",
            "regimes": regime_checks,
        }
        decisions.append(decision)
        if passed:
            nominated.append(formal_id)

    pair = scale_selection.get("selected", {}).get("pair", {})
    if not isinstance(pair, dict):
        raise ValueError("Scale selection lacks the pair decision")
    pair_checks = {
        "nonzero_pair_scale": float(pair.get("scale", 0.0)) > 0.0,
        "fc_discovery_signal": (
            float(pair.get("fc_delta_vs_zero", float("-inf")))
            >= NOMINATION_FC_WITH_POSITIVE_FOLD_CI
        ),
        "residual_guardrail": (
            float(pair.get("relevant_residual_delta_vs_zero", float("-inf")))
            >= 0.0
        ),
        "high_effect_guardrail": (
            float(pair.get("high_effect_pcc_delta_vs_zero", float("-inf")))
            >= -HIGH_EFFECT_MAX_DROP
        ),
        "locked_regimes": pair.get("guardrail_regimes") == ["R11", "RT"],
    }
    pair_pass = bool(all(pair_checks.values()))
    decisions.append(
        {
            "candidate": "CONF-M7.4-PAIR",
            "source_discovery_candidate": str(pair.get("experiment_id", "")),
            "source_control": "pair_scale_zero",
            "promotion_eligible_from_discovery": False,
            "nominate_for_confirmation": pair_pass,
            "decision": "confirm" if pair_pass else "screen_only",
            "checks": pair_checks,
            "note": (
                "Only frozen M7.4 is automatically nominated: the pair grid "
                "does not contain a fold-matched joint-finetune ablation."
            ),
        }
    )
    if pair_pass:
        nominated.append("CONF-M7.4-PAIR")
    return nominated, decisions


def _dependency_closure(
    experiments: Iterable[dict[str, Any]],
    candidates: Iterable[str],
) -> set[str]:
    metadata = {str(item["id"]): item for item in experiments}
    required = set(map(str, candidates))
    pending = list(required)
    while pending:
        current = pending.pop()
        if current not in metadata:
            raise ValueError(f"Nominated confirmation is absent after materialization: {current}")
        item = metadata[current]
        dependencies = []
        primary = item.get("primary_control")
        if primary:
            dependencies.append(str(primary))
        negatives = item.get("required_negative_controls", [])
        if not isinstance(negatives, list):
            raise ValueError(f"Invalid negative-control list for {current}")
        dependencies.extend(map(str, negatives))
        for dependency in dependencies:
            if dependency not in required:
                required.add(dependency)
                pending.append(dependency)
    return required


def prepare_confirmation(
    *,
    template_path: str | Path,
    scale_selection_path: str | Path,
    scale_candidates_path: str | Path,
    identity_audit_path: str | Path,
    calibration_audit_path: str | Path,
    fair_expert_audit_path: str | Path,
    fair_paired_summary_path: str | Path,
    quick_summary_dir: str | Path,
    research_summary_dir: str | Path,
    output_matrix_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    template_file = Path(template_path).resolve()
    scale_file = Path(scale_selection_path).resolve()
    output = Path(output_dir).resolve()
    matrix_path = Path(output_matrix_path).resolve()
    template = yaml.safe_load(template_file.read_text(encoding="utf-8"))
    selection = yaml.safe_load(scale_file.read_text(encoding="utf-8"))
    if not isinstance(template, dict) or not isinstance(selection, dict):
        raise ValueError("Template and scale selection must be mappings")

    summary_evidence: dict[str, Any] = {}
    for label, value in (
        ("quick", quick_summary_dir),
        ("research", research_summary_dir),
    ):
        directory = Path(value).resolve()
        required = (directory / "local_oof_report_zh.md", directory / "regime_summary.csv")
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise ValueError(f"Missing {label} summary evidence: {missing}")
        summary_evidence[label] = {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in required
        }

    scale_grid = validate_scale_candidate_grid(scale_candidates_path, selection)
    evidence = validate_preconfirmation_evidence(
        identity_audit=identity_audit_path,
        calibration_audit=calibration_audit_path,
        fair_expert_audit=fair_expert_audit_path,
        scale_selection=selection,
        scale_selection_path=scale_file,
    )
    fair_path = Path(fair_paired_summary_path).resolve()
    if not fair_path.is_file():
        raise ValueError(f"Fair paired discovery summary is unavailable: {fair_path}")
    paired = pd.read_csv(fair_path)
    nominated, nomination_decisions = nominate_m7_candidates(paired, selection)

    base = Path(str(template["base_config"]))
    base = base if base.is_absolute() else (template_file.parent / base).resolve()
    materialized = materialize_confirm_matrix(
        template,
        selection,
        absolute_base_config=str(base),
        preconfirmation_evidence=evidence,
    )
    all_experiments = list(materialized.get("experiments", []))
    nomination_contract = materialized.get("non_binding_expert_scale_nomination", {})
    if (
        not isinstance(nomination_contract, dict)
        or nomination_contract.get("binding_to_formal_predictions") is not False
        or nomination_contract.get("formal_policy")
        != "per_outer_fold_nested_inner_oof"
    ):
        raise ValueError(
            "Formal confirmation materializer still binds a global discovery scale"
        )
    for item in all_experiments:
        if item.get("kind") != "model_confirm":
            continue
        model = item.get("overrides", {}).get("model", {})
        formal_scale = item.get("formal_expert_scale_selection", {})
        if (
            model.get("nested_expert_scale_selection") is not True
            or int(model.get("nested_expert_scale_inner_folds", 0)) < 2
            or any(
                float(model.get(name, float("nan"))) != 1.0
                for name in (
                    "strain_expert_scale",
                    "chemical_expert_scale",
                    "pair_expert_scale",
                )
            )
            or not isinstance(formal_scale, dict)
            or formal_scale.get("method") != "per_outer_fold_nested_inner_oof"
            or formal_scale.get("global_nomination_binding") is not False
        ):
            raise ValueError(
                f"Formal candidate {item.get('id')} lacks train-only nested scale selection"
            )
    runnable_m8 = [
        str(item["id"])
        for item in all_experiments
        if str(item.get("model_id", "")).startswith("M8")
    ]
    # This automatic wave is deliberately M7-only.  If identity evidence is
    # later promoted, a separately reviewed M8 wave must be authorized rather
    # than silently consuming GPU in an old overnight command.
    if runnable_m8:
        raise ValueError(
            "M8 semantics became runnable; automatic M7-only confirmation "
            f"refuses to start them: {runnable_m8}"
        )
    blocked_m8 = [
        item
        for item in materialized.get("blocked_confirm_candidates", [])
        if str(item.get("model_id", "")).startswith("M8")
    ]
    template_m8 = [
        str(item["id"])
        for item in template.get("confirm_candidates", [])
        if str(item.get("model_id", "")).startswith("M8")
    ]
    if {str(item.get("id", "")) for item in blocked_m8} != set(template_m8):
        raise ValueError("Every formal M8 template candidate must have an explicit block receipt")

    required_ids = _dependency_closure(all_experiments, nominated)
    selected_experiments = [
        deepcopy(item)
        for item in all_experiments
        if str(item["id"]) in required_ids
    ]
    materialized["experiments"] = selected_experiments
    materialized["automatic_confirmation_selection"] = {
        "policy": "discovery_nomination_only_then_strict_confirmation",
        "promotion_candidates": nominated,
        "matrix_experiments": [str(item["id"]) for item in selected_experiments],
        "quick_or_fair_results_can_promote": False,
    }
    materialized["selection_receipt"] = {
        "path": str(scale_file),
        "sha256": sha256_file(scale_file),
        "selection_scope": "global_discovery_nomination_only",
        "discovery_selection_binding": False,
        "formal_scale_source": "outer_fold_train_only_nested_inner_oof",
        "note": (
            "The global quick/research scale grid can nominate a confirmation "
            "but cannot configure a formal outer fold. Each producer outer "
            "fold must select scales using its own train-only nested inner OOF."
        ),
    }
    _atomic_yaml(matrix_path, materialized)

    output.mkdir(parents=True, exist_ok=True)
    preconfirmation_path = output / "preconfirmation_evidence.json"
    _write_hashed_json(preconfirmation_path, evidence)
    m8_receipt = {
        "schema": "goai.m7_m8.semantic_block_receipt.v1",
        "protocol_label": PROTOCOL_LABEL,
        "generated_at": _now(),
        "status": "blocked",
        "route": "M8",
        "gpu_tasks_started": False,
        "identity_semantic_promotion_status": evidence["identity"][
            "semantic_promotion_status"
        ],
        "blocked_candidates": blocked_m8,
        "reason": (
            "Formal verified-only semantic identity/fit coverage is not ready; "
            "M8 experiments were not materialized into the selected matrix."
        ),
    }
    m8_path = output / "m8_blocked_receipt.json"
    _write_hashed_json(m8_path, m8_receipt)

    receipt = {
        "schema": "goai.m7_m8.confirmation_selection_receipt.v1",
        "protocol_label": PROTOCOL_LABEL,
        "official_score_status": "NOT_OFFICIAL",
        "generated_at": _now(),
        "status": "prepared",
        "discovery_results_can_promote": False,
        "discovery_scale_selection_binding": False,
        "formal_scale_source": "outer_fold_train_only_nested_inner_oof",
        "nomination_thresholds": {
            "fc_absolute_delta": NOMINATION_FC_ABSOLUTE,
            "fc_delta_with_positive_fold_ci": NOMINATION_FC_WITH_POSITIVE_FOLD_CI,
            "residual_min_delta": 0.0,
            "high_effect_max_drop": HIGH_EFFECT_MAX_DROP,
        },
        "promotion_candidates": nominated,
        "matrix_experiments": [str(item["id"]) for item in selected_experiments],
        "nomination_decisions": nomination_decisions,
        "matrix": {"path": str(matrix_path), "sha256": sha256_file(matrix_path)},
        "preconfirmation_evidence": {
            "path": str(preconfirmation_path),
            "sha256": sha256_file(preconfirmation_path),
        },
        "m8_blocked_receipt": {
            "path": str(m8_path),
            "sha256": sha256_file(m8_path),
        },
        "fair_paired_summary": {
            "path": str(fair_path),
            "sha256": sha256_file(fair_path),
        },
        "scale_candidate_grid": scale_grid,
        "summary_evidence": summary_evidence,
    }
    receipt_path = output / "confirmation_selection_receipt.json"
    _write_hashed_json(receipt_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--scale-selection", required=True)
    parser.add_argument("--scale-candidates", required=True)
    parser.add_argument("--identity-audit", required=True)
    parser.add_argument("--calibration-audit", required=True)
    parser.add_argument("--fair-expert-audit", required=True)
    parser.add_argument("--fair-paired-summary", required=True)
    parser.add_argument("--quick-summary-dir", required=True)
    parser.add_argument("--research-summary-dir", required=True)
    parser.add_argument("--output-matrix", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    receipt = prepare_confirmation(
        template_path=args.template,
        scale_selection_path=args.scale_selection,
        scale_candidates_path=args.scale_candidates,
        identity_audit_path=args.identity_audit,
        calibration_audit_path=args.calibration_audit,
        fair_expert_audit_path=args.fair_expert_audit,
        fair_paired_summary_path=args.fair_paired_summary,
        quick_summary_dir=args.quick_summary_dir,
        research_summary_dir=args.research_summary_dir,
        output_matrix_path=args.output_matrix,
        output_dir=args.output_dir,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
