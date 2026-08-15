from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import scripts.nightly.promotion_gate as promotion_gate_module

from scripts.nightly.promotion_gate import (
    CONFIRM_SEEDS,
    PROTOCOL_LABEL,
    _cluster_bootstrap,
    _parse_scenarios,
    _validate_confirmation_runs,
    decide_promotion,
    semantic_coverage_passes,
    validate_joint_primary_contract,
)
from scripts.nightly.compare_matrix import discovery_decisions


def _units(
    scenarios: tuple[str, ...] = ("R10", "R01"),
    *,
    fc_delta: float = 0.02,
    residual_delta: float = 0.015,
    high_effect_delta: float = -0.004,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    definitions = {
        "R00": (("Strains", "perturbation_no_concentration"), ()),
        "R10": (("perturbation_no_concentration",), ("context_residual_pcc",)),
        "R01": (("Strains",), ("drug_residual_pcc",)),
        "R11": (("Strains+perturbation_no_concentration",), ("context_residual_pcc", "drug_residual_pcc")),
        "RT": (("time_group",), ("context_residual_pcc", "drug_residual_pcc")),
    }
    for scenario in scenarios:
        axes, residuals = definitions[scenario]
        for axis in axes:
            for entity_index in range(8):
                for seed in CONFIRM_SEEDS:
                    jitter = (entity_index - 3.5) * 0.0002 + (seed - 52) * 0.00001
                    values = {
                        "fc_pcc": fc_delta + jitter,
                        "high_effect_pcc": high_effect_delta + jitter,
                        "high_effect_f1": high_effect_delta + jitter,
                        **{metric: residual_delta + jitter for metric in residuals},
                    }
                    for metric, delta in values.items():
                        rows.append(
                            {
                                "model_seed": seed,
                                "scenario": scenario,
                                "cluster_axis": axis,
                                "heldout_entity": f"entity-{entity_index}",
                                "metric": metric,
                                "delta": delta,
                            }
                        )
    return pd.DataFrame(rows)


def test_strict_promotion_passes_only_with_entity_ci_three_seeds_and_guardrails() -> None:
    checks, receipt = decide_promotion(
        _units(), ("R10", "R01"), bootstrap_draws=2_000
    )
    assert receipt["promoted"] is True
    assert receipt["status"] == "promoted"
    assert bool(checks["passed"].all())
    assert set(checks["metric"]) == {
        "fc_pcc",
        "context_residual_pcc",
        "drug_residual_pcc",
        "high_effect_pcc",
        "high_effect_f1",
    }
    assert checks.loc[checks["metric"].eq("fc_pcc"), "ci_low"].gt(0).all()


@pytest.mark.parametrize(
    ("mutation", "expected_metric"),
    [
        ("fc_below_point01", "fc_pcc"),
        ("one_seed_wrong_direction", "fc_pcc"),
        ("residual_not_up", "context_residual_pcc"),
        ("high_effect_pcc_drop", "high_effect_pcc"),
        ("high_effect_f1_drop", "high_effect_f1"),
    ],
)
def test_each_locked_guardrail_blocks_promotion(mutation: str, expected_metric: str) -> None:
    units = _units(("R10",))
    if mutation == "fc_below_point01":
        units.loc[units["metric"].eq("fc_pcc"), "delta"] = 0.009
    elif mutation == "one_seed_wrong_direction":
        units.loc[
            units["metric"].eq("fc_pcc") & units["model_seed"].eq(62), "delta"
        ] = -0.001
    elif mutation == "residual_not_up":
        units.loc[units["metric"].eq("context_residual_pcc"), "delta"] = 0.0
    elif mutation == "high_effect_pcc_drop":
        units.loc[units["metric"].eq("high_effect_pcc"), "delta"] = -0.0051
    elif mutation == "high_effect_f1_drop":
        units.loc[units["metric"].eq("high_effect_f1"), "delta"] = -0.0051
    checks, receipt = decide_promotion(units, ("R10",), bootstrap_draws=1_000)
    assert receipt["promoted"] is False
    failed = checks.loc[~checks["passed"]]
    assert expected_metric in set(failed["metric"])


def test_r00_alone_cannot_bypass_required_residual_improvement() -> None:
    _, receipt = decide_promotion(_units(("R00",)), ("R00",), bootstrap_draws=500)
    assert receipt["promoted"] is False
    assert any("R00 alone" in reason for reason in receipt["reasons"])


def test_cluster_bootstrap_uses_entities_after_seed_averaging() -> None:
    frame = pd.DataFrame(
        [
            {"heldout_entity": entity, "model_seed": seed, "delta": value}
            for entity, value in (("A", 0.01), ("B", 0.03), ("C", 0.05))
            for seed in CONFIRM_SEEDS
        ]
    )
    result = _cluster_bootstrap(frame, seed=1, draws=2_000)
    assert result["n_heldout_entities"] == 3
    assert result["mean_delta"] == pytest.approx(0.03)
    assert result["all_seeds_positive"] is True
    assert set(result["seed_means"]) == {"42", "52", "62"}


def _write_discovery_contract(root: Path, kind: str) -> None:
    items = []
    for producer in ("CONTROL", "CANDIDATE"):
        item = {"id": producer, "kind": kind}
        if kind == "model_confirm":
            epochs = 64 if producer == "CONTROL" else 80
            item.update(
                {
                    "promotion_eligible": producer != "CONTROL",
                    "confirmation_training_variant": (
                        "universal_parent_control"
                        if producer == "CONTROL"
                        else "frozen_residual_only"
                    ),
                    "overrides": {"model": {"epochs": epochs}},
                }
            )
        items.append(item)
    matrix = root / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "protocol_label": PROTOCOL_LABEL,
                "experiments": items,
            }
        ),
        encoding="utf-8",
    )
    (root / "environment.json").write_text(
        json.dumps({"matrix": str(matrix)}), encoding="utf-8"
    )


def test_discovery_quick_kind_is_never_accepted_as_confirmation(tmp_path: Path) -> None:
    _write_discovery_contract(tmp_path, "model_screen")
    with pytest.raises(ValueError, match="kind=model_confirm"):
        _validate_confirmation_runs(
            tmp_path, ("CONTROL", "CANDIDATE"), ("R10",)
        )


def _write_complete_confirmation_contract(root: Path) -> None:
    _write_discovery_contract(root, "model_confirm")
    assignment = pd.DataFrame(
        {
            "scenario": ["R10"],
            "fold": [0],
            "sample_ID": ["sample-1"],
            "Strains": ["strain-1"],
            "perturbation_no_concentration": ["chemical-1"],
            "time_group": ["time-1"],
            "eligible": [True],
            "exclusion_reason": [""],
        }
    )
    for producer in ("CONTROL", "CANDIDATE"):
        for seed in CONFIRM_SEEDS:
            run = root / "producers" / producer / f"S{seed}"
            (run / "oof_predictions").mkdir(parents=True)
            (run / "oof_predictions" / "R10.npz").write_bytes(b"placeholder")
            assignment.to_csv(run / "fold_assignments.csv", index=False)
            pd.DataFrame({"scenario": ["R10"], "fc_pcc_mean": [0.3]}).to_csv(
                run / "oof_summary.csv", index=False
            )
            contract = {
                "protocol": "support_regime_oof_run_contract_v3",
                "fingerprint_sha256": f"fingerprint-{producer}-{seed}",
                "n_folds": 4,
                "seed": 42,
                "model_seed": seed,
                "scenarios": ["R00", "R10", "R01", "R11", "RT"],
                "effective_config": {
                    "model": {"epochs": 64 if producer == "CONTROL" else 80}
                },
                "source_fingerprint": {"sha256": "same-source"},
                "input_hashes": {"metadata": "same-data"},
            }
            (run / "run_contract.json").write_text(
                json.dumps(contract), encoding="utf-8"
            )
            (run / "oof_manifest.json").write_text(
                json.dumps(
                    {
                        "protocol": "support_regime_oof_v2",
                        "n_folds": 4,
                        "model_seed": seed,
                        "audit_only": False,
                    }
                ),
                encoding="utf-8",
            )
            for fit in range(44):
                completed = run / "folds" / f"fit-{fit}" / "completed.json"
                completed.parent.mkdir(parents=True)
                completed.write_text(
                    json.dumps(
                        {
                            "training_receipt": {
                                "enabled": True,
                                "universal_state_sha256": "universal",
                                "copied_universal_state_sha256": "universal",
                                "post_frozen_expert_universal_state_sha256": "universal",
                                "final_universal_state_sha256": "final",
                                "common_state_unchanged_during_frozen_experts": True,
                            }
                        }
                    ),
                    encoding="utf-8",
                )


def test_confirmation_contract_requires_complete_4fold_3seed_80epoch_receipts(
    tmp_path: Path,
) -> None:
    _write_complete_confirmation_contract(tmp_path)
    result = _validate_confirmation_runs(
        tmp_path, ("CONTROL", "CANDIDATE"), ("R10",)
    )
    assert len(result["checks"]) == 6
    assert {row["completed_fits"] for row in result["checks"]} == {44}
    assert {row["fold_matched_training_receipts"] for row in result["checks"]} == {44}
    assert result["source_fingerprint_sha256"] == "same-source"

    path = tmp_path / "producers" / "CANDIDATE" / "S52" / "run_contract.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract["effective_config"]["model"]["epochs"] = 12
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="confirmation contract"):
        _validate_confirmation_runs(
            tmp_path, ("CONTROL", "CANDIDATE"), ("R10",)
        )


def test_formal_nested_scale_contract_requires_all_44_fold_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_complete_confirmation_contract(tmp_path)
    matrix_path = tmp_path / "matrix.yaml"
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    matrix["non_binding_expert_scale_nomination"] = {
        "binding_to_formal_predictions": False,
        "formal_policy": "per_outer_fold_nested_inner_oof",
    }
    for item in matrix["experiments"]:
        item.setdefault("overrides", {}).setdefault("model", {}).update(
            {
                "nested_expert_scale_selection": True,
                "nested_expert_scale_inner_folds": 2,
                "strain_expert_scale": 1.0,
                "chemical_expert_scale": 1.0,
                "pair_expert_scale": 1.0,
            }
        )
        item["formal_expert_scale_selection"] = {
            "method": "per_outer_fold_nested_inner_oof",
            "global_nomination_binding": False,
        }
    matrix_path.write_text(yaml.safe_dump(matrix), encoding="utf-8")
    environment_path = tmp_path / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["matrix_sha256"] = promotion_gate_module._sha256(matrix_path)
    environment_path.write_text(json.dumps(environment), encoding="utf-8")

    for producer in ("CONTROL", "CANDIDATE"):
        for seed in CONFIRM_SEEDS:
            run = tmp_path / "producers" / producer / f"S{seed}"
            contract_path = run / "run_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["effective_config"]["model"].update(
                {
                    "nested_expert_scale_selection": True,
                    "nested_expert_scale_inner_folds": 2,
                    "strain_expert_scale": 1.0,
                    "chemical_expert_scale": 1.0,
                    "pair_expert_scale": 1.0,
                }
            )
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            manifest_path = run / "oof_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["nested_expert_scale_selection"] = {
                "enabled": True,
                "protocol": promotion_gate_module.NESTED_SCALE_PROTOCOL,
                "inner_n_folds": 2,
                "global_scale_used": False,
                "outer_validation_labels_used": False,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            for completed_path in (run / "folds").glob("*/completed.json"):
                completed = json.loads(completed_path.read_text(encoding="utf-8"))
                completed.update(
                    {
                        "scenario": "R10",
                        "fold": 0,
                        "train_ids_sha256": "train",
                        "validation_ids_sha256": "validation",
                        "nested_expert_scale_protocol": promotion_gate_module.NESTED_SCALE_PROTOCOL,
                        "nested_expert_scale_receipt_sha256": "nested-hash",
                    }
                )
                completed_path.write_text(json.dumps(completed), encoding="utf-8")

    calls: list[dict[str, object]] = []

    def validate_nested(directory, **kwargs):
        calls.append({"directory": directory, **kwargs})
        return {"status": "selected"}

    monkeypatch.setattr(
        promotion_gate_module, "validate_nested_scale_receipt", validate_nested
    )
    result = _validate_confirmation_runs(
        tmp_path, ("CONTROL", "CANDIDATE"), ("R10",)
    )
    assert len(calls) == 2 * len(CONFIRM_SEEDS) * 44
    assert {row["nested_scale_receipts"] for row in result["checks"]} == {44}
    assert all(row["nested_scale_required"] is True for row in result["checks"])
    assert all(call["expected_sha256"] == "nested-hash" for call in calls)

    broken = (
        tmp_path
        / "producers"
        / "CANDIDATE"
        / "S52"
        / "folds"
        / "fit-0"
        / "completed.json"
    )
    payload = json.loads(broken.read_text(encoding="utf-8"))
    payload["nested_expert_scale_receipt_sha256"] = ""
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks a nested scale receipt"):
        _validate_confirmation_runs(
            tmp_path, ("CONTROL", "CANDIDATE"), ("R10",)
        )


def test_formal_scale_nomination_cannot_be_changed_back_to_global(
    tmp_path: Path,
) -> None:
    _write_complete_confirmation_contract(tmp_path)
    matrix_path = tmp_path / "matrix.yaml"
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    matrix["non_binding_expert_scale_nomination"] = {
        "binding_to_formal_predictions": True,
        "formal_policy": "global_discovery_scale",
    }
    matrix_path.write_text(yaml.safe_dump(matrix), encoding="utf-8")
    with pytest.raises(ValueError, match="non-binding/nested-inner-OOF"):
        _validate_confirmation_runs(
            tmp_path, ("CONTROL", "CANDIDATE"), ("R10",)
        )


def test_legacy_quick_compare_can_nominate_but_never_promote() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate": "QUICK",
                "model_seed": 42,
                "scenario": "R10",
                "metric": "fc_pcc",
                "mean_delta": 0.5,
                "ci_low": 0.4,
            },
            {
                "candidate": "QUICK",
                "model_seed": 42,
                "scenario": "R10",
                "metric": "high_effect_pcc",
                "mean_delta": 0.2,
                "ci_low": 0.1,
            },
        ]
    )
    [decision] = discovery_decisions(frame)
    assert decision["nominate_for_confirmation"] is True
    assert decision["promotion_eligible"] is False
    assert decision["promote"] is False
    assert decision["protocol_label"] == PROTOCOL_LABEL


def test_promotion_regimes_are_required_and_read_from_confirmation_metadata() -> None:
    metadata = {"promotion_regimes": ["R00", "R10"]}
    assert _parse_scenarios(None, metadata) == ("R00", "R10")
    assert _parse_scenarios(["R01"], metadata) == ("R01",)
    with pytest.raises(ValueError, match="missing"):
        _parse_scenarios(None, {})


def test_m8_metadata_separates_primary_zero_semantic_and_shuffled_controls() -> None:
    project = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load(
        (
            project
            / "configs/nightly/20260813-m7-m8/confirm_candidates.yaml"
        ).read_text(encoding="utf-8")
    )
    metadata = {item["id"]: item for item in payload["confirm_candidates"]}
    semantic_candidates = [
        item
        for item in metadata.values()
        if str(item.get("model_id", "")).startswith("M8")
        and item.get("promotion_eligible") is not False
    ]
    assert semantic_candidates
    for item in semantic_candidates:
        expected_primary = (
            "CONF-M7.4-PAIR"
            if item["id"] == "CONF-M8.3-DUAL-PAIR"
            else "CONF-M7.3-ENTITIES"
        )
        assert item["primary_control"] == expected_primary
        assert "CONF-M7.3-ENTITIES" not in item.get("required_negative_controls", [])
    for experiment_id in (
        "CONF-M8.0-MORGAN",
        "CONF-M8.1-STRAIN-REAL",
        "CONF-M8.0-CHEMBERTA-REAL",
    ):
        controls = metadata[experiment_id]["required_negative_controls"]
        assert len(controls) == 1
        assert metadata[controls[0]]["promotion_eligible"] is False
    # Parent/component proxy fusion remains a labelled quick-screen research
    # ablation; it cannot be materialized in verified-only confirmation.
    assert "CONF-M8.0-FUSION-REAL" not in metadata
    assert "CONF-M8.0-FUSION-SHUFFLED" not in metadata


def test_m8_zero_or_partial_semantic_coverage_can_never_promote() -> None:
    assert semantic_coverage_passes({}, semantic_candidate=True) is False
    assert semantic_coverage_passes(
        {"chemical": {"passed": False}}, semantic_candidate=True
    ) is False
    assert semantic_coverage_passes(
        {"chemical": {"passed": True}, "strain": {"passed": False}},
        semantic_candidate=True,
    ) is False
    assert semantic_coverage_passes(
        {"chemical": {"passed": True}, "strain": {"passed": True}},
        semantic_candidate=True,
    ) is True
    # M7 contains no open-knowledge semantic block, so this specific gate is
    # not applicable; it still must pass all metric/confirmation gates.
    assert semantic_coverage_passes({}, semantic_candidate=False) is True


def test_joint_primary_must_have_same_universal_update_budget() -> None:
    candidate = {
        "confirmation_training_variant": "joint_finetune",
        "universal_update_budget": {"total_universal_update_epochs": 96},
    }
    same_update = {
        "confirmation_training_variant": "same_universal_update_control",
        "universal_update_budget": {"total_universal_update_epochs": 96},
    }
    joint_control = {
        "confirmation_training_variant": "joint_finetune",
        "universal_update_budget": {"total_universal_update_epochs": 96},
    }
    validate_joint_primary_contract(candidate, same_update)
    validate_joint_primary_contract(candidate, joint_control)
    with pytest.raises(ValueError, match="different universal-update budgets"):
        validate_joint_primary_contract(
            candidate,
            {
                "confirmation_training_variant": "same_universal_update_control",
                "universal_update_budget": {"total_universal_update_epochs": 95},
            },
        )
    with pytest.raises(ValueError, match="same_universal_update_control"):
        validate_joint_primary_contract(
            candidate,
            {
                "confirmation_training_variant": "frozen_residual_only",
                "universal_update_budget": {"total_universal_update_epochs": 96},
            },
        )
