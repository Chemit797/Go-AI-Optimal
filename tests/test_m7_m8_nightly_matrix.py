from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from goai_response.config import load_response_config
from goai_response.nested_scale import PROTOCOL as NESTED_SCALE_PROTOCOL
from goai_response.nested_scale import write_receipt as write_nested_scale_receipt
from scripts.nightly.build_m7_m8_confirm import (
    PROTOCOL_LABEL,
    materialize_confirm_matrix,
)
from scripts.nightly.audit_fair_expert_receipts import audit_receipts
from scripts.nightly.run_matrix import (
    _absolute_config,
    _deep_update,
    _resolve_project_paths,
    _validate_completed_run,
)
from scripts.nightly.run_matrix import _write_environment
from scripts.nightly.summarize_m7_m8 import (
    ALLOWED_SCALES,
    _four_regime_macro,
    _paired_deltas,
    _select_expert_scales,
)


PROJECT = Path(__file__).resolve().parents[1]
MATRIX_DIR = PROJECT / "configs" / "nightly" / "20260813-m7-m8"
CORE = {"R00", "R10", "R01", "R11", "RT"}


def _load(name: str) -> dict:
    payload = yaml.safe_load((MATRIX_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _validate_effective_configs(matrix: dict, matrix_path: Path, experiments: list[dict], tmp_path: Path) -> None:
    base_path = Path(matrix["base_config"])
    base_path = base_path if base_path.is_absolute() else (matrix_path.parent / base_path).resolve()
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    absolute = _absolute_config(base, base_path, tmp_path / "runs")
    for experiment in experiments:
        payload = _resolve_project_paths(
            _deep_update(deepcopy(absolute), experiment.get("overrides", {}))
        )
        path = tmp_path / f"{experiment['id']}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        config = load_response_config(path)
        staged = (
            config.model.universal_epochs
            + config.model.strain_expert_epochs
            + config.model.chemical_expert_epochs
            + config.model.pair_expert_epochs
        )
        assert staged <= config.model.epochs


def test_quick_screen_is_fair_parseable_and_has_fixed_scale_grid(tmp_path: Path) -> None:
    matrix = _load("quick_screen.yaml")
    experiments = matrix["experiments"]
    ids = [str(item["id"]) for item in experiments]
    by_id = {str(item["id"]): item for item in experiments}
    assert len(ids) == len(set(ids))
    assert matrix["n_folds"] == 2
    assert matrix["protocol_label"] == PROTOCOL_LABEL

    primary = [item for item in experiments if item.get("kind") == "model_screen"]
    required = {
        "SCR-M7.0-GENERAL",
        "SCR-M7.1-STRAIN",
        "SCR-M7.2-CHEMICAL",
        "SCR-M7.2-CHEM-PROTOTYPE",
        "SCR-M7.3-ENTITIES",
        "SCR-M7.4-PAIR",
        "SCR-M8.1-STRAIN-REAL",
        "SCR-M8.1-STRAIN-SHUFFLED",
        "SCR-M8.2-DUAL",
        "SCR-M8.0-CHEMBERTA-REAL",
        "SCR-M8.0-CHEMBERTA-SHUFFLED",
        "SCR-M8.0-FUSION-REAL",
        "SCR-M8.0-FUSION-SHUFFLED",
        "SCR-M8.0-MORGAN-EXACT",
        "SCR-M8.0-MORGAN-PARENT",
        "SCR-M8.0-MORGAN-ZERO-RISKY",
        "SCR-M8.0-CHEMBERTA-PARENT",
        "SCR-M8.0-CHEMBERTA-ZERO-RISKY",
        "SCR-M8.3-DUAL-PAIR",
    }
    assert required.issubset({item["id"] for item in primary})
    for item in primary:
        assert set(item["scenarios"]) == CORE
        assert item["seeds"] == [42]
        assert item["overrides"]["model"]["epochs"] == 12

    scale_jobs = [item for item in experiments if item.get("kind") == "expert_scale_screen"]
    assert len(scale_jobs) == 10
    for axis, scenario in (("strain", "R10"), ("chemical", "R01")):
        jobs = [item for item in scale_jobs if item["selection_axis"] == axis]
        assert {float(item["selection_value"]) for item in jobs} == set(ALLOWED_SCALES)
        assert all(item["scenarios"] == [scenario] for item in jobs)
        assert all(item["overrides"]["model"][f"{axis}_expert_scale"] in ALLOWED_SCALES for item in jobs)

    # Pair scale is an explicit, separate research screen because it is only
    # identifiable in seen-entity R11/RT regimes.
    pair_matrix = _load("research_prior_pair_ablation.yaml")
    pair_jobs = [
        item
        for item in pair_matrix["experiments"]
        if item.get("selection_axis") == "pair"
    ]
    assert {float(item["selection_value"]) for item in pair_jobs} == set(ALLOWED_SCALES)

    _validate_effective_configs(matrix, MATRIX_DIR / "quick_screen.yaml", experiments, tmp_path)


def test_fold_matched_expert_ablation_has_exact_universal_receipt_contract(
    tmp_path: Path,
) -> None:
    matrix = _load("fair_expert_ablation.yaml")
    experiments = matrix["experiments"]
    ids = {str(item["id"]): item for item in experiments}
    assert set(ids) == {
        "FAIR-M7.0-U9",
        "FAIR-M7.1-U9-S2-FROZEN",
        "FAIR-M7.1-U9-S2-J3",
        "FAIR-M7.2-U9-C2-FROZEN",
        "FAIR-M7.2-U9-C2-J3",
        "FAIR-M7.3-U9-S2-C2-FROZEN",
        "FAIR-M7.3-U9-S2-C2-J3",
    }
    assert all(set(item["scenarios"]) == CORE for item in experiments)
    for item in experiments:
        model = item["overrides"]["model"]
        assert model["universal_epochs"] == 9
        assert model["fold_matched_universal_warm_start"] is True
    assert ids["FAIR-M7.0-U9"]["overrides"]["model"]["epochs"] == 9
    for entity in ("M7.1-U9-S2", "M7.2-U9-C2"):
        frozen_id = f"FAIR-{entity}-FROZEN"
        joint_id = f"FAIR-{entity}-J3"
        assert ids[frozen_id]["overrides"]["model"]["epochs"] == 11
        assert ids[joint_id]["overrides"]["model"]["epochs"] == 14
    frozen = ids["FAIR-M7.3-U9-S2-C2-FROZEN"]["overrides"]["model"]
    assert frozen["epochs"] == 13
    assert frozen["strain_expert_epochs"] == 2
    assert frozen["chemical_expert_epochs"] == 2
    joint = ids["FAIR-M7.3-U9-S2-C2-J3"]["overrides"]["model"]
    assert joint["epochs"] == 16
    # R00=4, R10=2, R01=2, R11=4, RT=2 fits per task at two folds.
    assert matrix["expected_fits_per_task"] == 14
    assert matrix["expected_total_fits"] == len(experiments) * 14
    _validate_effective_configs(
        matrix,
        MATRIX_DIR / "fair_expert_ablation.yaml",
        experiments,
        tmp_path,
    )


def test_fair_receipt_auditor_checks_all_source_parent_and_fit_receipts(
    tmp_path: Path,
) -> None:
    producers = {
        "FAIR-M7.0-U9",
        "FAIR-M7.1-U9-S2-FROZEN",
        "FAIR-M7.1-U9-S2-J3",
        "FAIR-M7.2-U9-C2-FROZEN",
        "FAIR-M7.2-U9-C2-J3",
        "FAIR-M7.3-U9-S2-C2-FROZEN",
        "FAIR-M7.3-U9-S2-C2-J3",
    }
    folds = [
        *(('R00', fold) for fold in range(4)),
        *(('R10', fold) for fold in range(2)),
        *(('R01', fold) for fold in range(2)),
        *(('R11', fold) for fold in range(4)),
        *(('RT', fold) for fold in range(2)),
    ]
    import json

    for producer in producers:
        run = tmp_path / "producers" / producer / "S42"
        run.mkdir(parents=True)
        (run / "run_contract.json").write_text(
            json.dumps(
                {
                    "fingerprint_sha256": f"contract-{producer}",
                    "source_fingerprint": {"sha256": "source-v1"},
                }
            ),
            encoding="utf-8",
        )
        for scenario, fold in folds:
            parent = f"parent-{scenario}-{fold}"
            is_frozen = producer.endswith("-FROZEN")
            final = parent if is_frozen or producer == "FAIR-M7.0-U9" else f"joint-{parent}"
            payload = {
                "scenario": scenario,
                "fold": fold,
                "training_receipt": {
                    "universal_state_sha256": parent,
                    "copied_universal_state_sha256": parent,
                    "post_frozen_expert_universal_state_sha256": parent,
                    "final_universal_state_sha256": final,
                    "common_state_unchanged_during_frozen_experts": True,
                },
            }
            path = run / "folds" / f"{scenario}_fold_{fold}" / "completed.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(payload), encoding="utf-8")

    table, summary = audit_receipts(tmp_path)
    assert len(table) == 98
    assert summary["status"] == "valid"
    assert summary["invalid_receipt_rows"] == 0
    assert set(summary["task_fit_counts"].values()) == {14}

    tampered = tmp_path / "producers" / "FAIR-M7.2-U9-C2-FROZEN" / "S42" / "folds" / "R01_fold_0" / "completed.json"
    payload = json.loads(tampered.read_text(encoding="utf-8"))
    payload["training_receipt"]["copied_universal_state_sha256"] = "wrong"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    _, invalid = audit_receipts(tmp_path)
    assert invalid["status"] == "invalid"
    assert invalid["invalid_receipt_rows"] == 1


def test_calibration_audit_covers_rank_shuffle_no_plate_and_dropout(tmp_path: Path) -> None:
    matrix = _load("calibration_audit.yaml")
    experiments = matrix["experiments"]
    ids = [str(item["id"]) for item in experiments]
    assert len(ids) == len(set(ids))
    assert matrix["n_folds"] == 2
    assert {4, 8, 16}.issubset(
        {int(item["overrides"]["model"]["calibration_rank"]) for item in experiments}
    )
    assert any(item["overrides"]["model"].get("calibration_plate_shuffle") for item in experiments)
    assert any(not item["overrides"]["model"].get("calibration_use_plate", True) for item in experiments)
    assert {0.0, 0.3, 0.5}.issubset(
        {float(item["overrides"]["model"]["calibration_plate_dropout"]) for item in experiments}
    )
    for item in experiments:
        assert {"plate", *CORE}.issubset(set(item["scenarios"]))
        assert item["seeds"] == [42]
        assert item["overrides"]["model"]["epochs"] == 12
    _validate_effective_configs(matrix, MATRIX_DIR / "calibration_audit.yaml", experiments, tmp_path)


def test_confirm_requires_selection_then_materializes_4fold_3seed_80epoch(tmp_path: Path) -> None:
    template = _load("confirm_candidates.yaml")
    incomplete = {"selection_status": "incomplete", "protocol_label": PROTOCOL_LABEL}
    with pytest.raises(ValueError, match="incomplete"):
        materialize_confirm_matrix(template, incomplete)

    selection = {
        "selection_status": "selected",
        "protocol_label": PROTOCOL_LABEL,
        "method": "synthetic test receipt",
        "selected": {
            "strain": {"scale": 0.5},
            "chemical": {"scale": 0.75},
            "pair": {"scale": 0.25},
        },
    }
    base_path = (MATRIX_DIR / template["base_config"]).resolve()
    matrix = materialize_confirm_matrix(
        template,
        selection,
        absolute_base_config=str(base_path),
    )
    experiments = matrix["experiments"]
    ids = [str(item["id"]) for item in experiments]
    by_id = {str(item["id"]): item for item in experiments}
    assert len(ids) == len(set(ids))
    assert matrix["n_folds"] == 4
    assert any(item["id"].endswith("-JOINT") for item in experiments)
    assert {item["id"] for item in matrix["blocked_confirm_candidates"]} == {
        "CONF-M8.0-MORGAN",
        "CONF-M8.0-MORGAN-SHUFFLED",
        "CONF-M8.1-STRAIN-REAL",
        "CONF-M8.1-STRAIN-SHUFFLED",
        "CONF-M8.2-DUAL",
        "CONF-M8.3-DUAL-PAIR",
        "CONF-M8.0-CHEMBERTA-REAL",
        "CONF-M8.0-CHEMBERTA-SHUFFLED",
    }
    for item in experiments:
        assert item["seeds"] == [42, 52, 62]
        assert set(item["scenarios"]) == CORE
        assert item["overrides"]["model"]["epochs"] >= 80
        expected_universal = (
            96
            if item["id"] == "CONF-M7.0-GENERAL-U96"
            else 80
        )
        assert item["overrides"]["model"]["universal_epochs"] == expected_universal
        assert item["overrides"]["model"]["fold_matched_universal_warm_start"] is True
        staged = sum(
            int(item["overrides"]["model"].get(name, 0))
            for name in (
                "universal_epochs",
                "strain_expert_epochs",
                "chemical_expert_epochs",
                "pair_expert_epochs",
            )
        )
        if item["confirmation_training_variant"].startswith("frozen_residual_only"):
            assert item["overrides"]["model"]["epochs"] == staged
        elif item["confirmation_training_variant"] == "joint_finetune":
            assert item["overrides"]["model"]["epochs"] == staged + 16
            assert item["universal_update_budget"][
                "total_universal_update_epochs"
            ] == 96
        elif item["confirmation_training_variant"] == "same_universal_update_control":
            assert item["id"] == "CONF-M7.0-GENERAL-U96"
            assert item["overrides"]["model"]["epochs"] == 96
            assert item["universal_update_budget"][
                "total_universal_update_epochs"
            ] == 96
        policy = item["nested_scale_policy"]
        model = item["overrides"]["model"]
        assert policy in {"none", "strain", "chemical", "both", "all"}
        assert model["strain_expert_scale"] == 1.0
        assert model["chemical_expert_scale"] == 1.0
        assert model["pair_expert_scale"] == 1.0
        assert model["nested_expert_scale_selection"] is True
        assert model["nested_expert_scale_inner_folds"] == 2
        assert item["formal_expert_scale_selection"][
            "global_nomination_binding"
        ] is False
    assert "fixed_expert_scale_selection" not in matrix
    assert matrix["non_binding_expert_scale_nomination"] == {
        "strain_expert_scale": 0.5,
        "chemical_expert_scale": 0.75,
        "pair_expert_scale": 0.25,
        "method": "synthetic test receipt",
        "binding_to_formal_predictions": False,
        "formal_policy": "per_outer_fold_nested_inner_oof",
    }
    assert by_id["CONF-M7.1-STRAIN"]["primary_control"] == "CONF-M7.0-GENERAL"
    assert by_id["CONF-M7.2-CHEMICAL"]["primary_control"] == "CONF-M7.0-GENERAL"
    assert by_id["CONF-M7.3-ENTITIES"]["primary_control"] == "CONF-M7.0-GENERAL"
    for candidate in (
        "CONF-M7.1-STRAIN-JOINT",
        "CONF-M7.2-CHEMICAL-JOINT",
        "CONF-M7.3-ENTITIES-JOINT",
    ):
        assert by_id[candidate]["primary_control"] == "CONF-M7.0-GENERAL-U96"
    assert by_id["CONF-M7.4-PAIR"]["primary_control"] == "CONF-M7.3-ENTITIES"
    assert by_id["CONF-M7.4-PAIR-JOINT"]["primary_control"] == (
        "CONF-M7.3-ENTITIES-JOINT"
    )
    # Every formal comparison is self-contained in the materialized matrix.
    for item in experiments:
        primary = item.get("primary_control")
        if primary:
            assert primary in by_id
        for negative in item.get("required_negative_controls", []):
            assert negative in by_id
    _validate_effective_configs(matrix, MATRIX_DIR / "confirm_candidates.yaml", experiments, tmp_path)


def test_experiment_ids_are_unique_across_m7_m8_suite() -> None:
    ids: list[str] = []
    for name, key in (
        ("quick_screen.yaml", "experiments"),
        ("fair_expert_ablation.yaml", "experiments"),
        ("calibration_audit.yaml", "experiments"),
        ("confirm_candidates.yaml", "confirm_candidates"),
        ("research_prior_pair_ablation.yaml", "experiments"),
    ):
        ids.extend(str(item["id"]) for item in _load(name)[key])
    assert len(ids) == len(set(ids))


def test_research_prior_and_pair_scale_ablation_is_non_promotable_and_parseable(
    tmp_path: Path,
) -> None:
    matrix = _load("research_prior_pair_ablation.yaml")
    experiments = matrix["experiments"]
    assert matrix["promotion_eligible"] is False
    assert all(item["promotion_eligible"] is False for item in experiments)
    prior_jobs = [item for item in experiments if item["kind"] == "research_prior_ablation"]
    assert {item["overrides"]["model"]["response_prior_mode"] for item in prior_jobs} == {
        "none",
        "chemical",
        "strain",
        "both",
    }
    pair_jobs = [item for item in experiments if item.get("selection_axis") == "pair"]
    assert {item["selection_value"] for item in pair_jobs} == set(ALLOWED_SCALES)
    assert all(item["scenarios"] == ["R11", "RT"] for item in pair_jobs)
    _validate_effective_configs(
        matrix,
        MATRIX_DIR / "research_prior_pair_ablation.yaml",
        experiments,
        tmp_path,
    )


def test_matrix_path_resolution_covers_both_entity_registries(tmp_path: Path) -> None:
    matrix_path = MATRIX_DIR / "quick_screen.yaml"
    matrix = _load("quick_screen.yaml")
    base_path = (matrix_path.parent / matrix["base_config"]).resolve()
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    absolute = _absolute_config(base, base_path, tmp_path / "runs")
    experiment = next(item for item in matrix["experiments"] if item["id"] == "SCR-M8.2-DUAL")
    payload = _resolve_project_paths(_deep_update(deepcopy(absolute), experiment["overrides"]))
    for key in ("chemical_registry", "strain_registry", "chemical_parent_views"):
        path = Path(payload["entity"][key])
        assert path.is_absolute()
        assert path.is_file()
    zero = next(
        item
        for item in matrix["experiments"]
        if item["id"] == "SCR-M8.0-MORGAN-ZERO-RISKY"
    )
    zero_payload = _resolve_project_paths(
        _deep_update(deepcopy(absolute), zero["overrides"])
    )
    risk_path = Path(zero_payload["entity"]["chemical_identity_risks"])
    assert risk_path.is_absolute()
    assert risk_path.is_file()


def test_environment_records_project_and_run_root_filesystems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.nightly.run_matrix as runner

    run_root = tmp_path / "large-run-mount"
    run_root.mkdir()
    matrix_path = MATRIX_DIR / "quick_screen.yaml"

    def fake_free(path: Path) -> float:
        return 19.0 if Path(path).resolve() == runner.PROJECT.resolve() else 2900.0

    monkeypatch.setattr(runner, "_disk_free_gb", fake_free)
    _write_environment(run_root, matrix_path)
    import json

    payload = json.loads((run_root / "environment.json").read_text(encoding="utf-8"))
    assert payload["project_free_gb_at_start"] == 19.0
    assert payload["run_root_free_gb_at_start"] == 2900.0
    assert payload["disk_free_gb_at_start"] == 2900.0
    assert payload["disk_guard_path"] == str(run_root.resolve())
    assert payload["run_root_path"] == str(run_root.resolve())


def _completed_contract() -> dict:
    return {
        "protocol": "support_regime_oof_run_contract_v3",
        "fingerprint_sha256": "current-fingerprint",
        "response_config_sha256": "config-sha",
        "effective_config": {"model": {"epochs": 12}},
        "input_hashes": {"metadata_train_val_sha256": "input-sha"},
        "input_audit": {
            "metadata_train_val_rows": 8958,
            "metadata_train_val_sha256": "input-sha",
        },
        "external_artifacts": {"chemical_registry": {"sha256": "artifact-sha"}},
        "n_folds": 2,
        "seed": 42,
        "model_seed": 52,
        "scenarios": ["R00", "R10"],
        "source_fingerprint": {"sha256": "source-sha", "files": []},
    }


def _write_completed_contract(run: Path, contract: dict) -> None:
    import json

    run.mkdir(parents=True)
    (run / "oof_summary.csv").write_text(
        "scenario,fc_pcc_mean\nR00,0.1\n", encoding="utf-8"
    )
    (run / "run_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    (run / "oof_manifest.json").write_text(
        json.dumps(
            {"run_contract_fingerprint_sha256": contract["fingerprint_sha256"]}
        ),
        encoding="utf-8",
    )


def test_completed_matrix_task_is_reused_only_after_full_contract_validation(
    tmp_path: Path,
) -> None:
    contract = _completed_contract()
    run = tmp_path / "producer"
    _write_completed_contract(run, contract)

    receipt = _validate_completed_run(run, deepcopy(contract))

    assert receipt["fingerprint_sha256"] == "current-fingerprint"
    assert set(receipt["validated_sections"]) == {
        "source",
        "config",
        "inputs",
        "artifacts",
        "folds_and_seeds",
        "manifest_binding",
    }


@pytest.mark.parametrize(
    ("section", "mutation"),
    [
        ("source", lambda value: value["source_fingerprint"].update(sha256="new-source")),
        ("config", lambda value: value.update(response_config_sha256="new-config")),
        ("inputs", lambda value: value["input_hashes"].update(metadata_train_val_sha256="new-input")),
        ("artifacts", lambda value: value["external_artifacts"]["chemical_registry"].update(sha256="new-artifact")),
        ("folds_and_seeds", lambda value: value.update(model_seed=62)),
    ],
)
def test_completed_matrix_task_hard_rejects_any_contract_drift(
    tmp_path: Path,
    section: str,
    mutation,
) -> None:
    existing = _completed_contract()
    run = tmp_path / section
    _write_completed_contract(run, existing)
    expected = deepcopy(existing)
    mutation(expected)
    expected["fingerprint_sha256"] = "new-fingerprint"

    with pytest.raises(ValueError, match=section):
        _validate_completed_run(run, expected)


def test_completed_matrix_task_rejects_unbound_manifest(tmp_path: Path) -> None:
    import json

    contract = _completed_contract()
    run = tmp_path / "producer"
    _write_completed_contract(run, contract)
    (run / "oof_manifest.json").write_text(
        json.dumps({"run_contract_fingerprint_sha256": "stale"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest is not bound"):
        _validate_completed_run(run, contract)


def test_completed_formal_matrix_task_validates_nested_receipt_on_resume(
    tmp_path: Path,
) -> None:
    import json

    contract = _completed_contract()
    contract["effective_config"]["model"]["nested_expert_scale_selection"] = True
    run = tmp_path / "formal-producer"
    _write_completed_contract(run, contract)
    fold = run / "folds" / "R00_fold_0"
    receipt_payload = {
        "protocol": NESTED_SCALE_PROTOCOL,
        "status": "not_applicable",
        "scenario": "R00",
        "outer_fold": 0,
        "outer_train_ids_sha256": "train",
        "outer_validation_ids_sha256": "validation",
        "source_contract_fingerprint_sha256": contract["fingerprint_sha256"],
        "global_scale_used": False,
        "outer_validation_labels_used": False,
        "canonical_training_scales": {
            "strain": 1.0,
            "chemical": 1.0,
            "pair": 1.0,
        },
        "active_axes": [],
        "inner_n_folds": 2,
        "selected_scales": {"strain": 0.0, "chemical": 0.0, "pair": 0.0},
    }
    _, receipt_hash = write_nested_scale_receipt(
        fold / "nested_expert_scale",
        payload=receipt_payload,
        assignments=pd.DataFrame(
            columns=["scenario", "fold", "sample_ID", "eligible"]
        ),
        candidates=pd.DataFrame(
            [{"scenario": "R00", "status": "not_applicable"}]
        ),
        fit_receipts=[],
    )
    (fold / "completed.json").write_text(
        json.dumps(
            {
                "scenario": "R00",
                "fold": 0,
                "train_ids_sha256": "train",
                "validation_ids_sha256": "validation",
                "nested_expert_scale_receipt_sha256": receipt_hash,
            }
        ),
        encoding="utf-8",
    )
    validation = _validate_completed_run(run, contract)
    assert validation["nested_inner_oof_scale_receipts"] == 1
    assert "nested_inner_oof_scale_receipts" in validation["validated_sections"]

    with (fold / "nested_expert_scale" / "candidate_metrics.csv").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="tampered"):
        _validate_completed_run(run, contract)


def _write_single_completed_matrix(tmp_path: Path, contract: dict) -> tuple[Path, Path]:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "baseline_config": "unused-baseline.yaml",
                "model": {"seed": 42, "device": "cpu"},
                "entity": {},
                "graph": {},
                "runtime": {"runs_dir": "runs"},
            }
        ),
        encoding="utf-8",
    )
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "base_config": str(base),
                "n_folds": 2,
                "fold_seed": 42,
                "experiments": [
                    {
                        "id": "ONE",
                        "seeds": [52],
                        "scenarios": ["R00", "R10"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    run_root = tmp_path / "runs"
    _write_completed_contract(run_root / "producers" / "ONE" / "S52", contract)
    return matrix, run_root


def test_run_matrix_preflights_valid_completion_before_skipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import sys
    import scripts.nightly.run_matrix as runner

    contract = _completed_contract()
    matrix, run_root = _write_single_completed_matrix(tmp_path, contract)
    monkeypatch.setattr(
        runner,
        "_expected_run_contract",
        lambda *_args, **_kwargs: deepcopy(contract),
    )
    monkeypatch.setattr(runner, "_disk_free_gb", lambda _path: 1000.0)
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("validated completion must not launch"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_matrix.py",
            "--matrix",
            str(matrix),
            "--run-root",
            str(run_root),
            "--gpus",
            "0",
            "--min-free-gb",
            "0",
        ],
    )

    runner.main()

    status = json.loads((run_root / "batch_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "complete"
    assert status["pending"] == []
    assert status["failed"] == []
    assert status["skipped"][0]["reason"] == "already_complete_contract_validated"
    assert status["skipped"][0]["fingerprint_sha256"] == "current-fingerprint"


def test_run_matrix_contract_rejection_happens_before_any_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import sys
    import scripts.nightly.run_matrix as runner

    existing = _completed_contract()
    matrix, run_root = _write_single_completed_matrix(tmp_path, existing)
    expected = deepcopy(existing)
    expected["source_fingerprint"]["sha256"] = "changed-source"
    expected["fingerprint_sha256"] = "changed-fingerprint"
    monkeypatch.setattr(
        runner,
        "_expected_run_contract",
        lambda *_args, **_kwargs: deepcopy(expected),
    )
    monkeypatch.setattr(runner, "_disk_free_gb", lambda _path: 1000.0)
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("contract rejection must precede launch"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_matrix.py",
            "--matrix",
            str(matrix),
            "--run-root",
            str(run_root),
            "--gpus",
            "0",
            "--min-free-gb",
            "0",
        ],
    )

    with pytest.raises(ValueError, match="source"):
        runner.main()

    status = json.loads((run_root / "batch_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "contract_rejected"
    assert status["pending"] == []
    assert status["failed"][0]["reason"] == "completed_run_contract_rejected"


def test_parent_view_jobs_carry_reviewed_parent_contract() -> None:
    expected = "data/processed/entities/chemical_parent_normalized_views.tsv"
    for name, key in (("quick_screen.yaml", "experiments"),):
        matrix = _load(name)
        parent_jobs = [
            item
            for item in matrix[key]
            if item.get("overrides", {}).get("entity", {}).get("chemical_structure_view") == "parent"
        ]
        assert parent_jobs
        for item in parent_jobs:
            assert item["overrides"]["entity"]["chemical_parent_views"] == expected
    formal = _load("confirm_candidates.yaml")["confirm_candidates"]
    assert not any(
        item.get("overrides", {}).get("entity", {}).get("chemical_structure_view") == "parent"
        for item in formal
    )


def test_zero_risky_jobs_are_row_level_and_morgan_duplicate_is_removed() -> None:
    matrix = _load("quick_screen.yaml")
    ids = {item["id"] for item in matrix["experiments"]}
    assert "SCR-M8.0-MORGAN" not in ids
    zero_jobs = [
        item
        for item in matrix["experiments"]
        if "ZERO-RISKY" in str(item["id"])
    ]
    assert {item["id"] for item in zero_jobs} == {
        "SCR-M8.0-MORGAN-ZERO-RISKY",
        "SCR-M8.0-CHEMBERTA-ZERO-RISKY",
    }
    for item in zero_jobs:
        entity = item["overrides"]["entity"]
        assert entity["chemical_structure_view"] == "zero_risky"
        assert entity["chemical_structure_view"] != "zero"
        assert entity["chemical_parent_views"] is None
        assert entity["chemical_identity_risks"] == (
            "resources/entities/chemical_identity_risk_review.tsv"
        )


def test_scale_selection_uses_residual_and_high_effect_guardrails() -> None:
    rows: list[dict] = []
    metadata: dict[str, dict] = {}
    for axis in ("strain", "chemical", "pair"):
        scenarios = {
            "strain": ("R10",),
            "chemical": ("R01",),
            "pair": ("R11", "RT"),
        }[axis]
        residual = (
            "drug_residual_pcc_mean"
            if axis == "chemical"
            else "context_residual_pcc_mean"
        )
        for index, scale in enumerate(ALLOWED_SCALES):
            experiment_id = f"GRID-{axis}-{index}"
            metadata[experiment_id] = {
                "kind": "expert_scale_screen",
                "selection_axis": axis,
                "selection_value": scale,
            }
            fc = {0.0: 0.300, 0.25: 0.305, 0.5: 0.320, 0.75: 0.330, 1.0: 0.340}[scale]
            residual_value = 0.10 if scale == 0 else 0.10 + scale * 0.01
            high_effect = 0.50
            if scale == 0.75 and axis != "pair":
                high_effect = 0.49  # fails the -0.005 guardrail
            if scale == 1.0:
                residual_value = 0.09  # fails the residual guardrail
            for scenario in scenarios:
                # R11 alone would select 0.75, while the locked R11+RT macro
                # selects 0.5.  This catches post-result regime cherry-picking.
                scenario_fc = (
                    0.28
                    if axis == "pair" and scenario == "RT" and scale == 0.75
                    else fc
                )
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "scenario": scenario,
                        "model_seed": 42,
                        "fc_pcc_mean": scenario_fc,
                        "context_residual_pcc_mean": residual_value if axis in {"strain", "pair"} else float("nan"),
                        "drug_residual_pcc_mean": residual_value if axis in {"chemical", "pair"} else float("nan"),
                        "high_effect_pcc_mean": high_effect,
                        "high_effect_f1_mean": 0.20,
                    }
                )
    candidates, selection = _select_expert_scales(pd.DataFrame(rows), metadata)
    assert len(candidates) == 15
    assert selection["selection_status"] == "selected"
    assert selection["selected"]["strain"]["scale"] == 0.5
    assert selection["selected"]["chemical"]["scale"] == 0.5
    assert selection["selected"]["pair"]["scale"] == 0.5
    assert selection["selected"]["pair"]["scenario"] == "R11+RT"
    assert selection["selected"]["pair"]["guardrail_regimes"] == ["R11", "RT"]
    assert selection["selection_rule_id"] == "goai.expert_scale_selection.v2"

    incomplete_metadata = dict(metadata)
    incomplete_metadata.pop("GRID-strain-4")
    _, incomplete = _select_expert_scales(pd.DataFrame(rows), incomplete_metadata)
    assert incomplete["selection_status"] == "incomplete"
    assert incomplete["selected"]["strain"] is None


def test_summary_builds_four_regime_macro_and_fold_paired_deltas(tmp_path: Path) -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "run_root": str(tmp_path),
                "experiment_id": "CANDIDATE",
                "model_id": "M7.test",
                "kind": "model_screen",
                "model_seed": 42,
                "scenario": scenario,
                "fc_pcc_mean": 0.30 + index * 0.01,
                "context_residual_pcc_mean": 0.10,
                "drug_residual_pcc_mean": 0.11,
                "high_effect_pcc_mean": 0.50,
                "high_effect_f1_mean": 0.20,
                "absolute_sample_r2_median_mean": 0.90,
            }
            for index, scenario in enumerate(sorted(CORE - {"RT"}))
        ]
    )
    macro = _four_regime_macro(leaderboard)
    assert len(macro) == 1
    assert bool(macro.iloc[0]["complete_four_regimes"])
    assert macro.iloc[0]["four_regime_macro_fc_pcc"] == pytest.approx(0.315)

    assignment = pd.DataFrame(
        {
            "scenario": ["R00", "R00"],
            "fold": [0, 1],
            "sample_ID": ["A", "B"],
            "eligible": [True, True],
            "exclusion_reason": ["", ""],
        }
    )
    for experiment in ("CONTROL", "CANDIDATE"):
        run = tmp_path / "producers" / experiment / "S42"
        run.mkdir(parents=True)
        pd.DataFrame({"scenario": ["R00"], "fc_pcc_mean": [0.3]}).to_csv(
            run / "oof_summary.csv", index=False
        )
        assignment.to_csv(run / "fold_assignments.csv", index=False)
    control_metrics = pd.DataFrame(
        {"scenario": ["R00", "R00"], "fold": [0, 1], "fc_pcc": [0.30, 0.32]}
    )
    candidate_metrics = pd.DataFrame(
        {"scenario": ["R00", "R00"], "fold": [0, 1], "fc_pcc": [0.32, 0.34]}
    )
    control_metrics.to_csv(
        tmp_path / "producers" / "CONTROL" / "S42" / "oof_official_proxy_metrics.csv",
        index=False,
    )
    candidate_metrics.to_csv(
        tmp_path / "producers" / "CANDIDATE" / "S42" / "oof_official_proxy_metrics.csv",
        index=False,
    )
    folds, paired = _paired_deltas([tmp_path], "CONTROL")
    assert len(folds) == 2
    assert len(paired) == 1
    assert paired.iloc[0]["mean_delta"] == pytest.approx(0.02)
    assert set(paired["protocol_label"]) == {PROTOCOL_LABEL}
