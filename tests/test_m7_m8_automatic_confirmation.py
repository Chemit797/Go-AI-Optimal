from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

import scripts.nightly.audit_identity_for_confirmation as identity_gate_module
from scripts.nightly.confirmation_evidence import PROTOCOL_LABEL, sha256_file
from scripts.nightly.prepare_m7_confirmation import (
    FAIR_NOMINATIONS,
    nominate_m7_candidates,
    prepare_confirmation,
)
from scripts.nightly.run_m7_promotion_batch import run_promotion_batch


PROJECT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _identity(tmp_path: Path) -> Path:
    artifacts = {}
    for entity in ("chemical", "strain"):
        registry = tmp_path / f"{entity}.tsv"
        registry.write_text("entity\nplaceholder\n", encoding="utf-8")
        artifacts[entity] = {
            "path": str(registry),
            "sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
        }
    return _write_json(
        tmp_path / "identity.json",
        {
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
            "promotion_blockers": [{"reason": "verified coverage is zero"}],
            "issues": [
                {
                    "severity": "error",
                    "code": "test_only_promotion_blocked",
                    "message": "formal semantics are intentionally blocked",
                }
            ],
            "registry_artifacts": artifacts,
        },
    )


def _calibration(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "calibration.json",
        {
            "schema": "goai.m7_m8.calibration_audit_receipt.v1",
            "protocol_label": PROTOCOL_LABEL,
            "status": "approved",
            "audit_complete": True,
            "decision_reason": "locked deterministic selection",
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
        },
    )


def _fair_audit(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "fair.json",
        {
            "schema": "fair_expert_receipt_audit_v1",
            "status": "valid",
            "source_fingerprints_are_identical": True,
            "source_fingerprint_sha256": "source",
            "invalid_receipt_rows": 0,
            "task_fit_counts": {f"task-{index}": 14 for index in range(7)},
        },
    )


def _selection() -> dict:
    return {
        "selection_status": "selected",
        "protocol_label": PROTOCOL_LABEL,
        "selection_rule_id": "goai.expert_scale_selection.v2",
        "method": "locked synthetic grid",
        "pair_selection_rule": {"regimes": ["R11", "RT"]},
        "selected": {
            "strain": {
                "scale": 0.5,
                "experiment_id": "SCALE-M7.3-STRAIN-P050",
            },
            "chemical": {
                "scale": 0.75,
                "experiment_id": "SCALE-M7.3-CHEMICAL-P075",
            },
            "pair": {
                "scale": 0.0,
                "experiment_id": "SCALE-M7.4-PAIR-P000",
                "scenario": "R11+RT",
                "guardrail_regimes": ["R11", "RT"],
                "fc_delta_vs_zero": 0.0,
                "relevant_residual_delta_vs_zero": 0.0,
                "high_effect_pcc_delta_vs_zero": 0.0,
            },
        },
    }


def _scale_grid(path: Path) -> Path:
    rows = []
    selected_ids = {
        ("strain", 0.5): "SCALE-M7.3-STRAIN-P050",
        ("chemical", 0.75): "SCALE-M7.3-CHEMICAL-P075",
        ("pair", 0.0): "SCALE-M7.4-PAIR-P000",
    }
    for axis in ("strain", "chemical", "pair"):
        for index, scale in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
            experiment_id = selected_ids.get(
                (axis, scale), f"SCALE-{axis.upper()}-{index}"
            )
            rows.append(
                {
                    "axis": axis,
                    "scale": scale,
                    "experiment_id": experiment_id,
                    "complete": True,
                    "metrics_finite": True,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _fair_paired_frame() -> pd.DataFrame:
    rows = []
    for rule in FAIR_NOMINATIONS.values():
        for scenario, residual in rule["regimes"].items():
            for metric in ("fc_pcc", residual, "high_effect_pcc", "high_effect_f1"):
                mean = 0.006 if metric == "fc_pcc" else 0.001
                rows.append(
                    {
                        "candidate": rule["source"],
                        "control": "FAIR-M7.0-U9",
                        "model_seed": 42,
                        "scenario": scenario,
                        "metric": metric,
                        "n_paired_folds": 2,
                        "mean_delta": mean,
                        "ci_low": 0.0005,
                        "ci_high": 0.01,
                    }
                )
    return pd.DataFrame(rows).drop_duplicates(
        ["candidate", "control", "model_seed", "scenario", "metric"]
    )


def test_discovery_can_nominate_but_never_promote_and_pair_zero_is_skipped() -> None:
    nominated, decisions = nominate_m7_candidates(
        _fair_paired_frame(), _selection()
    )
    assert set(nominated) == set(FAIR_NOMINATIONS)
    assert "CONF-M7.4-PAIR" not in nominated
    assert all(item["promotion_eligible_from_discovery"] is False for item in decisions)
    assert decisions[-1]["decision"] == "screen_only"


def test_nonfinite_fair_metric_screens_one_candidate_without_failing_others() -> None:
    frame = _fair_paired_frame()
    source = FAIR_NOMINATIONS["CONF-M7.1-STRAIN"]["source"]
    mask = frame["candidate"].eq(source) & frame["metric"].eq("high_effect_f1")
    frame.loc[mask, ["mean_delta", "ci_low", "ci_high"]] = float("nan")
    nominated, decisions = nominate_m7_candidates(frame, _selection())
    assert "CONF-M7.1-STRAIN" not in nominated
    assert "CONF-M7.2-CHEMICAL" in nominated
    decision = next(
        item for item in decisions if item["candidate"] == "CONF-M7.1-STRAIN"
    )
    assert decision["decision"] == "screen_only"


def test_identity_wrapper_accepts_semantic_block_as_valid_m7_outcome(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_run(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text("{}", encoding="utf-8")
        return type("Result", (), {"returncode": 1, "stdout": "semantic blocked\n"})()

    monkeypatch.setattr(identity_gate_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        identity_gate_module,
        "validate_identity_audit",
        lambda path: {
            "semantic_promotion_status": "blocked",
            "core_registry_integrity": "valid",
        },
    )
    receipt = identity_gate_module.audit_identity(
        output_dir=tmp_path / "identity", python="python-goai"
    )
    assert receipt["raw_audit_returncode"] == 1
    assert receipt["m7_confirmation_allowed"] is True
    assert receipt["m8_confirmation_status"] == "blocked"
    path = tmp_path / "identity" / "identity_gate_receipt.json"
    assert Path(str(path) + ".sha256").read_text(encoding="utf-8").strip() == sha256_file(path)


def test_preparation_filters_to_selected_m7_controls_and_receipts_m8_block(
    tmp_path: Path,
) -> None:
    selection_path = tmp_path / "selection.yaml"
    selection_path.write_text(yaml.safe_dump(_selection()), encoding="utf-8")
    grid = _scale_grid(tmp_path / "scale_grid.csv")
    paired = tmp_path / "paired.csv"
    _fair_paired_frame().to_csv(paired, index=False)
    summary_dirs = {}
    for name in ("quick", "research"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "local_oof_report_zh.md").write_text("local only\n", encoding="utf-8")
        (directory / "regime_summary.csv").write_text("scenario,fc_pcc\nR00,0.1\n", encoding="utf-8")
        summary_dirs[name] = directory
    output = tmp_path / "preparation"
    matrix_path = tmp_path / "confirm" / "confirm_matrix.yaml"
    receipt = prepare_confirmation(
        template_path=PROJECT
        / "configs/nightly/20260813-m7-m8/confirm_candidates.yaml",
        scale_selection_path=selection_path,
        scale_candidates_path=grid,
        identity_audit_path=_identity(tmp_path),
        calibration_audit_path=_calibration(tmp_path),
        fair_expert_audit_path=_fair_audit(tmp_path),
        fair_paired_summary_path=paired,
        quick_summary_dir=summary_dirs["quick"],
        research_summary_dir=summary_dirs["research"],
        output_matrix_path=matrix_path,
        output_dir=output,
    )
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    ids = {item["id"] for item in matrix["experiments"]}
    assert ids == set(FAIR_NOMINATIONS) | {
        "CONF-M7.0-GENERAL",
        "CONF-M7.0-GENERAL-U96",
    }
    assert not any(str(item.get("model_id", "")).startswith("M8") for item in matrix["experiments"])
    m8 = json.loads((output / "m8_blocked_receipt.json").read_text(encoding="utf-8"))
    assert m8["status"] == "blocked"
    assert m8["gpu_tasks_started"] is False
    assert receipt["discovery_results_can_promote"] is False
    assert receipt["matrix"]["sha256"] == sha256_file(matrix_path)


def test_promotion_batch_treats_statistical_block_as_complete_and_resumes(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "confirm_matrix.yaml"
    matrix = {
        "protocol_label": PROTOCOL_LABEL,
        "experiments": [
            {
                "id": "CONF-M7.0-GENERAL",
                "kind": "model_confirm",
                "promotion_eligible": False,
            },
            {
                "id": "CONF-M7.1-STRAIN",
                "kind": "model_confirm",
                "model_id": "M7.1",
                "primary_control": "CONF-M7.0-GENERAL",
                "promotion_regimes": ["R10"],
                "required_negative_controls": [],
            },
        ],
    }
    matrix_path.write_text(yaml.safe_dump(matrix), encoding="utf-8")
    root = tmp_path / "run"
    root.mkdir()
    _write_json(
        root / "environment.json",
        {"matrix": str(matrix_path), "matrix_sha256": sha256_file(matrix_path)},
    )
    selection_path = _write_json(
        tmp_path / "selection.json",
        {
            "schema": "goai.m7_m8.confirmation_selection_receipt.v1",
            "protocol_label": PROTOCOL_LABEL,
            "status": "prepared",
            "promotion_candidates": ["CONF-M7.1-STRAIN"],
            "matrix": {"path": str(matrix_path), "sha256": sha256_file(matrix_path)},
        },
    )
    calls = []

    def gate_runner(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "schema": "goai.m7_m8.promotion_receipt.v1",
            "protocol_label": PROTOCOL_LABEL,
            "candidate": "CONF-M7.1-STRAIN",
            "status": "blocked",
            "promoted": False,
            "confirmation_contract": {"matrix_sha256": sha256_file(matrix_path)},
        }

    output = tmp_path / "promotion"
    first = run_promotion_batch(
        run_root=root,
        selection_receipt_path=selection_path,
        output_dir=output,
        bootstrap_draws=50,
        gate_runner=gate_runner,
    )
    assert first["status"] == "complete"
    assert first["blocked_count"] == 1
    assert len(calls) == 1
    second = run_promotion_batch(
        run_root=root,
        selection_receipt_path=selection_path,
        output_dir=output,
        bootstrap_draws=50,
        gate_runner=gate_runner,
    )
    assert second["results"][0]["resumed"] is True
    assert len(calls) == 1
