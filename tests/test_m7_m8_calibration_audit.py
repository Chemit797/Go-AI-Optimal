from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.nightly.audit_calibration_results import (
    EXPECTED_IDS,
    EXPECTED_SCENARIOS,
    PROTOCOL_LABEL,
    audit_calibration_results,
)


PROJECT = Path(__file__).resolve().parents[1]
MATRIX = PROJECT / "configs/nightly/20260813-m7-m8/calibration_audit.yaml"


def _fixture(
    root: Path,
    fc_scores: dict[str, float] | None = None,
    plate_fc_scores: dict[str, float] | None = None,
) -> None:
    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    copied_matrix = root / "calibration.yaml"
    copied_matrix.write_text(yaml.safe_dump(matrix), encoding="utf-8")
    (root / "environment.json").write_text(
        json.dumps({"matrix": str(copied_matrix)}), encoding="utf-8"
    )
    metadata = {item["id"]: item for item in matrix["experiments"]}
    default_scores = {
        "CAL-M7.0-R16-BASE": 0.305,
        "CAL-M7.0-R4": 0.306,
        "CAL-M7.0-R8": 0.320,
        "CAL-M7.0-PLATE-SHUFFLE": 0.290,
        "CAL-M7.0-NO-PLATE": 0.300,
        "CAL-M7.0-DROPOUT-P000": 0.304,
        "CAL-M7.0-DROPOUT-P050": 0.310,
    }
    default_scores.update(fc_scores or {})
    default_plate_scores = dict(default_scores)
    default_plate_scores.update(plate_fc_scores or {})
    for experiment_id in EXPECTED_IDS:
        run = root / "producers" / experiment_id / "S42"
        (run / "folds").mkdir(parents=True)
        model = metadata[experiment_id]["overrides"]["model"]
        contract = {
            "protocol": "support_regime_oof_run_contract_v3",
            "n_folds": 2,
            "seed": 42,
            "model_seed": 42,
            "scenarios": sorted(EXPECTED_SCENARIOS),
            "source_fingerprint": {"sha256": "same-source"},
            "input_hashes": {"metadata": "same-input"},
            "effective_config": {"model": model},
        }
        (run / "run_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        (run / "oof_manifest.json").write_text(
            json.dumps({"audit_only": False}), encoding="utf-8"
        )
        scenarios = sorted(EXPECTED_SCENARIOS)
        pd.DataFrame(
            {
                "scenario": scenarios,
                "fc_pcc_mean": [
                    default_plate_scores[experiment_id]
                    if scenario == "plate"
                    else default_scores[experiment_id]
                    for scenario in scenarios
                ],
                "context_residual_pcc_mean": [0.20] * 6,
                "drug_residual_pcc_mean": [0.21] * 6,
                "high_effect_pcc_mean": [0.50] * 6,
                "high_effect_f1_mean": [0.20] * 6,
            }
        ).to_csv(
            run / "oof_summary.csv", index=False
        )
        for index in range(16):
            completed = run / "folds" / f"fit-{index}" / "completed.json"
            completed.parent.mkdir()
            completed.write_text("{}", encoding="utf-8")


def test_complete_seven_by_sixteen_calibration_matrix_is_approved(tmp_path: Path) -> None:
    _fixture(tmp_path)
    table, receipt = audit_calibration_results(tmp_path)
    assert len(table) == 7
    assert set(table["completed_fits"]) == {16}
    assert receipt["status"] == "approved"
    assert receipt["audit_complete"] is True
    assert all(receipt["checks"].values())
    assert receipt["selection"]["selected_experiment_id"] == "CAL-M7.0-R8"
    assert receipt["selection"]["selected_config"] == {
        "calibration_rank": 8,
        "calibration_use_plate": True,
        "calibration_plate_dropout": 0.3,
        "calibration_plate_shuffle": False,
    }


def test_shuffle_winning_forces_no_plate_instead_of_retaining_confounding(
    tmp_path: Path,
) -> None:
    _fixture(
        tmp_path,
        {
            "CAL-M7.0-R8": 0.33,
            "CAL-M7.0-PLATE-SHUFFLE": 0.34,
            "CAL-M7.0-NO-PLATE": 0.30,
        },
    )
    _, receipt = audit_calibration_results(tmp_path)
    assert receipt["selection"]["selected_experiment_id"] == "CAL-M7.0-NO-PLATE"
    assert receipt["selection"]["selected_config"]["calibration_use_plate"] is False
    assert receipt["selection"]["plate_beats_no_plate_and_shuffle"] is False


def test_dropout_variant_can_win_when_rank_candidates_are_weaker(tmp_path: Path) -> None:
    _fixture(
        tmp_path,
        {
            "CAL-M7.0-DROPOUT-P050": 0.35,
            "CAL-M7.0-PLATE-SHUFFLE": 0.29,
            "CAL-M7.0-NO-PLATE": 0.30,
        },
    )
    _, receipt = audit_calibration_results(tmp_path)
    selected = receipt["selection"]
    assert selected["selected_experiment_id"] == "CAL-M7.0-DROPOUT-P050"
    assert selected["selected_config"]["calibration_rank"] == 16
    assert selected["selected_config"]["calibration_plate_dropout"] == 0.5


def test_large_leave_one_plate_out_drop_forces_no_plate(tmp_path: Path) -> None:
    _fixture(
        tmp_path,
        {
            "CAL-M7.0-R8": 0.36,
            "CAL-M7.0-NO-PLATE": 0.30,
            "CAL-M7.0-PLATE-SHUFFLE": 0.29,
        },
        {
            **{
                experiment_id: 0.20
                for experiment_id in EXPECTED_IDS
                if experiment_id not in {
                    "CAL-M7.0-NO-PLATE",
                    "CAL-M7.0-PLATE-SHUFFLE",
                }
            },
            "CAL-M7.0-NO-PLATE": 0.30,
            "CAL-M7.0-PLATE-SHUFFLE": 0.29,
        },
    )
    table, receipt = audit_calibration_results(tmp_path)
    assert receipt["selection"]["selected_experiment_id"] == "CAL-M7.0-NO-PLATE"
    r8 = table.set_index("experiment_id").loc["CAL-M7.0-R8"]
    assert r8["leave_one_plate_out_fc_delta_vs_no_plate"] == pytest.approx(-0.10)
    assert bool(r8["guardrails_pass"]) is False


def test_missing_fit_blocks_calibration_receipt(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = (
        tmp_path
        / "producers/CAL-M7.0-R4/S42/folds/fit-15/completed.json"
    )
    path.unlink()
    with pytest.raises(ValueError, match="declared contract"):
        audit_calibration_results(tmp_path)


def test_source_mismatch_blocks_calibration_receipt(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "producers/CAL-M7.0-R8/S42/run_contract.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source_fingerprint"]["sha256"] = "different"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source fingerprint"):
        audit_calibration_results(tmp_path)
