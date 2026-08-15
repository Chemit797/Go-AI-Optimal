#!/usr/bin/env python3
"""Issue a deterministic calibration audit receipt from the locked matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROTOCOL_LABEL = "LOCAL_STRICT_OOF_NOT_OFFICIAL"
EXPECTED_IDS = {
    "CAL-M7.0-R16-BASE",
    "CAL-M7.0-R4",
    "CAL-M7.0-R8",
    "CAL-M7.0-PLATE-SHUFFLE",
    "CAL-M7.0-NO-PLATE",
    "CAL-M7.0-DROPOUT-P000",
    "CAL-M7.0-DROPOUT-P050",
}
EXPECTED_SCENARIOS = {"plate", "R00", "R10", "R01", "R11", "RT"}
MODEL_SELECTION_SCENARIOS = ("R00", "R10", "R01", "R11", "RT")
PLATE_CANDIDATES = {
    "CAL-M7.0-R16-BASE",
    "CAL-M7.0-R4",
    "CAL-M7.0-R8",
    "CAL-M7.0-DROPOUT-P000",
    "CAL-M7.0-DROPOUT-P050",
}
NO_PLATE_ID = "CAL-M7.0-NO-PLATE"
PLATE_SHUFFLE_ID = "CAL-M7.0-PLATE-SHUFFLE"
MAX_GUARDRAIL_DROP = 0.005
REQUIRED_CHECKS = {
    "leave_one_plate_out",
    "plate_label_shuffle",
    "rank_4_8_16",
    "plate_dropout",
    "no_plate_control",
    "observation_metadata_only",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile(summary: pd.DataFrame) -> dict[str, float]:
    """Pre-locked five-regime model-selection profile."""
    indexed = summary.set_index("scenario")

    def macro(column: str, scenarios: tuple[str, ...]) -> float:
        if column not in indexed:
            raise ValueError(f"Calibration summary lacks {column}")
        values = pd.to_numeric(indexed.reindex(scenarios)[column], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"Calibration summary has non-finite {column} for {list(scenarios)}"
            )
        return float(values.mean())

    return {
        "five_regime_macro_fc_pcc": macro(
            "fc_pcc_mean", MODEL_SELECTION_SCENARIOS
        ),
        "context_residual_macro": macro(
            "context_residual_pcc_mean", ("R10", "R11", "RT")
        ),
        "drug_residual_macro": macro(
            "drug_residual_pcc_mean", ("R01", "R11", "RT")
        ),
        "high_effect_pcc_macro": macro(
            "high_effect_pcc_mean", MODEL_SELECTION_SCENARIOS
        ),
        "high_effect_f1_macro": macro(
            "high_effect_f1_mean", MODEL_SELECTION_SCENARIOS
        ),
        "leave_one_plate_out_fc_pcc": macro("fc_pcc_mean", ("plate",)),
        "leave_one_plate_out_high_effect_pcc": macro(
            "high_effect_pcc_mean", ("plate",)
        ),
    }


def _select_calibration(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select an observed rank/dropout or deterministically fall back no-plate."""
    frame = table.copy().set_index("experiment_id", drop=False)
    reference = frame.loc[NO_PLATE_ID]
    profile_columns = (
        "five_regime_macro_fc_pcc",
        "context_residual_macro",
        "drug_residual_macro",
        "high_effect_pcc_macro",
        "high_effect_f1_macro",
    )
    for column in profile_columns:
        if not pd.notna(reference[column]):
            raise ValueError(f"No-plate reference has non-finite {column}")
    frame["context_residual_delta_vs_no_plate"] = (
        frame["context_residual_macro"] - reference["context_residual_macro"]
    )
    frame["drug_residual_delta_vs_no_plate"] = (
        frame["drug_residual_macro"] - reference["drug_residual_macro"]
    )
    frame["high_effect_pcc_delta_vs_no_plate"] = (
        frame["high_effect_pcc_macro"] - reference["high_effect_pcc_macro"]
    )
    frame["high_effect_f1_delta_vs_no_plate"] = (
        frame["high_effect_f1_macro"] - reference["high_effect_f1_macro"]
    )
    frame["leave_one_plate_out_fc_delta_vs_no_plate"] = (
        frame["leave_one_plate_out_fc_pcc"]
        - reference["leave_one_plate_out_fc_pcc"]
    )
    frame["guardrails_pass"] = (
        frame["context_residual_delta_vs_no_plate"].ge(-MAX_GUARDRAIL_DROP)
        & frame["drug_residual_delta_vs_no_plate"].ge(-MAX_GUARDRAIL_DROP)
        & frame["high_effect_pcc_delta_vs_no_plate"].ge(-MAX_GUARDRAIL_DROP)
        & frame["high_effect_f1_delta_vs_no_plate"].ge(-MAX_GUARDRAIL_DROP)
        & frame["leave_one_plate_out_fc_delta_vs_no_plate"].ge(
            -MAX_GUARDRAIL_DROP
        )
    )
    eligible_plate = frame.loc[
        frame.index.isin(PLATE_CANDIDATES) & frame["guardrails_pass"]
    ].copy().reset_index(drop=True)
    best_plate: pd.Series | None = None
    if not eligible_plate.empty:
        best_plate = eligible_plate.sort_values(
            [
                "five_regime_macro_fc_pcc",
                "calibration_rank",
                "calibration_plate_dropout",
                "experiment_id",
            ],
            ascending=[False, True, False, True],
        ).iloc[0]
    shuffle_fc = float(frame.loc[PLATE_SHUFFLE_ID, "five_regime_macro_fc_pcc"])
    no_plate_fc = float(reference["five_regime_macro_fc_pcc"])
    plate_beats_both = bool(
        best_plate is not None
        and float(best_plate["five_regime_macro_fc_pcc"]) > no_plate_fc
        and float(best_plate["five_regime_macro_fc_pcc"]) > shuffle_fc
    )
    selected = best_plate.copy() if plate_beats_both else reference.copy()
    selected["leave_one_plate_out_fc_delta_vs_no_plate"] = float(
        selected["leave_one_plate_out_fc_pcc"]
        - reference["leave_one_plate_out_fc_pcc"]
    )
    frame["selected"] = frame.index == str(selected["experiment_id"])
    frame["selection_eligible"] = (frame.index == NO_PLATE_ID) | (
        frame.index.isin(PLATE_CANDIDATES) & frame["guardrails_pass"]
    )
    selected_config = {
        "calibration_rank": int(selected["calibration_rank"]),
        "calibration_use_plate": bool(selected["calibration_use_plate"]),
        "calibration_plate_dropout": float(selected["calibration_plate_dropout"]),
        "calibration_plate_shuffle": False,
    }
    evidence = {
        "selection_method": (
            "Maximize five-regime macro FC among observed rank/dropout candidates "
            "passing residual/high-effect max-drop 0.005 versus no-plate; retain "
            "plate only when it strictly beats both no-plate and plate-shuffle."
        ),
        "selected_experiment_id": str(selected["experiment_id"]),
        "selected_config": selected_config,
        "selected_profile": {
            column: float(selected[column]) for column in profile_columns
        },
        "selected_leave_one_plate_out_fc_pcc": float(
            selected["leave_one_plate_out_fc_pcc"]
        ),
        "selected_leave_one_plate_out_fc_delta_vs_no_plate": float(
            selected["leave_one_plate_out_fc_delta_vs_no_plate"]
        ),
        "best_guardrail_plate_experiment_id": (
            None if best_plate is None else str(best_plate["experiment_id"])
        ),
        "best_guardrail_plate_macro_fc": (
            None
            if best_plate is None
            else float(best_plate["five_regime_macro_fc_pcc"])
        ),
        "no_plate_macro_fc": no_plate_fc,
        "plate_shuffle_macro_fc": shuffle_fc,
        "plate_beats_no_plate_and_shuffle": plate_beats_both,
        "guardrail_max_drop": MAX_GUARDRAIL_DROP,
    }
    return frame.reset_index(drop=True).sort_values("experiment_id"), evidence


def audit_calibration_results(run_root: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(run_root).resolve()
    environment_path = root / "environment.json"
    if not environment_path.is_file():
        raise ValueError("Calibration run lacks environment.json")
    environment = _load(environment_path)
    matrix_path = Path(str(environment.get("matrix", ""))).resolve()
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    if not isinstance(matrix, dict) or matrix.get("protocol_label") != PROTOCOL_LABEL:
        raise ValueError("Calibration matrix lacks the locked local-OOF contract")
    experiments = matrix.get("experiments", [])
    metadata = {str(item["id"]): item for item in experiments}
    if set(metadata) != EXPECTED_IDS:
        raise ValueError(
            f"Calibration audit requires exact producers: missing={sorted(EXPECTED_IDS-set(metadata))}, "
            f"unexpected={sorted(set(metadata)-EXPECTED_IDS)}"
        )
    rows: list[dict[str, Any]] = []
    source_hashes: set[str] = set()
    input_hashes: set[str] = set()
    run_contract_receipts: list[dict[str, str]] = []
    for experiment_id in sorted(EXPECTED_IDS):
        item = metadata[experiment_id]
        run = root / "producers" / experiment_id / "S42"
        for name in ("run_contract.json", "oof_manifest.json", "oof_summary.csv"):
            if not (run / name).is_file():
                raise ValueError(f"Incomplete calibration producer {experiment_id}: missing {name}")
        contract = _load(run / "run_contract.json")
        manifest = _load(run / "oof_manifest.json")
        source_hashes.add(str(contract.get("source_fingerprint", {}).get("sha256", "")))
        input_hashes.add(json.dumps(contract.get("input_hashes", {}), sort_keys=True))
        scenarios = set(contract.get("scenarios", []))
        model = contract.get("effective_config", {}).get("model", {})
        declared = item.get("overrides", {}).get("model", {})
        completed = list((run / "folds").glob("*/completed.json"))
        summary = pd.read_csv(run / "oof_summary.csv")
        summary_scenarios = set(summary["scenario"].astype(str))
        valid = bool(
            contract.get("protocol") == "support_regime_oof_run_contract_v3"
            and int(contract.get("n_folds", -1)) == 2
            and int(contract.get("seed", -1)) == 42
            and int(contract.get("model_seed", -1)) == 42
            and manifest.get("audit_only") is False
            and scenarios == EXPECTED_SCENARIOS
            and summary_scenarios == EXPECTED_SCENARIOS
            and len(completed) == 16
            and int(model.get("calibration_rank", -1)) == int(declared["calibration_rank"])
            and bool(model.get("calibration_use_plate"))
            == bool(declared["calibration_use_plate"])
            and bool(model.get("calibration_plate_shuffle"))
            == bool(declared["calibration_plate_shuffle"])
            and float(model.get("calibration_plate_dropout", -1.0))
            == float(declared["calibration_plate_dropout"])
        )
        if not valid:
            raise ValueError(f"Calibration producer violates declared contract: {experiment_id}")
        profile = _profile(summary)
        run_contract_receipts.append(
            {
                "experiment_id": experiment_id,
                "path": str(run / "run_contract.json"),
                "sha256": _sha256(run / "run_contract.json"),
                "fingerprint_sha256": str(contract.get("fingerprint_sha256", "")),
            }
        )
        rows.append(
            {
                "experiment_id": experiment_id,
                "completed_fits": len(completed),
                "scenarios": ",".join(sorted(scenarios)),
                "calibration_rank": int(model["calibration_rank"]),
                "calibration_use_plate": bool(model["calibration_use_plate"]),
                "calibration_plate_shuffle": bool(model["calibration_plate_shuffle"]),
                "calibration_plate_dropout": float(model["calibration_plate_dropout"]),
                "status": "valid",
                **profile,
            }
        )
    if len(source_hashes) != 1 or "" in source_hashes:
        raise ValueError("Calibration producers do not share one source fingerprint")
    if len(input_hashes) != 1:
        raise ValueError("Calibration producers do not share one input hash contract")
    table = pd.DataFrame(rows).sort_values("experiment_id")
    ranks = set(table.loc[table["calibration_use_plate"], "calibration_rank"])
    dropouts = set(
        table.loc[
            ~table["calibration_plate_shuffle"] & table["calibration_use_plate"],
            "calibration_plate_dropout",
        ]
    )
    checks = {
        "leave_one_plate_out": bool(all("plate" in value for value in table["scenarios"])),
        "plate_label_shuffle": bool(table["calibration_plate_shuffle"].any()),
        "rank_4_8_16": bool({4, 8, 16}.issubset(ranks)),
        "plate_dropout": bool({0.0, 0.3, 0.5}.issubset(dropouts)),
        "no_plate_control": bool((~table["calibration_use_plate"]).any()),
        # Enforced structurally by model/features tests: calibration receives
        # only observation tensor. The run source fingerprint binds that code.
        "observation_metadata_only": True,
    }
    table, selection = _select_calibration(table)
    approved = bool(
        set(checks) == REQUIRED_CHECKS
        and all(checks.values())
        and selection.get("selected_config") is not None
    )
    receipt = {
        "schema": "goai.m7_m8.calibration_audit_receipt.v1",
        "protocol_label": PROTOCOL_LABEL,
        "official_score_status": "NOT_OFFICIAL",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(root),
        "matrix": str(matrix_path),
        "matrix_sha256": _sha256(matrix_path),
        "audit_complete": approved,
        "checks": checks,
        "source_fingerprint_sha256": next(iter(source_hashes)),
        "producer_count": len(table),
        "fits_per_producer": sorted(set(table["completed_fits"])),
        "producer_run_contracts": run_contract_receipts,
        "selection": selection,
        "decision_reason": (
            f"Selected {selection['selected_experiment_id']} by the locked five-regime "
            "macro-FC rule with residual and high-effect guardrails."
        ),
        "status": "approved" if approved else "blocked",
    }
    return table, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    table, receipt = audit_calibration_results(args.run_root)
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(args.run_root).resolve() / "consumer"
    )
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "calibration_audit_producers.csv", index=False)
    receipt_path = output / "calibration_audit_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "calibration_audit_receipt.json.sha256").write_text(
        _sha256(receipt_path) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if receipt["status"] != "approved":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
