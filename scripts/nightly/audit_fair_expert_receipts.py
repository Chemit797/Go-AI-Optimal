"""Audit fold-matched M7 universal-parent receipts before comparing scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


EXPECTED_PRODUCERS = {
    "FAIR-M7.0-U9",
    "FAIR-M7.1-U9-S2-FROZEN",
    "FAIR-M7.1-U9-S2-J3",
    "FAIR-M7.2-U9-C2-FROZEN",
    "FAIR-M7.2-U9-C2-J3",
    "FAIR-M7.3-U9-S2-C2-FROZEN",
    "FAIR-M7.3-U9-S2-C2-J3",
}


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fold_receipts(run: Path) -> dict[tuple[str, int], tuple[Path, dict]]:
    result: dict[tuple[str, int], tuple[Path, dict]] = {}
    for path in sorted((run / "folds").glob("*/completed.json")):
        payload = _load(path)
        key = (str(payload.get("scenario", "")), int(payload.get("fold", -1)))
        if not key[0] or key[1] < 0:
            raise ValueError(f"Fold completion lacks scenario/fold: {path}")
        if key in result:
            raise ValueError(f"Duplicate fold receipt {key} in {run}")
        receipt = payload.get("training_receipt")
        if not isinstance(receipt, dict):
            raise ValueError(f"Fold completion lacks training_receipt: {path}")
        result[key] = (path, receipt)
    return result


def audit_receipts(
    run_root: str | Path,
    *,
    control_id: str = "FAIR-M7.0-U9",
    expected_fits_per_task: int = 14,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Verify source and universal-parent identity for every fair producer."""
    root = Path(run_root).resolve()
    producers = root / "producers"
    candidates = sorted(path for path in producers.glob("FAIR-*") if path.is_dir())
    if not candidates:
        raise ValueError(f"No FAIR producers found under {producers}")
    actual_producers = {path.name for path in candidates}
    if actual_producers != EXPECTED_PRODUCERS:
        raise ValueError(
            "Fair receipt audit requires the complete seven-producer matrix: "
            f"missing={sorted(EXPECTED_PRODUCERS - actual_producers)}, "
            f"unexpected={sorted(actual_producers - EXPECTED_PRODUCERS)}"
        )
    control_roots = sorted((producers / control_id).glob("S*"))
    if not control_roots:
        raise ValueError(f"Missing control producer: {control_id}")

    rows: list[dict[str, object]] = []
    source_hashes: set[str] = set()
    task_counts: dict[str, int] = {}
    for control_run in control_roots:
        seed = control_run.name
        control_contract_path = control_run / "run_contract.json"
        control_contract = _load(control_contract_path)
        control_source = str(control_contract.get("source_fingerprint", {}).get("sha256", ""))
        if not control_source:
            raise ValueError(f"Control source fingerprint is missing: {control_contract_path}")
        control_receipts = _fold_receipts(control_run)
        if len(control_receipts) != expected_fits_per_task:
            raise ValueError(
                f"{control_id}/{seed} has {len(control_receipts)} folds; "
                f"expected {expected_fits_per_task}"
            )
        source_hashes.add(control_source)
        task_counts[f"{control_id}/{seed}"] = len(control_receipts)

        for producer_root in candidates:
            candidate_run = producer_root / seed
            if not candidate_run.is_dir():
                raise ValueError(f"Missing seed-matched producer: {candidate_run}")
            contract_path = candidate_run / "run_contract.json"
            contract = _load(contract_path)
            source = str(contract.get("source_fingerprint", {}).get("sha256", ""))
            source_hashes.add(source)
            receipts = _fold_receipts(candidate_run)
            task_key = f"{producer_root.name}/{seed}"
            task_counts[task_key] = len(receipts)
            if len(receipts) != expected_fits_per_task:
                raise ValueError(
                    f"{task_key} has {len(receipts)} folds; expected {expected_fits_per_task}"
                )
            if set(receipts) != set(control_receipts):
                raise ValueError(f"Fold keys differ between {task_key} and the control")

            frozen_candidate = producer_root.name.endswith("-FROZEN")
            for (scenario, fold), (path, receipt) in receipts.items():
                control_path, parent_receipt = control_receipts[(scenario, fold)]
                parent = str(parent_receipt.get("universal_state_sha256", ""))
                candidate_parent = str(receipt.get("universal_state_sha256", ""))
                copied = str(receipt.get("copied_universal_state_sha256", ""))
                post_frozen = str(
                    receipt.get("post_frozen_expert_universal_state_sha256", "")
                )
                final = str(receipt.get("final_universal_state_sha256", ""))
                is_control = producer_root.name == control_id
                checks = {
                    "source_matches_control": bool(source == control_source),
                    "parent_matches_control": bool(candidate_parent == parent),
                    "copy_matches_parent": bool(copied == parent),
                    "frozen_state_matches_parent": bool(post_frozen == parent),
                    "frozen_final_matches_parent": bool(
                        (not frozen_candidate) or final == parent
                    ),
                    "receipt_declares_frozen_unchanged": bool(
                        receipt.get("common_state_unchanged_during_frozen_experts") is True
                    ),
                }
                valid = all(checks.values()) and bool(parent)
                rows.append(
                    {
                        "producer": producer_root.name,
                        "seed": seed,
                        "scenario": scenario,
                        "fold": fold,
                        "is_control": is_control,
                        "is_frozen_candidate": frozen_candidate,
                        "source_fingerprint_sha256": source,
                        "run_contract_fingerprint_sha256": str(
                            contract.get("fingerprint_sha256", "")
                        ),
                        "run_contract_file_sha256": _sha256(contract_path),
                        "parent_control_completed_json": str(control_path),
                        "parent_universal_state_sha256": parent,
                        "candidate_universal_state_sha256": candidate_parent,
                        "copied_universal_state_sha256": copied,
                        "post_frozen_expert_universal_state_sha256": post_frozen,
                        "final_universal_state_sha256": final,
                        **checks,
                        "receipt_valid": valid,
                        "completed_json": str(path),
                    }
                )

    table = pd.DataFrame(rows).sort_values(
        ["producer", "seed", "scenario", "fold"]
    )
    summary: dict[str, object] = {
        "schema": "fair_expert_receipt_audit_v1",
        "run_root": str(root),
        "control_id": control_id,
        "expected_fits_per_task": expected_fits_per_task,
        "task_fit_counts": task_counts,
        "source_fingerprint_sha256": next(iter(source_hashes))
        if len(source_hashes) == 1
        else "",
        "source_fingerprints_are_identical": len(source_hashes) == 1,
        "receipt_rows": int(len(table)),
        "invalid_receipt_rows": int((~table["receipt_valid"]).sum()),
        "status": "valid"
        if len(source_hashes) == 1 and bool(table["receipt_valid"].all())
        else "invalid",
    }
    return table, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--control-id", default="FAIR-M7.0-U9")
    parser.add_argument("--expected-fits-per-task", type=int, default=14)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    table, summary = audit_receipts(
        args.run_root,
        control_id=args.control_id,
        expected_fits_per_task=args.expected_fits_per_task,
    )
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(args.run_root).resolve() / "consumer" / "fair_expert_receipts"
    )
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "receipt_audit.csv", index=False)
    with (output / "receipt_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
