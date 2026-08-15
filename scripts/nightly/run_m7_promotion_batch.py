#!/usr/bin/env python3
"""Run or resume strict promotion gates for selected M7 confirmations.

A statistically blocked candidate is a valid completed result and therefore
does not fail the overnight orchestrator.  Missing/tampered confirmation
artifacts still fail closed.  M8 is never accepted by this M7-only consumer.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

try:
    from scripts.nightly.confirmation_evidence import PROTOCOL_LABEL, sha256_file
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from confirmation_evidence import PROTOCOL_LABEL, sha256_file  # type: ignore


GateRunner = Callable[..., Any]


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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    sidecar = Path(str(path) + ".sha256")
    if sidecar.is_file():
        expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
        if expected != sha256_file(path):
            raise ValueError(f"Receipt sidecar hash mismatch: {path}")
    return payload


def _valid_existing_receipt(
    path: Path,
    *,
    candidate: str,
    matrix_sha256: str,
) -> dict[str, Any] | None:
    if not path.is_file() or not Path(str(path) + ".sha256").is_file():
        return None
    payload = _load_json(path)
    if (
        payload.get("schema") != "goai.m7_m8.promotion_receipt.v1"
        or payload.get("protocol_label") != PROTOCOL_LABEL
        or payload.get("candidate") != candidate
        or payload.get("status") not in {"promoted", "blocked"}
        or payload.get("confirmation_contract", {}).get("matrix_sha256")
        != matrix_sha256
    ):
        return None
    return payload


def run_promotion_batch(
    *,
    run_root: str | Path,
    selection_receipt_path: str | Path,
    output_dir: str | Path,
    bootstrap_draws: int,
    gate_runner: GateRunner | None = None,
) -> dict[str, Any]:
    if gate_runner is None:
        try:
            from scripts.nightly.promotion_gate import run_promotion_gate
        except ModuleNotFoundError:  # pragma: no cover - direct script execution
            from promotion_gate import run_promotion_gate  # type: ignore

        gate_runner = run_promotion_gate
    root = Path(run_root).resolve()
    selection_path = Path(selection_receipt_path).resolve()
    selection = _load_json(selection_path)
    if (
        selection.get("schema")
        != "goai.m7_m8.confirmation_selection_receipt.v1"
        or selection.get("status") != "prepared"
        or selection.get("protocol_label") != PROTOCOL_LABEL
    ):
        raise ValueError("Confirmation selection receipt is invalid")
    matrix_record = selection.get("matrix", {})
    matrix_path = Path(str(matrix_record.get("path", ""))).resolve()
    if not matrix_path.is_file() or sha256_file(matrix_path) != str(
        matrix_record.get("sha256", "")
    ):
        raise ValueError("Confirmation matrix differs from the selection receipt")
    environment_path = root / "environment.json"
    if not environment_path.is_file():
        raise ValueError("Confirmation matrix stage has not written environment.json")
    environment = _load_json(environment_path)
    if (
        Path(str(environment.get("matrix", ""))).resolve() != matrix_path
        or environment.get("matrix_sha256") != sha256_file(matrix_path)
    ):
        raise ValueError("Confirmation run root is bound to a different matrix")
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    if not isinstance(matrix, dict) or not isinstance(matrix.get("experiments"), list):
        raise ValueError("Confirmation matrix is invalid")
    metadata = {str(item["id"]): item for item in matrix["experiments"]}
    candidates = [str(item) for item in selection.get("promotion_candidates", [])]
    if len(candidates) != len(set(candidates)):
        raise ValueError("Confirmation selection contains duplicate candidates")
    if any(candidate.startswith("CONF-M8") for candidate in candidates):
        raise ValueError("M8 cannot enter the automatic M7 promotion batch")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        item = metadata.get(candidate)
        if not isinstance(item, dict) or item.get("kind") != "model_confirm":
            raise ValueError(f"Selected candidate is absent/not formal: {candidate}")
        primary = str(item.get("primary_control", ""))
        scenarios = tuple(str(value) for value in item.get("promotion_regimes", []))
        negatives = tuple(str(value) for value in item.get("required_negative_controls", []))
        if not primary or not scenarios:
            raise ValueError(f"Selected candidate lacks promotion contract: {candidate}")
        candidate_output = output / candidate
        receipt_path = candidate_output / "promotion_receipt.json"
        receipt = _valid_existing_receipt(
            receipt_path,
            candidate=candidate,
            matrix_sha256=sha256_file(matrix_path),
        )
        resumed = receipt is not None
        if receipt is None:
            receipt = gate_runner(
                root,
                candidate,
                primary,
                scenarios,
                candidate_output,
                negative_control_ids=negatives,
                bootstrap_draws=bootstrap_draws,
            )
        if receipt.get("status") not in {"promoted", "blocked"}:
            raise ValueError(f"Promotion gate returned an invalid status for {candidate}")
        if not receipt_path.is_file():
            # Production gate writes its own receipt.  Tests/custom runners are
            # allowed only if the same durable contract is materialized here.
            _atomic_json(receipt_path, receipt)
            Path(str(receipt_path) + ".sha256").write_text(
                sha256_file(receipt_path) + "\n", encoding="utf-8"
            )
        results.append(
            {
                "candidate": candidate,
                "primary_control": primary,
                "required_scenarios": list(scenarios),
                "status": str(receipt["status"]),
                "promoted": bool(receipt.get("promoted", False)),
                "resumed": resumed,
                "receipt": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
            }
        )

    receipt = {
        "schema": "goai.m7_m8.promotion_batch_receipt.v1",
        "protocol_label": PROTOCOL_LABEL,
        "official_score_status": "NOT_OFFICIAL",
        "generated_at": _now(),
        "status": "complete",
        "run_root": str(root),
        "selection_receipt": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "matrix": {"path": str(matrix_path), "sha256": sha256_file(matrix_path)},
        "candidate_count": len(candidates),
        "promoted_count": sum(bool(item["promoted"]) for item in results),
        "blocked_count": sum(item["status"] == "blocked" for item in results),
        "results": results,
        "note": (
            "A blocked statistical decision is a completed confirmation result; "
            "discovery evidence never promotes a model."
        ),
    }
    receipt_path = output / "promotion_batch_receipt.json"
    _atomic_json(receipt_path, receipt)
    Path(str(receipt_path) + ".sha256").write_text(
        sha256_file(receipt_path) + "\n", encoding="utf-8"
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--selection-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    receipt = run_promotion_batch(
        run_root=args.run_root,
        selection_receipt_path=args.selection_receipt,
        output_dir=args.output_dir,
        bootstrap_draws=int(args.bootstrap_draws),
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
