#!/usr/bin/env python3
"""Refresh and classify the entity-identity receipt for M7/M8 confirmation.

The underlying strict registry audit intentionally exits non-zero while
candidate/proxy identities remain unsuitable for formal M8 semantics.  That
is a valid outcome for the closed-data M7 route, not an orchestration error.
This wrapper therefore accepts the raw audit only after the independent
pre-confirmation validator proves that all core registry contracts and hash
chains are intact, then records M8 as either ready or blocked.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.nightly.confirmation_evidence import (
        PROTOCOL_LABEL,
        sha256_file,
        validate_identity_audit,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from confirmation_evidence import (  # type: ignore
        PROTOCOL_LABEL,
        sha256_file,
        validate_identity_audit,
    )


PROJECT = Path(__file__).resolve().parents[2]


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


def audit_identity(
    *,
    output_dir: str | Path,
    python: str = sys.executable,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_receipt = output / "entity_registry_audit.json"
    command = (
        str(python),
        str(PROJECT / "scripts" / "audit_entity_registry.py"),
        "--strict-semantic",
        "--output",
        str(raw_receipt),
    )
    result = subprocess.run(
        list(command),
        cwd=str(PROJECT),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log = output / "entity_registry_audit.log"
    log.write_text(str(result.stdout or ""), encoding="utf-8")
    if not raw_receipt.is_file():
        raise RuntimeError(
            "Entity registry audit did not emit its raw receipt; "
            f"returncode={result.returncode}, log={log}"
        )

    # This raises on core identity errors or a broken registry hash chain.  A
    # semantic_promotion_status of blocked is deliberately accepted for M7.
    identity = validate_identity_audit(raw_receipt)
    semantic_status = str(identity["semantic_promotion_status"])
    receipt: dict[str, Any] = {
        "schema": "goai.m7_m8.identity_gate_receipt.v1",
        "protocol_label": PROTOCOL_LABEL,
        "generated_at": _now(),
        "status": "valid_for_m7",
        "m7_confirmation_allowed": True,
        "m8_confirmation_status": (
            "ready" if semantic_status == "ready" else "blocked"
        ),
        "raw_audit_returncode": int(result.returncode),
        "raw_audit": {
            "path": str(raw_receipt),
            "sha256": sha256_file(raw_receipt),
        },
        "validated_identity": identity,
        "decision_reason": (
            "Core registry and hash-chain contracts are valid; verified-only "
            "semantic evidence is ready."
            if semantic_status == "ready"
            else "Core registry and hash-chain contracts are valid for M7, but "
            "verified-only semantic evidence is blocked, so M8 cannot start."
        ),
    }
    receipt_path = output / "identity_gate_receipt.json"
    _atomic_json(receipt_path, receipt)
    Path(str(receipt_path) + ".sha256").write_text(
        sha256_file(receipt_path) + "\n", encoding="utf-8"
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    return parser


def main() -> int:
    args = _parser().parse_args()
    receipt = audit_identity(output_dir=args.output_dir, python=args.python)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
