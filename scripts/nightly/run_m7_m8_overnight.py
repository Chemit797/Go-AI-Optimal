"""Run the resumable M7/M8 overnight pipeline in a fail-fast order.

The producer work is delegated to ``run_matrix.py`` so its fold-level resume,
per-run ``STOP`` sentinel, and disk guard remain the single implementation of
those behaviours.  This wrapper only sequences producers and consumers and
writes an inspectable pipeline receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence, Tuple

import yaml


PROJECT = Path(__file__).resolve().parents[2]
MATRIX_DIR = PROJECT / "configs" / "nightly" / "20260813-m7-m8"
DEFAULT_BASE_ROOT = PROJECT / "runs" / "nightly" / "20260813-m7-m8-overnight"
STATUS_SCHEMA = "goai_m7_m8_overnight_v1"
GPU_WAIT_TIMEOUT_RETURN_CODE = 75


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hashed_receipt_valid(path: Path) -> bool:
    sidecar = Path(str(path) + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        return False
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    return bool(expected and expected == _sha256(path))


def parse_gpu_indices(value: str) -> tuple[int, ...]:
    """Parse a comma-separated physical GPU index list without ambiguity."""
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    if not parts:
        raise ValueError("--gpus must contain at least one explicit GPU index")
    try:
        indices = tuple(int(item) for item in parts)
    except ValueError as error:
        raise ValueError("--gpus accepts only comma-separated integer GPU indices") from error
    if any(index < 0 for index in indices):
        raise ValueError("--gpus indices must be non-negative")
    if len(set(indices)) != len(indices):
        raise ValueError("--gpus must not contain duplicate indices")
    return indices


@dataclass(frozen=True)
class PipelineOptions:
    base_root: Path
    python: str = sys.executable
    gpus: str = "0,1"
    min_free_gb: float = 25.0
    poll_seconds: float = 5.0
    min_gpu_free_mb: int = 0
    max_gpu_utilization: float = 100.0
    gpu_wait_poll_seconds: float = 15.0
    gpu_wait_timeout_seconds: float = 0.0

    def __post_init__(self) -> None:
        parse_gpu_indices(self.gpus)
        if self.min_gpu_free_mb < 0:
            raise ValueError("min_gpu_free_mb must be non-negative")
        if not 0.0 <= self.max_gpu_utilization <= 100.0:
            raise ValueError("max_gpu_utilization must be between 0 and 100")
        if self.gpu_wait_poll_seconds <= 0:
            raise ValueError("gpu_wait_poll_seconds must be positive")
        if self.gpu_wait_timeout_seconds < 0:
            raise ValueError("gpu_wait_timeout_seconds must be non-negative")

    @property
    def gpu_indices(self) -> tuple[int, ...]:
        return parse_gpu_indices(self.gpus)

    @property
    def normalized_gpus(self) -> str:
        return ",".join(str(index) for index in self.gpu_indices)

    @property
    def gpu_gate_enabled(self) -> bool:
        return self.min_gpu_free_mb > 0 or self.max_gpu_utilization < 100.0


@dataclass(frozen=True)
class Stage:
    stage_id: str
    kind: str
    command: tuple[str, ...]
    run_root: Path
    matrix: Path | None = None
    output_dir: Path | None = None


@dataclass(frozen=True)
class GPUReading:
    index: int
    memory_free_mb: int
    utilization_percent: float


GPUQueryRunner = Callable[..., subprocess.CompletedProcess]
Clock = Callable[[], float]
Sleeper = Callable[[float], None]


def query_gpu_readings(
    gpu_indices: Sequence[int],
    *,
    command_runner: GPUQueryRunner | None = None,
) -> tuple[GPUReading, ...]:
    """Read physical GPU capacity from one inspectable ``nvidia-smi`` call."""
    indices = tuple(int(index) for index in gpu_indices)
    if not indices:
        raise ValueError("At least one GPU index is required")
    runner = command_runner or subprocess.run
    command = [
        "nvidia-smi",
        f"--id={','.join(str(index) for index in indices)}",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = runner(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if int(result.returncode) != 0:
        detail = str(getattr(result, "stderr", "") or "").strip()
        raise RuntimeError(
            f"nvidia-smi GPU capacity query failed with code {result.returncode}: {detail}"
        )

    readings: dict[int, GPUReading] = {}
    for line in str(getattr(result, "stdout", "") or "").splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise RuntimeError(f"Unexpected nvidia-smi output row: {line!r}")
        try:
            index = int(fields[0])
            memory_free_mb = int(float(fields[1]))
            utilization_percent = float(fields[2])
        except ValueError as error:
            raise RuntimeError(f"Invalid nvidia-smi output row: {line!r}") from error
        if index in readings:
            raise RuntimeError(f"nvidia-smi returned duplicate GPU index {index}")
        readings[index] = GPUReading(index, memory_free_mb, utilization_percent)

    requested = set(indices)
    returned = set(readings)
    if returned != requested:
        raise RuntimeError(
            "nvidia-smi did not return exactly the requested GPUs: "
            f"requested={sorted(requested)}, returned={sorted(returned)}"
        )
    return tuple(readings[index] for index in indices)


def _gpu_gate_payload(
    options: PipelineOptions,
    readings: Sequence[GPUReading],
    *,
    state: str,
    stage_id: str,
    wait_started_at: str,
    elapsed_seconds: float,
) -> dict:
    rows = []
    for reading in readings:
        memory_ready = reading.memory_free_mb >= options.min_gpu_free_mb
        utilization_ready = reading.utilization_percent <= options.max_gpu_utilization
        rows.append(
            {
                "index": reading.index,
                "memory_free_mb": reading.memory_free_mb,
                "utilization_percent": reading.utilization_percent,
                "memory_ready": memory_ready,
                "utilization_ready": utilization_ready,
                "ready": memory_ready and utilization_ready,
            }
        )
    return {
        "state": state,
        "stage": stage_id,
        "target_gpu_indices": list(options.gpu_indices),
        "thresholds": {
            "min_gpu_free_mb": options.min_gpu_free_mb,
            "max_gpu_utilization": options.max_gpu_utilization,
            "poll_seconds": options.gpu_wait_poll_seconds,
            "timeout_seconds": options.gpu_wait_timeout_seconds,
        },
        "readings": rows,
        "all_ready": bool(rows) and all(bool(row["ready"]) for row in rows),
        "wait_started_at": wait_started_at,
        "checked_at": _now(),
        "elapsed_seconds": round(max(0.0, elapsed_seconds), 3),
    }


def wait_for_gpu_capacity(
    options: PipelineOptions,
    *,
    stage_id: str,
    status: dict,
    status_path: Path,
    stop_path: Path,
    query_runner: GPUQueryRunner | None = None,
    monotonic: Clock = time.monotonic,
    sleeper: Sleeper = time.sleep,
) -> str:
    """Wait without occupying a GPU; return ready, stopped, timeout, or error."""
    if not options.gpu_gate_enabled:
        return "ready"

    started_at = _now()
    started_clock = monotonic()
    while True:
        if stop_path.is_file():
            finished = _now()
            status.update(
                {
                    "state": "stopped",
                    "current_stage": None,
                    "stop_reason": f"STOP sentinel present: {stop_path}",
                    "resume_available": True,
                    "updated_at": finished,
                }
            )
            _atomic_json(status_path, status)
            return "stopped"

        elapsed = max(0.0, monotonic() - started_clock)
        try:
            readings = query_gpu_readings(
                options.gpu_indices,
                command_runner=query_runner,
            )
        except Exception as error:
            finished = _now()
            status.update(
                {
                    "state": "gpu_gate_error",
                    "current_stage": None,
                    "gpu_gate_error": str(error),
                    "resume_available": True,
                    "updated_at": finished,
                }
            )
            _atomic_json(status_path, status)
            return "error"

        all_ready = all(
            reading.memory_free_mb >= options.min_gpu_free_mb
            and reading.utilization_percent <= options.max_gpu_utilization
            for reading in readings
        )
        gate_state = "ready" if all_ready else "waiting_for_gpu"
        status["gpu_gate"] = _gpu_gate_payload(
            options,
            readings,
            state=gate_state,
            stage_id=stage_id,
            wait_started_at=started_at,
            elapsed_seconds=elapsed,
        )
        status["current_stage"] = stage_id
        status["updated_at"] = _now()
        if all_ready:
            status["state"] = "running"
            _atomic_json(status_path, status)
            return "ready"

        status["state"] = "waiting_for_gpu"
        _atomic_json(status_path, status)
        timeout = options.gpu_wait_timeout_seconds
        if timeout > 0 and elapsed >= timeout:
            finished = _now()
            status["state"] = "gpu_wait_timeout"
            status["current_stage"] = None
            status["resume_available"] = True
            status["gpu_gate"]["state"] = "timeout"
            status["updated_at"] = finished
            _atomic_json(status_path, status)
            return "timeout"
        sleeper(options.gpu_wait_poll_seconds)


def _matrix_command(
    options: PipelineOptions,
    matrix: Path,
    run_root: Path,
) -> tuple[str, ...]:
    return (
        options.python,
        str(PROJECT / "scripts" / "nightly" / "run_matrix.py"),
        "--matrix",
        str(matrix),
        "--run-root",
        str(run_root),
        "--python",
        options.python,
        "--gpus",
        options.normalized_gpus,
        "--min-free-gb",
        str(options.min_free_gb),
        "--poll-seconds",
        str(options.poll_seconds),
    )


def _consumer_command(
    options: PipelineOptions,
    script: str,
    *arguments: str,
) -> tuple[str, ...]:
    return (
        options.python,
        str(PROJECT / "scripts" / "nightly" / script),
        *arguments,
    )


def build_stages(options: PipelineOptions) -> list[Stage]:
    """Build the fixed, evidence-gated overnight stage order."""
    base = options.base_root.resolve()
    fair = base / "fair-experts"
    quick = base / "quick-screen"
    research = base / "research-prior-pair"
    calibration = base / "calibration-audit"
    identity = base / "identity-audit"
    confirm = base / "formal-m7-confirm"
    fair_matrix = MATRIX_DIR / "fair_expert_ablation.yaml"
    quick_matrix = MATRIX_DIR / "quick_screen.yaml"
    research_matrix = MATRIX_DIR / "research_prior_pair_ablation.yaml"
    calibration_matrix = MATRIX_DIR / "calibration_audit.yaml"
    confirm_template = MATRIX_DIR / "confirm_candidates.yaml"
    confirm_matrix = confirm / "confirm_matrix.yaml"
    preparation = confirm / "consumer" / "preparation"
    return [
        Stage(
            "fair_expert_ablation",
            "matrix",
            _matrix_command(options, fair_matrix, fair),
            fair,
            matrix=fair_matrix,
        ),
        Stage(
            "audit_fair_expert_receipts",
            "audit",
            _consumer_command(
                options,
                "audit_fair_expert_receipts.py",
                "--run-root",
                str(fair),
            ),
            fair,
            output_dir=fair / "consumer" / "fair_expert_receipts",
        ),
        Stage(
            "summarize_fair_experts",
            "summary",
            _consumer_command(
                options,
                "summarize_m7_m8.py",
                "--run-root",
                str(fair),
                "--control-id",
                "FAIR-M7.0-U9",
            ),
            fair,
            output_dir=fair / "consumer" / "m7_m8",
        ),
        Stage(
            "quick_screen",
            "matrix",
            _matrix_command(options, quick_matrix, quick),
            quick,
            matrix=quick_matrix,
        ),
        Stage(
            "research_prior_pair_ablation",
            "matrix",
            _matrix_command(options, research_matrix, research),
            research,
            matrix=research_matrix,
        ),
        Stage(
            "summarize_quick_screen",
            "summary",
            _consumer_command(
                options,
                "summarize_m7_m8.py",
                "--run-root",
                str(quick),
                "--run-root",
                str(research),
                "--control-id",
                "SCR-M7.0-GENERAL",
            ),
            quick,
            output_dir=quick / "consumer" / "m7_m8",
        ),
        Stage(
            "summarize_research_prior_pair",
            "summary",
            _consumer_command(
                options,
                "summarize_m7_m8.py",
                "--run-root",
                str(research),
                "--control-id",
                "RESEARCH-PRIOR-NONE",
            ),
            research,
            output_dir=research / "consumer" / "m7_m8",
        ),
        Stage(
            "calibration_audit",
            "matrix",
            _matrix_command(options, calibration_matrix, calibration),
            calibration,
            matrix=calibration_matrix,
        ),
        Stage(
            "audit_calibration_results",
            "calibration_audit",
            _consumer_command(
                options,
                "audit_calibration_results.py",
                "--run-root",
                str(calibration),
            ),
            calibration,
            output_dir=calibration / "consumer",
        ),
        Stage(
            "summarize_calibration_audit",
            "summary",
            _consumer_command(
                options,
                "summarize_m7_m8.py",
                "--run-root",
                str(calibration),
                "--control-id",
                "CAL-M7.0-R16-BASE",
            ),
            calibration,
            output_dir=calibration / "consumer" / "m7_m8",
        ),
        Stage(
            "audit_confirmation_identity",
            "identity_audit",
            _consumer_command(
                options,
                "audit_identity_for_confirmation.py",
                "--output-dir",
                str(identity),
                "--python",
                options.python,
            ),
            identity,
            output_dir=identity,
        ),
        Stage(
            "prepare_m7_confirmation",
            "confirmation_prepare",
            _consumer_command(
                options,
                "prepare_m7_confirmation.py",
                "--template",
                str(confirm_template),
                "--scale-selection",
                str(quick / "consumer" / "m7_m8" / "expert_scale_selection.yaml"),
                "--scale-candidates",
                str(quick / "consumer" / "m7_m8" / "expert_scale_candidates.csv"),
                "--identity-audit",
                str(identity / "entity_registry_audit.json"),
                "--calibration-audit",
                str(calibration / "consumer" / "calibration_audit_receipt.json"),
                "--fair-expert-audit",
                str(fair / "consumer" / "fair_expert_receipts" / "receipt_audit.json"),
                "--fair-paired-summary",
                str(fair / "consumer" / "m7_m8" / "paired_delta_summary.csv"),
                "--quick-summary-dir",
                str(quick / "consumer" / "m7_m8"),
                "--research-summary-dir",
                str(research / "consumer" / "m7_m8"),
                "--output-matrix",
                str(confirm_matrix),
                "--output-dir",
                str(preparation),
            ),
            confirm,
            matrix=confirm_matrix,
            output_dir=preparation,
        ),
        Stage(
            "run_selected_m7_confirmation",
            "matrix",
            _matrix_command(options, confirm_matrix, confirm),
            confirm,
            matrix=confirm_matrix,
        ),
        Stage(
            "gate_selected_m7_confirmation",
            "promotion_batch",
            _consumer_command(
                options,
                "run_m7_promotion_batch.py",
                "--run-root",
                str(confirm),
                "--selection-receipt",
                str(preparation / "confirmation_selection_receipt.json"),
                "--output-dir",
                str(confirm / "consumer" / "promotion"),
            ),
            confirm,
            output_dir=confirm / "consumer" / "promotion",
        ),
    ]


def _expected_producer_summaries(stage: Stage) -> list[Path]:
    if stage.matrix is None:
        return []
    payload = yaml.safe_load(stage.matrix.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("experiments"), list):
        raise ValueError(f"Invalid matrix: {stage.matrix}")
    expected: list[Path] = []
    for experiment in payload["experiments"]:
        experiment_id = str(experiment["id"])
        for seed in experiment.get("seeds", [42]):
            expected.append(
                stage.run_root
                / "producers"
                / experiment_id
                / f"S{int(seed)}"
                / "oof_summary.csv"
            )
    return expected


def verify_stage(stage: Stage) -> tuple[bool, str]:
    """Verify the durable receipt expected from a successful stage."""
    if stage.kind == "matrix":
        missing = [path for path in _expected_producer_summaries(stage) if not path.is_file()]
        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            return False, f"missing {len(missing)} producer summaries; first: {preview}"
        return True, "all matrix producer summaries are present"
    if stage.kind == "audit":
        receipt = (stage.output_dir or stage.run_root) / "receipt_audit.json"
        if not receipt.is_file():
            return False, f"missing audit receipt: {receipt}"
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        if payload.get("status") != "valid":
            return False, f"audit receipt status is {payload.get('status')!r}"
        return True, "fair expert receipts are valid"
    if stage.kind == "calibration_audit":
        receipt = (stage.output_dir or stage.run_root) / "calibration_audit_receipt.json"
        if not receipt.is_file():
            return False, f"missing calibration audit receipt: {receipt}"
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        if payload.get("status") != "approved" or payload.get("audit_complete") is not True:
            return False, f"calibration audit receipt status is {payload.get('status')!r}"
        return True, "calibration audit receipt is approved and complete"
    if stage.kind == "identity_audit":
        receipt = (stage.output_dir or stage.run_root) / "identity_gate_receipt.json"
        if not _hashed_receipt_valid(receipt):
            return False, f"missing identity gate receipt: {receipt}"
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        raw_record = payload.get("raw_audit", {})
        raw_path = Path(str(raw_record.get("path", "")))
        if (
            payload.get("status") != "valid_for_m7"
            or payload.get("m7_confirmation_allowed") is not True
            or payload.get("m8_confirmation_status") not in {"blocked", "ready"}
            or not raw_path.is_file()
            or _sha256(raw_path) != str(raw_record.get("sha256", ""))
        ):
            return False, "identity receipt does not authorize the M7 confirmation route"
        return True, "identity core is valid; M8 semantic status is explicit"
    if stage.kind == "confirmation_prepare":
        output = stage.output_dir or stage.run_root
        receipt_path = output / "confirmation_selection_receipt.json"
        evidence_path = output / "preconfirmation_evidence.json"
        m8_path = output / "m8_blocked_receipt.json"
        if not all(
            _hashed_receipt_valid(path)
            for path in (receipt_path, evidence_path, m8_path)
        ):
            return False, "missing confirmation preparation/evidence/M8 block receipt"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        m8 = json.loads(m8_path.read_text(encoding="utf-8"))
        matrix = stage.matrix
        matrix_record = receipt.get("matrix", {})
        evidence_record = receipt.get("preconfirmation_evidence", {})
        m8_record = receipt.get("m8_blocked_receipt", {})
        if (
            receipt.get("status") != "prepared"
            or evidence.get("status") != "valid"
            or m8.get("status") != "blocked"
            or m8.get("gpu_tasks_started") is not False
            or matrix is None
            or not matrix.is_file()
            or _sha256(matrix) != str(matrix_record.get("sha256", ""))
            or Path(str(matrix_record.get("path", ""))).resolve() != matrix.resolve()
            or _sha256(evidence_path) != str(evidence_record.get("sha256", ""))
            or _sha256(m8_path) != str(m8_record.get("sha256", ""))
        ):
            return False, "confirmation preparation contracts are incomplete"
        matrix_payload = yaml.safe_load(matrix.read_text(encoding="utf-8"))
        experiments = matrix_payload.get("experiments", []) if isinstance(matrix_payload, dict) else []
        ids = [str(item.get("id", "")) for item in experiments if isinstance(item, dict)]
        if ids != list(receipt.get("matrix_experiments", [])):
            return False, "selected confirmation matrix differs from its receipt"
        if any(str(item.get("model_id", "")).startswith("M8") for item in experiments):
            return False, "M8 was materialized despite its blocked receipt"
        return True, "M7 candidates and controls materialized; M8 is blocked"
    if stage.kind == "promotion_batch":
        receipt = (stage.output_dir or stage.run_root) / "promotion_batch_receipt.json"
        if not _hashed_receipt_valid(receipt):
            return False, f"missing promotion batch receipt: {receipt}"
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        results = payload.get("results", [])
        receipts_valid = isinstance(results, list) and all(
            isinstance(item, dict)
            and Path(str(item.get("receipt", ""))).is_file()
            and _sha256(Path(str(item["receipt"])))
            == str(item.get("receipt_sha256", ""))
            for item in results
        )
        if (
            payload.get("status") != "complete"
            or payload.get("protocol_label") != "LOCAL_STRICT_OOF_NOT_OFFICIAL"
            or int(payload.get("candidate_count", -1))
            != len(results)
            or not receipts_valid
        ):
            return False, "promotion batch receipt is incomplete"
        return True, "all selected M7 candidates have strict promotion receipts"
    if stage.kind == "summary":
        output = stage.output_dir or stage.run_root
        required = (output / "local_oof_report_zh.md", output / "regime_summary.csv")
        missing = [path for path in required if not path.is_file()]
        if missing:
            return False, f"missing summary outputs: {', '.join(map(str, missing))}"
        return True, "summary report and regime table are present"
    return False, f"unknown stage kind: {stage.kind}"


CommandRunner = Callable[..., subprocess.CompletedProcess]
StageVerifier = Callable[[Stage], Tuple[bool, str]]


def _initial_status(options: PipelineOptions, stages: Sequence[Stage]) -> dict:
    return {
        "schema": STATUS_SCHEMA,
        "created_at": _now(),
        "updated_at": _now(),
        "state": "pending",
        "current_stage": None,
        "project_root": str(PROJECT),
        "base_root": str(options.base_root.resolve()),
        "python": options.python,
        "gpus": options.normalized_gpus,
        "gpu_indices": list(options.gpu_indices),
        "min_free_gb": options.min_free_gb,
        "poll_seconds": options.poll_seconds,
        "min_gpu_free_mb": options.min_gpu_free_mb,
        "max_gpu_utilization": options.max_gpu_utilization,
        "gpu_wait_poll_seconds": options.gpu_wait_poll_seconds,
        "gpu_wait_timeout_seconds": options.gpu_wait_timeout_seconds,
        "sequence": [stage.stage_id for stage in stages],
        "stages": {
            stage.stage_id: {
                "kind": stage.kind,
                "state": "pending",
                "run_root": str(stage.run_root),
                "command": list(stage.command),
                "attempts": [],
            }
            for stage in stages
        },
    }


def _load_or_initialize_status(
    path: Path,
    options: PipelineOptions,
    stages: Sequence[Stage],
) -> dict:
    if not path.is_file():
        return _initial_status(options, stages)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != STATUS_SCHEMA:
        raise ValueError(f"Unsupported status schema in {path}")
    if Path(str(payload.get("base_root", ""))).resolve() != options.base_root.resolve():
        raise ValueError(f"Status base_root does not match requested base_root: {path}")
    payload["python"] = options.python
    payload["gpus"] = options.normalized_gpus
    payload["gpu_indices"] = list(options.gpu_indices)
    payload["min_free_gb"] = options.min_free_gb
    payload["poll_seconds"] = options.poll_seconds
    payload["min_gpu_free_mb"] = options.min_gpu_free_mb
    payload["max_gpu_utilization"] = options.max_gpu_utilization
    payload["gpu_wait_poll_seconds"] = options.gpu_wait_poll_seconds
    payload["gpu_wait_timeout_seconds"] = options.gpu_wait_timeout_seconds
    payload["sequence"] = [stage.stage_id for stage in stages]
    records = payload.setdefault("stages", {})
    for stage in stages:
        record = records.setdefault(
            stage.stage_id,
            {"state": "pending", "attempts": []},
        )
        record["kind"] = stage.kind
        record["run_root"] = str(stage.run_root)
        record["command"] = list(stage.command)
        record.setdefault("attempts", [])
    return payload


def _write_stage_log(path: Path, output: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(output)
        if output and not output.endswith("\n"):
            handle.write("\n")


def _record_stop(status: dict, status_path: Path, stop_path: Path) -> None:
    stopped_at = _now()
    status.update(
        {
            "state": "stopped",
            "current_stage": None,
            "stop_reason": f"STOP sentinel present: {stop_path}",
            "resume_available": True,
            "updated_at": stopped_at,
        }
    )
    _atomic_json(status_path, status)


def run_pipeline(
    options: PipelineOptions,
    *,
    command_runner: CommandRunner | None = None,
    stage_verifier: StageVerifier | None = None,
    gpu_query_runner: GPUQueryRunner | None = None,
    monotonic: Clock = time.monotonic,
    sleeper: Sleeper = time.sleep,
) -> int:
    """Execute or resume the pipeline and return a process-style return code."""
    runner = command_runner or subprocess.run
    verifier = stage_verifier or verify_stage
    base = options.base_root.resolve()
    base.mkdir(parents=True, exist_ok=True)
    stages = build_stages(options)
    status_path = base / "overnight_status.json"
    stop_path = base / "STOP"
    status = _load_or_initialize_status(status_path, options, stages)
    if stop_path.is_file():
        _record_stop(status, status_path, stop_path)
        return 0
    for stage in stages:
        stage.run_root.mkdir(parents=True, exist_ok=True)
        if os.stat(stage.run_root).st_dev != os.stat(base).st_dev:
            raise ValueError(f"Stage root is not on the base-root filesystem: {stage.run_root}")
    status["state"] = "running"
    status["updated_at"] = _now()
    status.pop("stop_reason", None)
    status.pop("resume_available", None)
    status.pop("gpu_gate_error", None)
    _atomic_json(status_path, status)

    initial_gpu_gate_done = False
    for stage in stages:
        if stop_path.is_file():
            _record_stop(status, status_path, stop_path)
            return 0
        record = status["stages"][stage.stage_id]
        if record.get("state") == "completed":
            valid, detail = verifier(stage)
            if valid:
                record["resume_skipped_at"] = _now()
                record["verification"] = detail
                status["updated_at"] = _now()
                _atomic_json(status_path, status)
                continue
            record["state"] = "pending"
            record["verification"] = f"previous completion invalidated: {detail}"

        if not initial_gpu_gate_done or stage.kind == "matrix":
            gate_result = wait_for_gpu_capacity(
                options,
                stage_id=stage.stage_id,
                status=status,
                status_path=status_path,
                stop_path=stop_path,
                query_runner=gpu_query_runner,
                monotonic=monotonic,
                sleeper=sleeper,
            )
            initial_gpu_gate_done = True
            if gate_result == "stopped":
                return 0
            if gate_result == "timeout":
                return GPU_WAIT_TIMEOUT_RETURN_CODE
            if gate_result == "error":
                return 1
        if stop_path.is_file():
            _record_stop(status, status_path, stop_path)
            return 0

        started = _now()
        log_path = base / "orchestrator_logs" / f"{stage.stage_id}.log"
        attempt = {
            "started_at": started,
            "command": list(stage.command),
            "log": str(log_path),
            "state": "running",
        }
        record["attempts"].append(attempt)
        record["state"] = "running"
        record["started_at"] = started
        record["returncode"] = None
        status["current_stage"] = stage.stage_id
        status["updated_at"] = _now()
        _atomic_json(status_path, status)

        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            result = runner(
                list(stage.command),
                cwd=str(PROJECT),
                env=environment,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except KeyboardInterrupt:
            finished = _now()
            attempt.update({"finished_at": finished, "state": "interrupted"})
            record.update({"finished_at": finished, "state": "interrupted"})
            status.update(
                {
                    "state": "interrupted",
                    "current_stage": stage.stage_id,
                    "updated_at": finished,
                }
            )
            _atomic_json(status_path, status)
            return 130
        output = str(getattr(result, "stdout", "") or "")
        _write_stage_log(log_path, output)
        returncode = int(result.returncode)
        finished = _now()
        attempt.update(
            {
                "finished_at": finished,
                "returncode": returncode,
                "state": "completed" if returncode == 0 else "failed",
            }
        )
        record.update({"finished_at": finished, "returncode": returncode})
        if returncode != 0:
            record["state"] = "failed"
            status.update(
                {
                    "state": "failed",
                    "failed_stage": stage.stage_id,
                    "current_stage": None,
                    "finished_at": finished,
                    "updated_at": finished,
                }
            )
            _atomic_json(status_path, status)
            return returncode

        valid, detail = verifier(stage)
        record["verification"] = detail
        if not valid:
            record["state"] = "failed_verification"
            attempt["state"] = "failed_verification"
            attempt["verification"] = detail
            status.update(
                {
                    "state": "failed",
                    "failed_stage": stage.stage_id,
                    "current_stage": None,
                    "finished_at": finished,
                    "updated_at": finished,
                }
            )
            _atomic_json(status_path, status)
            return 1
        record["state"] = "completed"
        attempt["verification"] = detail
        status["current_stage"] = None
        status["updated_at"] = _now()
        _atomic_json(status_path, status)

    finished = _now()
    status.update(
        {
            "state": "completed",
            "current_stage": None,
            "finished_at": finished,
            "updated_at": finished,
        }
    )
    status.pop("failed_stage", None)
    _atomic_json(status_path, status)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-root",
        default=str(DEFAULT_BASE_ROOT),
        help="One large-disk parent for every producer, consumer, log, and status artifact.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--min-free-gb", type=float, default=25.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--min-gpu-free-mb",
        type=int,
        default=0,
        help="Wait until every target GPU has at least this much free memory; 0 disables the memory gate.",
    )
    parser.add_argument(
        "--max-gpu-utilization",
        type=float,
        default=100.0,
        help="Wait until every target GPU is at or below this utilization percentage.",
    )
    parser.add_argument("--gpu-wait-poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--gpu-wait-timeout-seconds",
        type=float,
        default=0.0,
        help="GPU wait timeout; 0 waits indefinitely and remains resumable via the base-root STOP sentinel.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    options = PipelineOptions(
        base_root=Path(args.base_root),
        python=str(args.python),
        gpus=str(args.gpus),
        min_free_gb=float(args.min_free_gb),
        poll_seconds=float(args.poll_seconds),
        min_gpu_free_mb=int(args.min_gpu_free_mb),
        max_gpu_utilization=float(args.max_gpu_utilization),
        gpu_wait_poll_seconds=float(args.gpu_wait_poll_seconds),
        gpu_wait_timeout_seconds=float(args.gpu_wait_timeout_seconds),
    )
    return run_pipeline(options)


if __name__ == "__main__":
    raise SystemExit(main())
