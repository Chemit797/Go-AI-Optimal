from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

import yaml

from scripts.nightly.run_m7_m8_overnight import (
    GPU_WAIT_TIMEOUT_RETURN_CODE,
    PipelineOptions,
    Stage,
    build_stages,
    parse_gpu_indices,
    query_gpu_readings,
    run_pipeline,
    verify_stage,
)


EXPECTED_ORDER = [
    "fair_expert_ablation",
    "audit_fair_expert_receipts",
    "summarize_fair_experts",
    "quick_screen",
    "research_prior_pair_ablation",
    "summarize_quick_screen",
    "summarize_research_prior_pair",
    "calibration_audit",
    "audit_calibration_results",
    "summarize_calibration_audit",
    "audit_confirmation_identity",
    "prepare_m7_confirmation",
    "run_selected_m7_confirmation",
    "gate_selected_m7_confirmation",
]


def _success_verifier(stage: Stage) -> tuple[bool, str]:
    return True, f"verified {stage.stage_id}"


def test_overnight_builds_fixed_order_on_one_configurable_base_disk(tmp_path: Path) -> None:
    options = PipelineOptions(
        base_root=tmp_path / "large-disk",
        python="/opt/goai/bin/python",
        gpus="2,3",
        min_free_gb=80.0,
        poll_seconds=1.5,
    )
    stages = build_stages(options)
    assert [stage.stage_id for stage in stages] == EXPECTED_ORDER
    assert {stage.run_root.parent for stage in stages} == {options.base_root.resolve()}
    for stage in stages:
        assert isinstance(stage.command, tuple)
        assert stage.command[0] == options.python
    matrix_commands = [stage.command for stage in stages if stage.kind == "matrix"]
    assert all(("--gpus", "2,3") == command[command.index("--gpus") : command.index("--gpus") + 2] for command in matrix_commands)
    assert all("80.0" in command for command in matrix_commands)


def test_overnight_records_every_stage_and_repeat_safely_skips_completed(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n")

    options = PipelineOptions(base_root=tmp_path / "overnight", python="python-goai")
    assert run_pipeline(
        options,
        command_runner=runner,
        stage_verifier=_success_verifier,
    ) == 0
    assert len(calls) == len(EXPECTED_ORDER)
    assert [Path(command[1]).stem for command in calls] == [
        "run_matrix",
        "audit_fair_expert_receipts",
        "summarize_m7_m8",
        "run_matrix",
        "run_matrix",
        "summarize_m7_m8",
        "summarize_m7_m8",
        "run_matrix",
        "audit_calibration_results",
        "summarize_m7_m8",
        "audit_identity_for_confirmation",
        "prepare_m7_confirmation",
        "run_matrix",
        "run_m7_promotion_batch",
    ]

    status_path = options.base_root / "overnight_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    assert status["sequence"] == EXPECTED_ORDER
    assert all(status["stages"][name]["returncode"] == 0 for name in EXPECTED_ORDER)
    assert all(len(status["stages"][name]["attempts"]) == 1 for name in EXPECTED_ORDER)

    calls.clear()
    assert run_pipeline(
        options,
        command_runner=runner,
        stage_verifier=_success_verifier,
    ) == 0
    assert calls == []
    resumed = json.loads(status_path.read_text(encoding="utf-8"))
    assert all("resume_skipped_at" in resumed["stages"][name] for name in EXPECTED_ORDER)


def test_fair_receipt_audit_failure_stops_before_any_summary_or_later_matrix(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        returncode = 9 if Path(command[1]).name == "audit_fair_expert_receipts.py" else 0
        return subprocess.CompletedProcess(command, returncode, stdout="synthetic\n")

    options = PipelineOptions(base_root=tmp_path / "overnight")
    assert run_pipeline(
        options,
        command_runner=runner,
        stage_verifier=_success_verifier,
    ) == 9
    assert [Path(command[1]).name for command in calls] == [
        "run_matrix.py",
        "audit_fair_expert_receipts.py",
    ]
    status = json.loads((options.base_root / "overnight_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["failed_stage"] == "audit_fair_expert_receipts"
    assert status["stages"]["summarize_fair_experts"]["state"] == "pending"
    assert status["stages"]["quick_screen"]["state"] == "pending"


def test_matrix_stage_verification_requires_every_experiment_seed_summary(
    tmp_path: Path,
) -> None:
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "experiments": [
                    {"id": "ONE", "seeds": [42, 52]},
                    {"id": "TWO", "seeds": [62]},
                ]
            }
        ),
        encoding="utf-8",
    )
    stage = Stage(
        stage_id="synthetic",
        kind="matrix",
        command=("python", "run_matrix.py"),
        run_root=tmp_path / "run",
        matrix=matrix,
    )
    valid, detail = verify_stage(stage)
    assert not valid
    assert "missing 3 producer summaries" in detail
    for experiment, seed in (("ONE", 42), ("ONE", 52), ("TWO", 62)):
        path = stage.run_root / "producers" / experiment / f"S{seed}" / "oof_summary.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("scenario,fc_pcc_mean\nR00,0.1\n", encoding="utf-8")
    assert verify_stage(stage) == (True, "all matrix producer summaries are present")


def test_confirmation_preparation_verifier_accepts_explicit_m8_block_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    output = tmp_path / "preparation"
    output.mkdir()
    matrix = tmp_path / "confirm_matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "experiments": [
                    {"id": "CONF-M7.0-GENERAL", "model_id": "M7.0"}
                ]
            }
        ),
        encoding="utf-8",
    )

    def write_hashed(name: str, payload: dict) -> Path:
        path = output / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        Path(str(path) + ".sha256").write_text(
            hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="utf-8"
        )
        return path

    evidence = write_hashed("preconfirmation_evidence.json", {"status": "valid"})
    m8 = write_hashed(
        "m8_blocked_receipt.json",
        {"status": "blocked", "gpu_tasks_started": False},
    )
    matrix_hash = hashlib.sha256(matrix.read_bytes()).hexdigest()
    write_hashed(
        "confirmation_selection_receipt.json",
        {
            "status": "prepared",
            "matrix_experiments": ["CONF-M7.0-GENERAL"],
            "matrix": {"path": str(matrix), "sha256": matrix_hash},
            "preconfirmation_evidence": {
                "path": str(evidence),
                "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            },
            "m8_blocked_receipt": {
                "path": str(m8),
                "sha256": hashlib.sha256(m8.read_bytes()).hexdigest(),
            },
        },
    )
    stage = Stage(
        "prepare",
        "confirmation_prepare",
        ("python", "prepare.py"),
        tmp_path,
        matrix=matrix,
        output_dir=output,
    )
    assert verify_stage(stage) == (
        True,
        "M7 candidates and controls materialized; M8 is blocked",
    )
    matrix.write_text("experiments: []\n", encoding="utf-8")
    valid, detail = verify_stage(stage)
    assert valid is False
    assert "incomplete" in detail


def test_zero_returncode_with_missing_receipt_is_a_pipeline_failure(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="stopped by disk guard\n")

    def reject_first(stage: Stage) -> tuple[bool, str]:
        return False, f"missing durable receipt for {stage.stage_id}"

    options = PipelineOptions(base_root=tmp_path / "overnight")
    assert run_pipeline(
        options,
        command_runner=runner,
        stage_verifier=reject_first,
    ) == 1
    assert len(calls) == 1
    status = json.loads((options.base_root / "overnight_status.json").read_text(encoding="utf-8"))
    assert status["failed_stage"] == "fair_expert_ablation"
    assert status["stages"]["fair_expert_ablation"]["state"] == "failed_verification"


def test_gpu_indices_and_nvidia_smi_query_are_explicit_and_ordered() -> None:
    calls: list[list[str]] = []

    def query_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="1, 31000, 12\n3, 42000, 4\n",
            stderr="",
        )

    assert parse_gpu_indices("3, 1") == (3, 1)
    readings = query_gpu_readings((3, 1), command_runner=query_runner)
    assert [reading.index for reading in readings] == [3, 1]
    assert [reading.memory_free_mb for reading in readings] == [42000, 31000]
    assert calls == [
        [
            "nvidia-smi",
            "--id=3,1",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    ]


def test_gpu_gate_waits_without_launching_then_checks_each_matrix_stage(
    tmp_path: Path,
) -> None:
    stage_calls: list[list[str]] = []
    query_calls: list[list[str]] = []
    sleep_calls: list[float] = []
    waiting_snapshots: list[dict] = []
    clock = [0.0]
    base_root = tmp_path / "overnight"

    def monotonic() -> float:
        return clock[0]

    def sleeper(seconds: float) -> None:
        sleep_calls.append(seconds)
        clock[0] += seconds

    def query_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        query_calls.append(command)
        if len(query_calls) == 2:
            waiting_snapshots.append(
                json.loads((base_root / "overnight_status.json").read_text(encoding="utf-8"))
            )
        output = (
            "0, 12000, 91\n1, 50000, 2\n"
            if len(query_calls) == 1
            else "0, 50000, 2\n1, 51000, 1\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    def stage_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        stage_calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n")

    options = PipelineOptions(
        base_root=base_root,
        min_gpu_free_mb=30000,
        max_gpu_utilization=20,
        gpu_wait_poll_seconds=2,
        gpu_wait_timeout_seconds=10,
    )
    assert run_pipeline(
        options,
        command_runner=stage_runner,
        stage_verifier=_success_verifier,
        gpu_query_runner=query_runner,
        monotonic=monotonic,
        sleeper=sleeper,
    ) == 0
    assert len(stage_calls) == len(EXPECTED_ORDER)
    assert len(query_calls) == 6  # first matrix waits once; later four matrices recheck
    assert sleep_calls == [2]
    waiting = waiting_snapshots[0]
    assert waiting["state"] == "waiting_for_gpu"
    assert waiting["gpu_gate"]["thresholds"]["min_gpu_free_mb"] == 30000
    assert waiting["gpu_gate"]["thresholds"]["max_gpu_utilization"] == 20
    assert waiting["gpu_gate"]["readings"][0]["ready"] is False
    final = json.loads((base_root / "overnight_status.json").read_text(encoding="utf-8"))
    assert final["state"] == "completed"
    assert final["gpu_gate"]["state"] == "ready"


def test_base_stop_sentinel_returns_resumable_without_gpu_query_or_stage_launch(
    tmp_path: Path,
) -> None:
    base_root = tmp_path / "overnight"
    base_root.mkdir()
    (base_root / "STOP").write_text("", encoding="utf-8")
    stage_calls: list[list[str]] = []
    query_calls: list[list[str]] = []

    def stage_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        stage_calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="should not run\n")

    def query_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        query_calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="0, 50000, 0\n1, 50000, 0\n")

    options = PipelineOptions(base_root=base_root, min_gpu_free_mb=30000)
    assert run_pipeline(
        options,
        command_runner=stage_runner,
        stage_verifier=_success_verifier,
        gpu_query_runner=query_runner,
    ) == 0
    assert stage_calls == []
    assert query_calls == []
    status = json.loads((base_root / "overnight_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "stopped"
    assert status["resume_available"] is True
    assert all(not record["attempts"] for record in status["stages"].values())


def test_gpu_wait_timeout_is_resumable_and_never_launches_a_stage(tmp_path: Path) -> None:
    stage_calls: list[list[str]] = []
    query_calls: list[list[str]] = []
    clock = [0.0]

    def monotonic() -> float:
        return clock[0]

    def sleeper(seconds: float) -> None:
        clock[0] += seconds

    def query_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        query_calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="0, 10000, 90\n1, 12000, 80\n",
            stderr="",
        )

    def stage_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        stage_calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="should not run\n")

    options = PipelineOptions(
        base_root=tmp_path / "overnight",
        min_gpu_free_mb=30000,
        max_gpu_utilization=20,
        gpu_wait_poll_seconds=3,
        gpu_wait_timeout_seconds=2,
    )
    assert run_pipeline(
        options,
        command_runner=stage_runner,
        stage_verifier=_success_verifier,
        gpu_query_runner=query_runner,
        monotonic=monotonic,
        sleeper=sleeper,
    ) == GPU_WAIT_TIMEOUT_RETURN_CODE
    assert stage_calls == []
    assert len(query_calls) == 2
    status = json.loads(
        (options.base_root / "overnight_status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "gpu_wait_timeout"
    assert status["resume_available"] is True
    assert status["gpu_gate"]["state"] == "timeout"
