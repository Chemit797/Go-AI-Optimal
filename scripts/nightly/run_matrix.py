"""Run an inspectable two-GPU GOAI OOF producer matrix with resume support."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import yaml


PROJECT = Path(__file__).resolve().parents[2]


def _atomic_json(path: Path, payload: object) -> None:
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


def _code_fingerprint() -> str:
    digest = hashlib.sha256()
    for root in (PROJECT / "src", PROJECT / "scripts" / "nightly", PROJECT / "tests"):
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(PROJECT)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _deep_update(target: dict, changes: dict) -> dict:
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _absolute_config(payload: dict, base_path: Path, run_root: Path) -> dict:
    result = deepcopy(payload)
    base_dir = base_path.parent
    baseline = Path(result["baseline_config"])
    result["baseline_config"] = str((baseline if baseline.is_absolute() else base_dir / baseline).resolve())
    for section, key in (
        ("entity", "chemical_map"),
        ("entity", "chemical_features"),
        ("entity", "strain_features"),
        ("entity", "chemical_registry"),
        ("entity", "strain_registry"),
        ("entity", "chemical_parent_views"),
        ("entity", "chemical_identity_risks"),
        ("graph", "artifact"),
    ):
        value = result.get(section, {}).get(key)
        if value:
            path = Path(value)
            result[section][key] = str((path if path.is_absolute() else base_dir / path).resolve())
    result.setdefault("runtime", {})["runs_dir"] = str(run_root.resolve())
    result.setdefault("model", {})["device"] = "cuda:0"
    return result


def _resolve_project_paths(payload: dict) -> dict:
    for section, key in (
        ("entity", "chemical_map"),
        ("entity", "chemical_features"),
        ("entity", "strain_features"),
        ("entity", "chemical_registry"),
        ("entity", "strain_registry"),
        ("entity", "chemical_parent_views"),
        ("entity", "chemical_identity_risks"),
        ("graph", "artifact"),
    ):
        value = payload.get(section, {}).get(key)
        if value and not Path(value).is_absolute():
            payload[section][key] = str((PROJECT / value).resolve())
    return payload


def _disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def _task_key(task: dict) -> str:
    return f"{task['id']}-S{task['seed']}"


def _expected_run_contract(
    config_path: Path,
    matrix: dict,
    task: dict,
    input_audit_cache: dict[str, dict[str, object]] | None = None,
) -> dict:
    """Rebuild the exact producer contract before reusing a completed run."""

    # The launcher is executed as a file, so the project source tree is not
    # necessarily importable unless the caller installed the package.  Producer
    # subprocesses receive the same path explicitly below.
    source_root = str(PROJECT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from goai_baseline.audit import audit_inputs
    from goai_response.config import load_response_config
    from goai_response.oof import _run_contract

    config = load_response_config(config_path)
    cache_key = str(config.baseline.path.resolve())
    input_audit = (
        input_audit_cache.get(cache_key)
        if input_audit_cache is not None
        else None
    )
    if input_audit is None:
        input_audit = audit_inputs(config.baseline)
        if input_audit_cache is not None:
            input_audit_cache[cache_key] = input_audit
    return _run_contract(
        config,
        input_audit,
        int(matrix.get("n_folds", 4)),
        int(matrix.get("fold_seed", 42)),
        int(task["seed"]),
        tuple(str(item) for item in task["scenarios"]),
    )


def _contract_section_drift(existing: dict, expected: dict) -> list[str]:
    """Name every immutable contract section that differs."""

    sections = {
        "protocol": ("protocol",),
        "config": ("response_config_sha256", "effective_config"),
        "inputs": ("input_hashes", "input_audit"),
        "artifacts": ("external_artifacts",),
        "folds_and_seeds": ("n_folds", "seed", "model_seed", "scenarios"),
        "source": ("source_fingerprint",),
    }
    return [
        label
        for label, fields in sections.items()
        if any(existing.get(field) != expected.get(field) for field in fields)
    ]


def _validate_completed_run(run_dir: Path, expected_contract: dict) -> dict:
    """Fail closed unless an existing summary belongs to this exact task.

    Merely finding ``oof_summary.csv`` is insufficient: code, effective config,
    source data, semantic artifacts, folds, and seeds may have changed since it
    was written.  The OOF manifest must also bind itself to the same contract.
    """

    required = ("oof_summary.csv", "run_contract.json", "oof_manifest.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(
            f"Refusing to reuse incomplete completed run {run_dir}: missing {missing}"
        )
    try:
        existing = json.loads((run_dir / "run_contract.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "oof_manifest.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Refusing to reuse unreadable completed run {run_dir}") from error
    if not isinstance(existing, dict) or not isinstance(manifest, dict):
        raise ValueError(f"Refusing to reuse malformed completed run {run_dir}")

    expected_fingerprint = str(expected_contract.get("fingerprint_sha256", ""))
    existing_fingerprint = str(existing.get("fingerprint_sha256", ""))
    drift = _contract_section_drift(existing, expected_contract)
    if not expected_fingerprint or existing_fingerprint != expected_fingerprint or drift:
        detail = ", ".join(drift or ["contract_fingerprint"])
        raise ValueError(
            "Refusing to reuse completed OOF run because its immutable contract "
            f"drifted ({detail}): {run_dir}"
        )
    if str(manifest.get("run_contract_fingerprint_sha256", "")) != expected_fingerprint:
        raise ValueError(
            "Refusing to reuse completed OOF run because its manifest is not bound "
            f"to the current contract: {run_dir}"
        )
    nested_enabled = bool(
        expected_contract.get("effective_config", {})
        .get("model", {})
        .get("nested_expert_scale_selection", False)
    )
    nested_receipts = 0
    if nested_enabled:
        source_root = str(PROJECT / "src")
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        from goai_response.nested_scale import validate_receipt

        completed_paths = sorted((run_dir / "folds").glob("*/completed.json"))
        if not completed_paths:
            raise ValueError(
                f"Completed formal run has no fold receipts: {run_dir}"
            )
        for completed_path in completed_paths:
            try:
                completed = json.loads(completed_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(
                    f"Unreadable formal fold completion: {completed_path}"
                ) from error
            receipt_hash = str(
                completed.get("nested_expert_scale_receipt_sha256", "")
            )
            if not receipt_hash:
                raise ValueError(
                    f"Formal fold lacks nested scale receipt: {completed_path}"
                )
            validate_receipt(
                completed_path.parent / "nested_expert_scale",
                expected_sha256=receipt_hash,
                expected_scenario=str(completed.get("scenario", "")),
                expected_fold=int(completed.get("fold", -1)),
                expected_train_ids_sha256=str(
                    completed.get("train_ids_sha256", "")
                ),
                expected_validation_ids_sha256=str(
                    completed.get("validation_ids_sha256", "")
                ),
                expected_source_contract_sha256=expected_fingerprint,
            )
            nested_receipts += 1
    return {
        "fingerprint_sha256": expected_fingerprint,
        "run_contract_sha256": _sha256(run_dir / "run_contract.json"),
        "oof_manifest_sha256": _sha256(run_dir / "oof_manifest.json"),
        "validated_sections": [
            "source",
            "config",
            "inputs",
            "artifacts",
            "folds_and_seeds",
            "manifest_binding",
            *( ["nested_inner_oof_scale_receipts"] if nested_enabled else []),
        ],
        "nested_inner_oof_scale_receipts": nested_receipts,
    }


def _write_environment(root: Path, matrix_path: Path) -> None:
    project_free = _disk_free_gb(PROJECT)
    run_root_free = _disk_free_gb(root)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "matrix": str(matrix_path.resolve()),
        "matrix_sha256": _sha256(matrix_path),
        "code_fingerprint_sha256": _code_fingerprint(),
        # Retain the historical key but make its target explicit.  The disk
        # guard itself follows run_root, which may live on a larger mount.
        "disk_free_gb_at_start": run_root_free,
        "disk_guard_path": str(root.resolve()),
        "run_root_path": str(root.resolve()),
        "run_root_free_gb_at_start": run_root_free,
        "project_path": str(PROJECT.resolve()),
        "project_free_gb_at_start": project_free,
        "note": "Runtime Python is retained for historical comparability even when it differs from pyproject minimum.",
    }
    _atomic_json(root / "environment.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--include", default="")
    parser.add_argument("--min-free-gb", type=float, default=25.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    matrix_path = Path(args.matrix).resolve()
    with matrix_path.open("r", encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle)
    if not isinstance(matrix, dict) or not isinstance(matrix.get("experiments"), list):
        raise ValueError("matrix must contain an experiments list")
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "configs").mkdir(exist_ok=True)
    (run_root / "logs").mkdir(exist_ok=True)
    (run_root / "producers").mkdir(exist_ok=True)
    _write_environment(run_root, matrix_path)

    base_path = Path(matrix["base_config"])
    if not base_path.is_absolute():
        base_path = (matrix_path.parent / base_path).resolve()
    with base_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    base = _absolute_config(base, base_path, run_root)
    selected = {item.strip() for item in args.include.split(",") if item.strip()}
    tasks: list[dict] = []
    ids: set[str] = set()
    for experiment in matrix["experiments"]:
        experiment_id = str(experiment["id"])
        if experiment_id in ids:
            raise ValueError(f"duplicate experiment id: {experiment_id}")
        ids.add(experiment_id)
        if selected and experiment_id not in selected:
            continue
        for seed in experiment.get("seeds", [42]):
            tasks.append({**experiment, "seed": int(seed)})
    tasks.sort(key=lambda item: (int(item.get("priority", 100)), str(item["id"]), int(item["seed"])))

    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    status = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "state": "running",
        "pending": [_task_key(task) for task in tasks],
        "running": {},
        "completed": [],
        "failed": [],
        "skipped": [],
    }
    _atomic_json(run_root / "batch_status.json", status)
    # Validate every reusable output before launching any subprocess.  A stale
    # task must fail the batch without leaving a sibling GPU process orphaned.
    input_audit_cache: dict[str, dict[str, object]] = {}
    for task in tasks:
        key = _task_key(task)
        config = _resolve_project_paths(
            _deep_update(deepcopy(base), task.get("overrides", {}))
        )
        config["model"]["seed"] = int(task["seed"])
        config_path = run_root / "configs" / f"{key}.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        task["_config_path"] = str(config_path)
        run_dir = run_root / "producers" / str(task["id"]) / f"S{task['seed']}"
        if not (run_dir / "oof_summary.csv").is_file():
            continue
        try:
            expected_contract = _expected_run_contract(
                config_path,
                matrix,
                task,
                input_audit_cache,
            )
            task["_completed_validation"] = _validate_completed_run(
                run_dir, expected_contract
            )
        except Exception as error:
            record = {
                "task": key,
                "reason": "completed_run_contract_rejected",
                "run_dir": str(run_dir),
                "error": str(error),
            }
            status["failed"].append(record)
            status["pending"] = [item for item in status["pending"] if item != key]
            status["state"] = "contract_rejected"
            status["finished_at"] = datetime.now().isoformat(timespec="seconds")
            _atomic_json(run_root / "batch_status.json", status)
            raise
    processes: dict[str, dict] = {}

    while tasks or processes:
        finished: list[str] = []
        for gpu, active in processes.items():
            code = active["process"].poll()
            if code is None:
                continue
            active["log_handle"].close()
            key = active["key"]
            status["running"].pop(key, None)
            record = {
                "task": key,
                "returncode": int(code),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "run_dir": str(active["run_dir"]),
                "log": str(active["log_path"]),
            }
            if code == 0 and (active["run_dir"] / "oof_summary.csv").is_file():
                status["completed"].append(record)
            else:
                status["failed"].append(record)
                _atomic_json(active["run_dir"] / "failure.json", record) if active["run_dir"].is_dir() else None
            finished.append(gpu)
        for gpu in finished:
            processes.pop(gpu)

        free_gpus = [gpu for gpu in gpus if gpu not in processes]
        while tasks and free_gpus:
            if (run_root / "STOP").exists():
                status["state"] = "stopping"
                status["skipped"].extend(_task_key(task) for task in tasks)
                tasks.clear()
                break
            free_gb = _disk_free_gb(run_root)
            if free_gb < args.min_free_gb:
                status["state"] = "disk_guard_stopped"
                status["skipped"].extend(_task_key(task) for task in tasks)
                status["disk_free_gb"] = free_gb
                tasks.clear()
                break
            task = tasks.pop(0)
            key = _task_key(task)
            run_dir = run_root / "producers" / str(task["id"]) / f"S{task['seed']}"
            config_path = Path(str(task["_config_path"]))
            validation = task.get("_completed_validation")
            if isinstance(validation, dict):
                status["skipped"].append(
                    {
                        "task": key,
                        "reason": "already_complete_contract_validated",
                        "run_dir": str(run_dir),
                        **validation,
                    }
                )
                status["pending"] = [item for item in status["pending"] if item != key]
                continue
            if run_dir.exists() and not (run_dir / "run_contract.json").is_file():
                status["failed"].append({"task": key, "reason": "existing_directory_without_contract", "run_dir": str(run_dir)})
                status["pending"] = [item for item in status["pending"] if item != key]
                continue
            gpu = free_gpus.pop(0)
            command = [
                args.python, "-m", "goai_response.oof",
                "--config", str(config_path),
                "--run-dir", str(run_dir),
                "--n-folds", str(matrix.get("n_folds", 4)),
                "--seed", str(matrix.get("fold_seed", 42)),
                "--model-seed", str(task["seed"]),
                "--scenarios", *[str(item) for item in task["scenarios"]],
                "--npz-only",
            ]
            if run_dir.exists():
                command.append("--resume")
            log_path = run_root / "logs" / f"{key}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(PROJECT / "src")
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["OMP_NUM_THREADS"] = str(matrix.get("cpu_threads_per_job", 8))
            env["MKL_NUM_THREADS"] = str(matrix.get("cpu_threads_per_job", 8))
            process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            processes[gpu] = {
                "process": process,
                "log_handle": log_handle,
                "log_path": log_path,
                "run_dir": run_dir,
                "key": key,
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
            status["pending"] = [item for item in status["pending"] if item != key]
            status["running"][key] = {"gpu": gpu, "pid": process.pid, "run_dir": str(run_dir), "log": str(log_path)}
        status["heartbeat_at"] = datetime.now().isoformat(timespec="seconds")
        status["disk_guard_path"] = str(run_root)
        status["disk_free_gb"] = _disk_free_gb(run_root)
        _atomic_json(run_root / "batch_status.json", status)
        if tasks or processes:
            time.sleep(args.poll_seconds)

    status["finished_at"] = datetime.now().isoformat(timespec="seconds")
    status["state"] = "complete" if not status["failed"] else "complete_with_failures"
    _atomic_json(run_root / "batch_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
