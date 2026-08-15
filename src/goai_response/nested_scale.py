"""Leakage-safe per-outer-fold expert-scale selection utilities.

Formal M7/M8 confirmation trains expert heads at the canonical scale ``1``.
Only predictions from inner OOF fits on the current outer-training rows may
choose the inference-time scales.  Discovery-wide scale screens are therefore
informative nominations, never formal hyper-parameters.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from .entities import load_json_with_hash, write_json_with_hash


PROTOCOL = "goai.nested_inner_oof_expert_scale.v1"
ALLOWED_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
METRICS = (
    "fc_pcc",
    "context_residual_pcc",
    "drug_residual_pcc",
    "high_effect_pcc",
    "high_effect_f1",
)
RELEVANT_RESIDUALS = {
    "R10": ("context_residual_pcc",),
    "R01": ("drug_residual_pcc",),
    "R11": ("context_residual_pcc", "drug_residual_pcc"),
    "RT": ("context_residual_pcc", "drug_residual_pcc"),
}
SCENARIO_AXES = {
    "R00": (),
    "R10": ("strain",),
    "R01": ("chemical",),
    "R11": ("strain", "chemical"),
    "RT": ("strain", "chemical", "pair"),
}
COMPONENT_NAMES = ("B_U", "B_s", "C_obs", "R_U", "R_s", "R_c", "R_sc")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(ids: pd.Index) -> str:
    return hashlib.sha256("\n".join(ids.astype(str)).encode("utf-8")).hexdigest()


def active_expert_axes(model: Any, scenario: str) -> tuple[str, ...]:
    """Return only the expert axes both identifiable and enabled in a regime."""

    if scenario not in SCENARIO_AXES:
        raise ValueError(f"Nested expert-scale selection does not support {scenario!r}")
    enabled = {
        "strain": bool(
            model.background_strain_expert_enabled
            or model.response_strain_expert_enabled
        ),
        "chemical": bool(model.response_chemical_expert_enabled),
        "pair": bool(model.response_pair_expert_enabled),
    }
    return tuple(axis for axis in SCENARIO_AXES[scenario] if enabled[axis])


def scale_grid(active_axes: tuple[str, ...]) -> list[dict[str, float]]:
    """Enumerate a small post-processing grid; no model is retrained per scale."""

    if any(axis not in {"strain", "chemical", "pair"} for axis in active_axes):
        raise ValueError(f"Unknown expert scale axes: {active_axes}")
    if not active_axes:
        return [{"strain": 0.0, "chemical": 0.0, "pair": 0.0}]
    rows: list[dict[str, float]] = []
    for values in itertools.product(ALLOWED_SCALES, repeat=len(active_axes)):
        current = {"strain": 0.0, "chemical": 0.0, "pair": 0.0}
        current.update(dict(zip(active_axes, map(float, values))))
        rows.append(current)
    return rows


def compose_scaled_prediction(
    components: Mapping[str, np.ndarray],
    treatment: np.ndarray,
    scales: Mapping[str, float],
) -> np.ndarray:
    """Recompose one natural-log2 prediction from canonical-scale components."""

    missing = sorted(set(COMPONENT_NAMES) - set(components))
    if missing:
        raise ValueError(f"Named prediction components are incomplete: {missing}")
    treatment_array = np.asarray(treatment, dtype=np.float32).reshape(-1, 1)
    base = (
        np.asarray(components["B_U"], dtype=np.float32)
        + float(scales.get("strain", 0.0))
        * np.asarray(components["B_s"], dtype=np.float32)
        + np.asarray(components["C_obs"], dtype=np.float32)
    )
    response = (
        np.asarray(components["R_U"], dtype=np.float32)
        + float(scales.get("strain", 0.0))
        * np.asarray(components["R_s"], dtype=np.float32)
        + float(scales.get("chemical", 0.0))
        * np.asarray(components["R_c"], dtype=np.float32)
        + float(scales.get("pair", 0.0))
        * np.asarray(components["R_sc"], dtype=np.float32)
    )
    result = base + treatment_array * response
    if not np.isfinite(result).all():
        raise ValueError("Scaled expert prediction contains non-finite values")
    return result


def predict_fit_components(
    fit: Any,
    metadata: pd.DataFrame,
    ids: pd.Index,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Export canonical-scale named components directly from one in-memory fit."""

    if fit.model.interaction_mode != "shared_general_experts":
        raise ValueError("Nested scale selection requires shared_general_experts")
    for name in (
        "strain_expert_scale",
        "chemical_expert_scale",
        "pair_expert_scale",
    ):
        if float(getattr(fit.model, name)) != 1.0:
            raise ValueError(
                "Nested scale selection requires canonical scale=1 training; "
                f"{name}={getattr(fit.model, name)}"
            )
    current = metadata.loc[ids]
    features = fit.builder.transform(current)
    response_prior = fit.builder.response_prior(current)
    collected = {name: [] for name in COMPONENT_NAMES}
    fit.model.eval()
    target_mean = torch.as_tensor(
        fit.target_mean, dtype=torch.float32, device=fit.device
    )
    target_scale = torch.as_tensor(
        fit.target_scale, dtype=torch.float32, device=fit.device
    )
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            output = fit.model.forward_named_components(
                torch.from_numpy(features.response[start:end]).to(fit.device),
                torch.from_numpy(features.background[start:end]).to(fit.device),
                torch.from_numpy(features.observation[start:end]).to(fit.device),
                torch.from_numpy(features.is_treatment[start:end]).to(fit.device),
                torch.from_numpy(features.cell[start:end]).to(fit.device),
                torch.from_numpy(features.perturbation[start:end]).to(fit.device),
                torch.from_numpy(response_prior[start:end]).to(fit.device),
                torch.from_numpy(features.general_cell[start:end]).to(fit.device),
                torch.from_numpy(features.general_perturbation[start:end]).to(fit.device),
                torch.from_numpy(features.strain_indices[start:end]).to(fit.device),
                torch.from_numpy(features.chemical_indices[start:end]).to(fit.device),
                torch.from_numpy(features.strain_seen[start:end]).to(fit.device),
                torch.from_numpy(features.chemical_seen[start:end]).to(fit.device),
                torch.from_numpy(features.pair_indices[start:end]).to(fit.device),
                torch.from_numpy(features.pair_seen[start:end]).to(fit.device),
            )
            values = {
                "B_U": output.background_universal * target_scale + target_mean,
                "B_s": output.background_strain * target_scale,
                "C_obs": output.calibration * target_scale,
                "R_U": (output.response_universal + output.response_prior)
                * target_scale,
                "R_s": output.response_strain * target_scale,
                "R_c": output.response_chemical * target_scale,
                "R_sc": output.response_pair * target_scale,
            }
            for name, value in values.items():
                collected[name].append(
                    value.detach().cpu().numpy().astype(np.float32, copy=False)
                )
    return {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}


def select_nested_scales(
    candidates: pd.DataFrame,
    scenario: str,
    active_axes: tuple[str, ...],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Apply the preregistered FC objective and biological guardrails."""

    required = {"strain_scale", "chemical_scale", "pair_scale", *METRICS}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Nested scale candidate table lacks columns: {missing}")
    if scenario not in RELEVANT_RESIDUALS:
        raise ValueError(f"No scale objective is defined for {scenario}")
    frame = candidates.copy()
    scale_columns = {
        "strain": "strain_scale",
        "chemical": "chemical_scale",
        "pair": "pair_scale",
    }
    for column in (*scale_columns.values(), *METRICS):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    baseline_mask = np.ones(len(frame), dtype=bool)
    for axis in active_axes:
        baseline_mask &= frame[scale_columns[axis]].eq(0.0).to_numpy()
    baseline_rows = frame.loc[baseline_mask]
    if len(baseline_rows) != 1:
        raise ValueError("Nested scale grid must contain exactly one all-zero baseline")
    baseline = baseline_rows.iloc[0]
    required_metrics = {
        "fc_pcc",
        "high_effect_pcc",
        "high_effect_f1",
        *RELEVANT_RESIDUALS[scenario],
    }
    if not np.isfinite([float(baseline[name]) for name in required_metrics]).all():
        raise ValueError("All-zero inner-OOF baseline has non-finite gate metrics")
    frame["fc_delta_vs_zero"] = frame["fc_pcc"] - float(baseline["fc_pcc"])
    guard_columns: list[str] = []
    for metric in RELEVANT_RESIDUALS[scenario]:
        column = f"{metric}_delta_vs_zero"
        frame[column] = frame[metric] - float(baseline[metric])
        guard_columns.append(column)
    for metric in ("high_effect_pcc", "high_effect_f1"):
        column = f"{metric}_delta_vs_zero"
        frame[column] = frame[metric] - float(baseline[metric])
        guard_columns.append(column)
    finite = np.isfinite(frame.loc[:, list(required_metrics)].to_numpy(dtype=float)).all(
        axis=1
    )
    residual_pass = frame[
        [f"{name}_delta_vs_zero" for name in RELEVANT_RESIDUALS[scenario]]
    ].ge(-1e-12).all(axis=1)
    high_pass = frame[
        ["high_effect_pcc_delta_vs_zero", "high_effect_f1_delta_vs_zero"]
    ].ge(-0.005 - 1e-12).all(axis=1)
    frame["guardrail_pass"] = finite & residual_pass & high_pass
    eligible = frame.loc[frame["guardrail_pass"]].copy()
    if eligible.empty:
        raise ValueError("No nested expert-scale candidate passed locked guardrails")
    # The FC objective is the only optimization target.  Exact ties choose the
    # lower scale lexicographically in the preregistered strain/chemical/pair
    # order, avoiding result-dependent secondary objectives.
    order = [scale_columns[axis] for axis in ("strain", "chemical", "pair") if axis in active_axes]
    best = eligible.sort_values(
        ["fc_pcc", *order], ascending=[False, *([True] * len(order))], kind="mergesort"
    ).iloc[0]
    selected = {
        axis: float(best[scale_columns[axis]])
        for axis in ("strain", "chemical", "pair")
    }
    frame["selected"] = False
    matches = np.ones(len(frame), dtype=bool)
    for axis, column in scale_columns.items():
        matches &= frame[column].eq(selected[axis]).to_numpy()
    frame.loc[matches, "selected"] = True
    if int(frame["selected"].sum()) != 1:
        raise RuntimeError("Nested scale selection did not identify one candidate")
    return selected, frame


def write_receipt(
    directory: str | Path,
    *,
    payload: dict[str, Any],
    assignments: pd.DataFrame,
    candidates: pd.DataFrame,
    fit_receipts: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Atomically publish all auditable inner-OOF evidence and its hash chain."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    assignments_path = destination / "inner_assignments.csv"
    candidates_path = destination / "candidate_metrics.csv"
    fits_path = destination / "inner_fit_receipts.json"
    assignments.to_csv(assignments_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    fits_hash = write_json_with_hash(fits_path, {"fits": fit_receipts})
    evidence_hashes = {
        "inner_assignments.csv": sha256_file(assignments_path),
        "candidate_metrics.csv": sha256_file(candidates_path),
        "inner_fit_receipts.json": fits_hash,
    }
    receipt = {**payload, "evidence_hashes": evidence_hashes}
    receipt_hash = write_json_with_hash(destination / "receipt.json", receipt)
    return receipt, receipt_hash


def validate_receipt(
    directory: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_scenario: str | None = None,
    expected_fold: int | None = None,
    expected_train_ids_sha256: str | None = None,
    expected_validation_ids_sha256: str | None = None,
    expected_source_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed on missing, global, stale, or tampered nested evidence."""

    source = Path(directory)
    receipt = load_json_with_hash(source / "receipt.json", expected_sha256)
    if receipt.get("protocol") != PROTOCOL:
        raise ValueError("Nested scale receipt has an unsupported protocol")
    if receipt.get("global_scale_used") is not False:
        raise ValueError("Formal confirmation cannot use a global expert scale")
    if receipt.get("outer_validation_labels_used") is not False:
        raise ValueError("Nested scale receipt used outer validation labels")
    canonical = receipt.get("canonical_training_scales")
    if canonical != {"strain": 1.0, "chemical": 1.0, "pair": 1.0}:
        raise ValueError("Nested scale inner models were not trained at scale 1")
    expected = {
        "scenario": expected_scenario,
        "outer_fold": expected_fold,
        "outer_train_ids_sha256": expected_train_ids_sha256,
        "outer_validation_ids_sha256": expected_validation_ids_sha256,
        "source_contract_fingerprint_sha256": expected_source_contract_sha256,
    }
    for key, value in expected.items():
        if value is not None and receipt.get(key) != value:
            raise ValueError(f"Nested scale receipt {key} does not match outer fold")
    status = str(receipt.get("status", ""))
    if status not in {"selected", "not_applicable"}:
        raise ValueError(f"Nested scale receipt is not effective: {status!r}")
    axes = tuple(receipt.get("active_axes", ()))
    if status == "selected":
        if int(receipt.get("inner_n_folds", 0)) < 2 or not axes:
            raise ValueError("Selected nested scale lacks at least two inner folds")
        selected = receipt.get("selected_scales")
        if not isinstance(selected, dict) or set(selected) != {
            "strain",
            "chemical",
            "pair",
        }:
            raise ValueError("Nested scale receipt lacks all selected scales")
        if any(float(value) not in ALLOWED_SCALES for value in selected.values()):
            raise ValueError("Nested scale receipt contains an unlocked scale")
    elif axes:
        raise ValueError("Applicable nested scale receipt cannot be not_applicable")
    evidence = receipt.get("evidence_hashes")
    expected_files = {
        "inner_assignments.csv",
        "candidate_metrics.csv",
        "inner_fit_receipts.json",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected_files:
        raise ValueError("Nested scale receipt evidence hash chain is incomplete")
    for name, expected_hash in evidence.items():
        path = source / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Nested scale receipt evidence was tampered: {name}")
    fits = load_json_with_hash(
        source / "inner_fit_receipts.json", evidence["inner_fit_receipts.json"]
    )
    fit_rows = fits.get("fits")
    if not isinstance(fit_rows, list):
        raise ValueError("Nested scale fit receipt list is malformed")
    if status == "selected":
        if not fit_rows:
            raise ValueError("Selected nested scale has no inner fit receipts")
        for item in fit_rows:
            if not isinstance(item, dict) or not all(
                str(item.get(name, ""))
                for name in (
                    "train_ids_sha256",
                    "validation_ids_sha256",
                    "support_manifest_sha256",
                    "artifact_chain_sha256",
                    "source_contract_fingerprint_sha256",
                )
            ):
                raise ValueError("Nested scale inner fit support/source receipt is incomplete")
            if (
                expected_source_contract_sha256 is not None
                and item["source_contract_fingerprint_sha256"]
                != expected_source_contract_sha256
            ):
                raise ValueError("Nested scale inner fit source contract drifted")
    return receipt
