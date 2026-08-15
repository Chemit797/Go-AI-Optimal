"""Create a support-routed complete GOAI submission from M6 and M9 responses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from goai_baseline.schema import SAMPLE_ID, require_metadata_columns, require_unique_sample_ids
from goai_baseline.submission import verify_submission
from goai_response.config import load_response_config
from goai_response.entities import load_json_with_hash, manifest_sha256
from goai_response.predict import load_response_checkpoint
from goai_response.routing import support_route_audit
from goai_response.train import _predict_core_components


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _average_m6_components(
    config_path: Path,
    runs: list[Path],
    metadata: pd.DataFrame,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[str], list[dict[str, object]]]:
    config = load_response_config(config_path)
    totals: dict[str, np.ndarray] | None = None
    proteins: list[str] | None = None
    records = []
    for run in runs:
        checkpoint = run / "checkpoint.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("fit_scope") != "all_released_labeled_rows":
            raise ValueError(f"M6 checkpoint is not an all-label refit: {checkpoint}")
        model, builder, current_proteins, mean, scale = load_response_checkpoint(
            checkpoint, device, config
        )
        if proteins is None:
            proteins = current_proteins
        elif proteins != current_proteins:
            raise ValueError("M6 seed checkpoints use different protein contracts")
        current = _predict_core_components(
            model,
            builder,
            metadata,
            metadata.index,
            mean,
            scale,
            device,
            config.model.batch_size,
        )
        if totals is None:
            totals = {
                name: np.asarray(current[name], dtype=np.float64)
                for name in ("background_plus_calibration", "response", "final")
            }
        else:
            for name in totals:
                totals[name] += current[name]
        error = float(
            np.max(
                np.abs(
                    current["background_plus_calibration"]
                    + current["is_treatment"] * current["response"]
                    - current["final"]
                )
            )
        )
        if error > 5e-5:
            raise ValueError(f"M6 checkpoint component reconstruction failed: {error}")
        records.append(
            {
                "run": str(run),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "fit_scope": payload.get("fit_scope"),
                "fit_sample_count": int(payload.get("fit_sample_count", 0)),
                "component_reconstruction_max_abs_error": error,
            }
        )
        del model, builder, current
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert totals is not None and proteins is not None
    return (
        {name: (values / len(runs)).astype(np.float32) for name, values in totals.items()},
        proteins,
        records,
    )


def _average_m9(
    runs: list[Path],
    sample_ids: pd.Index,
    proteins: list[str],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    totals = {
        "m9_base_delta": np.zeros((len(sample_ids), len(proteins)), dtype=np.float64),
        "m9_op3_delta": np.zeros((len(sample_ids), len(proteins)), dtype=np.float64),
    }
    records = []
    for run in runs:
        path = run / "test_response.npz"
        with np.load(path, allow_pickle=False) as payload:
            current_ids = pd.Index(payload["sample_ids"].astype(str))
            current_proteins = payload["proteins"].astype(str).tolist()
            if not current_ids.equals(sample_ids) or current_proteins != proteins:
                raise ValueError(f"M9 test response contract differs: {path}")
            treatment = np.asarray(payload["is_treatment"], dtype=np.float32)
            if treatment.shape != (len(sample_ids), 1):
                raise ValueError(f"M9 treatment gate has an invalid shape: {path}")
            for name in totals:
                values = np.asarray(payload[name], dtype=np.float32)
                if values.shape != totals[name].shape or not np.isfinite(values).all():
                    raise ValueError(f"M9 array {name} is invalid: {path}")
                totals[name] += values
        manifest_path = run / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("fit_scope") != "all_released_labeled_rows":
            raise ValueError(f"M9 refit has the wrong fit scope: {manifest_path}")
        records.append(
            {
                "run": str(run),
                "prediction": str(path),
                "prediction_sha256": _sha256(path),
                "checkpoint_sha256": manifest.get("checkpoint_sha256"),
                "seed": int(manifest["seed"]),
                "fit_treatment_count": int(manifest["fit_treatment_count"]),
            }
        )
    return (
        {name: (values / len(runs)).astype(np.float32) for name, values in totals.items()},
        records,
    )


def _fuse_response(
    mode: str,
    weight: float,
    m6_response: np.ndarray,
    m9_op3_delta: np.ndarray,
    m9_base_delta: np.ndarray,
    specialist_threshold: float,
    specialist_takeover: float,
) -> tuple[np.ndarray, str]:
    if mode in {"blend", "high_specialist"}:
        fused = (1.0 - weight) * m6_response + weight * m9_op3_delta
        if mode == "high_specialist":
            gate = np.abs(m6_response) >= specialist_threshold
            fused = fused + specialist_takeover * gate * (m6_response - fused)
            formula = (
                "B6+C6+blend+takeover*I(abs(R6)>=threshold)*(R6-blend)"
            )
        else:
            formula = "B6+C6+(1-w)*R6+w*R9.6"
    elif mode == "op3_residual":
        fused = m6_response + weight * (m9_op3_delta - m9_base_delta)
        formula = "B6+C6+R6+g*(R9.6-R9.0)"
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")
    return fused.astype(np.float32), formula


def _semantic_shrink(
    current: np.ndarray, semantic: np.ndarray, scale: float
) -> np.ndarray:
    return (current + scale * (semantic - current)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-prediction", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--support-manifest", required=True)
    parser.add_argument("--m6-config", required=True)
    parser.add_argument("--m6-run", action="append", required=True)
    parser.add_argument("--m9-run", action="append", required=True)
    parser.add_argument(
        "--mode", choices=("blend", "op3_residual", "high_specialist"), default="blend"
    )
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--specialist-threshold", type=float, default=0.75)
    parser.add_argument("--specialist-takeover", type=float, default=0.25)
    parser.add_argument("--route", action="append", required=True, choices=("R00", "R01", "R10", "R11"))
    parser.add_argument("--expert-general-config")
    parser.add_argument("--expert-general-run", action="append", default=[])
    parser.add_argument("--expert-config")
    parser.add_argument("--expert-run", action="append", default=[])
    parser.add_argument("--expert-scale", type=float, default=1.0)
    parser.add_argument("--expert-residual-route", action="append", default=[], choices=("R00", "R01", "R10", "R11"))
    parser.add_argument("--expert-full-route", action="append", default=[], choices=("R00", "R01", "R10", "R11"))
    parser.add_argument("--semantic-config")
    parser.add_argument("--semantic-run", action="append", default=[])
    parser.add_argument("--semantic-fusion-route", action="append", default=[], choices=("R00", "R01", "R10", "R11"))
    parser.add_argument("--semantic-full-route", action="append", default=[], choices=("R00", "R01", "R10", "R11"))
    parser.add_argument("--semantic-residual-route", action="append", default=[], choices=("R00", "R01", "R10", "R11"))
    parser.add_argument("--semantic-scale", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-id", default="GOAI-M11.0")
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    if not np.isfinite(args.weight) or args.weight < 0.0:
        raise ValueError("Fusion weight must be finite and non-negative")
    if not np.isfinite(args.specialist_threshold) or args.specialist_threshold < 0.0:
        raise ValueError("Specialist threshold must be finite and non-negative")
    if not np.isfinite(args.specialist_takeover) or not 0.0 <= args.specialist_takeover <= 1.0:
        raise ValueError("Specialist takeover must be finite and in [0, 1]")
    if not np.isfinite(args.expert_scale) or args.expert_scale < 0.0:
        raise ValueError("Expert scale must be finite and non-negative")
    if not np.isfinite(args.semantic_scale) or not 0.0 <= args.semantic_scale <= 1.0:
        raise ValueError("Semantic scale must be finite and in [0, 1]")
    expert_requested = bool(args.expert_residual_route or args.expert_full_route)
    expert_inputs = bool(
        args.expert_general_config
        and args.expert_general_run
        and args.expert_config
        and args.expert_run
    )
    if expert_requested != expert_inputs:
        raise ValueError(
            "Expert routing requires both general/expert configs and non-empty run lists"
        )
    semantic_requested = bool(
        args.semantic_fusion_route
        or args.semantic_full_route
        or args.semantic_residual_route
    )
    semantic_inputs = bool(args.semantic_config and args.semantic_run)
    if semantic_requested != semantic_inputs:
        raise ValueError(
            "Semantic routing requires a semantic config and a non-empty run list"
        )
    direct_route_groups = {
        "m6_m9_fusion": set(args.route),
        "semantic_fusion": set(args.semantic_fusion_route),
        "semantic_full": set(args.semantic_full_route),
        "semantic_residual": set(args.semantic_residual_route),
        "expert_full": set(args.expert_full_route),
    }
    direct_route_names = list(direct_route_groups)
    for left_index, left_name in enumerate(direct_route_names):
        for right_name in direct_route_names[left_index + 1 :]:
            overlap = direct_route_groups[left_name] & direct_route_groups[right_name]
            if overlap:
                raise ValueError(
                    f"Direct route assignments overlap for {left_name}/{right_name}: {sorted(overlap)}"
                )

    metadata_path = Path(args.metadata).resolve()
    metadata = pd.read_csv(metadata_path, keep_default_na=False, low_memory=False)
    require_metadata_columns(metadata)
    require_unique_sample_ids(metadata, "test metadata")
    metadata = metadata.set_index(SAMPLE_ID, verify_integrity=True)
    base_path = Path(args.base_prediction).resolve()
    base = pd.read_csv(base_path).set_index(SAMPLE_ID, verify_integrity=True)
    if not base.index.equals(metadata.index):
        raise ValueError("Base prediction row order differs from test metadata")

    support_path = Path(args.support_manifest).resolve()
    support = load_json_with_hash(support_path)
    audit = support_route_audit(metadata, support)
    expected_counts = {"R00": 425, "R01": 1594, "R10": 2072, "R11": 135, "control": 228}
    actual_counts = audit["support_regime"].value_counts().to_dict()
    if actual_counts != expected_counts:
        raise ValueError(f"Unexpected full-refit support routing: {actual_counts}")

    device = torch.device(args.device)
    m6, proteins, m6_records = _average_m6_components(
        Path(args.m6_config).resolve(),
        [Path(path).resolve() for path in args.m6_run],
        metadata,
        device,
    )
    if base.columns.tolist() != proteins:
        raise ValueError("M6 protein order differs from the frozen base submission")
    m9, m9_records = _average_m9(
        [Path(path).resolve() for path in args.m9_run], metadata.index, proteins
    )
    fused_response, formula = _fuse_response(
        args.mode,
        args.weight,
        m6["response"],
        m9["m9_op3_delta"],
        m9["m9_base_delta"],
        args.specialist_threshold,
        args.specialist_takeover,
    )
    routed_prediction = m6["background_plus_calibration"] + fused_response
    if not np.isfinite(routed_prediction).all():
        raise ValueError("Routed M6/M9 prediction contains NaN or infinity")

    route_values = audit["support_regime"].astype(str)
    selected_routes = set(args.route)
    selected = route_values.isin(selected_routes).to_numpy()
    if not selected.any():
        raise ValueError("No test rows match the selected support routes")
    values = base.to_numpy(dtype=np.float32, copy=True)
    values[selected] = routed_prediction[selected]
    route_source = np.full(len(metadata), "GOAI-M5.2", dtype=object)
    route_source[selected] = "m6_m9_fusion"

    semantic_records: list[dict[str, object]] = []
    semantic_routes = (
        set(args.semantic_fusion_route)
        | set(args.semantic_full_route)
        | set(args.semantic_residual_route)
    )
    if semantic_requested:
        semantic, semantic_proteins, semantic_records = _average_m6_components(
            Path(args.semantic_config).resolve(),
            [Path(path).resolve() for path in args.semantic_run],
            metadata,
            device,
        )
        if semantic_proteins != proteins:
            raise ValueError("Semantic checkpoints use a different protein contract")
        semantic_fusion_mask = route_values.isin(args.semantic_fusion_route).to_numpy()
        semantic_full_mask = route_values.isin(args.semantic_full_route).to_numpy()
        semantic_residual_mask = route_values.isin(args.semantic_residual_route).to_numpy()
        values[semantic_fusion_mask] = (
            semantic["background_plus_calibration"][semantic_fusion_mask]
            + fused_response[semantic_fusion_mask]
        )
        values[semantic_full_mask] = semantic["final"][semantic_full_mask]
        values[semantic_residual_mask] = _semantic_shrink(
            values[semantic_residual_mask],
            semantic["final"][semantic_residual_mask],
            args.semantic_scale,
        )
        route_source[semantic_fusion_mask] = "semantic_background_m6_m9_response"
        route_source[semantic_full_mask] = "semantic_full"
        route_source[semantic_residual_mask] = "semantic_residual"

    expert_general_records: list[dict[str, object]] = []
    expert_records: list[dict[str, object]] = []
    expert_routes = set(args.expert_full_route) | set(args.expert_residual_route)
    if expert_requested:
        expert_general, expert_general_proteins, expert_general_records = _average_m6_components(
            Path(args.expert_general_config).resolve(),
            [Path(path).resolve() for path in args.expert_general_run],
            metadata,
            device,
        )
        expert, expert_proteins, expert_records = _average_m6_components(
            Path(args.expert_config).resolve(),
            [Path(path).resolve() for path in args.expert_run],
            metadata,
            device,
        )
        if expert_general_proteins != proteins or expert_proteins != proteins:
            raise ValueError("Expert checkpoints use a different protein contract")
        expert_full_mask = route_values.isin(args.expert_full_route).to_numpy()
        expert_residual_mask = route_values.isin(args.expert_residual_route).to_numpy()
        values[expert_full_mask] = expert["final"][expert_full_mask]
        route_source[expert_full_mask] = "expert_full"
        expert_delta = expert["final"] - expert_general["final"]
        values[expert_residual_mask] += args.expert_scale * expert_delta[expert_residual_mask]
        route_source[expert_residual_mask] = np.char.add(
            route_source[expert_residual_mask].astype(str), "+expert_residual"
        )

    all_overridden_routes = selected_routes | semantic_routes | expert_routes
    all_overridden = route_values.isin(all_overridden_routes).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Final prediction contains NaN or infinity")

    output = Path(args.output_csv).resolve()
    output.parent.mkdir(parents=True, exist_ok=False)
    result = pd.DataFrame(values, index=metadata.index, columns=proteins)
    result.index.name = SAMPLE_ID
    result.to_csv(output)
    audit = audit.copy()
    audit["selected_model"] = np.where(all_overridden, args.model_id, "GOAI-M5.2")
    audit["route_source"] = route_source
    audit["fusion_mode"] = np.where(all_overridden, args.mode, "none")
    audit["fusion_weight"] = np.where(all_overridden, args.weight, 0.0)
    audit["specialist_threshold"] = np.where(
        all_overridden & (args.mode == "high_specialist"), args.specialist_threshold, 0.0
    )
    audit["specialist_takeover"] = np.where(
        all_overridden & (args.mode == "high_specialist"), args.specialist_takeover, 0.0
    )
    audit["expert_scale"] = np.where(
        route_values.isin(args.expert_residual_route), args.expert_scale, 0.0
    )
    audit["semantic_scale"] = np.where(
        route_values.isin(args.semantic_residual_route), args.semantic_scale, 0.0
    )
    audit.reset_index(drop=True).to_csv(output.parent / "route_audit.csv", index=False)

    verification = verify_submission(output, metadata_path, proteins)
    contract = {
        **verification,
        "status": "complete",
        "model": args.model_id,
        "score_status": "local_strict_oof_selected_not_official",
        "base_prediction": str(base_path),
        "base_prediction_sha256": _sha256(base_path),
        "support_manifest": str(support_path),
        "support_manifest_sha256": manifest_sha256(support),
        "support_route_counts": actual_counts,
        "overridden_routes": sorted(all_overridden_routes),
        "overridden_rows": int(all_overridden.sum()),
        "fusion": {
            "mode": args.mode,
            "weight": args.weight,
            "specialist_threshold": args.specialist_threshold,
            "specialist_takeover": args.specialist_takeover,
            "formula": formula,
        },
        "route_composition": {
            "m6_m9_fusion": sorted(selected_routes),
            "semantic_background_m6_m9_response": sorted(args.semantic_fusion_route),
            "semantic_full": sorted(args.semantic_full_route),
            "semantic_residual": sorted(args.semantic_residual_route),
            "semantic_scale": args.semantic_scale,
            "expert_full": sorted(args.expert_full_route),
            "expert_residual": sorted(args.expert_residual_route),
            "expert_scale": args.expert_scale,
        },
        "m6_config": str(Path(args.m6_config).resolve()),
        "m6_config_sha256": _sha256(Path(args.m6_config)),
        "m6_checkpoints": m6_records,
        "m9_refits": m9_records,
        "semantic_config": str(Path(args.semantic_config).resolve()) if semantic_requested else None,
        "semantic_checkpoints": semantic_records,
        "expert_general_config": str(Path(args.expert_general_config).resolve()) if expert_requested else None,
        "expert_general_checkpoints": expert_general_records,
        "expert_config": str(Path(args.expert_config).resolve()) if expert_requested else None,
        "expert_checkpoints": expert_records,
        "route_audit": str(output.parent / "route_audit.csv"),
        "prediction": str(output),
        "prediction_sha256": _sha256(output),
        "official_submission_not_performed": True,
    }
    with (output.parent / "prediction_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, ensure_ascii=False, indent=2)
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
