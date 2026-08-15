#!/usr/bin/env python3
"""Reconstruct the complete GOAI-M12.0 prediction from released checkpoints.

The script intentionally does not consume cached M5/M9 predictions.  It loads
the 15 final checkpoints, rebuilds each parent model, applies the frozen route
contract, and writes a fresh prediction plus an auditable receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from goai_baseline.schema import (
    SAMPLE_ID,
    SPLIT,
    require_metadata_columns,
    require_unique_sample_ids,
    treatment_mask,
)
from goai_baseline.submission import verify_submission
from goai_response.entities import load_json_with_hash, manifest_sha256
from goai_response.predict import load_response_checkpoint
from goai_response.routing import support_route_audit
from goai_response.train import _predict, _predict_core_components
from goai_rna_transfer.common import morgan_fingerprint
from goai_rna_transfer.goai_data import encode_context
from goai_rna_transfer.models import ChemicalEncoder
from goai_rna_transfer.train_residual_gate import (
    ChemicalResidualGate,
    frozen_context_outputs,
    predict_correction,
)
from goai_rna_transfer.train_s1 import build_model


SEEDS = (42, 43, 2026)
M5_HUBER_WEIGHTS = {
    "test_chem_only": 0.15,
    "test_strain_only": 0.0,
    "test_both": 0.0,
    "test_time": 0.30,
}
EXPECTED_ROUTE_COUNTS = {
    "R10": 2072,
    "R01": 1594,
    "R00": 425,
    "R11": 135,
    "control": 228,
}
M12_WEIGHT = 1.075
M12_THRESHOLD = 0.5
M12_TAKEOVER = 0.15


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_paths(root: Path, family: str) -> list[Path]:
    paths = [root / family / f"S{seed}" / "checkpoint.pt" for seed in SEEDS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {missing}")
    return paths


def average_response_predictions(
    checkpoints: list[Path],
    metadata: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, list[str], list[dict[str, object]]]:
    total: np.ndarray | None = None
    proteins: list[str] | None = None
    records: list[dict[str, object]] = []
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("fit_scope") != "all_released_labeled_rows":
            raise ValueError(f"Checkpoint has the wrong fit scope: {path}")
        model, builder, current_proteins, mean, scale = load_response_checkpoint(
            path, device
        )
        if proteins is None:
            proteins = current_proteins
        elif proteins != current_proteins:
            raise ValueError("Response checkpoints use different protein contracts")
        current = _predict(
            model,
            builder,
            metadata,
            metadata.index,
            current_proteins,
            mean,
            scale,
            device,
            batch_size,
        ).to_numpy(dtype=np.float64)
        total = current if total is None else total + current
        records.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "fit_sample_count": int(payload.get("fit_sample_count", 0)),
            }
        )
        del model, builder, current
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert total is not None and proteins is not None
    return (total / len(checkpoints)).astype(np.float32), proteins, records


def average_m6_components(
    checkpoints: list[Path],
    metadata: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], list[str], list[dict[str, object]]]:
    totals: dict[str, np.ndarray] | None = None
    proteins: list[str] | None = None
    records: list[dict[str, object]] = []
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("fit_scope") != "all_released_labeled_rows":
            raise ValueError(f"Checkpoint has the wrong fit scope: {path}")
        model, builder, current_proteins, mean, scale = load_response_checkpoint(
            path, device
        )
        if proteins is None:
            proteins = current_proteins
        elif proteins != current_proteins:
            raise ValueError("M6 checkpoints use different protein contracts")
        current = _predict_core_components(
            model,
            builder,
            metadata,
            metadata.index,
            mean,
            scale,
            device,
            batch_size,
        )
        reconstruct_error = float(
            np.max(
                np.abs(
                    current["background_plus_calibration"]
                    + current["is_treatment"] * current["response"]
                    - current["final"]
                )
            )
        )
        if reconstruct_error > 5e-5:
            raise ValueError(f"M6 component reconstruction failed: {path}")
        if totals is None:
            totals = {
                name: np.asarray(current[name], dtype=np.float64)
                for name in ("background_plus_calibration", "response", "final")
            }
        else:
            for name in totals:
                totals[name] += current[name]
        records.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "fit_sample_count": int(payload.get("fit_sample_count", 0)),
                "component_reconstruction_max_abs_error": reconstruct_error,
            }
        )
        del model, builder, current
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert totals is not None and proteins is not None
    return (
        {name: (values / len(checkpoints)).astype(np.float32) for name, values in totals.items()},
        proteins,
        records,
    )


def chemical_fingerprints(
    metadata: pd.DataFrame,
    chemical_map_path: Path,
    n_bits: int,
) -> np.ndarray:
    mapping = pd.read_csv(
        chemical_map_path, sep="\t", keep_default_na=False
    ).set_index("raw_name", verify_integrity=True)
    structure_column = (
        "canonical_smiles"
        if "canonical_smiles" in mapping.columns
        else "isomeric_smiles"
    )
    smiles = mapping[structure_column].astype(str).to_dict()
    is_treatment = treatment_mask(metadata).to_numpy(dtype=bool)
    names = metadata.loc[is_treatment, "perturbation_no_concentration"].astype(str)
    missing = sorted({name for name in names.unique() if not smiles.get(name)})
    if missing:
        raise ValueError(f"Treatment chemicals lack a structure: {missing}")
    by_name = {
        name: morgan_fingerprint(smiles[name], radius=2, n_bits=n_bits)
        for name in sorted(names.unique())
    }
    result = np.zeros((len(metadata), n_bits), dtype=np.float32)
    result[is_treatment] = np.stack(names.map(by_name).to_numpy()).astype(np.float32)
    return result


def predict_m9_checkpoint(
    path: Path,
    metadata: pd.DataFrame,
    chemical_map_path: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "base_model",
        "residual_model",
        "target_mean",
        "target_scale",
        "context_fields",
        "context_vocabulary",
        "context_cardinalities",
        "proteins",
        "residual_scale",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"M9 checkpoint lacks fields {missing}: {path}")
    proteins = [str(value) for value in payload["proteins"]]
    cardinalities = np.asarray(payload["context_cardinalities"], dtype=np.int64)
    architecture = {
        "fingerprint": {"n_bits": 2048},
        "rna_pretraining": {
            "encoder_hidden": 256,
            "encoder_dim": 64,
            "dropout": 0.20,
        },
        "goai_s1": {
            "context_dim": 64,
            "fusion_hidden": 256,
            "dropout": 0.15,
        },
    }
    base = build_model(
        architecture,
        cardinalities,
        len(proteins),
        checkpoint=None,
        freeze_encoder=True,
        device=device,
    )
    base.load_state_dict(payload["base_model"], strict=True)
    encoder = ChemicalEncoder(2048, 256, 64, 0.20)
    residual = ChemicalResidualGate(
        encoder,
        context_dim=64,
        chemical_dim=64,
        hidden=128,
        n_proteins=len(proteins),
        dropout=0.10,
    ).to(device)
    residual.load_state_dict(payload["residual_model"], strict=True)

    fields = [str(value) for value in payload["context_fields"]]
    vocabulary = {
        str(field): [str(value) for value in values]
        for field, values in payload["context_vocabulary"].items()
    }
    context = encode_context(metadata, fields, vocabulary)
    fingerprints = chemical_fingerprints(metadata, chemical_map_path, n_bits=2048)
    base_standardized, context_features = frozen_context_outputs(
        base, fingerprints, context, device, batch_size
    )
    correction_standardized = predict_correction(
        residual, fingerprints, context_features, device, batch_size
    )
    target_mean = np.asarray(payload["target_mean"], dtype=np.float32)
    target_scale = np.asarray(payload["target_scale"], dtype=np.float32)
    base_delta = base_standardized * target_scale + target_mean
    op3_delta = base_delta + float(payload["residual_scale"]) * (
        correction_standardized * target_scale
    )
    if not np.isfinite(base_delta).all() or not np.isfinite(op3_delta).all():
        raise ValueError(f"M9 produced a non-finite response: {path}")
    record = {
        "path": str(path),
        "sha256": sha256(path),
        "seed": int(payload.get("seed", -1)),
        "fit_treatment_count": int(payload.get("fit_treatment_count", 0)),
        "residual_scale": float(payload["residual_scale"]),
    }
    del base, residual
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        base_delta.astype(np.float32),
        op3_delta.astype(np.float32),
        proteins,
        record,
    )


def average_m9(
    checkpoints: list[Path],
    metadata: pd.DataFrame,
    chemical_map_path: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], list[str], list[dict[str, object]]]:
    base_total: np.ndarray | None = None
    op3_total: np.ndarray | None = None
    proteins: list[str] | None = None
    records: list[dict[str, object]] = []
    for path in checkpoints:
        base, op3, current_proteins, record = predict_m9_checkpoint(
            path, metadata, chemical_map_path, device, batch_size
        )
        if proteins is None:
            proteins = current_proteins
        elif proteins != current_proteins:
            raise ValueError("M9 checkpoints use different protein contracts")
        base_total = base.astype(np.float64) if base_total is None else base_total + base
        op3_total = op3.astype(np.float64) if op3_total is None else op3_total + op3
        records.append(record)
    assert base_total is not None and op3_total is not None and proteins is not None
    count = float(len(checkpoints))
    return (
        {
            "base": (base_total / count).astype(np.float32),
            "op3": (op3_total / count).astype(np.float32),
        },
        proteins,
        records,
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-test", type=Path, required=True)
    parser.add_argument(
        "--chemical-map",
        type=Path,
        default=root / "resources/entities/chemical_entity_map.tsv",
    )
    parser.add_argument(
        "--support-manifest",
        type=Path,
        default=root / "resources/entities/support_manifest_fit_all_labeled.json",
    )
    parser.add_argument("--weights-root", type=Path, default=root / "weights")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--skip-official-count-check",
        action="store_true",
        help="Allow non-official metadata whose route counts differ from the frozen contract.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    metadata_path = args.metadata_test.resolve()
    metadata = pd.read_csv(metadata_path, keep_default_na=False, low_memory=False)
    require_metadata_columns(metadata)
    require_unique_sample_ids(metadata, "test metadata")
    metadata = metadata.set_index(SAMPLE_ID, verify_integrity=True)
    unknown_splits = sorted(set(metadata[SPLIT].astype(str)) - set(M5_HUBER_WEIGHTS))
    if unknown_splits:
        raise ValueError(f"Unknown split_final values: {unknown_splits}")

    weights_root = args.weights_root.resolve()
    mse, proteins, mse_records = average_response_predictions(
        checkpoint_paths(weights_root, "m2/mse"), metadata, device, args.batch_size
    )
    huber, huber_proteins, huber_records = average_response_predictions(
        checkpoint_paths(weights_root, "m2/huber"), metadata, device, args.batch_size
    )
    if huber_proteins != proteins:
        raise ValueError("M2 MSE and Huber protein contracts differ")
    huber_weight = metadata[SPLIT].astype(str).map(M5_HUBER_WEIGHTS).to_numpy()[:, None]
    m5_1 = ((1.0 - huber_weight) * mse + huber_weight * huber).astype(np.float32)

    concat, concat_proteins, concat_records = average_m6_components(
        checkpoint_paths(weights_root, "m6/concat256"),
        metadata,
        device,
        args.batch_size,
    )
    film, film_proteins, film_records = average_m6_components(
        checkpoint_paths(weights_root, "m6/film256"),
        metadata,
        device,
        args.batch_size,
    )
    if concat_proteins != proteins or film_proteins != proteins:
        raise ValueError("M2/M6 protein contracts differ")
    m5_2 = m5_1.copy()
    split = metadata[SPLIT].astype(str)
    m5_2[split.eq("test_chem_only").to_numpy()] = concat["final"][
        split.eq("test_chem_only").to_numpy()
    ]
    m5_2[split.eq("test_time").to_numpy()] = film["final"][
        split.eq("test_time").to_numpy()
    ]

    m9, m9_proteins, m9_records = average_m9(
        checkpoint_paths(weights_root, "m9/op3_residual"),
        metadata,
        args.chemical_map.resolve(),
        device,
        args.batch_size,
    )
    if m9_proteins != proteins:
        raise ValueError("M6 and M9 protein contracts differ")

    support = load_json_with_hash(args.support_manifest.resolve())
    audit = support_route_audit(metadata, support)
    route_counts = {
        str(key): int(value)
        for key, value in audit["support_regime"].value_counts().items()
    }
    if not args.skip_official_count_check and route_counts != EXPECTED_ROUTE_COUNTS:
        raise ValueError(
            f"Support route counts differ from the frozen official contract: {route_counts}"
        )

    blend = (1.0 - M12_WEIGHT) * concat["response"] + M12_WEIGHT * m9["op3"]
    gate = np.abs(concat["response"]) >= M12_THRESHOLD
    response_m12 = blend + M12_TAKEOVER * gate * (concat["response"] - blend)
    r10_prediction = concat["background_plus_calibration"] + response_m12
    r10 = audit["support_regime"].astype(str).eq("R10").to_numpy()
    final = m5_2.copy()
    final[r10] = r10_prediction[r10]
    if not np.isfinite(final).all():
        raise ValueError("Final M12.0 prediction contains NaN or infinity")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(final, index=metadata.index, columns=proteins)
    frame.index.name = SAMPLE_ID
    frame.to_csv(output)
    audit = audit.copy()
    audit["selected_model"] = np.where(r10, "GOAI-M12.0", "GOAI-M5.2")
    audit["route_source"] = np.where(r10, "M6.11+M9.6", "M5.2 fallback")
    audit.reset_index(drop=True).to_csv(output.parent / "route_audit.csv", index=False)

    verification = verify_submission(output, metadata_path, proteins)
    receipt = {
        **verification,
        "model": "GOAI-M12.0",
        "status": "complete",
        "score_status": "local_strict_oof_selected_not_official",
        "formula": (
            "B6+C6+blend+0.15*I(abs(R6)>=0.5)*(R6-blend); "
            "blend=-0.075*R6+1.075*R9.6"
        ),
        "support_manifest_sha256": manifest_sha256(support),
        "route_counts": route_counts,
        "prediction_sha256": sha256(output),
        "components": {
            "m2_mse": mse_records,
            "m2_huber": huber_records,
            "m6_concat256": concat_records,
            "m6_film256": film_records,
            "m9_op3_residual": m9_records,
        },
        "official_submission_not_performed": True,
    }
    receipt_path = output.parent / "prediction_contract.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
