"""Refit the M9.6 response model on all released labels and predict GOAI test rows."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import load_config, morgan_fingerprint, seed_everything, sha256, write_json
from .goai_data import (
    CHEMICAL,
    CONTROLS,
    MATCH_FIELDS,
    QUALITY_CONTROL,
    SAMPLE_ID,
    encode_context,
    fit_context_vocab,
    normalized_name,
)
from .models import ChemicalEncoder
from .train_residual_gate import (
    ChemicalResidualGate,
    fit_residual,
    frozen_context_outputs,
    predict_correction,
)
from .train_s1 import build_model, fit, masked_location_scale


def _treatment_mask(metadata: pd.DataFrame) -> np.ndarray:
    names = normalized_name(metadata[CHEMICAL])
    return ~names.isin(CONTROLS | {QUALITY_CONTROL}).to_numpy()


def _load_log2_proteome(
    metadata: pd.DataFrame,
    proteome_path: Path,
    proteins: np.ndarray,
) -> pd.DataFrame:
    usecols = [SAMPLE_ID, *proteins.astype(str).tolist()]
    raw = pd.read_csv(proteome_path, usecols=usecols).set_index(SAMPLE_ID)
    raw = raw.reindex(metadata.index).to_numpy(dtype=np.float64)
    observed = np.isfinite(raw) & (raw > 0)
    values = np.full(raw.shape, np.nan, dtype=np.float32)
    values[observed] = np.log2(raw[observed]).astype(np.float32)
    return pd.DataFrame(values, index=metadata.index, columns=proteins.astype(str))


def _matched_delta(
    metadata: pd.DataFrame,
    log2: pd.DataFrame,
) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    treatment_ids = metadata.index[_treatment_mask(metadata)]
    names = normalized_name(metadata[CHEMICAL])
    control_ids = metadata.index[names.isin(CONTROLS).to_numpy()]
    control_keys = pd.MultiIndex.from_frame(
        metadata.loc[control_ids, list(MATCH_FIELDS)].astype(str)
    )
    control_values = log2.loc[control_ids].copy()
    control_values.index = control_keys
    control_mean = control_values.groupby(
        level=list(range(len(MATCH_FIELDS))), sort=False
    ).mean()
    treatment_keys = pd.MultiIndex.from_frame(
        metadata.loc[treatment_ids, list(MATCH_FIELDS)].astype(str)
    )
    matched = control_mean.reindex(treatment_keys)
    matched.index = treatment_ids
    delta = (log2.loc[treatment_ids] - matched).to_numpy(dtype=np.float32)
    return treatment_ids, np.nan_to_num(delta, nan=0.0), np.isfinite(delta)


def _fingerprints(
    metadata: pd.DataFrame,
    treatment_rows: np.ndarray,
    chemical_map_path: Path,
    config: dict,
) -> np.ndarray:
    mapping = pd.read_csv(
        chemical_map_path, sep="\t", keep_default_na=False
    ).set_index("raw_name")
    smiles = mapping["canonical_smiles"].astype(str).to_dict()
    result = np.zeros(
        (len(metadata), int(config["fingerprint"]["n_bits"])), dtype=np.float32
    )
    names = metadata.iloc[np.flatnonzero(treatment_rows)][CHEMICAL].astype(str)
    missing = sorted({name for name in names.unique() if not smiles.get(name)})
    if missing:
        raise ValueError(f"Treatment chemicals lack structures: {missing}")
    by_name = {
        name: morgan_fingerprint(
            smiles[name],
            int(config["fingerprint"]["radius"]),
            int(config["fingerprint"]["n_bits"]),
        )
        for name in sorted(names.unique())
    }
    result[treatment_rows] = np.stack(names.map(by_name).to_numpy()).astype(np.float32)
    return result


def run(args: argparse.Namespace) -> Path:
    started = time.time()
    seed_everything(args.seed)
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    paths = config["paths"]
    metadata_path = Path(paths["goai_metadata"]).resolve()
    proteome_path = Path(paths["goai_proteome"]).resolve()
    chemical_map_path = Path(paths["goai_chemical_map"]).resolve()
    test_metadata_path = Path(args.metadata_test).resolve()
    encoder_path = Path(args.encoder_checkpoint).resolve()
    if not encoder_path.is_file():
        raise FileNotFoundError(encoder_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    cache_path = Path(paths["private_cache"]) / "goai_s1_delta.npz"
    with np.load(cache_path, allow_pickle=False) as payload:
        proteins = payload["proteins"].astype(str)
    metadata = pd.read_csv(metadata_path, keep_default_na=False).set_index(
        SAMPLE_ID, drop=False, verify_integrity=True
    )
    test_metadata = pd.read_csv(test_metadata_path, keep_default_na=False).set_index(
        SAMPLE_ID, drop=False, verify_integrity=True
    )
    log2 = _load_log2_proteome(metadata, proteome_path, proteins)
    treatment_ids, delta, mask = _matched_delta(metadata, log2)
    fit_rows = mask.any(axis=1)
    if not fit_rows.any():
        raise ValueError("No exact-control treatment delta is available for full refit")

    fields = list(config["goai_s1"]["context_fields"])
    vocabulary = fit_context_vocab(metadata, fields)
    cardinalities = np.asarray(
        [len(vocabulary[field]) + 1 for field in fields], dtype=np.int64
    )
    train_context = encode_context(metadata.loc[treatment_ids], fields, vocabulary)
    test_context = encode_context(test_metadata, fields, vocabulary)
    train_treatment = np.ones(len(treatment_ids), dtype=bool)
    test_treatment = _treatment_mask(test_metadata)
    train_fp = _fingerprints(
        metadata.loc[treatment_ids], train_treatment, chemical_map_path, config
    )
    test_fp = _fingerprints(test_metadata, test_treatment, chemical_map_path, config)

    target_mean, target_scale = masked_location_scale(delta[fit_rows], mask[fit_rows])
    standardized = ((delta - target_mean) / target_scale).astype(np.float32)
    standardized[~mask] = 0.0
    base = build_model(
        config,
        cardinalities,
        len(proteins),
        checkpoint=None,
        freeze_encoder=True,
        device=device,
    )
    base_history = fit(
        base,
        np.zeros_like(train_fp[fit_rows]),
        train_context[fit_rows],
        standardized[fit_rows],
        mask[fit_rows],
        delta[fit_rows],
        config,
        device,
        args.seed,
        pretrained=False,
        epochs=args.base_epochs,
    )
    base_train, context_train = frozen_context_outputs(
        base,
        train_fp,
        train_context,
        device,
        int(config["goai_s1"]["batch_size"]),
    )
    base_test, context_test = frozen_context_outputs(
        base,
        test_fp,
        test_context,
        device,
        int(config["goai_s1"]["batch_size"]),
    )

    encoder_payload = torch.load(encoder_path, map_location="cpu", weights_only=False)
    declared = str(encoder_payload.get("arm", "")).casefold()
    if "shuffle" in declared:
        raise ValueError("M9.6 full refit cannot use a shuffled RNA encoder")
    encoder = ChemicalEncoder(
        int(config["fingerprint"]["n_bits"]),
        int(config["rna_pretraining"]["encoder_hidden"]),
        int(config["rna_pretraining"]["encoder_dim"]),
        float(config["rna_pretraining"]["dropout"]),
    )
    encoder.load_state_dict(encoder_payload["chemical_encoder"], strict=True)
    residual = ChemicalResidualGate(
        encoder,
        context_train.shape[1],
        int(config["rna_pretraining"]["encoder_dim"]),
        hidden=128,
        n_proteins=len(proteins),
        dropout=0.10,
    ).to(device)
    residual_history = fit_residual(
        residual,
        train_fp[fit_rows],
        context_train[fit_rows],
        base_train[fit_rows],
        standardized[fit_rows],
        mask[fit_rows],
        delta[fit_rows],
        device,
        args.residual_epochs,
        int(config["goai_s1"]["batch_size"]),
        float(config["goai_s1"]["learning_rate"]),
        float(config["goai_s1"]["weight_decay"]),
        float(config["goai_s1"]["high_effect_weight"]),
        args.residual_penalty,
        args.seed,
    )
    correction_test = predict_correction(
        residual,
        test_fp,
        context_test,
        device,
        int(config["goai_s1"]["batch_size"]),
    )
    base_delta_test = base_test * target_scale + target_mean
    correction_delta_test = correction_test * target_scale
    op3_delta_test = base_delta_test + args.residual_scale * correction_delta_test
    for values in (base_delta_test, correction_delta_test, op3_delta_test):
        if values.shape != (len(test_metadata), len(proteins)) or not np.isfinite(values).all():
            raise ValueError("M9 full-refit test response has an invalid shape or value")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "test_response.npz"
    np.savez_compressed(
        prediction_path,
        sample_ids=np.asarray(test_metadata.index.astype(str), dtype=str),
        proteins=proteins,
        is_treatment=test_treatment.astype(np.float32).reshape(-1, 1),
        m9_base_delta=base_delta_test.astype(np.float32),
        op3_correction_delta=correction_delta_test.astype(np.float32),
        m9_op3_delta=op3_delta_test.astype(np.float32),
        residual_scale=np.asarray([args.residual_scale], dtype=np.float32),
    )
    checkpoint_path = output / "checkpoint.pt"
    torch.save(
        {
            "base_model": copy.deepcopy(base).cpu().state_dict(),
            "residual_model": copy.deepcopy(residual).cpu().state_dict(),
            "target_mean": target_mean,
            "target_scale": target_scale,
            "context_fields": fields,
            "context_vocabulary": vocabulary,
            "context_cardinalities": cardinalities,
            "proteins": proteins,
            "seed": args.seed,
            "fit_scope": "all_released_labeled_rows",
            "fit_treatment_count": int(len(treatment_ids)),
            "fit_exact_control_count": int(fit_rows.sum()),
            "encoder_checkpoint": str(encoder_path),
            "residual_scale": args.residual_scale,
        },
        checkpoint_path,
    )
    pd.DataFrame(base_history).to_csv(output / "base_history.csv", index=False)
    pd.DataFrame(residual_history).to_csv(output / "residual_history.csv", index=False)
    manifest = {
        "status": "complete",
        "model": "M9.6-full-refit",
        "knowledge_track": "open-knowledge",
        "fit_scope": "all_released_labeled_rows",
        "seed": args.seed,
        "device": str(device),
        "base_epochs": args.base_epochs,
        "residual_epochs": args.residual_epochs,
        "residual_penalty": args.residual_penalty,
        "residual_scale": args.residual_scale,
        "fit_treatment_count": int(len(treatment_ids)),
        "fit_exact_control_count": int(fit_rows.sum()),
        "test_count": int(len(test_metadata)),
        "test_treatment_count": int(test_treatment.sum()),
        "n_proteins": int(len(proteins)),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "prediction": str(prediction_path),
        "prediction_sha256": sha256(prediction_path),
        "inputs": {
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "metadata": {"path": str(metadata_path), "sha256": sha256(metadata_path)},
            "proteome": {"path": str(proteome_path), "sha256": sha256(proteome_path)},
            "test_metadata": {"path": str(test_metadata_path), "sha256": sha256(test_metadata_path)},
            "chemical_map": {"path": str(chemical_map_path), "sha256": sha256(chemical_map_path)},
            "encoder": {"path": str(encoder_path), "sha256": sha256(encoder_path)},
        },
        "elapsed_seconds": time.time() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--metadata-test", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-epochs", type=int, default=70)
    parser.add_argument("--residual-epochs", type=int, default=40)
    parser.add_argument("--residual-penalty", type=float, default=0.02)
    parser.add_argument("--residual-scale", type=float, default=0.2)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
