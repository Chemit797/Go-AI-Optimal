from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import load_config, pearson_flat, seed_everything, sha256, write_json
from .models import ChemicalEncoder, ProteinDeltaModel


def masked_location_scale(
    values: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    counts = mask.sum(axis=0).astype(np.float64)
    safe_counts = np.maximum(counts, 1.0)
    mean = (values * mask).sum(axis=0, dtype=np.float64) / safe_counts
    centered = (values - mean.astype(np.float32)) * mask
    variance = (centered.astype(np.float64) ** 2).sum(axis=0) / safe_counts
    scale = np.sqrt(variance)
    mean[counts == 0] = 0.0
    scale[(counts < 2) | (scale < 0.10)] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def build_model(
    config: dict,
    cardinalities: np.ndarray,
    n_proteins: int,
    checkpoint: Path | None,
    freeze_encoder: bool,
    device: torch.device,
) -> ProteinDeltaModel:
    fp_config = config["fingerprint"]
    rna_config = config["rna_pretraining"]
    s1_config = config["goai_s1"]
    encoder = ChemicalEncoder(
        n_bits=int(fp_config["n_bits"]),
        hidden=int(rna_config["encoder_hidden"]),
        output_dim=int(rna_config["encoder_dim"]),
        dropout=float(rna_config["dropout"]),
    )
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("chemical_encoder", payload)
        encoder.load_state_dict(state, strict=True)
    if freeze_encoder:
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    return ProteinDeltaModel(
        chemical_encoder=encoder,
        cardinalities=[int(value) for value in cardinalities],
        chemical_dim=int(rna_config["encoder_dim"]),
        context_dim=int(s1_config["context_dim"]),
        fusion_hidden=int(s1_config["fusion_hidden"]),
        n_proteins=n_proteins,
        dropout=float(s1_config["dropout"]),
    ).to(device)


def optimizer_for(
    model: ProteinDeltaModel,
    learning_rate: float,
    weight_decay: float,
    encoder_lr_scale: float,
    pretrained: bool,
) -> torch.optim.Optimizer:
    encoder_parameters = [
        parameter for parameter in model.chemical_encoder.parameters() if parameter.requires_grad
    ]
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in encoder_ids
    ]
    groups: list[dict] = []
    if encoder_parameters:
        groups.append(
            {
                "params": encoder_parameters,
                "lr": learning_rate * (encoder_lr_scale if pretrained else 1.0),
            }
        )
    if other_parameters:
        groups.append({"params": other_parameters, "lr": learning_rate})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def fit(
    model: ProteinDeltaModel,
    fingerprint: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    raw_delta: np.ndarray,
    config: dict,
    device: torch.device,
    seed: int,
    pretrained: bool,
    epochs: int,
) -> list[dict[str, float]]:
    s1_config = config["goai_s1"]
    optimizer = optimizer_for(
        model,
        float(s1_config["learning_rate"]),
        float(s1_config["weight_decay"]),
        float(s1_config["encoder_lr_scale"]),
        pretrained,
    )
    dataset = TensorDataset(
        torch.from_numpy(fingerprint),
        torch.from_numpy(context),
        torch.from_numpy(target),
        torch.from_numpy(mask),
        torch.from_numpy(raw_delta),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(s1_config["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    history: list[dict[str, float]] = []
    high_effect_weight = float(s1_config["high_effect_weight"])
    for epoch in range(epochs):
        model.train()
        if not any(
            parameter.requires_grad for parameter in model.chemical_encoder.parameters()
        ):
            # Freezing a representation also freezes its stochastic behavior.
            # `model.train()` would otherwise reactivate encoder dropout.
            model.chemical_encoder.eval()
        total_loss = 0.0
        total_weight = 0.0
        for batch_fp, batch_context, batch_target, batch_mask, batch_raw in loader:
            batch_fp = batch_fp.to(device, non_blocking=True)
            batch_context = batch_context.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            batch_raw = batch_raw.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(batch_fp, batch_context)
                element_loss = nn.functional.smooth_l1_loss(
                    prediction, batch_target, beta=0.5, reduction="none"
                )
                weights = batch_mask.float() * (
                    1.0 + high_effect_weight * (batch_raw.abs() > 1.0).float()
                )
                loss = (element_loss * weights).sum() / weights.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float((element_loss.detach() * weights).sum().cpu())
            total_weight += float(weights.sum().cpu())
        history.append(
            {"epoch": float(epoch + 1), "train_weighted_smooth_l1": total_loss / total_weight}
        )
    return history


@torch.no_grad()
def predict(
    model: ProteinDeltaModel,
    fingerprint: np.ndarray,
    context: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    dataset = TensorDataset(torch.from_numpy(fingerprint), torch.from_numpy(context))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    pieces: list[np.ndarray] = []
    for batch_fp, batch_context in loader:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            value = model(batch_fp.to(device), batch_context.to(device))
        pieces.append(value.float().cpu().numpy())
    return np.concatenate(pieces, axis=0)


def fold_context_reference(
    delta: np.ndarray,
    mask: np.ndarray,
    keys: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    references: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in np.unique(keys[train]):
        selected = train & (keys == key)
        count = mask[selected].sum(axis=0)
        total = (delta[selected] * mask[selected]).sum(axis=0, dtype=np.float64)
        mean = np.zeros(delta.shape[1], dtype=np.float32)
        observed = count > 0
        mean[observed] = (total[observed] / count[observed]).astype(np.float32)
        references[str(key)] = (mean, observed)
    valid_indices = np.flatnonzero(valid)
    value = np.zeros((len(valid_indices), delta.shape[1]), dtype=np.float32)
    observed = np.zeros((len(valid_indices), delta.shape[1]), dtype=bool)
    for local_index, global_index in enumerate(valid_indices):
        reference = references.get(str(keys[global_index]))
        if reference is not None:
            value[local_index], observed[local_index] = reference
    return value, observed


def fold_diagnostics(
    prediction: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    context_reference: np.ndarray,
    context_mask: np.ndarray,
) -> dict[str, float | int]:
    response_mask = mask & np.isfinite(prediction)
    residual_mask = response_mask & context_mask
    high_true = response_mask & (np.abs(truth) > 1.0)
    high_pred = response_mask & (np.abs(prediction) > 1.0)
    true_positive = high_true & high_pred & (np.sign(prediction) == np.sign(truth))
    precision = float(true_positive.sum() / high_pred.sum()) if high_pred.any() else float("nan")
    recall = float(true_positive.sum() / high_true.sum()) if high_true.any() else float("nan")
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(precision + recall) and precision + recall
        else float("nan")
    )
    return {
        "response_n_observed_values": int(response_mask.sum()),
        "context_n_observed_values": int(residual_mask.sum()),
        "fc_pcc": pearson_flat(prediction, truth, response_mask),
        "context_residual_pcc": pearson_flat(
            prediction - context_reference, truth - context_reference, residual_mask
        ),
        "high_effect_pcc": pearson_flat(prediction, truth, high_true),
        "high_effect_precision": precision,
        "high_effect_recall": recall,
        "high_effect_f1": f1,
    }


def permute_validation_fingerprints(
    fingerprints: np.ndarray,
    chemicals: np.ndarray,
    valid_rows: np.ndarray,
) -> np.ndarray:
    """Deterministically derange fingerprints at whole-chemical granularity."""
    valid_fingerprints = fingerprints[valid_rows]
    valid_chemicals = chemicals[valid_rows].astype(str)
    unique = sorted(np.unique(valid_chemicals).tolist())
    if len(unique) < 2:
        raise ValueError("Chemical sensitivity control needs at least two held-out drugs")
    representative = {
        name: valid_fingerprints[np.flatnonzero(valid_chemicals == name)[0]] for name in unique
    }
    mapping = dict(zip(unique, unique[1:] + unique[:1]))
    return np.stack([representative[mapping[name]] for name in valid_chemicals]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--fold", required=True, type=int, choices=range(4))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    source_hashes = {
        name: sha256(Path(__file__).with_name(name))
        for name in ("train_s1.py", "models.py", "common.py")
    }
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = Path(args.encoder_checkpoint) if args.encoder_checkpoint else None
    if checkpoint is not None and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    transfer_named = any(
        token in args.arm.casefold() for token in ("rna", "op3", "l1000", "real", "shuffle")
    )
    if transfer_named and checkpoint is None:
        raise ValueError(f"Transfer arm {args.arm!r} requires --encoder-checkpoint")
    if args.arm.casefold().endswith("_frozen") and not args.freeze_encoder:
        raise ValueError("An *_frozen arm requires --freeze-encoder")
    if args.arm.casefold().endswith("_ft") and args.freeze_encoder:
        raise ValueError("An *_ft arm must fine-tune rather than freeze the encoder")
    if args.arm.casefold() in {"morgan_scratch", "no_chemical"} and checkpoint is not None:
        raise ValueError(f"Scratch arm {args.arm!r} must not receive a pretrained checkpoint")
    if checkpoint is not None:
        checkpoint_metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
        declared = str(
            checkpoint_metadata.get("arm", checkpoint_metadata.get("label_mode", ""))
        ).casefold()
        requested = args.arm.casefold()
        if "real" in requested and "shuffle" in declared:
            raise ValueError("A real-transfer arm cannot load a shuffled checkpoint")
        if "shuffle" in requested and declared and "shuffle" not in declared:
            raise ValueError("A shuffle-control arm must load a shuffled checkpoint")
    seed_everything(args.seed)
    source = Path(config["paths"]["private_cache"]) / "goai_s1_delta.npz"
    if not source.exists():
        raise FileNotFoundError(f"Run src.goai_data first: {source}")
    with np.load(source, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    folds = arrays["folds"]
    train_rows = folds != args.fold
    valid_rows = folds == args.fold
    delta = arrays["delta"].astype(np.float32, copy=False)
    mask = arrays["mask"].astype(bool, copy=False)
    # The 12 rows without an exact matched control contain no delta target and
    # are retained for prediction/coverage accounting but cannot train the loss.
    fit_rows = train_rows & mask.any(axis=1)
    if set(np.unique(folds)) != {0, 1, 2, 3} or not valid_rows.any():
        raise ValueError("Unexpected frozen S1 fold assignment")

    fingerprints = arrays["fingerprints"].astype(np.float32, copy=False)
    if args.arm == "no_chemical":
        fingerprints = np.zeros_like(fingerprints)
    context = arrays["contexts"].astype(np.int64, copy=False)
    target_mean, target_scale = masked_location_scale(delta[fit_rows], mask[fit_rows])
    standardized = ((delta - target_mean) / target_scale).astype(np.float32)
    standardized[~mask] = 0.0

    model = build_model(
        config,
        arrays["context_cardinalities"],
        delta.shape[1],
        checkpoint,
        args.freeze_encoder or args.arm == "no_chemical",
        device,
    )
    initial_encoder = copy.deepcopy(model.chemical_encoder).cpu().state_dict()
    epochs = args.epochs or int(config["goai_s1"]["max_epochs"])
    output = (
        Path(config["paths"]["output_root"])
        / "oof"
        / args.arm
        / f"seed_{args.seed}"
        / f"fold_{args.fold}"
    )
    if (output / "manifest.json").exists() and not args.force:
        raise FileExistsError(
            f"Completed artifact exists at {output}; use a new arm name or --force"
        )
    started = time.time()
    history = fit(
        model,
        fingerprints[fit_rows],
        context[fit_rows],
        standardized[fit_rows],
        mask[fit_rows],
        delta[fit_rows],
        config,
        device,
        args.seed,
        checkpoint is not None,
        epochs,
    )
    prediction_standardized = predict(
        model,
        fingerprints[valid_rows],
        context[valid_rows],
        device,
        int(config["goai_s1"]["batch_size"]),
    )
    prediction = prediction_standardized * target_scale + target_mean
    permuted_standardized = predict(
        model,
        permute_validation_fingerprints(
            fingerprints, arrays["chemicals"], valid_rows
        ),
        context[valid_rows],
        device,
        int(config["goai_s1"]["batch_size"]),
    )
    permuted_prediction = permuted_standardized * target_scale + target_mean
    reference, reference_mask = fold_context_reference(
        delta, mask, arrays["context_keys"].astype(str), train_rows, valid_rows
    )
    diagnostics = fold_diagnostics(
        prediction,
        delta[valid_rows],
        mask[valid_rows],
        reference,
        reference_mask,
    )
    permuted_diagnostics = fold_diagnostics(
        permuted_prediction,
        delta[valid_rows],
        mask[valid_rows],
        reference,
        reference_mask,
    )
    diagnostics.update(
        {f"permuted_chemical_{key}": value for key, value in permuted_diagnostics.items()}
    )

    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.npz"
    np.savez_compressed(
        prediction_path,
        sample_ids=arrays["sample_ids"][valid_rows],
        chemicals=arrays["chemicals"][valid_rows],
        proteins=arrays["proteins"],
        pred_delta=prediction.astype(np.float32),
        pred_delta_permuted_chemical=permuted_prediction.astype(np.float32),
        fold=np.asarray([args.fold], dtype=np.int64),
    )
    model_path = output / "model.pt"
    torch.save(
        {
            "model": copy.deepcopy(model).cpu().state_dict(),
            "initial_chemical_encoder": initial_encoder,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "arm": args.arm,
            "fold": args.fold,
            "seed": args.seed,
            "encoder_checkpoint": str(checkpoint) if checkpoint else None,
        },
        model_path,
    )
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    pd.DataFrame([{**{"fold": args.fold, "arm": args.arm, "seed": args.seed}, **diagnostics}]).to_csv(
        output / "fold_metrics.csv", index=False
    )
    write_json(
        output / "manifest.json",
        {
            "status": "complete",
            "knowledge_track": "open-knowledge",
            "architecture_family": "independent-rna-transfer",
            "arm": args.arm,
            "fold": args.fold,
            "model_seed": args.seed,
            "fold_seed": int(config["seed"]),
            "epochs": epochs,
            "n_fit_rows": int(fit_rows.sum()),
            "n_validation_rows": int(valid_rows.sum()),
            "encoder_checkpoint": str(checkpoint) if checkpoint else None,
            "encoder_checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
            "encoder_frozen": bool(args.freeze_encoder or args.arm == "no_chemical"),
            "validation_chemical_sensitivity": (
                "same fitted model; whole-heldout-drug fingerprint derangement"
            ),
            "private_cache": str(source),
            "private_cache_sha256": sha256(source),
            "predictions": str(prediction_path),
            "predictions_sha256": sha256(prediction_path),
            "model": str(model_path),
            "model_sha256": sha256(model_path),
            "metrics": diagnostics,
            "elapsed_seconds": time.time() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "rdkit": rdkit.__version__,
            "config": config,
            "source_code_sha256_at_process_start": source_hashes,
        },
    )
    print(json.dumps({"status": "complete", "output": str(output), **diagnostics}, indent=2))


if __name__ == "__main__":
    main()
