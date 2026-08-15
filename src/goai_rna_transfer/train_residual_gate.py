"""Train a zero-initialized RNA chemical residual on a frozen context model."""

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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import load_config, seed_everything, sha256, write_json
from .models import ChemicalEncoder
from .train_s1 import (
    build_model,
    fold_context_reference,
    fold_diagnostics,
    permute_validation_fingerprints,
)


class ChemicalResidualGate(nn.Module):
    def __init__(
        self,
        chemical_encoder: ChemicalEncoder,
        context_dim: int,
        chemical_dim: int,
        hidden: int,
        n_proteins: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.chemical_encoder = chemical_encoder
        for parameter in self.chemical_encoder.parameters():
            parameter.requires_grad = False
        self.context_gate = nn.Sequential(nn.Linear(context_dim, chemical_dim), nn.Tanh())
        self.head = nn.Sequential(
            nn.Linear(chemical_dim * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_proteins),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def train(self, mode: bool = True) -> "ChemicalResidualGate":
        super().train(mode)
        # A frozen representation must also have frozen dropout behavior.
        self.chemical_encoder.eval()
        return self

    def forward(self, fingerprint: torch.Tensor, context_feature: torch.Tensor) -> torch.Tensor:
        chemical = self.chemical_encoder(fingerprint)
        gated = chemical * self.context_gate(context_feature)
        return self.head(torch.cat([chemical, gated], dim=1))


@torch.no_grad()
def frozen_context_outputs(
    base: nn.Module,
    fingerprint: np.ndarray,
    context: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    base.eval()
    dataset = TensorDataset(torch.from_numpy(fingerprint), torch.from_numpy(context))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions: list[np.ndarray] = []
    features: list[np.ndarray] = []
    for batch_fingerprint, batch_context in loader:
        batch_fingerprint = batch_fingerprint.to(device)
        batch_context = batch_context.to(device)
        zero = torch.zeros_like(batch_fingerprint)
        predictions.append(base(zero, batch_context).float().cpu().numpy())
        features.append(base.context_encoder(batch_context).float().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(features)


def fit_residual(
    model: ChemicalResidualGate,
    fingerprint: np.ndarray,
    context_feature: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    raw_delta: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    high_effect_weight: float,
    residual_penalty: float,
    seed: int,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    dataset = TensorDataset(
        torch.from_numpy(fingerprint),
        torch.from_numpy(context_feature),
        torch.from_numpy(baseline),
        torch.from_numpy(target),
        torch.from_numpy(mask),
        torch.from_numpy(raw_delta),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0
        correction_square_sum = 0.0
        for batch_fp, batch_context, batch_base, batch_target, batch_mask, batch_raw in loader:
            batch_fp = batch_fp.to(device, non_blocking=True)
            batch_context = batch_context.to(device, non_blocking=True)
            batch_base = batch_base.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            batch_raw = batch_raw.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                correction = model(batch_fp, batch_context)
                prediction = batch_base + correction
                element = nn.functional.smooth_l1_loss(
                    prediction, batch_target, beta=0.5, reduction="none"
                )
                weights = batch_mask.float() * (
                    1.0 + high_effect_weight * (batch_raw.abs() > 1.0).float()
                )
                data_loss = (element * weights).sum() / weights.sum().clamp_min(1.0)
                shrinkage = (
                    (correction.square() * batch_mask.float()).sum()
                    / batch_mask.sum().clamp_min(1)
                )
                loss = data_loss + residual_penalty * shrinkage
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float((element.detach() * weights).sum().cpu())
            weight_sum += float(weights.sum().cpu())
            correction_square_sum += float(shrinkage.detach().cpu())
        history.append(
            {
                "epoch": float(epoch + 1),
                "weighted_smooth_l1": loss_sum / max(weight_sum, 1.0),
                "batch_mean_correction_square": correction_square_sum / len(loader),
            }
        )
    return history


@torch.no_grad()
def predict_correction(
    model: ChemicalResidualGate,
    fingerprint: np.ndarray,
    context_feature: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(fingerprint), torch.from_numpy(context_feature)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    pieces = [
        model(batch_fp.to(device), batch_context.to(device)).float().cpu().numpy()
        for batch_fp, batch_context in loader
    ]
    return np.concatenate(pieces)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--base-arm", default="no_chemical")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--residual-penalty", type=float, default=0.02)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    source_hashes = {
        name: sha256(Path(__file__).with_name(name))
        for name in ("train_residual_gate.py", "train_s1.py", "models.py", "common.py")
    }
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint = Path(args.encoder_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    declared = str(checkpoint_payload.get("arm", "")).casefold()
    if "shuffle" in args.arm.casefold() and "shuffle" not in declared:
        raise ValueError("Shuffle residual arm must load a shuffled encoder")
    if "real" in args.arm.casefold() and "shuffle" in declared:
        raise ValueError("Real residual arm cannot load a shuffled encoder")

    root = Path(config["paths"]["output_root"])
    output = root / "oof" / args.arm / f"seed_{args.seed}" / f"fold_{args.fold}"
    if (output / "manifest.json").exists() and not args.force:
        raise FileExistsError(f"Completed artifact exists: {output}")
    base_path = (
        root
        / "oof"
        / args.base_arm
        / f"seed_{args.seed}"
        / f"fold_{args.fold}"
        / "model.pt"
    )
    if not base_path.is_file():
        raise FileNotFoundError(f"Train the frozen context base first: {base_path}")
    base_payload = torch.load(base_path, map_location="cpu", weights_only=False)
    cache_path = Path(config["paths"]["private_cache"]) / "goai_s1_delta.npz"
    with np.load(cache_path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    folds = arrays["folds"]
    train_rows = folds != args.fold
    valid_rows = folds == args.fold
    fit_rows = train_rows & arrays["mask"].any(axis=1)
    target_mean = np.asarray(base_payload["target_mean"], dtype=np.float32)
    target_scale = np.asarray(base_payload["target_scale"], dtype=np.float32)
    standardized = ((arrays["delta"] - target_mean) / target_scale).astype(np.float32)
    standardized[~arrays["mask"]] = 0.0

    base = build_model(
        config,
        arrays["context_cardinalities"],
        len(arrays["proteins"]),
        checkpoint=None,
        freeze_encoder=False,
        device=device,
    )
    base.load_state_dict(base_payload["model"], strict=True)
    for parameter in base.parameters():
        parameter.requires_grad = False
    baseline_all, context_feature_all = frozen_context_outputs(
        base,
        arrays["fingerprints"].astype(np.float32),
        arrays["contexts"].astype(np.int64),
        device,
        int(config["goai_s1"]["batch_size"]),
    )
    encoder = ChemicalEncoder(
        int(config["fingerprint"]["n_bits"]),
        int(config["rna_pretraining"]["encoder_hidden"]),
        int(config["rna_pretraining"]["encoder_dim"]),
        float(config["rna_pretraining"]["dropout"]),
    )
    encoder.load_state_dict(checkpoint_payload["chemical_encoder"], strict=True)
    residual = ChemicalResidualGate(
        encoder,
        context_feature_all.shape[1],
        int(config["rna_pretraining"]["encoder_dim"]),
        hidden=128,
        n_proteins=len(arrays["proteins"]),
        dropout=0.10,
    ).to(device)
    started = time.time()
    history = fit_residual(
        residual,
        arrays["fingerprints"][fit_rows].astype(np.float32),
        context_feature_all[fit_rows].astype(np.float32),
        baseline_all[fit_rows].astype(np.float32),
        standardized[fit_rows],
        arrays["mask"][fit_rows],
        arrays["delta"][fit_rows],
        device,
        args.epochs,
        int(config["goai_s1"]["batch_size"]),
        float(config["goai_s1"]["learning_rate"]),
        float(config["goai_s1"]["weight_decay"]),
        float(config["goai_s1"]["high_effect_weight"]),
        args.residual_penalty,
        args.seed,
    )
    correction = predict_correction(
        residual,
        arrays["fingerprints"][valid_rows].astype(np.float32),
        context_feature_all[valid_rows].astype(np.float32),
        device,
        int(config["goai_s1"]["batch_size"]),
    )
    permuted_correction = predict_correction(
        residual,
        permute_validation_fingerprints(
            arrays["fingerprints"].astype(np.float32), arrays["chemicals"], valid_rows
        ),
        context_feature_all[valid_rows].astype(np.float32),
        device,
        int(config["goai_s1"]["batch_size"]),
    )
    prediction = (
        baseline_all[valid_rows] + args.residual_scale * correction
    ) * target_scale + target_mean
    permuted_prediction = (
        baseline_all[valid_rows] + args.residual_scale * permuted_correction
    ) * target_scale + target_mean
    reference, reference_mask = fold_context_reference(
        arrays["delta"], arrays["mask"], arrays["context_keys"].astype(str), train_rows, valid_rows
    )
    metrics = fold_diagnostics(
        prediction,
        arrays["delta"][valid_rows],
        arrays["mask"][valid_rows],
        reference,
        reference_mask,
    )
    permuted_metrics = fold_diagnostics(
        permuted_prediction,
        arrays["delta"][valid_rows],
        arrays["mask"][valid_rows],
        reference,
        reference_mask,
    )
    metrics.update({f"permuted_chemical_{key}": value for key, value in permuted_metrics.items()})
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
    torch.save(
        {
            "residual_model": copy.deepcopy(residual).cpu().state_dict(),
            "base_model": str(base_path),
            "encoder_checkpoint": str(checkpoint),
            "target_mean": target_mean,
            "target_scale": target_scale,
        },
        output / "model.pt",
    )
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    pd.DataFrame([{"arm": args.arm, "seed": args.seed, "fold": args.fold, **metrics}]).to_csv(
        output / "fold_metrics.csv", index=False
    )
    write_json(
        output / "manifest.json",
        {
            "status": "complete",
            "knowledge_track": "open-knowledge",
            "architecture": "frozen-context-plus-zero-init-frozen-rna-residual-gate",
            "arm": args.arm,
            "fold": args.fold,
            "seed": args.seed,
            "epochs": args.epochs,
            "residual_penalty": args.residual_penalty,
            "residual_scale": args.residual_scale,
            "base_model": str(base_path),
            "base_model_sha256": sha256(base_path),
            "encoder_checkpoint": str(checkpoint),
            "encoder_checkpoint_sha256": sha256(checkpoint),
            "predictions": str(prediction_path),
            "predictions_sha256": sha256(prediction_path),
            "metrics": metrics,
            "elapsed_seconds": time.time() - started,
            "source_code_sha256_at_process_start": source_hashes,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    )
    print(json.dumps({"status": "complete", "output": str(output), **metrics}, indent=2))


if __name__ == "__main__":
    main()
