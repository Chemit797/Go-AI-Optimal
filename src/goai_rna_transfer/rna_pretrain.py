from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rdkit
import sklearn
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import (
    load_config,
    morgan_fingerprint,
    parent_connectivity_key,
    rowwise_pearson_mean,
    seed_everything,
    sha256,
    write_json,
)
from .models import RNAResponseModel


def _decode_strings(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset).astype(str)


def _categorical(group: h5py.Group) -> np.ndarray:
    categories = _decode_strings(group["categories"])
    return categories[np.asarray(group["codes"], dtype=np.int64)]


def load_op3(path: Path, layer: str, clip_abs: float) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        observation = pd.DataFrame(
            {
                name: _categorical(handle["obs"][name])
                for name in ("cell_type", "sm_lincs_id", "sm_name", "SMILES", "split")
            }
        )
        target = np.asarray(handle[f"layers/{layer}"], dtype=np.float32)
        genes = _decode_strings(handle["var/gene"])
    if target.shape != (len(observation), len(genes)) or not np.isfinite(target).all():
        raise ValueError("Unexpected OP3 target shape or non-finite values")
    return observation, np.clip(target, -clip_abs, clip_abs), genes


def group_folds(groups: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    mapping = {group: index % n_splits for index, group in enumerate(shuffled)}
    return np.asarray([mapping[group] for group in groups], dtype=np.int64)


def permute_drug_fingerprints(
    observation: pd.DataFrame,
    fingerprints: np.ndarray,
    selected: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Assign every row of a drug the same, deliberately wrong drug structure.

    The response targets, cell labels, missingness pattern, optimizer, and model
    are unchanged.  A derangement avoids accidentally leaving a compound paired
    with its own fingerprint and is a stricter negative control than row shuffling.
    """
    selected_indices = np.flatnonzero(selected)
    names = observation.loc[selected_indices, "sm_lincs_id"].to_numpy()
    unique_names = np.unique(names)
    if len(unique_names) < 2:
        raise ValueError("Drug-level shuffle needs at least two compounds")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(1, len(unique_names)))
    permuted_names = np.roll(rng.permutation(unique_names), shift)
    original_names = np.roll(permuted_names, -shift)
    mapping = dict(zip(original_names, permuted_names))
    # A random roll of a random ordering is a derangement in ordering space.  Be
    # defensive in case a future refactor changes the construction.
    if any(source == destination for source, destination in mapping.items()):
        ordered = sorted(unique_names)
        mapping = dict(zip(ordered, ordered[1:] + ordered[:1]))
    first_row = observation.reset_index().groupby("sm_lincs_id", sort=False)["index"].first()
    replacement = np.stack(
        [fingerprints[int(first_row.loc[mapping[name]])] for name in names]
    ).astype(np.float32)
    return replacement


def fit_model(
    model: RNAResponseModel,
    fingerprint: np.ndarray,
    cell: np.ndarray,
    target: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> list[dict[str, float]]:
    seed_everything(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.SmoothL1Loss(beta=0.5)
    dataset = TensorDataset(
        torch.from_numpy(fingerprint), torch.from_numpy(cell), torch.from_numpy(target)
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for batch_fingerprint, batch_cell, batch_target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_fingerprint.to(device), batch_cell.to(device))
            loss = loss_function(prediction, batch_target.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses))})
    return history


@torch.no_grad()
def predict(model: RNAResponseModel, fingerprint: np.ndarray, cell: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(fingerprint).to(device), torch.from_numpy(cell).to(device)).cpu().numpy()


def build_model(config: dict, n_cells: int, output_dim: int) -> RNAResponseModel:
    fingerprint = config["fingerprint"]
    pretrain = config["rna_pretraining"]
    return RNAResponseModel(
        n_bits=int(fingerprint["n_bits"]),
        encoder_hidden=int(pretrain["encoder_hidden"]),
        encoder_dim=int(pretrain["encoder_dim"]),
        n_cells=n_cells,
        cell_dim=int(pretrain["cell_dim"]),
        fusion_hidden=int(pretrain["fusion_hidden"]),
        output_dim=output_dim,
        dropout=float(pretrain["dropout"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Required: the legacy OP3 diagnostic uses a global label PCA and is not a leakage-safe OOF score.",
    )
    parser.add_argument("--scope", choices=("strict", "all"), default="strict")
    args = parser.parse_args()
    if not args.skip_cv:
        raise ValueError(
            "OP3 compound CV is disabled because its label transform is not fold-local; "
            "use --skip-cv for full external pretraining"
        )
    config = load_config(args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    pretrain = config["rna_pretraining"]
    source = Path(config["paths"]["op3_h5ad"])
    output = Path(config["paths"]["output_root"]) / "models" / "rna_pretraining"
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    observation, target, genes = load_op3(source, str(pretrain["target_layer"]), float(pretrain["clip_abs_logfc"]))
    excluded_compounds: list[str] = []
    if args.scope == "strict":
        goai_map = pd.read_csv(config["paths"]["goai_chemical_map"], sep="\t", keep_default_na=False)
        goai_keys = {
            key
            for key in goai_map["canonical_smiles"].map(parent_connectivity_key)
            if key is not None
        }
        op3_keys = observation["SMILES"].map(parent_connectivity_key)
        excluded_compounds = sorted(observation.loc[op3_keys.isin(goai_keys), "sm_name"].unique())
        keep = ~op3_keys.isin(goai_keys).to_numpy()
        observation = observation.loc[keep].reset_index(drop=True)
        target = target[keep]
    cells = sorted(observation["cell_type"].unique())
    cell_lookup = {name: index for index, name in enumerate(cells)}
    cell = observation["cell_type"].map(cell_lookup).to_numpy(dtype=np.int64)
    chemical = observation[["sm_lincs_id", "SMILES"]].drop_duplicates("sm_lincs_id")
    fp_lookup = {
        row.sm_lincs_id: morgan_fingerprint(
            row.SMILES, int(config["fingerprint"]["radius"]), int(config["fingerprint"]["n_bits"])
        )
        for row in chemical.itertuples(index=False)
    }
    fingerprints = np.stack(observation["sm_lincs_id"].map(fp_lookup).to_numpy()).astype(np.float32)

    # Remove human cell-type main effects so the chemical encoder cannot win by
    # letting the decoder predict only the PBMC context. This transformation is
    # learned exclusively from external RNA labels and never sees GOAI labels.
    if bool(pretrain.get("residualize_cell_mean", True)):
        residual_target = target.copy()
        for cell_name in cells:
            selected = observation["cell_type"].eq(cell_name).to_numpy()
            residual_target[selected] -= target[selected].mean(axis=0, keepdims=True)
    else:
        residual_target = target

    # PCA is learned only from external RNA labels. It never sees GOAI labels.
    pca = PCA(n_components=int(pretrain["pca_rank"]), whiten=True, random_state=seed)
    latent = pca.fit_transform(residual_target).astype(np.float32)
    latent_scale = np.maximum(latent.std(axis=0, keepdims=True), 1e-6).astype(np.float32)
    latent = latent / latent_scale
    np.savez_compressed(
        output / f"rna_target_pca_{args.scope}.npz",
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        latent_scale=latent_scale,
        genes=genes,
    )

    folds = group_folds(observation["sm_lincs_id"].to_numpy(), int(pretrain["n_folds"]), seed)
    oof_real = np.full_like(latent, np.nan)
    oof_shuffle = np.full_like(latent, np.nan)
    fold_rows: list[dict[str, float | int | str]] = []
    if not args.skip_cv:
        for fold in range(int(pretrain["n_folds"])):
            train = folds != fold
            valid = folds == fold
            for arm in ("real", "shuffle"):
                train_fingerprints = fingerprints[train]
                if arm == "shuffle":
                    train_fingerprints = permute_drug_fingerprints(
                        observation, fingerprints, train, seed + 1000 + fold
                    )
                seed_everything(seed + fold)
                model = build_model(config, len(cells), latent.shape[1])
                history = fit_model(
                    model,
                    train_fingerprints,
                    cell[train],
                    latent[train],
                    int(pretrain["cv_epochs"]),
                    int(pretrain["batch_size"]),
                    float(pretrain["learning_rate"]),
                    float(pretrain["weight_decay"]),
                    torch.device(args.device),
                    seed + fold,
                )
                fold_prediction = predict(model, fingerprints[valid], cell[valid], torch.device(args.device))
                if arm == "real":
                    oof_real[valid] = fold_prediction
                else:
                    oof_shuffle[valid] = fold_prediction
                fold_rows.append(
                    {
                        "fold": fold,
                        "arm": arm,
                        "n_train": int(train.sum()),
                        "n_valid": int(valid.sum()),
                        "latent_rmse": float(np.sqrt(np.mean((fold_prediction - latent[valid]) ** 2))),
                        "latent_rowwise_pcc": rowwise_pearson_mean(fold_prediction, latent[valid]),
                        "last_train_loss": history[-1]["loss"],
                    }
                )
        pd.DataFrame(fold_rows).to_csv(output / f"rna_oof_metrics_{args.scope}.csv", index=False)
        np.savez_compressed(output / f"rna_oof_predictions_{args.scope}.npz", folds=folds, real=oof_real, shuffle=oof_shuffle, truth=latent)

    checkpoints: dict[str, str] = {}
    for arm in ("real", "shuffle"):
        fit_fingerprints = fingerprints
        if arm == "shuffle":
            fit_fingerprints = permute_drug_fingerprints(
                observation, fingerprints, np.ones(len(observation), dtype=bool), seed + 9001
            )
        seed_everything(seed)
        model = build_model(config, len(cells), latent.shape[1])
        history = fit_model(
            model,
            fit_fingerprints,
            cell,
            latent,
            int(pretrain["epochs"]),
            int(pretrain["batch_size"]),
            float(pretrain["learning_rate"]),
            float(pretrain["weight_decay"]),
            torch.device(args.device),
            seed,
        )
        path = output / f"rna_{arm}_{args.scope}_encoder.pt"
        torch.save(
            {
                "chemical_encoder": copy.deepcopy(model.chemical_encoder).cpu().state_dict(),
                "arm": arm,
                "config": config,
                "cells": cells,
                "source_sha256": sha256(source),
            },
            path,
        )
        pd.DataFrame(history).to_csv(output / f"rna_{arm}_{args.scope}_history.csv", index=False)
        checkpoints[arm] = str(path)

    write_json(
        output / f"manifest_{args.scope}.json",
        {
            "status": "complete",
            "knowledge_track": "open-knowledge",
            "source": str(source),
            "source_sha256": sha256(source),
            "source_url": "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/2023-09-14_kaggle_upload/2023-09-12_de_by_cell_type_train.h5ad",
            "source_dataset": "OP3 / Open Problems - Single-Cell Perturbations",
            "target_layer": str(pretrain["target_layer"]),
            "pretraining_scope": args.scope,
            "goai_parent_structures_excluded": excluded_compounds,
            "n_rows": len(observation),
            "n_compounds": int(observation["sm_lincs_id"].nunique()),
            "n_cells": len(cells),
            "n_genes": len(genes),
            "pca_rank": int(pretrain["pca_rank"]),
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "checkpoints": checkpoints,
            "elapsed_seconds": time.time() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "rdkit": rdkit.__version__,
            "config": config,
        },
    )
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
