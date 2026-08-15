"""L1000FWD structure-to-transcriptome pretraining for the GOAI transfer lab.

The public L1000FWD files may live on the large shared mount.  All learned
artifacts and diagnostics are deliberately written beneath this experiment's
local ``output_root`` because the strict scope consults the private GOAI
chemical map when deciding which public compounds to remove.

Examples
--------
Download and structurally/hash verify the four public files::

    python -m goai_rna_transfer.l1000_pretrain download \
        --data-dir data/external/l1000fwd

Run the leakage-safe strict arm, including five-fold drug-held-out diagnostics::

    python -m goai_rna_transfer.l1000_pretrain train --config configs/experiment.yaml \
        --scope external-minus-goai --label-mode real --device cuda:0

The matched negative control changes only the drug-to-fingerprint pairing::

    python -m goai_rna_transfer.l1000_pretrain train --config configs/experiment.yaml \
        --scope external-minus-goai --label-mode input-shuffle --device cuda:1
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import pandas as pd
import rdkit
import sklearn
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
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
from .models import ChemicalEncoder


BASE_URL = "https://maayanlab.cloud/l1000fwd/download"
PUBLIC_FILES: dict[str, dict[str, Any]] = {
    "CD_signatures_LM_42809x978.gctx": {
        "url": f"{BASE_URL}/CD_signatures_LM_42809x978.gctx",
        "size": 346_694_992,
        "sha256": "6f50ade23f49d4d062938d936d98951be7528aced043cb2d77f9414c18545492",
    },
    "CD_signature_metadata.csv": {
        "url": f"{BASE_URL}/CD_signature_metadata.csv",
        "size": 5_075_455,
        "sha256": "b4495715d35f9bc17412c5ffa900d802812b25419ecd6291be6dd6fa4f38a557",
    },
    "Drugs_metadata.csv": {
        "url": f"{BASE_URL}/Drugs_metadata.csv",
        "size": 7_797_325,
        "sha256": "6447c511ac7d4111f2bd5e46cc95f0ca872d98d579d5e4ccaef6594ed7b899e3",
    },
    "Probes_L1000_metadata.csv": {
        "url": f"{BASE_URL}/Probes_L1000_metadata.csv",
        "size": 16_690,
        "sha256": "500aa68b119bbca5a805326328bd6cf5a754528b9fe89fd99c08364e6dd96f2e",
    },
}

GCTX_NAME = "CD_signatures_LM_42809x978.gctx"
SIGNATURE_METADATA_NAME = "CD_signature_metadata.csv"
DRUG_METADATA_NAME = "Drugs_metadata.csv"
PROBE_METADATA_NAME = "Probes_L1000_metadata.csv"
DEFAULT_DATA_DIR = Path("data/external/l1000fwd")


def _decode(values: h5py.Dataset | np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind in {"S", "O"}:
        return np.asarray(
            [item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item) for item in array],
            dtype=object,
        )
    return array.astype(str)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _download_one(url: str, destination: Path, expected_size: int, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == expected_size and not force:
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    if force:
        partial.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "goai-rna-transfer/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        append = offset > 0 and status == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=4 * 1024 * 1024)
    if partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"Incomplete download for {destination.name}: "
            f"{partial.stat().st_size} bytes, expected {expected_size}"
        )
    partial.replace(destination)


def download_public_files(data_dir: Path, force: bool = False) -> dict[str, Any]:
    """Download the four immutable L1000FWD files with resumable transfers."""
    started = time.time()
    for name, record in PUBLIC_FILES.items():
        _download_one(str(record["url"]), data_dir / name, int(record["size"]), force)
    report = verify_public_files(data_dir, verify_hashes=True)
    report["elapsed_seconds"] = time.time() - started
    return report


def verify_public_files(data_dir: Path, verify_hashes: bool = True) -> dict[str, Any]:
    """Verify sizes, optional SHA-256 digests, schemas, and GCTX/CSV alignment."""
    records: dict[str, Any] = {}
    for name, expected in PUBLIC_FILES.items():
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing public L1000FWD file: {path}")
        size = path.stat().st_size
        if size != int(expected["size"]):
            raise ValueError(f"Unexpected size for {name}: {size} != {expected['size']}")
        digest = sha256(path) if verify_hashes else None
        if verify_hashes and digest != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {name}: {digest}")
        records[name] = {"size": size, "sha256": digest, "url": expected["url"]}

    signatures = pd.read_csv(data_dir / SIGNATURE_METADATA_NAME, usecols=["sig_id", "pert_id"])
    drugs = pd.read_csv(data_dir / DRUG_METADATA_NAME, usecols=["pert_id", "canonical_smiles"])
    probes = pd.read_csv(data_dir / PROBE_METADATA_NAME, usecols=["pr_id", "pr_gene_symbol"])
    if tuple(signatures.shape) != (42_809, 2):
        raise ValueError(f"Unexpected signature metadata shape: {signatures.shape}")
    if len(probes) != 978 or probes["pr_id"].duplicated().any():
        raise ValueError("Expected exactly 978 unique L1000 probe IDs")
    if drugs["pert_id"].duplicated().any():
        raise ValueError("Drugs_metadata.csv contains duplicate pert_id values")

    with h5py.File(data_dir / GCTX_NAME, "r") as handle:
        matrix = handle["0/DATA/0/matrix"]
        signature_ids = _decode(handle["0/META/ROW/id"])
        probe_ids = _decode(handle["0/META/COL/id"])
        if matrix.shape != (978, 42_809):
            raise ValueError(f"Unexpected GCTX matrix shape: {matrix.shape}")
        if not np.array_equal(signature_ids, signatures["sig_id"].astype(str).to_numpy()):
            raise ValueError("GCTX signature IDs are not aligned to CD_signature_metadata.csv")
        if not np.array_equal(probe_ids, probes["pr_id"].astype(str).to_numpy()):
            raise ValueError("GCTX probe IDs are not aligned to Probes_L1000_metadata.csv")
    return {
        "status": "verified",
        "data_dir": str(data_dir),
        "hashes_verified": verify_hashes,
        "files": records,
        "matrix_shape": [42_809, 978],
        "n_signature_compounds": int(signatures["pert_id"].nunique()),
        "n_structured_compounds": int(
            drugs.loc[drugs["canonical_smiles"].notna(), "pert_id"].nunique()
        ),
    }


def load_signature_table(data_dir: Path) -> pd.DataFrame:
    signatures = pd.read_csv(data_dir / SIGNATURE_METADATA_NAME)
    required = {"sig_id", "cell_id", "pert_dose", "pert_id", "pert_time"}
    missing = required.difference(signatures.columns)
    if missing:
        raise ValueError(f"Signature metadata is missing columns: {sorted(missing)}")
    drugs = pd.read_csv(
        data_dir / DRUG_METADATA_NAME,
        usecols=["pert_id", "pert_iname", "canonical_smiles", "inchi_key"],
    )
    if drugs["pert_id"].duplicated().any():
        raise ValueError("Drug structure table must contain one row per pert_id")
    # Standardization (especially InChI generation) is expensive. Compute the
    # parent key once per compound, not once for every one of its signatures.
    valid_drug_smiles = drugs["canonical_smiles"].fillna("").astype(str).str.len().gt(0)
    drugs = drugs.loc[valid_drug_smiles].copy()
    drugs["parent_key"] = drugs["canonical_smiles"].map(parent_connectivity_key)
    drugs = drugs.loc[drugs["parent_key"].notna()].copy()
    table = signatures.merge(drugs, on="pert_id", how="left", validate="many_to_one")
    table["_matrix_index"] = np.arange(len(table), dtype=np.int64)
    valid_smiles = table["canonical_smiles"].fillna("").astype(str).str.len().gt(0)
    table = table.loc[valid_smiles].copy()
    return table.reset_index(drop=True)


def goai_parent_keys(chemical_map: Path) -> set[str]:
    mapping = pd.read_csv(chemical_map, sep="\t", keep_default_na=False)
    candidates: list[str] = []
    for column in ("canonical_smiles", "isomeric_smiles"):
        if column in mapping:
            candidates.extend(mapping[column].astype(str).tolist())
    return {key for key in map(parent_connectivity_key, candidates) if key is not None}


def apply_scope(
    table: pd.DataFrame,
    scope: str,
    goai_chemical_map: Path | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    before_rows = len(table)
    before_drugs = int(table["pert_id"].nunique())
    excluded_drugs = 0
    if scope == "external-minus-goai":
        if goai_chemical_map is None:
            raise ValueError("external-minus-goai requires a GOAI chemical map")
        forbidden = goai_parent_keys(goai_chemical_map)
        excluded = table["parent_key"].isin(forbidden)
        excluded_drugs = int(table.loc[excluded, "pert_id"].nunique())
        table = table.loc[~excluded].copy()
    elif scope != "external-all":
        raise ValueError(f"Unknown scope: {scope}")
    return table.reset_index(drop=True), {
        "structured_rows_before_scope": before_rows,
        "structured_compounds_before_scope": before_drugs,
        "rows_excluded_by_goai_parent": before_rows - len(table),
        "compounds_excluded_by_goai_parent": excluded_drugs,
        "rows_after_scope": len(table),
        "compounds_after_scope": int(table["pert_id"].nunique()),
    }


def deterministic_subsample_drugs(table: pd.DataFrame, max_drugs: int | None, seed: int) -> pd.DataFrame:
    """Subsample whole standardized parent structures (used only for smoke tests)."""
    if max_drugs is None or table["parent_key"].nunique() <= max_drugs:
        return table
    rng = np.random.default_rng(seed)
    compounds = np.sort(table["parent_key"].unique())
    selected = set(rng.choice(compounds, size=max_drugs, replace=False).tolist())
    return table.loc[table["parent_key"].isin(selected)].reset_index(drop=True)


def load_expression(data_dir: Path, matrix_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read selected signatures as observations x 978 landmarks.

    The source is an unchunked gene-by-signature HDF5 matrix. Reading it once
    contiguously is substantially faster than thousands of HDF5 fancy seeks and
    remains comfortably below one GiB of RAM.
    """
    with h5py.File(data_dir / GCTX_NAME, "r") as handle:
        dataset = handle["0/DATA/0/matrix"]
        probe_ids = _decode(handle["0/META/COL/id"])
        complete = np.asarray(dataset, dtype=np.float32).T
    selected = complete[np.asarray(matrix_indices, dtype=np.int64)].copy()
    del complete
    if not np.isfinite(selected).all():
        raise ValueError("L1000 signature matrix contains non-finite values")
    return selected, probe_ids


def build_fingerprints(
    table: pd.DataFrame,
    radius: int,
    n_bits: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    compounds = table[["pert_id", "canonical_smiles"]].drop_duplicates("pert_id")
    lookup = {
        str(row.pert_id): morgan_fingerprint(str(row.canonical_smiles), radius, n_bits)
        for row in compounds.itertuples(index=False)
    }
    fingerprints = np.stack([lookup[str(name)] for name in table["pert_id"]]).astype(np.float32)
    return fingerprints, lookup


def permute_drug_fingerprints(
    groups: Sequence[str],
    fingerprints: np.ndarray,
    selected: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Return drug-level deranged inputs while leaving every RNA label intact."""
    groups_array = np.asarray(groups, dtype=str)
    selected = np.asarray(selected, dtype=bool)
    selected_groups = np.unique(groups_array[selected])
    if len(selected_groups) < 2:
        raise ValueError("Input-shuffle requires at least two training compounds")
    rng = np.random.default_rng(seed)
    order = selected_groups.copy()
    rng.shuffle(order)
    mapping = dict(zip(order.tolist(), np.roll(order, -1).tolist()))
    first_row = {group: int(np.flatnonzero(groups_array == group)[0]) for group in selected_groups}
    result = np.stack(
        [fingerprints[first_row[mapping[group]]] for group in groups_array[selected]]
    ).astype(np.float32)
    if any(source == target for source, target in mapping.items()):
        raise AssertionError("Drug-level input permutation is not a derangement")
    return result


@dataclass
class Residualizer:
    global_mean: np.ndarray
    cell_means: dict[str, np.ndarray]


def fit_cell_residualizer(target: np.ndarray, cells: Sequence[str]) -> Residualizer:
    cells_array = np.asarray(cells, dtype=str)
    global_mean = target.mean(axis=0, dtype=np.float64).astype(np.float32)
    means = {
        cell: target[cells_array == cell].mean(axis=0, dtype=np.float64).astype(np.float32)
        for cell in np.unique(cells_array)
    }
    return Residualizer(global_mean=global_mean, cell_means=means)


def apply_cell_residualizer(
    target: np.ndarray,
    cells: Sequence[str],
    residualizer: Residualizer,
) -> np.ndarray:
    cells_array = np.asarray(cells, dtype=str)
    result = np.empty_like(target, dtype=np.float32)
    for cell in np.unique(cells_array):
        mean = residualizer.cell_means.get(cell, residualizer.global_mean)
        result[cells_array == cell] = target[cells_array == cell] - mean
    return result


@dataclass
class ContextTransform:
    cells: list[str]
    numeric_columns: list[str]
    numeric_mean: np.ndarray
    numeric_scale: np.ndarray


def fit_context(table: pd.DataFrame) -> ContextTransform:
    cells = sorted(table["cell_id"].fillna("<missing>").astype(str).unique())
    numeric_columns = [name for name in ("pert_time", "pert_dose") if name in table.columns]
    values = context_numeric_values(table, numeric_columns)
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-6)
    return ContextTransform(cells=cells, numeric_columns=numeric_columns, numeric_mean=mean, numeric_scale=scale)


def context_numeric_values(table: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for name in columns:
        values = pd.to_numeric(table[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if name == "pert_dose":
            values = np.log1p(np.maximum(values, 0.0)).astype(np.float32)
        pieces.append(values)
    return np.stack(pieces, axis=1) if pieces else np.zeros((len(table), 0), dtype=np.float32)


def transform_context(table: pd.DataFrame, transform: ContextTransform) -> tuple[np.ndarray, np.ndarray]:
    lookup = {cell: index + 1 for index, cell in enumerate(transform.cells)}
    cell = (
        table["cell_id"].fillna("<missing>").astype(str).map(lookup).fillna(0).to_numpy(dtype=np.int64)
    )
    numeric = context_numeric_values(table, transform.numeric_columns)
    numeric = (numeric - transform.numeric_mean) / transform.numeric_scale
    return cell, numeric.astype(np.float32)


class L1000ResponseModel(nn.Module):
    def __init__(
        self,
        n_bits: int,
        encoder_hidden: int,
        encoder_dim: int,
        n_cells: int,
        cell_dim: int,
        numeric_dim: int,
        fusion_hidden: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.chemical_encoder = ChemicalEncoder(n_bits, encoder_hidden, encoder_dim, dropout)
        self.cell_embedding = nn.Embedding(n_cells + 1, cell_dim)
        context_input = cell_dim + numeric_dim
        self.context_projector = nn.Sequential(
            nn.Linear(context_input, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(encoder_dim * 3, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, output_dim),
        )

    def forward(
        self,
        fingerprint: torch.Tensor,
        cell_index: torch.Tensor,
        numeric_context: torch.Tensor,
    ) -> torch.Tensor:
        chemical = self.chemical_encoder(fingerprint)
        context_input = torch.cat([self.cell_embedding(cell_index), numeric_context], dim=1)
        context = self.context_projector(context_input)
        return self.head(torch.cat([chemical, context, chemical * context], dim=1))


def build_model(
    config: dict[str, Any],
    n_cells: int,
    numeric_dim: int,
    output_dim: int,
) -> L1000ResponseModel:
    fingerprint = config["fingerprint"]
    pretrain = config["rna_pretraining"]
    return L1000ResponseModel(
        n_bits=int(fingerprint["n_bits"]),
        encoder_hidden=int(pretrain["encoder_hidden"]),
        encoder_dim=int(pretrain["encoder_dim"]),
        n_cells=n_cells,
        cell_dim=int(pretrain["cell_dim"]),
        numeric_dim=numeric_dim,
        fusion_hidden=int(pretrain["fusion_hidden"]),
        output_dim=output_dim,
        dropout=float(pretrain["dropout"]),
    )


def fit_model(
    model: L1000ResponseModel,
    fingerprint: np.ndarray,
    cell: np.ndarray,
    numeric: np.ndarray,
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
        torch.from_numpy(fingerprint),
        torch.from_numpy(cell),
        torch.from_numpy(numeric),
        torch.from_numpy(target),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        rows = 0
        for batch_fingerprint, batch_cell, batch_numeric, batch_target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch_fingerprint.to(device, non_blocking=True),
                batch_cell.to(device, non_blocking=True),
                batch_numeric.to(device, non_blocking=True),
            )
            loss = loss_function(prediction, batch_target.to(device, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(batch_target)
            rows += len(batch_target)
        history.append({"epoch": float(epoch + 1), "loss": loss_sum / max(rows, 1)})
    return history


@torch.no_grad()
def predict(
    model: L1000ResponseModel,
    fingerprint: np.ndarray,
    cell: np.ndarray,
    numeric: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    dataset = TensorDataset(
        torch.from_numpy(fingerprint), torch.from_numpy(cell), torch.from_numpy(numeric)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    predictions: list[np.ndarray] = []
    for batch_fingerprint, batch_cell, batch_numeric in loader:
        result = model(
            batch_fingerprint.to(device, non_blocking=True),
            batch_cell.to(device, non_blocking=True),
            batch_numeric.to(device, non_blocking=True),
        )
        predictions.append(result.cpu().numpy())
    return np.concatenate(predictions, axis=0)


def _flat_pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = left.astype(np.float64, copy=False).ravel()
    y = right.astype(np.float64, copy=False).ravel()
    x -= x.mean()
    y -= y.mean()
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denominator) if denominator > 0 else float("nan")


def _fit_pca(target: np.ndarray, requested_rank: int, seed: int) -> PCA:
    rank = min(int(requested_rank), target.shape[1], target.shape[0] - 1)
    if rank < 1:
        raise ValueError("PCA requires at least two training signatures")
    pca = PCA(n_components=rank, whiten=True, svd_solver="randomized", random_state=seed)
    pca.fit(target)
    return pca


def run_cv(
    table: pd.DataFrame,
    target: np.ndarray,
    fingerprints: np.ndarray,
    config: dict[str, Any],
    label_mode: str,
    device: torch.device,
    n_splits: int,
    epochs: int,
    batch_size: int,
    rank: int,
    seed: int,
) -> list[dict[str, Any]]:
    # Connectivity-level parent grouping prevents aliases, salts, and duplicate
    # L1000 perturbagen IDs for the same chemistry crossing a diagnostic fold.
    groups = table["parent_key"].astype(str).to_numpy()
    if np.unique(groups).size < n_splits:
        raise ValueError("Fewer compounds than requested GroupKFold splits")
    splitter = GroupKFold(n_splits=n_splits)
    rows: list[dict[str, Any]] = []
    for fold, (train_indices, valid_indices) in enumerate(splitter.split(table, groups=groups)):
        train = np.zeros(len(table), dtype=bool)
        train[train_indices] = True
        valid = ~train
        residualizer = fit_cell_residualizer(target[train], table.loc[train, "cell_id"])
        train_residual = apply_cell_residualizer(
            target[train], table.loc[train, "cell_id"], residualizer
        )
        valid_residual = apply_cell_residualizer(
            target[valid], table.loc[valid, "cell_id"], residualizer
        )
        pca = _fit_pca(train_residual, rank, seed + fold)
        train_latent = pca.transform(train_residual).astype(np.float32)
        valid_latent = pca.transform(valid_residual).astype(np.float32)

        context_transform = fit_context(table.loc[train])
        train_cell, train_numeric = transform_context(table.loc[train], context_transform)
        valid_cell, valid_numeric = transform_context(table.loc[valid], context_transform)
        train_fingerprint = fingerprints[train]
        if label_mode == "input-shuffle":
            train_fingerprint = permute_drug_fingerprints(
                groups, fingerprints, train, seed + 10_000 + fold
            )
        elif label_mode != "real":
            raise ValueError(f"Unknown label mode: {label_mode}")

        seed_everything(seed + fold)
        model = build_model(
            config,
            n_cells=len(context_transform.cells),
            numeric_dim=train_numeric.shape[1],
            output_dim=train_latent.shape[1],
        )
        history = fit_model(
            model,
            train_fingerprint,
            train_cell,
            train_numeric,
            train_latent,
            epochs,
            batch_size,
            float(config["rna_pretraining"]["learning_rate"]),
            float(config["rna_pretraining"]["weight_decay"]),
            device,
            seed + fold,
        )
        prediction = predict(
            model,
            fingerprints[valid],
            valid_cell,
            valid_numeric,
            device,
            batch_size * 2,
        )
        reconstructed = pca.inverse_transform(prediction).astype(np.float32)
        rows.append(
            {
                "fold": fold,
                "label_mode": label_mode,
                "n_train_rows": int(train.sum()),
                "n_valid_rows": int(valid.sum()),
                "n_train_compounds": int(np.unique(groups[train]).size),
                "n_valid_compounds": int(np.unique(groups[valid]).size),
                "pca_rank": int(train_latent.shape[1]),
                "pca_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
                "latent_rmse": float(np.sqrt(np.mean((prediction - valid_latent) ** 2))),
                "latent_flat_pcc": _flat_pearson(prediction, valid_latent),
                "latent_rowwise_pcc": rowwise_pearson_mean(prediction, valid_latent),
                "signature_residual_rowwise_pcc": rowwise_pearson_mean(
                    reconstructed, valid_residual
                ),
                "last_train_loss": float(history[-1]["loss"]),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def run_full_fit(
    table: pd.DataFrame,
    target: np.ndarray,
    fingerprints: np.ndarray,
    probe_ids: np.ndarray,
    config: dict[str, Any],
    scope: str,
    label_mode: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    rank: int,
    seed: int,
    output: Path,
) -> tuple[Path, Path, list[dict[str, float]], dict[str, Any]]:
    groups = table["parent_key"].astype(str).to_numpy()
    residualizer = fit_cell_residualizer(target, table["cell_id"])
    residual = apply_cell_residualizer(target, table["cell_id"], residualizer)
    pca = _fit_pca(residual, rank, seed)
    latent = pca.transform(residual).astype(np.float32)
    context_transform = fit_context(table)
    cell, numeric = transform_context(table, context_transform)
    fit_fingerprints = fingerprints
    if label_mode == "input-shuffle":
        fit_fingerprints = permute_drug_fingerprints(
            groups, fingerprints, np.ones(len(table), dtype=bool), seed + 90_001
        )
    elif label_mode != "real":
        raise ValueError(f"Unknown label mode: {label_mode}")

    seed_everything(seed)
    model = build_model(
        config,
        n_cells=len(context_transform.cells),
        numeric_dim=numeric.shape[1],
        output_dim=latent.shape[1],
    )
    history = fit_model(
        model,
        fit_fingerprints,
        cell,
        numeric,
        latent,
        epochs,
        batch_size,
        float(config["rna_pretraining"]["learning_rate"]),
        float(config["rna_pretraining"]["weight_decay"]),
        device,
        seed,
    )
    scope_slug = scope.replace("external-", "").replace("-", "_")
    mode_slug = label_mode.replace("-", "_")
    checkpoint = output / f"l1000_{mode_slug}_{scope_slug}_encoder.pt"
    # Real and shuffled jobs are commonly run concurrently on separate GPUs;
    # never let them race on a shared target-transform artifact.
    pca_path = output / f"l1000_target_pca_{mode_slug}_{scope_slug}.npz"
    encoder_state = copy.deepcopy(model.chemical_encoder).cpu().state_dict()
    torch.save(
        {
            # Keep the same key as src.rna_pretrain so the protein trainer can
            # compare OP3 and L1000 encoders without format-specific logic.
            "chemical_encoder": encoder_state,
            "scope": scope,
            "arm": label_mode,
            "source_dataset": "L1000FWD CD signatures (42,809 x 978)",
            "architecture": {
                "n_bits": int(config["fingerprint"]["n_bits"]),
                "radius": int(config["fingerprint"]["radius"]),
                "encoder_hidden": int(config["rna_pretraining"]["encoder_hidden"]),
                "encoder_dim": int(config["rna_pretraining"]["encoder_dim"]),
                "dropout": float(config["rna_pretraining"]["dropout"]),
            },
        },
        checkpoint,
    )
    np.savez_compressed(
        pca_path,
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        explained_variance=pca.explained_variance_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        probe_ids=np.asarray(probe_ids, dtype=str),
    )
    details = {
        "pca_rank": int(latent.shape[1]),
        "pca_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "n_cells": len(context_transform.cells),
        "numeric_context": context_transform.numeric_columns,
        "last_train_loss": float(history[-1]["loss"]),
    }
    return checkpoint, pca_path, history, details


def _data_dir_from_args(args: argparse.Namespace) -> Path:
    return Path(args.data_dir).expanduser().resolve()


def train(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    # This host is shared by several GPU experiments.  Randomized SVD and
    # PyTorch CPU kernels otherwise claim every logical CPU and can make the
    # machine unusable even though the actual neural network lives on one GPU.
    cpu_threads = max(1, int(args.cpu_threads))
    threadpool_limits(limits=cpu_threads)
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting inter-op threads only once per process.
        pass
    config = load_config(args.config)
    source_hashes = {
        name: sha256(Path(__file__).with_name(name))
        for name in ("l1000_pretrain.py", "models.py", "common.py")
    }
    seed = int(config.get("seed", 42)) if args.seed is None else int(args.seed)
    seed_everything(seed)
    data_dir = _data_dir_from_args(args)
    verify_report = verify_public_files(data_dir, verify_hashes=not args.skip_hash)
    output_root = Path(config["paths"]["output_root"]).expanduser().resolve()
    output = output_root / "models" / "l1000_pretraining"
    if _inside(output, data_dir):
        raise ValueError("Learned artifacts must not be written inside the read-only data directory")
    output.mkdir(parents=True, exist_ok=True)

    table = load_signature_table(data_dir)
    table, scope_counts = apply_scope(
        table,
        args.scope,
        Path(config["paths"]["goai_chemical_map"]) if args.scope == "external-minus-goai" else None,
    )
    table = deterministic_subsample_drugs(table, args.max_drugs, seed)
    target, probe_ids = load_expression(data_dir, table["_matrix_index"].to_numpy())
    fingerprints, _ = build_fingerprints(
        table,
        radius=int(config["fingerprint"]["radius"]),
        n_bits=int(config["fingerprint"]["n_bits"]),
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but torch.cuda.is_available() is false")

    suffix = f"{args.label_mode.replace('-', '_')}_{args.scope.replace('external-', '').replace('-', '_')}"
    cv_path = output / f"l1000_cv_{suffix}.csv"
    if not args.skip_full:
        full_targets = (
            output / f"l1000_{suffix}_encoder.pt",
            output / f"l1000_target_pca_{suffix}.npz",
            output / f"l1000_history_{suffix}.csv",
            output / f"manifest_{suffix}.json",
        )
        existing_full = [str(path) for path in full_targets if path.exists()]
        if existing_full and not args.force_full:
            raise RuntimeError(
                "Refusing to overwrite completed full-fit artifacts; use a new output "
                f"scope/label or --force-full. Existing: {existing_full}"
            )
    if args.skip_cv and cv_path.exists():
        raise RuntimeError(
            f"Refusing to leave a stale diagnostic at {cv_path}; move it aside or run CV"
        )
    if not args.skip_cv and cv_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing diagnostic: {cv_path}")
    cv_rows: list[dict[str, Any]] = []
    if not args.skip_cv:
        cv_rows = run_cv(
            table,
            target,
            fingerprints,
            config,
            args.label_mode,
            device,
            args.cv_folds,
            args.cv_epochs,
            args.batch_size,
            args.rank,
            seed,
        )
        pd.DataFrame(cv_rows).to_csv(cv_path, index=False)

    checkpoint: Path | None = None
    pca_path: Path | None = None
    full_details: dict[str, Any] = {}
    history: list[dict[str, float]] = []
    if not args.skip_full:
        checkpoint, pca_path, history, full_details = run_full_fit(
            table,
            target,
            fingerprints,
            probe_ids,
            config,
            args.scope,
            args.label_mode,
            device,
            args.epochs,
            args.batch_size,
            args.rank,
            seed,
            output,
        )
        pd.DataFrame(history).to_csv(output / f"l1000_history_{suffix}.csv", index=False)

    manifest = {
        "status": "complete",
        "knowledge_track": "open-knowledge",
        "source_dataset": "L1000FWD CD signatures (42,809 x 978 landmarks)",
        "source_urls": {name: record["url"] for name, record in PUBLIC_FILES.items()},
        "source_sha256": {
            name: record["sha256"] for name, record in verify_report["files"].items()
        },
        "scope": args.scope,
        "label_mode": args.label_mode,
        # Deliberately record only counts: no private GOAI structures or names
        # are copied into an artifact derived from the public dataset.
        "scope_counts": scope_counts,
        "n_rows_used": len(table),
        "n_pert_ids_used": int(table["pert_id"].nunique()),
        "n_parent_structures_used": int(table["parent_key"].nunique()),
        "n_landmark_genes": int(target.shape[1]),
        "context_fields": ["cell_id", "pert_time", "log1p(pert_dose)"],
        "target_transform": "fold-train cell residualization -> fold-train whitened PCA",
        "cv_protocol": "five-fold standardized parent-connectivity GroupKFold; validation drug fingerprints are always real",
        "cv_metrics": cv_rows,
        "cv_artifact": str(cv_path) if cv_rows else None,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "target_pca": str(pca_path) if pca_path else None,
        "full_fit": full_details,
        "epochs": args.epochs,
        "cv_epochs": args.cv_epochs,
        "cv_folds": args.cv_folds,
        "batch_size": args.batch_size,
        "cpu_threads": cpu_threads,
        "max_parent_structures": args.max_drugs,
        "skip_cv": bool(args.skip_cv),
        "skip_full": bool(args.skip_full),
        "skip_hash": bool(args.skip_hash),
        "seed": seed,
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "rdkit": rdkit.__version__,
        "source_code_sha256_at_process_start": source_hashes,
    }
    # Full-fit and CV-only are independent stages. Their provenance must never
    # overwrite each other when diagnostics are launched after checkpointing.
    stage = "cv_only" if args.skip_full else ("full_only" if args.skip_cv else "full_and_cv")
    manifest["stage"] = stage
    if stage == "full_only":
        manifest_name = f"manifest_{suffix}.json"  # stable downstream checkpoint manifest
    else:
        manifest_name = f"manifest_{stage}_{suffix}.json"
    manifest_path = output / manifest_name
    write_json(manifest_path, manifest)
    return {"status": "complete", "manifest": str(manifest_path), "checkpoint": manifest["checkpoint"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("download", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
        if command == "download":
            child.add_argument("--force", action="store_true")
        else:
            child.add_argument("--skip-hash", action="store_true")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    train_parser.add_argument(
        "--scope", choices=("external-minus-goai", "external-all"), default="external-minus-goai"
    )
    train_parser.add_argument("--label-mode", choices=("real", "input-shuffle"), default="real")
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--rank", type=int, default=64)
    train_parser.add_argument("--cv-folds", type=int, default=5)
    train_parser.add_argument("--cv-epochs", type=int, default=25)
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--batch-size", type=int, default=512)
    train_parser.add_argument("--cpu-threads", type=int, default=2)
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--max-drugs", type=int)
    train_parser.add_argument("--skip-cv", action="store_true")
    train_parser.add_argument("--skip-full", action="store_true")
    train_parser.add_argument("--skip-hash", action="store_true")
    train_parser.add_argument("--force-full", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "download":
        result = download_public_files(_data_dir_from_args(args), force=args.force)
    elif args.command == "verify":
        result = verify_public_files(_data_dir_from_args(args), verify_hashes=not args.skip_hash)
    else:
        result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
