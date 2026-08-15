from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(destination)


def morgan_fingerprint(smiles: str, radius: int, n_bits: int) -> np.ndarray:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprint = generator.GetFingerprint(molecule)
    return np.asarray(fingerprint, dtype=np.float32)


def parent_connectivity_key(smiles: str) -> str | None:
    """Return a salt/formulation-tolerant parent structure key."""
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    parent = rdMolStandardize.FragmentParent(molecule)
    parent = rdMolStandardize.Uncharger().uncharge(parent)
    key = Chem.MolToInchiKey(parent)
    return key.split("-", 1)[0] if key else None


def pearson_flat(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    observed = mask.astype(bool)
    if int(observed.sum()) < 2:
        return float("nan")
    x = prediction[observed].astype(np.float64, copy=False)
    y = truth[observed].astype(np.float64, copy=False)
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denominator) if denominator > 0 else float("nan")


def rowwise_pearson_mean(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction = prediction.astype(np.float64, copy=False)
    truth = truth.astype(np.float64, copy=False)
    prediction = prediction - prediction.mean(axis=1, keepdims=True)
    truth = truth - truth.mean(axis=1, keepdims=True)
    numerator = np.sum(prediction * truth, axis=1)
    denominator = np.sqrt(np.sum(prediction**2, axis=1) * np.sum(truth**2, axis=1))
    valid = denominator > 0
    return float(np.mean(numerator[valid] / denominator[valid])) if valid.any() else float("nan")
