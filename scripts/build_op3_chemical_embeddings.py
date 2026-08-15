#!/usr/bin/env python3
"""Build frozen OP3 chemical embeddings with a paired shuffled-RNA control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RNA_TRANSFER_ROOT = PROJECT_ROOT.parent / "goai-rna-transfer"
if str(RNA_TRANSFER_ROOT) not in sys.path:
    sys.path.insert(0, str(RNA_TRANSFER_ROOT))

from src.models import ChemicalEncoder  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def morgan_fingerprints(
    mapping: pd.DataFrame,
    *,
    smiles_column: str,
    radius: int,
    n_bits: int,
) -> tuple[np.ndarray, np.ndarray]:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    )
    matrix = np.zeros((len(mapping), n_bits), dtype=np.float32)
    resolved = np.zeros(len(mapping), dtype=bool)
    for row, record in mapping.iterrows():
        smiles = str(record[smiles_column]).strip()
        if str(record["status"]).strip() != "resolved" or not smiles:
            continue
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(
                f"Resolved chemical has invalid SMILES: {record['raw_name']!r}"
            )
        matrix[row] = np.asarray(generator.GetFingerprint(molecule), dtype=np.float32)
        resolved[row] = True
    return matrix, resolved


def load_encoder(path: Path, *, expected_arm: str, n_bits: int) -> ChemicalEncoder:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    arm = str(payload.get("arm", "")).strip().casefold()
    if arm != expected_arm:
        raise ValueError(f"Expected encoder arm {expected_arm!r}, found {arm!r}")
    state = payload.get("chemical_encoder")
    if not isinstance(state, dict):
        raise ValueError(f"Encoder checkpoint lacks chemical_encoder: {path}")
    first = state.get("network.0.weight")
    last = state.get("network.4.weight")
    if first is None or last is None:
        raise ValueError(f"Encoder checkpoint has an unsupported architecture: {path}")
    hidden, checkpoint_bits = map(int, first.shape)
    embedding_dim = int(last.shape[0])
    if checkpoint_bits != n_bits or int(last.shape[1]) != hidden:
        raise ValueError("Encoder dimensions do not match the requested fingerprint")
    encoder = ChemicalEncoder(n_bits, hidden, embedding_dim, dropout=0.0)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    return encoder


@torch.inference_mode()
def encode(encoder: ChemicalEncoder, fingerprints: np.ndarray) -> np.ndarray:
    values = encoder(torch.from_numpy(fingerprints)).cpu().numpy().astype(np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("OP3 chemical embeddings are invalid")
    return values


def write_table(path: Path, names: pd.Series, values: np.ndarray) -> None:
    columns = [f"op3_{index:04d}" for index in range(values.shape[1])]
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, "raw_name", names.astype(str).to_numpy())
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--real-encoder", type=Path, required=True)
    parser.add_argument("--shuffled-encoder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smiles-column", default="canonical_smiles")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    args = parser.parse_args()

    source = args.source.resolve()
    real_encoder_path = args.real_encoder.resolve()
    shuffled_encoder_path = args.shuffled_encoder.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(source, sep="\t", dtype=str, keep_default_na=False)
    required = {"raw_name", "status", args.smiles_column}
    missing = sorted(required.difference(mapping.columns))
    if missing:
        raise ValueError(f"Chemical mapping lacks columns: {missing}")
    if mapping.empty or mapping["raw_name"].duplicated().any():
        raise ValueError("Chemical mapping must contain unique non-empty rows")
    fingerprints, resolved = morgan_fingerprints(
        mapping,
        smiles_column=args.smiles_column,
        radius=args.radius,
        n_bits=args.n_bits,
    )
    real_encoder = load_encoder(
        real_encoder_path, expected_arm="real", n_bits=args.n_bits
    )
    shuffled_encoder = load_encoder(
        shuffled_encoder_path, expected_arm="shuffle", n_bits=args.n_bits
    )
    real = encode(real_encoder, fingerprints)
    shuffled = encode(shuffled_encoder, fingerprints)
    # Unresolved structures are explicit zeros, never learned constants.
    real[~resolved] = 0.0
    shuffled[~resolved] = 0.0

    real_path = output / "op3_real.tsv"
    shuffled_path = output / "op3_shuffled.tsv"
    write_table(real_path, mapping["raw_name"], real)
    write_table(shuffled_path, mapping["raw_name"], shuffled)
    manifest = {
        "model": "OP3-RNA-ChemicalEncoder",
        "model_revision": sha256(real_encoder_path),
        "frozen": True,
        "pooling": "pretrained Morgan-2048 MLP output",
        "source": str(source),
        "source_sha256": sha256(source),
        "smiles_column": args.smiles_column,
        "rows": int(len(mapping)),
        "resolved_rows": int(resolved.sum()),
        "embedding_dim": int(real.shape[1]),
        "fingerprint_radius": args.radius,
        "fingerprint_bits": args.n_bits,
        "real_encoder": str(real_encoder_path),
        "real_encoder_sha256": sha256(real_encoder_path),
        "shuffled_encoder": str(shuffled_encoder_path),
        "shuffled_encoder_sha256": sha256(shuffled_encoder_path),
        "negative_control": "same structures encoded by the RNA-target-shuffled OP3 encoder",
        "real_path": str(real_path),
        "real_sha256": sha256(real_path),
        "shuffled_path": str(shuffled_path),
        "shuffled_sha256": sha256(shuffled_path),
    }
    manifest_path = output / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
