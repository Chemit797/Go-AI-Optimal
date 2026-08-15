"""Build frozen ChemBERTa features and a deterministic row-shuffled control."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chemical-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-deps", default=".runtime/chembert")
    parser.add_argument("--model", default="DeepChem/ChemBERTa-77M-MLM")
    parser.add_argument(
        "--revision",
        default="ed8a5374f2024ec8da53760af91a33fb8f6a15ff",
        help="Pinned Hugging Face commit; never use a moving main revision for scored features",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--shuffle-seed", type=int, default=991)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dependency_path = Path(args.runtime_deps).resolve()
    sys.path.insert(0, str(dependency_path))
    from transformers import AutoModel, AutoTokenizer

    source = Path(args.chemical_map).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(source, sep="\t", keep_default_na=False)
    if table["raw_name"].duplicated().any():
        raise ValueError("chemical map contains duplicate raw_name values")
    smiles_column = "isomeric_smiles" if "isomeric_smiles" in table else "canonical_smiles"
    usable = table[smiles_column].astype(str).str.len().gt(0) & table["status"].astype(str).eq("resolved")
    names = table["raw_name"].astype(str).tolist()
    smiles = table[smiles_column].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModel.from_pretrained(args.model, revision=args.revision).to(torch.device(args.device)).eval()
    hidden = int(model.config.hidden_size)
    values = np.zeros((len(table), hidden), dtype=np.float32)
    valid_rows = np.flatnonzero(usable.to_numpy())
    with torch.inference_mode():
        for start in range(0, len(valid_rows), args.batch_size):
            rows = valid_rows[start:start + args.batch_size]
            encoded = tokenizer(
                [smiles[index] for index in rows],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            states = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(states.dtype)
            pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            values[rows] = pooled.cpu().numpy().astype(np.float32)

    columns = [f"chemberta_{index:04d}" for index in range(hidden)]
    real = pd.DataFrame(values, columns=columns)
    real.insert(0, "raw_name", names)
    real_path = output / "chemberta_real.tsv"
    real.to_csv(real_path, sep="\t", index=False)

    rng = np.random.default_rng(args.shuffle_seed)
    permutation = rng.permutation(len(table))
    shuffled = pd.DataFrame(values[permutation], columns=columns)
    shuffled.insert(0, "raw_name", names)
    shuffled_path = output / "chemberta_shuffled.tsv"
    shuffled.to_csv(shuffled_path, sep="\t", index=False)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "model_revision": args.revision,
        "transformers_version": __import__("transformers").__version__,
        "pooling": "attention-mask mean pooling",
        "max_length": args.max_length,
        "frozen": True,
        "source": str(source),
        "source_sha256": _sha256(source),
        "smiles_column": smiles_column,
        "rows": len(table),
        "resolved_rows": int(usable.sum()),
        "embedding_dim": hidden,
        "device": args.device,
        "shuffle_seed": args.shuffle_seed,
        "shuffle_permutation": permutation.tolist(),
        "real_path": str(real_path),
        "real_sha256": _sha256(real_path),
        "shuffled_path": str(shuffled_path),
        "shuffled_sha256": _sha256(shuffled_path),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("model", "rows", "resolved_rows", "embedding_dim", "real_path", "shuffled_path")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
