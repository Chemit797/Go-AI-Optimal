#!/usr/bin/env python3
"""Build explicit exact, parent-normalized, and zero-risk chemical views.

The competition names sometimes denote salts or mixtures.  This script never
silently overwrites the exact registry.  It creates labeled ablation tables so
that a parent/component representation can only be selected intentionally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exact-map",
        type=Path,
        default=PROJECT / "data/processed/chemical_entity_map.tsv",
    )
    parser.add_argument(
        "--parent-views",
        type=Path,
        default=PROJECT / "data/processed/entities/chemical_parent_normalized_views.tsv",
    )
    parser.add_argument(
        "--identity-risks",
        type=Path,
        default=PROJECT / "resources/entities/chemical_identity_risk_review.tsv",
        help="Reviewed identities to suppress only in the zero_risky ablation",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT / "data/processed/chemical_views"
    )
    args = parser.parse_args()

    exact = pd.read_csv(args.exact_map, sep="\t", dtype=str, keep_default_na=False)
    parents = pd.read_csv(args.parent_views, sep="\t", dtype=str, keep_default_na=False)
    risks = pd.read_csv(args.identity_risks, sep="\t", dtype=str, keep_default_na=False)
    if (
        exact["raw_name"].duplicated().any()
        or parents["raw_name"].duplicated().any()
        or risks["raw_name"].duplicated().any()
    ):
        raise ValueError("Chemical view inputs contain duplicate raw_name rows")
    required_risk_columns = {"raw_name", "risk_class", "zero_risky", "evidence_path"}
    missing_columns = sorted(required_risk_columns - set(risks.columns))
    if missing_columns:
        raise ValueError(f"Identity-risk review is missing columns: {missing_columns}")
    risk_gate = risks["zero_risky"].str.strip().str.casefold()
    if not risk_gate.isin({"true", "false"}).all():
        raise ValueError("Identity-risk zero_risky values must be true or false")
    risks = risks.loc[risk_gate.eq("true")].copy()
    missing = sorted((set(parents["raw_name"]) | set(risks["raw_name"])) - set(exact["raw_name"]))
    if missing:
        raise ValueError(f"Reviewed chemical names are absent from exact map: {missing}")
    missing_evidence = sorted(
        row["raw_name"]
        for row in risks.to_dict(orient="records")
        if not (PROJECT / row["evidence_path"]).is_file()
    )
    if missing_evidence:
        raise ValueError(f"Identity-risk evidence snapshots are missing for: {missing_evidence}")

    parent = exact.copy().set_index("raw_name", drop=False)
    for row in parents.to_dict(orient="records"):
        name = row["raw_name"]
        canonical_id = row["parent_canonical_id"]
        if not canonical_id.startswith("pubchem:"):
            raise ValueError(f"Unsupported parent canonical ID for {name}: {canonical_id}")
        parent.loc[name, "query_name"] = row["parent_name"]
        parent.loc[name, "cid"] = canonical_id.split(":", 1)[1]
        parent.loc[name, "title"] = row["parent_name"]
        parent.loc[name, "inchikey"] = row["parent_inchikey"]
        parent.loc[name, "isomeric_smiles"] = row["parent_isomeric_smiles"]
        parent.loc[name, "canonical_smiles"] = row["parent_isomeric_smiles"]
        parent.loc[name, "status"] = "resolved"
        parent.loc[name, "error"] = ""
        parent.loc[name, "source"] = "reviewed parent/component ablation view"
        parent.loc[name, "retrieved_utc"] = "2026-08-13"
    parent = parent.reset_index(drop=True)

    zero = exact.copy().set_index("raw_name", drop=False)
    risky = risks["raw_name"].tolist()
    for name in risky:
        for column in (
            "cid", "title", "isomeric_smiles", "canonical_smiles", "inchikey"
        ):
            zero.loc[name, column] = ""
        zero.loc[name, "query_name"] = name
        zero.loc[name, "status"] = "unresolved"
        zero.loc[name, "error"] = "intentional zero-risk ablation"
        zero.loc[name, "source"] = "GOAI structure-view gate"
        zero.loc[name, "retrieved_utc"] = "2026-08-13"
    zero = zero.reset_index(drop=True)

    # Structure-only negative control: keep the row identity, registry flags,
    # model architecture and dimensions fixed while deranging the resolved
    # non-control structures across chemical names.
    shuffled = exact.copy()
    resolved = (
        exact["status"].astype(str).eq("resolved")
        & ~exact["is_control"].astype(str).str.casefold().isin({"true", "1", "yes"})
        & exact["isomeric_smiles"].astype(str).str.len().gt(0)
    ).to_numpy()
    recipients = np.flatnonzero(resolved)
    if len(recipients) < 2:
        raise ValueError("Need at least two resolved treatment structures for shuffled control")
    rng = np.random.default_rng(1701)
    rotation = int(rng.integers(1, len(recipients)))
    donors = np.roll(recipients, rotation)
    structure_columns = [
        "query_name",
        "cid",
        "title",
        "isomeric_smiles",
        "canonical_smiles",
        "inchikey",
        "status",
        "error",
        "source",
        "retrieved_utc",
    ]
    shuffled.loc[recipients, structure_columns] = exact.loc[
        donors, structure_columns
    ].to_numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "exact": args.output_dir / "chemical_entity_map_exact.tsv",
        "parent_normalized": args.output_dir / "chemical_entity_map_parent_normalized.tsv",
        "zero_risky": args.output_dir / "chemical_entity_map_zero_risky.tsv",
        "exact_shuffled": args.output_dir / "chemical_entity_map_exact_shuffled.tsv",
    }
    exact.to_csv(outputs["exact"], sep="\t", index=False)
    parent.to_csv(outputs["parent_normalized"], sep="\t", index=False)
    zero.to_csv(outputs["zero_risky"], sep="\t", index=False)
    shuffled.to_csv(outputs["exact_shuffled"], sep="\t", index=False)
    payload = {
        "schema_version": "goai.chemical-structure-views.v3",
        "identity_policy": (
            "exact is default; parent/component substitutions and identity-risk zeroing "
            "are separate, explicitly reviewed ablations"
        ),
        "parent_raw_names": parents["raw_name"].tolist(),
        "zero_risky_raw_names": risky,
        "shuffled_exact": {
            "seed": 1701,
            "rotation": rotation,
            "permutation": [
                {
                    "recipient": str(exact.loc[recipient, "raw_name"]),
                    "donor": str(exact.loc[donor, "raw_name"]),
                }
                for recipient, donor in zip(recipients, donors)
            ],
        },
        # Compatibility alias for readers of the v1 manifest.  In v2 this means
        # identity-risk rows, not merely rows with a parent/component substitute.
        "risky_raw_names": risky,
        "inputs": {
            "exact_map": {"path": str(args.exact_map.resolve()), "sha256": _sha256(args.exact_map)},
            "parent_views": {"path": str(args.parent_views.resolve()), "sha256": _sha256(args.parent_views)},
            "identity_risks": {"path": str(args.identity_risks.resolve()), "sha256": _sha256(args.identity_risks)},
        },
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in outputs.items()
        },
    }
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest.with_suffix(".json.sha256").write_text(_sha256(manifest) + "\n", encoding="ascii")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
