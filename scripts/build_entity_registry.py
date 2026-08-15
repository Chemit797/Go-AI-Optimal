#!/usr/bin/env python3
"""Build reviewed GOAI chemical/strain registries and a train-only support manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goai_response.entities import (  # noqa: E402
    build_support_manifest,
    load_registry,
    manifest_sha256,
    sha256_file,
    validate_registry,
    write_json_with_hash,
)


CHEMICAL = "perturbation_no_concentration"
STRAIN = "Strains"
RISKY_PROXIES = {
    "Hoechst 33258": "formulation_unspecified_parent_normalized",
    "Oligomycin": "mixture_to_component",
    "Tunicamycin": "mixture_to_component",
}
REVIEW_PROMOTION_STATUSES = {"verified", "high_confidence_candidate", "proxy"}
CYCLOPIAZONIC = {
    "cid": "54682463",
    "title": "Cyclopiazonic acid",
    "isomeric_smiles": "CC(=O)C1=C([C@@H]2[C@@H]3[C@@H](CC4=C5C3=CNC5=CC=C4)C(N2C1=O)(C)C)O",
    "canonical_smiles": "CC(=O)C1=C(C2C3C(CC4=C5C3=CNC5=CC=C4)C(N2C1=O)(C)C)O",
    "inchikey": "SZINUGQCTHLQAZ-DQYPLSBCSA-N",
    "status": "resolved",
    "source": "PubChem PUG REST",
    "retrieved_utc": "2026-08-13",
}


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized or "unknown"


def _read_metadata(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path, low_memory=False) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    required = {CHEMICAL, STRAIN, "chemical_role", "strain_role"}
    missing = sorted(required.difference(combined.columns))
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")
    return combined


def _roles(metadata: pd.DataFrame, entity_column: str, role_column: str) -> dict[str, str]:
    grouped = metadata.groupby(entity_column, dropna=False)[role_column].agg(
        lambda values: sorted(set(str(value).strip() for value in values))
    )
    conflicts = grouped[grouped.map(len) != 1]
    if not conflicts.empty:
        raise ValueError(f"Conflicting {role_column} assignments: {conflicts.to_dict()}")
    return {str(entity): roles[0] for entity, roles in grouped.items()}


def build_chemical_registry(
    source: pd.DataFrame,
    metadata: pd.DataFrame,
    second_source_review: pd.DataFrame,
) -> pd.DataFrame:
    source = source.copy().fillna("")
    second_source_review = second_source_review.copy().fillna("")
    required_review = {
        "raw_name", "promotion_status", "review_outcome", "secondary_source",
        "secondary_id", "secondary_name", "secondary_inchikey",
        "selected_evidence_path", "secondary_source_url", "notes",
    }
    missing_review_columns = sorted(required_review.difference(second_source_review.columns))
    if missing_review_columns:
        raise ValueError(f"Second-source review missing columns: {missing_review_columns}")
    if second_source_review["raw_name"].duplicated().any():
        duplicates = second_source_review.loc[
            second_source_review["raw_name"].duplicated(False), "raw_name"
        ].tolist()
        raise ValueError(f"Duplicate second-source review rows: {duplicates}")
    invalid_promotions = sorted(
        set(second_source_review["promotion_status"]) - REVIEW_PROMOTION_STATUSES
    )
    if invalid_promotions:
        raise ValueError(f"Invalid second-source promotion statuses: {invalid_promotions}")
    review_by_name = second_source_review.set_index("raw_name").to_dict(orient="index")
    source.loc[source["raw_name"].eq("Cyclopiazonic acid"), list(CYCLOPIAZONIC)] = [
        CYCLOPIAZONIC[column] for column in CYCLOPIAZONIC
    ]
    roles = _roles(metadata, CHEMICAL, "chemical_role")
    source_names = set(source["raw_name"].astype(str))
    metadata_names = set(metadata[CHEMICAL].astype(str))
    if source_names != metadata_names:
        raise ValueError(
            f"Chemical map/metadata mismatch: missing={sorted(metadata_names-source_names)}, "
            f"extra={sorted(source_names-metadata_names)}"
        )

    rows = []
    for row in source.to_dict(orient="records"):
        raw = str(row["raw_name"])
        cid = str(row.get("cid", "")).strip()
        review = review_by_name.get(raw, {})
        is_control = raw.casefold() in {"water", "dmso"}
        is_qc = raw.casefold() == "quality control"
        if raw in RISKY_PROXIES:
            if review and review["promotion_status"] != "proxy":
                raise ValueError(
                    f"{raw}: proxy policy conflicts with review promotion "
                    f"{review['promotion_status']!r}"
                )
            status = "proxy"
            evidence = "D_proxy_assumption"
            canonical_id = f"goai-chemical:{_slug(raw)}"
            canonical_name = raw
            proxy_target = f"pubchem:{cid}" if cid else str(row.get("query_name", ""))
            formulation = RISKY_PROXIES[raw]
        elif str(row.get("status", "")).strip() == "resolved" and cid:
            requested_status = str(review.get("promotion_status", "")).strip()
            if requested_status == "verified":
                primary_key = str(row.get("inchikey", "")).strip()
                secondary_key = str(review.get("secondary_inchikey", "")).strip()
                if not primary_key or primary_key != secondary_key:
                    raise ValueError(
                        f"{raw}: verified promotion requires matching primary/secondary "
                        f"InChIKeys, found {primary_key!r} vs {secondary_key!r}"
                    )
                status = "verified"
                evidence = "A_verified"
            else:
                status = "high_confidence_candidate"
                evidence = "C_database_lookup"
            canonical_id = f"pubchem:{cid}"
            canonical_name = str(row.get("title", "")) or raw
            proxy_target = ""
            formulation = "exact_database_candidate"
        else:
            status = "unresolved"
            evidence = "E_unresolved"
            canonical_id = f"goai-chemical:{_slug(raw)}"
            canonical_name = raw
            proxy_target = ""
            formulation = "quality_control" if is_qc else "unresolved"
        source_url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" if cid else ""
        evidence_snapshot = str(review.get("selected_evidence_path", "")).strip()
        evidence_snapshot_path = ""
        evidence_snapshot_sha256 = ""
        if evidence_snapshot:
            candidate = Path(evidence_snapshot)
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"Second-source evidence snapshot for {raw!r} is missing: {candidate}"
                )
            evidence_snapshot_path = str(candidate.resolve())
            evidence_snapshot_sha256 = sha256_file(candidate)
        rows.append(
            {
                "raw_name": raw,
                "canonical_id": canonical_id,
                "canonical_name": canonical_name,
                "query_name": str(row.get("query_name", "")),
                "role": roles[raw],
                "is_control": is_control,
                "is_quality_control": is_qc,
                "mapping_status": status,
                "evidence_tier": evidence,
                "pubchem_cid": cid,
                "inchikey": str(row.get("inchikey", "")),
                "isomeric_smiles": str(row.get("isomeric_smiles", "")),
                "canonical_smiles": str(row.get("canonical_smiles", "")),
                "formulation_class": formulation,
                "proxy_target": proxy_target,
                "source_url": source_url,
                "source_version": str(row.get("source", "")),
                "retrieved_utc": str(row.get("retrieved_utc", "")),
                "notes": str(row.get("error", "")),
                "identity_review_outcome": str(review.get("review_outcome", "")),
                "secondary_source": str(review.get("secondary_source", "")),
                "secondary_id": str(review.get("secondary_id", "")),
                "secondary_name": str(review.get("secondary_name", "")),
                "secondary_inchikey": str(review.get("secondary_inchikey", "")),
                "secondary_source_url": str(review.get("secondary_source_url", "")),
                "secondary_review_notes": str(review.get("notes", "")),
                "evidence_snapshot_path": evidence_snapshot_path,
                "evidence_snapshot_sha256": evidence_snapshot_sha256,
            }
        )
    return pd.DataFrame(rows).sort_values("raw_name").reset_index(drop=True)


def build_strain_registry(source: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    source = source.copy().fillna("")
    roles = _roles(metadata, STRAIN, "strain_role")
    source_names = set(source["strain_code"].astype(str))
    metadata_names = set(metadata[STRAIN].astype(str))
    if source_names != metadata_names:
        raise ValueError(
            f"Strain source/metadata mismatch: missing={sorted(metadata_names-source_names)}, "
            f"extra={sorted(source_names-metadata_names)}"
        )
    source["role"] = source["strain_code"].map(roles)
    return source.sort_values("strain_code").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-train-val", type=Path,
        default=PROJECT_ROOT / "WAYB_WAYC_metadata_train_val.csv",
    )
    parser.add_argument(
        "--metadata-test", type=Path,
        default=PROJECT_ROOT / "WAYB_WAYC_metadata_test.csv",
    )
    parser.add_argument(
        "--chemical-map", type=Path,
        default=PROJECT_ROOT / "data/processed/chemical_entity_map.tsv",
    )
    parser.add_argument(
        "--strain-source", type=Path,
        default=PROJECT_ROOT / "resources/entities/strain_identity_candidates.tsv",
    )
    parser.add_argument(
        "--chemical-second-source-review", type=Path,
        default=PROJECT_ROOT / "resources/entities/chemical_second_source_review.tsv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data/processed/entities",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_train_val = pd.read_csv(args.metadata_train_val, low_memory=False)
    metadata = _read_metadata([args.metadata_train_val, args.metadata_test])
    chemical = build_chemical_registry(
        pd.read_csv(args.chemical_map, sep="\t", dtype=str, keep_default_na=False),
        metadata,
        pd.read_csv(
            args.chemical_second_source_review,
            sep="\t", dtype=str, keep_default_na=False,
        ),
    )
    strain = build_strain_registry(
        pd.read_csv(args.strain_source, sep="\t", dtype=str, keep_default_na=False), metadata
    )
    # Every registry row carries the exact reviewed input snapshot hash.  This
    # keeps row-level provenance available after registries are copied outside
    # this build directory; source URLs alone are not a reproducibility record.
    chemical["source_snapshot_sha256"] = sha256_file(args.chemical_map)
    strain["source_snapshot_sha256"] = sha256_file(args.strain_source)
    strain["evidence_snapshot_sha256"] = ""
    strain_evidence_manifest = (
        PROJECT_ROOT / "data/external/strain_identity_evidence/manifest.json"
    )
    if strain_evidence_manifest.is_file():
        selected = strain["evidence_snapshot_path"].astype(str).str.strip().ne("")
        strain.loc[selected, "evidence_snapshot_sha256"] = sha256_file(
            strain_evidence_manifest
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chemical_path = args.output_dir / "chemical_registry.tsv"
    strain_path = args.output_dir / "strain_registry.tsv"
    chemical.to_csv(chemical_path, sep="\t", index=False)
    strain.to_csv(strain_path, sep="\t", index=False)

    registries = {
        "chemical": load_registry(chemical_path, "chemical"),
        "strain": load_registry(strain_path, "strain"),
    }
    for registry in registries.values():
        validate_registry(registry).require_ok()
    fit_metadata = metadata_train_val.loc[
        metadata_train_val["split_final"].astype(str).eq("train")
    ].copy()
    support_manifest = build_support_manifest(fit_metadata, registries)
    support_path = args.output_dir / "support_manifest_fit_train.json"
    support_file_sha = write_json_with_hash(support_path, support_manifest)
    all_labeled_support = build_support_manifest(metadata_train_val, registries)
    all_labeled_support_path = args.output_dir / "support_manifest_fit_all_labeled.json"
    all_labeled_support_file_sha = write_json_with_hash(
        all_labeled_support_path, all_labeled_support
    )

    build_manifest = {
        "schema_version": "goai.entity-registry-build.v1",
        "entity_counts": {"chemical": len(chemical), "strain": len(strain)},
        "registry_semantic_sha256": {
            name: registry.sha256 for name, registry in registries.items()
        },
        "file_sha256": {
            "chemical_registry": sha256_file(chemical_path),
            "strain_registry": sha256_file(strain_path),
            "chemical_source": sha256_file(args.chemical_map),
            "chemical_second_source_review": sha256_file(
                args.chemical_second_source_review
            ),
            "strain_source": sha256_file(args.strain_source),
            "metadata_train_val": sha256_file(args.metadata_train_val),
            "metadata_test": sha256_file(args.metadata_test),
            "support_manifest_fit_train": support_file_sha,
            "support_manifest_fit_all_labeled": all_labeled_support_file_sha,
        },
        "support_manifest_content_sha256": {
            "fit_train": manifest_sha256(support_manifest),
            "fit_all_labeled": manifest_sha256(all_labeled_support),
        },
        "identity_policy": {
            "pert_id": "namespaced by (data_source, pert_id); never a global entity ID",
            "proxy": "kept distinct from proxy_target",
            "verified_chemical": (
                "requires a reviewed independent source with the same InChIKey; "
                "database conflicts remain candidates"
            ),
            "fit_support": (
                "two immutable snapshots: split_final == train and every released "
                "labeled row; checkpoints must still persist their actual fold-fit manifest"
            ),
            "expert_support": (
                "pair and exact strain/chemical/medium/temperature/time/time-unit "
                "vocabularies and counts contain treatment rows only"
            ),
        },
    }
    write_json_with_hash(args.output_dir / "registry_manifest.json", build_manifest)
    print(f"Built {len(chemical)} chemical and {len(strain)} strain registry rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
