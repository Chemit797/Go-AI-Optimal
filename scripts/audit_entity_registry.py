#!/usr/bin/env python3
"""Audit GOAI registries; strict mode is an execution gate for semantic models."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goai_response.entities import (  # noqa: E402
    build_support_manifest,
    load_registry,
    normalize_entity_key,
    sha256_file,
    support_flags,
    validate_registry,
    write_json_with_hash,
)


EXPECTED_STRAIN_IDENTITIES = {
    "BAH": ("peter2018:SX3", "SAMEA3895227", "ERR1309120"),
    "BAI": ("peter2018:BJ6", "SAMEA3895228", "ERR1309197"),
    "CEK": ("peter2018:JCM_2985-4B", "SAMEA3895619", "ERR1309167"),
    "CGD": ("peter2018:UCD_09-448", "SAMEA3895648", "ERR1309434"),
    "CRD": ("peter2018:FIMA_3", "SAMEA3895807", "ERR1308959"),
}


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
        "--chemical-registry", type=Path,
        default=PROJECT_ROOT / "data/processed/entities/chemical_registry.tsv",
    )
    parser.add_argument(
        "--strain-registry", type=Path,
        default=PROJECT_ROOT / "data/processed/entities/strain_registry.tsv",
    )
    parser.add_argument("--strict-semantic", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _issue(severity: str, code: str, message: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message}


def main() -> int:
    args = parse_args()
    train_val = pd.read_csv(args.metadata_train_val, low_memory=False)
    test = pd.read_csv(args.metadata_test, low_memory=False)
    all_metadata = pd.concat([train_val, test], ignore_index=True)
    registries = {
        "chemical": load_registry(args.chemical_registry, "chemical"),
        "strain": load_registry(args.strain_registry, "strain"),
    }
    issues: list[dict[str, str]] = []
    required = {
        "chemical": all_metadata["perturbation_no_concentration"].tolist(),
        "strain": all_metadata["Strains"].tolist(),
    }
    for entity_type, registry in registries.items():
        validation = validate_registry(registry, required[entity_type])
        issues.extend([issue.__dict__ for issue in validation.errors + validation.warnings])

    expected_counts = {"chemical": 57, "strain": 6}
    actual_counts = {
        "chemical": all_metadata["perturbation_no_concentration"].map(normalize_entity_key).nunique(),
        "strain": all_metadata["Strains"].map(normalize_entity_key).nunique(),
    }
    for entity_type, expected in expected_counts.items():
        if actual_counts[entity_type] != expected:
            issues.append(_issue(
                "error", "competition_entity_count",
                f"Expected {expected} {entity_type} entities, found {actual_counts[entity_type]}",
            ))

    # Role counts are an immutable competition-partition gate.
    role_counts = {
        "chemical": dict(Counter(
            all_metadata.drop_duplicates("perturbation_no_concentration")["chemical_role"].astype(str)
        )),
        "strain": dict(Counter(
            all_metadata.drop_duplicates("Strains")["strain_role"].astype(str)
        )),
    }
    expected_role_counts = {
        "chemical": {"train": 40, "val": 6, "test": 11},
        "strain": {"train": 4, "val": 1, "test": 1},
    }
    if role_counts != expected_role_counts:
        issues.append(_issue(
            "error", "role_partition_count",
            f"Expected {expected_role_counts}, found {role_counts}",
        ))

    fit = train_val.loc[train_val["split_final"].astype(str).eq("train")]
    manifest = build_support_manifest(fit, registries)
    flags = support_flags(test, manifest)
    semantic_counts = {
        column: int(flags[column].sum())
        for column in (
            "chemical_seen_in_fit", "strain_seen_in_fit",
            "chemical_semantic_supported", "strain_semantic_supported",
            "chemical_proxy", "strain_proxy", "chemical_missing", "strain_missing",
        )
    }

    # Test-only chemicals have no response labels in fit.  Structure is their
    # only transferable input, so any blank or unparsable structure is a hard
    # execution failure rather than an all-zero fallback.
    chemical_table = registries["chemical"].table
    test_only = chemical_table.loc[
        chemical_table["role"].eq("test")
        & ~chemical_table["is_control"].astype(bool)
        & ~chemical_table["is_quality_control"].astype(bool)
    ]
    test_structure_failures: list[dict[str, object]] = []
    try:
        from rdkit import Chem
    except ImportError:
        Chem = None
        issues.append(_issue(
            "error", "rdkit_unavailable",
            "RDKit is required to validate all test-only chemical structures",
        ))
    required_structure_fields = ("pubchem_cid", "inchikey", "isomeric_smiles")
    for _, row in test_only.iterrows():
        missing_fields = [
            column for column in required_structure_fields if not str(row[column]).strip()
        ]
        parseable = bool(
            Chem is not None
            and str(row["isomeric_smiles"]).strip()
            and Chem.MolFromSmiles(str(row["isomeric_smiles"]).strip()) is not None
        )
        if missing_fields or not parseable:
            failure = {
                "raw_name": str(row["raw_name"]),
                "missing_fields": missing_fields,
                "rdkit_parseable": parseable,
            }
            test_structure_failures.append(failure)
            issues.append(_issue(
                "error", "test_only_structure_gate",
                f"{row['raw_name']!r}: missing={missing_fields}, rdkit_parseable={parseable}",
            ))

    # Publication/competition provenance is a promotion gate, not a reason to
    # mislabel every successful PubChem lookup as verified.  Keep the exact
    # blocked list scoped to the 11 test-only treatments.
    promotion_blockers = [
        {
            "raw_name": str(row["raw_name"]),
            "mapping_status": str(row["mapping_status"]),
            "evidence_tier": str(row["evidence_tier"]),
            "reason": (
                "formulation/component proxy requires explicit opt-in"
                if row["mapping_status"] == "proxy"
                else str(row.get("identity_review_outcome", "")).strip()
                or "database structure found, but no independent second-source identity verification"
            ),
        }
        for _, row in test_only.iterrows()
        if row["mapping_status"] != "verified"
    ]
    if promotion_blockers:
        issues.append(_issue(
            "error" if args.strict_semantic else "warning",
            "test_only_promotion_blocked",
            f"{len(promotion_blockers)} test-only chemicals are not independently verified",
        ))

    verified_test_only = test_only.loc[test_only["mapping_status"].eq("verified")]
    verified_evidence_failures: list[dict[str, str]] = []
    verified_required = {
        "secondary_source", "secondary_id", "secondary_inchikey",
        "evidence_snapshot_path", "evidence_snapshot_sha256",
    }
    missing_verified_columns = sorted(verified_required.difference(chemical_table.columns))
    if missing_verified_columns:
        issues.append(_issue(
            "error", "verified_evidence_columns",
            f"Chemical registry lacks verified-evidence columns: {missing_verified_columns}",
        ))
    else:
        for _, row in chemical_table.loc[chemical_table["mapping_status"].eq("verified")].iterrows():
            failure_reasons: list[str] = []
            if row["evidence_tier"] != "A_verified":
                failure_reasons.append("evidence_tier_not_A_verified")
            if not str(row["secondary_source"]).strip() or not str(row["secondary_id"]).strip():
                failure_reasons.append("missing_secondary_identity")
            if str(row["inchikey"]).strip() != str(row["secondary_inchikey"]).strip():
                failure_reasons.append("primary_secondary_inchikey_mismatch")
            snapshot = Path(str(row["evidence_snapshot_path"]).strip())
            if not snapshot.is_file():
                failure_reasons.append("missing_evidence_snapshot")
            elif sha256_file(snapshot) != str(row["evidence_snapshot_sha256"]).strip():
                failure_reasons.append("evidence_snapshot_hash_mismatch")
            if failure_reasons:
                failure = {
                    "raw_name": str(row["raw_name"]),
                    "reasons": ",".join(failure_reasons),
                }
                verified_evidence_failures.append(failure)
                issues.append(_issue(
                    "error", "verified_evidence_gate",
                    f"{row['raw_name']!r}: {failure['reasons']}",
                ))

    for entity_type, registry in registries.items():
        table = registry.table
        treatment = table
        if entity_type == "chemical":
            treatment = table.loc[
                ~table["is_control"].astype(bool) & ~table["is_quality_control"].astype(bool)
            ]
        proxies = treatment.loc[treatment["mapping_status"].eq("proxy"), registry.key_column].tolist()
        unresolved = treatment.loc[
            treatment["mapping_status"].eq("unresolved"), registry.key_column
        ].tolist()
        candidates = treatment.loc[
            treatment["mapping_status"].eq("high_confidence_candidate"), registry.key_column
        ].tolist()
        if proxies:
            issues.append(_issue(
                "error" if args.strict_semantic else "warning",
                f"{entity_type}_proxy", f"Proxy-only mappings: {proxies}",
            ))
        if unresolved:
            issues.append(_issue(
                "error", f"{entity_type}_unresolved", f"Unresolved mappings: {unresolved}",
            ))
        if candidates:
            issues.append(_issue(
                "warning",
                f"{entity_type}_candidate",
                f"Not independently verified: {candidates}",
            ))

    # A public alias chain is useful evidence, but it does not establish that
    # the identically named GOAI entity came from that public isolate.
    strain_rows = registries["strain"].table.set_index("strain_code")
    for code, (canonical, sample, run) in EXPECTED_STRAIN_IDENTITIES.items():
        if (
            code not in strain_rows.index
            or strain_rows.loc[code, "canonical_id"] != canonical
            or strain_rows.loc[code, "mapping_status"] != "high_confidence_candidate"
            or strain_rows.loc[code, "ena_sample_accession"] != sample
            or strain_rows.loc[code, "ena_run_accession"] != run
            or strain_rows.loc[code, "competition_identity_evidence"] != "absent"
        ):
            issues.append(_issue(
                "error", "strain_identity_chain",
                f"{code} must retain its reviewed public candidate chain without organizer promotion",
            ))
    if (
        "DHY210" not in strain_rows.index
        or strain_rows.loc["DHY210", "mapping_status"] != "unresolved"
        or strain_rows.loc["DHY210", "proxy_target"] != ""
        or strain_rows.loc["DHY210", "canonical_id"] != "goai-strain:DHY210"
    ):
        issues.append(_issue(
            "error", "dhy210_identity_gate",
            "DHY210 must remain unresolved and distinct; S288C cannot enter the production identity/proxy fields",
        ))

    report = {
        "schema_version": "goai.entity-registry-audit.v1",
        "ok": not any(issue["severity"] == "error" for issue in issues),
        "strict_semantic": args.strict_semantic,
        "entity_counts": actual_counts,
        "role_counts": role_counts,
        "test_support_counts": semantic_counts,
        "test_only_structure_failures": test_structure_failures,
        "test_only_verified_count": int(len(verified_test_only)),
        "verified_evidence_failures": verified_evidence_failures,
        "promotion_blockers": promotion_blockers,
        # Bind the audit decision to the exact registry bytes.  Formal M8
        # materialization re-hashes these paths, so editing a registry after a
        # successful audit invalidates the evidence instead of inheriting its
        # old approval.
        "registry_artifacts": {
            "chemical": {
                "path": str(args.chemical_registry.resolve()),
                "sha256": sha256_file(args.chemical_registry),
            },
            "strain": {
                "path": str(args.strain_registry.resolve()),
                "sha256": sha256_file(args.strain_registry),
            },
        },
        "issues": issues,
    }
    if args.output:
        write_json_with_hash(args.output, report)
    print(f"entity audit ok={report['ok']} errors={sum(i['severity']=='error' for i in issues)} "
          f"warnings={sum(i['severity']=='warning' for i in issues)}")
    for issue in issues:
        print(f"[{issue['severity']}] {issue['code']}: {issue['message']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
