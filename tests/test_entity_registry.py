from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from goai_response.entities import (
    EntityRegistryError,
    build_support_manifest,
    canonical_chemical,
    canonical_strain,
    load_json_with_hash,
    load_registry,
    manifest_sha256,
    normalize_entity_key,
    registry_from_frame,
    stable_json_dumps,
    support_flags,
    validate_registry,
    write_json_with_hash,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _chemical_registry():
    return registry_from_frame(
        pd.DataFrame(
            [
                {
                    "raw_name": "Water",
                    "canonical_id": "pubchem:962",
                    "canonical_name": "Water",
                    "mapping_status": "high_confidence_candidate",
                    "evidence_tier": "C_database_lookup",
                    "proxy_target": "",
                    "is_control": True,
                    "is_quality_control": False,
                    "pubchem_cid": "962",
                    "inchikey": "XLYOFNOQVPJJNP-UHFFFAOYSA-N",
                    "isomeric_smiles": "O",
                    "canonical_smiles": "O",
                },
                {
                    "raw_name": "Drug A salt",
                    "canonical_id": "goai-chemical:drug-a-salt",
                    "canonical_name": "Drug A salt",
                    "mapping_status": "proxy",
                    "evidence_tier": "D_proxy_assumption",
                    "proxy_target": "pubchem:123",
                    "is_control": False,
                    "is_quality_control": False,
                    "pubchem_cid": "123",
                    "inchikey": "",
                    "isomeric_smiles": "CC",
                    "canonical_smiles": "CC",
                },
                {
                    "raw_name": "Mystery",
                    "canonical_id": "goai-chemical:mystery",
                    "canonical_name": "Mystery",
                    "mapping_status": "unresolved",
                    "evidence_tier": "E_unresolved",
                    "proxy_target": "",
                    "is_control": False,
                    "is_quality_control": False,
                    "pubchem_cid": "",
                    "inchikey": "",
                    "isomeric_smiles": "",
                    "canonical_smiles": "",
                },
            ]
        ),
        "chemical",
    )


def _strain_registry():
    return registry_from_frame(
        pd.DataFrame(
            [
                {
                    "strain_code": "BAH",
                    "canonical_id": "peter2018:SX3",
                    "canonical_name": "SX3",
                    "mapping_status": "high_confidence_candidate",
                    "evidence_tier": "B_primary_candidate",
                    "proxy_target": "",
                    "is_control": False,
                    "is_quality_control": False,
                },
                {
                    "strain_code": "ProxyStrain",
                    "canonical_id": "goai-strain:proxy-strain",
                    "canonical_name": "ProxyStrain",
                    "mapping_status": "proxy",
                    "evidence_tier": "D_proxy_assumption",
                    "proxy_target": "reference:S288C",
                    "is_control": False,
                    "is_quality_control": False,
                },
            ]
        ),
        "strain",
    )


def _metadata(rows):
    return pd.DataFrame(
        rows,
        columns=["data_source", "pert_id", "perturbation_no_concentration", "Strains"],
    )


def test_normalization_is_conservative_and_collisions_are_rejected():
    assert normalize_entity_key("  Drug\u3000A  ") == "drug a"
    table = _chemical_registry().table.copy()
    table = pd.concat([table, table.iloc[[0]].assign(raw_name=" WATER ")], ignore_index=True)
    registry = registry_from_frame(table, "chemical")
    report = validate_registry(registry)
    assert not report.ok
    assert "normalized_key_collision" in {issue.code for issue in report.errors}


def test_proxy_resolution_never_collapses_identity_into_target():
    chemical = canonical_chemical(" drug  a salt ", _chemical_registry())
    assert chemical.canonical_id == "goai-chemical:drug-a-salt"
    assert chemical.proxy_target == "pubchem:123"
    assert chemical.is_proxy and not chemical.semantic_supported
    blocked = canonical_chemical("Drug A salt", _chemical_registry(), allow_proxy=False)
    assert blocked.is_proxy and blocked.is_missing and not blocked.semantic_supported

    strain = canonical_strain("ProxyStrain", _strain_registry())
    assert strain.canonical_id != strain.proxy_target
    assert strain.proxy_target == "reference:S288C"


def test_pert_id_is_namespaced_by_source_and_raw_name_can_have_multiple_ids():
    fit = _metadata(
        [
            ("WAYB", "#2", "Drug A salt", "BAH"),
            ("WAYC", "#2", "Water", "BAH"),
            ("WAYB", "#77", "Drug A salt", "BAH"),
        ]
    )
    manifest = build_support_manifest(
        fit, {"chemical": _chemical_registry(), "strain": _strain_registry()}
    )
    records = manifest["perturbation_ids"]["records"]
    assert len(records) == 3
    assert {(row["data_source"], row["pert_id"]) for row in records} == {
        ("WAYB", "#2"), ("WAYC", "#2"), ("WAYB", "#77")
    }
    assert manifest["seen_pair_keys"] == [["bah", "drug a salt"]]
    assert manifest["pair_counts"] == [
        {"key": ["bah", "drug a salt"], "count": 2}
    ]
    # This minimal fixture lacks exact condition/time columns.  Production
    # metadata fills the corresponding treatment-only vocabulary and counts.
    assert manifest["seen_context_time_keys"] == []
    assert manifest["context_time_counts"] == []

    conflict = _metadata(
        [("WAYB", "#2", "Water", "BAH"), ("WAYB", "#2", "Drug A salt", "BAH")]
    )
    with pytest.raises(EntityRegistryError, match="maps to multiple chemicals"):
        build_support_manifest(
            conflict, {"chemical": _chemical_registry(), "strain": _strain_registry()}
        )


def test_fold_local_support_is_separate_from_semantic_support():
    fit = _metadata([("WAYB", "#1", "Water", "BAH")])
    manifest = build_support_manifest(
        fit, {"chemical": _chemical_registry(), "strain": _strain_registry()}
    )
    query = _metadata(
        [
            ("WAYB", "#1", "Water", "BAH"),
            ("WAYB", "#2", "Drug A salt", "BAH"),
            ("WAYB", "#3", "Water", "ProxyStrain"),
            ("WAYB", "#4", "Mystery", "Unknown"),
        ]
    )
    flags = support_flags(query, manifest)
    assert flags["support_route"].tolist() == [
        "seen_seen", "unseen_chemical", "unseen_strain", "unseen_both"
    ]
    assert not bool(flags.loc[1, "chemical_semantic_supported"])
    assert bool(flags.loc[1, "chemical_proxy"])
    assert not bool(flags.loc[1, "chemical_seen_in_fit"])
    assert bool(flags.loc[2, "strain_proxy"])
    assert bool(flags.loc[3, "chemical_missing"])
    assert bool(flags.loc[3, "strain_missing"])


def test_canonical_support_keys_merge_aliases_but_never_proxy_targets():
    chemical_registry = registry_from_frame(
        pd.DataFrame(
            [
                {
                    "raw_name": "Drug alias A",
                    "canonical_id": "pubchem:123",
                    "canonical_name": "Drug",
                    "mapping_status": "verified",
                    "evidence_tier": "A_verified",
                    "proxy_target": "",
                    "is_control": False,
                    "is_quality_control": False,
                },
                {
                    "raw_name": "Drug alias B",
                    "canonical_id": "pubchem:123",
                    "canonical_name": "Drug",
                    "mapping_status": "high_confidence_candidate",
                    "evidence_tier": "B_primary_candidate",
                    "proxy_target": "",
                    "is_control": False,
                    "is_quality_control": False,
                },
                {
                    "raw_name": "Drug formulation proxy",
                    "canonical_id": "goai-chemical:drug-formulation-proxy",
                    "canonical_name": "Drug formulation proxy",
                    "mapping_status": "proxy",
                    "evidence_tier": "D_proxy_assumption",
                    "proxy_target": "pubchem:123",
                    "is_control": False,
                    "is_quality_control": False,
                },
            ]
        ),
        "chemical",
    )
    strain_registry = registry_from_frame(
        pd.DataFrame(
            [
                {
                    "strain_code": alias,
                    "canonical_id": "goai-strain:DHY210",
                    "canonical_name": "DHY210",
                    "mapping_status": "unresolved",
                    "evidence_tier": "E_unresolved",
                    "proxy_target": "",
                    "is_control": False,
                    "is_quality_control": False,
                }
                for alias in ("DHY210 alias A", "DHY210 alias B")
            ]
        ),
        "strain",
    )
    registries = {"chemical": chemical_registry, "strain": strain_registry}

    fit_alias = _metadata(
        [("WAYB", "#fit", "Drug alias A", "DHY210 alias A")]
    )
    alias_manifest = build_support_manifest(fit_alias, registries)
    assert alias_manifest["seen_support_keys"]["chemical"] == ["pubchem:123"]
    assert alias_manifest["seen_support_keys"]["strain"] == [
        "goai-strain:DHY210"
    ]
    assert alias_manifest["seen_raw_keys"]["chemical"] == ["drug alias a"]
    alias_query = _metadata(
        [("WAYB", "#query", "Drug alias B", "DHY210 alias B")]
    )
    alias_flags = support_flags(alias_query, alias_manifest)
    assert alias_flags.loc[0, "chemical_seen_in_fit"]
    assert alias_flags.loc[0, "strain_seen_in_fit"]
    assert alias_flags.loc[0, "chemical_support_key"] == "pubchem:123"
    assert alias_flags.loc[0, "strain_support_key"] == "goai-strain:DHY210"

    fit_parent = _metadata(
        [("WAYB", "#parent", "Drug alias A", "DHY210 alias A")]
    )
    parent_manifest = build_support_manifest(fit_parent, registries)
    proxy_query = _metadata(
        [("WAYB", "#proxy", "Drug formulation proxy", "DHY210 alias B")]
    )
    proxy_flags = support_flags(proxy_query, parent_manifest)
    assert not proxy_flags.loc[0, "chemical_seen_in_fit"]
    assert proxy_flags.loc[0, "chemical_support_key"] == (
        "goai-chemical:drug-formulation-proxy"
    )
    assert proxy_flags.loc[0, "chemical_canonical_id"] != "pubchem:123"


def test_support_flags_reject_pert_id_identity_drift():
    fit = _metadata([("WAYB", "#1", "Water", "BAH")])
    manifest = build_support_manifest(
        fit, {"chemical": _chemical_registry(), "strain": _strain_registry()}
    )
    drift = _metadata([("WAYB", "#1", "Drug A salt", "BAH")])
    with pytest.raises(EntityRegistryError, match="identity drift"):
        support_flags(drift, manifest)


def test_registry_tsv_and_json_hash_roundtrip(tmp_path):
    table = _chemical_registry().table
    path = tmp_path / "chemical.tsv"
    table.to_csv(path, sep="\t", index=False)
    loaded = load_registry(path)
    assert loaded.sha256 == _chemical_registry().sha256

    payload = {"β": [2, 1], "a": {"z": True}}
    assert stable_json_dumps(payload) == stable_json_dumps(json.loads(stable_json_dumps(payload)))
    assert manifest_sha256(payload) == manifest_sha256(dict(reversed(list(payload.items()))))
    json_path = tmp_path / "manifest.json"
    digest = write_json_with_hash(json_path, payload)
    assert load_json_with_hash(json_path, digest) == payload
    json_path.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(EntityRegistryError, match="hash mismatch"):
        load_json_with_hash(json_path)


def test_production_dhy210_is_missing_not_s288c_proxy():
    registry = load_registry(
        PROJECT_ROOT / "resources/entities/strain_identity_candidates.tsv", "strain"
    )
    resolution = canonical_strain("DHY210", registry)
    assert resolution.canonical_id == "goai-strain:DHY210"
    assert resolution.mapping_status == "unresolved"
    assert resolution.proxy_target == ""
    assert resolution.is_missing
    assert not resolution.is_proxy
    assert not resolution.semantic_supported


def test_production_strain_candidates_retain_accession_chain_without_promotion():
    table = pd.read_csv(
        PROJECT_ROOT / "resources/entities/strain_identity_candidates.tsv",
        sep="\t", dtype=str, keep_default_na=False,
    ).set_index("strain_code")
    expected = {
        "BAH": ("peter2018:SX3", "BAH", "SAMEA3895227", "ERR1309120"),
        "BAI": ("peter2018:BJ6", "BAI", "SAMEA3895228", "ERR1309197"),
        "CEK": ("peter2018:JCM_2985-4B", "CEK", "SAMEA3895619", "ERR1309167"),
        "CGD": ("peter2018:UCD_09-448", "CGD", "SAMEA3895648", "ERR1309434"),
        "CRD": ("peter2018:FIMA_3", "CRD", "SAMEA3895807", "ERR1308959"),
    }
    for code, values in expected.items():
        canonical, standardised, sample, run = values
        assert table.loc[code, "canonical_id"] == canonical
        assert table.loc[code, "peter_standardized_name"] == standardised
        assert table.loc[code, "ena_sample_accession"] == sample
        assert table.loc[code, "ena_run_accession"] == run
        assert table.loc[code, "competition_identity_evidence"] == "absent"
        assert table.loc[code, "mapping_status"] == "high_confidence_candidate"

    assert table.loc["DHY210", "competition_identity_evidence"] == "absent"
    assert table.loc["DHY210", "mapping_status"] == "unresolved"
    assert table.loc["DHY210", "ncbi_assembly_accession"] == ""


def test_production_formulation_proxies_are_excluded_from_default_semantic_gate():
    registry = load_registry(
        PROJECT_ROOT / "data/processed/entities/chemical_registry.tsv", "chemical"
    )
    expected = {
        "Hoechst 33258",
        "Oligomycin",
        "Tunicamycin",
    }
    rows = registry.table.loc[registry.table["mapping_status"].eq("proxy")]
    assert set(rows["raw_name"]) == expected
    for raw_name in expected:
        resolution = canonical_chemical(raw_name, registry)
        assert resolution.is_proxy
        assert resolution.proxy_target
        assert resolution.canonical_id != resolution.proxy_target
        assert not resolution.semantic_supported

    # The exact formulation entry replaces the former parent-doxycycline
    # proxy.  It remains a database candidate until independent verification.
    doxycycline = canonical_chemical("Doxycycline hyclate", registry)
    assert doxycycline.canonical_id == "pubchem:54705095"
    assert not doxycycline.is_proxy
    assert doxycycline.semantic_supported

    # Exact PubChem salt replaces the former parent-only LY294002 proxy.  It
    # stays a database candidate because ChEBI/ChEMBL only confirm the parent.
    ly_hydrochloride = canonical_chemical("LY 294002 hydrochloride", registry)
    assert ly_hydrochloride.canonical_id == "pubchem:11957589"
    assert ly_hydrochloride.mapping_status == "high_confidence_candidate"
    assert not ly_hydrochloride.is_proxy
    assert ly_hydrochloride.semantic_supported


def test_test_only_second_source_promotions_preserve_conflict_blockers():
    registry = load_registry(
        PROJECT_ROOT / "data/processed/entities/chemical_registry.tsv", "chemical"
    )
    test_rows = registry.table.loc[registry.table["role"].eq("test")]
    assert set(test_rows.loc[test_rows["mapping_status"].eq("verified"), "raw_name"]) == {
        "(S)-(+)-Camptothecin",
        "Abietic acid",
        "Fluconazole",
        "H2O2",
        "MMS",
        "Neomycin B",
        "Plumbagin",
        "Tamoxifen",
    }
    assert set(
        test_rows.loc[test_rows["mapping_status"].eq("high_confidence_candidate"), "raw_name"]
    ) == {"Doxycycline hyclate", "G418", "Hygromycin B"}
    verified = test_rows.loc[test_rows["mapping_status"].eq("verified")]
    assert verified["evidence_tier"].eq("A_verified").all()
    assert verified["inchikey"].eq(verified["secondary_inchikey"]).all()
