"""Validation and checkpoint identities for external semantic artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .entities import sha256_file, stable_json_dumps


def _manifest_path(feature_path: Path) -> Path:
    return feature_path.parent / "manifest.json"


def _strain_manifest_path(feature_path: Path) -> Path:
    return feature_path.parent / "strain_semantics_manifest.json"


def _declared_path(manifest_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _entity_order_sha256(names: list[str]) -> str:
    return sha256(stable_json_dumps(names).encode("utf-8")).hexdigest()


def validate_chemical_feature_artifact(
    feature_path: Path | None,
    *,
    manifest_required: bool = False,
) -> dict[str, object] | None:
    """Validate a frozen chemical feature table against its build manifest.

    Formal M8 artifacts opt into ``manifest_required``.  Historical ad-hoc
    feature tables without a manifest remain loadable, but their checkpoint
    receipt is explicitly marked ``legacy_unverified`` rather than silently
    implying a mapping-to-embedding provenance chain.
    """

    if feature_path is None:
        if manifest_required:
            raise ValueError(
                "chemical_features_manifest_required=true requires chemical_features"
            )
        return None
    feature_path = feature_path.resolve()
    if not feature_path.is_file():
        raise FileNotFoundError(f"Chemical feature table does not exist: {feature_path}")
    manifest_path = _manifest_path(feature_path)
    if not manifest_path.is_file():
        if manifest_required:
            raise FileNotFoundError(
                f"Formal chemical feature artifact requires {manifest_path}"
            )
        warnings.warn(
            f"Chemical feature artifact {feature_path} has no manifest; "
            "checkpoint will be marked legacy_unverified",
            RuntimeWarning,
            stacklevel=2,
        )
        return {"status": "legacy_unverified"}

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Chemical feature manifest is not a mapping: {manifest_path}")
    required = {
        "model", "model_revision", "source", "source_sha256", "smiles_column",
        "rows", "resolved_rows", "embedding_dim", "real_path", "real_sha256",
        "shuffled_path", "shuffled_sha256",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"Chemical feature manifest missing fields: {missing}")
    if not str(manifest["model"]).strip() or not str(manifest["model_revision"]).strip():
        raise ValueError("Chemical feature manifest must pin model and revision")

    declared = {
        "real": (
            _declared_path(manifest_path, manifest["real_path"]),
            str(manifest["real_sha256"]),
        ),
        "shuffled": (
            _declared_path(manifest_path, manifest["shuffled_path"]),
            str(manifest["shuffled_sha256"]),
        ),
    }
    selected = next(
        (name for name, (path, _) in declared.items() if path.resolve() == feature_path),
        None,
    )
    if selected is None:
        raise ValueError(
            f"Chemical feature table {feature_path} is not declared by {manifest_path}"
        )
    expected_feature_sha = declared[selected][1]
    actual_feature_sha = sha256_file(feature_path)
    if actual_feature_sha != expected_feature_sha:
        raise ValueError(
            f"Chemical feature hash mismatch for {selected}: "
            f"{actual_feature_sha} != {expected_feature_sha}"
        )

    source_path = _declared_path(manifest_path, manifest["source"])
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Chemical feature source mapping is unavailable: {source_path}"
        )
    actual_source_sha = sha256_file(source_path)
    expected_source_sha = str(manifest["source_sha256"])
    if actual_source_sha != expected_source_sha:
        raise ValueError(
            f"Chemical mapping hash drift: {actual_source_sha} != {expected_source_sha}"
        )

    source = pd.read_csv(source_path, sep="\t", dtype=str, keep_default_na=False)
    features = pd.read_csv(feature_path, sep="\t", keep_default_na=False)
    if "raw_name" not in source.columns or "raw_name" not in features.columns:
        raise ValueError("Chemical mapping and feature table must both contain raw_name")
    if source["raw_name"].duplicated().any() or features["raw_name"].duplicated().any():
        raise ValueError("Chemical mapping/feature table contains duplicate raw_name values")
    source_names = source["raw_name"].astype(str).tolist()
    feature_names = features["raw_name"].astype(str).tolist()
    if feature_names != source_names:
        raise ValueError("Chemical feature entity order does not match its source mapping")
    rows = int(manifest["rows"])
    if rows != len(source) or rows != len(features):
        raise ValueError(
            f"Chemical feature row count mismatch: manifest={rows}, "
            f"source={len(source)}, features={len(features)}"
        )
    feature_columns = [column for column in features.columns if column != "raw_name"]
    embedding_dim = int(manifest["embedding_dim"])
    if embedding_dim != len(feature_columns) or embedding_dim <= 0:
        raise ValueError(
            f"Chemical feature dimension mismatch: manifest={embedding_dim}, "
            f"table={len(feature_columns)}"
        )
    values = features[feature_columns].apply(pd.to_numeric, errors="raise").to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Chemical feature table contains non-finite values")

    smiles_column = str(manifest["smiles_column"])
    if smiles_column not in source.columns or "status" not in source.columns:
        raise ValueError(
            "Chemical source mapping lacks manifest smiles_column or status"
        )
    resolved_rows = int(
        (
            source["status"].astype(str).eq("resolved")
            & source[smiles_column].astype(str).str.len().gt(0)
        ).sum()
    )
    if resolved_rows != int(manifest["resolved_rows"]):
        raise ValueError(
            f"Chemical resolved-row mismatch: computed={resolved_rows}, "
            f"manifest={manifest['resolved_rows']}"
        )

    return {
        "status": "verified_manifest",
        "manifest_sha256": sha256_file(manifest_path),
        "source_sha256": actual_source_sha,
        "selected_kind": selected,
        "selected_feature_sha256": actual_feature_sha,
        "rows": rows,
        "resolved_rows": resolved_rows,
        "embedding_dim": embedding_dim,
        "entity_order_sha256": _entity_order_sha256(source_names),
        "model": str(manifest["model"]),
        "model_revision": str(manifest["model_revision"]),
    }


def validate_chemical_structure_artifact(
    mapping_path: Path | None,
    *,
    manifest_required: bool = False,
) -> dict[str, object] | None:
    """Validate exact Morgan inputs and their structure-permutation control."""

    if mapping_path is None:
        if manifest_required:
            raise ValueError(
                "chemical_structure_manifest_required=true requires chemical_map"
            )
        return None
    mapping_path = mapping_path.resolve()
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Chemical structure map does not exist: {mapping_path}")
    manifest_path = mapping_path.parent / "manifest.json"
    if not manifest_path.is_file():
        if manifest_required:
            raise FileNotFoundError(
                f"Formal chemical structure artifact requires {manifest_path}"
            )
        return {"status": "legacy_unverified"}
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Chemical structure manifest is not a mapping: {manifest_path}")
    if str(manifest.get("schema_version")) != "goai.chemical-structure-views.v3":
        if manifest_required:
            raise ValueError("Formal Morgan artifact requires structure-view manifest v3")
        return {"status": "legacy_unverified"}
    outputs = manifest.get("outputs")
    inputs = manifest.get("inputs")
    permutation_contract = manifest.get("shuffled_exact")
    if not isinstance(outputs, dict) or not isinstance(inputs, dict) or not isinstance(
        permutation_contract, dict
    ):
        raise ValueError("Chemical structure manifest lacks output/input/permutation contracts")
    required_outputs = {"exact", "parent_normalized", "zero_risky", "exact_shuffled"}
    if not required_outputs.issubset(outputs):
        raise ValueError("Chemical structure manifest lacks paired real/shuffled outputs")
    declared: dict[str, tuple[Path, str]] = {}
    for kind in sorted(required_outputs):
        record = outputs[kind]
        if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
            raise ValueError(f"Chemical structure output contract is incomplete: {kind}")
        path = _declared_path(manifest_path, record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Declared chemical structure view is unavailable: {path}")
        actual = sha256_file(path)
        expected = str(record["sha256"])
        if actual != expected:
            raise ValueError(
                f"Chemical structure hash mismatch for {kind}: {actual} != {expected}"
            )
        declared[kind] = (path, actual)
    selected = next(
        (kind for kind, (path, _) in declared.items() if path.resolve() == mapping_path),
        None,
    )
    if selected is None:
        raise ValueError(
            f"Chemical structure map {mapping_path} is not declared by {manifest_path}"
        )

    input_receipts: dict[str, str] = {}
    for name, record in inputs.items():
        if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
            raise ValueError(f"Chemical structure input contract is incomplete: {name}")
        path = _declared_path(manifest_path, record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Chemical structure input is unavailable: {path}")
        actual = sha256_file(path)
        if actual != str(record["sha256"]):
            raise ValueError(f"Chemical structure input hash drift: {name}")
        input_receipts[f"{name}_sha256"] = actual
    exact_input = inputs.get("exact_map")
    if not isinstance(exact_input, dict):
        raise ValueError("Chemical structure manifest lacks exact-map source contract")
    source_path = _declared_path(manifest_path, exact_input["path"])
    source_sha = input_receipts["exact_map_sha256"]

    real = pd.read_csv(declared["exact"][0], sep="\t", dtype=str, keep_default_na=False)
    source = pd.read_csv(source_path, sep="\t", dtype=str, keep_default_na=False)
    shuffled = pd.read_csv(
        declared["exact_shuffled"][0], sep="\t", dtype=str, keep_default_na=False
    )
    required_columns = {
        "raw_name", "is_control", "status", "cid", "inchikey", "isomeric_smiles",
        "canonical_smiles",
    }
    if not required_columns.issubset(real.columns) or real.columns.tolist() != shuffled.columns.tolist():
        raise ValueError("Chemical real/shuffled map columns are inconsistent")
    if real["raw_name"].tolist() != shuffled["raw_name"].tolist():
        raise ValueError("Chemical shuffled map changed entity order")
    if not source.equals(real):
        raise ValueError("Chemical exact structure view disagrees with its source map")
    if not real[["raw_name", "is_control"]].equals(shuffled[["raw_name", "is_control"]]):
        raise ValueError("Chemical shuffled map changed row identity/control flags")
    if real["raw_name"].duplicated().any():
        raise ValueError("Chemical structure views contain duplicate raw names")
    permutation = permutation_contract.get("permutation")
    if not isinstance(permutation, list) or not permutation:
        raise ValueError("Chemical shuffled permutation is empty")
    lookup = real.set_index("raw_name", verify_integrity=True)
    shuffled_lookup = shuffled.set_index("raw_name", verify_integrity=True)
    recipients: list[str] = []
    donors: list[str] = []
    structure_columns = ["cid", "inchikey", "isomeric_smiles", "canonical_smiles", "status"]
    for record in permutation:
        if not isinstance(record, dict):
            raise ValueError("Chemical shuffled permutation record is invalid")
        recipient = str(record.get("recipient", ""))
        donor = str(record.get("donor", ""))
        if recipient not in lookup.index or donor not in lookup.index or recipient == donor:
            raise ValueError("Chemical shuffled permutation is not a valid derangement")
        if shuffled_lookup.loc[recipient, structure_columns].tolist() != lookup.loc[
            donor, structure_columns
        ].tolist():
            raise ValueError("Chemical shuffled structure disagrees with declared donor")
        recipients.append(recipient)
        donors.append(donor)
    if len(set(recipients)) != len(recipients) or set(recipients) != set(donors):
        raise ValueError("Chemical shuffled permutation is not bijective")

    return {
        "status": "verified_manifest",
        "manifest_sha256": sha256_file(manifest_path),
        "source_sha256": source_sha,
        "selected_kind": selected,
        "selected_mapping_sha256": sha256_file(mapping_path),
        "paired_real_sha256": declared["exact"][1],
        "paired_shuffled_sha256": declared["exact_shuffled"][1],
        "rows": len(real),
        "permuted_rows": len(recipients),
        "entity_order_sha256": _entity_order_sha256(real["raw_name"].tolist()),
        **input_receipts,
    }


def validate_strain_feature_artifact(
    feature_path: Path | None,
    *,
    manifest_required: bool = False,
) -> dict[str, object] | None:
    """Validate strain semantics all the way back to their public evidence.

    The real and shuffled tables are one paired experiment.  Validation checks
    both members even when only one is selected, so a negative control cannot
    silently drift away from the source/evidence chain used by the real view.
    """

    if feature_path is None:
        if manifest_required:
            raise ValueError(
                "strain_features_manifest_required=true requires strain_features"
            )
        return None
    feature_path = feature_path.resolve()
    if not feature_path.is_file():
        raise FileNotFoundError(f"Strain feature table does not exist: {feature_path}")
    manifest_path = _strain_manifest_path(feature_path)
    if not manifest_path.is_file():
        if manifest_required:
            raise FileNotFoundError(
                f"Formal strain semantic artifact requires {manifest_path}"
            )
        warnings.warn(
            f"Strain semantic artifact {feature_path} has no manifest; "
            "checkpoint will be marked legacy_unverified",
            RuntimeWarning,
            stacklevel=2,
        )
        return {"status": "legacy_unverified"}

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Strain semantic manifest is not a mapping: {manifest_path}")
    required = {
        "protocol",
        "target_strains",
        "candidate_strains",
        "unresolved_strains",
        "feature_columns",
        "source_table",
        "source_table_sha256",
        "distance_matrix",
        "distance_matrix_sha256",
        "identity_evidence_registry",
        "identity_evidence_registry_sha256",
        "identity_evidence_manifest",
        "identity_evidence_manifest_sha256",
        "real_path",
        "real_sha256",
        "shuffled_path",
        "shuffled_sha256",
        "resolved_row_permutation",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"Strain semantic manifest missing fields: {missing}")
    if str(manifest["protocol"]) != "goai_peter2018_strain_semantics_v1":
        raise ValueError("Unsupported strain semantic protocol")

    declared = {
        "real": (
            _declared_path(manifest_path, manifest["real_path"]),
            str(manifest["real_sha256"]),
        ),
        "shuffled": (
            _declared_path(manifest_path, manifest["shuffled_path"]),
            str(manifest["shuffled_sha256"]),
        ),
    }
    selected = next(
        (name for name, (path, _) in declared.items() if path.resolve() == feature_path),
        None,
    )
    if selected is None:
        raise ValueError(
            f"Strain feature table {feature_path} is not declared by {manifest_path}"
        )
    for kind, (path, expected) in declared.items():
        if not path.is_file():
            raise FileNotFoundError(f"Declared {kind} strain feature table is unavailable: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Strain feature hash mismatch for {kind}: {actual} != {expected}"
            )

    evidence_receipt: dict[str, str] = {}
    for name, path_field, hash_field in (
        ("source_table", "source_table", "source_table_sha256"),
        ("distance_matrix", "distance_matrix", "distance_matrix_sha256"),
        (
            "identity_evidence_registry",
            "identity_evidence_registry",
            "identity_evidence_registry_sha256",
        ),
        (
            "identity_evidence_manifest",
            "identity_evidence_manifest",
            "identity_evidence_manifest_sha256",
        ),
    ):
        path = _declared_path(manifest_path, manifest[path_field])
        if not path.is_file():
            raise FileNotFoundError(f"Strain semantic {name} is unavailable: {path}")
        actual = sha256_file(path)
        expected = str(manifest[hash_field])
        if actual != expected:
            raise ValueError(
                f"Strain semantic {name} hash drift: {actual} != {expected}"
            )
        evidence_receipt[f"{name}_sha256"] = actual

    real = pd.read_csv(declared["real"][0], sep="\t", keep_default_na=False)
    shuffled = pd.read_csv(declared["shuffled"][0], sep="\t", keep_default_na=False)
    target_strains = [str(value) for value in manifest["target_strains"]]
    feature_columns = [str(value) for value in manifest["feature_columns"]]
    expected_columns = ["strain_code", *feature_columns]
    for kind, table in (("real", real), ("shuffled", shuffled)):
        if table.columns.astype(str).tolist() != expected_columns:
            raise ValueError(f"{kind} strain feature columns disagree with manifest")
        if table["strain_code"].astype(str).tolist() != target_strains:
            raise ValueError(f"{kind} strain feature entity order disagrees with manifest")
        values = table[feature_columns].apply(pd.to_numeric, errors="raise").to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{kind} strain feature table contains non-finite values")

    status_columns = ["resolved", "missing", "proxy"]
    if not set(status_columns).issubset(feature_columns):
        raise ValueError("Strain feature manifest lacks resolved/missing/proxy build flags")
    if not real[status_columns].equals(shuffled[status_columns]):
        raise ValueError("Strain shuffled control changed identity-status flags")
    candidate_strains = [str(value) for value in manifest["candidate_strains"]]
    unresolved_strains = [str(value) for value in manifest["unresolved_strains"]]
    indexed_status = real.set_index("strain_code", verify_integrity=True)[status_columns]
    if set(candidate_strains).difference(indexed_status.index) or set(
        unresolved_strains
    ).difference(indexed_status.index):
        raise ValueError("Strain semantic status lists reference unknown entities")
    if not np.allclose(
        indexed_status.loc[candidate_strains].to_numpy(dtype=np.float64),
        np.asarray([[1.0, 0.0, 0.0]] * len(candidate_strains)),
    ):
        raise ValueError("Candidate strain build flags disagree with manifest")
    if not np.allclose(
        indexed_status.loc[unresolved_strains].to_numpy(dtype=np.float64),
        np.asarray([[0.0, 1.0, 0.0]] * len(unresolved_strains)),
    ):
        raise ValueError("Unresolved strain build flags disagree with manifest")
    semantic_columns = [column for column in feature_columns if column not in status_columns]
    permutation = np.asarray(manifest["resolved_row_permutation"], dtype=np.int64)
    resolved_rows = np.flatnonzero(
        pd.to_numeric(real["resolved"], errors="raise").to_numpy() > 0.5
    )
    if permutation.shape != resolved_rows.shape or set(permutation.tolist()) != set(
        resolved_rows.tolist()
    ):
        raise ValueError("Strain shuffled permutation is not a bijection over resolved rows")
    if len(permutation) > 1 and np.any(permutation == resolved_rows):
        raise ValueError("Strain shuffled permutation is not a derangement")
    expected_shuffled = real.loc[permutation, semantic_columns].to_numpy(dtype=np.float64)
    actual_shuffled = shuffled.loc[resolved_rows, semantic_columns].to_numpy(dtype=np.float64)
    if not np.allclose(actual_shuffled, expected_shuffled, rtol=0.0, atol=1e-9):
        raise ValueError("Strain shuffled features disagree with declared permutation")

    return {
        "status": "verified_manifest",
        "manifest_sha256": sha256_file(manifest_path),
        "selected_kind": selected,
        "selected_feature_sha256": sha256_file(feature_path),
        "paired_real_sha256": declared["real"][1],
        "paired_shuffled_sha256": declared["shuffled"][1],
        "rows": len(real),
        "embedding_dim": len(feature_columns),
        "entity_order_sha256": _entity_order_sha256(target_strains),
        "candidate_strains": candidate_strains,
        "unresolved_strains": unresolved_strains,
        **evidence_receipt,
    }
