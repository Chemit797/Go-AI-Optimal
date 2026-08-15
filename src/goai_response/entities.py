"""Canonical GOAI entity registries and fold-local support gates.

The important distinction in this module is between identity and support.  A raw
competition label can be present in the registry while its biological identity
is still a candidate, a proxy, or unresolved.  Callers therefore receive an
``EntityResolution`` and must not silently treat a proxy as an exact mapping.

``pert_id`` is deliberately *not* an entity identifier.  It is only validated
inside ``data_source`` because the same number denotes different chemicals in
WAYB and WAYC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import unicodedata

import pandas as pd

from goai_baseline.schema import MEDIUM, TEMPERATURE, TIME, TIME_UNIT, treatment_mask


SCHEMA_VERSION = "goai.entity-registry.v1"
MANIFEST_SCHEMA_VERSION = "goai.entity-support-manifest.v1"

ENTITY_KEYS = {"chemical": "raw_name", "strain": "strain_code"}
MAPPING_STATUSES = frozenset(
    {"verified", "high_confidence_candidate", "proxy", "unresolved"}
)
EVIDENCE_TIERS = frozenset(
    {
        "A_verified",
        "B_primary_candidate",
        "C_database_lookup",
        "D_proxy_assumption",
        "E_unresolved",
    }
)
SEMANTIC_STATUSES = frozenset({"verified", "high_confidence_candidate"})

CHEMICAL_COLUMN = "perturbation_no_concentration"
STRAIN_COLUMN = "Strains"
DATA_SOURCE_COLUMN = "data_source"
PERT_ID_COLUMN = "pert_id"


class EntityRegistryError(ValueError):
    """Raised when an entity registry or support manifest is unsafe to use."""


def normalize_entity_key(value: object) -> str:
    """Return the sole normalization allowed for registry lookup.

    This intentionally performs no chemistry or strain alias substitution.
    Aliases belong in a reviewed registry row, where their evidence is visible.
    """

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.strip().split()).casefold()


def entity_support_key(
    value: object,
    record: Mapping[str, Any] | EntityResolution | None = None,
) -> str:
    """Return the stable key used by fold-fit experts and routing gates.

    A reviewed registry identity wins regardless of whether its mapping is
    verified, a candidate, a proxy, or unresolved.  In particular, a proxy
    keeps *its own* ``canonical_id`` and is never collapsed onto
    ``proxy_target``.  Raw identity is only a fallback when no registry record
    (or no non-empty canonical ID) exists.
    """

    canonical_id = ""
    if isinstance(record, EntityResolution):
        canonical_id = str(record.canonical_id).strip()
    elif record is not None:
        canonical_id = str(record.get("canonical_id", "")).strip()
    if canonical_id:
        return canonical_id
    return f"raw:{normalize_entity_key(value)}"


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    raise EntityRegistryError(f"Invalid boolean registry value: {value!r}")


def stable_json_dumps(payload: Any) -> str:
    """Serialize JSON deterministically for manifests and content hashes."""

    def default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, set):
            return sorted(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=default,
    )


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(payload: Mapping[str, Any]) -> str:
    return sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def write_json_with_hash(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Write canonical JSON and a ``.sha256`` sidecar, returning the digest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = stable_json_dumps(payload) + "\n"
    destination.write_text(text, encoding="utf-8")
    digest = sha256(text.encode("utf-8")).hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        digest + "\n", encoding="ascii"
    )
    return digest


def load_json_with_hash(
    path: str | Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    source = Path(path)
    raw = source.read_bytes()
    actual = sha256(raw).hexdigest()
    sidecar = source.with_suffix(source.suffix + ".sha256")
    expected = expected_sha256
    if expected is None and sidecar.exists():
        expected = sidecar.read_text(encoding="ascii").strip()
    if expected is not None and actual != expected:
        raise EntityRegistryError(
            f"JSON hash mismatch for {source}: expected {expected}, got {actual}"
        )
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise EntityRegistryError(f"Expected a JSON object in {source}")
    return payload


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str


@dataclass
class ValidationReport:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, code: str, message: str) -> None:
        self.errors.append(ValidationIssue("error", code, message))

    def add_warning(self, code: str, message: str) -> None:
        self.warnings.append(ValidationIssue("warning", code, message))

    def require_ok(self) -> None:
        if self.errors:
            joined = "; ".join(f"{issue.code}: {issue.message}" for issue in self.errors)
            raise EntityRegistryError(joined)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [asdict(issue) for issue in self.errors],
            "warnings": [asdict(issue) for issue in self.warnings],
        }


@dataclass(frozen=True)
class EntityRegistry:
    entity_type: str
    table: pd.DataFrame
    key_column: str
    source_path: str | None = None
    sha256: str = ""

    def records_by_normalized_key(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for row in self.table.to_dict(orient="records"):
            key = normalize_entity_key(row[self.key_column])
            records[key] = {name: _json_scalar(value) for name, value in row.items()}
        return records


@dataclass(frozen=True)
class EntityResolution:
    entity_type: str
    raw_value: str
    normalized_key: str
    canonical_id: str
    canonical_name: str
    mapping_status: str
    evidence_tier: str
    proxy_target: str
    is_proxy: bool
    is_missing: bool
    is_control: bool = False
    is_quality_control: bool = False

    @property
    def semantic_supported(self) -> bool:
        # Proxies are deliberately *not* admitted by the default semantic gate.
        # A caller that chooses to run a proxy-only experiment can inspect
        # ``is_proxy`` and ``proxy_target`` explicitly; it cannot accidentally
        # present that representation as an exact entity mapping.
        return not self.is_missing and self.mapping_status in SEMANTIC_STATUSES

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["semantic_supported"] = self.semantic_supported
        return data


def _json_scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if hasattr(value, "item"):
        return value.item()
    return value


def _registry_content_hash(entity_type: str, table: pd.DataFrame) -> str:
    records = [
        {column: _json_scalar(row[column]) for column in sorted(table.columns)}
        for row in table.to_dict(orient="records")
    ]
    records.sort(key=lambda row: normalize_entity_key(row[ENTITY_KEYS[entity_type]]))
    return manifest_sha256(
        {"schema_version": SCHEMA_VERSION, "entity_type": entity_type, "records": records}
    )


def registry_from_frame(
    table: pd.DataFrame,
    entity_type: str,
    *,
    source_path: str | Path | None = None,
) -> EntityRegistry:
    if entity_type not in ENTITY_KEYS:
        raise EntityRegistryError(f"Unknown entity_type {entity_type!r}")
    clean = table.copy()
    clean.columns = [str(column).strip() for column in clean.columns]
    for column in clean.columns:
        clean[column] = clean[column].map(_json_scalar)
    for column in ("is_control", "is_quality_control"):
        if column in clean.columns:
            clean[column] = clean[column].map(_bool_value)
    digest = _registry_content_hash(entity_type, clean)
    return EntityRegistry(
        entity_type=entity_type,
        table=clean,
        key_column=ENTITY_KEYS[entity_type],
        source_path=str(Path(source_path).resolve()) if source_path is not None else None,
        sha256=digest,
    )


def load_registry(
    path: str | Path, entity_type: str | None = None
) -> EntityRegistry:
    """Load a UTF-8 TSV registry and compute a semantic, row-order-independent hash."""

    source = Path(path)
    table = pd.read_csv(source, sep="\t", dtype=str, keep_default_na=False)
    if entity_type is None:
        if "raw_name" in table.columns and "strain_code" not in table.columns:
            entity_type = "chemical"
        elif "strain_code" in table.columns and "raw_name" not in table.columns:
            entity_type = "strain"
        elif "entity_type" in table.columns and table["entity_type"].nunique() == 1:
            entity_type = str(table["entity_type"].iloc[0]).strip()
        else:
            raise EntityRegistryError(f"Cannot infer registry type from {source}")
    return registry_from_frame(table, entity_type, source_path=source)


def validate_registry(
    registry: EntityRegistry,
    required_entities: Sequence[object] | None = None,
    *,
    require_verified: bool = False,
) -> ValidationReport:
    """Validate identity, evidence, proxy and collision gates."""

    report = ValidationReport()
    required_columns = {
        registry.key_column,
        "canonical_id",
        "canonical_name",
        "mapping_status",
        "evidence_tier",
        "proxy_target",
        "is_control",
        "is_quality_control",
    }
    missing_columns = sorted(required_columns.difference(registry.table.columns))
    if missing_columns:
        report.add_error("missing_columns", f"Missing columns: {missing_columns}")
        return report

    table = registry.table
    keys = table[registry.key_column].map(normalize_entity_key)
    if (keys == "").any():
        report.add_error("blank_key", "Registry contains a blank entity key")
    duplicates = sorted(set(keys[keys.duplicated(keep=False)].tolist()))
    if duplicates:
        report.add_error(
            "normalized_key_collision",
            f"NFKC/casefold normalization collides for: {duplicates[:10]}",
        )

    statuses = table["mapping_status"].astype(str).str.strip()
    bad_statuses = sorted(set(statuses).difference(MAPPING_STATUSES))
    if bad_statuses:
        report.add_error("invalid_mapping_status", f"Invalid statuses: {bad_statuses}")
    tiers = table["evidence_tier"].astype(str).str.strip()
    bad_tiers = sorted(set(tiers).difference(EVIDENCE_TIERS))
    if bad_tiers:
        report.add_error("invalid_evidence_tier", f"Invalid tiers: {bad_tiers}")

    for index, row in table.iterrows():
        label = str(row[registry.key_column])
        status = str(row["mapping_status"]).strip()
        canonical_id = str(row["canonical_id"]).strip()
        proxy_target = str(row["proxy_target"]).strip()
        if not canonical_id:
            report.add_error("blank_canonical_id", f"{label!r} has no stable canonical_id")
        if status == "proxy":
            if not proxy_target:
                report.add_error("proxy_without_target", f"{label!r} has no proxy_target")
            if canonical_id and canonical_id == proxy_target:
                report.add_error(
                    "proxy_collapsed_to_target",
                    f"{label!r} uses the same ID as its proxy target",
                )
        elif proxy_target:
            report.add_warning(
                "unexpected_proxy_target", f"{label!r} is {status!r} but has proxy_target"
            )
        if status == "unresolved" and registry.entity_type == "chemical":
            populated = [
                column
                for column in ("pubchem_cid", "inchikey", "isomeric_smiles", "canonical_smiles")
                if column in table.columns and str(row[column]).strip()
            ]
            if populated:
                report.add_error(
                    "unresolved_has_structure",
                    f"{label!r} is unresolved but has {populated}",
                )
        if require_verified and status != "verified":
            report.add_error(
                "not_verified", f"{label!r} has status {status!r}, not 'verified'"
            )
        elif status in {"high_confidence_candidate", "proxy", "unresolved"}:
            report.add_warning("mapping_not_verified", f"{label!r}: {status}")

        try:
            is_control = _bool_value(row["is_control"])
            is_qc = _bool_value(row["is_quality_control"])
        except EntityRegistryError as error:
            report.add_error("invalid_boolean", f"{label!r}: {error}")
            continue
        if is_control and is_qc:
            report.add_error("control_qc_conflict", f"{label!r} is both control and QC")

    if required_entities is not None:
        available = set(keys)
        missing = sorted(
            key
            for key in {normalize_entity_key(value) for value in required_entities}
            if key and key not in available
        )
        if missing:
            report.add_error("missing_required_entities", f"Missing keys: {missing[:20]}")
    return report


def _resolve(
    value: object,
    registry: EntityRegistry,
    expected_type: str,
    *,
    allow_proxy: bool,
) -> EntityResolution:
    if registry.entity_type != expected_type:
        raise EntityRegistryError(
            f"Expected {expected_type} registry, got {registry.entity_type}"
        )
    key = normalize_entity_key(value)
    row = registry.records_by_normalized_key().get(key)
    if row is None:
        return EntityResolution(
            entity_type=expected_type,
            raw_value="" if value is None else str(value),
            normalized_key=key,
            canonical_id="",
            canonical_name="",
            mapping_status="unresolved",
            evidence_tier="E_unresolved",
            proxy_target="",
            is_proxy=False,
            is_missing=True,
        )
    status = str(row["mapping_status"])
    is_proxy = status == "proxy"
    is_missing = status == "unresolved" or (is_proxy and not allow_proxy)
    return EntityResolution(
        entity_type=expected_type,
        raw_value="" if value is None else str(value),
        normalized_key=key,
        canonical_id=str(row["canonical_id"]),
        canonical_name=str(row["canonical_name"]),
        mapping_status=status,
        evidence_tier=str(row["evidence_tier"]),
        proxy_target=str(row["proxy_target"]),
        is_proxy=is_proxy,
        is_missing=is_missing,
        is_control=_bool_value(row.get("is_control", False)),
        is_quality_control=_bool_value(row.get("is_quality_control", False)),
    )


def canonical_chemical(
    value: object, registry: EntityRegistry, allow_proxy: bool = True
) -> EntityResolution:
    return _resolve(value, registry, "chemical", allow_proxy=allow_proxy)


def canonical_strain(
    value: object, registry: EntityRegistry, allow_proxy: bool = True
) -> EntityResolution:
    return _resolve(value, registry, "strain", allow_proxy=allow_proxy)


def _coerce_registries(
    registries: Mapping[str, EntityRegistry | str | Path],
) -> dict[str, EntityRegistry]:
    resolved: dict[str, EntityRegistry] = {}
    for entity_type in ("chemical", "strain"):
        if entity_type not in registries:
            raise EntityRegistryError(f"Missing {entity_type} registry")
        value = registries[entity_type]
        resolved[entity_type] = (
            value if isinstance(value, EntityRegistry) else load_registry(value, entity_type)
        )
        validate_registry(resolved[entity_type]).require_ok()
    return resolved


def _validate_metadata_columns(metadata: pd.DataFrame) -> None:
    required = {
        CHEMICAL_COLUMN,
        STRAIN_COLUMN,
        DATA_SOURCE_COLUMN,
        PERT_ID_COLUMN,
    }
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise EntityRegistryError(f"Metadata is missing required columns: {missing}")


def _perturbation_id_records(metadata: pd.DataFrame) -> list[dict[str, str]]:
    """Validate and serialize the safe ``(data_source, pert_id)`` namespace."""

    _validate_metadata_columns(metadata)
    working = metadata[
        [DATA_SOURCE_COLUMN, PERT_ID_COLUMN, CHEMICAL_COLUMN]
    ].copy()
    for column in working.columns:
        working[column] = working[column].map(lambda value: "" if pd.isna(value) else str(value).strip())
    if (working[[DATA_SOURCE_COLUMN, PERT_ID_COLUMN]] == "").any().any():
        raise EntityRegistryError("Metadata contains blank data_source or pert_id")
    working["chemical_key"] = working[CHEMICAL_COLUMN].map(normalize_entity_key)
    conflict_counts = working.groupby(
        [DATA_SOURCE_COLUMN, PERT_ID_COLUMN], dropna=False
    )["chemical_key"].nunique()
    conflicts = conflict_counts[conflict_counts > 1]
    if not conflicts.empty:
        examples = [f"{source}/{pert_id}" for source, pert_id in conflicts.index[:10]]
        raise EntityRegistryError(
            "A (data_source, pert_id) pair maps to multiple chemicals: " + ", ".join(examples)
        )
    unique = working.drop_duplicates(
        [DATA_SOURCE_COLUMN, PERT_ID_COLUMN, "chemical_key"]
    ).sort_values([DATA_SOURCE_COLUMN, PERT_ID_COLUMN, "chemical_key"])
    return [
        {
            "data_source": str(row[DATA_SOURCE_COLUMN]),
            "pert_id": str(row[PERT_ID_COLUMN]),
            "chemical_key": str(row["chemical_key"]),
        }
        for _, row in unique.iterrows()
    ]


def _metadata_content_hash(metadata: pd.DataFrame) -> str:
    columns = [DATA_SOURCE_COLUMN, PERT_ID_COLUMN, CHEMICAL_COLUMN, STRAIN_COLUMN]
    rows = [
        {column: "" if pd.isna(row[column]) else str(row[column]) for column in columns}
        for _, row in metadata[columns].iterrows()
    ]
    return manifest_sha256({"columns": columns, "rows": rows})


def _support_key_from_records(
    value: object,
    records: Mapping[str, Mapping[str, Any]],
) -> str:
    normalized = normalize_entity_key(value)
    return entity_support_key(value, records.get(normalized))


def _treatment_support_records(
    fit_metadata: pd.DataFrame,
    entities: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Serialize model-aligned pair and exact context/time support.

    Response experts never act on Water, DMSO, or QC rows, so these
    vocabularies deliberately use the same treatment predicate as the model.
    Lists are stable-sorted and counts are explicit to make fold support fully
    inspectable and part of the manifest content hash.
    """

    treatment = fit_metadata.loc[treatment_mask(fit_metadata)]
    raw_pair_counts: dict[tuple[str, str], int] = {}
    raw_context_counts: dict[tuple[str, ...], int] = {}
    support_pair_counts: dict[tuple[str, str], int] = {}
    support_context_counts: dict[tuple[str, ...], int] = {}
    context_fields = (STRAIN_COLUMN, CHEMICAL_COLUMN, MEDIUM, TEMPERATURE, TIME, TIME_UNIT)
    has_context = all(field in treatment.columns for field in context_fields)
    for _, row in treatment.iterrows():
        raw_pair = (
            normalize_entity_key(row[STRAIN_COLUMN]),
            normalize_entity_key(row[CHEMICAL_COLUMN]),
        )
        support_pair = (
            _support_key_from_records(row[STRAIN_COLUMN], entities["strain"]),
            _support_key_from_records(row[CHEMICAL_COLUMN], entities["chemical"]),
        )
        raw_pair_counts[raw_pair] = raw_pair_counts.get(raw_pair, 0) + 1
        support_pair_counts[support_pair] = support_pair_counts.get(support_pair, 0) + 1
        if has_context:
            raw_context = tuple(
                normalize_entity_key(row[field]) for field in context_fields
            )
            support_context = (
                support_pair[0],
                support_pair[1],
                *(normalize_entity_key(row[field]) for field in context_fields[2:]),
            )
            raw_context_counts[raw_context] = raw_context_counts.get(raw_context, 0) + 1
            support_context_counts[support_context] = (
                support_context_counts.get(support_context, 0) + 1
            )
    return {
        # These original raw-key fields remain part of the identity-drift audit
        # contract.  Model eligibility uses the support-key fields below.
        "seen_pair_keys": [list(key) for key in sorted(raw_pair_counts)],
        "pair_counts": [
            {"key": list(key), "count": raw_pair_counts[key]}
            for key in sorted(raw_pair_counts)
        ],
        "seen_context_time_keys": [list(key) for key in sorted(raw_context_counts)],
        "context_time_counts": [
            {"key": list(key), "count": raw_context_counts[key]}
            for key in sorted(raw_context_counts)
        ],
        "seen_pair_support_keys": [
            list(key) for key in sorted(support_pair_counts)
        ],
        "support_pair_counts": [
            {"key": list(key), "count": support_pair_counts[key]}
            for key in sorted(support_pair_counts)
        ],
        "seen_context_time_support_keys": [
            list(key) for key in sorted(support_context_counts)
        ],
        "support_context_time_counts": [
            {"key": list(key), "count": support_context_counts[key]}
            for key in sorted(support_context_counts)
        ],
    }


def build_support_manifest(
    fit_metadata: pd.DataFrame,
    registries: Mapping[str, EntityRegistry | str | Path],
) -> dict[str, Any]:
    """Snapshot fold-local entity support without using held-out metadata.

    The registry snapshot makes inference auditable even if the TSV later
    changes.  ``seen_support_keys`` is the gate for expert routing; raw keys are
    retained separately for identity-drift audit.  Semantic support is a
    separate property and never implies that an entity was seen in-fit.
    """

    _validate_metadata_columns(fit_metadata)
    resolved_registries = _coerce_registries(registries)
    pert_records = _perturbation_id_records(fit_metadata)

    entities: dict[str, dict[str, dict[str, Any]]] = {"chemical": {}, "strain": {}}
    for entity_type, registry in resolved_registries.items():
        resolver = canonical_chemical if entity_type == "chemical" else canonical_strain
        for row in registry.table.to_dict(orient="records"):
            raw = row[registry.key_column]
            resolution = resolver(raw, registry, allow_proxy=True)
            entities[entity_type][resolution.normalized_key] = resolution.to_dict()

    seen_raw_keys = {
        "chemical": sorted(
            {normalize_entity_key(value) for value in fit_metadata[CHEMICAL_COLUMN]}
        ),
        "strain": sorted({normalize_entity_key(value) for value in fit_metadata[STRAIN_COLUMN]}),
    }
    seen_canonical_ids: dict[str, list[str]] = {}
    seen_support_keys: dict[str, Any] = {}
    for entity_type in ("chemical", "strain"):
        ids = {
            entities[entity_type][key]["canonical_id"]
            for key in seen_raw_keys[entity_type]
            if key in entities[entity_type]
            and not entities[entity_type][key]["is_missing"]
            and entities[entity_type][key]["canonical_id"]
        }
        seen_canonical_ids[entity_type] = sorted(ids)

        seen_support_keys[entity_type] = sorted(
            {
                _support_key_from_records(key, entities[entity_type])
                for key in seen_raw_keys[entity_type]
            }
        )

    treatment_support = _treatment_support_records(fit_metadata, entities)
    seen_support_keys["pair"] = treatment_support["seen_pair_support_keys"]
    seen_support_keys["context_time"] = treatment_support[
        "seen_context_time_support_keys"
    ]

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "fit_row_count": int(len(fit_metadata)),
        "fit_metadata_sha256": _metadata_content_hash(fit_metadata),
        "registry_sha256": {
            entity_type: registry.sha256
            for entity_type, registry in resolved_registries.items()
        },
        "seen_raw_keys": seen_raw_keys,
        "seen_support_keys": seen_support_keys,
        "seen_canonical_ids": seen_canonical_ids,
        "entities": entities,
        "perturbation_ids": {
            "namespace": "(data_source, pert_id)",
            "records": pert_records,
            "sha256": manifest_sha256({"records": pert_records}),
        },
        **treatment_support,
    }


def _resolution_from_manifest(
    value: object, entity_type: str, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    key = normalize_entity_key(value)
    records = manifest.get("entities", {}).get(entity_type, {})
    record = records.get(key)
    if record is not None:
        return dict(record)
    return EntityResolution(
        entity_type=entity_type,
        raw_value="" if value is None else str(value),
        normalized_key=key,
        canonical_id="",
        canonical_name="",
        mapping_status="unresolved",
        evidence_tier="E_unresolved",
        proxy_target="",
        is_proxy=False,
        is_missing=True,
    ).to_dict()


def support_flags(metadata: pd.DataFrame, manifest: Mapping[str, Any]) -> pd.DataFrame:
    """Return per-row seen/semantic/proxy/missing gates and the OOD route."""

    _validate_metadata_columns(metadata)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise EntityRegistryError(
            f"Unsupported support manifest schema: {manifest.get('schema_version')!r}"
        )

    # A reused pair with a changed raw name is corruption, not OOD.
    current_pert = _perturbation_id_records(metadata)
    fit_pairs = {
        (row["data_source"], row["pert_id"]): row["chemical_key"]
        for row in manifest.get("perturbation_ids", {}).get("records", [])
    }
    for row in current_pert:
        pair = (row["data_source"], row["pert_id"])
        if pair in fit_pairs and fit_pairs[pair] != row["chemical_key"]:
            raise EntityRegistryError(
                f"pert_id identity drift for {pair[0]}/{pair[1]}: "
                f"fit={fit_pairs[pair]!r}, current={row['chemical_key']!r}"
            )

    support_vocabulary = manifest.get("seen_support_keys")
    use_canonical_support = isinstance(support_vocabulary, Mapping)
    if use_canonical_support:
        seen_chemical = set(support_vocabulary.get("chemical", []))
        seen_strain = set(support_vocabulary.get("strain", []))
    else:
        # Backward compatibility for checkpoints/manifests written before
        # canonical support keys were persisted.
        seen_chemical = set(manifest.get("seen_raw_keys", {}).get("chemical", []))
        seen_strain = set(manifest.get("seen_raw_keys", {}).get("strain", []))
    rows: list[dict[str, Any]] = []
    for _, metadata_row in metadata.iterrows():
        chemical = _resolution_from_manifest(
            metadata_row[CHEMICAL_COLUMN], "chemical", manifest
        )
        strain = _resolution_from_manifest(metadata_row[STRAIN_COLUMN], "strain", manifest)
        chemical_support_key = entity_support_key(
            metadata_row[CHEMICAL_COLUMN], chemical
        )
        strain_support_key = entity_support_key(metadata_row[STRAIN_COLUMN], strain)
        chemical_gate_key = (
            chemical_support_key if use_canonical_support else chemical["normalized_key"]
        )
        strain_gate_key = (
            strain_support_key if use_canonical_support else strain["normalized_key"]
        )
        chemical_seen = chemical_gate_key in seen_chemical
        strain_seen = strain_gate_key in seen_strain
        if chemical_seen and strain_seen:
            route = "seen_seen"
        elif not chemical_seen and strain_seen:
            route = "unseen_chemical"
        elif chemical_seen and not strain_seen:
            route = "unseen_strain"
        else:
            route = "unseen_both"
        rows.append(
            {
                "chemical_key": chemical["normalized_key"],
                "strain_key": strain["normalized_key"],
                "chemical_support_key": chemical_support_key,
                "strain_support_key": strain_support_key,
                "chemical_canonical_id": chemical["canonical_id"],
                "strain_canonical_id": strain["canonical_id"],
                "chemical_seen_in_fit": chemical_seen,
                "strain_seen_in_fit": strain_seen,
                "chemical_semantic_supported": bool(chemical["semantic_supported"]),
                "strain_semantic_supported": bool(strain["semantic_supported"]),
                "chemical_proxy": bool(chemical["is_proxy"]),
                "strain_proxy": bool(strain["is_proxy"]),
                "chemical_missing": bool(chemical["is_missing"]),
                "strain_missing": bool(strain["is_missing"]),
                "is_control": bool(chemical.get("is_control", False)),
                "is_quality_control": bool(
                    chemical.get("is_quality_control", False)
                ),
                "support_route": route,
            }
        )
    return pd.DataFrame(rows, index=metadata.index)


__all__ = [
    "EntityRegistry",
    "EntityRegistryError",
    "EntityResolution",
    "ValidationIssue",
    "ValidationReport",
    "build_support_manifest",
    "canonical_chemical",
    "canonical_strain",
    "entity_support_key",
    "load_json_with_hash",
    "load_registry",
    "manifest_sha256",
    "normalize_entity_key",
    "registry_from_frame",
    "sha256_file",
    "stable_json_dumps",
    "support_flags",
    "validate_registry",
    "write_json_with_hash",
]
