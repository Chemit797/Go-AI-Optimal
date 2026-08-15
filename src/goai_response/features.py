"""Strictly train-fitted biological, observation, and entity feature blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from goai_response.entities import (
    EntityRegistryError,
    entity_support_key,
    load_registry,
    normalize_entity_key,
    validate_registry,
)

from goai_baseline.schema import (
    CHEMICAL,
    DATA_SOURCE,
    INSTRUMENT,
    MEDIUM,
    PLATE,
    STRAIN,
    TEMPERATURE,
    TIME,
    treatment_mask,
)


BIOLOGICAL_CATEGORIES = (STRAIN, CHEMICAL, MEDIUM, TEMPERATURE)
BACKGROUND_CATEGORIES = (STRAIN, MEDIUM, TEMPERATURE)
OBSERVATION_CATEGORIES = (DATA_SOURCE, INSTRUMENT, PLATE)
STATUS_COLUMNS = ("resolved", "missing", "proxy")
SEMANTIC_MAPPING_STATUSES = {"verified", "high_confidence_candidate"}
IDENTITY_STATUS_COLUMNS = ("verified", "candidate", "proxy", "missing")


def _categories(values: pd.Series) -> list[str]:
    return sorted(values.astype(str).drop_duplicates().tolist())


def _one_hot(values: pd.Series, categories: list[str]) -> np.ndarray:
    result = np.zeros((len(values), len(categories)), dtype=np.float32)
    lookup = {value: index for index, value in enumerate(categories)}
    positions = values.astype(str).map(lookup)
    valid = positions.notna().to_numpy()
    if valid.any():
        rows = np.flatnonzero(valid)
        result[rows, positions.iloc[rows].astype(int).to_numpy()] = 1.0
    return result


def _value_entity_key(
    value: object,
    records: dict[str, dict[str, object]],
    mode: str,
) -> str:
    if mode == "canonical_support_v1":
        normalized = normalize_entity_key(value)
        return entity_support_key(value, records.get(normalized))
    return normalize_entity_key(value)


def _entity_indices(
    values: pd.Series,
    keys: list[str],
    records: dict[str, dict[str, object]],
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return padding-safe indices (unknown=0) and an explicit fold-fit gate."""
    lookup = {key: index + 1 for index, key in enumerate(keys)}
    indices = np.asarray(
        [lookup.get(_value_entity_key(value, records, mode), 0) for value in values],
        dtype=np.int64,
    )
    return indices, (indices > 0).astype(np.float32).reshape(-1, 1)


def _pair_indices(
    strains: pd.Series,
    chemicals: pd.Series,
    keys: list[tuple[str, str]],
    strain_records: dict[str, dict[str, object]],
    chemical_records: dict[str, dict[str, object]],
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {tuple(key): index + 1 for index, key in enumerate(keys)}
    indices = np.asarray(
        [
            lookup.get(
                (
                    _value_entity_key(strain, strain_records, mode),
                    _value_entity_key(chemical, chemical_records, mode),
                ),
                0,
            )
            for strain, chemical in zip(strains, chemicals)
        ],
        dtype=np.int64,
    )
    return indices, (indices > 0).astype(np.float32).reshape(-1, 1)


@dataclass
class ResponseFeatures:
    response: np.ndarray
    background: np.ndarray
    cell: np.ndarray
    perturbation: np.ndarray
    observation: np.ndarray
    is_treatment: np.ndarray
    general_cell: np.ndarray
    general_perturbation: np.ndarray
    strain_indices: np.ndarray
    chemical_indices: np.ndarray
    strain_seen: np.ndarray
    chemical_seen: np.ndarray
    pair_indices: np.ndarray
    pair_seen: np.ndarray


@dataclass
class ResponseFeatureBuilder:
    """Separates biology from acquisition context by construction.

    The optional strain table must contain ``strain_code`` and purely numeric
    feature columns.  An empty table is treated as unavailable, never as a
    fabricated genotype representation.
    """

    chemical_map: Path | None = None
    strain_features_path: Path | None = None
    strain_feature_columns: tuple[str, ...] | None = None
    strain_feature_transform: str = "scaled"
    chemical_bits: int = 512
    chemical_features_path: Path | None = None
    chemical_registry_path: Path | None = None
    strain_registry_path: Path | None = None
    chemical_parent_views_path: Path | None = None
    chemical_identity_risks_path: Path | None = None
    allow_proxy_semantics: bool = False
    chemical_structure_view: str = "exact"
    semantic_identity_policy: str | None = None
    semantic_training_coverage_required: bool = False
    calibration_use_plate: bool = True
    calibration_plate_shuffle: bool = False
    calibration_shuffle_seed: int = 42
    biological_categories: dict[str, list[str]] = field(default_factory=dict)
    observation_categories: dict[str, list[str]] = field(default_factory=dict)
    max_train_time: float | None = None
    chemical_names: list[str] = field(default_factory=list)
    chemical_matrix: np.ndarray | None = None
    chemical_mean: np.ndarray | None = None
    chemical_scale: np.ndarray | None = None
    chemical_semantic_table: pd.DataFrame | None = None
    chemical_semantic_columns: list[str] = field(default_factory=list)
    chemical_semantic_mean: np.ndarray | None = None
    chemical_semantic_scale: np.ndarray | None = None
    chemical_registry_records: dict[str, dict[str, object]] = field(default_factory=dict)
    strain_registry_records: dict[str, dict[str, object]] = field(default_factory=dict)
    chemical_registry_sha256: str = ""
    strain_registry_sha256: str = ""
    chemical_parent_records: dict[str, dict[str, object]] = field(default_factory=dict)
    chemical_identity_risk_records: dict[str, dict[str, object]] = field(default_factory=dict)
    strain_table: pd.DataFrame | None = None
    strain_columns: list[str] = field(default_factory=list)
    strain_semantic_columns: list[str] = field(default_factory=list)
    strain_status_columns: list[str] = field(default_factory=list)
    strain_mean: np.ndarray | None = None
    strain_scale: np.ndarray | None = None
    strain_kernel_centers: np.ndarray | None = None
    strain_kernel_bandwidth: float | None = None
    plate_training_assignments: dict[str, str] = field(default_factory=dict)
    observation_slices: dict[str, tuple[int, int]] = field(default_factory=dict)
    response_prior_mode: str = "none"
    response_prior_alpha: float = 4.0
    response_prior_dim: int = 0
    response_prior_global_sum: np.ndarray | None = None
    response_prior_global_count: np.ndarray | None = None
    response_prior_chemical: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    response_prior_strain: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    strain_entity_keys: list[str] = field(default_factory=list)
    chemical_entity_keys: list[str] = field(default_factory=list)
    pair_entity_keys: list[tuple[str, str]] = field(default_factory=list)
    context_entity_keys: list[tuple[str, ...]] = field(default_factory=list)
    strain_raw_entity_keys: list[str] = field(default_factory=list)
    chemical_raw_entity_keys: list[str] = field(default_factory=list)
    pair_raw_entity_keys: list[tuple[str, str]] = field(default_factory=list)
    context_raw_entity_keys: list[tuple[str, ...]] = field(default_factory=list)
    entity_key_mode: str = "canonical_support_v1"
    semantic_training_coverage: dict[str, object] = field(default_factory=dict)

    def fit(self, metadata: pd.DataFrame, train_ids: pd.Index) -> "ResponseFeatureBuilder":
        train = metadata.loc[train_ids]
        treatment_train = train.loc[treatment_mask(train)]
        self._fit_registry_gates(train)
        self.biological_categories = {field: _categories(train[field]) for field in BIOLOGICAL_CATEGORIES}
        self.entity_key_mode = "canonical_support_v1"
        self.strain_raw_entity_keys = sorted(
            {normalize_entity_key(value) for value in train[STRAIN]}
        )
        self.chemical_raw_entity_keys = sorted(
            {normalize_entity_key(value) for value in train[CHEMICAL]}
        )
        self.strain_entity_keys = sorted(
            {
                _value_entity_key(
                    value, self.strain_registry_records, self.entity_key_mode
                )
                for value in train[STRAIN]
            }
        )
        self.chemical_entity_keys = sorted(
            {
                _value_entity_key(
                    value, self.chemical_registry_records, self.entity_key_mode
                )
                for value in train[CHEMICAL]
            }
        )
        self.pair_raw_entity_keys = sorted(
            {
                (normalize_entity_key(strain), normalize_entity_key(chemical))
                for strain, chemical in zip(
                    treatment_train[STRAIN], treatment_train[CHEMICAL]
                )
            }
        )
        self.pair_entity_keys = sorted(
            {
                (
                    _value_entity_key(
                        strain, self.strain_registry_records, self.entity_key_mode
                    ),
                    _value_entity_key(
                        chemical, self.chemical_registry_records, self.entity_key_mode
                    ),
                )
                for strain, chemical in zip(treatment_train[STRAIN], treatment_train[CHEMICAL])
            }
        )
        self.context_raw_entity_keys = self._context_keys(
            treatment_train, use_support_keys=False
        )
        self.context_entity_keys = self._context_keys(
            treatment_train, use_support_keys=True
        )
        observation_fields = (DATA_SOURCE, INSTRUMENT, PLATE) if self.calibration_use_plate else (DATA_SOURCE, INSTRUMENT)
        self.observation_categories = {field: _categories(train[field]) for field in observation_fields}
        offset = 0
        self.observation_slices = {}
        for field in observation_fields:
            width = len(self.observation_categories[field])
            self.observation_slices[field] = (offset, offset + width)
            offset += width
        if self.calibration_plate_shuffle and self.calibration_use_plate:
            rng = np.random.default_rng(self.calibration_shuffle_seed)
            shuffled = rng.permutation(train[PLATE].astype(str).to_numpy())
            self.plate_training_assignments = dict(zip(train.index.astype(str), shuffled.astype(str)))
        values = pd.to_numeric(train[TIME], errors="raise").to_numpy(dtype=np.float32)
        if np.any(values < 0):
            raise ValueError("pert_time must be non-negative")
        self.max_train_time = max(float(values.max()), 1.0)
        self._fit_chemical_features(train[CHEMICAL])
        self._fit_chemical_semantics(train[CHEMICAL])
        self._fit_strain_features(train[STRAIN])
        self._audit_semantic_training_coverage(treatment_train)
        return self

    def _audit_semantic_training_coverage(self, treatment_train: pd.DataFrame) -> None:
        """Record fold-fit semantic support and fail closed for formal M8.

        Identity flags and ID-expert support do not count as semantic coverage.
        At least one treatment entity must have an admitted, non-zero continuous
        representation under the selected identity policy.
        """

        receipt: dict[str, object] = {
            "identity_policy": self.semantic_identity_policy or "legacy_v1",
            "required": self.semantic_training_coverage_required,
        }
        chemical_configured = (
            self.chemical_structure_view != "zero"
            and (self.chemical_map is not None or self.chemical_features_path is not None)
        )
        if chemical_configured:
            values = treatment_train[CHEMICAL]
            blocks: list[np.ndarray] = []
            present = np.zeros(len(values), dtype=bool)
            matrix, matrix_present = self._chemical_matrix_block(values)
            if matrix.shape[1]:
                blocks.append(matrix)
                present |= matrix_present
            semantic, semantic_present = self._chemical_semantic_raw_block(values)
            if semantic.shape[1]:
                blocks.append(semantic)
                present |= semantic_present
            continuous = (
                np.concatenate(blocks, axis=1)
                if blocks
                else np.empty((len(values), 0), dtype=np.float32)
            )
            gate, _ = self._chemical_status_and_gate(values)
            nonzero = (
                np.any(np.abs(continuous) > 0, axis=1)
                if continuous.shape[1]
                else np.zeros(len(values), dtype=bool)
            )
            admitted = present & gate & nonzero
            receipt["chemical"] = {
                "configured": True,
                "admitted_treatment_rows": int(admitted.sum()),
                "admitted_unique_entities": int(
                    values.loc[admitted].map(normalize_entity_key).nunique()
                ),
                "treatment_rows": int(len(values)),
            }
        else:
            receipt["chemical"] = {"configured": False}

        strain_configured = self.strain_features_path is not None
        if strain_configured:
            values = treatment_train[STRAIN]
            raw, present = self._strain_raw_block(values)
            semantic_positions = [
                self.strain_columns.index(column)
                for column in self.strain_semantic_columns
            ]
            continuous = raw[:, semantic_positions]
            gate, _ = self._strain_status_and_gate(values)
            if self.strain_status_columns:
                status_positions = [
                    self.strain_columns.index(column) for column in STATUS_COLUMNS
                ]
                table_flags = raw[:, status_positions]
                table_gate = (
                    (table_flags[:, 0] > 0.5)
                    & (table_flags[:, 1] < 0.5)
                    & ((table_flags[:, 2] < 0.5) | self.allow_proxy_semantics)
                )
                gate &= table_gate
            nonzero = (
                np.any(np.abs(continuous) > 0, axis=1)
                if continuous.shape[1]
                else np.zeros(len(values), dtype=bool)
            )
            admitted = present & gate & nonzero
            receipt["strain"] = {
                "configured": True,
                "admitted_treatment_rows": int(admitted.sum()),
                "admitted_unique_entities": int(
                    values.loc[admitted].map(normalize_entity_key).nunique()
                ),
                "treatment_rows": int(len(values)),
            }
        else:
            receipt["strain"] = {"configured": False}
        self.semantic_training_coverage = receipt

        if self.semantic_training_coverage_required:
            empty = [
                name
                for name in ("chemical", "strain")
                if bool(dict(receipt[name]).get("configured"))
                and int(dict(receipt[name]).get("admitted_unique_entities", 0)) <= 0
            ]
            if empty:
                raise ValueError(
                    "Formal semantic training coverage is empty for fold-fit "
                    f"modalities {empty} under policy {self.semantic_identity_policy!r}"
                )

    def _context_keys(
        self,
        metadata: pd.DataFrame,
        *,
        use_support_keys: bool,
    ) -> list[tuple[str, ...]]:
        """Stable exact biological context/time support; this is audit state, not an embedding."""
        time_unit = "pert_time_unit"
        fields = (STRAIN, CHEMICAL, MEDIUM, TEMPERATURE, TIME, time_unit)
        if any(field not in metadata for field in fields):
            return []
        keys: set[tuple[str, ...]] = set()
        for _, row in metadata.loc[:, fields].iterrows():
            if use_support_keys:
                strain_key = _value_entity_key(
                    row[STRAIN], self.strain_registry_records, self.entity_key_mode
                )
                chemical_key = _value_entity_key(
                    row[CHEMICAL], self.chemical_registry_records, self.entity_key_mode
                )
            else:
                strain_key = normalize_entity_key(row[STRAIN])
                chemical_key = normalize_entity_key(row[CHEMICAL])
            keys.add(
                (
                    strain_key,
                    chemical_key,
                    normalize_entity_key(row[MEDIUM]),
                    normalize_entity_key(row[TEMPERATURE]),
                    normalize_entity_key(row[TIME]),
                    normalize_entity_key(row[time_unit]),
                )
            )
        return sorted(keys)

    def _fit_registry_gates(self, train: pd.DataFrame) -> None:
        """Load reviewed identity state without making registry rows model targets."""
        if (self.chemical_registry_path is None) != (self.strain_registry_path is None):
            raise ValueError("Chemical and strain registries must be supplied together")
        # Structure-view contracts are model inputs in their own right.  Load
        # them even in a legacy/no-registry run so their row-level gates cannot
        # silently disappear.
        self._load_parent_view_contract()
        self._load_identity_risk_contract()
        if self.chemical_registry_path is None:
            return
        chemical_registry = load_registry(self.chemical_registry_path, "chemical")
        strain_registry = load_registry(self.strain_registry_path, "strain")  # type: ignore[arg-type]
        chemical_report = validate_registry(chemical_registry, train[CHEMICAL].tolist())
        strain_report = validate_registry(strain_registry, train[STRAIN].tolist())
        try:
            chemical_report.require_ok()
            strain_report.require_ok()
        except EntityRegistryError as error:
            raise ValueError(f"Entity registry gate failed for fold-fit rows: {error}") from error
        self.chemical_registry_records = chemical_registry.records_by_normalized_key()
        self.strain_registry_records = strain_registry.records_by_normalized_key()
        self.chemical_registry_sha256 = chemical_registry.sha256
        self.strain_registry_sha256 = strain_registry.sha256
        unknown_risks = sorted(
            set(self.chemical_identity_risk_records) - set(self.chemical_registry_records)
        )
        if unknown_risks:
            raise ValueError(
                "chemical_identity_risks contains identities absent from the registry: "
                f"{unknown_risks}"
            )
        self._validate_chemical_map_registry_contract()

    def _load_parent_view_contract(self) -> None:
        if self.chemical_structure_view != "parent":
            return
        if self.chemical_structure_view == "parent" and not self.allow_proxy_semantics:
            raise ValueError("Parent structure view requires allow_proxy_semantics=true")
        if self.chemical_parent_views_path is None or not self.chemical_parent_views_path.is_file():
            raise ValueError("Parent structure view requires a reviewed chemical_parent_views table")
        table = pd.read_csv(
            self.chemical_parent_views_path, sep="\t", dtype=str, keep_default_na=False
        )
        required = {
            "raw_name",
            "parent_canonical_id",
            "parent_inchikey",
            "parent_isomeric_smiles",
            "view_status",
        }
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"chemical_parent_views is missing columns: {missing}")
        keys = table["raw_name"].map(normalize_entity_key)
        if keys.duplicated().any():
            raise ValueError("chemical_parent_views has normalized-name collisions")
        valid_status = {"parent_normalized_ablation", "component_proxy_ablation"}
        if not set(table["view_status"]).issubset(valid_status):
            raise ValueError("chemical_parent_views contains an unsupported view_status")
        if (table["parent_canonical_id"].str.strip() == "").any() or (
            table["parent_isomeric_smiles"].str.strip() == ""
        ).any():
            raise ValueError("chemical_parent_views contains an incomplete parent contract")
        self.chemical_parent_records = {
            key: {str(column): str(value) for column, value in row.items()}
            for key, row in zip(keys, table.to_dict(orient="records"))
        }

    def _load_identity_risk_contract(self) -> None:
        """Load the identity-risk zeroing contract independently of parent views."""
        if self.chemical_structure_view != "zero_risky":
            return
        path = self.chemical_identity_risks_path
        if path is None or not path.is_file():
            raise ValueError(
                "zero_risky structure view requires a reviewed chemical_identity_risks table"
            )
        table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        required = {"raw_name", "risk_class", "zero_risky", "evidence_path"}
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"chemical_identity_risks is missing columns: {missing}")
        keys = table["raw_name"].map(normalize_entity_key)
        if keys.duplicated().any():
            raise ValueError("chemical_identity_risks has normalized-name collisions")
        gate = table["zero_risky"].str.strip().str.casefold()
        if not gate.isin({"true", "false"}).all():
            raise ValueError("chemical_identity_risks zero_risky values must be true or false")
        table = table.loc[gate.eq("true")].copy()
        keys = keys.loc[gate.eq("true")]
        if (table["risk_class"].str.strip() == "").any() or (
            table["evidence_path"].str.strip() == ""
        ).any():
            raise ValueError("chemical_identity_risks contains an incomplete review contract")
        self.chemical_identity_risk_records = {
            key: {str(column): str(value) for column, value in row.items()}
            for key, row in zip(keys, table.to_dict(orient="records"))
        }

    def _validate_chemical_map_registry_contract(self) -> None:
        """Reject an exact-view map that still contains a parent or different formulation."""
        if self.chemical_map is None or not self.chemical_registry_records:
            return
        table = pd.read_csv(self.chemical_map, sep="\t", dtype=str, keep_default_na=False)
        if "raw_name" not in table:
            raise ValueError("chemical map requires raw_name")
        keys = table["raw_name"].map(normalize_entity_key)
        if keys.duplicated().any():
            collisions = sorted(set(keys[keys.duplicated(keep=False)]))
            raise ValueError(f"chemical map normalized-name collisions: {collisions[:10]}")
        for index, row in table.iterrows():
            key = keys.iloc[index]
            record = self.chemical_registry_records.get(key)
            if record is None:
                continue
            status = self._mapping_status(record)
            parent_record = self.chemical_parent_records.get(key)
            identity_risk_record = self.chemical_identity_risk_records.get(key)
            proxy_is_used = self.chemical_structure_view == "parent" and parent_record is not None
            exact_is_used = self._exact_semantics_allowed(status) and not (
                (self.chemical_structure_view == "parent" and parent_record is not None)
                or (
                    self.chemical_structure_view == "zero_risky"
                    and identity_risk_record is not None
                )
            )
            if self.chemical_structure_view in {"zero", "shuffled"} or not (
                exact_is_used or proxy_is_used
            ):
                # Identity-risk rows are intentionally absent from zero_risky;
                # all other exact rows still pass through this contract.
                continue
            is_control = str(record.get("is_control", "")).strip().casefold() in {
                "true",
                "1",
                "yes",
            } or record.get("is_control") is True
            map_status = str(row.get("status", "")).strip().casefold()
            smiles = str(row.get("isomeric_smiles", "")).strip()
            if not is_control and (map_status != "resolved" or not smiles):
                raise ValueError(
                    f"Registry admits {row['raw_name']!r}, but its chemical map has no resolved structure"
                )
            comparisons = (
                (("cid", "parent_canonical_id"), ("inchikey", "parent_inchikey"))
                if proxy_is_used
                else (("cid", "pubchem_cid"), ("inchikey", "inchikey"))
            )
            for map_column, registry_column in comparisons:
                map_value = str(row.get(map_column, "")).strip()
                reference = parent_record if proxy_is_used else record
                registry_value = str(reference.get(registry_column, "")).strip()
                if map_column == "cid" and registry_value.startswith("pubchem:"):
                    registry_value = registry_value.split(":", 1)[1]
                if map_value and registry_value and map_value.casefold() != registry_value.casefold():
                    raise ValueError(
                        f"Chemical map {map_column} disagrees with registry for {row['raw_name']!r}: "
                        f"map={map_value!r}, registry={registry_value!r}"
                    )

    @staticmethod
    def _mapping_status(record: dict[str, object] | None) -> str:
        return "unresolved" if record is None else str(record.get("mapping_status", "unresolved"))

    def _exact_semantics_allowed(self, status: str) -> bool:
        """Apply identity quality without relabelling blocked candidates."""
        if self.semantic_identity_policy == "verified_only":
            return status == "verified"
        # None reproduces historical/direct builders.  Configured research
        # runs use the explicit research_allow_candidate value.
        return status in SEMANTIC_MAPPING_STATUSES

    def _identity_flags(
        self,
        status: str,
        *,
        representation_is_proxy: bool = False,
    ) -> np.ndarray:
        """Return legacy v1 flags or the explicit verified/candidate/proxy/missing v2 flags."""
        if self.semantic_identity_policy is None:
            is_proxy = status == "proxy" or representation_is_proxy
            is_resolved = status in SEMANTIC_MAPPING_STATUSES and not representation_is_proxy
            is_missing = not is_resolved
            if representation_is_proxy:
                # A reviewed parent/component representation was available in
                # v1, so its overlapping proxy flag did not imply missing.
                is_missing = False
            return np.asarray(
                # Historical v1 intentionally used overlapping missing/proxy
                # flags: a proxy was [0, 1, 1], not a one-hot category.
                [float(is_resolved), float(is_missing), float(is_proxy)],
                dtype=np.float32,
            )
        flags = np.zeros(len(IDENTITY_STATUS_COLUMNS), dtype=np.float32)
        if representation_is_proxy or status == "proxy":
            flags[2] = 1.0
        elif status == "verified":
            flags[0] = 1.0
        elif status == "high_confidence_candidate":
            flags[1] = 1.0
        else:
            flags[3] = 1.0
        return flags

    def _chemical_status_and_gate(self, values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Return eligibility plus explicit identity-quality flags."""
        eligible = np.ones(len(values), dtype=bool)
        flag_width = len(STATUS_COLUMNS) if self.semantic_identity_policy is None else len(IDENTITY_STATUS_COLUMNS)
        flags = np.zeros((len(values), flag_width), dtype=np.float32)
        if not self.chemical_registry_records:
            flags[:, 0] = 1.0
            return eligible, flags
        for row, value in enumerate(values):
            record = self.chemical_registry_records.get(normalize_entity_key(value))
            status = self._mapping_status(record)
            key = normalize_entity_key(value)
            has_parent_view = key in self.chemical_parent_records
            has_identity_risk = key in self.chemical_identity_risk_records
            is_proxy = status == "proxy" or (
                self.chemical_structure_view == "parent" and has_parent_view
            )
            is_exact = self._exact_semantics_allowed(status)
            usable_proxy = (
                has_parent_view
                and self.chemical_structure_view == "parent"
                and self.allow_proxy_semantics
            )
            risky_zero = self.chemical_structure_view == "zero_risky" and has_identity_risk
            use_semantics = self.chemical_structure_view != "zero" and not risky_zero and (
                (is_exact and not (self.chemical_structure_view == "parent" and has_parent_view))
                or usable_proxy
            )
            eligible[row] = use_semantics
            flags[row] = self._identity_flags(
                status,
                representation_is_proxy=(
                    self.chemical_structure_view == "parent" and has_parent_view
                ),
            )
        return eligible, flags

    def _strain_status_and_gate(self, values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        eligible = np.ones(len(values), dtype=bool)
        flag_width = len(STATUS_COLUMNS) if self.semantic_identity_policy is None else len(IDENTITY_STATUS_COLUMNS)
        flags = np.zeros((len(values), flag_width), dtype=np.float32)
        if not self.strain_registry_records:
            flags[:, 0] = 1.0
            return eligible, flags
        for row, value in enumerate(values):
            record = self.strain_registry_records.get(normalize_entity_key(value))
            status = self._mapping_status(record)
            is_proxy = status == "proxy"
            is_resolved = self._exact_semantics_allowed(status)
            usable_proxy = is_proxy and self.allow_proxy_semantics
            eligible[row] = is_resolved or usable_proxy
            flags[row] = self._identity_flags(status)
        return eligible, flags

    def _fit_chemical_features(self, train_chemicals: pd.Series) -> None:
        if self.chemical_map is None or self.chemical_structure_view == "zero":
            return
        if not self.chemical_map.exists():
            raise FileNotFoundError(f"Chemical map does not exist: {self.chemical_map}")
        try:
            from goai_graph.chemistry import load_chemical_features
        except ImportError as error:  # pragma: no cover - package is local
            raise RuntimeError("Chemical feature module is unavailable") from error
        features = load_chemical_features(self.chemical_map, n_bits=self.chemical_bits)
        self.chemical_names = features.names
        self.chemical_matrix = features.matrix.astype(np.float32)
        matrix, present = self._chemical_matrix_block(train_chemicals)
        semantic_gate, _ = self._chemical_status_and_gate(train_chemicals)
        valid = present & semantic_gate
        fit_matrix = matrix[valid]
        if not len(fit_matrix):
            self.chemical_mean = np.zeros(matrix.shape[1], dtype=np.float32)
            self.chemical_scale = np.ones(matrix.shape[1], dtype=np.float32)
            return
        self.chemical_mean = fit_matrix.mean(axis=0)
        # A training-unseen Morgan bit can validly be 1 for a new compound.
        # Dividing it by an epsilon-sized training standard deviation would
        # turn a legitimate OOD feature into a million-scale activation.
        # Unit floor keeps binary bits bounded and still standardises varying
        # physicochemical descriptors whose scale is naturally larger.
        self.chemical_scale = np.maximum(fit_matrix.std(axis=0), 1.0)

    def _chemical_matrix_block(self, values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        if self.chemical_matrix is None:
            return np.empty((len(values), 0), dtype=np.float32), np.zeros(len(values), dtype=bool)
        result = np.zeros((len(values), self.chemical_matrix.shape[1]), dtype=np.float32)
        lookup = {normalize_entity_key(name): index for index, name in enumerate(self.chemical_names)}
        positions = values.map(lambda value: lookup.get(normalize_entity_key(value)))
        present = positions.notna().to_numpy()
        if present.any():
            rows = np.flatnonzero(present)
            result[rows] = self.chemical_matrix[positions.iloc[rows].astype(int).to_numpy()]
        if self.chemical_structure_view == "zero_risky":
            intentional_zero = values.map(normalize_entity_key).isin(
                self.chemical_identity_risk_records
            ).to_numpy()
            result[intentional_zero] = 0.0
            # A deliberately zeroed reviewed row remains present; eligibility
            # gates decide that its continuous vector must stay zero.
        return result, present

    def _fit_strain_features(self, train_strains: pd.Series) -> None:
        if self.strain_features_path is None:
            return
        if not self.strain_features_path.exists():
            raise FileNotFoundError(f"Strain feature table does not exist: {self.strain_features_path}")
        table = pd.read_csv(self.strain_features_path, sep="\t", keep_default_na=False)
        if table.empty:
            return
        if "strain_code" not in table.columns:
            raise ValueError("strain feature table requires a strain_code column")
        columns = [column for column in table.columns if column != "strain_code"]
        if not columns:
            raise ValueError("strain feature table has no numeric feature columns")
        if self.strain_feature_columns is not None:
            requested = list(self.strain_feature_columns)
            forbidden = sorted(set(requested).intersection(STATUS_COLUMNS))
            if forbidden:
                raise ValueError(
                    "strain_feature_columns selects status columns explicitly: "
                    f"{forbidden}; status flags are appended automatically"
                )
            missing = sorted(set(requested).difference(columns))
            if missing:
                raise ValueError(
                    f"strain_feature_columns are absent from the table: {missing}"
                )
            status = [column for column in STATUS_COLUMNS if column in columns]
            columns = requested + status
        present_status = [column for column in STATUS_COLUMNS if column in columns]
        if present_status and set(present_status) != set(STATUS_COLUMNS):
            raise ValueError(
                "strain feature table must contain all of resolved/missing/proxy when any status flag is present"
            )
        numeric = table.loc[:, columns].apply(pd.to_numeric, errors="raise").astype(np.float32)
        normalized_index = table["strain_code"].map(normalize_entity_key)
        self.strain_table = numeric.set_axis(normalized_index, axis=0)
        if self.strain_table.index.duplicated().any():
            raise ValueError("strain feature table has duplicate strain_code rows")
        self.strain_columns = columns
        self.strain_status_columns = present_status
        self.strain_semantic_columns = [column for column in columns if column not in STATUS_COLUMNS]
        if self.strain_status_columns and self.strain_registry_records:
            status_positions = [self.strain_columns.index(column) for column in STATUS_COLUMNS]
            table_flags_all = self.strain_table.iloc[:, status_positions].to_numpy(dtype=np.float32)
            registry_flags_all = np.zeros_like(table_flags_all)
            for row, value in enumerate(self.strain_table.index):
                status = self._mapping_status(
                    self.strain_registry_records.get(normalize_entity_key(value))
                )
                is_proxy = status == "proxy"
                is_resolved = status in SEMANTIC_MAPPING_STATUSES
                registry_flags_all[row] = np.asarray(
                    [
                        float(is_resolved),
                        float(not (is_resolved or is_proxy)),
                        float(is_proxy),
                    ],
                    dtype=np.float32,
                )
            disagreement = np.any(np.abs(table_flags_all - registry_flags_all) > 0.5, axis=1)
            if disagreement.any():
                names = self.strain_table.index[np.flatnonzero(disagreement)].astype(str).tolist()
                raise ValueError(f"Strain numeric flags disagree with registry for: {names}")
        # A genome coordinate is an entity property.  Fit its scaler over the
        # unique fold-supported strain records, not over sample frequencies
        # (which would let heavily assayed strains dominate the geometry).
        fit_strains = pd.Series(
            sorted({normalize_entity_key(value) for value in train_strains}),
            dtype=str,
        )
        raw, present = self._strain_raw_block(fit_strains)
        registry_gate, registry_flags = self._strain_status_and_gate(fit_strains)
        if self.strain_status_columns:
            status_positions = [self.strain_columns.index(column) for column in STATUS_COLUMNS]
            table_flags = raw[:, status_positions]
            table_gate = (
                table_flags[:, 0] > 0.5
            ) & (table_flags[:, 1] < 0.5) & (
                (table_flags[:, 2] < 0.5) | self.allow_proxy_semantics
            )
            valid = present & table_gate & registry_gate
        else:
            valid = present & registry_gate
        semantic_positions = [self.strain_columns.index(column) for column in self.strain_semantic_columns]
        semantic = raw[:, semantic_positions]
        fit_matrix = semantic[valid]
        if not len(fit_matrix):
            self.strain_mean = np.zeros(len(semantic_positions), dtype=np.float32)
            self.strain_scale = np.ones(len(semantic_positions), dtype=np.float32)
            if self.strain_feature_transform in {"rbf", "nearest"}:
                raise ValueError(
                    f"{self.strain_feature_transform} strain features require at least one "
                    "fold-supported strain"
                )
        else:
            self.strain_mean = fit_matrix.mean(axis=0)
            # Only a few strains are supported in each fold, so binary
            # clade/ecology columns are often constant.  An epsilon-scale
            # denominator would turn a held-out value of one into a
            # million-scale input.  Unit flooring leaves count/coverage
            # features standardized while keeping bounded semantics bounded.
            self.strain_scale = np.maximum(fit_matrix.std(axis=0), 1.0)
            if self.strain_feature_transform in {"rbf", "nearest"}:
                self.strain_kernel_centers = fit_matrix.astype(np.float32, copy=True)
            if self.strain_feature_transform == "rbf":
                distance = np.sqrt(
                    np.maximum(
                        ((fit_matrix[:, None, :] - fit_matrix[None, :, :]) ** 2).sum(axis=2),
                        0.0,
                    )
                )
                nonzero = distance[np.triu_indices(len(distance), k=1)]
                nonzero = nonzero[nonzero > 1e-8]
                self.strain_kernel_bandwidth = (
                    float(np.median(nonzero)) if len(nonzero) else 1.0
                )

    def _fit_chemical_semantics(self, train_chemicals: pd.Series) -> None:
        if self.chemical_features_path is None or self.chemical_structure_view == "zero":
            return
        if not self.chemical_features_path.exists():
            raise FileNotFoundError(f"Chemical feature table does not exist: {self.chemical_features_path}")
        table = pd.read_csv(self.chemical_features_path, sep="\t", keep_default_na=False)
        if table.empty or "raw_name" not in table.columns:
            raise ValueError("chemical feature table requires non-empty raw_name rows")
        columns = [column for column in table.columns if column != "raw_name"]
        numeric = table.loc[:, columns].apply(pd.to_numeric, errors="raise").astype(np.float32)
        self.chemical_semantic_table = numeric.set_axis(table["raw_name"].map(normalize_entity_key), axis=0)
        if self.chemical_semantic_table.index.duplicated().any():
            raise ValueError("chemical feature table has duplicate raw_name rows")
        self.chemical_semantic_columns = columns
        matrix, present = self._chemical_semantic_raw_block(train_chemicals)
        semantic_gate, _ = self._chemical_status_and_gate(train_chemicals)
        fit_matrix = matrix[present & semantic_gate]
        if not len(fit_matrix):
            self.chemical_semantic_mean = np.zeros(matrix.shape[1], dtype=np.float32)
            self.chemical_semantic_scale = np.ones(matrix.shape[1], dtype=np.float32)
        else:
            self.chemical_semantic_mean = fit_matrix.mean(axis=0)
            self.chemical_semantic_scale = np.maximum(fit_matrix.std(axis=0), 1e-6)

    def _chemical_semantic_raw_block(self, values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        if self.chemical_semantic_table is None:
            return np.empty((len(values), 0), dtype=np.float32), np.zeros(len(values), dtype=bool)
        keys = values.map(normalize_entity_key).to_numpy()
        present = pd.Index(keys).isin(self.chemical_semantic_table.index)
        result = self.chemical_semantic_table.reindex(keys).to_numpy(dtype=np.float32)
        result = np.nan_to_num(result, nan=0.0)
        if self.chemical_structure_view == "zero_risky":
            intentional_zero = pd.Index(keys).isin(self.chemical_identity_risk_records)
            result[intentional_zero] = 0.0
        return result, np.asarray(present, dtype=bool)

    def _chemical_block(self, values: pd.Series) -> np.ndarray:
        blocks: list[np.ndarray] = []
        semantic_gate, status_flags = self._chemical_status_and_gate(values)
        if self.chemical_matrix is not None and self.chemical_mean is not None and self.chemical_scale is not None:
            result, present = self._chemical_matrix_block(values)
            valid = present & semantic_gate
            scaled = np.zeros_like(result, dtype=np.float32)
            scaled[valid] = (
                result[valid] - self.chemical_mean[None, :]
            ) / self.chemical_scale[None, :]
            blocks.append(scaled)
        semantic, semantic_present = self._chemical_semantic_raw_block(values)
        if semantic.shape[1]:
            if self.chemical_semantic_mean is None or self.chemical_semantic_scale is None:
                raise RuntimeError("Chemical semantic scaling was not fit")
            valid = semantic_present & semantic_gate
            scaled = np.zeros_like(semantic, dtype=np.float32)
            scaled[valid] = (
                semantic[valid] - self.chemical_semantic_mean[None, :]
            ) / self.chemical_semantic_scale[None, :]
            blocks.append(scaled)
        if self.chemical_registry_records and (
            self.chemical_map is not None or self.chemical_features_path is not None
        ):
            blocks.append(status_flags)
        return np.concatenate(blocks, axis=1, dtype=np.float32) if blocks else np.empty((len(values), 0), dtype=np.float32)

    def _strain_raw_block(self, values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        if self.strain_table is None:
            return np.empty((len(values), 0), dtype=np.float32), np.zeros(len(values), dtype=bool)
        keys = values.map(normalize_entity_key).to_numpy()
        present = pd.Index(keys).isin(self.strain_table.index)
        result = self.strain_table.reindex(keys).to_numpy(dtype=np.float32)
        return np.nan_to_num(result, nan=0.0), np.asarray(present, dtype=bool)

    def _strain_block(self, values: pd.Series) -> np.ndarray:
        """Scale biological semantics, then hard-zero missing/proxy rows and append status."""
        if self.strain_table is None:
            return np.empty((len(values), 0), dtype=np.float32)
        if self.strain_mean is None or self.strain_scale is None:
            raise RuntimeError("Strain feature scaling was not fit")
        raw, present = self._strain_raw_block(values)
        registry_gate, registry_flags = self._strain_status_and_gate(values)
        semantic_positions = [self.strain_columns.index(column) for column in self.strain_semantic_columns]
        semantic = raw[:, semantic_positions]
        valid = present & registry_gate
        if self.strain_status_columns:
            status_positions = [self.strain_columns.index(column) for column in STATUS_COLUMNS]
            table_flags = raw[:, status_positions]
            table_gate = (
                table_flags[:, 0] > 0.5
            ) & (table_flags[:, 1] < 0.5) & (
                (table_flags[:, 2] < 0.5) | self.allow_proxy_semantics
            )
            valid &= table_gate
            flags = table_flags.copy()
            flags[~present] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            blocked_proxy = (flags[:, 2] > 0.5) & ~valid
            flags[blocked_proxy, 0] = 0.0
            flags[blocked_proxy, 1] = 1.0
            if self.strain_registry_records:
                flags = registry_flags
        else:
            flags = (
                registry_flags
                if self.strain_registry_records
                else np.empty((len(values), 0), dtype=np.float32)
            )
        if self.strain_feature_transform in {"rbf", "nearest"}:
            if self.strain_kernel_centers is None:
                raise RuntimeError(
                    f"{self.strain_feature_transform} strain centers were not fit"
                )
            transformed = np.zeros(
                (len(values), len(self.strain_kernel_centers)), dtype=np.float32
            )
            if valid.any():
                distance_square = (
                    (
                        semantic[valid, None, :]
                        - self.strain_kernel_centers[None, :, :]
                    )
                    ** 2
                ).sum(axis=2)
                if self.strain_feature_transform == "rbf":
                    if self.strain_kernel_bandwidth is None:
                        raise RuntimeError("RBF strain bandwidth was not fit")
                    transformed[valid] = np.exp(
                        -distance_square
                        / (2.0 * max(self.strain_kernel_bandwidth, 1e-6) ** 2)
                    ).astype(np.float32)
                else:
                    nearest = np.argmin(distance_square, axis=1)
                    transformed[np.flatnonzero(valid), nearest] = 1.0
        else:
            transformed = np.zeros_like(semantic, dtype=np.float32)
            transformed[valid] = (
                semantic[valid] - self.strain_mean[None, :]
            ) / self.strain_scale[None, :]
            np.clip(transformed, -5.0, 5.0, out=transformed)
        return np.concatenate([transformed, flags], axis=1, dtype=np.float32)

    def _time_block(self, metadata: pd.DataFrame) -> np.ndarray:
        if self.max_train_time is None:
            raise RuntimeError("ResponseFeatureBuilder must be fit before transform")
        time = pd.to_numeric(metadata[TIME], errors="raise").to_numpy(dtype=np.float32)
        scaled = time / self.max_train_time
        angle = 2.0 * np.pi * scaled
        return np.stack((scaled, np.log1p(time) / np.log1p(self.max_train_time), np.sin(angle), np.cos(angle)), axis=1)

    def transform(self, metadata: pd.DataFrame) -> ResponseFeatures:
        if not self.biological_categories or not self.observation_categories:
            raise RuntimeError("ResponseFeatureBuilder must be fit before transform")
        chemical_one_hot = _one_hot(metadata[CHEMICAL], self.biological_categories[CHEMICAL])
        chemical_structure = self._chemical_block(metadata[CHEMICAL])
        response_blocks = [_one_hot(metadata[field], self.biological_categories[field]) for field in BIOLOGICAL_CATEGORIES]
        response_blocks.extend([self._time_block(metadata), chemical_structure])
        strain = self._strain_block(metadata[STRAIN])
        if strain.shape[1]:
            response_blocks.append(strain)
        background_blocks = [_one_hot(metadata[field], self.biological_categories[field]) for field in BACKGROUND_CATEGORIES]
        background_blocks.append(self._time_block(metadata))
        if strain.shape[1]:
            background_blocks.append(strain)
        observation_blocks: list[np.ndarray] = []
        for field, categories in self.observation_categories.items():
            values = metadata[field]
            if field == PLATE and self.calibration_plate_shuffle:
                values = pd.Series(
                    [self.plate_training_assignments.get(str(index), str(value)) for index, value in zip(metadata.index, values)],
                    index=metadata.index,
                )
            observation_blocks.append(_one_hot(values, categories))
        observation = np.concatenate(observation_blocks, axis=1, dtype=np.float32)
        cell = np.concatenate(background_blocks, axis=1, dtype=np.float32)
        perturbation = np.concatenate([chemical_one_hot, chemical_structure], axis=1, dtype=np.float32)
        general_cell_blocks = [
            _one_hot(metadata[field], self.biological_categories[field])
            for field in (MEDIUM, TEMPERATURE)
        ]
        general_cell_blocks.append(self._time_block(metadata))
        if strain.shape[1]:
            general_cell_blocks.append(strain)
        strain_indices, strain_seen = _entity_indices(
            metadata[STRAIN],
            self.strain_entity_keys,
            self.strain_registry_records,
            self.entity_key_mode,
        )
        chemical_indices, chemical_seen = _entity_indices(
            metadata[CHEMICAL],
            self.chemical_entity_keys,
            self.chemical_registry_records,
            self.entity_key_mode,
        )
        pair_indices, pair_seen = _pair_indices(
            metadata[STRAIN],
            metadata[CHEMICAL],
            self.pair_entity_keys,
            self.strain_registry_records,
            self.chemical_registry_records,
            self.entity_key_mode,
        )
        return ResponseFeatures(
            response=np.concatenate(response_blocks, axis=1, dtype=np.float32),
            background=cell,
            cell=cell,
            perturbation=perturbation,
            observation=observation,
            is_treatment=treatment_mask(metadata).to_numpy(dtype=np.float32).reshape(-1, 1),
            general_cell=np.concatenate(general_cell_blocks, axis=1, dtype=np.float32),
            general_perturbation=chemical_structure,
            strain_indices=strain_indices,
            chemical_indices=chemical_indices,
            strain_seen=strain_seen,
            chemical_seen=chemical_seen,
            pair_indices=pair_indices,
            pair_seen=pair_seen,
        )

    @staticmethod
    def _group_tables(
        values: pd.Series,
        fc: np.ndarray,
        mask: np.ndarray,
        records: dict[str, dict[str, object]],
        entity_key_mode: str,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        tables: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        keys = values.map(
            lambda value: _value_entity_key(value, records, entity_key_mode)
        )
        for group in sorted(keys.unique()):
            rows = keys.eq(group).to_numpy()
            tables[group] = ((fc[rows] * mask[rows]).sum(axis=0), mask[rows].sum(axis=0))
        return tables

    def fit_response_priors(
        self,
        metadata: pd.DataFrame,
        fc: np.ndarray,
        mask: np.ndarray,
        mode: str,
        alpha: float,
    ) -> "ResponseFeatureBuilder":
        if mode not in {"none", "chemical", "strain", "both"}:
            raise ValueError(f"Unsupported response prior mode: {mode}")
        self.response_prior_mode = mode
        self.response_prior_alpha = float(alpha)
        if mode == "none":
            self.response_prior_dim = 0
            self.response_prior_global_sum = None
            self.response_prior_global_count = None
            self.response_prior_chemical = {}
            self.response_prior_strain = {}
            return self
        self.response_prior_dim = int(fc.shape[1])
        self.response_prior_global_sum = (fc * mask).sum(axis=0)
        self.response_prior_global_count = mask.sum(axis=0)
        if mode in {"chemical", "both"}:
            self.response_prior_chemical = self._group_tables(
                metadata[CHEMICAL],
                fc,
                mask,
                self.chemical_registry_records,
                self.entity_key_mode,
            )
        if mode in {"strain", "both"}:
            self.response_prior_strain = self._group_tables(
                metadata[STRAIN],
                fc,
                mask,
                self.strain_registry_records,
                self.entity_key_mode,
            )
        return self

    @staticmethod
    def _safe_mean(total: np.ndarray, count: np.ndarray) -> np.ndarray:
        return np.divide(total, count, out=np.zeros_like(total, dtype=np.float32), where=count > 0)

    def response_prior(
        self,
        metadata: pd.DataFrame,
        leave_one_out_fc: np.ndarray | None = None,
        leave_one_out_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.response_prior_dim <= 0:
            return np.zeros((len(metadata), 0), dtype=np.float32)
        if self.response_prior_global_sum is None or self.response_prior_global_count is None:
            raise RuntimeError("Response priors were not fitted")
        if (leave_one_out_fc is None) != (leave_one_out_mask is None):
            raise ValueError("leave-one-out fc and mask must be supplied together")
        result = np.zeros((len(metadata), self.response_prior_dim), dtype=np.float32)
        for row, (_, record) in enumerate(metadata.iterrows()):
            subtract_value = np.zeros(self.response_prior_dim, dtype=np.float32)
            subtract_mask = np.zeros(self.response_prior_dim, dtype=np.float32)
            if leave_one_out_fc is not None and leave_one_out_mask is not None:
                subtract_value = leave_one_out_fc[row] * leave_one_out_mask[row]
                subtract_mask = leave_one_out_mask[row]
            global_mean = self._safe_mean(
                self.response_prior_global_sum - subtract_value,
                self.response_prior_global_count - subtract_mask,
            )
            for field, tables in ((CHEMICAL, self.response_prior_chemical), (STRAIN, self.response_prior_strain)):
                if not tables:
                    continue
                registry_records = (
                    self.chemical_registry_records
                    if field == CHEMICAL
                    else self.strain_registry_records
                )
                table = tables.get(
                    _value_entity_key(
                        record[field], registry_records, self.entity_key_mode
                    )
                )
                if table is None:
                    continue
                total, count = table
                group_count = count - subtract_mask
                group_mean = self._safe_mean(total - subtract_value, group_count)
                shrink = group_count / (group_count + self.response_prior_alpha)
                result[row] += shrink * (group_mean - global_mean)
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            "chemical_map": None if self.chemical_map is None else str(self.chemical_map),
            "strain_features_path": None if self.strain_features_path is None else str(self.strain_features_path),
            "strain_feature_columns": None if self.strain_feature_columns is None else list(self.strain_feature_columns),
            "strain_feature_transform": self.strain_feature_transform,
            "chemical_bits": self.chemical_bits,
            "chemical_features_path": None if self.chemical_features_path is None else str(self.chemical_features_path),
            "chemical_registry_path": None if self.chemical_registry_path is None else str(self.chemical_registry_path),
            "strain_registry_path": None if self.strain_registry_path is None else str(self.strain_registry_path),
            "chemical_parent_views_path": None if self.chemical_parent_views_path is None else str(self.chemical_parent_views_path),
            "chemical_identity_risks_path": None if self.chemical_identity_risks_path is None else str(self.chemical_identity_risks_path),
            "allow_proxy_semantics": self.allow_proxy_semantics,
            "chemical_structure_view": self.chemical_structure_view,
            "semantic_identity_policy": self.semantic_identity_policy,
            "semantic_training_coverage_required": self.semantic_training_coverage_required,
            "semantic_training_coverage": self.semantic_training_coverage,
            "calibration_use_plate": self.calibration_use_plate,
            "calibration_plate_shuffle": self.calibration_plate_shuffle,
            "calibration_shuffle_seed": self.calibration_shuffle_seed,
            "biological_categories": self.biological_categories,
            "observation_categories": self.observation_categories,
            "max_train_time": self.max_train_time,
            "chemical_names": self.chemical_names,
            "chemical_matrix": self.chemical_matrix,
            "chemical_mean": self.chemical_mean,
            "chemical_scale": self.chemical_scale,
            "chemical_semantic_table": None if self.chemical_semantic_table is None else self.chemical_semantic_table.to_dict(orient="split"),
            "chemical_semantic_columns": self.chemical_semantic_columns,
            "chemical_semantic_mean": self.chemical_semantic_mean,
            "chemical_semantic_scale": self.chemical_semantic_scale,
            "chemical_registry_records": self.chemical_registry_records,
            "strain_registry_records": self.strain_registry_records,
            "chemical_registry_sha256": self.chemical_registry_sha256,
            "strain_registry_sha256": self.strain_registry_sha256,
            "chemical_parent_records": self.chemical_parent_records,
            "chemical_identity_risk_records": self.chemical_identity_risk_records,
            "strain_table": None if self.strain_table is None else self.strain_table.to_dict(orient="split"),
            "strain_columns": self.strain_columns,
            "strain_semantic_columns": self.strain_semantic_columns,
            "strain_status_columns": self.strain_status_columns,
            "strain_mean": self.strain_mean,
            "strain_scale": self.strain_scale,
            "strain_kernel_centers": self.strain_kernel_centers,
            "strain_kernel_bandwidth": self.strain_kernel_bandwidth,
            "plate_training_assignments": self.plate_training_assignments,
            "observation_slices": self.observation_slices,
            "response_prior_mode": self.response_prior_mode,
            "response_prior_alpha": self.response_prior_alpha,
            "response_prior_dim": self.response_prior_dim,
            "response_prior_global_sum": self.response_prior_global_sum,
            "response_prior_global_count": self.response_prior_global_count,
            "response_prior_chemical": self.response_prior_chemical,
            "response_prior_strain": self.response_prior_strain,
            "strain_entity_keys": self.strain_entity_keys,
            "chemical_entity_keys": self.chemical_entity_keys,
            "pair_entity_keys": self.pair_entity_keys,
            "context_entity_keys": self.context_entity_keys,
            "strain_raw_entity_keys": self.strain_raw_entity_keys,
            "chemical_raw_entity_keys": self.chemical_raw_entity_keys,
            "pair_raw_entity_keys": self.pair_raw_entity_keys,
            "context_raw_entity_keys": self.context_raw_entity_keys,
            "entity_key_mode": self.entity_key_mode,
        }

    def validate_support_manifest(self, manifest: dict[str, object]) -> None:
        """Prove that inference gates use the same fold-fit support vocabulary."""
        seen_raw = manifest.get("seen_raw_keys")
        if not isinstance(seen_raw, dict):
            raise ValueError("Support manifest is missing seen_raw_keys")
        seen_support = manifest.get("seen_support_keys")
        use_support = isinstance(seen_support, dict)
        seen = seen_support if use_support else seen_raw
        expected_strains = sorted(str(value) for value in seen.get("strain", []))
        expected_chemicals = sorted(str(value) for value in seen.get("chemical", []))
        if expected_strains != sorted(self.strain_entity_keys):
            raise ValueError("Builder strain expert vocabulary disagrees with support manifest")
        if expected_chemicals != sorted(self.chemical_entity_keys):
            raise ValueError("Builder chemical expert vocabulary disagrees with support manifest")
        expected_raw_strains = sorted(
            str(value) for value in seen_raw.get("strain", [])
        )
        expected_raw_chemicals = sorted(
            str(value) for value in seen_raw.get("chemical", [])
        )
        if self.strain_raw_entity_keys and expected_raw_strains != sorted(
            self.strain_raw_entity_keys
        ):
            raise ValueError("Builder strain raw-key audit disagrees with support manifest")
        if self.chemical_raw_entity_keys and expected_raw_chemicals != sorted(
            self.chemical_raw_entity_keys
        ):
            raise ValueError("Builder chemical raw-key audit disagrees with support manifest")
        pair_source = (
            seen.get("pair", [])
            if use_support
            else manifest.get("seen_pair_keys", [])
        )
        expected_pairs = sorted(
            tuple(str(part) for part in value)
            for value in pair_source
        )
        if expected_pairs != sorted(self.pair_entity_keys):
            raise ValueError("Builder pair expert vocabulary disagrees with support manifest")
        context_source = (
            seen.get("context_time", [])
            if use_support
            else manifest.get("seen_context_time_keys", [])
        )
        expected_contexts = sorted(
            tuple(str(part) for part in value)
            for value in context_source
        )
        if expected_contexts != sorted(self.context_entity_keys):
            raise ValueError("Builder context/time vocabulary disagrees with support manifest")
        expected_raw_pairs = sorted(
            tuple(str(part) for part in value)
            for value in manifest.get("seen_pair_keys", [])
        )
        if self.pair_raw_entity_keys and expected_raw_pairs != sorted(
            self.pair_raw_entity_keys
        ):
            raise ValueError("Builder pair raw-key audit disagrees with support manifest")
        expected_raw_contexts = sorted(
            tuple(str(part) for part in value)
            for value in manifest.get("seen_context_time_keys", [])
        )
        if self.context_raw_entity_keys and expected_raw_contexts != sorted(
            self.context_raw_entity_keys
        ):
            raise ValueError("Builder context/time raw-key audit disagrees with support manifest")
        registry_hashes = manifest.get("registry_sha256", {})
        if not isinstance(registry_hashes, dict):
            raise ValueError("Support manifest registry_sha256 must be a mapping")
        expected_hashes = {
            "chemical": self.chemical_registry_sha256,
            "strain": self.strain_registry_sha256,
        }
        for entity_type, expected in expected_hashes.items():
            actual = str(registry_hashes.get(entity_type, ""))
            if expected and actual != expected:
                raise ValueError(
                    f"Builder {entity_type} registry hash disagrees with support manifest"
                )

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ResponseFeatureBuilder":
        builder = cls(
            chemical_map=None if state["chemical_map"] is None else Path(str(state["chemical_map"])),
            strain_features_path=None if state["strain_features_path"] is None else Path(str(state["strain_features_path"])),
            strain_feature_columns=(
                None
                if state.get("strain_feature_columns") is None
                else tuple(str(value) for value in state["strain_feature_columns"])
            ),
            strain_feature_transform=str(
                state.get("strain_feature_transform", "scaled")
            ),
            chemical_bits=int(state["chemical_bits"]),
            chemical_features_path=None if state.get("chemical_features_path") is None else Path(str(state["chemical_features_path"])),
            chemical_registry_path=None if state.get("chemical_registry_path") is None else Path(str(state["chemical_registry_path"])),
            strain_registry_path=None if state.get("strain_registry_path") is None else Path(str(state["strain_registry_path"])),
            chemical_parent_views_path=None if state.get("chemical_parent_views_path") is None else Path(str(state["chemical_parent_views_path"])),
            chemical_identity_risks_path=None if state.get("chemical_identity_risks_path") is None else Path(str(state["chemical_identity_risks_path"])),
            allow_proxy_semantics=bool(state.get("allow_proxy_semantics", False)),
            chemical_structure_view=str(state.get("chemical_structure_view", "exact")),
            # Missing means a historical v1 checkpoint whose three status
            # columns and candidate-admission behaviour must remain unchanged.
            semantic_identity_policy=(
                None
                if state.get("semantic_identity_policy") is None
                else str(state.get("semantic_identity_policy"))
            ),
            semantic_training_coverage_required=bool(
                state.get("semantic_training_coverage_required", False)
            ),
            calibration_use_plate=bool(state.get("calibration_use_plate", True)),
            calibration_plate_shuffle=bool(state.get("calibration_plate_shuffle", False)),
            calibration_shuffle_seed=int(state.get("calibration_shuffle_seed", 42)),
        )
        builder.biological_categories = {str(k): list(v) for k, v in dict(state["biological_categories"]).items()}
        builder.observation_categories = {str(k): list(v) for k, v in dict(state["observation_categories"]).items()}
        builder.max_train_time = float(state["max_train_time"])
        builder.chemical_names = list(state["chemical_names"])
        for name in ("chemical_matrix", "chemical_mean", "chemical_scale", "chemical_semantic_mean", "chemical_semantic_scale", "strain_mean", "strain_scale", "strain_kernel_centers", "response_prior_global_sum", "response_prior_global_count"):
            value = state.get(name)
            if value is not None:
                setattr(builder, name, np.asarray(value, dtype=np.float32))
        bandwidth = state.get("strain_kernel_bandwidth")
        if bandwidth is not None:
            builder.strain_kernel_bandwidth = float(bandwidth)
        payload = state["strain_table"]
        if payload is not None:
            builder.strain_table = pd.DataFrame(payload["data"], columns=payload["columns"], index=payload["index"]).astype(np.float32)
            builder.strain_table.index = builder.strain_table.index.map(normalize_entity_key)
        builder.strain_columns = list(state["strain_columns"])
        builder.strain_status_columns = list(state.get("strain_status_columns", []))
        builder.strain_semantic_columns = list(
            state.get(
                "strain_semantic_columns",
                [
                    column
                    for column in builder.strain_columns
                    if column not in builder.strain_status_columns
                ],
            )
        )
        chemical_payload = state.get("chemical_semantic_table")
        if chemical_payload is not None:
            builder.chemical_semantic_table = pd.DataFrame(chemical_payload["data"], columns=chemical_payload["columns"], index=chemical_payload["index"]).astype(np.float32)
            builder.chemical_semantic_table.index = builder.chemical_semantic_table.index.map(normalize_entity_key)
        builder.chemical_semantic_columns = list(state.get("chemical_semantic_columns", []))
        builder.chemical_registry_records = {
            str(key): dict(value)
            for key, value in dict(state.get("chemical_registry_records", {})).items()
        }
        builder.strain_registry_records = {
            str(key): dict(value)
            for key, value in dict(state.get("strain_registry_records", {})).items()
        }
        builder.chemical_registry_sha256 = str(state.get("chemical_registry_sha256", ""))
        builder.strain_registry_sha256 = str(state.get("strain_registry_sha256", ""))
        builder.chemical_parent_records = {
            str(key): dict(value)
            for key, value in dict(state.get("chemical_parent_records", {})).items()
        }
        builder.chemical_identity_risk_records = {
            str(key): dict(value)
            for key, value in dict(state.get("chemical_identity_risk_records", {})).items()
        }
        builder.plate_training_assignments = {str(k): str(v) for k, v in dict(state.get("plate_training_assignments", {})).items()}
        builder.observation_slices = {str(k): tuple(v) for k, v in dict(state.get("observation_slices", {})).items()}
        builder.response_prior_mode = str(state.get("response_prior_mode", "none"))
        builder.response_prior_alpha = float(state.get("response_prior_alpha", 4.0))
        builder.response_prior_dim = int(state.get("response_prior_dim", 0))
        builder.response_prior_chemical = dict(state.get("response_prior_chemical", {}))
        builder.response_prior_strain = dict(state.get("response_prior_strain", {}))
        builder.entity_key_mode = str(
            state.get("entity_key_mode", "normalized_raw_v0")
        )
        builder.semantic_training_coverage = dict(
            state.get("semantic_training_coverage", {})
        )
        builder.strain_entity_keys = list(state.get("strain_entity_keys", []))
        builder.chemical_entity_keys = list(state.get("chemical_entity_keys", []))
        builder.pair_entity_keys = [
            (str(value[0]), str(value[1]))
            for value in state.get("pair_entity_keys", [])
        ]
        builder.context_entity_keys = [
            tuple(str(part) for part in value)
            for value in state.get("context_entity_keys", [])
        ]
        builder.strain_raw_entity_keys = list(
            state.get("strain_raw_entity_keys", [])
        )
        builder.chemical_raw_entity_keys = list(
            state.get("chemical_raw_entity_keys", [])
        )
        builder.pair_raw_entity_keys = [
            (str(value[0]), str(value[1]))
            for value in state.get("pair_raw_entity_keys", [])
        ]
        builder.context_raw_entity_keys = [
            tuple(str(part) for part in value)
            for value in state.get("context_raw_entity_keys", [])
        ]
        # Older checkpoints did not persist canonical expert vocabularies.  The
        # fallback changes no legacy model tensor and only makes feature state
        # safe to inspect with the new builder.
        if not builder.strain_entity_keys:
            builder.strain_entity_keys = sorted(
                {normalize_entity_key(value) for value in builder.biological_categories.get(STRAIN, [])}
            )
        if not builder.chemical_entity_keys:
            builder.chemical_entity_keys = sorted(
                {normalize_entity_key(value) for value in builder.biological_categories.get(CHEMICAL, [])}
            )
        return builder

    def summary(self) -> dict[str, object]:
        strain_encoded_dim = (
            len(self.strain_kernel_centers)
            if self.strain_feature_transform in {"rbf", "nearest"}
            and self.strain_kernel_centers is not None
            else len(self.strain_semantic_columns)
        )
        return {
            "biological_category_sizes": {key: len(value) for key, value in self.biological_categories.items()},
            "observation_category_sizes": {key: len(value) for key, value in self.observation_categories.items()},
            "uses_chemical_structure": self.chemical_matrix is not None,
            "uses_chemical_semantics": self.chemical_semantic_table is not None,
            "uses_strain_semantics": self.strain_table is not None,
            "chemical_feature_dim": (
                (0 if self.chemical_matrix is None else int(self.chemical_matrix.shape[1]))
                + len(self.chemical_semantic_columns)
                + (
                    (
                        len(STATUS_COLUMNS)
                        if self.semantic_identity_policy is None
                        else len(IDENTITY_STATUS_COLUMNS)
                    )
                    if self.chemical_registry_records
                    and (self.chemical_map is not None or self.chemical_features_path is not None)
                    else 0
                )
            ),
            "chemical_status_dim": (
                (
                    len(STATUS_COLUMNS)
                    if self.semantic_identity_policy is None
                    else len(IDENTITY_STATUS_COLUMNS)
                )
                if self.chemical_registry_records
                and (self.chemical_map is not None or self.chemical_features_path is not None)
                else 0
            ),
            "strain_feature_transform": self.strain_feature_transform,
            "strain_feature_dim": strain_encoded_dim + (
                (
                    len(STATUS_COLUMNS)
                    if self.semantic_identity_policy is None
                    else len(IDENTITY_STATUS_COLUMNS)
                )
                if self.strain_table is not None
                else 0
            ),
            "strain_semantic_dim": len(self.strain_semantic_columns),
            "strain_encoded_semantic_dim": strain_encoded_dim,
            "strain_kernel_center_count": (
                0
                if self.strain_kernel_centers is None
                else int(len(self.strain_kernel_centers))
            ),
            "strain_status_dim": (
                0
                if self.strain_table is None
                else (
                    len(STATUS_COLUMNS)
                    if self.semantic_identity_policy is None
                    else len(IDENTITY_STATUS_COLUMNS)
                )
            ),
            "allow_proxy_semantics": self.allow_proxy_semantics,
            "chemical_structure_view": self.chemical_structure_view,
            "semantic_identity_policy": (
                self.semantic_identity_policy or "legacy_v1"
            ),
            "semantic_training_coverage": self.semantic_training_coverage,
            "chemical_registry_sha256": self.chemical_registry_sha256,
            "strain_registry_sha256": self.strain_registry_sha256,
            "calibration_use_plate": self.calibration_use_plate,
            "calibration_plate_shuffle": self.calibration_plate_shuffle,
            "observation_slices": self.observation_slices,
            "response_prior_mode": self.response_prior_mode,
            "response_prior_alpha": self.response_prior_alpha,
            "strain_entity_count": len(self.strain_entity_keys),
            "chemical_entity_count": len(self.chemical_entity_keys),
            "pair_entity_count": len(self.pair_entity_keys),
            "context_time_count": len(self.context_entity_keys),
            "entity_key_mode": self.entity_key_mode,
            "entity_key_normalization": "unicode_nfkc_whitespace_casefold",
        }
