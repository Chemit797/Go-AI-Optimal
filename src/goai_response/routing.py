"""Deterministic row routing from the entities actually seen during fitting.

Scenario labels describe the organizer's original split.  They are not a
reliable proxy for a checkpoint's support after released validation rows are
added to a final refit.  This module therefore derives Rxy directly from the
    fit support manifest.  Expert eligibility follows the fold-fit canonical
    support key when the registry supplies one, otherwise ``raw:<normalized>``.
    Proxy identities keep their own canonical IDs and are never collapsed onto
    their proxy targets.  Raw keys remain in the audit for identity-drift checks.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from goai_baseline.schema import (
    CHEMICAL,
    MEDIUM,
    SAMPLE_ID,
    SPLIT,
    STRAIN,
    TEMPERATURE,
    TIME,
    TIME_UNIT,
    treatment_mask,
)
from .entities import normalize_entity_key

SUPPORT_REGIMES = ("R00", "R10", "R01", "R11")
CONTROL_ROUTE = "control"
_SUPPORT_TO_REGIME = {
    "unseen_both": "R00",
    "unseen_chemical": "R10",
    "unseen_strain": "R01",
    "seen_seen": "R11",
}
_REQUIRED_FLAG_COLUMNS = {
    "chemical_seen_in_fit",
    "strain_seen_in_fit",
    "chemical_semantic_supported",
    "strain_semantic_supported",
    "chemical_proxy",
    "strain_proxy",
    "chemical_missing",
    "strain_missing",
    "support_route",
}


def support_flags(metadata: pd.DataFrame, manifest: Mapping[str, object]) -> pd.DataFrame:
    """Lazy bridge to the authoritative entity registry implementation."""

    from .entities import support_flags as entity_support_flags

    return entity_support_flags(metadata, manifest)


def manifest_sha256(manifest: Mapping[str, object]) -> str:
    """Lazy bridge so legacy split routing does not require entity resources."""

    from .entities import manifest_sha256 as entity_manifest_sha256

    return entity_manifest_sha256(dict(manifest))


def support_route_audit(
    metadata: pd.DataFrame,
    support_manifest: Mapping[str, object],
) -> pd.DataFrame:
    """Classify every row without consulting split/role labels as model inputs.

    ``split_final`` is copied into the audit output only for reporting.  The
    route itself comes exclusively from canonical support-key membership in the
    supplied fit support manifest (with a legacy raw-key fallback); canonical
    IDs and raw keys are retained for audit.
    Controls and QC are isolated because they
    are not chemical-response expert examples.
    """

    flags = support_flags(metadata, support_manifest)
    if len(flags) != len(metadata):
        raise ValueError("Entity support flags do not align with metadata rows")
    flags = flags.copy()
    flags.index = metadata.index
    missing = sorted(_REQUIRED_FLAG_COLUMNS - set(flags.columns))
    if missing:
        raise ValueError(f"Entity support flags are missing columns: {missing}")
    unknown = sorted(set(flags["support_route"].astype(str)) - set(_SUPPORT_TO_REGIME))
    if unknown:
        raise ValueError(f"Unknown entity support routes: {unknown}")

    if SAMPLE_ID in metadata.columns:
        sample_ids = metadata[SAMPLE_ID].astype(str).to_numpy()
    else:
        sample_ids = metadata.index.astype(str).to_numpy()
    audit = pd.DataFrame(
        {
            SAMPLE_ID: sample_ids,
            SPLIT: metadata[SPLIT].astype(str).to_numpy() if SPLIT in metadata else "",
            STRAIN: metadata[STRAIN].astype(str).to_numpy(),
            CHEMICAL: metadata[CHEMICAL].astype(str).to_numpy(),
            "is_treatment": treatment_mask(metadata).to_numpy(dtype=bool),
        },
        index=metadata.index,
    )
    for column in flags.columns:
        audit[column] = flags[column].to_numpy()
    seen_support = support_manifest.get("seen_support_keys")
    if isinstance(seen_support, Mapping):
        pair_support = {
            tuple(str(part) for part in value)
            for value in seen_support.get("pair", [])
        }
        context_support = {
            tuple(str(part) for part in value)
            for value in seen_support.get("context_time", [])
        }
    else:
        pair_support = {
            tuple(str(part) for part in value)
            for value in support_manifest.get("seen_pair_keys", [])
        }
        context_support = {
            tuple(str(part) for part in value)
            for value in support_manifest.get("seen_context_time_keys", [])
        }
    strain_keys = flags["strain_support_key"].astype(str)
    chemical_keys = flags["chemical_support_key"].astype(str)
    pair_keys = list(zip(strain_keys, chemical_keys))
    if all(field in metadata for field in (MEDIUM, TEMPERATURE, TIME, TIME_UNIT)):
        context_keys = [
            (
                strain,
                chemical,
                normalize_entity_key(row[MEDIUM]),
                normalize_entity_key(row[TEMPERATURE]),
                normalize_entity_key(row[TIME]),
                normalize_entity_key(row[TIME_UNIT]),
            )
            for (_, row), strain, chemical in zip(
                metadata.iterrows(), strain_keys, chemical_keys
            )
        ]
    else:
        context_keys = [tuple() for _ in range(len(metadata))]
    audit["strain_seen"] = flags["strain_seen_in_fit"].astype(bool).to_numpy()
    audit["chemical_seen"] = flags["chemical_seen_in_fit"].astype(bool).to_numpy()
    audit["pair_seen"] = [key in pair_support for key in pair_keys]
    audit["context_time_seen"] = [
        bool(key) and key in context_support for key in context_keys
    ]
    treatment = audit["is_treatment"].astype(bool)
    audit["strain_expert_enabled"] = audit["strain_seen"]
    audit["chemical_expert_enabled"] = treatment & audit["chemical_seen"]
    audit["pair_expert_enabled"] = treatment & audit["pair_seen"]
    audit["strain_expert_reason"] = audit["strain_seen"].map(
        {True: "canonical_support_key_seen", False: "canonical_support_key_unseen"}
    )
    audit["chemical_expert_reason"] = np.where(
        ~treatment,
        "non_treatment_response_disabled",
        np.where(
            audit["chemical_seen"],
            "canonical_support_key_seen",
            "canonical_support_key_unseen",
        ),
    )
    audit["pair_expert_reason"] = np.where(
        ~treatment,
        "non_treatment_response_disabled",
        np.where(
            audit["pair_seen"],
            "canonical_pair_seen",
            "canonical_pair_unseen",
        ),
    )
    audit["context_time_reason"] = np.where(
        audit["context_time_seen"],
        "exact_context_time_seen",
        "exact_context_time_unseen",
    )
    audit["support_regime"] = flags["support_route"].astype(str).map(_SUPPORT_TO_REGIME).to_numpy()
    audit.loc[~audit["is_treatment"], "support_regime"] = CONTROL_ROUTE
    audit["fit_support_manifest_sha256"] = manifest_sha256(dict(support_manifest))
    return audit


def support_route_masks(audit: pd.DataFrame) -> dict[str, pd.Series]:
    """Return exhaustive, mutually exclusive masks for deterministic routing."""

    if "support_regime" not in audit:
        raise ValueError("Route audit is missing support_regime")
    allowed = {*SUPPORT_REGIMES, CONTROL_ROUTE}
    unknown = sorted(set(audit["support_regime"].astype(str)) - allowed)
    if unknown:
        raise ValueError(f"Route audit contains unsupported regimes: {unknown}")
    masks = {name: audit["support_regime"].astype(str).eq(name) for name in (*SUPPORT_REGIMES, CONTROL_ROUTE)}
    total = sum(mask.to_numpy(dtype=int) for mask in masks.values())
    if not (total == 1).all():
        raise AssertionError("Support routing is not exhaustive and mutually exclusive")
    return masks
