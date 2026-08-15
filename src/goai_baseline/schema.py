"""Canonical metadata schema and field-level rules."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


SAMPLE_ID = "sample_ID"
SPLIT = "split_final"
STRAIN = "Strains"
CHEMICAL = "perturbation_no_concentration"
MEDIUM = "Medium"
TEMPERATURE = "Temperature"
TIME = "pert_time"
TIME_UNIT = "pert_time_unit"
DATA_SOURCE = "data_source"
INSTRUMENT = "instrument"
PLATE = "Yeast_cell_plate"

REQUIRED_METADATA_COLUMNS = (
    SAMPLE_ID,
    DATA_SOURCE,
    STRAIN,
    MEDIUM,
    TEMPERATURE,
    TIME,
    TIME_UNIT,
    "pert_id",
    CHEMICAL,
    INSTRUMENT,
    PLATE,
    "protein_well",
    SPLIT,
    "strain_role",
    "chemical_role",
)

BIOLOGICAL_FIELDS = (STRAIN, CHEMICAL, MEDIUM, TEMPERATURE, TIME)
MATCH_CONTROL_FIELDS = (
    DATA_SOURCE,
    INSTRUMENT,
    PLATE,
    STRAIN,
    MEDIUM,
    TEMPERATURE,
    TIME,
    TIME_UNIT,
)
CONTROL_NAMES = frozenset({"water", "dmso"})
QUALITY_CONTROL_NAME = "quality control"


def require_metadata_columns(metadata: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_METADATA_COLUMNS if column not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")


def require_unique_sample_ids(metadata: pd.DataFrame, label: str) -> None:
    if metadata[SAMPLE_ID].isna().any():
        raise ValueError(f"{label} contains missing sample_ID values")
    duplicates = metadata.loc[metadata[SAMPLE_ID].duplicated(), SAMPLE_ID].head(5).tolist()
    if duplicates:
        raise ValueError(f"{label} contains duplicate sample_ID values: {duplicates}")


def normalise_name(values: Iterable[object]) -> pd.Series:
    return pd.Series(values, copy=False).astype(str).str.strip().str.casefold()


def control_mask(metadata: pd.DataFrame) -> pd.Series:
    return normalise_name(metadata[CHEMICAL]).isin(CONTROL_NAMES).set_axis(metadata.index)


def quality_control_mask(metadata: pd.DataFrame) -> pd.Series:
    return normalise_name(metadata[CHEMICAL]).eq(QUALITY_CONTROL_NAME).set_axis(metadata.index)


def treatment_mask(metadata: pd.DataFrame) -> pd.Series:
    return ~(control_mask(metadata) | quality_control_mask(metadata))
