"""Produce a machine-readable audit of the released GOAI virtual-cell data.

This script never reads a test proteome (none is configured) and never writes
outside the requested output directory.  It distinguishes controls available
to a training fold from controls that an evaluator may use after predictions
have been frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from goai_baseline.audit import audit_inputs
from goai_baseline.config import load_config
from goai_baseline.controls import exact_control_predictions
from goai_baseline.preprocess import feature_contract, prepare_data
from goai_baseline.schema import (
    CHEMICAL,
    DATA_SOURCE,
    INSTRUMENT,
    MATCH_CONTROL_FIELDS,
    MEDIUM,
    PLATE,
    SAMPLE_ID,
    SPLIT,
    STRAIN,
    TEMPERATURE,
    TIME,
    control_mask,
    quality_control_mask,
    treatment_mask,
)


def _plain(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    return value


def _sample_kind(metadata: pd.DataFrame) -> pd.Series:
    result = pd.Series("treatment", index=metadata.index, dtype=object)
    result.loc[control_mask(metadata)] = "control"
    result.loc[quality_control_mask(metadata)] = "quality_control"
    return result


def _level_audit(metadata: pd.DataFrame, train_ids: pd.Index, field: str) -> dict[str, object]:
    train_levels = set(metadata.loc[train_ids, field].astype(str))
    result: dict[str, object] = {"train_n_unique": len(train_levels)}
    for split, rows in metadata.groupby(SPLIT, sort=False):
        levels = set(rows[field].astype(str))
        result[str(split)] = {
            "n_unique": len(levels),
            "unseen_vs_train": sorted(levels - train_levels),
            "seen_vs_train_count": len(levels & train_levels),
        }
    return result


def _control_coverage(data) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, group in data.metadata.groupby(SPLIT, sort=False):
        ids = group.index[treatment_mask(group).to_numpy()]
        if len(ids) == 0:
            continue
        for label, pool in (
            ("fit_train_only", data.train_ids),
            ("all_released_train_val_scorer_only", data.metadata.index),
        ):
            matched = exact_control_predictions(data.metadata, data.y_log2, ids, pool)
            matched_rows = int(matched.has_exact_match.sum())
            rows.append(
                {
                    "split": str(split),
                    "control_pool": label,
                    "treatment_rows": int(len(ids)),
                    "matched_rows": matched_rows,
                    "matched_fraction": matched_rows / len(ids),
                    "observed_delta_values": int(
                        (data.y_log2.loc[ids] - matched.predictions.loc[ids]).notna().to_numpy().sum()
                    ),
                }
            )
    return rows


def _context_overlap(metadata: pd.DataFrame, train_ids: pd.Index) -> list[dict[str, object]]:
    train = metadata.loc[train_ids]
    train_biology = set(map(tuple, train.loc[:, [STRAIN, MEDIUM, TEMPERATURE, TIME]].astype(str).to_numpy()))
    train_observation = set(map(tuple, train.loc[:, [DATA_SOURCE, INSTRUMENT, PLATE]].astype(str).to_numpy()))
    rows: list[dict[str, object]] = []
    for split, group in metadata.groupby(SPLIT, sort=False):
        biology = list(map(tuple, group.loc[:, [STRAIN, MEDIUM, TEMPERATURE, TIME]].astype(str).to_numpy()))
        observation = list(map(tuple, group.loc[:, [DATA_SOURCE, INSTRUMENT, PLATE]].astype(str).to_numpy()))
        rows.append(
            {
                "split": str(split),
                "rows": int(len(group)),
                "biology_context_seen_fraction": float(np.mean([key in train_biology for key in biology])),
                "observation_context_seen_fraction": float(np.mean([key in train_observation for key in observation])),
            }
        )
    return rows


def _chemical_audit(config, data, test_metadata: pd.DataFrame) -> dict[str, object]:
    path = config.path.parent.parent / "data" / "processed" / "chemical_entity_map.tsv"
    if not path.is_file():
        return {"path": str(path), "exists": False}
    mapping = pd.read_csv(path, sep="\t", keep_default_na=False)
    all_names = set(data.metadata[CHEMICAL].astype(str)) | set(test_metadata[CHEMICAL].astype(str))
    mapped_names = set(mapping["raw_name"].astype(str))
    unresolved = mapping.loc[
        (~mapping["is_control"].astype(bool)) & mapping["status"].ne("resolved"), "raw_name"
    ].astype(str)
    resolved_keys = mapping.loc[
        mapping["status"].eq("resolved") & mapping["inchikey"].astype(str).ne(""), "inchikey"
    ].astype(str)
    return {
        "path": str(path.resolve()),
        "exists": True,
        "rows": int(len(mapping)),
        "metadata_name_count": len(all_names),
        "missing_metadata_names": sorted(all_names - mapped_names),
        "extra_mapping_names": sorted(mapped_names - all_names),
        "resolved_noncontrol_count": int(
            ((~mapping["is_control"].astype(bool)) & mapping["status"].eq("resolved")).sum()
        ),
        "unresolved_noncontrol_names": sorted(unresolved.tolist()),
        "duplicate_resolved_inchikeys": sorted(resolved_keys[resolved_keys.duplicated(keep=False)].unique().tolist()),
        "source_values": sorted(mapping["source"].astype(str).unique().tolist()),
        "retrieved_utc_min": str(mapping["retrieved_utc"].astype(str).min()),
        "retrieved_utc_max": str(mapping["retrieved_utc"].astype(str).max()),
    }


def audit(config_path: str | Path, output_dir: str | Path) -> Path:
    config = load_config(config_path)
    inputs = audit_inputs(config)
    data = prepare_data(config)
    test = pd.read_csv(config.data.metadata_test, low_memory=False).set_index(SAMPLE_ID, verify_integrity=True)
    metadata = data.metadata.copy()
    kinds = _sample_kind(metadata)

    split_rows = []
    for split, group in metadata.groupby(SPLIT, sort=False):
        ids = group.index
        values = data.y_log2.loc[ids].to_numpy(dtype=np.float32)
        split_rows.append(
            {
                "split": str(split),
                "rows": int(len(group)),
                "treatments": int((kinds.loc[ids] == "treatment").sum()),
                "controls": int((kinds.loc[ids] == "control").sum()),
                "quality_controls": int((kinds.loc[ids] == "quality_control").sum()),
                "observed_fraction_retained_proteins": float(np.isfinite(values).mean()),
                "strains": int(group[STRAIN].nunique()),
                "chemicals": int(group[CHEMICAL].nunique()),
                "times": sorted(pd.to_numeric(group[TIME], errors="raise").unique().tolist()),
            }
        )

    retained_missing = data.y_log2.loc[data.train_ids].isna().mean(axis=0)
    report = {
        "protocol": "goai_project_audit_v2",
        "inputs": inputs,
        "feature_contract": feature_contract(data, config),
        "protein_audit": {
            "raw_count": int(len(data.missing_rate)),
            "retained_count": int(len(data.proteins)),
            "removed_count": int(len(data.missing_rate) - len(data.proteins)),
            "threshold": float(config.data.missing_rate_threshold),
            "retained_train_missing_rate_min": float(retained_missing.min()),
            "retained_train_missing_rate_median": float(retained_missing.median()),
            "retained_train_missing_rate_max": float(retained_missing.max()),
            "removed_missing_rate_min": float(data.missing_rate.loc[~data.missing_rate.index.isin(data.proteins)].min()),
        },
        "split_summary": split_rows,
        "entity_levels": {
            field: _level_audit(metadata, data.train_ids, field)
            for field in (STRAIN, CHEMICAL, MEDIUM, TEMPERATURE, TIME, DATA_SOURCE, INSTRUMENT, PLATE)
        },
        "roles": {
            column: metadata.groupby(SPLIT, sort=False)[column].value_counts(dropna=False).to_dict()
            for column in ("strain_role", "chemical_role")
        },
        "control_match_fields": list(MATCH_CONTROL_FIELDS),
        "control_coverage": _control_coverage(data),
        "context_overlap": _context_overlap(metadata, data.train_ids),
        "test_summary": {
            "rows": int(len(test)),
            "split_counts": test[SPLIT].value_counts(dropna=False).to_dict(),
            "strains": sorted(test[STRAIN].astype(str).unique().tolist()),
            "chemicals": sorted(test[CHEMICAL].astype(str).unique().tolist()),
            "times": sorted(pd.to_numeric(test[TIME], errors="raise").unique().tolist()),
        },
        "chemical_mapping": _chemical_audit(config, data, test),
        "guardrails": {
            "test_proteome_configured": False,
            "outer_validation_labels_available_locally": True,
            "outer_validation_must_remain_frozen": True,
            "released_data_is_competition_restricted": True,
        },
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(_plain(report), handle, ensure_ascii=False, indent=2)
    pd.DataFrame(split_rows).to_csv(destination / "split_summary.csv", index=False)
    pd.DataFrame(report["control_coverage"]).to_csv(destination / "control_coverage.csv", index=False)
    pd.DataFrame(report["context_overlap"]).to_csv(destination / "context_overlap.csv", index=False)
    print(json.dumps(_plain(report), ensure_ascii=False, indent=2))
    print(f"Wrote audit: {destination.resolve()}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit released GOAI virtual-cell data and OOD structure")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output-dir", default="runs/audit_v2")
    args = parser.parse_args()
    audit(args.config, args.output_dir)


if __name__ == "__main__":
    main()
