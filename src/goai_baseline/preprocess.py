"""Sample alignment, train-only protein filtering, log2 conversion, and masks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import audit_inputs
from .config import BaselineConfig, load_config
from .schema import SAMPLE_ID, SPLIT, require_metadata_columns, require_unique_sample_ids


@dataclass
class PreprocessedData:
    metadata: pd.DataFrame
    y_log2: pd.DataFrame
    mask: pd.DataFrame
    proteins: list[str]
    train_ids: pd.Index
    missing_rate: pd.Series


def load_train_val(config: BaselineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(config.data.metadata_train_val, low_memory=False)
    proteome = pd.read_csv(config.data.proteome_train_val, low_memory=False)
    require_metadata_columns(metadata)
    require_unique_sample_ids(metadata, "train/validation metadata")
    if SAMPLE_ID not in proteome.columns:
        raise ValueError("Protein matrix is missing sample_ID")
    if proteome[SAMPLE_ID].isna().any() or proteome[SAMPLE_ID].duplicated().any():
        raise ValueError("Protein matrix sample_ID must be present and unique")
    if len(proteome.columns) <= 1:
        raise ValueError("Protein matrix has no protein columns")

    metadata = metadata.set_index(SAMPLE_ID, verify_integrity=True)
    proteome = proteome.set_index(SAMPLE_ID, verify_integrity=True)
    if set(metadata.index) != set(proteome.index):
        raise ValueError("Metadata and protein matrix sample_ID sets differ")
    proteome = proteome.reindex(metadata.index)
    try:
        proteome = proteome.astype(np.float32)
    except ValueError as error:
        raise ValueError("All protein columns must be numeric or missing") from error
    return metadata, proteome


def prepare_data(config: BaselineConfig) -> PreprocessedData:
    metadata, raw_proteome = load_train_val(config)
    train_ids = metadata.index[metadata[SPLIT].eq("train")]
    if train_ids.empty:
        raise ValueError("No split_final == 'train' rows found")

    finite = np.isfinite(raw_proteome.to_numpy(copy=False))
    invalid_non_missing = raw_proteome.notna().to_numpy(copy=False) & ~finite
    if invalid_non_missing.any():
        raise ValueError("Protein matrix contains non-finite observed values")
    nonpositive = raw_proteome.notna().to_numpy(copy=False) & (raw_proteome.to_numpy(copy=False) <= 0)
    if nonpositive.any():
        raise ValueError("Protein matrix contains non-positive observed intensities")

    missing_rate = raw_proteome.loc[train_ids].isna().mean(axis=0)
    keep = missing_rate < config.data.missing_rate_threshold
    if not keep.any():
        raise ValueError("Training-only missingness filter removed every protein")
    filtered = raw_proteome.loc[:, keep]
    y_log2 = np.log2(filtered)
    mask = y_log2.notna()
    proteins = y_log2.columns.astype(str).tolist()
    return PreprocessedData(
        metadata=metadata,
        y_log2=y_log2,
        mask=mask,
        proteins=proteins,
        train_ids=train_ids,
        missing_rate=missing_rate,
    )


def feature_contract(data: PreprocessedData, config: BaselineConfig) -> dict[str, object]:
    return {
        "protein_names": data.proteins,
        "n_raw_proteins": int(len(data.missing_rate)),
        "n_kept_proteins": int(len(data.proteins)),
        "missing_rate_threshold": config.data.missing_rate_threshold,
        "training_split": "train",
        "train_sample_count": int(len(data.train_ids)),
        "observed_fraction": float(data.mask.to_numpy(dtype=bool).mean()),
        "target_scale": "log2",
    }


def write_feature_contract(path: str | Path, data: PreprocessedData, config: BaselineConfig) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(feature_contract(data, config), handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GOAI baseline data")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="runs/preprocess/feature_contract.json")
    args = parser.parse_args()
    config = load_config(args.config)
    audit_inputs(config)
    data = prepare_data(config)
    write_feature_contract(args.output, data, config)
    print(f"Prepared {len(data.metadata):,} rows and {len(data.proteins):,} proteins")
    print(f"Wrote feature contract: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
