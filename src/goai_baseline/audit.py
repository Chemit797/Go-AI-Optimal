"""Input auditing and data-usage guardrails."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from .config import BaselineConfig, load_config
from .schema import SAMPLE_ID, require_metadata_columns, require_unique_sample_ids


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_allowed_inputs(config: BaselineConfig) -> None:
    """Validate the declared training and submission input layout."""
    expected_submission_metadata_name = "wayb_wayc_metadata_test.csv"
    if config.data.metadata_test.name.casefold() != expected_submission_metadata_name:
        raise ValueError("metadata_test must point to the official submission metadata file")
    if "test" in config.data.proteome_train_val.name.casefold():
        raise ValueError("proteome_train_val must point to a train/validation protein matrix")
    for path in (
        config.data.metadata_train_val,
        config.data.proteome_train_val,
        config.data.metadata_test,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Configured input does not exist: {path}")


def audit_inputs(config: BaselineConfig) -> dict[str, object]:
    assert_allowed_inputs(config)
    train_meta = pd.read_csv(config.data.metadata_train_val, low_memory=False)
    test_meta = pd.read_csv(config.data.metadata_test, low_memory=False)
    require_metadata_columns(train_meta)
    require_metadata_columns(test_meta)
    require_unique_sample_ids(train_meta, "train/validation metadata")
    require_unique_sample_ids(test_meta, "test metadata")
    overlap = set(train_meta[SAMPLE_ID]) & set(test_meta[SAMPLE_ID])
    if overlap:
        raise ValueError(f"Metadata train/test sample_ID overlap: {sorted(overlap)[:5]}")

    protein_ids = pd.read_csv(config.data.proteome_train_val, usecols=[SAMPLE_ID])[SAMPLE_ID]
    if protein_ids.isna().any() or protein_ids.duplicated().any():
        raise ValueError("Protein matrix sample_ID must be present and unique")
    if set(protein_ids) != set(train_meta[SAMPLE_ID]):
        missing_targets = sorted(set(train_meta[SAMPLE_ID]) - set(protein_ids))[:5]
        extra_targets = sorted(set(protein_ids) - set(train_meta[SAMPLE_ID]))[:5]
        raise ValueError(
            "Metadata/protein sample_ID sets differ; "
            f"missing targets={missing_targets}, extra targets={extra_targets}"
        )

    return {
        "metadata_train_val_rows": int(len(train_meta)),
        "metadata_test_rows": int(len(test_meta)),
        "protein_rows": int(len(protein_ids)),
        "train_split_rows": int((train_meta["split_final"] == "train").sum()),
        "metadata_train_val_sha256": sha256(config.data.metadata_train_val),
        "proteome_train_val_sha256": sha256(config.data.proteome_train_val),
        "metadata_test_sha256": sha256(config.data.metadata_test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit GOAI baseline inputs")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = audit_inputs(load_config(args.config))
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
