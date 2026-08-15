"""Submission contract checks for condition-model predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .schema import SAMPLE_ID, require_metadata_columns, require_unique_sample_ids


def verify_submission(
    prediction_path: str | Path,
    metadata_test_path: str | Path,
    proteins: list[str],
) -> dict[str, object]:
    prediction_file = Path(prediction_path)
    test_file = Path(metadata_test_path)
    submission = pd.read_csv(prediction_file)
    test_metadata = pd.read_csv(test_file, low_memory=False)
    require_metadata_columns(test_metadata)
    require_unique_sample_ids(test_metadata, "test metadata")
    if submission.columns.empty or submission.columns[0] != SAMPLE_ID:
        raise ValueError("Submission must start with a sample_ID column")
    if submission[SAMPLE_ID].isna().any() or submission[SAMPLE_ID].duplicated().any():
        raise ValueError("Submission sample_ID must be present and unique")
    expected_ids = test_metadata[SAMPLE_ID].tolist()
    if submission[SAMPLE_ID].tolist() != expected_ids:
        raise ValueError("Submission sample_ID order does not match official test metadata")
    predicted_proteins = submission.columns[1:].tolist()
    if predicted_proteins != proteins:
        raise ValueError("Submission protein columns do not match the frozen feature contract")
    values = submission.iloc[:, 1:].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Submission contains NaN or infinite values")
    return {
        "rows": int(len(submission)),
        "proteins": int(len(proteins)),
        "sample_id_order_matches_metadata_test": True,
        "finite_values": True,
        "prediction_scale": "log2",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a GOAI baseline submission")
    parser.add_argument("prediction_csv")
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-contract", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    with Path(args.feature_contract).open("r", encoding="utf-8") as handle:
        proteins = json.load(handle)["protein_names"]
    report = verify_submission(args.prediction_csv, config.data.metadata_test, proteins)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
