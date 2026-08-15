"""Quantify how strongly GOAI plate labels encode biological/task variables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import normalized_mutual_info_score


PLATE = "Yeast_cell_plate"
DEFAULT_TARGETS = (
    "Strains",
    "perturbation_no_concentration",
    "pert_time",
    "Medium",
    "Temperature",
    "data_source",
    "instrument",
    "split_final",
)


def _cramers_v(left: pd.Series, right: pd.Series) -> float:
    table = pd.crosstab(left.astype(str), right.astype(str)).to_numpy(dtype=np.float64)
    if table.size == 0:
        return float("nan")
    chi2 = float(chi2_contingency(table, correction=False)[0])
    n = float(table.sum())
    denominator = min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt((chi2 / n) / denominator)) if n > 0 and denominator > 0 else 0.0


def _weighted_purity(source: pd.Series, target: pd.Series) -> float:
    table = pd.crosstab(source.astype(str), target.astype(str)).to_numpy(dtype=np.int64)
    return float(table.max(axis=1).sum() / table.sum()) if table.size and table.sum() else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shuffle-seed", type=int, default=991)
    parser.add_argument("--shuffle-draws", type=int, default=200)
    args = parser.parse_args()

    metadata_path = Path(args.metadata).resolve()
    frame = pd.read_csv(metadata_path, low_memory=False)
    required = [PLATE, *DEFAULT_TARGETS]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")
    plate = frame[PLATE].fillna("<NA>").astype(str)
    rng = np.random.default_rng(args.shuffle_seed)
    rows: list[dict[str, object]] = []
    for target_name in DEFAULT_TARGETS:
        target = frame[target_name].fillna("<NA>").astype(str)
        observed_nmi = float(normalized_mutual_info_score(plate, target))
        shuffled = np.empty(args.shuffle_draws, dtype=np.float64)
        plate_values = plate.to_numpy()
        for draw in range(args.shuffle_draws):
            shuffled[draw] = normalized_mutual_info_score(rng.permutation(plate_values), target)
        rows.append(
            {
                "target": target_name,
                "target_levels": int(target.nunique()),
                "normalized_mutual_information": observed_nmi,
                "cramers_v": _cramers_v(plate, target),
                "plate_to_target_weighted_purity": _weighted_purity(plate, target),
                "target_to_plate_weighted_purity": _weighted_purity(target, plate),
                "shuffle_nmi_mean": float(shuffled.mean()),
                "shuffle_nmi_p95": float(np.quantile(shuffled, 0.95)),
                "shuffle_p_ge_observed": float((1 + np.sum(shuffled >= observed_nmi)) / (1 + len(shuffled))),
            }
        )

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values("normalized_mutual_information", ascending=False)
    result.to_csv(output / "plate_associations.csv", index=False)
    plate_sizes = plate.value_counts().rename_axis(PLATE).rename("n_rows").reset_index()
    plate_sizes.to_csv(output / "plate_sizes.csv", index=False)
    contract = {
        "protocol": "goai_plate_confounding_audit_v1",
        "metadata": str(metadata_path),
        "rows": int(len(frame)),
        "plates": int(plate.nunique()),
        "shuffle_seed": args.shuffle_seed,
        "shuffle_draws": args.shuffle_draws,
        "interpretation": "High association is evidence of confounding risk, not proof of target leakage. Model ablations and leave-one-plate-out OOF decide whether plate calibration is retained.",
    }
    with (output / "audit_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, ensure_ascii=False, indent=2)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
