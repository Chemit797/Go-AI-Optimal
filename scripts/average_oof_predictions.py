"""Average aligned OOF predictions across model seeds without tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from goai_baseline.schema import SAMPLE_ID


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def average(runs: list[Path], scenario: str, output_dir: Path) -> Path:
    if len(runs) < 2:
        raise ValueError("Seed averaging requires at least two OOF runs")
    assignment_hashes = {_sha256(run / "fold_assignments.csv") for run in runs}
    if len(assignment_hashes) != 1:
        raise ValueError("OOF seed runs do not share identical fold assignments")
    frames = [
        pd.read_csv(run / "oof_predictions" / f"{scenario}.csv").set_index(
            SAMPLE_ID, verify_integrity=True
        ).sort_index()
        for run in runs
    ]
    first = frames[0]
    for frame in frames[1:]:
        if not frame.index.equals(first.index) or not frame.columns.equals(first.columns):
            raise ValueError("OOF seed predictions are not aligned")
    result = sum(frames[1:], first.copy()) / len(frames)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "oof_predictions"
    prediction_dir.mkdir(exist_ok=True)
    result.index.name = SAMPLE_ID
    result.to_csv(prediction_dir / f"{scenario}.csv")
    shutil.copyfile(runs[0] / "fold_assignments.csv", output_dir / "fold_assignments.csv")
    manifest = {
        "protocol": "aligned_oof_seed_average_v1",
        "scenario": scenario,
        "seed_run_count": len(runs),
        "runs": [str(run.resolve()) for run in runs],
        "fold_assignments_sha256": next(iter(assignment_hashes)),
        "outer_validation_used": False,
        "tuned_parameters": [],
    }
    with (output_dir / "average_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Average aligned GOAI OOF seed predictions")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = average([Path(value) for value in args.runs], args.scenario, Path(args.output_dir))
    print(f"Wrote OOF seed average: {output.resolve()}")


if __name__ == "__main__":
    main()
