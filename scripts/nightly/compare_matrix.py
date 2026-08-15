"""Compare discovery producers against a same-seed control.

This is a quick-screen diagnostic only.  It deliberately cannot issue a
promotion decision; final promotion is owned by ``promotion_gate.py`` after a
complete 4-fold, three-model-seed confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "fc_pcc",
    "context_residual_pcc",
    "drug_residual_pcc",
    "high_effect_pcc",
    "high_effect_f1",
    "absolute_sample_r2_median",
)
PROTOCOL_LABEL = "LOCAL_STRICT_OOF_NOT_OFFICIAL"


def _scenario_hash(path: Path, scenario: str) -> str:
    frame = pd.read_csv(path, keep_default_na=False)
    frame = frame.loc[frame["scenario"].eq(scenario)].sort_values(["fold", "sample_ID"])
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def _bootstrap(delta: np.ndarray, seed: int, draws: int = 10000) -> dict[str, float | int]:
    finite = delta[np.isfinite(delta)]
    if not len(finite):
        return {"n_units": 0, "mean_delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "probability_positive": float("nan")}
    rng = np.random.default_rng(seed)
    sampled = finite[rng.integers(0, len(finite), size=(draws, len(finite)))].mean(axis=1)
    return {
        "n_units": int(len(finite)),
        "mean_delta": float(finite.mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "probability_positive": float((sampled > 0).mean()),
    }


def discovery_decisions(result: pd.DataFrame) -> list[dict[str, object]]:
    """Nominate confirmation candidates while hard-disabling promotion."""
    decisions: list[dict[str, object]] = []
    if result.empty:
        return decisions
    fc = result.loc[result["metric"].eq("fc_pcc")]
    high = result.loc[result["metric"].eq("high_effect_pcc")][
        ["candidate", "model_seed", "scenario", "mean_delta"]
    ].rename(columns={"mean_delta": "high_effect_delta"})
    merged = fc.merge(high, on=["candidate", "model_seed", "scenario"], how="left")
    for row in merged.to_dict(orient="records"):
        nominate = (
            (row["mean_delta"] >= 0.005 or (row["mean_delta"] >= 0.002 and row["ci_low"] > 0))
            and (pd.isna(row["high_effect_delta"]) or row["high_effect_delta"] >= -0.005)
        )
        decisions.append({
            **row,
            "protocol_label": PROTOCOL_LABEL,
            "promotion_eligible": False,
            "promote": False,
            "nominate_for_confirmation": bool(nominate),
            "decision": "candidate_for_confirmation" if nominate else "screen_only",
            "reason": (
                "Discovery/fold-bootstrap evidence can nominate confirmation only; "
                "it cannot satisfy held-out-entity CI, 4-fold, or three-seed promotion gates."
            ),
        })
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--control-id", default="N-CTRL-01")
    parser.add_argument(
        "--control-root",
        default=None,
        help="Optional matrix root containing the same-seed control producer.",
    )
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    control_root = Path(args.control_root).resolve() if args.control_root else root
    output = root / "consumer"
    output.mkdir(exist_ok=True)
    run_dirs = sorted(path.parent for path in (root / "producers").glob("*/S*/oof_summary.csv"))
    records: list[dict[str, object]] = []
    for candidate in run_dirs:
        candidate_id = candidate.parent.name
        seed = int(candidate.name[1:])
        if candidate_id == args.control_id:
            continue
        control = control_root / "producers" / args.control_id / candidate.name
        if not (control / "oof_summary.csv").is_file():
            continue
        candidate_folds = pd.read_csv(candidate / "oof_official_proxy_metrics.csv")
        control_folds = pd.read_csv(control / "oof_official_proxy_metrics.csv")
        for scenario in sorted(set(candidate_folds["scenario"]) & set(control_folds["scenario"])):
            candidate_hash = _scenario_hash(candidate / "fold_assignments.csv", scenario)
            control_hash = _scenario_hash(control / "fold_assignments.csv", scenario)
            if candidate_hash != control_hash:
                raise ValueError(f"fold assignment mismatch for {candidate_id} {scenario}")
            left = control_folds.loc[control_folds["scenario"].eq(scenario)]
            right = candidate_folds.loc[candidate_folds["scenario"].eq(scenario)]
            paired = left.merge(right, on=["scenario", "fold"], suffixes=("_control", "_candidate"), validate="one_to_one")
            for metric_index, metric in enumerate(METRICS):
                a, b = f"{metric}_control", f"{metric}_candidate"
                if a not in paired or b not in paired:
                    continue
                delta = paired[b].to_numpy(dtype=float) - paired[a].to_numpy(dtype=float)
                records.append({
                    "candidate": candidate_id,
                    "model_seed": seed,
                    "control": args.control_id,
                    "scenario": scenario,
                    "metric": metric,
                    "fold_assignment_sha256": candidate_hash,
                    **_bootstrap(delta, 8100 + metric_index),
                })
    result = pd.DataFrame(records)
    result.to_csv(output / "paired_bootstrap.csv", index=False)
    decisions = discovery_decisions(result)
    with (output / "discovery_decisions.json").open("w", encoding="utf-8") as handle:
        json.dump(decisions, handle, ensure_ascii=False, indent=2)
    # Retain the historical path for callers, but make it fail closed.  No
    # quick-screen record written here can carry promote=true.
    with (output / "promotion_decisions.json").open("w", encoding="utf-8") as handle:
        json.dump(decisions, handle, ensure_ascii=False, indent=2)
    print(result.loc[result["metric"].eq("fc_pcc")].sort_values(["scenario", "mean_delta"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
