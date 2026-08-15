"""Summarize M7/M8 regime OOF, paired deltas, and fixed expert scales.

All outputs are explicitly labelled local strict OOF.  This script never
calls the GOAI scorer and must not be used to describe an official score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


PROTOCOL_LABEL = "LOCAL_STRICT_OOF_NOT_OFFICIAL"
FOUR_REGIMES = ("R00", "R10", "R01", "R11")
ALL_REPORT_SCENARIOS = FOUR_REGIMES + ("RT", "plate")
ALLOWED_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
SCALE_SELECTION_RULE_ID = "goai.expert_scale_selection.v2"
METRICS = (
    "fc_pcc",
    "context_residual_pcc",
    "drug_residual_pcc",
    "high_effect_pcc",
    "high_effect_f1",
    "absolute_sample_r2_median",
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _matrix_metadata(matrix_path: Path) -> dict[str, dict[str, Any]]:
    if not matrix_path.is_file():
        return {}
    payload = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    experiments = payload.get("experiments", payload.get("confirm_candidates", []))
    if not isinstance(experiments, list):
        return {}
    return {
        str(item["id"]): dict(item)
        for item in experiments
        if isinstance(item, dict) and "id" in item
    }


def _root_matrix(root: Path, explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit.resolve()
    environment = root / "environment.json"
    if not environment.is_file():
        return None
    payload = json.loads(environment.read_text(encoding="utf-8"))
    value = payload.get("matrix")
    return Path(value).resolve() if value else None


def _collect_summaries(
    roots: Iterable[Path],
    matrix_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for root in roots:
        current_matrix = _root_matrix(root, matrix_path)
        if current_matrix is not None:
            metadata.update(_matrix_metadata(current_matrix))
        for path in sorted((root / "producers").glob("*/S*/oof_summary.csv")):
            experiment_id = path.parents[1].name
            seed_label = path.parent.name
            seed = int(seed_label[1:] if seed_label.startswith("S") else seed_label)
            summary = pd.read_csv(path)
            for row in summary.to_dict(orient="records"):
                rows.append(
                    {
                        "protocol_label": PROTOCOL_LABEL,
                        "run_root": str(root),
                        "experiment_id": experiment_id,
                        "model_id": metadata.get(experiment_id, {}).get("model_id", ""),
                        "kind": metadata.get(experiment_id, {}).get("kind", ""),
                        "model_seed": seed,
                        "run_dir": str(path.parent),
                        **row,
                    }
                )
    return pd.DataFrame(rows), metadata


def _assignment_hash(path: Path, scenario: str) -> str:
    frame = pd.read_csv(path, keep_default_na=False)
    frame = frame.loc[frame["scenario"].eq(scenario)].copy()
    columns = [column for column in ("scenario", "fold", "sample_ID", "eligible", "exclusion_reason") if column in frame]
    frame = frame.loc[:, columns].sort_values([column for column in ("fold", "sample_ID") if column in frame])
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def _bootstrap_mean(values: np.ndarray, seed: int, draws: int = 10000) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = finite[rng.integers(0, len(finite), size=(draws, len(finite)))].mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def _paired_deltas(roots: Iterable[Path], control_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    for root in roots:
        for candidate_summary in sorted((root / "producers").glob("*/S*/oof_summary.csv")):
            candidate = candidate_summary.parent
            candidate_id = candidate.parent.name
            if candidate_id == control_id:
                continue
            control = root / "producers" / control_id / candidate.name
            candidate_metrics_path = candidate / "oof_official_proxy_metrics.csv"
            control_metrics_path = control / "oof_official_proxy_metrics.csv"
            if not candidate_metrics_path.is_file() or not control_metrics_path.is_file():
                continue
            candidate_metrics = pd.read_csv(candidate_metrics_path)
            control_metrics = pd.read_csv(control_metrics_path)
            common = sorted(
                (set(candidate_metrics["scenario"]) & set(control_metrics["scenario"]))
                & set(ALL_REPORT_SCENARIOS)
            )
            for scenario in common:
                candidate_assignments = candidate / "fold_assignments.csv"
                control_assignments = control / "fold_assignments.csv"
                assignment_hash = "unavailable"
                if candidate_assignments.is_file() and control_assignments.is_file():
                    candidate_hash = _assignment_hash(candidate_assignments, scenario)
                    control_hash = _assignment_hash(control_assignments, scenario)
                    if candidate_hash != control_hash:
                        raise ValueError(
                            f"Fold assignment mismatch: {candidate_id} vs {control_id}, {scenario}"
                        )
                    assignment_hash = candidate_hash
                left = control_metrics.loc[control_metrics["scenario"].eq(scenario)]
                right = candidate_metrics.loc[candidate_metrics["scenario"].eq(scenario)]
                paired = left.merge(
                    right,
                    on=["scenario", "fold"],
                    suffixes=("_control", "_candidate"),
                    validate="one_to_one",
                )
                for metric in METRICS:
                    control_column = f"{metric}_control"
                    candidate_column = f"{metric}_candidate"
                    if control_column not in paired or candidate_column not in paired:
                        continue
                    for row in paired.to_dict(orient="records"):
                        control_value = float(row[control_column])
                        candidate_value = float(row[candidate_column])
                        fold_rows.append(
                            {
                                "protocol_label": PROTOCOL_LABEL,
                                "run_root": str(root),
                                "candidate": candidate_id,
                                "control": control_id,
                                "model_seed": int(candidate.name[1:]),
                                "scenario": scenario,
                                "fold": int(row["fold"]),
                                "metric": metric,
                                "control_value": control_value,
                                "candidate_value": candidate_value,
                                "delta": candidate_value - control_value,
                                "fold_assignment_sha256": assignment_hash,
                            }
                        )
    folds = pd.DataFrame(fold_rows)
    aggregates: list[dict[str, Any]] = []
    if not folds.empty:
        keys = ["run_root", "candidate", "control", "model_seed", "scenario", "metric"]
        for group_key, group in folds.groupby(keys, sort=True, dropna=False):
            values = group["delta"].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            ci_low, ci_high = _bootstrap_mean(values, seed=8401)
            aggregates.append(
                {
                    "protocol_label": PROTOCOL_LABEL,
                    **dict(zip(keys, group_key)),
                    "n_paired_folds": int(len(finite)),
                    "mean_delta": float(finite.mean()) if len(finite) else float("nan"),
                    "std_delta": float(finite.std(ddof=1)) if len(finite) > 1 else float("nan"),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return folds, pd.DataFrame(aggregates)


def _four_regime_macro(leaderboard: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty:
        return pd.DataFrame()
    subset = leaderboard.loc[leaderboard["scenario"].isin(FOUR_REGIMES)].copy()
    records: list[dict[str, Any]] = []
    keys = ["run_root", "experiment_id", "model_id", "kind", "model_seed"]
    for group_key, group in subset.groupby(keys, sort=True, dropna=False):
        present = tuple(sorted(set(group["scenario"])))
        row: dict[str, Any] = {
            "protocol_label": PROTOCOL_LABEL,
            **dict(zip(keys, group_key)),
            "n_regimes": len(present),
            "complete_four_regimes": set(present) == set(FOUR_REGIMES),
            "regimes_present": ",".join(present),
        }
        for metric in METRICS:
            column = f"{metric}_mean"
            if column in group:
                values = pd.to_numeric(group[column], errors="coerce")
                row[f"four_regime_macro_{metric}"] = float(values.mean()) if values.notna().any() else float("nan")
                row[f"four_regime_n_{metric}"] = int(values.notna().sum())
        records.append(row)
    return pd.DataFrame(records)


def _select_expert_scales(
    leaderboard: pd.DataFrame,
    metadata: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select locked expert scales with a predeclared OOF-only rule.

    Strain and chemical scales use R10 and R01 respectively.  Pair scale uses
    the macro FC PCC over *both* R11 and RT; neither context nor drug residual
    may regress in either regime, and both high-effect PCC/F1 guardrails apply
    per regime.  Scale zero is a measured candidate, never an implicit
    fallback, and post-result regime cherry-picking is impossible.
    """

    rows: list[dict[str, Any]] = []
    for experiment_id, item in metadata.items():
        if item.get("kind") != "expert_scale_screen":
            continue
        axis = str(item["selection_axis"])
        value = float(item["selection_value"])
        scenarios = {
            "strain": ("R10",),
            "chemical": ("R01",),
            "pair": ("R11", "RT"),
        }[axis]
        match = leaderboard.loc[
            leaderboard["experiment_id"].eq(experiment_id)
            & leaderboard["scenario"].isin(scenarios)
            & leaderboard["model_seed"].eq(42)
        ]
        record: dict[str, Any] = {
            "protocol_label": PROTOCOL_LABEL,
            "axis": axis,
            "scale": value,
            "experiment_id": experiment_id,
            "scenario": "+".join(scenarios),
            "complete": (
                len(match) == len(scenarios)
                and set(match["scenario"].astype(str)) == set(scenarios)
            ),
            "metrics_finite": False,
        }
        if bool(record["complete"]):
            by_scenario = {
                str(source["scenario"]): source
                for _, source in match.iterrows()
            }
            residual_column = {
                "strain": "context_residual_pcc_mean",
                "chemical": "drug_residual_pcc_mean",
                "pair": "context_and_drug_residuals_per_regime",
            }[axis]
            record["relevant_residual_metric"] = residual_column
            finite_values: list[float] = []
            for scenario in scenarios:
                source = by_scenario[scenario]
                for column in (
                    "fc_pcc_mean",
                    "context_residual_pcc_mean",
                    "drug_residual_pcc_mean",
                    "high_effect_pcc_mean",
                    "high_effect_f1_mean",
                ):
                    value = float(source.get(column, float("nan")))
                    record[f"{scenario}_{column}"] = value
                    # Only the axis-relevant residual is required for the
                    # single-regime screens; pair requires both residuals.
                    if column in {
                        "fc_pcc_mean",
                        "high_effect_pcc_mean",
                        "high_effect_f1_mean",
                    } or axis == "pair" or column == residual_column:
                        finite_values.append(value)
            record["fc_pcc_macro_mean"] = float(
                np.mean([record[f"{scenario}_fc_pcc_mean"] for scenario in scenarios])
            )
            # Preserve these simple columns for downstream inspection of the
            # one-regime axes.  Pair decisions deliberately use macro fields.
            if axis != "pair":
                scenario = scenarios[0]
                for column in (
                    "fc_pcc_mean",
                    residual_column,
                    "high_effect_pcc_mean",
                    "high_effect_f1_mean",
                ):
                    record[column] = record[f"{scenario}_{column}"]
            record["metrics_finite"] = bool(np.isfinite(finite_values).all())
        rows.append(record)
    candidates = pd.DataFrame(rows)
    selected: dict[str, Any] = {
        "selection_status": "incomplete",
        "protocol_label": PROTOCOL_LABEL,
        "selection_rule_id": SCALE_SELECTION_RULE_ID,
        "method": (
            "Strain:R10 and chemical:R01 maximize FC PCC. Pair maximizes the "
            "predeclared R11+RT macro FC PCC. All relevant residual deltas vs "
            "scale 0 must be non-negative per required regime; high-effect "
            "PCC and F1 deltas must each be >=-0.005. Scales are fixed."
        ),
        "pair_selection_rule": {
            "objective": "macro_mean_fc_pcc",
            "regimes": ["R11", "RT"],
            "residual_guardrails": [
                "R11.context_residual_pcc",
                "R11.drug_residual_pcc",
                "RT.context_residual_pcc",
                "RT.drug_residual_pcc",
            ],
            "high_effect_guardrails": [
                "R11.high_effect_pcc",
                "R11.high_effect_f1",
                "RT.high_effect_pcc",
                "RT.high_effect_f1",
            ],
        },
        "allowed_scales": list(ALLOWED_SCALES),
        "selected": {"strain": None, "chemical": None, "pair": None},
    }
    if candidates.empty:
        return candidates, selected
    complete_all = True
    decisions: dict[str, dict[str, Any]] = {}
    for axis in ("strain", "chemical", "pair"):
        axis_rows = candidates.loc[candidates["axis"].eq(axis)].copy()
        usable = axis_rows["complete"] & axis_rows["metrics_finite"]
        observed = set(axis_rows.loc[usable, "scale"].astype(float))
        if observed != set(ALLOWED_SCALES) or len(axis_rows) != len(ALLOWED_SCALES):
            complete_all = False
            continue
        baseline = axis_rows.loc[axis_rows["scale"].eq(0.0)].iloc[0]
        residual_column = str(baseline["relevant_residual_metric"])
        objective = (
            "fc_pcc_macro_mean" if axis == "pair" else "fc_pcc_mean"
        )
        axis_rows["fc_delta_vs_zero"] = axis_rows[objective] - baseline[objective]
        if axis == "pair":
            residual_deltas: list[str] = []
            high_effect_deltas: list[str] = []
            for scenario in ("R11", "RT"):
                for metric in (
                    "context_residual_pcc_mean",
                    "drug_residual_pcc_mean",
                ):
                    source_column = f"{scenario}_{metric}"
                    delta_column = f"{source_column}_delta_vs_zero"
                    axis_rows[delta_column] = (
                        axis_rows[source_column] - baseline[source_column]
                    )
                    residual_deltas.append(delta_column)
                for metric in ("high_effect_pcc_mean", "high_effect_f1_mean"):
                    source_column = f"{scenario}_{metric}"
                    delta_column = f"{source_column}_delta_vs_zero"
                    axis_rows[delta_column] = (
                        axis_rows[source_column] - baseline[source_column]
                    )
                    high_effect_deltas.append(delta_column)
            axis_rows["min_relevant_residual_delta_vs_zero"] = axis_rows[
                residual_deltas
            ].min(axis=1)
            axis_rows["min_high_effect_delta_vs_zero"] = axis_rows[
                high_effect_deltas
            ].min(axis=1)
            axis_rows["relevant_residual_delta_vs_zero"] = axis_rows[
                "min_relevant_residual_delta_vs_zero"
            ]
            axis_rows["high_effect_pcc_delta_vs_zero"] = axis_rows[
                "min_high_effect_delta_vs_zero"
            ]
            axis_rows["guardrail_pass"] = (
                axis_rows[residual_deltas].ge(-1e-12).all(axis=1)
                & axis_rows[high_effect_deltas].ge(-0.005).all(axis=1)
            )
        else:
            axis_rows["relevant_residual_delta_vs_zero"] = (
                axis_rows[residual_column] - baseline[residual_column]
            )
            axis_rows["high_effect_pcc_delta_vs_zero"] = (
                axis_rows["high_effect_pcc_mean"]
                - baseline["high_effect_pcc_mean"]
            )
            axis_rows["high_effect_f1_delta_vs_zero"] = (
                axis_rows["high_effect_f1_mean"]
                - baseline["high_effect_f1_mean"]
            )
            axis_rows["guardrail_pass"] = (
                axis_rows["relevant_residual_delta_vs_zero"].ge(-1e-12)
                & axis_rows["high_effect_pcc_delta_vs_zero"].ge(-0.005)
                & axis_rows["high_effect_f1_delta_vs_zero"].ge(-0.005)
            )
        candidates.loc[axis_rows.index, axis_rows.columns] = axis_rows
        eligible = axis_rows.loc[axis_rows["guardrail_pass"]].copy()
        if eligible.empty:
            complete_all = False
            continue
        best = eligible.sort_values(
            [objective, "relevant_residual_delta_vs_zero", "high_effect_pcc_delta_vs_zero", "scale"],
            ascending=[False, False, False, True],
        ).iloc[0]
        decisions[axis] = {
            "scale": float(best["scale"]),
            "experiment_id": str(best["experiment_id"]),
            "scenario": str(best["scenario"]),
            "fc_pcc_mean": float(best[objective]),
            "objective_metric": objective,
            "fc_delta_vs_zero": float(best["fc_delta_vs_zero"]),
            "relevant_residual_metric": residual_column,
            "relevant_residual_delta_vs_zero": float(best["relevant_residual_delta_vs_zero"]),
            "high_effect_pcc_delta_vs_zero": float(best["high_effect_pcc_delta_vs_zero"]),
        }
        if axis == "pair":
            decisions[axis]["guardrail_regimes"] = ["R11", "RT"]
            decisions[axis]["per_regime"] = {
                scenario: {
                    metric: float(best[f"{scenario}_{metric}"])
                    for metric in (
                        "fc_pcc_mean",
                        "context_residual_pcc_mean",
                        "drug_residual_pcc_mean",
                        "high_effect_pcc_mean",
                        "high_effect_f1_mean",
                    )
                }
                for scenario in ("R11", "RT")
            }
    if complete_all and set(decisions) == {"strain", "chemical", "pair"}:
        selected["selection_status"] = "selected"
        selected["selected"] = decisions
    return candidates.sort_values(["axis", "scale"]), selected


def _render_report(
    leaderboard: pd.DataFrame,
    macros: pd.DataFrame,
    paired: pd.DataFrame,
    scale_selection: dict[str, Any],
) -> str:
    lines = [
        "# GOAI M7/M8 本地严格 OOF 汇总",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "> **这些结果全部是 LOCAL STRICT OOF，不是 GOAI 官方 PSS，也不是排行榜成绩。**",
        "",
        "## 四种实体状态总体",
        "",
    ]
    if macros.empty:
        lines.append("尚无完整结果。")
    else:
        ordered = macros.sort_values("four_regime_macro_fc_pcc", ascending=False, na_position="last")
        for row in ordered.head(20).to_dict(orient="records"):
            lines.append(
                f"- `{row['experiment_id']}-S{row['model_seed']}`："
                f"覆盖 {row['n_regimes']}/4，macro FC PCC="
                f"{row.get('four_regime_macro_fc_pcc', float('nan')):.6f}"
            )
    lines.extend(["", "## 分状态最佳", ""])
    for scenario in FOUR_REGIMES + ("RT",):
        group = leaderboard.loc[leaderboard["scenario"].eq(scenario)]
        if group.empty or "fc_pcc_mean" not in group:
            lines.append(f"- {scenario}：尚无结果")
            continue
        best = group.sort_values("fc_pcc_mean", ascending=False).iloc[0]
        lines.append(f"- {scenario}：`{best['experiment_id']}-S{best['model_seed']}`，FC PCC={best['fc_pcc_mean']:.6f}")
    lines.extend(["", "## 语义对照解释", ""])
    lines.append(
        "- 菌株语义的 zero 对照共享 `SCR-M7.3-ENTITIES`：它与 M8.1 使用相同实体专家结构，但 `strain_features=null`。"
    )
    lines.append(
        "- `R10` 是菌株已见/药物未知，因此看 context-residual；`R01` 是菌株未知/药物已见，因此看 drug-residual。"
    )
    lines.extend(["", "## Paired OOF 差值", ""])
    fc = paired.loc[paired["metric"].eq("fc_pcc")] if not paired.empty else paired
    if fc.empty:
        lines.append("尚无可配对的 control 结果。")
    else:
        for row in fc.sort_values(["scenario", "mean_delta"], ascending=[True, False]).head(30).to_dict(orient="records"):
            lines.append(
                f"- {row['scenario']} `{row['candidate']}-S{row['model_seed']}`："
                f"ΔFC={row['mean_delta']:+.6f}，95% fold-bootstrap "
                f"[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}]"
            )
    lines.extend(["", "## 固定专家缩放选择", ""])
    if scale_selection.get("selection_status") != "selected":
        lines.append("缩放网格尚未完整跑完；只生成候选表，不生成可用于确认实验的伪选择。")
    else:
        for axis, decision in scale_selection["selected"].items():
            lines.append(
                f"- {axis}：固定 scale={decision['scale']:.2f}，来源 "
                f"`{decision['experiment_id']}` / {decision['scenario']}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", required=True, help="Repeat to combine screen/confirm roots")
    parser.add_argument("--matrix", default=None, help="Optional matrix path when environment.json is absent")
    parser.add_argument("--control-id", default="SCR-M7.0-GENERAL")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    roots = [Path(value).resolve() for value in args.run_root]
    matrix_path = Path(args.matrix).resolve() if args.matrix else None
    output = Path(args.output_dir).resolve() if args.output_dir else roots[0] / "consumer" / "m7_m8"
    output.mkdir(parents=True, exist_ok=True)

    leaderboard, metadata = _collect_summaries(roots, matrix_path)
    if leaderboard.empty:
        raise RuntimeError("No completed M7/M8 oof_summary.csv files found")
    leaderboard.sort_values(["scenario", "fc_pcc_mean"], ascending=[True, False]).to_csv(
        output / "regime_summary.csv", index=False
    )
    macros = _four_regime_macro(leaderboard)
    macros.to_csv(output / "four_regime_macro.csv", index=False)
    fold_deltas, paired = _paired_deltas(roots, args.control_id)
    fold_deltas.to_csv(output / "paired_fold_deltas.csv", index=False)
    paired.to_csv(output / "paired_delta_summary.csv", index=False)

    scale_candidates, selection = _select_expert_scales(leaderboard, metadata)
    scale_candidates.to_csv(output / "expert_scale_candidates.csv", index=False)
    selection["generated_at"] = datetime.now().isoformat(timespec="seconds")
    selection["source_run_roots"] = [str(root) for root in roots]
    _atomic_text(
        output / "expert_scale_selection.yaml",
        yaml.safe_dump(selection, sort_keys=False, allow_unicode=True),
    )
    _atomic_text(
        output / "local_oof_report_zh.md",
        _render_report(leaderboard, macros, paired, selection),
    )
    print((output / "local_oof_report_zh.md").resolve())


if __name__ == "__main__":
    main()
