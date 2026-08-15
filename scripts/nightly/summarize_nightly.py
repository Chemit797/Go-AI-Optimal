"""Build a compact GOAI OOF leaderboard and Chinese morning report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


PRIMARY = [
    "fc_pcc_mean",
    "context_residual_pcc_mean",
    "drug_residual_pcc_mean",
    "high_effect_pcc_mean",
    "high_effect_f1_mean",
    "absolute_sample_r2_median_mean",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--control-id", default="N-CTRL-01")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    consumer = root / "consumer"
    consumer.mkdir(exist_ok=True)
    rows = []
    for path in sorted((root / "producers").glob("*/S*/oof_summary.csv")):
        experiment_id = path.parents[1].name
        seed_label = path.parent.name
        seed = int(seed_label[1:] if seed_label.startswith("S") else seed_label)
        table = pd.read_csv(path)
        for record in table.to_dict(orient="records"):
            rows.append({"experiment_id": experiment_id, "model_seed": seed, "run_dir": str(path.parent), **record})
    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        raise RuntimeError("no completed OOF summaries found")
    controls = leaderboard.loc[leaderboard["experiment_id"].eq(args.control_id)].copy()
    merged = leaderboard.merge(
        controls[["model_seed", "scenario", *[column for column in PRIMARY if column in controls]]],
        on=["model_seed", "scenario"], how="left", suffixes=("", "_control"), validate="many_to_one",
    )
    for column in PRIMARY:
        control = f"{column}_control"
        if column in merged and control in merged:
            merged[f"delta_{column}"] = merged[column] - merged[control]
    merged.sort_values(["scenario", "fc_pcc_mean"], ascending=[True, False]).to_csv(consumer / "scenario_leaderboard.csv", index=False)

    status_path = root / "batch_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    report = [
        "# GOAI 夜间 OOD 实验晨报",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "> 所有分数均为本地 OOF / official-proxy 指标，不是 GOAI 官方 PSS 或排行榜成绩。",
        "",
        "## 执行概况",
        "",
        f"- 完成任务：{len(status.get('completed', []))}",
        f"- 失败任务：{len(status.get('failed', []))}",
        f"- 跳过任务：{len(status.get('skipped', []))}",
        f"- 最终状态：{status.get('state', 'unknown')}",
        "",
        "## 各场景单种子最佳结果",
        "",
    ]
    for scenario, group in merged.groupby("scenario", sort=False):
        best = group.sort_values("fc_pcc_mean", ascending=False).iloc[0]
        delta = best.get("delta_fc_pcc_mean")
        report.extend([
            f"### {scenario}", "",
            f"- 最佳：`{best['experiment_id']}-S{best['model_seed']}`",
            f"- FC PCC：{best['fc_pcc_mean']:.6f}",
            f"- 相对同 seed control：{delta:+.6f}" if pd.notna(delta) else "- 相对 control：不可用",
            f"- high-effect PCC：{best.get('high_effect_pcc_mean', float('nan')):.6f}",
            f"- high-effect F1：{best.get('high_effect_f1_mean', float('nan')):.6f}",
            "",
        ])
    if status.get("failed"):
        report.extend(["## 失败任务", "", "```json", json.dumps(status["failed"], ensure_ascii=False, indent=2), "```", ""])
    (consumer / "morning_report_zh.md").write_text("\n".join(report), encoding="utf-8")
    receipt = {
        "competition": "GOAI virtual yeast cell track 3",
        "status": "blocked_no_official_endpoint",
        "reason": "No official GOAI submission endpoint, scorer credentials, or accepted submission command is available in the workspace.",
        "local_artifact": str((consumer / "scenario_leaderboard.csv").resolve()),
        "next_action": "Provide the official submission portal/API, credentials, sample submission contract, and scoring availability.",
    }
    (consumer / "submission_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print((consumer / "morning_report_zh.md").resolve())


if __name__ == "__main__":
    main()
