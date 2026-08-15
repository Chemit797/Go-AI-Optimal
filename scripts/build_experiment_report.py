"""Build a Chinese experiment worksheet from one completed run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SCENARIO_LABELS = {
    "val_chem_only": "S1：新化合物",
    "val_strain_only": "S2：新菌株",
    "val_both": "S3：新菌株 + 新化合物",
    "val_time": "时间外推",
}

ARCHITECTURES = {
    "p0_onehot": "P0 条件 MLP：把菌株、化合物、培养基、温度和时间编码为类别特征，经两层神经网络直接预测全部蛋白。未见实体没有可迁移语义。",
    "p1_priors": "P1 统计先验 MLP：在 P0 基础上加入训练数据中的菌株平均蛋白图谱和化合物平均响应，帮助模型获得稳定背景信息。",
    "p1_oof_priors": "P1-OOF 统计先验 MLP：与 P1 相同，但训练每一行的统计先验不包含该行自身，避免模型从特征中直接读到自身标签。",
    "p2_crosses": "P2 交叉特征 MLP：在 P1 基础上显式加入菌株-培养基、化合物-温度组合特征。",
    "p3_time": "P3 时间编码 MLP：在 P2 基础上加入时间的 sin/cos 连续编码。",
    "p4_hash": "P4 名称散列 MLP：在 P3 基础上加入化合物名称的确定性数值编码；它不代表真实化学结构。",
    "v2_response_decomposition": "背景加响应模型：背景分支不输入化合物，响应分支输入完整条件；两者相加得到蛋白质组，并对响应分支施加匹配对照 FC 监督。",
}


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _score_path(run_dir: Path) -> Path | None:
    for name in ("official_proxy_metrics.csv", "official_proxy_rescore.csv"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    return None


def _format(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "不适用/无法定义"
    return f"{float(value):.{digits}f}"


def _comparison(current: float, previous: float) -> str:
    if pd.isna(current) or pd.isna(previous):
        return "无法比较"
    change = current - previous
    if abs(change) < 0.002:
        return "基本持平"
    return f"{'提高' if change > 0 else '降低'} {abs(change):.3f}"


def _metric_interpretation(row: pd.Series, previous: pd.Series | None) -> str:
    absolute = row.get("absolute_sample_r2_median")
    fc = row.get("fc_pcc")
    fragments: list[str] = []
    if pd.notna(absolute):
        if absolute < 0:
            fragments.append("绝对蛋白丰度预测差于用该场景真值均值作常数预测，说明该场景尚未可靠")
        elif absolute < 0.5:
            fragments.append("绝对蛋白丰度只捕捉到有限信号")
        else:
            fragments.append("绝对蛋白丰度存在可用的一致性，但仍需与简单参照比较")
    if pd.notna(fc):
        if fc < 0.05:
            fragments.append("扰动变化（FC）几乎没有可用相关性")
        elif fc < 0.20:
            fragments.append("扰动变化只捕捉到较弱信号，不能单独支持机制结论")
        else:
            fragments.append("扰动变化存在初步相关性，仍需检查高效应蛋白和拆分场景")
    if previous is not None:
        fragments.append(
            f"相对比较运行：绝对 R2 {_comparison(absolute, previous.get('absolute_sample_r2_median'))}，"
            f"FC PCC {_comparison(fc, previous.get('fc_pcc'))}"
        )
    return "；".join(fragments) + "。"


def build_report(run_dir: Path, compare_run: Path | None = None, label: str | None = None) -> str:
    manifest = _read_json(run_dir / "manifest.json")
    variant = str(manifest.get("variant", run_dir.name))
    score_path = _score_path(run_dir)
    metrics_path = run_dir / "metrics.csv"
    scores = pd.read_csv(score_path) if score_path else pd.DataFrame()
    metrics = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
    comparison_scores = pd.read_csv(_score_path(compare_run)) if compare_run and _score_path(compare_run) else pd.DataFrame()
    comparison_by_split = comparison_scores.set_index("split") if not comparison_scores.empty else pd.DataFrame()

    title = label or variant
    lines = [
        f"# 实验记录：{title}",
        "",
        "> 本文件由 `scripts/build_experiment_report.py` 自动生成。数值来自本次运行；标有“请填写”的内容由研究者补充。",
        "",
        "## 一、这次跑了什么",
        "",
        f"- 运行目录：`{run_dir}`",
        f"- 模型版本：`{variant}`",
        f"- 架构说明：{ARCHITECTURES.get(variant, '请在此补充该版本的模型结构和与父版本的差异。')}",
        f"- 随机种子：`{manifest.get('config', {}).get('model', {}).get('seed', '请填写')}`",
        f"- 训练轮数：`{manifest.get('config', {}).get('model', {}).get('epochs', '请填写')}`",
        "- 本次唯一想验证的假设：**请填写。例：去除训练行自身统计信息后，外部验证是否仍然稳定？**",
        "",
        "## 二、先看结果：自动填写",
        "",
        "### 指标速查",
        "",
        "- `abs R2`：绝对蛋白丰度是否接近真实值。越接近 1 越好；小于 0 表示该场景预测不可靠。",
        "- `FC PCC`：处理相对于匹配对照的变化方向和强弱是否一致。越接近 1 越好；接近 0 表示几乎没学到扰动变化。",
        "- `S1/S2/S3`：依次是新化合物、新菌株、二者都新；S3 最严格。",
        "",
    ]
    if scores.empty:
        lines.extend(["尚未找到官方评分代理 CSV。请先运行 `goai_baseline.score`，再重新生成本报告。", ""])
    else:
        lines.extend(["| 场景 | abs R2 | FC PCC | 高效应蛋白方向准确率 | 自动解读 |", "|---|---:|---:|---:|---|"])
        for _, row in scores.iterrows():
            previous = comparison_by_split.loc[row["split"]] if row["split"] in comparison_by_split.index else None
            lines.append(
                f"| {SCENARIO_LABELS.get(row['split'], row['split'])} | {_format(row.get('absolute_sample_r2_median'))} | "
                f"{_format(row.get('fc_pcc'))} | {_format(row.get('high_effect_direction_accuracy'))} | {_metric_interpretation(row, previous)} |"
            )
        lines.append("")
    if not metrics.empty:
        lines.extend(["### 与原始训练评估交叉核对", "", "| 场景 | log2 RMSE | 全局 R2 | 逐蛋白 R2 中位数 |", "|---|---:|---:|---:|"])
        for _, row in metrics.loc[metrics["subset"].eq("all_rows")].iterrows():
            lines.append(f"| {SCENARIO_LABELS.get(row['split'], row['split'])} | {_format(row.get('log2_rmse'))} | {_format(row.get('global_r2'))} | {_format(row.get('protein_r2_median'))} |")
        lines.append("")
    lines.extend([
        "## 三、生物学填写区：请填写",
        "",
        "| 问题 | 填写提示 | 本次填写 |",
        "|---|---|---|",
        "| 这次最重要的现象是什么？ | 只写一个数值事实，例如“S3 FC PCC 低于 B0”。 | 请填写 |",
        "| 可能的生物学解释是什么？ | 用“可能/假设”措辞；不要把模型相关性写成因果。 | 请填写 |",
        "| 哪些条件最值得单独查看？ | 可按菌株、化合物、培养基、温度、时间选择。 | 请填写 |",
        "| 是否出现高效应蛋白方向错误？ | 查看方向准确率和后续蛋白列表，不要只看总分。 | 请填写 |",
        "| 下一步该保留、回退还是继续？ | 只有同时改善目标场景和目标指标，才进入下一步。 | 请填写 |",
        "",
        "## 四、决策规则",
        "",
        "1. 若绝对 R2 和 FC PCC 都低于比较运行：回退，不继续堆叠模块。",
        "2. 若绝对 R2 提高但 FC PCC 不提高：它主要改善背景丰度，不能称为扰动响应提升。",
        "3. 若 FC PCC 提高但绝对 R2 明显下降：检查是否损害了完整蛋白质组预测；不要只挑有利指标。",
        "4. 只有在目标 OOD 场景中稳定改善，并且能在不同随机种子重复时，才进入下一版本。",
        "",
        "## 五、运行与复现信息",
        "",
        "- 配置文件：请填写本次使用的 YAML 路径。",
        "- 比较运行：" + (f"`{compare_run}`" if compare_run else "未提供，请填写。"),
        "- 外部资源：本版本未自动判断，请填写“无”或列出来源、版本和用途。",
        "- 结论状态：请填写（进入主线 / 保留探索 / 回退）。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Chinese experiment worksheet from a completed run")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--compare-run", type=Path, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.run_dir / "experiment_report.md"
    output.write_text(build_report(args.run_dir, args.compare_run, args.label), encoding="utf-8")
    print(f"Wrote experiment worksheet: {output.resolve()}")


if __name__ == "__main__":
    main()
