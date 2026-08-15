"""Score a GOAI prediction CSV against organizer-held test truth.

The public handbook specifies scoring modules and their weights, but not the
final within-module aggregation.  This module therefore reports the published
components exactly where possible and labels its transparent weighted
combination as a proxy rather than an official leaderboard score.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .official_metrics import (
    _frozen_delta_references,
    absolute_fidelity,
    response_metrics,
)
from .preprocess import PreprocessedData
from .schema import REQUIRED_METADATA_COLUMNS, SAMPLE_ID, SPLIT


DEFAULT_METADATA_TEST = "WAYB_WAYC_metadata_test.csv"
DEFAULT_METADATA_TRAIN = "WAYB_WAYC_metadata_train_val.csv"
DEFAULT_PROTEOME_TRAIN = "WAYB_WAYC_proteome_raw_train_val.csv"
DEFAULT_REFERENCES = "goai_scoring_references.npz"
TEST_SPLITS = ("test_chem_only", "test_strain_only", "test_both", "test_time")


@dataclass(frozen=True)
class CsvInputs:
    truth: pd.DataFrame
    prediction: pd.DataFrame
    proteins: list[str]
    truth_scale: str
    ignored_truth_columns: list[str]
    prediction_order_matched_truth: bool


def _read_csv(path: str | Path, label: str) -> pd.DataFrame:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise ValueError(f"{label}不存在: {csv_path}")
    try:
        frame = pd.read_csv(csv_path, low_memory=False)
    except Exception as error:
        raise ValueError(f"无法读取{label}: {csv_path}\n{error}") from error
    if frame.empty:
        raise ValueError(f"{label}为空: {csv_path}")
    if SAMPLE_ID not in frame.columns:
        raise ValueError(f"{label}缺少必需列 {SAMPLE_ID!r}")
    if frame[SAMPLE_ID].isna().any():
        raise ValueError(f"{label}的 {SAMPLE_ID} 存在缺失值")
    duplicates = frame.loc[frame[SAMPLE_ID].duplicated(), SAMPLE_ID].head(5).tolist()
    if duplicates:
        raise ValueError(f"{label}的 {SAMPLE_ID} 存在重复，例如: {duplicates}")
    return frame


def _numeric_values(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    try:
        numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}的蛋白预测列必须全部为数值") from error
    values = numeric.to_numpy(dtype=np.float64, copy=False)
    if np.isinf(values).any():
        raise ValueError(f"{label}包含正无穷或负无穷")
    return numeric


def _detect_truth_scale(values: np.ndarray) -> str:
    observed = values[np.isfinite(values)]
    if observed.size == 0:
        raise ValueError("标准答案没有任何可评分数值")
    # GOAI raw intensities are typically 1e3--1e10, while log2 values are
    # typically below 40.  A conservative threshold leaves a wide gap.
    return "raw" if float(np.nanmedian(observed)) > 100.0 else "log2"


def load_csv_inputs(
    truth_path: str | Path,
    prediction_path: str | Path,
    truth_scale: str = "auto",
) -> CsvInputs:
    """Load, validate, align, and scale the two required CSV files."""
    answer = _read_csv(truth_path, "标准答案 CSV")
    submitted = _read_csv(prediction_path, "预测 CSV")

    proteins = [str(column) for column in submitted.columns if column != SAMPLE_ID]
    if not proteins:
        raise ValueError("预测 CSV 没有蛋白预测列")
    if len(proteins) != len(set(proteins)):
        raise ValueError("预测 CSV 存在重复列名")
    metadata_columns = set(REQUIRED_METADATA_COLUMNS) - {SAMPLE_ID}
    answer_proteins = [str(column) for column in answer.columns if column != SAMPLE_ID and column not in metadata_columns]
    missing = [column for column in answer_proteins if column not in proteins]
    extra = [column for column in proteins if column not in answer_proteins]
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"预测缺少 {len(missing)} 个标准答案蛋白列，例如 {missing[:5]}")
        if extra:
            details.append(f"预测多出 {len(extra)} 个非答案列，例如 {extra[:5]}")
        raise ValueError("；".join(details))

    answer_ids = answer[SAMPLE_ID].astype(str)
    submitted_ids = submitted[SAMPLE_ID].astype(str)
    missing_ids = sorted(set(answer_ids) - set(submitted_ids))
    extra_ids = sorted(set(submitted_ids) - set(answer_ids))
    if missing_ids or extra_ids:
        details: list[str] = []
        if missing_ids:
            details.append(f"预测缺少 {len(missing_ids)} 个 sample_ID，例如 {missing_ids[:5]}")
        if extra_ids:
            details.append(f"预测多出 {len(extra_ids)} 个 sample_ID，例如 {extra_ids[:5]}")
        raise ValueError("；".join(details))

    order_matched = submitted_ids.tolist() == answer_ids.tolist()
    answer = answer.assign(**{SAMPLE_ID: answer_ids}).set_index(SAMPLE_ID)
    submitted = submitted.assign(**{SAMPLE_ID: submitted_ids}).set_index(SAMPLE_ID)
    submitted = submitted.reindex(answer.index)

    truth_numeric = _numeric_values(answer, proteins, "标准答案")
    prediction_numeric = _numeric_values(submitted, proteins, "预测 CSV")
    prediction_values = prediction_numeric.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(prediction_values).all():
        raise ValueError("预测 CSV 不能包含 NaN 或无穷值")

    resolved_scale = _detect_truth_scale(truth_numeric.to_numpy(dtype=np.float64, copy=False)) if truth_scale == "auto" else truth_scale
    if resolved_scale not in {"raw", "log2"}:
        raise ValueError("truth_scale 必须是 auto、raw 或 log2")
    if resolved_scale == "raw":
        truth_values = truth_numeric.to_numpy(dtype=np.float64, copy=True)
        invalid = np.isfinite(truth_values) & (truth_values <= 0.0)
        if invalid.any():
            raise ValueError("raw 标准答案包含非正数，无法进行 log2 转换")
        truth_numeric = pd.DataFrame(
            np.log2(truth_values),
            index=truth_numeric.index,
            columns=truth_numeric.columns,
        )

    ignored = [str(column) for column in answer.columns if column in metadata_columns]
    return CsvInputs(
        truth=truth_numeric.astype(np.float64),
        prediction=prediction_numeric.astype(np.float64),
        proteins=proteins,
        truth_scale=resolved_scale,
        ignored_truth_columns=ignored,
        prediction_order_matched_truth=order_matched,
    )


def _candidate_roots(truth_path: Path, prediction_path: Path) -> list[Path]:
    roots = [truth_path.parent, prediction_path.parent, Path.cwd(), Path(__file__).resolve().parents[2]]
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _resolve_support_file(explicit: str | Path | None, filename: str, roots: list[Path]) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"辅助文件不存在: {path}")
        return path
    for root in roots:
        candidate = root / filename
        if candidate.is_file():
            return candidate
    return None


def _metadata_from_answer(path: str | Path) -> pd.DataFrame | None:
    header = pd.read_csv(Path(path).expanduser(), nrows=0)
    if not set(REQUIRED_METADATA_COLUMNS).issubset(header.columns):
        return None
    metadata = pd.read_csv(Path(path).expanduser(), usecols=list(REQUIRED_METADATA_COLUMNS), low_memory=False)
    metadata[SAMPLE_ID] = metadata[SAMPLE_ID].astype(str)
    return metadata.set_index(SAMPLE_ID)


def _load_frozen_references(path: Path, proteins: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            stored_proteins = payload["proteins"].astype(str).tolist()
            if stored_proteins != proteins:
                raise ValueError("冻结评分参照的蛋白列与 prediction.csv 不一致")
            context_keys = payload["context_keys"].astype(str)
            context_values = payload["context_values"].astype(np.float64)
            drug_keys = payload["drug_keys"].astype(str)
            drug_values = payload["drug_values"].astype(np.float64)
    except (KeyError, OSError, ValueError) as error:
        if isinstance(error, ValueError) and "蛋白列" in str(error):
            raise
        raise ValueError(f"无法读取冻结评分参照: {path}") from error
    if context_keys.ndim != 2 or context_keys.shape[1] != 8:
        raise ValueError("冻结评分参照的 context key 形状错误")
    if context_values.shape != (len(context_keys), len(proteins)):
        raise ValueError("冻结评分参照的 context 数值形状错误")
    if drug_values.shape != (len(drug_keys), len(proteins)):
        raise ValueError("冻结评分参照的 drug 数值形状错误")
    context_index = pd.MultiIndex.from_arrays(context_keys.T)
    context = pd.DataFrame(context_values, index=context_index, columns=proteins)
    drug = pd.DataFrame(drug_values, index=pd.Index(drug_keys), columns=proteins)
    return context, drug


def _load_full_data(
    inputs: CsvInputs,
    truth_path: str | Path,
    prediction_path: str | Path,
    metadata_test_path: str | Path | None,
    metadata_train_path: str | Path | None,
    proteome_train_path: str | Path | None,
    references_path: str | Path | None,
) -> tuple[PreprocessedData, pd.Index, dict[str, str], pd.DataFrame, pd.DataFrame] | None:
    roots = _candidate_roots(Path(truth_path).expanduser().resolve(), Path(prediction_path).expanduser().resolve())
    test_path = _resolve_support_file(metadata_test_path, DEFAULT_METADATA_TEST, roots)
    frozen_path = (
        _resolve_support_file(references_path, DEFAULT_REFERENCES, roots)
        if references_path is not None or (metadata_train_path is None and proteome_train_path is None)
        else None
    )

    test_metadata = _metadata_from_answer(truth_path)
    if test_metadata is None and test_path is not None:
        test_metadata = _read_csv(test_path, "test metadata").set_index(SAMPLE_ID)
        test_metadata.index = test_metadata.index.astype(str)
    if test_metadata is None:
        return None
    missing_metadata = [
        column for column in REQUIRED_METADATA_COLUMNS if column != SAMPLE_ID and column not in test_metadata.columns
    ]
    if missing_metadata:
        raise ValueError(f"test metadata 缺少字段: {missing_metadata}")
    if set(test_metadata.index) != set(inputs.truth.index):
        if metadata_test_path is None and _metadata_from_answer(truth_path) is None:
            return None
        raise ValueError("test metadata 与标准答案的 sample_ID 集合不同")
    test_metadata = test_metadata.reindex(inputs.truth.index)

    if frozen_path is not None:
        context_reference, drug_reference = _load_frozen_references(frozen_path, inputs.proteins)
        data = PreprocessedData(
            metadata=test_metadata,
            y_log2=inputs.truth,
            mask=inputs.truth.notna(),
            proteins=inputs.proteins,
            train_ids=pd.Index([], dtype=object),
            missing_rate=pd.Series(dtype=np.float64),
        )
        paths = {
            "metadata_test": str(test_path) if test_path else "embedded_in_truth_csv",
            "frozen_references": str(frozen_path),
        }
        return data, test_metadata.index, paths, context_reference, drug_reference

    train_meta_path = _resolve_support_file(metadata_train_path, DEFAULT_METADATA_TRAIN, roots)
    train_proteome_path = _resolve_support_file(proteome_train_path, DEFAULT_PROTEOME_TRAIN, roots)
    if train_meta_path is None or train_proteome_path is None:
        return None

    train_metadata = _read_csv(train_meta_path, "train/val metadata").set_index(SAMPLE_ID)
    train_metadata.index = train_metadata.index.astype(str)
    missing_train_metadata = [
        column for column in REQUIRED_METADATA_COLUMNS if column != SAMPLE_ID and column not in train_metadata.columns
    ]
    if missing_train_metadata:
        raise ValueError(f"train/val metadata 缺少字段: {missing_train_metadata}")
    if set(train_metadata.index) & set(test_metadata.index):
        raise ValueError("train/val metadata 与 test metadata 的 sample_ID 不应重叠")

    usecols = [SAMPLE_ID, *inputs.proteins]
    try:
        raw_train = pd.read_csv(train_proteome_path, usecols=usecols, low_memory=False)
    except ValueError as error:
        raise ValueError("train/val 蛋白矩阵缺少预测所需的蛋白列") from error
    if raw_train[SAMPLE_ID].isna().any() or raw_train[SAMPLE_ID].duplicated().any():
        raise ValueError("train/val 蛋白矩阵的 sample_ID 必须非空且唯一")
    raw_train[SAMPLE_ID] = raw_train[SAMPLE_ID].astype(str)
    raw_train = raw_train.set_index(SAMPLE_ID)
    if set(raw_train.index) != set(train_metadata.index):
        raise ValueError("train/val metadata 与蛋白矩阵的 sample_ID 集合不同")
    raw_train = raw_train.reindex(train_metadata.index).apply(pd.to_numeric, errors="raise")
    raw_values = raw_train.to_numpy(dtype=np.float64, copy=False)
    if (np.isfinite(raw_values) & (raw_values <= 0.0)).any() or np.isinf(raw_values).any():
        raise ValueError("train/val raw 蛋白矩阵包含非正数或无穷值")
    train_log2 = np.log2(raw_train.astype(np.float64))

    metadata = pd.concat([train_metadata, test_metadata], axis=0)
    y_log2 = pd.concat([train_log2, inputs.truth], axis=0)
    train_ids = train_metadata.index[train_metadata[SPLIT].eq("train")]
    if train_ids.empty:
        raise ValueError("train/val metadata 中没有 split_final == 'train' 的样本")
    data = PreprocessedData(
        metadata=metadata,
        y_log2=y_log2,
        mask=y_log2.notna(),
        proteins=inputs.proteins,
        train_ids=train_ids,
        missing_rate=train_log2.loc[train_ids].isna().mean(axis=0),
    )
    paths = {
        "metadata_test": str(test_path) if test_path else "embedded_in_truth_csv",
        "metadata_train_val": str(train_meta_path),
        "proteome_train_val": str(train_proteome_path),
    }
    context_reference, drug_reference = _frozen_delta_references(data)
    return data, test_metadata.index, paths, context_reference, drug_reference


def _mean_finite(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if len(finite) else float("nan")


def _absolute_component(metrics: dict[str, Any]) -> float:
    return _mean_finite(
        [
            float(metrics["absolute_sample_pcc_median"]),
            float(metrics["absolute_sample_r2_median"]),
            float(metrics["absolute_protein_pcc_median"]),
            float(metrics["absolute_protein_r2_median"]),
        ]
    )


def _transparent_weighted_proxy(overall: dict[str, Any], by_split: list[dict[str, Any]]) -> dict[str, Any]:
    split_map = {str(row["split"]): row for row in by_split}
    absolute = _absolute_component(overall)
    fc = float(overall["fc_pcc"])
    context = float(split_map.get("test_chem_only", {}).get("context_residual_pcc", np.nan))
    drug = float(split_map.get("test_strain_only", {}).get("drug_residual_pcc", np.nan))
    extrapolation_values: list[float] = []
    for split in ("test_both", "test_time"):
        row = split_map.get(split)
        if row:
            extrapolation_values.append(_mean_finite([_absolute_component(row), float(row["fc_pcc"])]))
    extrapolation = _mean_finite(extrapolation_values)
    high_effect = _mean_finite(
        [
            float(overall["high_effect_direction_accuracy"]),
            float(overall["high_effect_pcc"]),
            float(overall["high_effect_f1"]),
        ]
    )
    components = {
        "absolute_fidelity_20pct": absolute,
        "matched_control_fc_25pct": fc,
        "context_residual_20pct": context,
        "drug_residual_20pct": drug,
        "double_unknown_time_10pct": extrapolation,
        "high_effect_5pct": high_effect,
    }
    weights = [0.20, 0.25, 0.20, 0.20, 0.10, 0.05]
    values = np.asarray(list(components.values()), dtype=np.float64)
    if not np.isfinite(values).all():
        score = float("nan")
    else:
        score = float(np.dot(values, np.asarray(weights)) * 100.0)
    return {
        "score": score,
        "components": components,
        "formula": "100 * (0.20*absolute + 0.25*fc + 0.20*context + 0.20*drug + 0.10*double_time + 0.05*high_effect)",
        "status": "TRANSPARENT_LOCAL_PROXY_NOT_OFFICIAL_FINAL_SCORE",
        "warning": "手册未规定模块内部聚合、归一化和全部边界处理；此总分仅用于本地检查。",
    }


def score_files(
    truth_path: str | Path,
    prediction_path: str | Path,
    *,
    truth_scale: str = "auto",
    metadata_test_path: str | Path | None = None,
    metadata_train_path: str | Path | None = None,
    proteome_train_path: str | Path | None = None,
    references_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return absolute metrics and, when support files exist, all public modules."""
    inputs = load_csv_inputs(truth_path, prediction_path, truth_scale)
    absolute = absolute_fidelity(inputs.prediction, inputs.truth)
    report: dict[str, Any] = {
        "status": "ok",
        "truth_csv": str(Path(truth_path).expanduser().resolve()),
        "prediction_csv": str(Path(prediction_path).expanduser().resolve()),
        "truth_input_scale": inputs.truth_scale,
        "scoring_scale": "log2",
        "n_proteins": len(inputs.proteins),
        "prediction_order_matched_truth": inputs.prediction_order_matched_truth,
        "ignored_truth_columns": inputs.ignored_truth_columns,
        "absolute_fidelity": absolute,
    }

    full = _load_full_data(
        inputs,
        truth_path,
        prediction_path,
        metadata_test_path,
        metadata_train_path,
        proteome_train_path,
        references_path,
    )
    if full is None:
        report["published_modules_status"] = "support_files_not_found"
        report["official_final_score"] = None
        report["note"] = (
            "仅凭答案与预测可计算绝对保真度；FC、残差和高效应模块还需要 test metadata "
            "以及冻结评分参照，或 train/val metadata 和 raw 蛋白矩阵。"
        )
        return report

    data, test_ids, support_paths, context_reference, drug_reference = full
    overall_response = response_metrics(
        data,
        inputs.prediction,
        test_ids,
        context_reference,
        drug_reference,
        control_pool_ids=test_ids,
    )
    overall = {**absolute, **overall_response}
    rows: list[dict[str, Any]] = []
    for split in TEST_SPLITS:
        ids = test_ids[data.metadata.loc[test_ids, SPLIT].eq(split).to_numpy()]
        if len(ids) == 0:
            continue
        split_prediction = inputs.prediction.reindex(ids)
        split_absolute = absolute_fidelity(split_prediction, inputs.truth.reindex(ids))
        split_response = response_metrics(
            data,
            split_prediction,
            ids,
            context_reference,
            drug_reference,
            control_pool_ids=test_ids,
        )
        rows.append({"split": split, **split_absolute, **split_response})

    report["published_modules_status"] = "computed_with_public_handbook_proxy"
    report["support_files"] = support_paths
    report["overall_metrics"] = overall
    report["split_metrics"] = rows
    report["weighted_proxy"] = _transparent_weighted_proxy(overall, rows)
    report["official_final_score"] = None
    report["note"] = "官方未发布完整最终聚合细节，因此不能从公开材料声称官方最终分数。"
    return report


def _prompt_path(label: str) -> str:
    value = input(label).strip().strip('"').strip("'")
    if not value:
        raise ValueError("文件地址不能为空")
    return value


def _print_report(report: dict[str, Any]) -> None:
    absolute = report["absolute_fidelity"]
    print("\n=== GOAI CSV 评分结果 ===")
    print(f"样本数: {absolute['absolute_n_samples']:,}")
    print(f"蛋白数: {report['n_proteins']:,}")
    print(f"有效真值数: {absolute['absolute_n_observed_values']:,}")
    print(f"标准答案输入尺度: {report['truth_input_scale']}（评分统一使用 log2）")
    print(f"逐样本 PCC 中位数: {absolute['absolute_sample_pcc_median']:.8f}")
    print(f"逐样本 R2 中位数:  {absolute['absolute_sample_r2_median']:.8f}")
    print(f"逐蛋白 PCC 中位数: {absolute['absolute_protein_pcc_median']:.8f}")
    print(f"逐蛋白 R2 中位数:  {absolute['absolute_protein_r2_median']:.8f}")

    if "weighted_proxy" in report:
        print("\n--- 按公开手册模块 ---")
        for row in report["split_metrics"]:
            print(
                f"{row['split']}: FC PCC={row['fc_pcc']:.8f}, "
                f"context residual={row['context_residual_pcc']:.8f}, "
                f"drug residual={row['drug_residual_pcc']:.8f}, "
                f"high-effect F1={row['high_effect_f1']:.8f}"
            )
        proxy = report["weighted_proxy"]
        print(f"\n公开权重代理总分: {proxy['score']:.8f}")
        print(f"状态: {proxy['status']}")
        print(f"注意: {proxy['warning']}")
    else:
        print(f"\n注意: {report['note']}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="比较 GOAI 标准答案 CSV 与 prediction.csv")
    parser.add_argument("truth_csv", nargs="?", help="组委会标准答案 CSV 的本地地址")
    parser.add_argument("prediction_csv", nargs="?", help="参赛者 prediction.csv 的本地地址")
    parser.add_argument("--truth-scale", choices=("auto", "raw", "log2"), default="auto")
    parser.add_argument("--metadata-test", default=None, help="可选；test metadata CSV")
    parser.add_argument("--metadata-train-val", default=None, help="可选；train/val metadata CSV")
    parser.add_argument("--proteome-train-val", default=None, help="可选；train/val raw proteome CSV")
    parser.add_argument("--references", default=None, help="可选；预先冻结的评分参照 NPZ")
    parser.add_argument("--output", default=None, help="可选；保存完整 JSON 报告的地址")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        truth_csv = args.truth_csv or _prompt_path("请输入标准答案 test.csv 的本地地址: ")
        prediction_csv = args.prediction_csv or _prompt_path("请输入 prediction.csv 的本地地址: ")
        report = score_files(
            truth_csv,
            prediction_csv,
            truth_scale=args.truth_scale,
            metadata_test_path=args.metadata_test,
            metadata_train_path=args.metadata_train_val,
            proteome_train_path=args.proteome_train_val,
            references_path=args.references,
        )
        _print_report(report)
        if args.output:
            output = Path(args.output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as handle:
                json.dump(_json_safe(report), handle, ensure_ascii=False, indent=2)
            print(f"\n完整报告已保存: {output}")
    except (OSError, ValueError) as error:
        raise SystemExit(f"评分失败: {error}") from error


if __name__ == "__main__":
    main()
