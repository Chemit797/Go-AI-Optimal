# Go-AI-Optimal: GOAI Virtual Yeast Cell M12.0

[中文](#中文说明) | [English](#english)

This repository is the reproducible release of **GOAI-M12.0**, our current
best local candidate for the GOAI virtual yeast-cell perturbation challenge.
For the layer-by-layer model explanation, read
[`description.md`](description.md) first.

## 中文说明

### 1. 这是什么

本仓库整理了 GOAI 虚拟酵母扰动响应任务当前最终候选 `GOAI-M12.0` 的：

- 完整训练、OOF、评估、路由和推理源码；
- 完整最终推理权重，共 15 个下游 checkpoint 和 1 个 OP3 预训练编码器；
- 可从 checkpoint 直接重建最终预测的单命令入口；
- 中英双语复现说明、模型架构文档、结果报告和权重哈希清单；
- 经过清理的化合物结构映射与逐行 support 路由清单。

仓库**不包含** GOAI 官方原始数据、蛋白标签、OOF 预测、test 预测或
`prediction.csv`。这些内容不应通过 GitHub 再分发。

`GOAI-M12.0` 不是一个单独的神经网络，而是一个经过 OOD 路由的组合系统：

```text
M2.0 / M2.31  --------> M5.1 fallback
M6.11 / M6.21 --------> M5.2 biological backbone
M9.6 OP3 transfer ----> unseen-chemical response specialist
                              |
canonical support router -----+----> GOAI-M12.0
```

详细的数据构造、输入维度、每一层、损失函数、融合公式和路由规则见
[`description.md`](description.md)。

### 2. 当前结果

以下均为**本地严格 chemical-held-out OOF proxy**，不是官方 PSS 或排行榜成绩：

| 指标 | GOAI-M12.0 |
|---|---:|
| FC PCC | **0.426342** |
| Context residual PCC | 0.060967 |
| High-effect PCC | **0.603184** |
| High-effect F1 | **0.233970** |
| Absolute sample R2 median | 0.979150 |
| Nested outer FC PCC | 0.425571 |

M12.0 相对 M11.0 的 FC 提升为 `+0.000203`，按 37 个 held-out chemical 做
paired bootstrap 的 95% CI 为 `[+0.000059, +0.000355]`；High-effect PCC
提高 `+0.002314`，但 Context residual PCC 下降 `-0.001277`。因此它是按当前
FC 主目标选择的候选，不声称所有指标同时上升。

### 3. 最终逐行路由

路由不直接相信 `split_final`，而是根据全量 refit 实际见过的 canonical strain、
chemical 和 pair 逐行重算：

| 路由 | 含义 | 行数 | 最终路径 |
|---|---|---:|---|
| R10 | 菌株已见，化合物未知 | 2,072 | M6.11 background/calibration + M6/M9.6 response fusion |
| R01 | 菌株未知，化合物已见 | 1,594 | M5.2/M2 fallback |
| R00 | 菌株、化合物均未知 | 425 | M5.2/M2 fallback |
| R11 | 两实体已见，目标 pair/time 未见 | 135 | M5.2，time 路由使用 M6.21 |
| control | Water/DMSO/QC | 228 | M5.2 background/control path |

`test_both` 不是单一状态，其中 treatment 行精确拆成 R10 `432`、R01 `272`、
R00 `425`。

### 4. 环境

已记录训练环境：Python `3.8.20`、PyTorch `2.4.1+cu124`、CUDA 12.4、
NumPy `1.24.4`、Pandas `2.0.3`、RDKit `2024.03.5`。

```bash
conda create -n goai-m12 python=3.8.20 -y
conda activate goai-m12
python -m pip install -r requirements.txt
python -m pip install -e .
```

GPU 版 PyTorch 应根据服务器 CUDA 版本从 PyTorch 官方索引安装。CPU 也可完成推理，
速度会更慢。

### 5. 数据放置

完整推理只需要组委会发布的 test metadata：

```text
/path/to/official-data/
└── WAYB_WAYC_metadata_test.csv
```

若要重新训练，还需要：

```text
WAYB_WAYC_metadata_train_val.csv
WAYB_WAYC_proteome_raw_train_val.csv
WAYB_WAYC_metadata_test.csv
```

化合物结构映射和全量 refit support manifest 已在 `resources/entities/`。它们不含
蛋白标签或预测值。

### 6. 一键复现最终预测

先校验全部 16 个权重文件：

```bash
python scripts/verify_release.py
```

再从 checkpoint 直接重建 M12.0：

```bash
python scripts/predict_m12.py \
  --metadata-test /path/to/official-data/WAYB_WAYC_metadata_test.csv \
  --output outputs/m12/prediction.csv \
  --device cuda:0 \
  --batch-size 256
```

输出目录包含：

```text
outputs/m12/
├── prediction.csv
├── route_audit.csv
└── prediction_contract.json
```

推理脚本会强制检查：

- test metadata 字段和 `sample_ID` 唯一性；
- 4,422 个蛋白列的顺序一致性；
- 15 个 checkpoint 的训练范围和输出合同；
- M6 的 `background + treatment × response = final` 重建误差；
- M2/M6/M9 三个家族的蛋白顺序；
- 官方 metadata 的五类逐行路由计数；
- 最终矩阵无 NaN/inf。

本仓库回放结果与内部冻结 M12.0 的 `19,695,588` 个数值最大绝对误差为
`3.0e-6`，没有元素超过 `1e-5`。机器可读记录见
[`manifests/reproduction_receipt.json`](manifests/reproduction_receipt.json)。

### 7. 目录

```text
Go-AI-Optimal/
├── README.md
├── description.md              # 超详细架构说明
├── EXTERNAL_RESOURCES.md       # 外部数据、许可与归因
├── src/
│   ├── goai_baseline/
│   ├── goai_response/          # M2/M6/M7/M8 及最终组件
│   ├── goai_graph/
│   └── goai_rna_transfer/      # M9 OP3 RNA->protein 迁移
├── scripts/
│   ├── predict_m12.py          # 最终全链路推理入口
│   ├── verify_release.py
│   └── ...                     # OOF、审计、训练和消融脚本
├── configs/
├── weights/
│   ├── manifest.json
│   ├── m2/
│   ├── m6/
│   ├── m9/
│   └── pretrained/
├── resources/entities/
├── manifests/
├── docs/
├── research/rna_transfer/
└── tests/
```

### 8. 初赛作品附件

按算法赛页面要求组织的单 ZIP 附件已放在
[`deliverables/AI4R_ALG_AIVC_队伍名待替换.zip`](deliverables/AI4R_ALG_AIVC_队伍名待替换.zip)。
它包含初赛方案 PDF/DOCX、源码、结果表和外部资源披露，不含官方数据或预测文件。

当前文件仍需填写真实队伍名后才能上传。具体替换位置、大小和 SHA256 见
[`deliverables/README.md`](deliverables/README.md)。

### 9. 测试

```bash
pytest -q
python scripts/verify_release.py
```

本仓库当前完整测试结果为 `202 passed, 2 skipped`。两个 skip 分别需要未再分发的
GOAI 官方 metadata，以及可选的 Peter et al. 1,011-genome 原始证据包。除此之外，
本仓库还执行了完整 checkpoint replay，结果见上面的复现收据。

### 10. 重要限制

- 所有成绩是本地 OOF proxy，没有官方提交 ID 或官方分数。
- 当前权威本地输出合同是 4,422 个 train-supported proteins；若官方入口要求
  5,243 列，必须先取得组委会的 sample-submission 列合同，不能盲目补列。
- M9.6 使用外部 OP3 RNA 扰动数据，属于开放知识路线；正式参赛时应按 GOAI 最新
  规则披露外部数据。
- 新菌株语义和已见实体专家都已真实实现并做过对照，但最终混合的最佳权重为 0，
  因而没有进入 M12.0。
- R01/R00 仍是主要瓶颈：当前 fallback 对完全新菌株和双未知组合缺少稳定可迁移语义。

### 11. 许可与引用

源码采用 [MIT License](LICENSE)。训练权重仍受 GOAI 官方数据使用条款和外部资源条款
约束。OP3 数据为 CC BY 4.0，来源、文件哈希和用途见
[`EXTERNAL_RESOURCES.md`](EXTERNAL_RESOURCES.md)。仓库不再分发 GOAI 原始数据。

## English

### 1. Overview

This repository releases the complete reproducible implementation of
`GOAI-M12.0`, our current best local candidate for the GOAI virtual yeast-cell
perturbation task. It includes the full source tree, all 15 downstream
checkpoints, one OP3 pretrained encoder, the deterministic support router, and
a checkpoint-only inference command.

It deliberately excludes official GOAI raw data, labels, OOF/test predictions,
and submission files.

M12.0 is a routed system, not one monolithic neural network:

```text
M2.0/M2.31 fallback + M6.11/M6.21 biological model
                         + M9.6 OP3 chemical response
                         + canonical per-row support routing
                         = GOAI-M12.0
```

See [`description.md`](description.md) for the complete data-to-tensor,
layer-by-layer, loss, ensemble, and routing specification.

### 2. Local validation

All values below are strict local chemical-held-out OOF proxies, not official
leaderboard scores.

| Metric | GOAI-M12.0 |
|---|---:|
| FC PCC | **0.426342** |
| Context residual PCC | 0.060967 |
| High-effect PCC | **0.603184** |
| High-effect F1 | **0.233970** |
| Median absolute sample R2 | 0.979150 |

M12.0 improves FC PCC over M11.0 by `+0.000203`, with a 37-chemical paired
bootstrap 95% CI of `[+0.000059, +0.000355]`. It improves high-effect PCC but
reduces context-residual PCC; it is therefore an FC-primary candidate rather
than a uniform Pareto improvement.

### 3. Installation and inference

```bash
conda create -n goai-m12 python=3.8.20 -y
conda activate goai-m12
python -m pip install -r requirements.txt
python -m pip install -e .

python scripts/verify_release.py

python scripts/predict_m12.py \
  --metadata-test /path/to/WAYB_WAYC_metadata_test.csv \
  --output outputs/m12/prediction.csv \
  --device cuda:0 \
  --batch-size 256
```

The command generates `prediction.csv`, `route_audit.csv`, and
`prediction_contract.json`. The public checkpoint replay matched the frozen
internal candidate within a maximum absolute difference of `3e-6` across all
`4,454 x 4,422` values.

### 4. Route contract

| Regime | Meaning | Rows | Model path |
|---|---|---:|---|
| R10 | seen strain, unseen chemical | 2,072 | M6.11 background/calibration + M6/M9.6 response fusion |
| R01 | unseen strain, seen chemical | 1,594 | M5.2/M2 fallback |
| R00 | both unseen | 425 | M5.2/M2 fallback |
| R11 | both seen, target pair/time absent | 135 | M5.2, including M6.21 time route |
| control | Water/DMSO/QC | 228 | M5.2 background/control path |

Routes are derived from the entities actually observed during full refitting,
not inferred from `split_final` labels.

### 5. Competition attachment

The staged single-ZIP preliminary attachment is available at
[`deliverables/AI4R_ALG_AIVC_队伍名待替换.zip`](deliverables/AI4R_ALG_AIVC_队伍名待替换.zip).
It contains the proposal documents, source code, result summaries, and resource
disclosure, but no official data or predictions. The actual team name must be
filled in before portal upload; see [`deliverables/README.md`](deliverables/README.md).

### 6. Limitations and licensing

The reported scores are local proxies. The current verified output contract is
4,422 train-supported proteins, while an authoritative organizer contract is
still needed before adapting to any 5,243-column interface. M9.6 uses external
OP3 RNA perturbation data and must be disclosed as an open-knowledge component.

Source code is MIT licensed. Checkpoint use remains subject to GOAI official
data terms and OP3 CC BY 4.0 attribution. See
[`EXTERNAL_RESOURCES.md`](EXTERNAL_RESOURCES.md) for complete provenance.
