# GOAI 多模型结构与关键指标详细对比

> 版本：2026-08-14
> 结论边界：本文所有数字均为本地冻结验证、严格 OOF 或研究性 proxy；当前没有官方提交回执、官方 PSS 或排行榜分数。

## 1. 技术摘要

1. **当前正式冻结交付系统仍是 `GOAI-M5.2`。** 它不是单个网络，而是按场景选择模型：S1 新化合物使用 3-seed `M6.11`，time 使用 3-seed `M6.21`，S2 新菌株和 S3 双未知保留 `M5.1`。
2. **最重要的历史跃迁是 `M1 -> M2`。** `M1` 直接用 absolute MSE 预测 4,422 维蛋白组；`M2` 首次显式训练 matched-control FC，并拆成 Background、Calibration、Response。按同一 `V0-FROZEN` 口径，`M1.2 -> M2.0` 的 FC PCC 在 S1/S2/S3/time 分别增加约 `+0.198/+0.195/+0.119/+0.376`。
3. **当前最佳已确认结构升级是 cell-conditioned response。** 严格 S1 OOF 中，`M6.11` 将 FC PCC 从三种子 `M2.0` 的 `0.360289` 提高到 `0.371674`；严格 time-forward 中，`M6.21` 从 `0.534982` 提高到 `0.578731`，且 residual 与 high-effect 指标同步提高。
4. **新药特异性仍是核心瓶颈。** `M6.11` 的 S1 FC PCC 为 `0.371674`，但 Context PCC 只有 `0.098239`，说明较多分数来自公共 context/stress pattern，而不是新药独特响应。
5. **`M7.1` 已证明“通用模型 + 已见菌株专家”有很强的插值价值。** 在菌株已见的 R10/R11/RT 中，相对 `M7.0` 的 FC PCC 约提高 `+0.079~+0.080`；菌株未见的 R00/R01 完全不变，说明硬门控工作正常。但这不是新菌株语义泛化，且最终 nested-scale/promotion gate 尚未完成。
6. **`M9.6` 是目前最有价值的开放知识研究组件，但不是可交付完整模型。** 它的 S1 response-only FC/Context PCC 为 `0.437656/0.139366`，并通过 real-vs-shuffled RNA 表示归因；但 high-effect PCC/F1 明显低于 `M6.11`，且没有 absolute 输出，不能直接替换 `M5.2`。

## 2. 如何读指标

模型迭代的主判断顺序应为：

```text
FC PCC
  -> 对应 OOD residual PCC（S1 看 Context；S2 看 Drug）
  -> High-effect PCC / F1
  -> Absolute sample R2
  -> log2 RMSE
```

| 指标 | 定义与含义 | 方向 | 主要用途 |
|---|---|---:|---|
| `FC PCC` | 对 exact matched control 计算 `delta = treatment - control`，在共同观测位置上计算预测 delta 与真实 delta 的 Pearson 相关 | 越高越好 | 所有 OOD 场景的第一主指标 |
| `Context PCC` | `PCC(pred_delta - context_mean, true_delta - context_mean)`；context mean 只由 fold-train truth 构建 | 越高越好 | S1：是否预测出新药相对公共 context 的独特响应 |
| `Drug PCC` | `PCC(pred_delta - drug_mean, true_delta - drug_mean)`；drug mean 只由 fold-train truth 构建 | 越高越好 | S2：是否预测出已见药物在新菌株中的特异修正 |
| `Absolute sample R2` | 对每个样本在 4,422 个蛋白上计算 absolute log2 R2，再取有限样本的中位数 | 越高越好 | 完整蛋白组保真度护栏；不能替代 FC |
| `High-effect PCC` | 仅在真实 `abs(delta) > 1` 的蛋白位置计算预测与真实 delta PCC | 越高越好 | 大效应蛋白的幅度与模式是否正确 |
| `High-effect F1` | 预测阳性同样要求 `abs(pred_delta) > 1`；TP 还要求预测与真实方向一致 | 越高越好 | 是否准确检出大效应蛋白，而非只拟合数值趋势 |
| `log2 RMSE` | absolute log2 预测与真实值的均方根误差 | 越低越好 | 数值稳定性诊断；容易被大量小变化蛋白主导 |

不同场景的优先级：

| 场景 | 第一主指标 | 第二主指标 | 护栏 |
|---|---|---|---|
| S1 / R10：新化合物 | FC PCC | Context PCC | High PCC/F1、Abs R2 |
| S2 / R01：新菌株 | FC PCC | Drug PCC | High PCC/F1、Abs R2 |
| S3 / R00：双未知 | FC PCC | 无可靠单实体 residual | High PCC/F1、Abs R2 |
| Time / RT | FC PCC | Context/Drug PCC（可定义时） | High PCC/F1、Abs R2、time-forward |

## 3. 数据和验证口径

- 已发布有标签数据共 `8,958` 行；早期模型选择使用其中 `5,920` 行，最终 refit 才使用全部标签。
- 原始蛋白列 `5,243` 个；按训练侧缺失率 `< 0.80` 保留 `4,422` 个。
- 目标是 `log2(raw intensity)`；NaN 在 loss 和指标中通过 mask 排除，不做伪标签填补。
- 训练侧约只有 37 个 treatment chemical、4 个训练菌株，实体有效样本量远低于行数。

以下协议禁止混成一个排行榜：

| 协议 | 含义 | 可比较范围 |
|---|---|---|
| `V0-FROZEN` | 5,920 行拟合，在冻结的 S1/S2/S3/time validation split 评估 | 只和同表、同 split 结果比 |
| `V1-ENTITY-OOF` | held-out chemical/strain 的全部行不进入 fold train；每折重拟合 scaler、support、control reference、prototype/SVD 和模型 | 严格实体 OOD 主证据 |
| `V1-TIME-FORWARD` | 用较早时间训练、较晚时间验证 | 时间外推主证据 |
| `R00/R10/R01/R11/RT` | 双未知 / 菌株已见药未知 / 菌株未知药已见 / 两实体已见但 pair 未见 / 时间条件外推 | M7 逐行 support router 证据 |
| `R1-ALL-LABELED` | 全部 8,958 行 refit | 只生成 test 预测，不能产生模型选择分数 |
| response-only research | 只预测 treatment delta，不重建 absolute 蛋白组 | 不能与完整模型比较 Abs R2/RMSE |

## 4. 模型家族总览

| 模型 | 核心结构 | 输出 | 主 loss | 未知实体能力 | 当前角色 |
|---|---|---|---|---|---|
| `M0.0` | 逐蛋白训练均值 | absolute 4,422 | 闭式均值 | 只有全局均值 | 统计基线 |
| `M0.1` | exact matched control 均值 | absolute 4,422 | 无 | test 无观测 control 时不可用 | 诊断参照 |
| `M1.0/M1.2` | `input -> 256 -> 256 -> 4422` MLP | absolute 4,422 | masked MSE | one-hot 未知类为全零；M1.2 加 OOF 先验 fallback | 历史基线 |
| `M2.0` | 独立 Background + Calibration + learned low-rank Response | absolute + FC | `0.25 Abs + 1 Bg + 1 FC` MSE | 未知实体依赖公共 response | M5 核心组件 |
| `M2.31` | 同 M2.0 | absolute + FC | 三项 Huber | 同 M2.0 | M5 融合辅助 |
| `M6.11` | shared cell encoder + concat response，rank 256 | absolute + FC | M2 三项 MSE | response 可读取 cell state；无药物语义 | M5.2 的 S1 路由 |
| `M6.21` | shared cell encoder + FiLM response，rank 256 | absolute + FC | M2 三项 MSE | 已见实体的时间/条件外推强 | M5.2 的 time 路由 |
| `M5.2` | 按场景路由 M6.11/M6.21/M5.1 | absolute 4,422 | 组件独立训练 | split-level 路由；不是逐行 support | 当前冻结交付系统 |
| `M7.0` | universal `B_U + C + I*R_U` | absolute + 组件 | MSE + calibration 约束 | 无 ID 专家；当前无实体语义配置 | 正式对照 |
| `M7.1` | `M7.0 + g_s(B_s + R_s)` | absolute + 组件 | staged residual + joint | 已见菌株强；未见菌株 gate=0 | 强候选，未冻结 |
| `M7.2/7.3/7.4` | 再加 chemical / 双实体 / pair-time 专家 | absolute + 组件 | staged residual + joint | 解决已见实体插值，不自动解决新实体 | 待完整确认 |
| `M8.0-8.3` | M7 + chemical/strain semantic adapters | absolute + 组件 | 同 M7 | 目标是真正新药/新菌株迁移 | identity gate 阻塞 |
| `M9.6` | frozen context predictor + frozen OP3 RNA chemical residual gate | FC 4,422 | weighted SmoothL1 + residual shrinkage | 有药物特异开放知识信号 | research component |
| `M10.*` | OP3/MoA/ADT2GEX 冠军结构直接迁移 | FC 4,422 | masked mixed loss 或 RMSE | 实验中未通过 chemical-sensitivity gate | 研究负结果 |

## 5. 历史冻结基线：M2 相对 M1 是任务定义级跃迁

共同口径：`V0-FROZEN`。每格为 `absolute sample R2 / FC PCC`。

| 模型 | S1 新化合物 | S2 新菌株 | S3 双未知 | time |
|---|---:|---:|---:|---:|
| `M0.0` protein mean | 0.860 / 0.152 | 0.906 / 0.181 | 0.862 / 0.146 | 0.902 / 0.164 |
| `M0.1` exact control | 0.986 / — | 0.984 / — | 0.986 / — | 0.984 / — |
| `M1.0` one-hot MLP | -0.146 / 0.124 | -0.499 / 0.133 | -4.078 / 0.083 | 0.858 / 0.127 |
| `M1.2` OOF-safe prior MLP | 0.828 / 0.137 | 0.819 / 0.162 | 0.822 / 0.131 | 0.819 / 0.147 |
| `M2.0-S42` response decomposition | — / 0.335218 | — / 0.356882 | — / 0.250255 | — / 0.522510 |
| `M2.10-S42` + Morgan/RDKit | — / 0.349377 | — / 0.358948 | — / 0.260427 | — / 0.533872 |

`M1.2 -> M2.0-S42` 的 FC PCC 增益：

| 场景 | M1.2 | M2.0 | 绝对增益 |
|---|---:|---:|---:|
| S1 | 0.137 | 0.335218 | +0.198218 |
| S2 | 0.162 | 0.356882 | +0.194882 |
| S3 | 0.131 | 0.250255 | +0.119255 |
| time | 0.147 | 0.522510 | +0.375510 |

这次跃迁来自四件同时发生的变化：FC 成为直接监督；control/background、observation calibration、treatment response 被拆开；response 使用低秩共享；plate/instrument/source 测量偏差被显式校正。因此它不是普通的“网络调参收益”。

## 6. 严格 S1 新化合物多指标对比

共同口径：`V1-ENTITY-OOF` S1，四折 chemical-held-out。`—` 表示未产出或不适用，不能自行补算后混入原结论。

| 模型 | 统计范围 | log2 RMSE ↓ | Abs R2 ↑ | FC PCC ↑ | Context PCC ↑ | High PCC ↑ | High F1 ↑ | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `M2.0` | seed 42 | 0.431284 | 0.979152 | 0.358665 | 0.102352 | 0.630771 | 0.170422 | 单 seed 主干 |
| `M2.10` | seed 42，Morgan-512/RDKit | 0.431798 | 0.979501 | 0.346803 | 0.111427 | 0.608116 | 0.162261 | Context 升但其余主指标下降 |
| `M2.11` | seed 42，Morgan-2048/RDKit | 0.436242 | 0.978932 | 0.340027 | 0.106357 | 0.602197 | 0.159551 | 未晋级 |
| `M2.30` | seed 42，FC=MSE+MAE | 0.425745 | 0.980054 | 0.360143 | 0.107577 | 0.624215 | 0.164959 | 数值改善，高效应退化 |
| `M2.31` | seed 42，all Huber | 0.423574 | 0.980408 | 0.363957 | 0.109040 | 0.614791 | 0.168915 | 仅作融合辅助 |
| `M2.0` | 3-seed bag | — | 0.979106 | 0.360289 | 0.102079 | 0.633293 | 0.170336 | M6 对照 |
| `M5.1` | 3-seed MSE/Huber blend | — | — | 0.361525 | 0.103346 | 0.631517 | 0.170129 | 上一版冻结 S1 |
| `M6.11` | 3-seed concat-256 | — | 0.979297 | **0.371674** | 0.098239 | **0.635453** | **0.182438** | 当前完整模型 S1 最强已确认组件 |

`M6.11 - M2.0 bag` 的 FC 增益为 `+0.011385`，四折 bootstrap 95% CI `[+0.008648,+0.014281]`；三个 seed 的逐 chemical 增益方向一致。但 Context PCC 下降 `-0.003840`，所以这是可靠结构进步，不是药物语义突破。

## 7. S2、S3 与时间外推

### 7.1 M5 的严格 OOF 场景结果

| 场景 / 模型 | FC PCC | Context PCC | Drug PCC | High PCC | High F1 | Abs R2 | log2 RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| S2 / `M2.0` | 0.280621 | — | 0.218011 | 0.614390 | 0.158383 | — | — |
| S3 / `M2.0` | 0.216675 | — | — | 0.497406 | 0.115913 | — | — |
| time / `M5.0` 70:30 blend | 0.508836 | 0.356763 | 0.425353 | 0.772590 | 0.306538 | — | — |
| time-forward / `M5.0` | 0.531470 | 0.380233 | 0.446083 | 0.777819 | 0.319064 | — | — |

### 7.2 Cell-conditioned FiLM 对时间外推的完整改善

共同口径：3-seed bagging。

| 协议 | 模型 | FC PCC | Context PCC | Drug PCC | High PCC | High F1 | Abs R2 |
|---|---|---:|---:|---:|---:|---:|---:|
| time | `M2.0` | 0.511024 | 0.358420 | 0.427350 | 0.776183 | 0.307357 | 0.983919 |
| time | `M6.20` FiLM-128 | 0.537243 | 0.383002 | 0.465072 | 0.787314 | 0.343632 | 0.984165 |
| time | `M6.21` FiLM-256 | **0.539912** | **0.383961** | **0.468130** | **0.789024** | **0.346425** | **0.984186** |
| time-forward | `M2.0` | 0.534982 | 0.383512 | 0.449532 | 0.782276 | 0.321945 | 0.980950 |
| time-forward | `M6.20` FiLM-128 | 0.576423 | 0.425908 | 0.503887 | 0.803385 | 0.379061 | 0.981455 |
| time-forward | `M6.21` FiLM-256 | **0.578731** | **0.427088** | **0.506483** | **0.804417** | **0.382575** | **0.981505** |

严格 time-forward 中，`M6.21 - M2.0` 的 FC 增益为 `+0.043749`，95% CI `[+0.040622,+0.046877]`；Context、Drug、High PCC、High F1 全部同向提高。这是目前最干净的结构性胜利。

## 8. M7 通用主干与已见实体专家

下表为 4-fold、3 seeds（42/52/62）producer 平均、expert scale=1 的正式快照；尚未经最终 nested-scale 和 promotion gate，因此不能写成冻结模型成绩。Residual 列在 R10/R11/RT 为 Context PCC，在 R01 为 Drug PCC。

| Regime | M7.0 FC | M7.1 FC | Delta FC | M7.0 residual | M7.1 residual | M7.0 High PCC | M7.1 High PCC | M7.0 High F1 | M7.1 High F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R00 双未知 | 0.202665 | 0.202665 | 0.000000 | — | — | 0.459523 | 0.459523 | 0.112986 | 0.112986 |
| R10 菌株已见/药未知 | 0.258148 | **0.337283** | **+0.079135** | 0.073982 | **0.105059** | 0.512829 | **0.553283** | 0.136026 | **0.162771** |
| R01 菌株未知/药已见 | 0.206093 | 0.206093 | 0.000000 | 0.226536 | 0.226536 | 0.464987 | 0.464987 | 0.114363 | 0.114363 |
| R11 两实体已见/pair 未见 | 0.260170 | **0.338905** | **+0.078735** | 0.079849 | **0.112337** | 0.526278 | **0.567746** | 0.135539 | **0.162191** |
| RT 时间/条件外推 | 0.261380 | **0.341656** | **+0.080276** | 0.090482 | **0.129375** | 0.519077 | **0.561354** | 0.137522 | **0.164775** |

解释：`M7.1` 只在 fit rows 确实见过该菌株时启用 `B_s/R_s`，所以 R00/R01 与 `M7.0` 数值完全一致。它证明 ID 残差专家有效，但不提供新菌株的 genome semantics。

## 9. 开放知识 response-only 研究

### 9.1 M9：RNA 扰动表示迁移

共同口径：V1 S1 四折，5,078 treatment rows；所有模型只预测 4,422 维 log2 FC，因此 Abs R2 和 absolute RMSE 不适用。

| 模型 | seed 范围 | FC PCC | Context PCC | High PCC | High F1 | 结论 |
|---|---|---:|---:|---:|---:|---|
| `M9.0` no chemical | S42 | 0.429196 | 0.131844 | 0.541367 | 0.107253 | 强 context-only 基线 |
| `M9.1` Morgan scratch | S42 | 0.334650 | 0.122943 | 0.470212 | 0.108172 | 化学直接输入破坏主干 |
| `M9.2` OP3 real | S42 | 0.362599 | 0.131629 | 0.504149 | 0.113014 | real 显著优于 shuffled |
| `M9.3` OP3 shuffled | S42 | 0.309715 | 0.086713 | 0.478143 | 0.109030 | 负对照 |
| `M9.4` L1000 real | S42 | 0.362795 | 0.117779 | 0.507710 | 0.114884 | 未显著胜 shuffled |
| `M9.5` L1000 shuffled | S42 | 0.351651 | 0.092366 | 0.493914 | 0.120598 | 负对照 |
| `M9.0` context-only bag | S42/43/2026 | 0.435627 | 0.136079 | 0.544178 | 0.106032 | M9.6 父模型 |
| `M9.7` shuffled residual bag | S42/43/2026 | 0.427173 | 0.123558 | 0.546507 | 0.108425 | 匹配负对照 |
| `M9.6` real OP3 residual bag | S42/43/2026 | **0.437656** | **0.139366** | **0.551712** | 0.108180 | 有真实 additive drug signal；research only |
| `M6.11` 完整模型对照 | S42/43/2026 | 0.371674 | 0.098239 | **0.635453** | **0.182438** | 可交付 absolute 模型 |

归因证据：`M9.6 - M9.7` 的 FC 为 `+0.010483`，95% CI `[+0.004429,+0.016535]`；Context 为 `+0.015808`，CI `[+0.004706,+0.035921]`，三个 seed 同向。因此 OP3 RNA 预训练表示确实含可迁移药物信息。它仍未晋级，因为相对 `M6.11` 的 High PCC/F1 分别低 `0.083742/0.074258`，且缺少 absolute background。

### 9.2 M10：邻近竞赛冠军网络的直接迁移

这些也是 response-only 模型。关键门禁不是 pooled FC 是否较高，而是正确化学表示是否优于同一 fitted model 的 chemical derangement 或架构匹配的 NoChem。

| 模型 | 架构 | FC PCC | Context PCC | High PCC | High F1 | chemical-sensitivity 结论 |
|---|---|---:|---:|---:|---:|---|
| `M10.3-Morgan` | MoA 3FC，S42 | 0.396813 | 0.089743 | 0.545506 | 0.106001 | deranged FC 0.401612；拒绝 |
| `M10.3-NoChem` | 同架构无化学输入 | **0.417022** | **0.117179** | **0.562222** | 0.093408 | 显著优于 Morgan 的三个相关指标 |
| `M10.5-Morgan` | 4x512 GELU + signed linear head | 0.410819 | 0.041099 | 0.560273 | 0.162372 | deranged 0.411786；拒绝 |
| `M10.0-MTR` | OP3 2-layer LSTM，25-epoch screen | 0.343581 | 0.043330 | 0.519740 | 0.094691 | deranged 0.371791；拒绝 |
| `M10.1-MTR` | OP3 2-layer GRU，25-epoch screen | 0.354576 | 0.038647 | 0.528999 | 0.111419 | deranged 0.379294；拒绝 |
| `M10.2-MTR` | OP3 1D-CNN，25-epoch screen | 0.338620 | 0.028652 | 0.521875 | 0.089842 | deranged 0.374798；拒绝 |
| `M10.7-Morgan` | MoA trunk + chemical-contrast residual fine-tune | 0.441819 | -0.059985 | 0.611169 | 0.164524 | NoChem FC 0.472500；拒绝 |
| `M10.8-Morgan` | 同上，冻结 trunk 只训输出权重 | 0.468183 | -0.045868 | 0.621298 | 0.164893 | NoChem FC 0.472500；拒绝 |

结论：这些网络能够从 context/background 获得很高 pooled FC，但正确药物配对不优于打乱或无药物输入。它们说明“照搬冠军架构”不能自动解决 GOAI 新药 OOD。

## 10. 经典模型的输入、网络、输出与 Loss

### 10.0 先固定维度口径

下列 ASCII 图默认使用 **D0 的 5,920 行模型选择 fit**，不是 8,958 行全量 refit。每个 OOF fold 的 vocabulary 还会随 fold-train support 略微缩小，原则上应读取该 checkpoint 的 `feature_summary.json`，不能把一个全局维度硬套给所有 fold。

| Block | D0 实际维数 | 加法来源 |
|---|---:|---|
| M1.0 basic one-hot | 54 | strain 4 + chemical 40 + medium 2 + temperature 2 + categorical time 6 |
| M1.2 完整输入 | 8,898 | basic 54 + strain absolute prior 4,422 + chemical delta prior 4,422 |
| M2 response input | 52 | strain 4 + chemical 40 + medium 2 + temperature 2 + continuous time 4 |
| M2/M6 cell/background input | 12 | strain 4 + medium 2 + temperature 2 + continuous time 4 |
| M2/M6 observation input | 155 | source 4 + instrument 7 + plate 144 |
| M6 perturbation input | 40 | chemical one-hot 40；当前晋级模型无结构向量 |
| M6 response MLP input | 168 | encoded cell 128 + perturbation 40 |
| M7 universal cell input | 8 | medium 2 + temperature 2 + continuous time 4；当前无 strain semantics |
| M7 universal perturbation | 0 | 当前 `chemical_map/features=null`，所以没有 chemical semantic 数值输入 |
| M9 chemical input | 2,048 -> 64 | Morgan-2048 经预训练 chemical encoder 压缩 |
| M9 context embedding | 47 -> 64 | 8 个 categorical field 的 embedding 共 47 维，再投影到 64 |

全量 8,958 行 refit 后，M6 的 strain/chemical vocabulary 变为 `5/46`，因此 cell input 为 `5+2+2+4=13`，perturbation input 为 46，response MLP input 为 `128+46=174`；observation 仍为 155。模型结构没变，变化的是 fit support。

### 10.1 M0：蛋白均值与 exact-control 参照

**M0.0 输入与输出**

- 不读取样本条件。
- 对每个蛋白仅用训练行已观测 log2 值求均值，输出固定 4,422 维向量。
- 它的 absolute R2 已达到约 0.86-0.91，证明蛋白基础丰度占绝对信号的大部分；但 FC PCC 仅 0.146-0.181。

**M0.1 exact-control**

- 对目标 treatment，按 `data_source + instrument + plate + strain + medium + temperature + time + time_unit` 找训练池中的 Water/DMSO control。
- 对多个匹配 control 逐蛋白求均值。
- 它是背景估计诊断，不是可部署模型：隐藏 test 没有可供模型读取的真实 control abundance。

### 10.2 M1.0/M1.2：直接 absolute MLP

**输入构造**

- 基础类别：strain、chemical、medium、temperature、time，各自在 fit rows 上建立 one-hot vocabulary。D0 中分别为 `4/40/2/2/6` 维，总计 54 维。
- 验证或 test 的未见类别变成对应 block 的全零向量。
- `M1.2` 额外拼接两个 `4,422` 维 fold-safe target prior：strain absolute mean、chemical matched-control delta mean。
- 训练行 prior 使用 leave-one-row-out；不存在实体统计时回退到 fold-train global mean/global delta。

**M1.0 网络**

```text
strain one-hot       [4] ----+
chemical one-hot    [40] ----|
medium one-hot       [2] ----+--> concat x_basic [54]
temperature one-hot  [2] ----|          |
time one-hot         [6] ----+          v
                                  Linear 54 -> 256
                                  ReLU + Dropout(0.10)
                                          |
                                          v
                                  Linear 256 -> 256
                                  ReLU + Dropout(0.10)
                                          |
                                          v
                                  Linear 256 -> 4,422
                                          |
                                          v
                               predicted absolute log2 y_hat
```

**M1.2 在哪里增加输入**

```text
x_basic                                      [54]
strain absolute mean prior                [4,422]
chemical matched-control delta prior      [4,422]
                                               |
                                               v
concat input = 54 + 4,422 + 4,422 =       [8,898]
                                               |
                                               v
                         same 256 -> 256 -> 4,422 MLP
```

两个 prior 都是目标统计，因此严格 OOF 的核心不是“算出 prior”本身，而是保证每个训练行不读取自己的标签、每个验证实体不读取验证标签。

**Loss**

```text
For sample i, protein p:
  y[i,p] = log2(raw_intensity[i,p])
  m[i,p] = 1 if y[i,p] is observed else 0

L_M1 = sum(i,p) m[i,p] * (y_hat[i,p] - y[i,p])^2
       ------------------------------------------------
                       sum(i,p) m[i,p]
```

这是 full-batch Adam、learning rate `1e-3`、50 epochs；M1 不做逐蛋白 target 标准化。问题在于 absolute loss 被蛋白基础丰度主导，treatment-control delta 相对很小，网络没有足够动力学习扰动响应。

### 10.3 M2.0：Background + Calibration + low-rank Response

**输入加工**

1. 原始 intensity 取 log2；对每个蛋白 `p`，仅用 fold-train 计算 `mu_p` 与 `s_p=max(std_p,0.10)`，网络目标是 `z=(log2(y)-mu_p)/s_p`。
2. Background 输入：strain、medium、temperature one-hot，加 4 维时间编码：`t/t_max`、`log1p(t)/log1p(t_max)`、`sin(2pi*t/t_max)`、`cos(...)`。D0 为 `4+2+2+4=12` 维。
3. Response 输入：strain、chemical、medium、temperature one-hot，加相同时间编码；M2.0 不含结构特征。D0 为 `4+40+2+2+4=52` 维。
4. Calibration 输入：data_source、instrument、plate one-hot；D0 为 `4+7+144=155` 维，不读取 chemical/strain semantics。
5. treatment mask：Water、DMSO、Quality Control 不进入 response；其他行才加 response。
6. FC target：只对能在 fold-train pool 找到 exact matched control 的 treatment 构建 `delta=(log2 y_treatment-log2 y_control)/s_p`。匹配键为 source、instrument、plate、strain、medium、temperature、time、time unit。

**网络与输出**

```text
BIOLOGICAL BACKGROUND                    OBSERVATION CALIBRATION
---------------------                    -----------------------
strain one-hot       [4] --+             source one-hot       [4] --+
medium one-hot       [2] --|             instrument one-hot   [7] --+--> x_obs [155]
temperature one-hot  [2] --+--> x_bg     plate one-hot      [144] --+        |
time continuous      [4] --+    [12]                                       v
                                |                               Linear 155 -> 16, no bias
                                v                                          |
                         Linear 12 -> 128                                  v
                         GELU + Dropout                         z_cal [16] @ W_cal [16,4422]
                                |                                          |
                                v                                          v
                         Linear 128 -> 4,422                         C [4,422]
                                |
                                v
                           B [4,422]

TREATMENT RESPONSE
------------------
strain one-hot       [4] --+
chemical one-hot    [40] --|
medium one-hot       [2] --+--> x_resp [52]
temperature one-hot  [2] --|         |
time continuous      [4] --+         v
                                Linear 52 -> 128
                                GELU + Dropout
                                       |
                                       v
                                Linear 128 -> 64
                                       |
                                       v
                                  z_resp [64]
                                       |
                                       v
                    response_center [4,422] + z_resp @ W_resp [64,4,422]
                                       |
                                       v
                                  R [4,422]

FINAL IN STANDARDIZED PROTEIN SPACE
-----------------------------------
z_hat = B + C + I[treatment] * R                         [4,422]
y_hat = mu_protein + s_protein * z_hat                   [4,422 log2]
```

M2 的 Background 和 Response 参数独立；它们只通过最终 absolute loss 以及各自的 Background/FC loss 间接协调。Calibration 同样独立，这是合理的观测偏差边界，但旧 M2 对其中心化约束较弱。

**Loss**

```text
MSE(P,T,M) = sum M*(P-T)^2 / sum M

L_absolute   = MSE(B + C + I[treatment]*R, z_true, observed_mask)

L_background = MSE(B + C, z_true,
                   observed_mask * background_row_mask)

L_FC         = MSE(R, delta_true_standardized, exact_control_delta_mask)

L_total      = 0.25 * L_absolute
             + 1.00 * L_background
             + 1.00 * L_FC
```

- `L_absolute`：所有训练行、所有可观测 absolute 位置。
- `L_background`：M2/M6 的 legacy 设置使用所有非-treatment 行，即 Water、DMSO、QC；M7 的 `controls_only` 只用 Water/DMSO。
- `L_FC`：只有 exact matched-control treatment 且 treatment/control 两侧都观测到的蛋白位置。
- 默认均为 masked MSE；AdamW，lr `1e-3`，weight decay `2e-4`，batch 128，80 epochs，gradient clip 5。
- `M2.31` 把三项的 elementwise error 全部改为 Huber，`delta=1.0`：

```text
Huber(e; delta=1) = 0.5*e^2       , if abs(e) <= 1
                  = abs(e) - 0.5  , otherwise
```

### 10.4 M6.11：共享 cell state 的 concat response

M6 修复 M2 中“Background 和 Response 各算各的”这一结构问题。

```text
strain one-hot       [4] --+
medium one-hot       [2] --|
temperature one-hot  [2] --+--> x_cell [12]
time continuous      [4] --+          |
                                      v
                               Linear 12 -> 128
                               GELU + Dropout
                                      |
                                      v
                                 h_cell [128]
                                  /         \
                                 /           \
                                v             v
                    Linear 128 -> 4,422    concat with x_pert [40]
                                |             128 + 40 = [168]
                                v                    |
                           B [4,422]                 v
                                              Linear 168 -> 128
                                              GELU + Dropout
                                                     |
                                                     v
                                              Linear 128 -> 256
                                                     |
                                                     v
                                                z_resp [256]
                                                     |
                                                     v
                                          z_resp @ W_resp[256,4,422]
                                                     |
                                                     v
                                                R [4,422]

x_pert [40] = chemical one-hot [40] + chemical semantics [0]
C [4,422]   = 与 M2 相同的 observation calibration
z_hat       = B + C + I[treatment] * R
```

- 输入、Calibration、matched-control target 和三项 MSE 与 M2 保持一致。
- 关键变化只有：Background 和 Response 共享 `h_cell`，response rank 从 64 扩为 256。
- 未见 chemical 的 one-hot 仍是全零，因此提升来自更合理的 cell-response 交互，不是结构语义。
- Loss 仍是上一节的 `0.25 L_absolute + L_background + L_FC`，三项均为 masked MSE；没有额外 concat loss。

### 10.5 M6.21：FiLM 调制的 cell-conditioned response

```text
x_cell [12] -> CellEncoder -> h_cell [128] -> Linear -> B [4,422]
                                  |
                                  |               chemical one-hot x_pert [40]
                                  |                              |
                                  |                              v
                                  |                    Linear 40 -> 256
                                  |                         split in half
                                  |                       /              \
                                  |                      v                v
                                  |                 scale [128]      shift [128]
                                  |                      \                /
                                  v                       \              /
                        h_mod = h_cell * (1 + tanh(scale)) + shift
                                  |
                                  +---- concat x_pert [40] ----> [168]
                                                                    |
                                                                    v
                                                          Linear 168 -> 128
                                                          GELU + Dropout
                                                                    |
                                                                    v
                                                          Linear 128 -> 256
                                                                    |
                                                                    v
                                                               z_resp [256]
                                                                    |
                                                                    v
                                                     z_resp @ W_resp[256,4,422]
                                                                    |
                                                                    v
                                                               R [4,422]
```

FiLM 允许不同 perturbation 对 cell state 各维做乘性与加性调制。Loss 与 M6.11 完全相同；因此 time-forward 的提升可以归因于交互形式与 rank，而不是换了目标函数。它在 time/time-forward 上最有效，但在 S2/S3 没有稳定晋级。

### 10.6 M5.1/M5.2：融合与路由，不是新网络

**M5.1**

- `M2.0` 三个 seed 先等权平均；`M2.31` 三个 seed 先等权平均。
- S1 使用 `85% M2.0 + 15% M2.31`；S2/S3 100% M2.0；time 使用 `70% M2.0 + 30% M2.31`。
- 权重由 inner OOF 冻结，outer 只做 confirmation。

**M5.2**

```text
test_chem_only   -> mean(M6.11 S42/S43/S2026)
test_time        -> mean(M6.21 S42/S43/S2026)
test_strain_only -> 保留 M5.1
test_both        -> 保留 M5.1
```

它完成了全标签 refit 和 `4,454 x 4,422` test 预测契约，是当前可交付系统。但它按整个 `split_final` 路由，不按每行 checkpoint support 路由，这是 M7 要修复的问题。

### 10.7 M7：通用主干 + fold-fit 残差专家

**统一方程**

```text
y_hat = B_U + g_s * B_s + C_obs
      + I[treatment] * (R_U + g_s*R_s + g_c*R_c + g_sc*R_sc)
```

**通用输入**

- `general_cell = medium one-hot + temperature one-hot + time(4) + optional strain semantics`。
- `general_perturbation = optional chemical structure/semantic vector`。
- 不把 strain/chemical ID one-hot 放进 universal trunk，避免未见实体全零与已见 ID 记忆混在一起。
- 重要限制：当前 `M7.0/M7.1` 配置的 chemical map、chemical features、strain features 都是 null，因此 `R_U` 实际没有药物语义输入，`B_U` 也没有菌株语义输入。它是通用结构骨架，不是已完成的新药/新菌株语义模型。

**网络**

```text
UNIVERSAL PATH (ID hidden)
--------------------------
medium one-hot       [2] --+
temperature one-hot  [2] --+--> general_cell [8]
time continuous      [4] --+          |
strain semantics     [0] ------------+   (current M7.0/M7.1)
                                      v
                                Linear 8 -> 128
                                GELU + Dropout
                                      |
                                      v
                                  h_cell [128]
                                   /          \
                                  /            \
                                 v              v
                      Linear 128 -> 4,422   concat chemical semantics [0]
                                 |              total [128]
                                 v                    |
                           B_U [4,422]                v
                                              Linear 128 -> 128
                                              GELU + Dropout
                                                     |
                                                     v
                                              Linear 128 -> 256
                                                     |
                                                     v
                                           z_U [256] @ shared_W[256,4,422]
                                                     |
                                                     v
                                                R_U [4,422]

SEEN-ENTITY RESIDUAL EXPERTS
----------------------------
canonical strain index --g_s--> Embedding(4+unknown, 128) -> Linear -> B_s [4,422]
                       \-g_s--> Embedding(4+unknown, 256) ----------> z_s [256]
canonical chemical idx --g_c--> Embedding(vocab+unknown,256) ------> z_c [256]
canonical pair index ---g_sc--> Embedding(pair_vocab+unknown,256)
                                   * (1+tanh(Linear(h_cell))) ------> z_sc[256]

R_s  = z_s  @ shared_W [256,4,422]
R_c  = z_c  @ shared_W [256,4,422]
R_sc = z_sc @ shared_W [256,4,422]

OBSERVATION PATH
----------------
[source 4 | instrument 7 | plate 144] -> centered x_obs [155]
  -> optional centered plate dropout(0.30)
  -> Linear 155 -> 16 -> @ W_cal[16,4,422] -> C_obs [4,422]

FINAL
-----
z_hat = B_U + g_s*B_s + C_obs
      + I[treatment] * (R_U + g_s*R_s + g_c*R_c + g_sc*R_sc)
```

所有 response 专家共享同一个 `256 x 4,422` protein decoder；Background strain expert 单独解码 absolute 残差。

**硬门控与训练**

- `g_s/g_c/g_sc` 由当前 fold 真实 fit rows 的 canonical support vocabulary 逐行计算，unknown index 固定为 0。
- staged training：universal -> strain expert -> chemical expert -> pair expert -> 小学习率 joint。
- joint 中通过 entity dropout 模拟 00/10/01/11 support 状态，防止专家吞掉通用能力。
- 专家 scale 只能从 `{0,0.25,0.5,0.75,1}` 由 outer-train 内 nested OOF 选择。

**Calibration 与 Loss**

- observation one-hot 在 fold-train 上硬中心化；plate dropout `0.30`。
- 增加 `1e-4 * mean(C)^2 + 1e-4 * mean(C^2)`，约束零均值和幅度。
- `L_absolute` 使用所有观测 absolute 位置，监督上图最终 `z_hat`。
- `L_background` 在 `controls_only` 设置下只使用 Water/DMSO 行，监督 `B_U+g_sB_s+C_obs`；QC 不进入这一项。
- `L_FC` 只使用 exact-control delta mask，监督 `R_U+g_sR_s+g_cR_c+g_scR_sc`。

```text
L_M7 = 0.25 * L_absolute
     + 1.00 * L_background
     + 1.00 * L_FC
     + 1e-4 * mean_protein(mean_batch(C_obs)^2)
     + 1e-4 * mean(C_obs^2)

L_FC-correlation weight = 0
L_PPI weight            = 0
```

专家阶段并没有另造一套标签：冻结 universal 后，仍用同一 absolute/background/FC loss，只允许目标专家参数更新，所以专家学到的是通用预测剩余的 residual。

### 10.8 M9.6：冻结 context 主干 + RNA chemical residual

**输入**

- chemical：2048-bit Morgan，经 OP3 RNA perturbation 任务预训练的 `2048 -> 256 -> 64` ChemicalEncoder。
- context：strain、medium、temperature、time、time unit、source、instrument、plate 的 fold-fit categorical embeddings。以 S1 fold 0 为例，各 embedding 输出为 `4+4+4+6+3+4+6+16=47` 维，再投影为 64 维；每列的 index 0 保留给 fold 未见类别。
- 输出：4,422 维 log2 delta，不预测 absolute background。

**网络**

```text
FROZEN CONTEXT BASE (M9.0)
--------------------------
8 categorical fields
  -> 8 embedding tables
  -> concat embeddings [47]
  -> Linear 47 -> 64 + LayerNorm + GELU
  -> context64 [64]

zero Morgan [2,048]
  -> frozen ChemicalEncoder: 2,048 -> 256 -> 64
  -> constant chemical64 [64]

[chemical64 64 | context64 64 | projected interaction 64]
  -> concat [192]
  -> Linear 192 -> 256 -> Linear 256 -> 256
  -> Linear 256 -> 4,422
  -> base_delta [4,422]

OP3 RNA CHEMICAL RESIDUAL (M9.6)
--------------------------------
real Morgan [2,048]
  -> frozen OP3 encoder: Linear 2,048->256, LN, GELU, Dropout(0.20),
                         Linear 256->64, LN, GELU
  -> chem64 [64]

context64 [64] -> Linear 64->64 -> tanh -> gate64 [64]
chem64 * gate64 ----------------------------------> gated64 [64]

[chem64 64 | gated64 64] -> concat [128]
  -> Linear 128 -> 128 -> LayerNorm -> GELU -> Dropout(0.10)
  -> Linear 128 -> 4,422, ZERO INITIALIZED
  -> correction_std [4,422]

pred_std_delta = frozen base_std_delta + 0.20 * correction_std
pred_delta     = target_mean[4,422] + target_scale[4,422] * pred_std_delta
```

残差头最后一层零初始化，确保训练起点精确等于 context-only parent。

**Loss**

```text
w[i,p] = observed_mask[i,p]
         * (1 + 0.35 * I[abs(raw_true_delta[i,p]) > 1])

true_std_delta = (raw_true_delta - target_mean) / target_scale
pred_train_std = frozen_base_std_delta + correction_std

L_data = sum w * SmoothL1(pred_train_std, true_std_delta; beta=0.5)
         ----------------------------------------------------------
                                  sum w

L_shrink = sum observed_mask * correction_std^2
           -------------------------------------
                       sum observed_mask

L_M9.6 = L_data + 0.02 * L_shrink

inference std output = frozen_base_std_delta + 0.20 * correction_std
inference raw delta  = target_mean + target_scale * inference_std_output
```

其中 `SmoothL1(beta=0.5)` 的逐位置形式为：

```text
SmoothL1(e; beta=0.5) = 0.5*e^2/beta  , if abs(e) <= beta
                       = abs(e)-0.5*beta, otherwise
```

训练 loss 内的 correction 以 scale 1 拟合，最终 `0.20` 是固定残差缩放；它由 seed42/fold0 设计，folds1-3 用作未参与选择的确认子集。它提供了真实药物特异信号，但还需要独立 absolute background 和 high-effect specialist 才可能进入完整系统。

### 10.9 M10：竞赛冠军结构迁移为何失败

**输入处理**

- chemical view 三选一：2048-bit Morgan、1,200 维 ChemBERTa-MTR，或空的 NoChem；连续特征只用 outer-train mean/std 标准化。
- categorical context 在 outer-train 建 vocabulary，未知类别编号为 0，再由每列独立 embedding 编码。
- 每行还拼接 `2 x 4,422 = 8,844` 维 fold-safe context target statistics：逐蛋白 delta mean 与 scale。outer-train 的每个 chemical 使用 leave-current-chemical-out 统计；outer-valid 只使用全部 outer-train，缺失 context 回退到 outer-train global statistics。
- 目标 delta 同样只用 outer-train 的逐蛋白 mean/scale 标准化，scale 小于 `0.10` 或证据不足时置为 1。
- 同一个 fitted model 的 chemical derangement 只置换整药 chemical view，context、target statistics、missingness 和样本顺序保持不变，因此能直接检验网络是否真正读取药物表示。

这 8,844 维 context target statistics 是 M10-NoChem 仍能获得较高 pooled FC 的重要原因。它们按 chemical leave-out 构建，不属于直接标签泄漏，但本质上是很强的公共 context prior；因此必须用 Context PCC 和 chemical derangement 区分“公共模式拟合”与“药物特异泛化”。

**网络与 loss**

- OP3 LSTM/GRU：把 chemical + context flat vector 按 128 宽切成序列，经过 2 层 hidden-128 RNN，拼接所有时刻与 hidden state，再走 `1024 -> 512 -> 4422`。
- OP3 CNN：flat vector 进入多层 1D Conv/Pool，再走 `1024 -> 512 -> 4422`。
- MoA 3FC：BatchNorm + weight-normalized Linear，隐藏宽 2048，三层 dropout 为 `0.15/0.30/0.25`，直接输出 4,422 维 delta。
- Novel ADT2GEX：四个 512 宽 GELU block，最后输出 4,422；`M10.5` 去掉 terminal GELU 以允许 signed delta。
- OP3/MoA 类 loss 为 mask-aware 混合损失：MSE、log-cosh、MAE 和 sigmoid-target BCE；Novel 类使用 masked RMSE。

代表性的 `M10.3-Morgan` 实际输入与网络为：

```text
Morgan structure                                      [2,048]
context delta mean + scale        2 * 4,422 =         [8,844]
                                                        |
                                                        v
standardized numeric block                         [10,892]

8 categorical context columns
  -> each Embedding(...,8)
  -> 8 * 8 =                                             [64]

concat flat input = 10,892 + 64 =                    [10,956]
                                                        |
                                                        v
BatchNorm(10,956) -> Dropout(0.15)
                                                        |
                                                        v
weight-normalized Linear 10,956 -> 2,048 -> LeakyReLU
                                                        |
                                                        v
BatchNorm(2,048) -> Dropout(0.30)
                                                        |
                                                        v
Linear 2,048 -> 2,048 -> LeakyReLU
                                                        |
                                                        v
BatchNorm(2,048) -> Dropout(0.25)
                                                        |
                                                        v
weight-normalized Linear 2,048 -> 4,422
                                                        |
                                                        v
standardized predicted delta [4,422]
  -> target_scale * prediction + target_mean
  -> raw log2 delta [4,422]
```

对所有 observed 位置令 `e=prediction-target`，其精确混合 loss 为：

```text
MSE      = masked_mean(e^2)
MAE      = masked_mean(abs(e))
LOGCOSH  = masked_mean(log(cosh(e/3)))
BCE      = masked_mean(BCEWithLogits(prediction, sigmoid(target)))

L_mixed  = 0.8 * (0.4*MSE + 0.3*LOGCOSH + 0.3*MAE)
         + 0.2 * BCE
```

`M10.3` 使用 Adam，初始 lr `5e-3`、weight decay `1e-5`，OneCycle max lr `1e-2`，15 epochs，batch 256。这里保留 BCE 是对原 MoA 多标签冠军 loss 的 fidelity 迁移，不代表它天然适合 signed continuous delta；chemical derangement 结果最终否定了该直接迁移。

这些结构的主要失败不是优化不收敛，而是 chemical derangement/NoChem 对照表明 pooled FC 主要由 context target statistics 获得，正确药物表示没有贡献可靠增量。

## 11. 已完成消融实验

### 11.1 Response space

严格 S1 entity OOF：

| 模型 | response space | 解释能量 | FC PCC | Context PCC | High PCC | High F1 | 决策 |
|---|---|---:|---:|---:|---:|---:|---|
| `M2.0` | learned-64 | — | **0.358665** | 0.102352 | **0.630771** | **0.170422** | 主干 |
| `M2.20` | fixed SVD-16 | 0.3652 | 0.347793 | **0.108180** | 0.620258 | 0.153086 | 拒绝 |
| `M2.21` | fixed SVD-32 | 0.4361 | 0.350815 | 0.107578 | 0.622768 | 0.157051 | 拒绝 |
| `M2.22` | fixed SVD-64 | 0.5105 | 0.353585 | 0.106777 | 0.624643 | 0.160061 | 拒绝 |
| `M2.23` | fixed SVD-128 | 0.5913 | 0.357453 | 0.106655 | 0.627920 | 0.164309 | 取舍，不晋级 |

`M2.23 - M2.0` 的逐 chemical FC 差为 `-0.003175`，95% CI `[-0.005278,-0.001172]`；Context 差为 `+0.004934`，CI `[+0.000322,+0.009151]`。固定统计低秩改善 residual 一点，但损失整体与高效应能力。

### 11.2 Loss

| 改动 | FC 变化 | Context 变化 | High PCC 变化 | 结论 |
|---|---:|---:|---:|---|
| MSE -> MSE+MAE (`M2.30`) | +0.001478 | +0.005225 | -0.006556 | 不晋级 |
| MSE -> all Huber (`M2.31`) | +0.005292 | +0.006688 | -0.015980 | 仅少量融合 |
| `M6.11 + 10% Huber` | +0.000715 | +0.000812 | -0.001125 | 收益太小且护栏下降 |

### 11.3 Chemical representation

| 表示 | 关键证据 | 结论 |
|---|---|---|
| Morgan-512/RDKit | V0 S1 提高，但严格 S1 FC `0.346803 < 0.358665` | 直接拼接拒绝 |
| Morgan-2048/RDKit | 严格 S1 FC `0.340027` | 拒绝 |
| ChemBERTa real/shuffled | real `0.255660 > 0.220838`，有真实语义；仍远低于 M2.0 | 语义存在，融合方式失败 |
| OP3 RNA real/shuffled residual | M9.6-M9.7 FC/Context CI 均大于 0 | 最可信开放知识 additive signal |
| M10 direct transfer | 多个 normal 不优于 deranged/NoChem | 架构照搬拒绝 |

### 11.4 Calibration

旧 `V0-FROZEN` seed42：

| 模型 | S1 FC | S2 FC | S3 FC | time FC | S1 Abs R2 |
|---|---:|---:|---:|---:|---:|
| `M2.10` 完整 | 0.3496 | 0.3589 | 0.2605 | 0.5337 | 约 0.981 |
| `M2.40` no calibration | 0.156 | 0.226 | 0.148 | 0.286 | 约 0.881 |

Calibration 是巨大有效信号，但 plate 与 time/instrument/source 强混杂。去 plate 的 S2 FC 只提高 `+0.001862`，bootstrap CI 跨 0，且 Abs R2/High PCC 下降，因此当前保留受约束 Calibration，而不是直接删除。

### 11.5 PPI 图

| 模型 | S1 FC | S2 FC | S3 FC | time FC |
|---|---:|---:|---:|---:|
| `M3.0` 无图 | 0.139888 | 0.219494 | 0.124684 | 0.267701 |
| `M3.1` 真实 PPI | 0.140221 | 0.220541 | 0.125456 | 0.267489 |
| `M3.2` 保度重连 PPI | **0.140652** | **0.221412** | **0.126300** | **0.267789** |

真实 STRING 图没有击败 degree-preserving rewired 负对照，说明当前静态平滑不能归因于真实生物拓扑，不进入主模型。

### 11.6 Prototype 与专家

- OOF-safe chemical/strain response prototypes 曾带来局部 FC 提升，但损害 high-effect 护栏，未进入 M5.2。
- M7 strain ID expert 在 R10/R11/RT 带来约 `+0.08` FC，且 residual/high-effect 同向，是已见菌株插值的强证据。
- M7 chemical/pair experts 仍需在相同 universal parent、相同 joint update 预算和 nested scale 下完成确认；discovery 结果不能当冻结结论。

## 12. 当前每个场景应该用谁

| 目标 | 当前选择 | 理由 | 主要缺口 |
|---|---|---|---|
| 可交付完整 test 预测 | `M5.2` | 已完成 OOF、outer confirmation、全标签 refit、输出契约 | 路由仍是 split-level |
| S1 新药完整模型 | `M6.11` 3-seed | FC/High F1 稳定胜 M2 | Context PCC 只有 0.098239；无药物语义 |
| S2 新菌株完整模型 | `M5.1` 的 M2 路由 | M6 候选未过护栏 | 无 genome semantics |
| S3 双未知 | `M5.1` 的 M2 路由 | 新候选未稳定超过 | 两侧均缺语义，最困难 |
| Time/time-forward | `M6.21` 3-seed | 所有关键指标同步提高 | 更复杂条件外推仍需验证 |
| 已见菌株插值候选 | `M7.1` | R10/R11/RT 大幅提高且 gate 正确 | promotion gate 未完成 |
| 开放知识药物 signal | `M9.6` research component | real-vs-shuffled 归因成立 | 无 absolute、高效应弱、不可直接交付 |

## 13. 为什么还没有第二次 M2 级跃迁

`M1 -> M2` 同时增加了正确监督目标、合理分解、低秩共享和观测偏差校正，相当于重新定义了问题。M2 之后的大多数实验只是在相同信息上换 rank、loss 或交互模块，理论上很难再产生 `+0.1~0.3` 的跳跃。

更直接地说：

- 未见 chemical 的 one-hot 全部变成同一个零向量；没有结构/机制语义时，模型无法区分 11 个 test-only treatment chemicals。
- 未见 strain 同样缺少 genome/pathway 表示；CRD/DHY210 等映射身份仍未完全解除门禁。
- 数据的有效独立实体少、重复稀疏，公共 response 往往比大胆预测特异大效应更稳定。
- Calibration 已吃掉大量批次/测量结构，剩余 residual 更接近真正困难的生物特异信号。
- M9/M10 的 NoChem 与 derangement 实验证明，高 pooled FC 可以在几乎不读取药物身份的情况下出现，所以不能再把 pooled FC 单独当突破。

下一次大幅上升需要“新信息事件”：可核验 chemical mechanism/structure semantics、可核验 strain genome/pathway semantics、protein-conditioned shared decoder、以及 replicate aggregation/可靠性加权。单纯扩大 MLP 或继续微调 Huber/rank 不足以跨过当前瓶颈。

## 14. 下一步实验优先级

1. 完成 `M7.1` joint/U96 公平对照、outer-train nested scale、held-out entity bootstrap 和 promotion receipt；确认 `+0.08` 中有多少来自 `B_s`、多少来自 `R_s`。
2. 以同一 universal checkpoint 和更新预算确认 `M7.2/M7.3/M7.4`，分别报告 R01 Drug PCC、R11/RT 双 residual，禁止用 overall macro 掩盖 R00/R01 退化。
3. 把 `M9.6` correction 作为冻结的低权重 response component，与独立 absolute `B+C` 和 high-effect specialist 组合；先做 OOF 融合，不能直接看 outer 调权。
4. 解锁 M8 identity gate 后，按 real / zero / shuffled / random-remap 四臂测试 chemical 和 strain semantics；晋级要求 FC 与对应 residual 同时提高。
5. 引入 replicate-aware reliability weight 和 biological-group aggregation，优先验证是否提高 Context/Drug PCC 与 High F1，而不是只降低 RMSE。
6. 只有在上述输入信息有效后，再考虑 protein-conditioned decoder/ESM；必须包含 shuffled protein embedding 对照。

## 15. 限制与可审计来源

- 当前 scorer 是依据公开手册实现的本地 proxy，不是主办方可执行官方 scorer。
- 不同协议、seed 范围和 response-only/absolute 输出不能横向作单一排名。
- `M7.1` 表是 producer 快照，未完成最终 nested-scale/promotion gate。
- `M8` 的 verified-only 训练语义覆盖门禁尚未解除，不能宣称已有有效 genome/chemical semantic 模型。
- `M9/M10` 使用开放知识，必须与 closed-data 路线分榜；外部数据许可与最终提交契约仍需确认。

主要证据文件：

- [`docs/model_ledger.md`](../model_ledger.md)
- [`docs/model_registry.md`](../model_registry.md)
- [`src/goai_baseline/official_metrics.py`](../../src/goai_baseline/official_metrics.py)
- [`src/goai_baseline/features.py`](../../src/goai_baseline/features.py)
- [`src/goai_baseline/controls.py`](../../src/goai_baseline/controls.py)
- [`src/goai_response/features.py`](../../src/goai_response/features.py)
- [`src/goai_response/model.py`](../../src/goai_response/model.py)
- [`src/goai_response/train.py`](../../src/goai_response/train.py)
- [`configs/response_mvp.yaml`](../../configs/response_mvp.yaml)
- [`configs/experiments/response_conditional_concat_l256.yaml`](../../configs/experiments/response_conditional_concat_l256.yaml)
- [`configs/experiments/response_conditional_film_l256.yaml`](../../configs/experiments/response_conditional_film_l256.yaml)
- [`configs/experiments/m7_general_only.yaml`](../../configs/experiments/m7_general_only.yaml)
- [`configs/experiments/m7_strain_expert.yaml`](../../configs/experiments/m7_strain_expert.yaml)
- [`GOAI_model_comparison_report_20260814.html`](GOAI_model_comparison_report_20260814.html)
- [`goai-rna-transfer/RESULTS.md`](../../../goai-rna-transfer/RESULTS.md)
- [`goai-kaggle-model-transfer/README.md`](../../../goai-kaggle-model-transfer/README.md)
