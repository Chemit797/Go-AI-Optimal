# GOAI 模型编号表

编号只表达模型的技术血缘，不使用额外名称。

模型改动、数据版本、实验表现和决策详见 [model_ledger.md](model_ledger.md)。

## 编号结构

| 主编号 | 模型家族 |
|---|---|
| `M0` | 非神经网络参照 |
| `M1` | ConditionMLP 基线阶梯 |
| `M2` | ResponseDecompositionRegressor 响应分解家族 |
| `M3` | StaticProteinGraphRegressor 静态图家族 |
| `M4` | ConditionalProteinGraphRegressor 条件图家族 |
| `M5` | 融合与最终推理系统 |
| `M6` | Cell-conditioned ResponseDecompositionRegressor 家族 |
| `M7` | 通用生物主干 + fold-fit 已见实体残差专家家族 |
| `M8` | `M7` + 可迁移化合物/菌株开放知识语义家族 |
| `M9` | 独立 RNA 扰动预训练 → 蛋白响应迁移家族（开放知识） |
| `M10` | 相邻竞赛冠军架构直接迁移家族（开放知识） |

随机种子不产生新模型编号，写作 `M2.0-S42`、`M2.0-S43`、`M2.0-S2026`。配置文件名和运行目录保留为复现别名。

## M0：参照

| 编号 | 原方法 | 说明 |
|---|---|---|
| `M0.0` | B0 protein mean | 蛋白均值预测 |
| `M0.1` | B1 exact matched control | 精确匹配对照预测 |

## M1：ConditionMLP 基线

编号顺序就是特征逐步增加的顺序。

| 编号 | 原变体 | 相对上一级的变化 |
|---|---|---|
| `M1.0` | `p0_onehot` | 条件独热编码起点 |
| `M1.1` | `p1_priors` | `M1.0` + 全量统计先验 |
| `M1.2` | `p1_oof_priors` | 将 `M1.1` 的训练行改为 leave-one-row-out 无自包含先验 |
| `M1.3` | `p2_crosses` | `M1.1` + 条件交叉项；不包含 `M1.2` 的 OOF 训练变换 |
| `M1.4` | `p3_time` | `M1.3` + 时间结构 |
| `M1.5` | `p4_hash` | `M1.4` + 化学哈希特征 |

## M2：响应分解家族

`M2.0` 是公共主干。第二段按实验轴分组：`1x` 化学特征、`2x` 响应空间、`3x` 损失函数、`4x` 校准、`5x` PPI。

| 编号 | 原配置/变体 | 相对 `M2.0` 的变化 |
|---|---|---|
| `M2.0` | `response_space_learned64`、`response_mvp_no_chemistry`、`chemistry_s1_none`、`response_loss_mse_control` | Learned-64 + MSE + 校准开启 + 无化学/PPI；当前主干 |
| `M2.10` | `response_mvp`、`chemistry_s1_morgan512` | 化学轴：Morgan 512 |
| `M2.11` | `chemistry_s1_morgan2048` | 化学轴：Morgan 2048 |
| `M2.12` | `response_chemberta_real` | 化学轴：frozen ChemBERTa-77M-MLM 384 维 |
| `M2.13` | `response_chemberta_shuffled` | 化学轴：打乱 ChemBERTa 行标签的负对照 |
| `M2.14` | `response_morgan_chemberta` | 化学轴：Morgan-512 + ChemBERTa |
| `M2.20` | `response_space_svd16` | 响应空间轴：固定 SVD-16 |
| `M2.21` | `response_space_svd32` | 响应空间轴：固定 SVD-32 |
| `M2.22` | `response_space_svd64` | 响应空间轴：固定 SVD-64 |
| `M2.23` | `response_space_svd128` | 响应空间轴：固定 SVD-128 |
| `M2.24` | `response_space_learned32` | 响应空间轴：learned rank 32 |
| `M2.25` | `response_space_learned128` | 响应空间轴：learned rank 128 |
| `M2.26` | `response_space_learned256` | 响应空间轴：learned rank 256 |
| `M2.27` | `response_space_svd256` | 响应空间轴：固定 SVD-256 |
| `M2.30` | `response_loss_fc_msemae` | 损失轴：FC 使用 MSE+MAE |
| `M2.31` | `response_loss_huber_all` | 损失轴：全部使用 Huber |
| `M2.40` | `response_mvp_no_calibration` | 校准轴：关闭低秩校准 |
| `M2.41` | `response_calibration_rank4` | 校准轴：rank 4 |
| `M2.42` | `response_calibration_rank8` | 校准轴：rank 8 |
| `M2.43` | `response_calibration_no_plate` | 校准轴：移除 plate 特征 |
| `M2.44` | `response_calibration_plate_dropout03` | 校准轴：plate dropout 0.3 |
| `M2.45` | `response_calibration_plate_dropout05` | 校准轴：plate dropout 0.5 |
| `M2.46` | `response_calibration_plate_shuffle` | 校准轴：打乱 plate 的负对照 |
| `M2.47` | `response_calibration_regularized` | 校准轴：加强校准正则 |
| `M2.48` | `response_calibration_controls_only` | 校准轴：只让 control 更新校准分支 |
| `M2.50` | `response_mvp_ppi_real` | PPI 轴：真实 PPI 正则 |
| `M2.51` | `response_mvp_ppi_rewired` | PPI 轴：保度重连负对照 |
| `M2.60` | `response_prior_chemical` | OOF-safe response prior：chemical 均值 |
| `M2.61` | `response_prior_strain` | OOF-safe response prior：strain 均值 |
| `M2.62` | `response_prior_both` | OOF-safe response prior：chemical + strain |
| `M2.63` | `response_prior_both_scaled` | `M2.62` + 可学习 prior scale |

## M3：静态图家族

末位统一表示图类型：`0` 无图对照、`1` 真实 PPI、`2` 重连 PPI。

| 编号 | 图变体 | 说明 |
|---|---|---|
| `M3.0` | `no_graph` | 单位邻接对照 |
| `M3.1` | `real_ppi` | 真实 STRING PPI 图 |
| `M3.2` | `rewired_ppi` | 保度重连负对照 |

## M4：条件图家族

中间位表示是否做节点扰动注入：`0` 不注入，`1` 注入。末位仍表示图类型：`0` 无图、`1` 真实 PPI、`2` 重连 PPI。

| 编号 | 图变体 | 节点注入 |
|---|---|---|
| `M4.0.0` | `no_graph` | 否 |
| `M4.0.1` | `real_ppi` | 否 |
| `M4.0.2` | `rewired_ppi` | 否 |
| `M4.1.0` | `no_graph` | 是 |
| `M4.1.1` | `real_ppi` | 是 |
| `M4.1.2` | `rewired_ppi` | 是 |

## M5：融合与最终系统

| 编号 | 组成 | 说明 |
|---|---|---|
| `M5.0` | `M2.0` + `M2.31` | 按 OOD 场景使用不同 MSE/Huber 权重的融合规则 |
| `M5.1` | 3×`M2.0` + 3×`M2.31` + `M5.0` | 三随机种子平均后进行场景融合；上一版冻结模型 |
| `M5.2` | `M5.1` + 3×`M6.11` + 3×`M6.21` | S1 路由到 concat-256，time 路由到 FiLM-256；S2/S3 保留 M5.1；当前冻结模型 |

当前冻结模型对外只写：**GOAI-M5.2**。

## 非模型诊断记录

| 编号 | 对象 | 说明 |
|---|---|---|
| `AUDIT-P01` | `M5.2` FC 平台期 | 未知 chemical/strain 全零回退、FC 信噪比、重复一致性和 test chemical 距离审计；不产生 checkpoint，不改变模型编号 |

## M6：cell-conditioned response

十位表示交互结构：`0x` 为早期 rank-64 结构对照，`1x` 为 concat 扩秩/损失轴，`2x` 为 FiLM 扩秩/损失轴。

| 编号 | 父模型 | 交互与 rank | 损失 |
|---|---|---|---|
| `M6.0` | `M2.0` | shared concat，rank 64 | MSE |
| `M6.1` | `M2.0` | shared gate，rank 64 | MSE |
| `M6.2` | `M2.0` | shared FiLM，rank 64 | MSE |
| `M6.10` | `M6.0` | shared concat，rank 128 | MSE |
| `M6.11` | `M6.10` | shared concat，rank 256 | MSE |
| `M6.12` | `M6.0` | shared concat，rank 64 | 全 Huber |
| `M6.13` | `M6.11` | shared concat，rank 256 | 全 Huber |
| `M6.20` | `M6.2` | shared FiLM，rank 128 | MSE |
| `M6.21` | `M6.20` | shared FiLM，rank 256 | MSE |
| `M6.22` | `M6.2` | shared FiLM，rank 64 | 全 Huber |
| `M6.23` | `M6.20` | shared FiLM，rank 128 | 全 Huber |

## M7：通用模型 + 已见实体专家（封闭数据）

`M7` 不再按 `split_final` 选择整套网络。每个 checkpoint 根据实际 fit rows
生成 strain、chemical 和 pair support vocabulary，并逐行硬门控低秩残差专家。
所有 response 组件共用同一个 4,422 蛋白 decoder；Calibration 仍是只读取
`data_source/instrument/plate` 的独立观测分支。

| 编号 | 父模型 | 唯一新增因素 |
|---|---|---|
| `M7.0` | `M6` 结构重构 | 共享 universal biological trunk、逐行 support router、受约束 Calibration；无实体专家 |
| `M7.1` | `M7.0` | 已见 strain 的 background/response 低秩残差专家 |
| `M7.2` | `M7.0` | 已见 chemical 的低秩残差专家 |
| `M7.3` | `M7.0` | 同时启用 strain 与 chemical 专家 |
| `M7.4` | `M7.3` | 仅在 exact strain×chemical pair 存在于 fit rows 时启用 pair/time 残差专家 |

专家缩放不是新的模型血缘；写作 `M7.3-Gs0.5-Gc0.75`，候选值只允许
`0/0.25/0.5/0.75/1`，并且必须由 inner OOF 冻结。

fold-safe chemical/strain prototype 是单独的 `research-prior-*` 研究组件，不属于
`M7.2/M7.3/M7.4` 编号，也不能直接晋级。它先按训练折（训练行使用 leave-one-row-out）
计算，再投影到共享 response decoder 的低秩空间；必须与 `prior=none` 单独对照。

## M8：开放知识实体语义

`M8` 与 `M7` 的配置、产物和结论隔离。高可信候选或 proxy 未通过身份门禁时，
只能标记为研究性 screen，不能进入冻结提交模型。

| 编号 | 父模型 | 唯一新增因素 |
|---|---|---|
| `M8.0` | `M7.3` | 化合物语义；Morgan/RDKit、frozen ChemBERTa、融合及 shuffled/zero 对照 |
| `M8.1` | `M7.3` | 1,011 酵母 SNP-MDS/菌株元数据语义及 shuffled/zero 对照 |
| `M8.2` | `M7.3` | 最佳化合物语义 + 最佳菌株语义 |
| `M8.3` | `M7.4` | 双实体语义 + exact pair/time 专家 |

当前编号仅表示已实现候选架构；是否晋级必须以 R00/R10/R01/R11/RT 的本地
严格 OOF、多 seed、实体 bootstrap 和 high-effect 护栏决定，不能由单次 screen 决定。

## M9：独立 RNA → protein 迁移（开放知识）

`M9` 是与 `M0–M8` 实现完全隔离的 greenfield 架构。它只读取冻结 S1
fold/data/metric 契约；不导入旧模型、checkpoint、预测或 teacher 特征。所有变体使用
同一独立 context×chemical 蛋白 delta consumer，区别仅为化学 encoder 初始化。

| 编号 | 父模型 | 唯一因素 |
|---|---|---|
| `M9.0` | 无 | context-only；化学输入恒为零 |
| `M9.1` | `M9.0` | Morgan-2048 encoder 从 GOAI fold-train 随机初始化 |
| `M9.2` | `M9.1` | OP3 `logFC` 预训练 encoder；全局排除 GOAI parent 后微调 |
| `M9.3` | `M9.2` | OP3 whole-drug input-shuffle 负对照 |
| `M9.4` | `M9.1` | L1000FWD PCA-64 encoder；全局排除 GOAI parent 后微调 |
| `M9.5` | `M9.4` | L1000FWD whole-parent input-shuffle 负对照 |
| `M9.6` | `M9.0` + `M9.2` | 冻结 context-only 主干 + zero-init OP3 real residual gate；固定 scale 0.20 |
| `M9.7` | `M9.6` | residual gate 的 OP3 whole-drug shuffle 负对照 |

`M9.0–M9.5` 为 direct-consumer discovery / rejected controls；`M9.6/M9.7`
完成 seeds 42/43/2026，保留为 research response component，尚非完整 absolute model。
它们均不改变当前冻结模型。
`M9` 输出 response delta，不能用观测 validation control 重构后声称 absolute fidelity。

## M10：相邻竞赛冠军架构迁移（开放知识）

`M10` 是独立 sibling 实验室中的 greenfield 架构家族，不导入 `M0–M9` 的模型、
checkpoint、预测或 teacher 特征。它只读取冻结 S1 数据/折分/评分契约，并按原竞赛
公开源码先复刻结构与训练参数，再对 GOAI 的有符号、缺失蛋白 delta 做最小必要适配。
化学输入写作后缀 `-Morgan`、`-MTR` 或 `-NoChem`；随机种子仍写作 `-S<seed>`。

| 编号 | 公开来源 | 唯一架构因素 |
|---|---|---|
| `M10.0` | Open Problems 2023 rank-1 | 两层 LSTM-128 + `1024→512` direct head |
| `M10.1` | Open Problems 2023 rank-1 | 两层 GRU-128 + `1024→512` direct head |
| `M10.2` | Open Problems 2023 rank-1 | 原 1D-CNN stack + `1024→512` direct head |
| `M10.3` | MoA 2020 rank-1 | first-stage 3FC-2048，BatchNorm/weight norm/OneCycle |
| `M10.4` | NeurIPS 2021 ADT→GEX rank-1 | 四层 512-GELU + terminal GELU fidelity head |
| `M10.5` | `M10.4` | 仅把 terminal GELU 改为 linear signed-delta head |
| `M10.6` | `M10.0–M10.5` | 只在完整 OOF 后确定的跨架构融合候选 |
| `M10.7` | `M10.3` | same-fold trunk warm-start + reset residual head + full fine-tune；formal S42 已拒绝 |
| `M10.8` | `M10.3` | same-fold trunk frozen + reset residual head，仅 head `weight_g/weight_v` 可训练；formal S42 已拒绝 |

`M10` 的 outer S1 fold 只在固定 epoch 结束后评分一次，不参与 checkpoint/epoch 选择；
训练行的 context target statistics 对当前 chemical 留一，验证行只用 outer-train。
每个 fit 必须同时保存同模型的 whole-chemical derangement 预测，并与 `-NoChem` 对照。

`M10.7` 每个 outer fold 从对应 same-fold `M10.3` outer-train checkpoint 读取
trunk，丢弃并重新初始化 residual head，再用 chemical-contrast
context-residual objective 全量 fine-tune trunk 与 head。formal S42 未击败
NoChem/PermChem 控制，已拒绝；不扩 seed、不进入 `M10.6`。

`M10.8` 是对 `M10.7` 负结果的归因诊断：复用相同 same-fold `M10.3`
trunk，冻结所有 trunk 参数与 BatchNorm running statistics，reset residual head
后仅允许 weight-normalized head 的 `weight_g/weight_v` 更新。`-NoChem` 和 same-fit
`-PermChem` 仍是必须匹配控制。formal S42 未击败两项控制，已拒绝；
不扩 seed、不进入 `M10.6`、不覆写 `M10.3/M10.7` 产物，也不改变
当前冻结 `GOAI-M5.2`。

## M11：M6 absolute 主干 + M9 chemical response

`M11` 把独立的 M9 response producer 接入能输出 absolute abundance 的 M6 系统，
并按全量 refit 的 canonical support 逐行路由。模型只在 R10（菌株已见、药物未知）
启用 M9 response；其他 regime 保留冻结 `M5.2`。

| 编号 | 父模型 | 唯一因素 | 状态 |
|---|---|---|---|
| `M11.0` | `M5.2 + M6.11 + M9.6` | `B6+C6-0.05*R6+1.05*R9.6`，逐行 R10 router | 冻结历史候选 |

## M12：M11 后续融合、专家与双实体语义

| 编号 | 父模型 | 唯一因素 | 状态 |
|---|---|---|---|
| `M12.0` | `M11.0` | 在 `abs(R6)>=0.5` 位置以 0.15 比例把 M9-heavy response 拉回 R6 | 当前本地最优完整候选 |
| `M12.1` | `M2/M12.0` | 新菌株 scaled SNP-MDS-4 强收缩语义残差 | 三种子 OOF 拒绝 |
| `M12.2` | `M12.0` | R00 chemical + strain semantics；RR/SR/RS/SS/ZZ 对照 | 完成，未击败 zero/M2，拒绝 |

`M12.0` 的固定公式为：

```text
blend = (1 - 1.075) * R6 + 1.075 * R9.6
R12   = blend + 0.15 * I(abs(R6) >= 0.5) * (R6 - blend)
y_hat = B6 + C6 + R12
```

另外完成的 seen-strain expert overlay 不是新模型编号。它按
`M12.0 + alpha*(expert-general)` 评估，最佳 `alpha=0`，因此不进入血缘。
详细证据见 `docs/experiments/m12_execution_results_20260815.md`。

## 版本规则

- 只改变随机种子：在编号后加 `-S<seed>`，主编号不变。
- 同一实验轴增加变体：在所属十位分组内顺延，例如新的损失变体为 `M2.32`。
- 新的基础架构：启用新的主编号。
- 最终融合组件或权重发生实质变化：从 `M5.1` 顺延为 `M5.2`。
