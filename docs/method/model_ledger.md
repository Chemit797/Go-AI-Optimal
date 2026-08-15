# GOAI 模型台账

> 最后更新：2026-08-15
> 当前冻结模型：`GOAI-M11.0`
> 冻结回退模型：`GOAI-M5.2`
> 当前长期执行路线：[GOAI_continuous_modeling_roadmap_20260814.md](plans/GOAI_continuous_modeling_roadmap_20260814.md)
> 规范编号见：[model_registry.md](model_registry.md)

本文件是项目唯一的模型事实台账。它既记录当前模型，也保留失败实验和历史结果。以后只要训练、评估、修改或融合模型，都必须在本文件追加记录；只保存 checkpoint 或口头汇报不算完成一次迭代。

## 1. 记录纪律

每次迭代至少记录以下信息：

1. 模型编号、父模型、日期、状态和实验假设；
2. 相对父模型调整了什么，以及哪些关键设置保持不变；
3. 训练数据范围、样本数、蛋白数、文件 SHA256、外部资源版本；
4. 验证协议、fold seed、模型 seed、OOF/outer 边界和防泄漏措施；
5. 各场景的样本覆盖和表现，包括 FC PCC、absolute sample R2、high-effect PCC/F1；
6. 相对父模型的差值、随机种子波动或置信区间；
7. 配置、代码、运行目录、checkpoint 和预测文件路径；
8. 晋级、保留、拒绝或仅作对照的结论及理由；
9. 官方分数；若未提交，必须明确写“无”，不能用本地代理冒充。

同一结果只能在相同数据边界和验证协议内比较。旧冻结验证、inner OOF、outer confirmation、全标注 refit 和官方 test 分数必须分开记录。

## 2. 数据版本

### D0：官方发布训练/冻结验证数据

| 项目 | 数值 |
|---|---:|
| 已发布有标签样本 | 8,958 |
| `split_final=train` | 5,920 |
| `val_chem_only` | 1,065 |
| `val_strain_only` | 1,547 |
| `val_both` | 269 |
| `val_time` | 157 |
| 原始蛋白列 | 5,243 |
| 训练行缺失率 `<0.80` 后保留蛋白 | 4,422 |
| 目标尺度 | raw intensity 取 `log2` |

| 文件 | 大小（bytes） | SHA256 |
|---|---:|---|
| `WAYB_WAYC_metadata_train_val.csv` | 904,646 | `9414f22d71e925a3b85544b49fde252613c87808d34738a84785003adb8131ef` |
| `WAYB_WAYC_proteome_raw_train_val.csv` | 289,769,736 | `a15d9a40a6ad4e8e84a4ce4ed08644fce78780d31ace5561928517c4a5fa7ccb` |
| `WAYB_WAYC_metadata_test.csv` | 461,272 | `42f2df9ea79f28da8344e96b5181edacc215744a858d1a4eaa729c2e1cc69d31` |

### D0-Chem：开放化学身份表

| 文件 | SHA256 | 用途 |
|---|---|---|
| `data/processed/chemical_entity_map.tsv` | `1ef08ad74ff807d6e85da4d4a00b06a083f4424bd281a81b49589f47dd754df6` | 名称到 PubChem/SMILES，生成 Morgan 指纹和 RDKit 描述符 |

上表为 `M2/M5` 历史实验使用的 v1 快照，不得用新文件哈希覆盖它。`M8`
开始使用下面的 `D3-ENTITY-20260813` 快照。

### D3-ENTITY-20260813-R2：M7/M8 实体、语义与 support 契约

本表为完成 PubChem + ChEBI/ChEMBL 二次化学审计、Peter/ENA/NCBI 菌株证据审计后的
当前 R2 快照。首批 `core-v2` discovery 使用此前 R1 registry/support；其真实 contract
继续随历史产物保存，但不得在 R2 源码或映射下续跑。

| 文件 | SHA256 | 用途/边界 |
|---|---|---|
| `data/processed/chemical_entity_map.tsv` | `928ebf5698b42c873a092672613d224aace4f94c4268b103220a13d6043e6b78` | 57 个 raw names 的 exact-default 结构视图；修正 LY294002 hydrochloride 盐型与 Hoechst 33258 错 CID |
| `data/processed/entities/chemical_registry.tsv` | `bfe670ce1c8b6d42291ccb106dbfec4b2264fc62868ef2675e538e91718b88bc` | canonical chemical identity/provenance；11 个 test-only 中 8 verified、3 candidate；另有 3 个 formulation/mixture proxy |
| `data/processed/entities/strain_registry.tsv` | `eec1a5353c0cf956f0ab27b1ec202a950acab5d74e9fdf8b71cf52b60c98ff2a` | 6 个菌株的 accession/evidence provenance；五个自然株仍为 candidate，DHY210 unresolved |
| `data/processed/entities/registry_manifest.json` | `dec6e32508352ac9c09664f92ef5bc5959d72c6412e6705288c569498d307867` | registry 文件、semantic hash、support content hash 总收据 |
| `data/processed/entities/strain_semantics_numeric.tsv` | `8ee0f31d33c8dccefa90eed33cb7f4eb949470ef6dfe0681beefecf52b7d72ce` | Peter 2018/1,011 yeast 的 32 维 SNP-MDS + 元数据 + status flags |
| `data/processed/entities/strain_semantics_shuffled.tsv` | `7f1a8d2a28f70d033a42416d59da9ac2e27022aed79b40926745f51cc0d096f7` | 菌株语义负对照，shuffle seed 991 |
| `data/processed/entities/strain_semantics_manifest.json` | `ba54ed7ccddd6d34663978b698ac7da39f13bd41e84937be8edbd31fbf795da2` | 语义数值、shuffle、Peter/ENA/NCBI 证据链 |
| `data/processed/entities/chemical_parent_normalized_views.tsv` | `c9c0c24d019748df667a7c6050e67773771752038db32a262c499b2855010b78` | 只允许显式 opt-in 的 parent/component 消融 |
| `data/processed/chemical_views/chemical_entity_map_exact.tsv` | `57bc9fde4a519a32d4d6c7fe13511e337aaf1383f14383ba7b100e38248bfd91` | exact 结构视图 |
| `data/processed/chemical_views/chemical_entity_map_parent_normalized.tsv` | `4f5ac8db3b150fdee025f9cd9fdbafd7371a6a4fbb5deac7df771251e6473c35` | parent/component 研究性消融，不可默认晋级 |
| `resources/entities/chemical_identity_risk_review.tsv` | `fbc2bb05b8bfe2add123c068b206bdd1d0c3492db9ab1488aae410b3ad856a3d` | zero-risky 的独立 7 实体审核契约 |
| `data/processed/chemical_views/chemical_entity_map_zero_risky.tsv` | `61da177d72d60344aa4fbabc8b4f200b7082bac8fc57554b0d3c548ec345850a` | 将 5 个配方/混合物风险与 G418/Hygromycin B 两个立体冲突共 7 项置零 |
| `data/processed/entities/support_manifest_fit_train.json` | `6f20f5a7e47b320e5237103f84b5292674e433bb6517738cf3d9a81f4f30dd54` | 5,920 行 fit support；content hash `e9983117...65b` |
| `data/processed/entities/support_manifest_fit_all_labeled.json` | `364ea2260c274d88d2d942bf32832732ce3c5cfa481261a27f0aa575a728b425` | 8,958 行全标注 refit support；content hash `98692514...e17` |
| `data/processed/chemical_embeddings/chemberta_77m_mlm/manifest.json` | `cb64148b66d58eff94881bfc241e85725d6a6bfe5dcb41713484b5d48e608f96` | corrected exact mapping → frozen ChemBERTa TSV 收据 |
| `data/processed/chemical_embeddings/chemberta_77m_mlm_parent/manifest.json` | `9e60ad7c1705902e796e4a17cc7abfe9c16cc7f0150321feb7620fdf520a718a` | corrected parent mapping → frozen ChemBERTa TSV 收据 |
| `data/processed/chemical_embeddings/chemberta_77m_mlm_zero_risky/manifest.json` | `05a051a044bedbab3d41c94ce7837d1952b54ba4f784bae77a9ca571ac8fab10` | 7 个 identity-risk 置零后的 frozen ChemBERTa 收据 |
| `data/processed/entities/audit_report.json` | `1173a387ae31909ab1e43f4ccc95368bbe33d48fa00b711385bb72ebdf261733` | 默认身份审计；只因 DHY210 unresolved 返回非零，不影响 M7 ID 专家 |
| `data/processed/entities/audit_report_strict.json` | `bac3ed6ff492759154915a3aef9ddf78bb52ea2957def142497c294403c69d29` | 正式开放语义门禁；完整保留 candidate/proxy/DHY210 blockers |

外部菌株资源快照：Peter et al. 2018 Table S1 SHA256
`b11de13b50bcf91bb2a40cdbd0f2f35372bdef6fc9a018d7538cf5e5eea7f273`；1,011 SNP distance
matrix SHA256 `140da4e5193584c01e60c554a2ba5075a542d925be540afe7c7a92b7377af928`。
五个自然株只标记 `HIGH_CONFIDENCE_CANDIDATE`；Peter/ENA/NCBI 能证明公开 code/isolate
链，但赛事材料没有证明 GOAI code 就是这些公开株。11 个 test-only chemicals 已完成
PubChem + ChEBI/ChEMBL 二次审计：8 verified；Doxycycline hyclate、G418、Hygromycin B
因计量/立体冲突仍为 candidate；Hoechst 33258、Oligomycin、Tunicamycin 仍是显式 proxy。
因此 `M8` 只允许带 real/shuffled/zero 与 proxy gate 的研究性本地 screen。
需要特别区分“11 个 test-only 药物中 8 个 verified”和“训练侧有可学的
verified semantics”：当前 `role=train` 的 chemical 与 strain 在 `verified_only`
政策下可接纳语义实体均为 0。因而正式 `M8.0–M8.3` confirmation 必须机器
阻断，否则所谓 M8 只是一个带全零语义块的 M7。研究性 quick screen 显式使用
`research_allow_candidate` 且 `promotion_eligible=false`，不得据此进入冻结模型。

R3 只读预审已确认：36 个训练侧 chemical candidates 中，28 个（含 26 个
treatment）具有 PubChem 与 ChEBI/ChEMBL 同键且名称/配方无冲突的保守升级路径，
足以在新 registry 版本中解锁 formal `M8.0`。本轮不修改 R2，避免使已启动的
run-contract 与 artifact chain 漂移。边界项、证据落盘要求与首选第二来源详见
`docs/audit/train_chemical_verification_plan_20260813.md`。

### D1：最终全标注 refit 数据

模型选择冻结后，`M5.1` 的六个组件分别用 D0 的全部 8,958 个有标签样本重新拟合；每个组件包含 7,884 个 exact-control FC 训练配对。该阶段不再产生可用于选择模型的验证分数。

### D2：`M5.2` 新路由 refit 数据

`M6.11-S42/S43/S2026` 与 `M6.21-S42/S43/S2026` 同样使用全部 8,958 个已发布有标签样本重新拟合；每个 checkpoint 都在文件内记录 `fit_scope=all_released_labeled_rows`、`fit_sample_count=8958`。`M5.2` 未重新拟合被保留的 S2/S3 路由，而是逐值复用已冻结 `M5.1` 预测。

### DT：官方 test metadata

| 场景 | 行数 |
|---|---:|
| `test_chem_only` | 1,640 |
| `test_strain_only` | 1,534 |
| `test_both` | 1,129 |
| `test_time` | 151 |
| 合计 | 4,454 |

## 3. 评估协议

| 协议 | 数据边界 | 用途 |
|---|---|---|
| `V0-FROZEN` | 用 5,920 个 `train` 样本拟合，在四个官方冻结 validation split 评估 | 历史模型与最终 outer confirmation |
| `V1-ENTITY-OOF` | 每折重新拟合特征、标准化、matched control 和模型；验证实体不进入训练状态 | 当前模型选择主协议 |
| `V1-TIME-FORWARD` | 同一上下文只用较早时间训练，最后时间点验证 | 更接近真实时间外推 |
| `R1-ALL-LABELED` | 用全部 8,958 个已发布标签 refit | 最终 test 推理；不得再把训练内分数当验证结果 |
| `OFFICIAL` | 主办方隐藏 test 评分 | 当前没有提交记录或官方分数 |

主要指标是 matched-control `FC PCC`。同时必须检查 `absolute sample R2`、`high-effect PCC` 和 `high-effect F1`，避免只提高相关性却破坏效应幅度。所有下表均为本地代理结果，不是官方排行榜分数。

## 4. 模型总表

| 编号 | 父模型 | 核心变化 | 当前状态 |
|---|---|---|---|
| `M0.0` | 无 | 逐蛋白训练均值 | 参照 |
| `M0.1` | 无 | exact matched control | 诊断参照，不能直接预测 test |
| `M1.0` | 无 | 条件 one-hot MLP | 历史基线 |
| `M1.1` | `M1.0` | 加全训练统计先验 | 历史基线，有训练特征泄漏风险 |
| `M1.2` | `M1.1` | 训练行改为 leave-one-row-out 先验 | 可信的统计先验对照 |
| `M1.3` | `M1.1` | 加条件交叉项 | 历史消融；不是从 `M1.2` 分支 |
| `M1.4` | `M1.3` | 加时间 sin/cos | 历史消融 |
| `M1.5` | `M1.4` | 加 32 维化学名称哈希 | 历史消融 |
| `M2.0` | 新架构 | Learned-64 响应分解、MSE、校准、无化学/PPI | 当前最终系统的 MSE 组件 |
| `M2.10` | `M2.0` | 加 Morgan-512 和 7 个 RDKit 描述符 | 旧 MVP 主模型；严格 S1 OOF 未晋级 |
| `M2.11` | `M2.0` | 加 Morgan-2048 和描述符 | 未晋级 |
| `M2.12`–`M2.14` | `M2.0` | ChemBERTa real/shuffled 负对照及 Morgan+ChemBERTa | real 有语义信号，但直接拼接整体拒绝 |
| `M2.20`–`M2.23` | `M2.0` | Learned-64 改为固定 SVD 16/32/64/128 | 未晋级 |
| `M2.24`–`M2.27` | `M2.0` | learned 32/128/256 与 fixed SVD-256 | learned 扩秩保留证据；fixed SVD-256 拒绝 |
| `M2.30` | `M2.0` | FC loss 改 MSE+MAE | 未晋级 |
| `M2.31` | `M2.0` | 三个 loss 全改 Huber | 仅作为融合辅助组件 |
| `M2.40` | `M2.10` | 关闭 observation calibration | 拒绝 |
| `M2.41`–`M2.48` | `M2.0` | calibration rank、去 plate、dropout、shuffle、正则、controls-only 审计 | 诊断保留；无变体进入最终路由 |
| `M2.50` | `M2.10` | 加真实 PPI 平滑正则 | 对照，未晋级 |
| `M2.51` | `M2.10` | 加保度重连 PPI 正则 | 负对照 |
| `M2.60`–`M2.63` | `M2.0` | OOF-safe chemical/strain response prototypes | 部分场景 FC 上升但破坏 high-effect；拒绝 |
| `M3.0`–`M3.2` | 新架构 | 静态低秩蛋白图：无图/真实/重连 | 未晋级 |
| `M4.*` | `M3.*` | 条件化图和可选节点扰动注入 | smoke 实验，未晋级 |
| `M5.0` | `M2.0` + `M2.31` | 按 OOD 场景冻结融合权重 | 已确认融合规则 |
| `M5.1` | `M5.0` | 两家族各 3 seed 先平均再融合 | 上一版冻结模型；仍提供 S2/S3 路由 |
| `M6.0`–`M6.23` | `M2.0` | response 显式读取 cell state；concat/gate/FiLM、rank 与 loss 消融 | `M6.11` 和 `M6.21` 晋级 |
| `M7.0`–`M7.4` | `M6` 结构重构 | 共享 universal trunk + fold-fit strain/chemical/pair 残差专家 + 逐行 support router | 已实现；本地严格 OOF 筛选中 |
| `M8.0`–`M8.3` | `M7.3/M7.4` | 可迁移 chemical/strain semantics 及 real/shuffled/zero/proxy 对照 | 已实现候选；identity promotion gate 未解除 |
| `M5.2` | `M5.1` + `M6.11` + `M6.21` | S1/time 场景路由替换，S2/S3 保留 | 冻结回退模型；未覆盖 |
| `M9.0`–`M9.7` | 独立开放知识路线 | OP3/L1000 RNA 预训练 chemical encoder；`M9.6` 为冻结 context 主干上的 OP3 residual | `M9.6` 已作为 R10 response 进入 `M11.0` |
| `M10.0`–`M10.8` | 独立冠军迁移路线 | OP3/MoA/ADT2GEX 结构与 ChemBERTa/Morgan 消融 | 全部未过 chemical-sensitivity 门禁 |
| `M11.0` | `M5.2` + `M6/M9.6` | 逐行 support router；R10 使用三种子 M6/M9.6 response 融合，其余保留 `M5.2` | **当前冻结主提交候选** |
| `M11.2` | `M11.0` + strain semantics | R00 使用语义 Background 与 M6/M9 response | 研究候选；无正 R00 OOF 证据，不作主提交 |

## 5. M0/M1：参照与 MLP 基线

共同条件：D0、`V0-FROZEN`、seed 42（可训练模型）、`configs/baseline.yaml`。数值为 `absolute sample R2 / FC PCC`。

| 模型 | S1 新化合物 | S2 新菌株 | S3 双重未知 | time |
|---|---:|---:|---:|---:|
| `M0.0` | 0.860 / 0.152 | 0.906 / 0.181 | 0.862 / 0.146 | 0.902 / 0.164 |
| `M0.1` | 0.986 / 不可定义 | 0.984 / 不可定义 | 0.986 / 不可定义 | 0.984 / 不可定义 |
| `M1.0` | -0.146 / 0.124 | -0.499 / 0.133 | -4.078 / 0.083 | 0.858 / 0.127 |
| `M1.1` | 0.816 / 0.137 | 0.801 / 0.162 | 0.807 / 0.131 | 0.800 / 0.147 |
| `M1.2` | 0.828 / 0.137 | 0.819 / 0.162 | 0.822 / 0.131 | 0.819 / 0.147 |
| `M1.3` | 0.835 / 0.140 | 0.880 / 0.162 | 0.839 / 0.134 | 0.873 / 0.148 |
| `M1.4` | 0.804 / 0.139 | 0.873 / 0.160 | 0.797 / 0.134 | 0.871 / 0.149 |
| `M1.5` | 0.847 / 0.135 | 0.864 / 0.158 | 0.847 / 0.130 | 0.863 / 0.145 |

结论：MLP 阶梯主要恢复绝对背景，没有一个版本在 FC PCC 上稳定超过 `M0.0`。`M1.2` 修复了 `M1.1` 的训练行先验泄漏，绝对 R2 小幅改善但 FC 不变，因此只保留为可信对照。

证据：`docs/experiments/official_evaluation_v1.md`、`docs/experiments/oof_priors_v4.md`。

## 6. M2：响应分解家族

共同架构为：

```text
log2 proteome
= background(strain, medium, temperature, time)
+ calibration(data_source, instrument, plate)
+ I[treatment] * response(condition, optional chemistry)
```

背景、绝对值和 matched-control FC 三项 loss 在逐蛋白标准化空间训练；默认 response rank=64、calibration rank=16、hidden=128。

### 6.1 化学特征轴：严格 S1 entity OOF

共同条件：D0 的训练侧、`V1-ENTITY-OOF` S1 四折、fold seed 42、model seed 42。数值为四折均值。

| 模型 | 改动 | log2 RMSE | abs R2 | FC PCC | context PCC | high PCC | high F1 | 决策 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `M2.0` | 无结构主干 | 0.431284 | 0.979152 | **0.358665** | 0.102352 | **0.630771** | **0.170422** | 主干 |
| `M2.10` | Morgan-512 + 描述符 | 0.431798 | 0.979501 | 0.346803 | **0.111427** | 0.608116 | 0.162261 | 未晋级 |
| `M2.11` | Morgan-2048 + 描述符 | 0.436242 | 0.978932 | 0.340027 | 0.106357 | 0.602197 | 0.159551 | 未晋级 |

虽然 `M2.10` 在早期固定 outer split 上优于无化学版本，但严格化学实体 OOF 的 FC 与 high-effect 指标更差，因此当前最终模型没有使用 Morgan 特征。不能用旧 outer 数字覆盖这项后来的反证。

### 6.2 化学轴：旧冻结 outer 结果

共同条件：D0、`V0-FROZEN`、seed 42。数值为 `FC PCC / high-effect F1`。

| 模型 | S1 | S2 | S3 | time |
|---|---:|---:|---:|---:|
| `M2.0-S42` | 0.335218 / 0.171417 | 0.356882 / 0.192129 | 0.250255 / 0.128672 | 0.522510 / 0.319212 |
| `M2.10-S42` | 0.349377 / 0.182904 | 0.358948 / 0.197906 | 0.260427 / 0.136559 | 0.533872 / 0.331728 |
| `M2.11-S42` | 0.332870 / 0.176639 | 0.365173 / 0.201191 | 0.266279 / 0.139413 | 0.531981 / 0.332939 |

旧交付文档 `MODEL_GUIDE_ZH.md` 的 `seed42-v3` 对应 `M2.10-S42`；其 seed 17/2026 是同一模型的复现实例，不是集成模型。

### 6.3 响应空间轴：严格 S1 entity OOF

| 模型 | response space | 解释能量 | FC PCC | context PCC | high PCC | high F1 | 决策 |
|---|---|---:|---:|---:|---:|---:|---|
| `M2.0` | learned-64 | 不适用 | **0.358665** | 0.102352 | **0.630771** | **0.170422** | 主干 |
| `M2.20` | fixed SVD-16 | 0.3652 | 0.347793 | **0.108180** | 0.620258 | 0.153086 | 不晋级 |
| `M2.21` | fixed SVD-32 | 0.4361 | 0.350815 | 0.107578 | 0.622768 | 0.157051 | 不晋级 |
| `M2.22` | fixed SVD-64 | 0.5105 | 0.353585 | 0.106777 | 0.624643 | 0.160061 | 不晋级 |
| `M2.23` | fixed SVD-128 | 0.5913 | 0.357453 | 0.106655 | 0.627920 | 0.164309 | 不晋级 |

`M2.23 - M2.0` 的逐化学 FC 差为 -0.003175，95% bootstrap CI `[-0.005278,-0.001172]`；context 差为 +0.004934，CI `[+0.000322,+0.009151]`。这是取舍，不是整体提升。

### 6.4 损失轴：严格 S1 entity OOF，seed 42

| 模型 | 改动 | log2 RMSE | abs R2 | FC PCC | context PCC | high PCC | high F1 | 决策 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `M2.0` | MSE control | 0.431284 | 0.979152 | 0.358665 | 0.102352 | **0.630771** | **0.170422** | 主干 |
| `M2.30` | FC=MSE+MAE | 0.425745 | 0.980054 | 0.360143 | 0.107577 | 0.624215 | 0.164959 | 不晋级 |
| `M2.31` | all Huber | **0.423574** | **0.980408** | **0.363957** | **0.109040** | 0.614791 | 0.168915 | 只作辅助组件 |

Huber 提高 FC/absolute/context，但明显降低 high-effect PCC，因此不能单独替换 `M2.0`。

### 6.5 MSE/Huber 跨种子稳定性

| model seed | `M2.0` FC | `M2.31` FC | 15% Huber blend FC | `M2.0` high PCC | blend high PCC |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.358665 | 0.363957 | 0.359806 | 0.630771 | 0.628955 |
| 43 | 0.357573 | 0.363960 | 0.358882 | 0.630771 | 0.629059 |
| 2026 | 0.356622 | 0.362828 | 0.357906 | 0.631344 | 0.629659 |

三个 seed 均由受约束网格独立选中 15% Huber。blend 对 MSE 的逐 held-out chemical FC 改善分别为 +0.001323、+0.001464、+0.001441，三个 95% CI 均大于 0。

### 6.6 校准与 PPI 消融

共同条件：旧 `V0-FROZEN`、seed 42；表中为 FC PCC。

| 模型 | S1 | S2 | S3 | time | 结论 |
|---|---:|---:|---:|---:|---|
| `M2.10` | 0.3496 | 0.3589 | 0.2605 | 0.5337 | 当时的完整 MVP |
| `M2.40` | 0.156 | 0.226 | 0.148 | 0.286 | 关闭校准大幅退化；拒绝 |
| `M2.50` | 0.3512 | 0.3586 | 0.2605 | 0.5325 | 真实 PPI 增益不稳定 |
| `M2.51` | 0.3515 | 0.3584 | 0.2605 | 0.5322 | 重连对照与真实图相当 |

`M2.40` 的 S1 absolute sample R2 也从约 0.981 降至 0.881。真实 PPI 没有稳定击败重连对照，因此不进入主模型。

证据：`runs/experiments/chemistry_s1_comparison_seed42/`、`runs/experiments/response_space_s1_20260812/`、`runs/experiments/response_loss_s1_20260812/`、`docs/experiments/response_mvp_v1.md`。

## 7. M3：静态图模型

设置：D0、`V0-FROZEN`、seed 42、30 epochs、每个变体 319,878 参数。STRING v12 physical、score≥400；4,407/4,422 蛋白完成 SGD 映射，54,452 条无向边，3,933 个蛋白有图覆盖。真实图和重连图参数量完全相同。

### FC PCC

| 模型 | S1 | S2 | S3 | time |
|---|---:|---:|---:|---:|
| `M3.0` 无图 | 0.139888 | 0.219494 | 0.124684 | 0.267701 |
| `M3.1` 真实 PPI | 0.140221 | 0.220541 | 0.125456 | 0.267489 |
| `M3.2` 重连 PPI | **0.140652** | **0.221412** | **0.126300** | **0.267789** |

### Absolute sample R2

| 模型 | S1 | S2 | S3 | time |
|---|---:|---:|---:|---:|
| `M3.0` | 0.870681 | 0.938532 | 0.865303 | **0.950991** |
| `M3.1` | **0.871892** | 0.938593 | **0.865494** | 0.949678 |
| `M3.2` | 0.871762 | **0.938893** | 0.863563 | 0.949638 |

结论：真实 PPI 在 FC 上没有击败重连负对照，整个静态图家族不晋级。证据：`docs/ppi_graph_mvp.md`。

## 8. M4：条件图模型

设置：D0、`V0-FROZEN`、seed 42、10 epochs smoke run。条件编码包含 Morgan-512、7 个描述符和时间特征；节点注入资源只有 8 个化合物、16 条人工审核靶点种子，其中 15 条映射到保留蛋白；菌株突变表为空。

| 模型 | 说明 | S1 FC | S2 FC | S3 FC | time FC | 状态 |
|---|---|---:|---:|---:|---:|---|
| `M4.0.0` | 无图、无节点注入 | 0.133870 | 0.213324 | 0.122324 | 0.214626 | 已运行 smoke |
| `M4.0.1` | 真实图、无注入 | 无结果 | 无结果 | 无结果 | 无结果 | 代码支持，未留存本轮结果 |
| `M4.0.2` | 重连图、无注入 | 无结果 | 无结果 | 无结果 | 无结果 | 代码支持，未留存本轮结果 |
| `M4.1.0` | 无图、节点注入 | 无结果 | 无结果 | 无结果 | 无结果 | 代码支持，未留存本轮结果 |
| `M4.1.1` | 真实图、节点注入 | 0.133251 | 0.212024 | 0.121142 | 0.215508 | 已运行 smoke |
| `M4.1.2` | 重连图、节点注入 | 0.130524 | 0.210285 | 0.117841 | 0.215577 | 已运行 smoke |

结论：三个已运行变体均低于 `M3.0`；靶点覆盖和菌株信息不足，不进入提交模型。证据：`docs/ppi_graph_evolution_v2.md`。

## 9. M5：融合与最终模型

### 9.1 `M5.0` 场景融合规则

先在 `V1-ENTITY-OOF` 内冻结权重，不使用 outer 结果回调：

| 测试场景 | `M2.0` 权重 | `M2.31` 权重 | 选择依据 |
|---|---:|---:|---|
| S1 / `test_chem_only` | 0.85 | 0.15 | 3 seeds 独立选中，逐化学 bootstrap 同向 |
| S2 / `test_strain_only` | 1.00 | 0.00 | Huber 降低 FC 和 high-effect |
| S3 / `test_both` | 1.00 | 0.00 | Huber 降低 FC 和 high-effect |
| time / `test_time` | 0.70 | 0.30 | time 与 time-forward 在 high-effect 护栏下共同选中 |

### 9.2 S2/S3/time inner OOF 证据

| 场景/选定模型 | FC PCC | context PCC | drug PCC | high PCC | high F1 |
|---|---:|---:|---:|---:|---:|
| S2，100% `M2.0` | 0.280621 | 不可定义 | 0.218011 | 0.614390 | 0.158383 |
| S3，100% `M2.0` | 0.216675 | 不可定义 | 不可定义 | 0.497406 | 0.115913 |
| time，70/30 `M5.0` | 0.508836 | 0.356763 | 0.425353 | 0.772590 | 0.306538 |
| time-forward，70/30 `M5.0` | 0.531470 | 0.380233 | 0.446083 | 0.777819 | 0.319064 |

### 9.3 冻结 outer confirmation

权重冻结后使用 seed 2026 在 `V0-FROZEN` 确认；outer 结果未用于再次调权。

| split | Huber 权重 | FC PCC | residual PCC | high PCC | high F1 |
|---|---:|---:|---:|---:|---:|
| `val_chem_only` | 0.15 | 0.334206 | context 0.114365 | 0.632166 | 0.170895 |
| `val_strain_only` | 0.00 | 0.349373 | drug 0.298391 | 0.686517 | 0.188816 |
| `val_both` | 0.00 | 0.244759 | 不可定义 | 0.568531 | 0.127746 |
| `val_time` | 0.30 | 0.524486 | context 0.372424 | 0.779696 | 0.318085 |

### 9.4 `M5.1` 三种子最终系统

组成：`M2.0-S42/S43/S2026` 和 `M2.31-S42/S43/S2026` 各自先等权平均，再应用 `M5.0` 场景权重。

模型选择证据：三 seed bagged S1 OOF 的 15% Huber blend：

| FC PCC | context PCC | high PCC | high F1 |
|---:|---:|---:|---:|
| 0.361525 | 0.103346 | 0.631517 | 0.170129 |

相对纯 bagged MSE：FC +0.001236，context +0.001267，high PCC -0.001776，high F1 -0.000207。逐化学 FC 差 +0.001402，95% CI `[+0.000715,+0.001959]`。

最终 refit 与 test 产物：

| 项目 | 记录 |
|---|---|
| 训练协议 | `R1-ALL-LABELED`，每个组件 8,958 行 |
| checkpoint 数 | 6 |
| test 输出 | 4,454 × 4,422，log2 |
| 数值范围 | `[9.676924, 34.273403]` |
| NaN / inf | 0 / 0 |
| `prediction.csv` | `runs/final/frozen_ensemble_20260812/prediction.csv` |
| prediction SHA256 | `b17004cdaf072177f380c842d27daeddce1d9977cc6cdd2f9c913b454771f86d` |
| single-seed 备份 SHA256 | `90b6a4d54940889a8823c94555bcda9de6f2adceb38d0ccd792351a618b38fa8` |
| 官方提交/分数 | 无；当前工作区没有提交端点、凭据或官方 scorer |

历史结论：`GOAI-M5.1` 是 2026-08-12 的冻结交付模型，现已由 `M5.2` 取代。不得把上述本地 FC PCC 称为官方 PSS、官方 delta 或排行榜分数。

证据：`docs/experiments/response_space_robust_loss_execution_20260812.md`、`runs/final/frozen_ensemble_20260812/decision_manifest.json`、`runs/final/frozen_ensemble_20260812/prediction_contract.json`。

### 9.5 `M5.2` cell-conditioned 场景路由系统

日期：2026-08-13。父模型：`M5.1`。状态：**晋级并冻结**。

实验假设：旧 response 分支独立读取条件，不能表达“同一药物作用在不同细胞状态上会产生不同响应”。`M6` 先编码 `h_cell = Encoder(strain, medium, temperature, time)`，再让 background 和 response 共享该状态；`M6.11` 用 concat，`M6.21` 用 FiLM。保持不变的部分包括 D0 数据边界、4,422 蛋白、matched-control FC、calibration rank 16、hidden 128、80 epochs、MSE 和 fold seed 42。

执行范围：42 个首轮/扩展 producer + 11 个最终确认 producer，共 53 个正式 matrix producer，失败 0；另运行 3 个新 outer confirmation 和 6 个 D2 全标注 refit。所有 OOF 的特征、标准化、matched-control 参照、SVD/prior 状态都只在 fold train 内拟合。

#### S1 新化合物：三 seed bagged OOF

| 模型 | FC PCC | context PCC | high PCC | high F1 | abs R2 | 决策 |
|---|---:|---:|---:|---:|---:|---|
| `M2.0` 三 seed | 0.360289 | 0.102079 | 0.633293 | 0.170336 | 0.979106 | 对照 |
| `M6.11` 三 seed | **0.371674** | 0.098239 | **0.635453** | **0.182438** | **0.979297** | 晋级 S1 |
| `M6.11` + 10% `M6.13` | 0.372388 | 0.099051 | 0.634329 | 0.182249 | 0.979444 | 拒绝；收益太小且 high PCC 明确下降 |

`M6.11 - M2.0`：FC `+0.011385`，四折 bootstrap 95% CI `[+0.008648,+0.014281]`；high PCC `+0.002160`，high F1 `+0.012102`，abs R2 `+0.000190`，context `-0.003840`。逐 held-out chemical 的 FC 增益在 seed 42/43/2026 分别为 `+0.013766 / +0.015264 / +0.013424`，三个 95% CI 均完全大于 0；因此不是单 fold 或单 seed 偶然。

10% Huber 相对纯 `M6.11` 虽有 FC `+0.000715`，CI `[+0.000468,+0.001079]`，但 high PCC `-0.001125`，CI `[-0.001570,-0.000487]`，high F1 也未改善，因此没有进入最终组件。

#### time：三 seed bagged OOF

| 场景 | 模型 | FC PCC | context PCC | drug PCC | high PCC | high F1 | abs R2 |
|---|---|---:|---:|---:|---:|---:|---:|
| time | `M2.0` | 0.511024 | 0.358420 | 0.427350 | 0.776183 | 0.307357 | 0.983919 |
| time | `M6.20` FiLM-128 | 0.537243 | 0.383002 | 0.465072 | 0.787314 | 0.343632 | 0.984165 |
| time | `M6.21` FiLM-256 | **0.539912** | **0.383961** | **0.468130** | **0.789024** | **0.346425** | **0.984186** |
| time-forward | `M2.0` | 0.534982 | 0.383512 | 0.449532 | 0.782276 | 0.321945 | 0.980950 |
| time-forward | `M6.20` FiLM-128 | 0.576423 | 0.425908 | 0.503887 | 0.803385 | 0.379061 | 0.981455 |
| time-forward | `M6.21` FiLM-256 | **0.578731** | **0.427088** | **0.506483** | **0.804417** | **0.382575** | **0.981505** |

`M6.21 - M2.0`：普通 time FC `+0.028888`，95% CI `[+0.027138,+0.031178]`；严格 time-forward FC `+0.043749`，CI `[+0.040622,+0.046877]`。time-forward 的 context、drug、high PCC、high F1 分别提高 `+0.043576 / +0.056951 / +0.022142 / +0.060631`，没有指标交换。`M6.21` 相对 `M6.20` 也在两种 time 协议上提高 FC、drug、high PCC 和 high F1，因此 rank 256 晋级。

#### S2/S3 拒绝证据

| 场景 | 候选 | 对照/候选 FC | ΔFC | high PCC 变化 | abs R2 变化 | 决策 |
|---|---|---:|---:|---:|---:|---|
| S2 | `M6.11` | 0.280663 / 0.274575 | -0.006088 | -0.013892 | -0.000326 | 拒绝 |
| S2 | `M2.43` 去 plate | 0.280663 / 0.282525 | +0.001862 | -0.000155 | -0.000727 | 拒绝；FC CI 跨 0 |
| S3 | concat/FiLM/rank/prior 候选 | 最佳未超过 `M2.0` | 非正 | 无稳定改善 | 无稳定改善 | 全部拒绝 |

去 plate 的 S2 FC bootstrap CI 为 `[-0.000927,+0.004651]`；outer `val_strain_only` FC 虽 `+0.002651`，但 high PCC `-0.002205`、abs R2 `-0.000671`。因此它只保留为 calibration confounding 诊断，不路由 test。

#### 冻结 outer confirmation

| split | 对照 FC | 晋级候选 FC | ΔFC | context/drug Δ | high PCC Δ | high F1 Δ | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `val_chem_only`，`M6.11-S42` | 0.335218 | 0.341661 | +0.006443 | context -0.003062 | +0.004590 | +0.009068 | 复现 S1 主方向 |
| `val_time`，`M6.21-S42` | 0.522510 | 0.540404 | +0.017894 | context +0.013257；drug +0.026036 | -0.002938 | +0.030770 | FC/F1/残差复现；记录 high PCC 小幅取舍 |

outer 只用于确认已由 OOF 冻结的方向，未用于重新搜索 rank、结构或权重。

#### 关键负结果与信息缺口

- learned rank 32 明显退化；learned 128/256 可用；fixed SVD-256 在 S1/S3/time-forward 均低于 learned，说明任务驱动低秩优于固定统计低秩。
- ChemBERTa real 的 S1 FC `0.255660` 高于 shuffled `0.220838`，证明嵌入含有真实语义；但 Morgan+real 也只有 `0.256427`，均远低于 `M2.0=0.358665`。当前“384 维直接拼接”拒绝，下一步应做压缩/门控而不是宣称预训练无用。
- OOF-safe target prototypes 最好只带来局部 FC 增益并破坏 high-effect；没有进入最终模型。
- plate 与 time/instrument/data source 的 NMI 分别为 `0.5318/0.5310/0.4022`，且 plate 对 time、instrument、source、medium、temperature 的加权 purity 都是 1.0。校准确有强混杂，但去 plate 未通过最终多指标护栏。
- strain genome 分支状态为 `BLOCKED_NO_VERIFIED_STRAIN_IDENTITY`：BAH/CEK/CGD/DHY210/BAI/CRD 没有经来源核验的身份映射。
- protein ESM 分支状态为 `BLOCKED_NO_PROTEIN_SEQUENCE_ARTIFACT`：缺少 4,422 输出列到 SGD/UniProt 序列的 ≥98% 覆盖映射。未伪造这两类输入。

#### `M5.2` 最终路由与产物

| test 场景 | 路由 |
|---|---|
| `test_chem_only`，1,640 行 | `M6.11-S42/S43/S2026` 等权平均 |
| `test_time`，151 行 | `M6.21-S42/S43/S2026` 等权平均 |
| `test_strain_only`，1,534 行 | 逐值保留 `M5.1` |
| `test_both`，1,129 行 | 逐值保留 `M5.1` |

| 项目 | 记录 |
|---|---|
| 新 refit checkpoint | 6 个；每个 8,958 行，fit scope 已验证 |
| test 输出 | 4,454 × 4,422，log2 |
| 数值范围 | `[9.816884, 34.273403]` |
| NaN / inf | 0 / 0 |
| `prediction.csv` | `runs/final/goai_m5_2_20260813/prediction.csv` |
| prediction SHA256 | `919e20020b7bff836058d0be36411c412193e4ccba92f112ff2854575f6c59b7` |
| 预测契约 | `runs/final/goai_m5_2_20260813/prediction_contract.json` |
| 决策清单 | `runs/final/goai_m5_2_20260813/decision_manifest.json` |
| 官方提交/分数 | 无；缺官方端点、凭据、scorer 和权威 sample-submission 蛋白列契约 |

结论：`GOAI-M5.2` 取代 `M5.1` 成为当前冻结交付模型。上述全部数字仍是本地 OOF/outer proxy，不是官方 PSS 或排行榜分数。

### 9.6 `AUDIT-P01`：FC 约 0.3 平台期诊断

日期：2026-08-13。性质：**只读诊断，不产生新模型编号、不改变 `M5.2` checkpoint 或预测值**。

诊断结论：当前 S1/S2/S3 的主要瓶颈不是 response rank 或网络深度，而是未知实体缺少可迁移语义、FC 标签信噪比较低，以及 pooled FC 指标被公共扰动模式主导。继续只在相同 one-hot 输入上扩网络，预期只能得到小幅波动。

#### 最终模型实际可见的未知实体信息

`M6.11` 和 `M6.21` 的全标注 refit 均记录：

| 特征 | 记录值 |
|---|---:|
| `uses_chemical_structure` | false |
| `uses_chemical_semantics` | false |
| `uses_strain_semantics` | false |
| `chemical_feature_dim` | 0 |
| `strain_feature_dim` | 0 |

全标注数据含 46 个 chemical category 和 5 个 strain category，但 test 还有 11 个从未出现在 train/val 标签中的新药，以及完全未出现过的菌株 `CRD`。因此：

- 11 个 test-only chemical 的 one-hot 在 response 分支中都变为同一个全零未知编码；同一 cell context 下，模型不能区分它们的药物特异响应。
- `CRD` 的 strain one-hot 和语义块同样为全零；模型不能学习 CRD 特异 background 或 strain×drug modulation。
- `test_both` 同时包含这些信息缺口；不存在可由扩大 MLP/rank 自动恢复的实体关系。
- time 是连续数值特征，且大部分 chemical/strain 已见，所以 FiLM time/time-forward 能达到 `0.539912/0.578731`；这与 S1/S2/S3 的平台期形成对照。

证据：`runs/final/goai_m5_2_20260813/refits/M6.11-S42/feature_summary.json` 与同家族其余五个 refit feature summaries。

#### pooled FC 掩盖药物特异能力

S1 三 seed bagging 的 raw FC PCC 为 `0.371674`，但 subtract 同 context 公共响应后的 context-residual PCC 只有 `0.098239`；high-effect recall 约 `0.14`。这说明当前分数主要来自共享 stress/context pattern，而不是正确区分每个新药。以后不能再把单独的 pooled FC 小幅上升称为实体 OOD 的大突破。

#### D0 inner-train 信噪比与重复覆盖

| 项目 | 诊断值 |
|---|---:|
| inner-train 总样本 | 5,920 |
| treatment / control / QC | 5,078 / 751 / 91 |
| 保留蛋白 | 4,422 |
| 观测蛋白值比例 | 84.665% |
| treatment biological groups | 3,337 |
| 每组重复数中位数 | 1 |
| 有 exact control 的 treatment rows | 5,066 |
| FC observed values | 18,613,603 |
| FC 标准差 | 0.403745 log2 |
| `median(|FC|)` | 0.170620 log2 |
| `P(|FC|>0.5)` | 12.883% |
| `P(|FC|>1)` | 2.893% |
| control leave-one-out residual 标准差 | 0.339030 log2 |
| control leave-one-out `median(|residual|)` | 0.156584 log2 |

以相同 `strain/chemical/medium/temperature/time`、但跨 plate/source/instrument 的 matched-control FC 作为跨批次重复，3,020 个样本对的 PCC 中位数为 `0.0942`，四分位区间 `[0.0117,0.2907]`。用其他重复的均值预测留一重复，池化 PCC 为 `0.2369`。这不是严格数学 noise ceiling，因为它混合了跨批次差异和有限重复估计误差，但足以证明小 FC 的可重复性很弱、均值收缩有强统计诱因。

#### test chemical 相似度比现有验证更困难

使用 canonical SMILES、Morgan radius 2、2,048 bits，并只与 36 个有解析结构的 inner-train treatment chemicals 比较最近邻：

| 集合 | chemical 数 | 最近训练 chemical Tanimoto 中位数 | 范围 | `>=0.4` |
|---|---:|---:|---:|---:|
| `val_chem_only` held-out | 6 | 0.246 | 0.129–0.860 | 2 |
| `test_chem_only` test-only | 11 | 0.167 | 0.143–0.897 | 1 |

test-only 中只有 Tamoxifen 与训练侧 4-Hydroxytamoxifen 高度相似；其余多数最近邻只有约 `0.14–0.29`。因此现有 S1 validation 仍可能比最终 test 乐观，裸结构特征在只有 36 种训练药物时也缺少足够邻域支持。

#### 对后续迭代的约束

1. 暂停只改变 hidden size、rank、MLP/Transformer 深度的主线搜索；它不能创造未知实体信息。
2. S2/S3 的实质晋级前置条件是经来源核验的 `CRD`/其他 strain 身份与 genome/genotype 表征；缺失时必须继续标为信息阻塞。
3. S1 优先验证 target/MoA/pathway 等机制表征，以及 fold-safe 的结构压缩/检索；不再直接拼接高维 ChemBERTa。
4. 新结构必须显式分解 `shared stress + chemical residual + strain modulation + interaction`，并直接报告 context/drug residual，而非只看 pooled FC。
5. 加入 replicate aggregation、可靠性权重和 high-effect 专用目标；同时构建按 test 最近邻相似度匹配的 remote-chemical folds。
6. 晋级口径收紧：单独 pooled FC 的千分位/百分位小涨不称为突破；至少要在相应 residual 与 high-effect 指标上同向，并经多 seed 与实体级 bootstrap 支持。

本诊断没有改变当前冻结模型：仍为 `GOAI-M5.2`；官方提交与官方 PSS 仍为无。

### 9.7 `M7/M8` 通用模型 + 已见实体专家重构

日期：2026-08-13。父模型：`M6` 生物交互结构与 `M5.2` 路由诊断。当前状态：
**代码与无泄漏实验系统已实现；首批 `M7.0/M7.3` 严格 OOF 已完成，失败 0；
结果只作 discovery 诊断，均未晋级。**

实验假设：原 `M5.2` 把整个 `split_final` 路由到不同完整网络，无法表达全量
refit 后 BAI/六个 validation chemicals 已从 unseen 变成 seen 的事实。`M7`
使用一个 universal 生物主干，在其上叠加 fold-fit strain/chemical/pair
低秩响应残差专家；`M8` 再增加开放 chemical/strain semantics。

已实现的自然尺度分解为：

```text
final = B_U + g_s B_s + C_obs
      + I[treatment] * (R_U + g_s R_s + g_c R_c + g_sc R_sc)
```

- `R_U/R_s/R_c/R_sc` 全部共享同一个 `rank × 4,422` protein decoder；
- `R_sc` 的 pair latent 由当前 cell state 调制，而不是静态 pair 常数；
- `Calibration` 只读 `data_source/instrument/plate`：每个 fold 只用 fit observation
  计算输入中心并硬中心化；输出零均值仍是软惩罚，另加 L2 和 centered plate dropout。
  旧 checkpoint 缺少 center buffer 时以零加载，已验证 `M6.11` 历史预测方程不变；
- universal → strain → chemical → pair → low-LR joint 分阶段训练；joint batch
  均衡模拟 R00/R10/R01/R11；
- 组件可导出 `B_U/B_s/C_obs/R_U/R_s/R_c/R_sc/final`，并执行 exact reconstruction 检查。

实体与路由契约：

- 专家 seen gate 优先使用 registry 中实体自身的 namespaced `canonical_id`；只在没有
  canonical ID 时回退 `raw:<normalized>`。同 canonical aliases 共享专家 index。
- proxy 只使用自身 `goai-chemical:*` ID，永不使用 `proxy_target`；DHY210 使用
  `goai-strain:DHY210` 作 ID 专家键，但通用菌株语义仍为 zero + missing flag。
- support manifest 同时保存 canonical support keys 与 raw audit keys；pair/context-time 只统计
  treatment rows。每个 checkpoint 保存 registry、artifact、config 和 support 哈希链。
- 用 8,958 个全标注行生成的 support 对 test 逐行审计，得到
  `R10=2,072 / R01=1,594 / R00=425 / R11=135 / control=228`；`test_both`
  treatment 精确为 `R10=432 / R01=272 / R00=425`。

验证与防泄漏：

- 固定 `R00/R10/R01/R11/RT`；每 fold 重新拟合 support、scaler、matched control、
  response prototype、optional SVD 和专家状态。
- `R10/R01` 分别只按 chemical/strain 分区，不再做无效的二维交叉重训。
  2-fold 五场景从 18 次降为 14 次 fit；4-fold 从 68 次降为 44 次，
  eligible treatment 在每个 regime 仍恰好评分一次。
- producer `run_contract` 已加入核心源码逐文件清单与 aggregate SHA256；源码改变后
  旧 fold 会被 `--resume` 硬拒绝。
- 当前独立全量单元/集成测试：`187 passed`；四份夜间矩阵的 55 个
  effective config 均已成功解析；quick-screen 的 32 个配置还已在真实
  5,920 行上完成 feature fit/transform 预检，失败 0。
- 计算预算按实际 OOF producer 计数：fair expert `98` fits；quick screen
  `328` fits；research prior/pair `86` fits；calibration audit `112` fits；夜间筛选合计
  `624` fits。任一正式确认候选为 4 folds × 3 seeds 下的 `132` fits。

首批运行契约（仅 discovery screen，不是晋级确认）：

| 项目 | `M7.0` | `M7.3` |
|---|---|---|
| matrix ID | `SCR-M7.0-GENERAL` | `SCR-M7.3-ENTITIES` |
| 唯一变化 | universal trunk，无专家 | 增加 strain + chemical residual experts |
| protocol | 两折 R00/R10/R01/R11/RT | 两折 R00/R10/R01/R11/RT |
| fold/model seed | `42 / 42` | `42 / 42` |
| epochs | 12（universal 9 + joint 3） | 12（universal 5 + strain 2 + chemical 2 + joint 3） |
| response rank / hidden | 256 / 128 | 256 / 128 |
| run-contract SHA256 | `ea82e522...1cd2` | `d2043b60...bd5f` |
| source fingerprint | `059484c6...ba03` | `059484c6...ba03` |
| 当前状态 | 完成；未晋级 | 完成；未晋级 |

运行目录：`/mnt/Omics_GPU/chenyuming/go-ai/runs/nightly/20260813-m7-m8-core-v2`。
运行介质为网络大盘（启动时约 2.9 TB 可用），避免根分区仅 23 GB 剩余导致中断。

一次早期 launcher 预检在产生任何 fold checkpoint/预测前主动终止：原因是发现旧
`run_contract` 尚未绑定 source fingerprint。该目录
`/mnt/Omics_GPU/chenyuming/go-ai/runs/nightly/20260813-m7-m8-screen` 只含约 80 KB 的配置/日志，
不是可评估实验，不得与 v2 结果混合。

`core-v2` 作为 R1 历史诊断冻结：chemical registry `36577a3c...c943`、strain registry
`67f56304...8c8c`、fit-train support file `787ee78a...dc8c`、all-labeled support file
`b728de15...e22dd`。二次审计修正后，R2 registry/support/embedding manifest 均已变化；
因此 `core-v2` 指标仍可解释其历史架构实验，但其 fold/checkpoint 不可被 R2 任务 resume，
也不可与 R2 producer 在同一个汇总或集成中混合。

首批本地严格 OOF 结果（`M7.3 - M7.0` 为同 fold 配对差值）：

| regime | `M7.0` FC PCC | `M7.3` FC PCC | ΔFC | 相关 residual Δ | high PCC Δ | high F1 Δ | 诊断 |
|---|---:|---:|---:|---:|---:|---:|---|
| R00 双未知 | 0.185584 | 0.168750 | **-0.016834** | 不适用 | -0.018704 | -0.019227 | 全面下降 |
| R10 菌株已见/药物未知 | 0.224603 | 0.246983 | **+0.022380** | context `+0.006016` | +0.019959 | +0.013831 | strain expert 有效信号 |
| R01 菌株未知/药物已见 | 0.187237 | 0.192642 | +0.005405 | drug **-0.006680** | +0.018056 | -0.000403 | chemical expert 未过护栏 |
| R11 实体已见/pair 未见 | 0.232988 | 0.265965 | **+0.032976** | context `+0.021889`；drug `+0.018057` | +0.031796 | +0.020098 | 两类专家均有信号 |
| RT pair/context/time 外推 | 0.226644 | 0.260555 | **+0.033911** | context `+0.024103`；drug `+0.019667` | +0.046609 | +0.018303 | 已见实体插值有信号 |

四种实体 regime 的等权 FC macro 为 `0.207603 → 0.218585`，但不能用它掩盖
R00 的退化。R10、R11、RT 的每个评分折方向一致；R01 未达到预设 `+0.01`，且真正的
drug residual 反向。产物 SHA256：`M7.0 oof_summary=5db36e0e...510c`，
`M7.3 oof_summary=08ec5918...e968`，配对汇总 `c6468cdc...d1d`。

本轮还存在明确的 schedule 混杂：`M7.0` 的 universal 阶段为 9 epochs，`M7.3`
只有 5 epochs；R00 推理时所有实体专家 gate 都为 0，因此其下降至少部分可能来自通用主干
少训练 4 epochs，不能据此判定“专家架构损害双未知”。下一步必须从同 fold、同初始化、
同 universal checkpoint 分叉训练专家，并把 frozen-expert residual 与 low-LR joint 分开消融；
当前不允许直接把 `M7.3` 送入确认矩阵。

该公平性修复现已落地，但尚未运行：opt-in `fold_matched_universal_warm_start` 先构造并训练
完全不含专家参数的 M7.0 通用父状态 9 epochs，再把所有共同 tensor 严格复制到零初始化
专家模型；复制后重置 optimizer、冻结共同状态训练 residual。每 fold 保存预训练、复制后、
frozen expert 后和可选 joint 后的共同 state SHA256 receipt，键/形状不一致立即失败。
独立 `fair_expert_ablation.yaml` 覆盖 M7.0 receipt control、strain-only frozen/joint、
chemical-only frozen/joint、both frozen/joint 共 7 项；R10/R01 分别作为 strain/chemical
专家首要证据。两折五协议每项 14 fits、总计 98 fits，不改写上面的历史 discovery 契约，
也未干扰已经完成的 core-v2 producer。旧 checkpoint 加载协议不变。

正式 confirmation 又在此基础上锁定了更严的公平对照：所有 M7 专家候选共享
同 fold 的 U80 universal parent；frozen 变体只训残差专家，joint 变体再追加
16 epochs 小学习率更新。`M7.1/M7.2/M7.3-JOINT` 必须对比同样更新预算的
纯 universal U96 control；`M7.4` 的 frozen/joint 分别对比同路径 `M7.3`。每个
fold 必须保存 parent/copied/post-frozen/final common-state SHA256 收据。

严格晋级门禁只接受 `kind=model_confirm`、4 folds、fold seed 42、model seeds
`42/52/62` 和每 seed 44 fits 的完整产物。相关 regime 要求 `ΔFC>=0.01`，
R10 context residual、R01 drug residual、R11/RT 两类 residual 同时改善；
high-effect PCC/F1 总体与每 seed 均不得下降 0.005 以上，三个 seed 方向一致，
且 held-out entity 聚类 bootstrap 95% CI 下界大于 0。quick/fair/fold-bootstrap 均不能
写入 `promoted=true`。R00 因无可识别 residual，不能单独支持晋级。

正式专家 scale 已从“全局 quick OOF 选一个数”修正为每个 outer fold 内的
train-only nested inner OOF。专家头统一以 scale 1 训练，每个 inner fit 只训一次，
再用实名组件预测对 `0/0.25/0.5/0.75/1` 做便宜重组：R10 选 strain，
R01 选 chemical，R11 选 strain×chemical，RT 选 strain×chemical×pair，R00 不调专家。
目标是 inner FC PCC，相对 all-zero 要求相关 residual 不退化、high-effect PCC/F1
不低于 `-0.005`，平手选更小 scale。每个 outer fold 保存 inner assignments、全候选
指标、support/artifact/source 哈希和选定收据；resume 与 promotion 逐折验证。
外层 validation 标签没有进入选择数据路径。全局 quick scale 现在只能提名候选，
不再绑定 formal prediction。

Calibration 审计锁定 7 组×16 fits，比较 rank 4/8/16、no-plate、plate shuffle
和 dropout 0/0.3/0.5，并执行 leave-one-plate-out。只有 plate 严格胜过 no-plate 和
shuffle，且 FC/residual/high-effect 护栏全过时才保留 plate；否则正式确认
自动注入 no-plate 选择。选择结果、完整性和哈希写入不可修改收据。

R2 overnight 已于 `2026-08-13 14:58 UTC` 启动后台等待器，tmux session 为
`goai_m7m8_overnight_r2_20260813`，根目录为
`/mnt/Omics_GPU/chenyuming/go-ai/runs/nightly/20260813-m7-m8-overnight-r2`。
启动时两张 A100 正被同账号下另一项 BioHub 训练占用，故状态为 `waiting_for_gpu`，未创建
任何 GOAI CUDA 进程。门禁要求 GPU 0/1 各自 free memory `>=30,000 MB` 且 utilization
`<=20%`，每 30 秒复核，无超时；满足后依次运行 fair 98 fits → receipt audit/summary →
  quick 328 fits/summary → research prior/pair 86 fits/summary → calibration 112 fits/summary。任何 receipt、source fingerprint、
artifact chain 或阶段返回码失败都会停止后续派发；`<run-root>/STOP` 可安全停止并续跑。
状态收据：`<run-root>/overnight_status.json`。

`2026-08-13 15:01 UTC` 独立逐条验收后，在任何 R2 producer 启动前写入 `STOP` 并安全
暂停等待器；状态收据为 `stopped + resume_available`，GPU/训练产物为 0。当时独立
审计发现了 promotion、identity gate、formal warm-start、pair frozen residual、完成产物
续跑合同以及 outer scale selection 六类风险。截至本节更新，六项均已修复：
完成产物在任何 GPU 子进程启动前重建并验证 source/config/input/artifact/fold/seed/
scenario/manifest 合同；pair frozen 阶段保留已训练 `R_s/R_c`，使 `R_sc` 学习与
推理一致的最终残差；formal scale 改为上述 nested inner OOF。全测为 `187 passed`。

自动闭环现包含 identity classification → fair/scale/calibration evidence preparation →
仅物化被提名的 M7 候选及其 U80/U96/M7.3 精确对照 → 4-fold×3-seed
confirmation → 逐候选 promotion receipt。统计不达标会写 `blocked`但视为有效完成；
数据/哈希/折分/训练收据缺失会 fail closed。当前 M8 formal 因 verified-only 训练语义
覆盖为 0 被明确记录为 blocked，不会启动伪 M8 GPU 任务；research M8 仍在 quick 路线运行。

`2026-08-13 16:10 UTC` 主代理独立复跑全测 `187 passed`，并完成 55 个 effective
config 解析、strict identity wrapper 和 14 阶段编排预演。随后删除审计期 `STOP`，
用同一 R2 run root 恢复 tmux `goai_m7m8_overnight_r2_20260813`。当时状态为
`waiting_for_gpu`：GPU 门禁要求两卡同时空闲显存 `>=30,000 MB` 且利用率 `<=20%`，
等待期间未启动 GOAI CUDA 进程。`16:12:43 UTC` 两卡同时通过门禁，
fair stage 正式开始；首批 `FAIR-M7.0-U9-S42` 与 `FAIR-M7.1-U9-S2-FROZEN-S42`
已分配 GPU 0/1，batch status 为 `running`、失败 0。恢复后顺序锁定为：fair → receipt audit →
quick → research prior/pair → calibration audit → identity classification → selected M7
confirmation → batch promotion gate。实时收据为
`/mnt/Omics_GPU/chenyuming/go-ai/runs/nightly/20260813-m7-m8-overnight-r2/overnight_status.json`。

结果边界：这是 2-fold、单 seed 42 的 discovery screen；R00/R11 因二维交叉各有 4 个
评分折，R10/R01/RT 各 2 个。它不是 GOAI 官方 PSS。`M5.2` 继续是当前冻结模型；
官方提交 ID/官方分数：无。

### 9.8 `M9.0–M9.7`：独立 RNA 扰动 → 蛋白响应迁移（开放知识）

- 模型编号：`M9.0-S42` context-only；`M9.1-S42` Morgan scratch；
  `M9.2-S42/M9.3-S42` OP3 real/shuffle；`M9.4-S42/M9.5-S42`
  L1000FWD real/shuffle；`M9.6/M9.7-S42/S43/S2026` 冻结 context 主干上的
  zero-init OP3 real/shuffle residual gate。
- 父模型：无。greenfield 实现，不导入或训练 `M0–M8` 模型代码；只读取 D0、冻结
  S1 assignment 和本地 proxy metric 契约。旧 `M6.11` 预测仅在所有新结果冻结后读入评分器。
- 状态：`M9.1–M9.5` direct consumer 拒绝；`M9.6` 保留为有证据的 research
  response component，`M9.7` 为负对照；仍非完整模型，`M5.2` 不变。
- 假设：外部人类 RNA 扰动响应可监督化学 encoder 学习可迁移药物机制，再由独立
  yeast context×chemical consumer 预测 4,422 维蛋白 log2 FC。
- 唯一变化：所有化学臂仅改变 2048-bit Morgan encoder 的初始化；protein consumer、
  optimizer、fold、seed、loss 和训练轮数完全一致。RNA real 与负对照仅改变整药/整 parent
  的结构—RNA 配对，target/context/missingness 不动。

数据与防泄漏：

- 数据版本：`D0 + D3-ENTITY-20260813 + D4-RNA-OPEN-20260813`；开放知识榜，
  不属于 closed-data 结果。
- GOAI：inner-train treatment 5,078 行、37 个 S1 chemicals、4,422 proteins；exact
  matched-control 可评分 5,066 行。metadata SHA `9414f22d...13f`，proteome SHA
  `a15d9a40...ccb`，当前 chemical map SHA `1685158b...2a6`。
- OP3：原始 H5AD SHA `19042e...c516`；主 target 为 `logFC`，不是 signed p-value；
  按标准化 parent 全局排除 Amiodarone、Clotrimazole、Hydroxyurea、Raloxifene 后为
  598 rows / 142 compounds / 18,211 genes；PCA-64。
- L1000FWD：matrix/signature/drug/probe SHA 分别为 `6f50ade2...5492`、
  `b4495715...557`、`6447c511...99e3`、`500aa68b...2e`；全局排除任何 GOAI
  parent 命中的 40 pert IDs / 3,350 signatures 后为 38,948 signatures、4,861 pert IDs、
  4,276 standardized parents、978 landmark genes；fold-train cell residualization +
  whitened PCA-64，full PCA explained variance `0.312278`。
- RNA/GOAI overlap 规则：strict 主臂从外部训练一次性删掉所有 GOAI parent，而不只删
  当前 fold held-out drug；因此没有同药外部 response 泄漏。shuffled L1000 与 CV 分组单位
  均为 parent connectivity，不是 pert ID。
- 验证：V1 S1，固定四折 sizes `1255/1311/1249/1263`，fold seed 42；direct
  discovery model seed 42，residual gate model seeds 42/43/2026；
  每折训练化合物与验证化合物交集为 0。context reference 只从其他三折 treatment truth
  构造。所有结果为本地 proxy，四折不加权均值。

结果（V1 S1 本地 proxy，单 seed 42）：

| model | n | FC PCC | context residual PCC | high PCC | high F1 | abs sample R² |
|---|---:|---:|---:|---:|---:|---:|
| `M9.0` no chemical | 5,078 | **0.429196** | **0.131844** | 0.541367 | 0.107253 | 不适用 |
| `M9.1` Morgan scratch | 5,078 | 0.334650 | 0.122943 | 0.470212 | 0.108172 | 不适用 |
| `M9.2` OP3 real | 5,078 | 0.362599 | 0.131629 | 0.504149 | 0.113014 | 不适用 |
| `M9.3` OP3 shuffled | 5,078 | 0.309715 | 0.086713 | 0.478143 | 0.109030 | 不适用 |
| `M9.4` L1000 real minus GOAI | 5,078 | 0.362795 | 0.117779 | 0.507710 | 0.114884 | 不适用 |
| `M9.5` L1000 shuffled minus GOAI | 5,078 | 0.351651 | 0.092366 | 0.493914 | 0.120598 | 不适用 |
| `M6.11` 3-seed（协议内旧 absolute 对照） | 5,078 | 0.371674 | 0.098239 | **0.635453** | **0.182438** | 0.979297 |

`M9` 是 response-only 模型。禁止用观测 validation control 重构 absolute prediction，
因此其 absolute R² 明确为不适用，不能与 `M6.11` 的 absolute fidelity 比较。

归因与不确定性（held-out chemical 为簇，10,000 次 paired bootstrap；每次在折内重算
PCC/F1 后四折宏平均）：

- `M9.2 - M9.3`：FC `+0.052884`，95% CI `[+0.022465,+0.076849]`；context
  `+0.044916` `[+0.008911,+0.093330]`；high PCC `+0.026005`
  `[+0.010730,+0.039794]`。OP3 real 相对匹配 shuffle 有可迁移表示信号。
- `M9.4 - M9.5`：FC `+0.011143` `[-0.006808,+0.029744]`；context
  `+0.025413` `[-0.006564,+0.063327]`；两项 CI 均跨 0。
- `M9.4 - M9.0`：FC `-0.066401` `[-0.085359,-0.044784]`；high PCC
  `-0.033657` `[-0.060328,-0.014136]`。RNA consumer 未击败最强独立 context-only。
- 同一 fitted `M9.4` 模型将 validation fingerprints 以 whole-drug derangement 替换后，
  correct-minus-permuted 四折均值为 FC `+0.012020`、context `+0.006904`、high PCC
  `+0.019234`；模型确实读取结构表示，但增益不足以晋级。
- 对保存的同一 `M9.2` 四折模型做同样 replay，correct-minus-permuted 为 FC
  `+0.021591`、context `+0.026970`、high PCC `+0.022970`、high F1 `+0.010453`；
  与 real-vs-shuffle 结果一致，说明 OP3 encoder 含 drug-specific 信号。

架构迭代（`M9.6/M9.7`）：

- direct consumer 会替换并破坏更强的 context predictor，因此冻结每折 `M9.0`，只训练
  zero-init、frozen-RNA chemical residual；residual scale `0.20` 由 seed42/fold0 设计，
  folds1–3 作为未参与该选择的确认子集。未读取 frozen outer validation。
- 三 seed 等权四折 OOF：`M9.6` FC `0.437656`、context `0.139366`、high PCC
  `0.551712`、high F1 `0.108180`；三 seed `M9.0` bag 分别为 `0.435627 / 0.136079 /
  0.544178 / 0.106032`；`M9.7` shuffled bag 为 `0.427173 / 0.123558 /
  0.546507 / 0.108425`。
- 三个 individual seeds 的 `M9.6-M9.0` FC/context/high PCC/high F1 均同向为正。
  三 seed bag 差为 `+0.002029/+0.003287/+0.007534/+0.002148`；chemical-cluster
  bootstrap 中 high PCC CI `[+0.002466,+0.012723]`、high F1
  `[+0.001086,+0.003430]`，FC/context CI 跨 0。
- `M9.6-M9.7` bag：FC `+0.010483` `[+0.004429,+0.016535]`；context
  `+0.015808` `[+0.004706,+0.035921]`，三个 seed 方向一致，支持 RNA-specific 归因。
- 移除设计 fold0 后，folds1–3 / 28 chemicals 的三 seed bag 相对 `M9.0`：FC
  `+0.001322` `[-0.002775,+0.006524]`、context `+0.001637`
  `[-0.004980,+0.018896]`、high PCC `+0.006500`
  `[+0.001538,+0.012475]`、high F1 `+0.002060`
  `[+0.000760,+0.003617]`。additive high-effect 增益通过 untouched-fold 确认。
- 相对 `M6.11`，`M9.6` bag 的 FC/context 为 `+0.065983/+0.041127`，满足本轮
  双指标探索目标；但 high PCC/F1 为 `-0.083742/-0.074258`，且 `M9.6` 不预测
  absolute background，因此未过完整 promotion guardrail。

训练与产物：

- 配置：`/home/chenyuming/Project/goai-rna-transfer/configs/experiment.yaml`；独立源码：
  `/home/chenyuming/Project/goai-rna-transfer/src`；工作区无 git repo，commit 不适用。
- L1000 real/shuffle encoder SHA：`4731c157...57ca` / `689ecf65...b5b`；80 epochs，
  loss `0.56407→0.46295` / `0.56665→0.46519`；约 120/124 秒。
- OOF：`/home/chenyuming/Project/goai-rna-transfer/oof`；完整 summary SHA
  `7810ef43...72f2`，paired bootstrap SHA `f4f4cba1...abdb`；报告：
  `/home/chenyuming/Project/goai-rna-transfer/RESULTS.md`。
- `M9.6` 三 seed persistent bag：
  `/home/chenyuming/Project/goai-rna-transfer/oof/ensembles/op3_real_residual_s02_s42_s43_s2026/S1.npz`
  （SHA `9d784179...496`）；三 seed summary SHA
  `8664e4d1...5799`，bootstrap SHA `30baa9f4...15d0`；untouched-fold summary/
  bootstrap SHA `8f852c9c...1dcc` / `8e7fe7f5...6d14`。
- 可复现测试：13 passed。L1000 formal external CV 未完成，不填数；早期 25-parent
  smoke 已隔离并明确 invalidated，未参与任何结论。
- 工作区不是 git repo。L1000 fold manifest 记录了进程启动时源码 hash；较早的 OP3、
  Morgan 和 context-only fold 产生于该机制加入前，虽保留 config/checkpoint/prediction，
  但 exact runtime source snapshot 缺失，作为明确的复现限制记录。
- 决策：`M9.6` 证明 RNA 表示能作为稳定 additive response residual，但只保留为
  research component；不使用 frozen outer validation、不做旧模型融合/test inference。
  在独立 absolute background 与 high-effect specialist 补齐并重新多 seed 验证前，不晋级。
- 官方提交 ID 与分数：无。GOAI 工作区没有 portal/API/auth/官方 sample schema，且
  4,422 vs 5,243 输出契约及 L1000FWD prize/commercial license 尚未确认；receipt 见
  `/home/chenyuming/Project/goai-rna-transfer/submissions/submission_receipt.json`。

## 10. 迭代日志

### 2026-08-13：`M10.0–M10.8` 相邻竞赛冠军架构迁移（开放知识）

- 模型编号：`M10.0–M10.5` 已实现；`M10.6` 仅预留给完整 OOF 后的融合；
  `M10.7` full fine-tune 与 `M10.8` frozen-trunk 诊断均已完成并拒绝。
- 父模型：无。独立 sibling greenfield 实现，不导入 `M0–M9` 代码、checkpoint、预测或 teacher 特征。
- 状态：实现与审计完成；下列 formal/screen 均使用完整四折 OOF，尚无任何
  `M10` 臂通过 chemical-sensitivity gate 或进入多 seed/融合。
- 假设：转录组扰动、药物 MoA、多模态单细胞竞赛的高维多输出冠军网络，可在仅替换输入/输出维度与连续缺失值损失后迁移到 GOAI protein delta。
- 调整内容：实现 OP3 rank-1 LSTM/GRU/CNN、MoA rank-1 first-stage 3FC、NeurIPS 2021 ADT→GEX rank-1 MLP，并增加 terminal-linear signed-output 适配臂。
- 保持不变：各来源的隐藏宽度、层数、dropout、optimizer、learning rate、batch、epoch、clip/OneCycle 等已核实参数；必要的模态适配均在 manifest 明示。

数据与验证：

- 数据版本：`D0 + D3-ENTITY-20260813 + D5-KAGGLE-TRANSFER-OPEN-20260813`；开放知识榜。
- GOAI S1：5,078 treatment rows、37 chemicals、4,422 proteins；固定四折 sizes `1255/1311/1249/1263`，fold seed 42。
- 私有 S1 cache SHA256：`f42d938a053cd5a88853e43eeb63a714cd941d4de09c42d143075cbdd99806c5`；只读、`0600`。
- OP3 ChemBERTa-MTR：37 chemicals × 1,200 features，artifact SHA256 `fbce74092e008c50c4ed8c5791402be34675d07800136013655647f70ae116a9`；公开模型 revision `66b895cab8adebea0cb59a8effa66b2020f204ca`，特征公式与 rank-1 源码逐值复核 max error `0`。
- 防泄漏：outer validation 不参与 epoch/checkpoint 选择；化学归一化、target scale、context vocabulary 均仅 fit outer-train；训练 context target stats 为 leave-current-chemical-out，outer-valid 只用全部 outer-train；每个 checkpoint 同时生成 normal 与验证 chemical derangement。
- 本地指标：冻结独立 evaluator，四折不加权均值；FC PCC、context residual PCC、high-effect PCC/F1；delta-only absolute R² 不适用。结果不得称为官方 GOAI 分数。

实现与复现：

- 独立目录：`/home/chenyuming/Project/goai-kaggle-model-transfer`；git commit `abd38739aafcba52db9d28cd8d3014aca81fb540`。
- 公开源码：OP3 rank-1 commit `78344dac529bff2cd2686fc09712c2bc3cdbcb41`（MIT）；MoA rank-1 commit `eeba59fe3285938721dc4f47791e1154e3334b06`（Apache-2.0）；NeurIPS 2021 topmethods commit `5782a87a3bb46b30eb264d85cca999724aaaf7d2`（MIT）。
- 测试：9 passed；真实 S1 六架构 forward/backward smoke、ChemBERTa 对齐、checkpoint exact replay（max abs diff `0`）、manifest-bound 四折 collector 均通过。
- 资源状态：两张 A100 在正式派发时仍由同账号另一 BioHub 训练占用约 31.7 GiB/卡；未抢占、未叠加 GOAI CUDA producer。GPU 空闲后先跑六个 1-epoch speed/memory smoke，再按完整四折结果推进。
- 官方提交 ID 与分数：无。GOAI portal/API/auth/官方 sample schema 仍缺失，且 4,422 vs 5,243 输出契约未解决。
- 决策：实现可进入正式 seed-42 四折；在 normal 相对 `-NoChem`、same-fit derangement 及 frozen baselines 同时通过前，不晋级、不与 `M5.2` 融合。

首个完成臂（`M10.3-Morgan-S42`，MoA first-stage 3FC）：

- 四折正式训练全部按公开参数完成：15 epochs、batch 256、Adam `lr=5e-3`、
  `weight_decay=1e-5`、OneCycle `max_lr=1e-2/pct_start=.2/div_factor=1000`；
  CPU 2 threads，总 fit wall time `636.15 s`，失败 0。
- 本地 V1 S1 四折宏平均：FC `0.396813`、context residual `0.089743`、
  high-effect PCC `0.545506`、high-effect F1 `0.106001`。
- 同一 fitted model 的 whole-chemical derangement：FC `0.401612`、context
  `0.100197`、high PCC `0.552585`、high F1 `0.104608`。normal-minus-deranged
  的 10,000 次 chemical-cluster bootstrap：FC `-0.004799`
  `[-0.024590,+0.013762]`；context `-0.010454`
  `[-0.046937,+0.028956]`；high PCC `-0.007079`
  `[-0.023503,+0.009598]`；high F1 `+0.001393`
  `[-0.006624,+0.010596]`。没有可靠 chemical-specific signal。
- 架构匹配 `M10.3-NoChem-S42` 为 FC `0.417022`、context `0.117179`、
  high PCC `0.562222`、high F1 `0.093408`。Morgan-minus-NoChem bootstrap：
  FC `-0.020209` `[-0.029588,-0.008920]`；context `-0.027435`
  `[-0.050241,-0.005220]`；high PCC `-0.016716`
  `[-0.026616,-0.004883]`；仅 high F1 `+0.012593`
  `[+0.006261,+0.021107]`。化学输入显著损害三个相关性指标。
- 产物：normal OOF SHA `f4efa737...ddcbe`；deranged OOF SHA
  `686c5be0...a6312`；主 summary SHA `67cdc58d...b67f`；derangement bootstrap
  SHA `617775af...b198`；Morgan-vs-NoChem bootstrap SHA `ccbebbe8...344b`。
- 决策：`M10.3-Morgan-S42` 拒绝，不跑多 seed，不进入 `M10.6`。它的较高 pooled
  FC 主要来自 context/background；打乱或移除药物表示不会使结果变差。`M5.2` 不变。
- `M10.0-MTR-S42` 第一次正式 GPU 尝试在 fold0 运行 `4m44s` 后发现另一个用户进程
  新进入同一 GPU，按预注册独占门禁安全 SIGINT 停止（exit 130）。未生成 prediction、
  checkpoint、SUCCESS 或 complete manifest，未收集/评分；immutable runtime YAML 与
  startup manifest 保留为失败证据。该次不计入结果，须以新 experiment ID 从 fold0 重跑。
- `M10.0-MTR-S42` retry1 在 GPU0 通过两次空闲检查后启动，fold0 运行 `3m13s`
  时又检测到新的外部 PID；10 秒守护只中止本次 GOAI 进程。仍无 prediction/checkpoint/
  SUCCESS/complete manifest，不计入结果。retry2 的启动门提高为 GPU 连续独占空闲 5 分钟，
  并在每折前重复检查；所有被抢占尝试均保留 immutable startup evidence。
- retry2 在 GPU0 连续 5 分钟（10/10 次）无进程、约 40.4 GiB free、util 0 后启动，
  但约 10 秒后再次有两个外部 producer 进入；本次模型尚在 CPU 预处理即于 `15.95s`
  安全停止。连续三次即时抢占表明需要显式 GPU 预约/协调，不再无上限机会式重试。

第二个完成臂（`M10.5-Morgan-S42`，Novel signed linear head）：

- 四折正式训练按公开代码参数完成：100 epochs、batch 64、Adam `lr=0.00041`、
  `weight_decay=0.0000139`、masked RMSE；CPU 2 threads，失败 0。
- 本地 V1 S1 四折宏平均：FC `0.410819`、context residual `0.041099`、
  high-effect PCC `0.560273`、high-effect F1 `0.162372`。
- 同一 fitted model whole-chemical derangement：FC `0.411786`、context
  `0.048647`、high PCC `0.561918`、high F1 `0.161656`。normal-minus-deranged
  的 10,000 次 chemical-cluster bootstrap：FC `-0.000966`
  `[-0.013647,+0.013304]`；context `-0.007548`
  `[-0.030752,+0.022324]`；high PCC `-0.001645`
  `[-0.010898,+0.008760]`；high F1 `+0.000716`
  `[-0.003901,+0.005698]`。四项均未通过 chemical-sensitivity gate。
- 产物：normal OOF SHA `089dc37f...5d157`；deranged OOF SHA
  `4d7d2104...5ed74`；summary SHA `6075a102...56416`；bootstrap SHA
  `b0521f9c...3c47c`。
- 决策：`M10.5-Morgan-S42` 拒绝，不跑多 seed、不进入 `M10.6`。虽然 pooled FC 与
  high-effect F1 较高，但正确药物配对不优于置乱，不能归因于可泛化化学表示。

OP3 successive-halving（25/250 epochs；显式 discovery budget）：

- `M10.2-MTR-S42` CNN 四折完成：normal FC `0.338620`、context `0.028652`、
  high PCC `0.521875`、high F1 `0.089842`；deranged 为 `0.374798/0.089688/
  0.549010/0.093056`。normal-minus-deranged bootstrap：FC `-0.036178`
  `[-0.064667,-0.004034]`；context `-0.061037`
  `[-0.104408,-0.017635]`；high PCC `-0.027135`
  `[-0.045341,-0.001757]`；high F1 `-0.003214`
  `[-0.016769,+0.010611]`。三个相关性指标显著为负，successive-halving 拒绝，
  不补到 250 epochs、不跑多 seed。
- `M10.0-MTR-S42` LSTM 四折 25-epoch screen：normal FC `0.343581`、context
  `0.043330`、high PCC `0.519740`、high F1 `0.094691`；deranged 为
  `0.371791/0.099187/0.545831/0.104774`。normal-minus-deranged bootstrap：
  FC `-0.028211` `[-0.066615,+0.008898]`；context `-0.055857`
  `[-0.110158,-0.005505]`；high PCC `-0.026091`
  `[-0.051344,+0.004455]`；high F1 `-0.010083`
  `[-0.025497,+0.005350]`。context chemical-sensitivity 显著为负，successive-halving
  拒绝，不补到 250 epochs、不跑多 seed。
- `M10.1-MTR-S42` GRU 四折 25-epoch screen 完成：batch 16、Adam
  `lr=3e-4`、gradient clip `5`、30% zero-feature doubled augmentation；四折 CPU fit
  wall time `3149.80 s`，失败 0。normal FC `0.354576`、context `0.038647`、
  high PCC `0.528999`、high F1 `0.111419`；deranged 为
  `0.379294/0.091939/0.548152/0.120929`。normal-minus-deranged 的 10,000 次
  chemical-cluster bootstrap：FC `-0.024718` `[-0.052776,+0.004983]`；
  context `-0.053292` `[-0.098536,-0.006708]`；high PCC `-0.019152`
  `[-0.041360,+0.006348]`；high F1 `-0.009510`
  `[-0.027616,+0.008342]`。normal OOF SHA256
  `fdbfcb0f2163de65cf3565cbd9522fbdb0229df1c7db105c0ad45ec677e91a7e`；
  deranged OOF SHA256
  `4a28635ff888d53e4f37a1f79abe1e1adc8d1cf1bcc6a7c5a8d263e0c435ffae`；
  summary SHA256
  `3563fc341f835918a242fd29e23eb1ec30b24cc1b3283f1ae6d19e82361a6b9d`；
  bootstrap SHA256
  `85ff0167e7220a570e8de23c001753b1cbf5222f83360b4d51001c6009bfcc73`；
  collected manifest SHA256
  `75d7160ded0e9e2eab1591aab87cbfb5d625ee4ba797df2d766a730511f580dd`；
  evaluation manifest SHA256
  `ff9381a442e893c742ee4a3d938a3c8fe938536ab3998c1fc5f354e8f37443e3`。
  主产物位于
  `/home/chenyuming/Project/goai-kaggle-model-transfer/oof/ensembles/screen25_s42_m10_1_op3_gru_mtr_cpu/`，
  评分位于
  `/home/chenyuming/Project/goai-kaggle-model-transfer/logs/evaluation/screen25_s42_m10_1_op3_gru_mtr_cpu_normal_vs_permuted/`。
  四项点估计均为负，且 context chemical-sensitivity 的 95% CI 整体小于 0；
  successive-halving 拒绝，不补到 250 epochs、不跑多 seed、不进入 `M10.6`。

`M10.7-Morgan-S42` chemical-contrast residual full fine-tune：

- 每个 outer fold 从对应 same-fold `M10.3` checkpoint warm-start trunk，reset
  residual head 后全量 fine-tune；固定四折 formal fit wall time `840.102349 s`，
  失败 0，四折 checkpoint replay max absolute difference 均为 `0`。实现 git commit
  `a6e816217d82f3290541ab0ed9ba36c2d6526db3`。
- normal 四折宏平均：FC `0.441819`、context residual `-0.059985`、
  high-effect PCC `0.611169`、high-effect F1 `0.164524`。same-fit PermChem 为
  `0.449175/-0.003081/0.624891/0.166235`；NoChem 为 FC `0.472500`、
  context residual `NaN`、high PCC `0.626251`、high F1 `0.164948`。
- normal-minus-PermChem 的 10,000 次 chemical-cluster bootstrap：FC `-0.007356`
  `[-0.037155,+0.021871]`；context `-0.056904` `[-0.134154,+0.022197]`；
  high PCC `-0.013722` `[-0.034253,+0.010042]`；high F1 `-0.001711`
  `[-0.007643,+0.006591]`。normal-minus-NoChem：FC `-0.030681`
  `[-0.046392,-0.014051]`；high PCC `-0.015082` `[-0.025264,-0.003008]`；
  high F1 `-0.000424` `[-0.005701,+0.002938]`；NoChem context 为 `NaN`，
  因此不声称或填造该项比较。
- 产物：normal OOF SHA256
  `a4def2c110c6de2f002525c900d5f6739b73501fc269b0426d3ae1d881a88f6c`；
  PermChem OOF SHA256
  `db2264ffb8428cfe96e8860e7111f83ec9523a028cce282757be1c3b6cb7f8e2`；
  NoChem OOF SHA256
  `b3c3cc7f9b0fcf1c640ed36cc4035f10c4a456b170363645a3f8180fda812d12`；
  collection manifest SHA256
  `d55b59ee244e3cda56865ccd6d56b00f14cb530c791a39e6b7e473f379ecc0d5`；
  summary SHA256
  `bffb3236b53a7f995509ee42deea4f8b2a0c9c25ae723273f1b3c6e640c0c057`；
  bootstrap SHA256
  `f3a7f1413dd75b8a87c2c4a2cd0da8c3ac885751fb443d7acc817d7dfae2b900`；
  evaluation manifest SHA256
  `95f140283fe5743d1b79f9d62544a3c0c2a9f619f78a64989b5104fb3b654b68`。
  OOF 位于
  `/home/chenyuming/Project/goai-kaggle-model-transfer/oof/ensembles/formal_s42_m10_7_moa_chemcontrast_residual_cpu_commit_a6e8162/`；
  评分位于
  `/home/chenyuming/Project/goai-kaggle-model-transfer/logs/evaluation/formal_s42_m10_7_moa_chemcontrast_residual_cpu_commit_a6e8162_paired/`。
- 决策：拒绝 `M10.7`。normal 没有击败 same-fit PermChem，且相对 NoChem 的
  FC 与 high-effect PCC 均显著变差；不扩展 seeds 43/2026、不进入 `M10.6`
  融合、不改变当前冻结 `GOAI-M5.2`。`M10.8` 只能作为 frozen-trunk
  归因诊断，不是对 `M10.7` 的追加 outer 调参。

`M10.8-Morgan-S42` frozen-trunk chemical-contrast residual 诊断：

- 每个 outer fold 复用对应 same-fold `M10.3` trunk，冻结 trunk、context
  embeddings、BatchNorm buffers 与 dropout state，reset weight-normalized residual head 后
  仅训练 `weight_g/weight_v`；`dense3` bias 保持精确零且冻结。四折全部使用
  2 CPU threads，formal fit wall time `437.349264 s`，失败 0；四折 checkpoint
  replay max absolute difference 均为 `0`，frozen state equality 全部通过。实现
  git commit `e8e187d225a322fec57704a67571375650a08cfc`，implementation source
  SHA256 `fb9286d0a6e81c3de5281c0d53f6a61f8869e41a48688f3bbceeab5c572b3ed4`。
- normal 四折宏平均：FC `0.468183`、context residual `-0.045868`、
  high-effect PCC `0.621298`、high-effect F1 `0.164893`。same-fit PermChem 为
  `0.471125/0.007588/0.624026/0.165006`；NoChem 为 FC `0.472500`、
  context residual `NaN`、high PCC `0.626251`、high F1 `0.164948`。
- normal-minus-PermChem 的 10,000 次 chemical-cluster bootstrap：FC `-0.002942`
  `[-0.008442,+0.003061]`；context `-0.053456` `[-0.128249,+0.035949]`；
  high PCC `-0.002727` `[-0.009032,+0.003831]`；high F1 `-0.000113`
  `[-0.001169,+0.001046]`。normal-minus-NoChem：FC `-0.004317`
  `[-0.007551,-0.000529]`；high PCC `-0.004952` `[-0.009542,-0.000318]`；
  high F1 `-0.000055` `[-0.000792,+0.000880]`；NoChem context 为 `NaN`，
  不声称或填造该项比较。
- 从实际冻结文件复核的产物：normal OOF SHA256
  `0e69329ec8b253c41ec2b56ad623f75b0653c0096aab22731202ff908625d854`；
  PermChem OOF SHA256
  `ab6a4a246dbad1e85c325bf7e0019076a1dafd29bc4c7103da10ec9c01bb7176`；
  NoChem OOF SHA256
  `b3c3cc7f9b0fcf1c640ed36cc4035f10c4a456b170363645a3f8180fda812d12`；
  collection manifest SHA256
  `2bfc1f08d095baf4a7727ba4dadfdbd26cd26945ef8d706095328883bd683fb9`；
  summary SHA256
  `656d6c316f22d36a8f4a35c7debff6b03abeb6366aa872512a9608b06020ea76`；
  bootstrap SHA256
  `ff205588464dcda924769b3ad83ac22e395cb6eda509adb78e995d854700e629`；
  evaluation manifest SHA256
  `6aa5a56862d7fd2030b92a2e02acf848d028c028cfc9dbd501e606cf89954d18`。
  OOF 位于
  `/home/chenyuming/Project/goai-kaggle-model-transfer/oof/ensembles/formal_s42_m10_8_moa_chemcontrast_residual_frozen_cpu_commit_e8e187d/`；
  评分位于
  `/home/chenyuming/Project/goai-kaggle-model-transfer/logs/evaluation/formal_s42_m10_8_moa_chemcontrast_residual_frozen_cpu_commit_e8e187d/`。
- 决策：拒绝 `M10.8`。冻结冠军 trunk 后 normal 仍未击败 same-fit PermChem，
  且相对 NoChem 的 FC 与 high-effect PCC 显著变差；不扩展 seeds
  43/2026、不进入 `M10.6` 融合、不改变当前冻结 `GOAI-M5.2`。

### 2026-08-12：历史模型统一登记

- 建立 `M0`–`M5` 编号体系。
- 将历史基线、响应分解、化学/SVD/loss/PPI 消融、两代图模型和最终融合系统统一映射。
- 当时的最终模型定为 `GOAI-M5.1`；2026-08-13 已由 `M5.2` 取代。
- 本次只整理事实和编号，没有重新训练或改变预测值。

### 2026-08-13：`M6` 全面 OOD 夜间实验与 `M5.2` 冻结

- 建立固定 S1/S2/S3/time/time-forward/plate OOF 协议并执行 cell-conditioned、calibration、rank/SVD、prototype、ChemBERTa 及负对照矩阵。
- 共完成 53 个正式 matrix producer，失败 0；完成 3 个 outer confirmation、6 个全标注 refit 和最终路由推理。
- `M6.11` 晋级 S1，`M6.21` 晋级 time；S2/S3 保留 `M5.1`。
- Huber 小权重、concat-on-S2、去 plate、fixed SVD-256、直接 ChemBERTa concat、response prototypes 均按多指标护栏拒绝。
- `GOAI-M5.2` 预测契约验证通过；官方提交仍被缺失端点/契约阻塞。

### 2026-08-13：`AUDIT-P01` 平台期根因审计

- 复核最终 refit 特征状态，确认 chemical/strain semantic dimensions 均为 0；11 个 test-only chemicals 与 CRD 均回退到全零未知编码。
- 量化 inner-train FC 信号、control 重复噪声、跨批次同条件一致性和 high-effect 稀疏度。
- 量化 test-only chemical 到 36 个 inner-train chemicals 的 Morgan 最近邻距离，确认 test chemical OOD 比现有 validation 更远。
- 明确 `0.371674` 是 pooled local FC proxy，不是官方 PSS，且 S1 context residual 只有 `0.098239`。
- 不产生新模型编号，不修改 `M5.2`；后续主线从“同输入扩网络”转向实体语义、去噪和特异 residual。

### 2026-08-13：`M7/M8` 通用主干与实体专家实现

- 确认并登记 `M7.0–M7.4` 与 `M8.0–M8.3`，保留 `M5.2` 为冻结基线。
- 实现 universal biological trunk、fold-fit strain/chemical/pair experts、共享 protein decoder、
  staged training、canonical support router、组件导出和 calibration 硬中心化。
- 建立并审计 57 chemical/6 strain registries，修正 Cyclopiazonic acid 与
  Doxycycline hyclate；二次来源审计又修正 LY294002 hydrochloride 盐型与 Hoechst 33258
  错误 CID。11 个 test-only chemicals 中 8 verified、3 candidate，另保留 3 个显式 proxy
  和 DHY210 missing boundary。
- 构建 1,011 yeast SNP-MDS 菌株 real/shuffled features，但五个映射在组织方确认前
  仍只允许 `M8` research screen。
- 新增 R00/R10/R01/R11/RT、source-fingerprinted producer contract、scale/confirm/calibration matrices
  与严格负对照；消除 R10/R01 重复 fit。
- 正式 M8 输入新增 mapping → embedding manifest → selected TSV → checkpoint 硬链；
  训练与预测均拒绝 mapping/manifest/TSV 漂移，无 manifest 的旧输入只标 legacy-unverified。
- 独立全测 `187 passed`；真实 test support 路由计数与计划精确一致。
- `M7.0-S42` 和 `M7.3-S42` 两折 discovery OOF 共完成 28 个 fold fits，失败 0；
  `M7.3` 在 R10/R11/RT 有明确正信号，但 R01 未过 residual 护栏，R00 下降且受
  universal epoch 不等的 schedule 混杂影响。两者均不晋级，完整结果与后续公平消融要求
  已记录在 9.7 节。
- 实现 fold-matched universal receipt、共同 state 严格复制、冻结专家/joint 分离，以及
  7 项/98-fit 公平实验矩阵；独立复核在 learned/fixed-SVD 与两张 A100 上通过共同状态
  bitwise SHA 收据。
- 新增正式 confirmation/promotion 硬门禁、严格身份语义覆盖门禁、Calibration
  确定性选择收据。formal M8 因当前 verified-only 训练语义覆盖为 0 而预检阻断；
  research M8 只允许作为不可晋级的方向屏查。全项目全测最终为 `187 passed`。
- R2 overnight 完整顺序为 fair 98 + quick 328 + research 86 + calibration 112，
  共 `624` fits；状态、GPU 门禁读数、阶段命令与返回码写入大盘原子收据。
  当前尚未新增模型结论或官方成绩。

### 2026-08-15：`GOAI-M11.0` M9 融合与逐行路由提交候选

- 模型编号：`GOAI-M11.0`。
- 父模型：`GOAI-M5.2`、`M6.11`、`M9.6`。
- 状态：**冻结主提交候选**；官方提交被外部入口/合同阻塞。
- 假设：`M9.6` 的 OP3 RNA 预训练 chemical residual 在新化合物上有药物特异信号，
  但它是 response-only 模型，不应覆盖新菌株、双未知、time 和 control 路径。因此只在
  当前 refit support 明确为“菌株已见、化合物未知”的 R10 行替换 Response。
- 调整内容：三种子 refit `M9.6`；导出三种子 `M6` 的 `B/C/R` 组件；在严格 S1 OOF
  上选择完整 Response 融合；实现逐行 canonical support router；补做 entity-expert
  residual 与 strain-semantic 接入实验。
- 保持不变：`M5.2` 原预测文件不覆盖；R01、R00、R11 和 control 逐值保留；最终仍输出
  绝对 log2 蛋白丰度，而不是只输出 FC。

数据：

- 数据版本：`D0 + D2 + D3-ENTITY-20260813 + D4-RNA-OPEN-20260813`。
- 全量 refit：8,958 条已发布标签；每个 `M9.6` seed 使用 7,884 个 exact-control
  treatment FC；测试 4,454 行。
- 输出蛋白数：4,422。
- model seeds：`42/43/2026`。
- support manifest：
  `/home/chenyuming/Project/go-ai/data/processed/entities/support_manifest_fit_all_labeled.json`，
  content SHA256 `9869251495f56970f206aa9a80ee6cf101018a1cb3388751040c627b00146e17`。
- `M9.6` 三个 refit checkpoint SHA256：
  `34a94594b57227609db5e699f679cddf43f40e0d6ac07bebcd70899665a3c50d`、
  `46093a47b16e6eab94dd887baeeff4b7573904ba84fc5aaf8a0b706e46bab454`、
  `0e8e328f2d04e660515d576ed23e591eca83dc0bf02ab756314a86391971a233`。

R10 网络与融合：

```text
M6: shared cell encoder -> Background B6
                      \-> conditional Response R6
observation metadata ----> independent Calibration C6

M9.6: Morgan-2048 -> frozen OP3 2048->256->64 encoder
                    + context-gated zero-init residual -> R9.6

M11 R10 = B6 + C6 + (1 - 1.05) * R6 + 1.05 * R9.6
```

`M9.6` 使用 mask-aware high-effect weighted SmoothL1，加 `0.02` correction shrinkage；
推理 residual scale 为 `0.20`。`M6` 与 `M9.6` 都先在 seed 内预测，再对
`42/43/2026` 等权平均。详细结构和 loss 见
`/home/chenyuming/Project/go-ai/runs/final/goai_m11_0_20260815/MODEL_CARD.md`。

严格 S1/R10 OOF 结果：

| 模型 | n | FC PCC | Context PCC | High PCC | High F1 | Abs R2 |
|---|---:|---:|---:|---:|---:|---:|
| M6 core | 5,078 | 0.371973 | 0.098530 | 0.635746 | 0.182687 | 0.979300 |
| M9 complete Response replacement | 5,078 | 0.426065 | 0.064850 | 0.603782 | 0.233621 | 0.979471 |
| **M11 blend w=1.05** | 5,078 | **0.426139** | 0.062244 | 0.600870 | **0.233938** | 0.979193 |
| OP3 conservative residual g=1.3 | 5,078 | 0.372978 | **0.099002** | **0.638486** | 0.186087 | 0.979048 |

- M11 blend 相对 M6 core 的 37-chemical cluster bootstrap：FC `+0.054166`，
  95% CI `[+0.027482,+0.075783]`；high F1 `+0.051251`，
  95% CI `[+0.045035,+0.056421]`。
- 代价：context residual PCC `-0.036286`，high-effect PCC `-0.034875`。因此这是按
  当前 FC 主目标选择的刷分候选，不声称所有指标全面提升。
- 证据：
  `/home/chenyuming/Project/go-ai/runs/score_sprint/20260814/m6_m9_fusion_s42_s43_s2026/`；
  summary SHA256 `febbaa0226ee043790bf881e1c9202994ecd996e9c8b3b18088b9d91314eb631`；
  bootstrap SHA256 `9cbd1055a6196f948f7b1077afa5025364f58755ade4f89315be0f6b5658961d`。

专家和菌株语义结果：

- entity expert residual scale `0/0.25/0.5/0.75/1` 的 R10 FC PCC 为
  `0.426139/0.425123/0.415428/0.398752/0.377530`，最优为 0；不进入主模型。
- `M2 + strain semantics` 权重 `0/0.25/0.5/0.75/1` 的 S2 FC PCC 为
  `0.281731/0.275621/0.266490/0.255106/0.242374`，所有非零权重下降。
- 独立 real-vs-shuffled 菌株语义在 R01 为 `0.191269 vs 0.182049`，real `+0.009220`；
  R00 real 反而低 `0.001161`。由于只有 4 个训练菌株、fold 内仅 3 个实体，不足以替换
  当前稳定 R01/R00 路由。
- `M11.2` 已生成，R10 使用 M9 融合、R00 使用语义 Background/M9 Response；但 R00
  没有正负对照晋级证据，只作为研究候选，prediction SHA256
  `969facbb70f9f2936353ab18228adc3aa1cb4f41b6a6bf512ff4033763a5f4ae`。

最终 test 路由：

| support regime | treatment rows | 最终路径 |
|---|---:|---|
| R10：菌株已见、药物未知 | 2,072 | M6 Background/Calibration + M6/M9.6 Response blend |
| R01：菌株未知、药物已见 | 1,594 | 保留 M5.2/M2 |
| R00：双未知 | 425 | 保留 M5.2/M2 |
| R11：双已见、pair/time 目标未知 | 135 | 保留 M5.2/M6.21 time |
| Water/DMSO/QC | 228 | 保留 M5.2 Control/Background |

其中 `test_both` treatment 精确拆为 R10 `432`、R01 `272`、R00 `425`；不再按整个
`split_final` 粗路由。只有 R10 的 2,072 行相对 `M5.2` 改变，其余逐值相同。

最终产物：

- 目录：`/home/chenyuming/Project/go-ai/runs/final/goai_m11_0_20260815/`。
- prediction SHA256：
  `c4c588e8573d0e38441fbee01dd66dfda81e338498c28be6a9a5d2a6c7379bab`。
- route audit SHA256：
  `16b2b1967cff8e31e131fb83a11a27c0399e0e8df1e94432ba4a0d5b922c40c1`。
- consumer script SHA256：
  `8d071653de60cbc90aeeabe9c1ac80f9fc9dd713f10260605d34f42c23211840`。
- 本地合同：4,454 × 4,422、sample ID 顺序匹配、全部 finite；M6 组件重建最大误差
  `3.814697e-06`；全项目测试 `188 passed`。
- 冻结回退：`GOAI-M5.2` prediction SHA256
  `919e20020b7bff836058d0be36411c412193e4ccba92f112ff2854575f6c59b7`。

决策与官方状态：

- `M11.0` 晋级为冻结主提交候选；`M5.2` 不删除、不覆盖，作为回退。
- 本地 M9 融合权重选择与报告使用同一冻结 OOF 表面，存在选择偏差；正式结果仍需官方
  分数验证。
- `M9.6` 使用 OP3 RNA 外部预训练，提交前必须确认赛事外部数据规则和许可。
- 官方提交 ID/官方分数：无。工作区没有官方 portal/API/auth/scorer，也缺能解决
  4,422 vs 5,243 列合同的权威 sample submission；精确阻塞和下一动作见
  `/home/chenyuming/Project/go-ai/runs/final/goai_m11_0_20260815/submission_receipt.json`。

### 2026-08-15：`GOAI-M12.0` 与 M12 语义/专家审计

- 模型编号：`GOAI-M12.0`；父模型 `GOAI-M11.0`。
- 状态：**当前本地 FC 最优完整候选**；`M12.1/M12.2` 与 seen-strain expert overlay 均完成并拒绝。
- 调整：R10 使用 high-effect specialist：
  `blend=-0.075*R6+1.075*R9.6`，在 `abs(R6)>=0.5` 位置以 0.15 拉回 R6。
- 严格 S1/R10 三种子 OOF：FC `0.426342`、Context `0.060967`、High PCC
  `0.603184`、High F1 `0.233970`、Abs R2 `0.979150`。
- 相对 M11.0：FC `+0.000203`，chemical bootstrap 95% CI
  `[+0.000059,+0.000355]`；High PCC `+0.002314`；Context `-0.001277`。
- M12.1：scaled SNP-MDS-4 real 相对 shuffled 的 FC `+0.006699`，CI 下界为正，
  但相对 zero-semantic M2 的最佳 FC 权重为 0；拒绝全量 refit与路由。
- seen-strain expert：standalone expert 稳定改善弱 general，但
  `M12.0+alpha*(expert-general)` 在 alpha 0.25 即降至 FC `0.425307`；最佳 alpha 0。
- M12.2 R00：RR/SR/RS/SS/ZZ 16-fold 完成。ZZ FC `0.210603`、RR `0.185253`、
  最佳 RR/SS mix `0.192641`；RR 对 ZZ delta `-0.025350`，95% CI
  `[-0.034241,-0.017386]`。R00 保留 M5.2/M2。
- 最终 prediction：4,454 × 4,422 absolute log2，SHA256
  `4179afee866920ef7df6da99025c17e26c3b901647591691562884af7e8159ab`。
- 产物：`runs/final/goai_m12_0_fine_20260815/`；完整报告：
  `docs/experiments/m12_execution_results_20260815.md`。
- 提交候选包：`deliverables/GOAI-M12.0_submission_candidate_20260815/`；候选预测固定命名为
  `submission/prediction_4422_local_contract.csv`，并附路由审计、模型卡、checkpoint 哈希索引
  和机器可读 artifact manifest。该命名刻意保留 `4422_local_contract`，在官方 schema
  解决前不得直接冒充已验收 submission。
- 初赛算法赛作品附件按实际页面规则另行整理为
  `deliverables/AI4R_ALG_AIVC_队伍名待替换.zip`：基于官方算法赛初赛模板的 DOCX/PDF、
  干净训练/推理源码、主要结果表/图和外部资源披露合并在一个 ZIP；不包含 prediction、
  官方数据、权重、OOF/cache 或完整日志。占位 ZIP 为 769,258 bytes，SHA256
  `92342e5a3f0e16a6909fb791fd600844c2ef44891141b466885dff36adc93e70`；完成真实队伍名、
  成员背景/分工/成果后必须重新生成并以 `AI4R_ALG_AIVC_实际队伍名.zip` 上传。
- 测试：`go-ai 204 passed`；`goai-rna-transfer 14 passed`。
- 官方提交：未执行；无官方端点/auth/scorer，4,422/5,243 合同及 OP3 外部数据许可未解决。

## 11. 新实验追加模板

复制下面一节到“迭代日志”末尾；没有的数据写“无”或“不适用”，不要删除字段。

```markdown
### YYYY-MM-DD：M?.? 实验标题

- 模型编号：
- 父模型：
- 状态：计划 / 运行中 / 完成待判断 / 晋级 / 保留对照 / 拒绝
- 假设：
- 调整内容：
- 保持不变：

数据：

- 数据版本：D0 / D1 / 新版本编号
- 训练样本数：
- 验证样本数与覆盖率：
- 输出蛋白数：
- 输入文件及 SHA256：
- 外部资源、版本、来源及 SHA256：

训练与验证：

- 配置文件：
- 代码版本/commit：
- 运行目录：
- checkpoint：
- 训练设备与耗时：
- fold 协议与 fold seed：
- model seeds：
- 防泄漏检查：

结果（本地/官方必须注明）：

| split | n samples | FC PCC | ΔFC vs parent | abs R2 | high PCC | high F1 | 其他 |
|---|---:|---:|---:|---:|---:|---:|---|
| S1 | | | | | | | |
| S2 | | | | | | | |
| S3 | | | | | | | |
| time | | | | | | | |

- 多 seed 均值/标准差：
- bootstrap CI 或显著性：
- 失败模式：
- 官方提交 ID 与分数：无 / 填写真实记录
- 产物路径和 SHA256：
- 决策：
- 决策理由：
- 下一步：
```

## 12. 待回填事项

- `M4.0.1`、`M4.0.2`、`M4.1.0` 虽有代码入口，但当前没有留存的正式结果。
- `M2.10-S17/S2026` 的旧冻结指标记录在历史交付文档，当前项目运行目录中未登记对应 checkpoint 路径。
- `GOAI-M5.1` 尚无官方提交 ID、官方 PSS 或排行榜分数；提交后必须回填本台账。
