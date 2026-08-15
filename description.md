# GOAI-M12.0 架构与原理：从原始数据到最终 4,422 维预测

版本：2026-08-15
模型编号：`GOAI-M12.0`
状态：当前本地严格 OOF 下 FC 最优的完整候选
重要口径：本文所有分数都是本地验证，不是官方 PSS 或排行榜成绩

---

## 0. 先用一句话说清楚 M12.0

`GOAI-M12.0` 不是“一个输入表、一个 MLP、一个输出表”的单网络，而是一个逐行
判断已见/未见状态的组合系统：

1. `M2.0/M2.31` 提供稳定的背景、扰动和测量校准 fallback；
2. `M6.11` 让药物响应读取同一个 cell state，负责新化合物路线的背景与响应；
3. `M6.21` 用 FiLM 交互处理已见实体上的 time extrapolation；
4. `M9.6` 把外部 RNA 扰动知识压进分子编码器，专门增强未见化合物的响应；
5. support router 根据全量训练时真正出现过的 canonical strain/chemical，逐行选择
   R10、R01、R00、R11 或 control；
6. 只有 R10 使用 M6/M9.6 融合，其他路线保留已经验证更稳的 M5.2 fallback。

最终 R10 公式为：

```text
blend = -0.075 * R6 + 1.075 * R9.6
gate  = I(abs(R6) >= 0.5)
R12   = blend + 0.15 * gate * (R6 - blend)
y_hat = B6 + C6 + R12
```

这里的 `I(...)` 是对每个样本、每个蛋白分别计算的硬门，不是每行一个标量。

---

## 1. 任务到底是什么

每个样本有两类输入：

- 生物条件：菌株、化合物、培养基、温度、时间；
- 观测条件：数据来源、仪器、plate。

模型输出该样本固定蛋白顺序下的完整 log2 蛋白丰度向量：

```text
input:  one metadata row
output: [protein_1, protein_2, ..., protein_4422] in log2 scale
```

训练标签原始值是蛋白丰度，矩阵包含缺失。模型不能把缺失值当成 0，也不能让缺失
位置参与 loss。最终本地合同保留 4,422 个在训练侧满足可用条件的蛋白。

### 1.1 为什么不能只预测 absolute abundance

绝对蛋白丰度的主要变化通常由以下因素共同决定：

- 菌株和培养条件形成的基础细胞状态；
- 药物带来的真实扰动；
- plate、instrument、data source 带来的测量偏移。

如果直接把所有输入丢进一个大 MLP，absolute MSE 很容易被高丰度蛋白和公共背景主导。
模型即使 absolute R2 很高，也可能根本没有学会 treatment 相对 control 的变化。

因此从 M2 开始，我们显式建模：

```text
absolute = Background + Calibration + treatment_gate * Response
```

这一改变是 M1 到 M2 出现最大跃升的根本原因。

---

## 2. 数据预处理与监督目标

## 2.1 原始蛋白值转 log2

设原始蛋白丰度为 `x[i,p]`，样本为 `i`，蛋白为 `p`：

```text
observed[i,p] = isfinite(x[i,p]) and x[i,p] > 0
y[i,p]        = log2(x[i,p])              if observed
                 NaN                       otherwise
mask[i,p]     = 1                          if observed
                 0                          otherwise
```

所有损失均乘以相应 mask，再除以有效元素数。缺失位置既不会贡献误差，也不会被模型
当作“真实值为 0”。

## 2.2 蛋白级标准化

每个 fold 只用 fold-train 行计算每个蛋白的均值和标准差：

```text
mu[p]    = mean(y_train[:,p], ignoring NaN)
sigma[p] = max(std(y_train[:,p], ignoring NaN), scale_floor)
z[i,p]   = (y[i,p] - mu[p]) / sigma[p]
```

模型内部预测标准化值 `z`。推理时还原：

```text
y_hat_absolute[i,p] = z_hat_absolute[i,p] * sigma[p] + mu[p]
```

Response 表示相对变化，所以还原时只乘 `sigma[p]`，不再加 `mu[p]`：

```text
delta_hat[i,p] = response_standardized[i,p] * sigma[p]
```

## 2.3 exact matched control

对 treatment 样本，control 必须在以下字段上精确一致：

```text
data_source
instrument
Yeast_cell_plate
Strains
Medium
Temperature
pert_time
pert_time_unit
```

Water 和 DMSO 是 control，Quality Control 单独隔离。若同一键有多个 control，则先对其
log2 蛋白值按蛋白求均值。于是 treatment 的真实 response 为：

```text
delta[i,p] = y_treatment[i,p] - mean(y_exact_control[:,p])
```

只有 treatment 和 matched control 在该蛋白上都可用时，`delta_mask[i,p]=1`。

这一步非常关键：Response loss 直接监督真实 FC，而不是希望网络从 absolute loss 中
自己“悟出”差分。

## 2.4 treatment gate

```text
treatment_gate = 0 for Water, DMSO, Quality Control
treatment_gate = 1 for actual perturbation chemicals
```

最终结构强制：

```text
absolute = background + calibration + treatment_gate * response
```

因此 control/QC 永远不会错误地经过药物响应分支。

---

## 3. M2/M6 的输入是怎么构造的

以下维度来自最终 full-refit checkpoint，而不是从配置猜测。

## 3.1 时间编码：4 维

训练侧最大时间为 `Tmax=240`。原始时间 `t` 转成：

```text
t1 = t / Tmax
t2 = log(1+t) / log(1+Tmax)
t3 = sin(2*pi*t/Tmax)
t4 = cos(2*pi*t/Tmax)
```

线性项支持趋势，log 项压缩长时间差，sin/cos 提供周期/相位表达。时间不是简单类别
one-hot，因此至少保留一定的连续外推能力。

## 3.2 one-hot 词表

full-refit 使用全部 8,958 条已发布有标签行拟合词表：

| 字段 | 维度 | 未见值处理 |
|---|---:|---|
| `Strains` | 5 | 全零 |
| `perturbation_no_concentration` | 46 | 全零 |
| `Medium` | 2 | 全零 |
| `Temperature` | 2 | 全零 |
| `data_source` | 4 | 全零 |
| `instrument` | 7 | 全零 |
| `Yeast_cell_plate` | 144 | 全零 |

“全零”不是说所有新药相同的化学结构，而是说 M2/M6 的 **ID one-hot 分支**无法区分
训练词表之外的不同实体。M9.6 的 Morgan/OP3 分支正是为了解决新药在结构上不可区分
的问题。

## 3.3 cell input：13 维

```text
strain one-hot       5
medium one-hot       2
temperature one-hot  2
time encoding        4
----------------------
cell input           13
```

记为 `x_cell in R^13`。

## 3.4 perturbation input：46 维

最终 M6.11/M6.21 没有直接拼 Morgan，也没有直接拼 ChemBERTa：

```text
x_drug = chemical ID one-hot in R^46
```

新化合物在该分支中确实是全零。新药的可迁移区别由独立 M9.6 读取 canonical SMILES 后
提供，而不是由 M6 假装提供。

## 3.5 legacy response input：59 维

M2 的独立 response 网络直接读取：

```text
strain 5 + chemical 46 + medium 2 + temperature 2 + time 4 = 59
```

记为 `x_response_legacy in R^59`。

## 3.6 observation input：155 维

```text
data_source one-hot  4
instrument one-hot   7
plate one-hot      144
----------------------
observation         155
```

记为 `x_obs in R^155`。Calibration 只能读取这 155 维，不能读取 strain、chemical、
medium、temperature 或 time。

---

## 4. M2.0：稳定 fallback 的完整结构

`M2.0` 是 learned-rank-64、MSE 版本；`M2.31` 网络相同，只把三项主损失换成 Huber。

```text
Background branch
x_cell[13]
   -> Linear(13,128)
   -> GELU
   -> Dropout(0.10)
   -> Linear(128,4422)
   -> B2[4422]

Response branch
x_response_legacy[59]
   -> Linear(59,128)
   -> GELU
   -> Dropout(0.10)
   -> Linear(128,64)
   -> r2_latent[64]
   -> matrix multiply W_response[64,4422]
   -> R2[4422]

Calibration branch
x_obs[155]
   -> Linear(155,16,bias=False)
   -> c2_latent[16]
   -> matrix multiply W_calibration[16,4422]
   -> C2[4422]

final = B2 + C2 + treatment_gate * R2
```

### 4.1 为什么 Response/Calibration 用低秩

直接从 128 hidden 输出 4,422 维需要大量独立参数，并且容易让每个蛋白只记自己的
噪声。低秩分解让所有蛋白共享 64 或 256 个响应方向：

```text
response[i,:] = latent[i,:] @ protein_basis
```

这不是固定 PCA。`protein_basis` 与网络一起端到端学习，所以能围绕任务 loss 选择方向。
固定 SVD/PCA 消融没有击败 learned basis。

### 4.2 M2 的不足

M2 的 Background 和 Response 参数完全独立。Response 虽然也收到 strain/medium/time
one-hot，但没有复用 Background 学到的 cell state。它等价于两个网络分别理解“同一个
细胞”，会浪费样本，也不符合“药物作用于当前细胞状态”的因果直觉。

M6 的核心改进就是让二者共享 cell encoder。

---

## 5. M6.11：共享 cell state 的 concat-256 主干

M6.11 的 full-refit 每个 seed 有约 1.83M 个可训练参数，response rank 为 256。

## 5.1 共享 cell encoder

```text
x_cell[13]
   -> Linear(13,128)
   -> GELU
   -> Dropout(0.10)
   -> h_cell[128]
```

这一个 `h_cell` 同时服务 Background 和 Response。

## 5.2 Background

```text
h_cell[128]
   -> Linear(128,4422)
   -> B6_standardized[4422]
```

## 5.3 Response

```text
h_cell[128] -----+
                 +-> concat[174]
x_drug[46] ------+
                        |
                        -> Linear(174,128)
                        -> GELU
                        -> Dropout(0.10)
                        -> Linear(128,256)
                        -> r6_latent[256]
                        -> @ W_response[256,4422]
                        -> R6_standardized[4422]
```

关键不是“rank 从 64 变 256”这么简单，而是 Response 直接读取 Background 同源的
`h_cell`。这使模型学习的是：

```text
response = f(current_cell_state, drug_identity)
```

而不是：

```text
response = f(separately_encoded metadata)
```

## 5.4 Calibration

M6.11 延续独立观测分支：

```text
x_obs[155]
   -> Linear(155,16,bias=False)
   -> c6_latent[16]
   -> @ W_calibration[16,4422]
   -> C6_standardized[4422]
```

它与 Background/Response 不共享输入，因为 plate/instrument 是测量偏差，不是生物状态。
“共享生物主干”不等于把所有 metadata 混成一个向量。

## 5.5 natural log2 输出

标准化空间：

```text
z_absolute = B6_std + C6_std + treatment_gate * R6_std
```

还原到 log2 空间：

```text
B6_plus_C6 = (B6_std + C6_std) * sigma + mu
R6         = R6_std * sigma
y_hat      = B6_plus_C6 + treatment_gate * R6
```

M12 融合发生在 natural log2 response 空间，而不是把两个不同标准化尺度的 latent 直接相加。

---

## 6. M6.21：FiLM-256 time 路由

M6.21 与 M6.11 使用相同 cell encoder、background decoder、rank-256 response decoder
和 rank-16 calibration。唯一核心差异是药物先调制 cell hidden：

```text
x_drug[46]
   -> Linear(46,256)
   -> split into scale[128], shift[128]

h_mod = h_cell * (1 + tanh(scale)) + shift

concat(h_mod[128], x_drug[46]) = 174
   -> Linear(174,128)
   -> GELU
   -> Dropout(0.10)
   -> Linear(128,256)
   -> @ W_response[256,4422]
```

FiLM 允许同一个细胞 hidden 在不同药物下做按维缩放和平移。它在 time/time-forward OOF
中比简单 concat 更强，因此只进入 M5.2 的 time 路由；它没有被强行推广到所有 OOD
场景。

---

## 7. M2/M6 的 loss：三种监督同时存在

M2.0、M6.11、M6.21 使用同一主损失权重：

```text
L_total = 0.25 * L_absolute
        + 1.00 * L_background
        + 1.00 * L_response
```

## 7.1 absolute loss

```text
L_absolute = sum(mask * (z_hat_absolute - z_target)^2) / sum(mask)
```

它保证最终完整蛋白向量在绝对尺度上正确。

## 7.2 background loss

只在 control/QC policy 允许的非 treatment 行上监督：

```text
L_background = sum(mask * background_selector
                   * (z_hat_background - z_target)^2)
               / sum(mask * background_selector)
```

它阻止 Background 依赖 Response 来解释 control。

## 7.3 response loss

只在存在 exact matched control 的 treatment 蛋白位置计算：

```text
delta_std = delta_log2 / sigma
L_response = sum(delta_mask * (R_std - delta_std)^2) / sum(delta_mask)
```

注意这里不减 `mu`，因为 delta 本身没有绝对丰度截距。

## 7.4 M2.31 的 Huber

M2.31 把三项 elementwise MSE 都替换为 `delta=1.0` 的 Huber：

```text
Huber(e) = 0.5*e^2                  if |e| <= 1
           |e| - 0.5               otherwise
```

Huber 对异常值更稳，FC 有小幅收益，但 high-effect PCC 往往下降，所以它只作为 M5.1
的小权重辅助，不单独替换 M2.0。

## 7.5 优化细节

- optimizer：AdamW；
- learning rate：`1e-3`；
- weight decay：`2e-4`；
- epochs：80；
- batch size：128；
- dropout：0.10；
- gradient norm clip：5.0；
- seeds：42、43、2026。

所有 scaler、词表、matched control、低秩统计和 support vocabulary 在 OOF 中只由
fold-train 拟合。

---

## 8. M9.6：从外部 RNA 扰动迁移到蛋白 response

M6 对未见化合物的 ID one-hot 为全零，无法区分两个都没见过的药。M9.6 不读取 GOAI
chemical ID，而是读取 canonical SMILES 的 Morgan 指纹，因此不同新药仍有不同输入。

M9.6 只预测 treatment-control response，不负责 absolute Background 和 Calibration。

## 8.1 化合物输入：Morgan-2048

```text
canonical SMILES
   -> RDKit Morgan fingerprint
   -> radius = 2
   -> 2048 binary dimensions
   -> x_morgan[2048]
```

映射链经过名称、CID、InChIKey、SMILES 和配方/盐型审计。推理时若 treatment chemical
没有结构，脚本直接报错，不允许静默给全零向量。

## 8.2 OP3 预训练 chemical encoder

外部 Open Problems 2023 RNA 扰动数据训练出 chemical encoder：

```text
x_morgan[2048]
   -> Linear(2048,256)
   -> LayerNorm(256)
   -> GELU
   -> Dropout(0.20)
   -> Linear(256,64)
   -> LayerNorm(64)
   -> GELU
   -> h_chem[64]
```

进入 M9.6 residual 训练后，该 encoder 参数冻结，且保持 `eval()`，因此预训练 encoder
中的 dropout 也冻结，不会在 residual 训练时制造随机漂移。

预训练时做了 GOAI parent structure 排除，并训练了 whole-drug shuffled 对照。真实 OP3
encoder 在相同 downstream 模型中稳定胜过 shuffled encoder，说明提升不只是额外参数。

## 8.3 context 输入：8 个字段

M9.6 context 使用以下 categorical fields：

```text
Strains
Medium
Temperature
pert_time
pert_time_unit
data_source
instrument
Yeast_cell_plate
```

每个字段预留 index 0 给 unknown。full-refit embedding 表形状为：

| 字段顺序 | embedding table | embedding width |
|---:|---:|---:|
| 0 | `6 x 4` | 4 |
| 1 | `3 x 4` | 4 |
| 2 | `3 x 4` | 4 |
| 3 | `7 x 6` | 6 |
| 4 | `2 x 3` | 3 |
| 5 | `5 x 4` | 4 |
| 6 | `8 x 6` | 6 |
| 7 | `145 x 16` | 16 |

拼接宽度：

```text
4 + 4 + 4 + 6 + 3 + 4 + 6 + 16 = 47
```

然后：

```text
context embeddings[47]
   -> Linear(47,64)
   -> LayerNorm(64)
   -> GELU
   -> h_context[64]
```

## 8.4 冻结 context-only base

M9.6 首先训练一个 chemical input 恒为 0 的 context-only base。虽然类中保留 chemical
encoder，但实际 forward 强制传入 zero fingerprint，因此它不能记药物身份。

```text
zero Morgan -> base chemical encoder -> q0[64]
context categorical -> context encoder -> h_context[64]
interaction = Linear(q0) * Linear(h_context) -> 64

concat(q0[64], h_context[64], interaction[64]) -> 192
   -> Linear(192,256)
   -> LayerNorm(256)
   -> GELU
   -> Dropout(0.15)
   -> Linear(256,256)
   -> GELU
   -> Dropout(0.15)
   -> Linear(256,4422)
   -> R9_base_standardized
```

这个 base 学公共 context response，不让化学分支一开始就破坏强公共模式。

## 8.5 zero-initialized chemical residual

在冻结的 context base 上增加：

```text
h_chem[64] ----------------------------+
                                        +-> concat[128]
h_context[64] -> Linear(64,64) -> Tanh -+
                     |                  |
                     +-> elementwise gate with h_chem

concat(h_chem, h_chem * tanh(Wg*h_context))[128]
   -> Linear(128,128)
   -> LayerNorm(128)
   -> GELU
   -> Dropout(0.10)
   -> Linear(128,4422), initialized to all zeros
   -> correction_standardized[4422]
```

最后一层全零初始化意味着训练开始时 M9.6 精确等于 context-only parent。模型只能在 loss
支持时逐步增加 drug-specific correction，而不是一开始随机破坏 base。

## 8.6 M9.6 loss

目标是标准化 matched-control delta。对每个有效蛋白：

```text
prediction_std = base_std + correction_std
weight          = delta_mask * (1 + 0.35 * I(abs(delta_log2) > 1))
```

数据损失：

```text
L_data = weighted SmoothL1(prediction_std, target_std, beta=0.5)
```

残差收缩：

```text
L_shrink = mean(delta_mask * correction_std^2)
L_M9     = L_data + 0.02 * L_shrink
```

训练完成后的推理仍做额外保守缩放：

```text
R9.6_log2 = R9_base_std * target_scale + target_mean_delta
            + 0.20 * correction_std * target_scale
```

也就是说 correction 在训练模型内学习，但最终只使用 20%，降低外部迁移信号过强的风险。

---

## 9. 三随机种子是怎样融合的

所有家族先在家族内部平均，再做跨家族运算，顺序不能交换成“每个 seed 各自路由后再随意
平均”。

```text
M2_MSE  = mean(M2.0-S42, M2.0-S43, M2.0-S2026)
M2_HUB  = mean(M2.31-S42, M2.31-S43, M2.31-S2026)
M6_CON  = mean(M6.11-S42, M6.11-S43, M6.11-S2026)
M6_FILM = mean(M6.21-S42, M6.21-S43, M6.21-S2026)
M9_OP3  = mean(M9.6-S42, M9.6-S43, M9.6-S2026)
```

M5.1 的 split-level Huber 权重：

| split_final | M2.0 weight | M2.31 weight |
|---|---:|---:|
| `test_chem_only` | 0.85 | 0.15 |
| `test_strain_only` | 1.00 | 0.00 |
| `test_both` | 1.00 | 0.00 |
| `test_time` | 0.70 | 0.30 |

M5.2 在 M5.1 上做两次替换：

```text
test_chem_only -> M6.11 three-seed average
test_time      -> M6.21 three-seed average
other rows     -> retain M5.1
```

M12.0 再按 canonical support router，把所有 R10 行替换为 M6/M9.6 融合。因此原先
`test_both` 中实际属于 R10 的 432 行也能得到新药模型，而不是被粗糙地留在 S3 fallback。

---

## 10. M12.0 的 R10 融合逐元素解释

先取 M6.11 三 seed 平均后的 natural log2 组件：

```text
B6_plus_C6[i,p]
R6[i,p]
```

再取 M9.6 三 seed 平均 response：

```text
R9[i,p]
```

基础 blend：

```text
blend[i,p] = -0.075 * R6[i,p] + 1.075 * R9[i,p]
```

权重大于 1 不是概率，而是 OOF 上选定的线性外推系数。它略微反向扣除 M6 response，
更强调 M9 对 FC 的预测。

随后高效应保护门：

```text
gate[i,p] = 1 if abs(R6[i,p]) >= 0.5 else 0
R12[i,p]  = blend[i,p]
             + 0.15 * gate[i,p] * (R6[i,p] - blend[i,p])
```

因此：

- `abs(R6)<0.5`：完全使用 blend；
- `abs(R6)>=0.5`：把 blend 向 M6 拉回 15%，保护 M6 更强的 high-effect PCC。

最后：

```text
y_hat[i,p] = B6_plus_C6[i,p] + R12[i,p]
```

R10 都是 treatment，所以这里 treatment gate 为 1。

---

## 11. 逐行 support router

## 11.1 为什么不能只看 split_final

最初的 `test_both` 是按早期 train/validation 角色定义的。最终模型用全部 8,958 条已发布
标签 refit 后，一部分原 validation 实体已经进入训练支持集。因此 `test_both` 里实际混有：

```text
432 rows: strain seen, chemical unseen  -> R10
272 rows: strain unseen, chemical seen  -> R01
425 rows: strain unseen, chemical unseen -> R00
```

若整块都按 `test_both` 路由，会把 704 行错误地当作双未知。

## 11.2 canonical support key

每个 raw entity 先经过 registry：

```text
raw name
  -> normalized name
  -> canonical/support ID
  -> seen in fit rows?
```

chemical 不能只看 `pert_id`，因为它只在 `data_source` 局部唯一。化合物 registry 保存
PubChem CID、InChIKey、SMILES、配方/parent 和证据级别。proxy 不会与其 proxy target
错误合并。

硬门定义：

```text
strain_seen   = strain support key in fit support
chemical_seen = chemical support key in fit support
is_treatment  = not Water/DMSO/QC
```

对应关系：

```text
strain_seen=0, chemical_seen=0 -> R00
strain_seen=1, chemical_seen=0 -> R10
strain_seen=0, chemical_seen=1 -> R01
strain_seen=1, chemical_seen=1 -> R11
non-treatment                  -> control
```

门控事实由 manifest 确定，不交给神经网络猜。

## 11.3 最终路由表

| 路由 | 行数 | 使用模型 | 原因 |
|---|---:|---|---|
| R10 | 2,072 | M12 R10 fusion | 新药有 Morgan/OP3 可迁移表示，M9 显著改善 FC |
| R01 | 1,594 | M5.2/M2 | 新菌株语义未稳定击败 fallback |
| R00 | 425 | M5.2/M2 | 双语义模型低于 zero/M2 对照 |
| R11 | 135 | M5.2/M6.21 time | 已见实体的 time 路线已有稳定 OOF 证据 |
| control | 228 | M5.2 background/control | 禁用 treatment response |

---

## 12. “通用模型 + 专家”做了没有，为什么最终看不到专家

做了。M7/M11 系列实现了：

```text
universal background + seen-strain background residual
universal response
  + seen-strain response residual
  + seen-chemical response residual
  + seen-pair/time response residual
```

专家使用 fold-fit support vocabulary 和硬门，未知实体 gate 必为 0。训练也实现了 universal
阶段、冻结 universal 训练专家阶段和小学习率联合阶段。

但“代码里实现”不等于“最终应该启用”。在与 M12.0 组合时：

```text
candidate = M12.0 + alpha * (expert - general)
```

| alpha | FC PCC | Context PCC | High PCC | High F1 |
|---:|---:|---:|---:|---:|
| 0.00 | **0.426342** | **0.060967** | **0.603184** | 0.233970 |
| 0.25 | 0.425307 | 0.060250 | 0.598980 | **0.234061** |
| 0.50 | 0.415593 | 0.056766 | 0.591271 | 0.230306 |
| 1.00 | 0.377658 | 0.046187 | 0.566662 | 0.210104 |

最优 FC 权重是 `alpha=0`。专家能改善较弱的 general parent，但与 M9-heavy response
叠加后互相冲突。因此最终 M12.0 没有隐藏启用专家，不把负结果包装成成功组件。

---

## 13. 菌株语义做了没有，为什么最终仍回退

第一版公开菌株表示使用 1,011 yeast genomes 的 SNP distance 派生 MDS/RBF/nearest
特征，并与 zero、shuffled 同协议比较。

M12.1 的 scaled SNP-MDS-4 在三个 seed 上都优于 shuffled，说明公开菌株距离确实包含
生物信号。但把它叠加到稳定 M2 fallback：

| semantic alpha | FC PCC | Drug residual PCC | High PCC |
|---:|---:|---:|---:|
| 0.00 | **0.280663** | 0.219185 | **0.614390** |
| 0.05 | 0.280592 | 0.219394 | 0.614338 |
| 0.10 | 0.280484 | 0.219573 | 0.614246 |
| 0.20 | 0.280162 | **0.219841** | 0.613935 |

它改善 drug residual，却略降主 FC 和 high-effect，最佳主指标权重仍是 0。双未知 M12.2
中，real-real FC `0.185253`，zero-zero 对照 `0.210603`，也没有晋级。

所以最终 R01/R00 回退不是忘了做菌株表示，而是做完严格负对照后主动拒绝。

---

## 14. 无泄漏验证协议

模型选择使用实体级 OOF，而不是随机拆行：

| 协议 | 验证含义 | 训练移除规则 |
|---|---|---|
| R10/S1 | 已见菌株、新化合物 | held-out chemical 的全部行移除 |
| R01/S2 | 新菌株、已见化合物 | held-out strain 的全部行移除 |
| R00/S3 | 双未知 | held-out strain 与 chemical 的相关行全部移除 |
| R11 | 两实体已见、pair 未见 | held-out pair block 移除 |
| RT | time/context 外推 | exact pair/context 的目标时间移除 |

每折重新拟合：

- one-hot vocabulary；
- target mean/scale；
- matched controls；
- PCA/SVD 或 learned decoder训练；
- target statistics/prototypes；
- entity support vocabulary；
- 专家 gate 和融合权重的内层选择。

M9 还包含 real vs whole-drug shuffled RNA encoder 对照，保证收益来自正确化学语义，而
不是更多参数或训练时间。

---

## 15. 指标如何计算和解读

## 15.1 FC PCC

在 exact matched-control 可用位置，把预测 absolute 转成预测 FC，再与真实 FC 做 Pearson：

```text
pred_delta = pred_absolute - matched_control_absolute
FC PCC     = pearson(pred_delta[mask], true_delta[mask])
```

它是当前主要选择指标。

## 15.2 Context residual PCC

先从 fold-train 构造公共 context response，再比较去除公共模式后的相关性。它更强调“这个
药相对同 context 平均到底有什么独特响应”。M12 FC 上升而 Context PCC 下降，说明它更强
地利用了可预测的公共响应，但 drug-specific residual 仍是短板。

## 15.3 High-effect PCC 与 F1

high-effect 定义使用真实 `abs(delta)>1` 的位置。PCC 衡量这些大效应的连续值相关；F1
同时要求预测超过阈值且方向一致，用来防止模型只把所有 FC 压向 0。

## 15.4 Absolute sample R2 median

对每个样本的 4,422 维绝对蛋白向量算 R2，再取中位数。它检查完整丰度是否合理，但不能
替代 FC 指标，因为绝对背景本身就很容易贡献很高 R2。

---

## 16. 最终主要结果

严格 S1/R10，三 seed：

| 模型 | FC PCC | Context PCC | High PCC | High F1 | Abs R2 |
|---|---:|---:|---:|---:|---:|
| M6 core | 0.371973 | 0.098530 | **0.635746** | 0.182687 | 0.979300 |
| M9 response replacement | 0.426065 | 0.064850 | 0.603782 | 0.233621 | **0.979471** |
| M11 blend | 0.426139 | 0.062244 | 0.600870 | 0.233938 | 0.979193 |
| **M12.0** | **0.426342** | 0.060967 | 0.603184 | **0.233970** | 0.979150 |

M11 相对 M6 的 FC 增益是大跃升来源：`+0.054166`，37-chemical bootstrap 95% CI
`[+0.027482,+0.075783]`。M12 在 M11 上只做小幅 high-effect 修正。

---

## 17. 从 checkpoint 到 prediction.csv 的精确执行顺序

`scripts/predict_m12.py` 执行：

```text
1. 读 metadata_test，验证字段、sample_ID 和 split_final。
2. 加载 3 个 M2.0 checkpoint，逐 seed 预测并平均。
3. 加载 3 个 M2.31 checkpoint，逐 seed 预测并平均。
4. 按 split 权重形成 M5.1。
5. 加载 3 个 M6.11，导出 B+C、R、final，验证组件重建，再平均。
6. 加载 3 个 M6.21，导出组件并平均。
7. test_chem_only 用 M6.11、test_time 用 M6.21，形成 M5.2。
8. 对 test treatment 名称查 chemical map；缺结构立即失败。
9. 生成 Morgan-2048。
10. 加载 3 个 M9.6 checkpoint，重建 context base 和 OP3 residual，再平均。
11. 读取 full-refit support manifest，逐行得到 R10/R01/R00/R11/control。
12. 验证官方 route counts。
13. 对全部蛋白计算 M12 R10 融合和 high-effect gate。
14. 只替换 R10 行，其他行保留 M5.2。
15. 写 prediction.csv、route_audit.csv、prediction_contract.json。
16. 重新读取 prediction.csv，验证 4,422 列顺序、sample ID 顺序和 finite。
```

该入口不读取缓存的 base prediction 或 M9 test response，因此权重和源码本身足够完成重放。

---

## 18. 权重与可复现合同

最终推理需要：

| 家族 | checkpoint 数 | 单 seed 作用 |
|---|---:|---|
| M2.0 | 3 | MSE fallback |
| M2.31 | 3 | Huber fallback auxiliary |
| M6.11 | 3 | shared-concat rank-256 |
| M6.21 | 3 | FiLM rank-256 time route |
| M9.6 | 3 | OP3 chemical residual response |
| OP3 encoder | 1 | 预训练来源/单独复核 artifact |

所有文件大小和 SHA256 位于 `weights/manifest.json`。执行：

```bash
python scripts/verify_release.py
```

公开仓库全链路回放与内部冻结结果比较：

```text
rows                         4,454
proteins                     4,422
values compared         19,695,588
max absolute difference      3.0e-6
mean absolute difference     2.98e-9
count(abs diff > 1e-5)       0
```

CSV SHA 不同来自浮点输出格式和微小舍入，不是模型预测差异。仓库不包含回放生成的
prediction 文件，只保存 `manifests/reproduction_receipt.json`。

---

## 19. 当前没有解决的问题

1. **R01 新菌株**：训练实体太少，公开 SNP-MDS 有信号但不足以稳定击败 M2 fallback。
2. **R00 双未知**：同时缺 chemical 和 strain support，两个语义 encoder 的交互仍不稳。
3. **drug-specific residual**：M12 的 Context residual PCC 低于 M6，说明 FC 提升更多来自
   公共可预测模式，而不是完全解决每个新药的独特反应。
4. **Calibration confounding**：plate 与 source/instrument/time 强相关。当前保留 calibration
   是因为去 plate 没有通过最终多指标护栏，但解释上仍需谨慎。
5. **4,422/5,243 输出合同**：本地训练支持 4,422 个蛋白；任何扩展必须以组委会权威
   sample-submission 为准，不能用均值或 0 随意补齐。
6. **外部数据路线**：M9.6 使用 OP3，必须在正式提交材料中披露来源、许可和开放知识属性。
7. **官方成绩缺失**：没有官方提交 ID，因此本文不能把 OOF proxy 写成官方 PSS。

---

## 20. 代码索引

| 内容 | 文件 |
|---|---|
| 最终推理 | `scripts/predict_m12.py` |
| 权重哈希验证 | `scripts/verify_release.py` |
| M2/M6 模型 | `src/goai_response/model.py` |
| M2/M6 特征 | `src/goai_response/features.py` |
| M2/M6 训练与 loss | `src/goai_response/train.py` |
| support router | `src/goai_response/routing.py` |
| registry/support | `src/goai_response/entities.py` |
| M9 chemical/context 模型 | `src/goai_rna_transfer/models.py` |
| M9 zero-init residual | `src/goai_rna_transfer/train_residual_gate.py` |
| M9 full refit | `src/goai_rna_transfer/refit_m9_response.py` |
| 模型编号 | `docs/method/model_registry.md` |
| 完整实验结果 | `docs/results/m12_execution_results.md` |
| 多模型对比 | `docs/results/model_comparison.md` |
| 外部数据与许可 | `EXTERNAL_RESOURCES.md` |

---

## 21. 最后再明确一次最终模型边界

进入最终 `GOAI-M12.0` 的是：

```text
M2.0 + M2.31 + M6.11 + M6.21 + M9.6 + deterministic support router
```

没有进入最终 M12.0 的是：

```text
strain SNP semantics       (real signal, best final alpha = 0)
seen-entity expert overlay (implemented, best final alpha = 0)
dual chemical/strain M12.2 (below zero/M2 controls)
ChemBERTa direct concat    (below M2)
fixed SVD decoder          (below learned decoder)
PPI graph variants         (real graph did not beat controls)
```

这一区分保证文档描述的是实际提交路径，而不是把所有做过的实验都堆进一张看起来复杂的
架构图。
