# GOAI M12 执行结果

日期：2026-08-15
状态：M12.0 完成并冻结；M12.1、expert overlay、M12.2 全部完成
评分口径：本地严格 OOF proxy，不是官方 PSS

## 1. 当前提交候选

当前最优完整候选为 `GOAI-M12.0`，父模型为 `GOAI-M11.0`。它只替换测试集中
R10（菌株已见、化合物未知）的 2,072 个 treatment 行，其余 R01、R00、R11 和
control 行逐值保留 `GOAI-M5.2`。

```text
blend = (1 - 1.075) * R6 + 1.075 * R9.6
gate  = I(abs(R6) >= 0.5)
R12   = blend + 0.15 * gate * (R6 - blend)
y_hat = B6 + C6 + R12
```

严格 S1/R10、三种子 `42/43/2026`：

| 模型 | FC PCC | Context PCC | High PCC | High F1 | Abs R2 |
|---|---:|---:|---:|---:|---:|
| M11.0 | 0.426139 | 0.062244 | 0.600870 | 0.233938 | 0.979193 |
| **M12.0** | **0.426342** | 0.060967 | **0.603184** | **0.233970** | 0.979150 |

M12.0 相对 M11.0 的 chemical-cluster paired bootstrap：

- FC `+0.000203`，95% CI `[+0.000059,+0.000355]`。
- High PCC `+0.002314`，95% CI `[+0.001804,+0.002790]`。
- High F1 `+0.000031`，CI 跨零。
- Context `-0.001277`，95% CI 全为负。

最终 prediction 为 `4,454 x 4,422` absolute log2，全部 finite，SHA256：
`4179afee866920ef7df6da99025c17e26c3b901647591691562884af7e8159ab`。

## 2. M12.1 新菌株语义

完成 R01 seed 42 的 RBF、nearest、scaled SNP-MDS 4/8/16 维筛选，并对唯一保留的
scaled-4 补齐 seeds `43/2026`。所有 strain scaler/transform 均只在 fold train 拟合；
real、shuffled 和 zero 使用相同 assignments。

scaled-4 裸模型：

| seed | Real FC | Shuffled FC | Real drug residual | Shuffled drug residual |
|---:|---:|---:|---:|---:|
| 42 | 0.272335 | 0.264101 | 0.216582 | 0.208510 |
| 43 | 0.272279 | 0.264199 | 0.217730 | 0.212002 |
| 2026 | 0.265931 | 0.262030 | 0.212677 | 0.210886 |

real 在三个 seed 均高于 shuffled，说明 SNP-MDS 含真实信号；但把它作为强收缩残差叠加
到三种子 zero-semantic M2 后，FC 最优为 `alpha=0`：

real 相对 shuffled 的四 fold paired bootstrap：FC `+0.006699`，95% CI
`[+0.000052,+0.017036]`；drug residual `+0.005179`；high PCC `+0.017463`；
high F1 `+0.004989`，四项 CI 下界均为正。

| alpha | FC PCC | Drug residual PCC | High PCC | High F1 |
|---:|---:|---:|---:|---:|
| 0 | **0.280663** | 0.219185 | **0.614390** | **0.157890** |
| 0.05 | 0.280592 | 0.219394 | 0.614338 | 0.157855 |
| 0.10 | 0.280484 | 0.219573 | 0.614246 | 0.157778 |
| 0.20 | 0.280162 | **0.219841** | 0.613935 | 0.157632 |

结论：真实菌株语义改善 drug residual，但损害 FC 与 high-effect，未通过提交门禁；
`M12.1` 不做全量 refit、不进入 router。需要改变语义与输出的交互方式，而不是继续放大 alpha。

## 3. 已见菌株专家叠加

先补齐 canonical R10 general/expert 的 seeds `43/2026`。三种子 standalone expert 相对
general 的 FC 约提高 `+0.029~+0.030`，证明专家分支确实学到已见菌株残差。

随后发现历史 M12 S1 与 canonical R10 的 fold identity 不同。没有放宽合同，而是新增
raw-normalized legacy-S1 配置；audit assignment SHA256 与 M12 source 均为
`7f8c891ea0c64d4724ec168fdd2e6b28d1f86e29f5c5535a5583945c13c595cd`，再重训
general/expert 三种子并评估：

```text
candidate = M12.0 + alpha * (expert - general)
```

| alpha | FC PCC | Context PCC | High PCC | High F1 | Abs R2 |
|---:|---:|---:|---:|---:|---:|
| 0 | **0.426342** | **0.060967** | **0.603184** | 0.233970 | **0.979150** |
| 0.25 | 0.425307 | 0.060250 | 0.598980 | **0.234061** | 0.979140 |
| 0.50 | 0.415593 | 0.056766 | 0.591271 | 0.230306 | 0.978190 |
| 0.75 | 0.398898 | 0.051676 | 0.580352 | 0.222493 | 0.976512 |
| 1.00 | 0.377658 | 0.046187 | 0.566662 | 0.210104 | 0.973802 |

结论：专家能改善弱 general parent，但其残差与 M9-heavy M12 response 冲突；最佳仍为
`alpha=0`。专家已真实实现并验证，但不进入当前提交候选。

## 4. M12.2 双未知联合语义

状态：完成，未晋级。

固定 R00 16-fold、seed 42、shared-concat rank 256，比较：

- `RR`：real chemical + real strain。
- `SR`：shuffled chemical + real strain。
- `RS`：real chemical + shuffled strain。
- `SS`：double shuffled。
- `ZZ`：chemical/strain semantics 全零，同 fold、同网络补充对照。

四格单模型结果：

| 输入 | FC PCC | High PCC | High F1 | Abs R2 |
|---|---:|---:|---:|---:|
| RR | 0.185253 | 0.418647 | 0.104083 | 0.947116 |
| SR | 0.177628 | 0.416289 | 0.095634 | 0.943641 |
| RS | **0.188209** | **0.425575** | **0.105192** | **0.947704** |
| SS | 0.179000 | 0.419146 | 0.095991 | 0.943925 |
| ZZ | **0.210603** | **0.480765** | **0.114181** | **0.951831** |

RR 与控制的混合结果：

| 对照 | 最佳 RR 权重 | 最佳 FC | delta vs 对照 | FC 95% CI |
|---|---:|---:|---:|---|
| SR（隔离 chemical） | 0.50 | 0.191307 | +0.013679 | [+0.005792,+0.021715] |
| RS（隔离 strain） | 0.25 | 0.189288 | +0.001079 | [-0.000185,+0.002333] |
| SS（双打乱） | 0.50 | **0.192641** | +0.013641 | [+0.005083,+0.022374] |

chemical 语义贡献明确；strain 语义在双未知组合里没有稳定增益。当前最佳 0.192641
仍低于冻结 M2/S3 约 0.216675，也低于同 fold ZZ 0.210603。

RR 相对 ZZ 的结果随 alpha 单调恶化：alpha 0.25 时 FC `0.209990`，alpha 0.5 时
`0.204541`，纯 RR `0.185253`。纯 RR 相对 ZZ 的 FC delta 为 `-0.025350`，95% CI
`[-0.034241,-0.017386]`；high PCC delta 为 `-0.062118`。因此 M12.2 明确拒绝，
R00 继续使用冻结 M5.2/M2 路径。

## 5. 代码与验证

- 新增 `scaled/rbf/nearest` strain transform，missing/proxy 标准化后严格清零。
- 新增 test consumer 的 semantic residual route：`current + alpha*(semantic-current)`。
- 新增 `scripts/evaluate_oof_residual_overlay.py`，强制 base/general/expert 的
  sample、protein 和 fold hash 一致。
- 源码外部更新后最终全量测试：`go-ai 204 passed`，`goai-rna-transfer 14 passed`。
- residual overlay 针对性测试：`4 passed`。

## 6. 官方提交阻塞

当前没有官方 portal/API/auth/scorer，且权威输出合同尚未解决 `4,422 vs 5,243` 蛋白列差异。
M12.0 使用 OP3 RNA 外部预训练，正式提交前还必须确认外部数据规则和许可。因此本文所有
分数均只能称本地严格 OOF proxy。
