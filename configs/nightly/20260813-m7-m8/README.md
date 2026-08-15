# M7/M8 两 GPU 实验编排

这里的所有指标都只能称为 **本地严格 OOF**，不是 GOAI 官方 PSS 或排行榜成绩。所有任务使用现有的断点续跑机制：重新执行同一命令和同一 `run-root` 会跳过已有 `oof_summary.csv` 的任务，并对未完成任务追加 `--resume`。

当前矩阵的工作量口径以一次 fold 训练为一次 `fit`：

- quick screen 共 32 个任务：22 个五场景主候选各 14 fits，加 10 个单场景 scale 候选各 2 fits，共 **328 fits**；
- research prior/pair 共 9 个任务：4 个五场景 prior 对照各 14 fits，加 5 个
  `R11+RT` pair-scale 对照各 6 fits，共 **86 fits**；
- Calibration audit 共 7 个任务：每个任务为 `plate=2` 加五场景 `14`，共 **16 fits/任务、112 fits**；
- confirmation 的五场景在 4-fold 下为 **44 个 outer fits/seed**；固定三个 seeds 后是
  **132 个 outer fits/候选**，另加每个适用 outer fold 内至少 2-fold 的 scale=1 inner
  OOF。只有 discovery 提名且全部门禁通过的候选才应物化运行。

这里不是简单的“场景数 × folds”：R00/R11 使用 strain×chemical 二维交叉块，R10/R01 已去除重复训练，只分别按 chemical/strain 分区。

身份语义有两条机器可审计路线：quick M8 明确使用
`semantic_identity_policy: research_allow_candidate`，仅可标为研究性候选身份实验；formal
confirmation 默认 `verified_only`，候选实体的语义向量严格置零但仍保留 candidate 身份 flag
和独立的 seen/support gate。Morgan real 必须与 `MORGAN-SHUFFLED` 同架构负对照成对比较，
菌株 real/shuffled 也必须通过同一个来源、证据与哈希 manifest。

## 1. 快速筛选与专家 scale 提名搜索

`quick_screen.yaml` 已显式标记 `promotion_eligible: false` 与
`schedule_confounded: true`：各候选总 epoch 相同，但 universal/expert 更新次数不同，
只能找方向，不能用它晋级。实体专家归因必须通过 2.1 的公平消融。

```bash
PY=/dev/shm/chenyuming-discobax/envs/genedisco-repro/bin/python
$PY scripts/nightly/run_matrix.py \
  --matrix configs/nightly/20260813-m7-m8/quick_screen.yaml \
  --run-root runs/nightly/20260813-m7-m8-screen \
  --python "$PY" --gpus 0,1
```

只跑指定任务时使用逗号分隔的实验 ID：

```bash
$PY scripts/nightly/run_matrix.py \
  --matrix configs/nightly/20260813-m7-m8/quick_screen.yaml \
  --run-root runs/nightly/20260813-m7-m8-screen \
  --python "$PY" --gpus 0,1 \
  --include SCR-M7.0-GENERAL,SCR-M7.3-ENTITIES
```

## 2. 汇总四种 regime、paired delta 和非绑定 scale 提名

```bash
$PY scripts/nightly/summarize_m7_m8.py \
  --run-root runs/nightly/20260813-m7-m8-screen \
  --control-id SCR-M7.0-GENERAL
```

关键产物在 `consumer/m7_m8/`：

- `regime_summary.csv`：R00/R10/R01/R11/RT 分状态结果；
- `four_regime_macro.csv`：四种实体状态总体；
- `paired_fold_deltas.csv` 与 `paired_delta_summary.csv`：同 fold 配对差值；
- `expert_scale_candidates.csv`：`0/0.25/0.5/0.75/1` 完整候选；
- `expert_scale_selection.yaml`：只有 strain、chemical、pair 三条网格都完整时状态才会是
  `selected`。pair 预先锁定以 `R11+RT` 的 FC PCC macro 选择，并要求两个 regime 的
  context/drug residual 均不退、high-effect PCC/F1 均满足护栏，不能事后挑单一场景。
  该文件只负责决定是否值得进入昂贵确认；其中数值不会注入 formal config。

## 2.1 严格公平的实体专家消融

原 12-epoch quick screen 的 M7.0 使用 `universal 9 + joint 3`，而 M7.3 使用
`universal 5 + strain 2 + chemical 2 + joint 3`。它只能作为早期诊断，不能把
R00 差值归因于专家。公平消融使用独立矩阵，不改写该历史契约：

```bash
$PY scripts/nightly/run_matrix.py \
  --matrix configs/nightly/20260813-m7-m8/fair_expert_ablation.yaml \
  --run-root runs/nightly/20260813-m7-m8-fair-experts \
  --python "$PY" --gpus 0,1
```

七个任务都先以同一 seed、同一 fold、同一个不含专家参数的 M7.0 构造训练
universal 9 epochs。之后才扩展零初始化专家，并逐 tensor 复制所有共同状态；复制前后
SHA256 必须相同，残差-only 阶段共同状态也必须保持不变。每个 fold 的
`folds/*/completed.json` 保存 `training_receipt`：

- `universal_state_sha256`：通用预训练父状态收据；
- `copied_universal_state_sha256`：复制到专家模型后的共同状态收据；
- `post_frozen_expert_universal_state_sha256`：专家冻结训练后的共同状态收据；
- `final_universal_state_sha256`：可选 joint 后的最终共同状态收据。

任务覆盖 M7.0 receipt control，以及 strain-only、chemical-only、both-expert 各自的
frozen/joint 对照。R10 优先判断 strain-only，R01 优先判断 chemical-only，避免 combined
模型掩盖单类专家。两折五协议每任务 14 fits，总计 98 fits；这是
`LOCAL_STRICT_OOF_NOT_OFFICIAL`，不会自动晋级模型。

分数汇总前先强制核验七项 × 14 folds 的 source/parent receipts：

```bash
$PY scripts/nightly/audit_fair_expert_receipts.py \
  --run-root runs/nightly/20260813-m7-m8-fair-experts
```

任何 source fingerprint、fold key、parent/copy/frozen hash 不一致都会返回失败；明细写到
`consumer/fair_expert_receipts/receipt_audit.csv`，同时记录每个任务的实际 fit 数、
run-contract 文件哈希与 parent universal state 哈希。

## 3. 生成并运行确认矩阵

确认矩阵不会把 discovery 全局 scale 当成正式超参数。必须先用上一步的完整 OOF
收据确认候选方向，再生成启用 fold-local nested selection 的矩阵：

```bash
$PY scripts/nightly/build_m7_m8_confirm.py \
  --template configs/nightly/20260813-m7-m8/confirm_candidates.yaml \
  --selection runs/nightly/20260813-m7-m8-screen/consumer/m7_m8/expert_scale_selection.yaml \
  --identity-audit runs/audits/entity_registry_audit.json \
  --calibration-audit runs/nightly/20260813-m7-m8-calibration/consumer/calibration_audit_receipt.json \
  --fair-expert-audit runs/nightly/20260813-m7-m8-fair-experts/consumer/fair_expert_receipts/receipt_audit.json \
  --output runs/nightly/20260813-m7-m8-confirm/confirm_matrix.yaml
```

strain、chemical、pair 三条 discovery 提名网格，实体身份审计、Calibration 完整审计和公平专家
parent receipt 任一缺失/失败时，程序都会拒绝生成可运行矩阵；四份证据路径与 SHA256
嵌入物化矩阵且明确标记 `binding_to_formal_predictions: false`。身份审计还锁定两张
registry 的文件 SHA256；审计后再改 registry 会直接
拒绝物化。网格未跑完时写出 `.scale_candidates.csv`，不会伪造选择。
完成后可用 `--include` 只确认筛选晋级者。正式候选共享同一个 80-epoch universal
父训练：frozen 变体之后只训残差专家，`-JOINT` 变体再追加 16 epochs 小学习率联合微调；
两种路径都由 materializer 显式生成，不能把 quick screen 的不公平 schedule 带入确认。
frozen 的 M7.1/M7.2/M7.3 对比 U80 M7.0，joint 版本对比无专家的 U96 M7.0 控制；
M7.4 frozen/joint 分别对比相同训练变体的 M7.3。这样 joint 的提升不能由“主干多更新
16 epochs”单独解释。物化器会验证所有 primary/negative control 引用存在且 universal
update budget 完全一致。
每个变体固定运行 4-fold 五场景、seeds `42/52/62`，即 132 个 outer fits/候选变体；
适用的 outer fold 还会只在其 train rows 内建立至少 2-fold inner OOF。inner 模型只训练
一次 canonical scale=1，然后从 `B_U/B_s/C_obs/R_U/R_s/R_c/R_sc` 组件后处理枚举网格，
不会为 5/25/125 个组合重复训练：

当前正式 `verified_only` 身份门禁下，训练角色中可接纳的 verified 化学实体和菌株实体
都为 0；因此所有 M8 语义候选都会进入 `blocked_confirm_candidates`，不会生成 GPU 任务。
它们仍可在明确 `promotion_eligible:false` 的 research/quick 路线研究，只有一手证据把训练
实体升级为 verified 后，semantic coverage preflight 才会放行正式 M8 确认。
实际每个 fold 训练前仍会再次检查非零 admitted coverage，并把计数写入
`training_receipt.semantic_training_coverage`；该检查不能靠模板物化结果绕过。

```bash
$PY scripts/nightly/run_matrix.py \
  --matrix runs/nightly/20260813-m7-m8-confirm/confirm_matrix.yaml \
  --run-root runs/nightly/20260813-m7-m8-confirm \
  --python "$PY" --gpus 0,1 \
  --include CONF-M7.3-ENTITIES,CONF-M8.2-DUAL
```

确认完成后使用严格晋级消费者。quick/fair/scale 只能提名 confirmation，不能写
`promoted=true`：

```bash
$PY scripts/nightly/promotion_gate.py \
  --run-root runs/nightly/20260813-m7-m8-confirm \
  --candidate CONF-M8.0-MORGAN \
  --output-dir runs/nightly/20260813-m7-m8-confirm/consumer/promotion/CONF-M8.0-MORGAN
```

`promotion_regimes`、`primary_control` 和 `required_negative_controls` 均从物化确认矩阵读取。
M7 frozen 以相同 U80 预算的 M7 control 为主对照；M7.1/M7.2/M7.3 joint 以纯 U96
M7.0 为同 universal-update 对照，M7.4 joint 以 M7.3 joint 为对照。M8.0/M8.1/M8.2
以 M7.3 零语义同构模型为
主对照，M8.3 pair 以 M7.4 为主对照，并额外胜过相应 shuffled 语义对照。JOINT 候选只能
对比 JOINT 或明确的 pure-universal same-update 控制，且 universal-update epoch 数必须相同。

晋级收据要求：相关 regime FC PCC 至少 +0.01；R10 context residual、R01 drug residual、
R11/RT 两类 residual 同升；high-effect PCC/F1 总体及每个 seed 降幅都不超过 0.005；三个
模型 seed 方向一致；held-out chemical/strain/pair/time-group 聚类 bootstrap 的 95% CI 下界
大于 0。R00 本身没有可识别 residual，因此不能单独晋级。确认必须完整覆盖 4 folds、
fold seed 42、模型 seeds `42/52/62`，每个 fold 的 warm-start/common-state 收据完整。

M8 还会重算 fold-fit semantic coverage：配置了的每个语义轴必须有至少两个非零、身份门禁
准入实体，并在每个相关 OOD fold 的训练侧及合并验证侧都真正出现。`verified_only` 导致
admitted=0 时，即便指标偶然上涨也会以“零语义伪 M8”拒绝晋级。产物
`promotion_receipt.json`、`promotion_gate_checks.csv`、`heldout_entity_metric_units.csv` 全部
标记 `LOCAL_STRICT_OOF_NOT_OFFICIAL`，不是官方 PSS。

## 4. Calibration 排雷

```bash
$PY scripts/nightly/run_matrix.py \
  --matrix configs/nightly/20260813-m7-m8/calibration_audit.yaml \
  --run-root runs/nightly/20260813-m7-m8-calibration \
  --python "$PY" --gpus 0,1

$PY scripts/nightly/summarize_m7_m8.py \
  --run-root runs/nightly/20260813-m7-m8-calibration \
  --control-id CAL-M7.0-R16-BASE

$PY scripts/nightly/audit_calibration_results.py \
  --run-root runs/nightly/20260813-m7-m8-calibration
```

`audit_calibration_results.py` 只有在 7 个锁定任务各完成 16 fits，且明确覆盖
leave-one-plate-out、rank 4/8/16、plate shuffle、no-plate、dropout 0/0.3/0.5、
observation-metadata-only 时才写 `status=approved` 的 `calibration_audit_receipt.json`。

运行时可在 `run-root` 放一个名为 `STOP` 的文件，让编排器完成当前 GPU 上的任务后停止继续派发；删除它并重跑相同命令即可续跑。

## 5. 一条命令按证据门禁跑完整晚

推荐入口会严格按下面顺序执行，并把所有生产、汇总和审计产物放在同一个大盘父目录：

```text
fair expert ablation
→ receipt audit（失败立即停止）
→ fair summary
→ quick screen
→ research prior/pair screen
→ quick+research 三轴 scale summary
→ research 专项 summary
→ calibration audit
→ calibration summary
→ deterministic calibration receipt（失败立即停止）
→ strict identity receipt（M7 可继续；M8 ready/blocked 分流）
→ discovery-only M7 confirmation nomination + pre-evidence materialization
→ selected M7 candidates + exact controls 4-fold/3-seed confirmation
→ held-out-entity promotion receipts（promoted 或 blocked 均为有效结论）
```

```bash
PY=/dev/shm/chenyuming-discobax/envs/genedisco-repro/bin/python
$PY scripts/nightly/run_m7_m8_overnight.py \
  --base-root runs/nightly/20260813-m7-m8-overnight \
  --python "$PY" \
  --gpus 0,1 \
  --min-free-gb 25 \
  --min-gpu-free-mb 30000 \
  --max-gpu-utilization 20 \
  --gpu-wait-poll-seconds 30 \
  --gpu-wait-timeout-seconds 0
```

不写 `--base-root` 时默认也是上面的项目内 ignored `runs/` 路径。producer root
分别是 `fair-experts/`、`quick-screen/`、`research-prior-pair/`、`calibration-audit/`
和按证据动态物化的 `formal-m7-confirm/`；
因此不会把大产物写入项目盘。scale 汇总同时读取 quick 与 research root，保证
strain/chemical/pair 三轴收据完整；research root 另产一份 prior/pair 专项报告。

GPU 门禁使用 `nvidia-smi` 按 `--gpus` 的**明确物理索引**查询。只有每一张目标卡同时满足
空闲显存不少于 `--min-gpu-free-mb`、利用率不高于 `--max-gpu-utilization`，才会启动任务；
首次可执行阶段前会检查一次，之后每个 matrix 阶段开始前都会重新检查。等待期间不创建 CUDA
进程，当前显存、利用率、阈值和更新时间会持续写入
`<base-root>/overnight_status.json`，其 `state` 为 `waiting_for_gpu`。

默认 `--min-gpu-free-mb 0 --max-gpu-utilization 100` 表示关闭等待门禁。
`--gpu-wait-timeout-seconds 0` 表示无限等待；设置正数后，超时会以可续跑状态
`gpu_wait_timeout` 返回，不会启动当前阶段。正式长跑建议使用上例的 `30000 MB` 门槛。

入口复用 `run_matrix.py` 的 fold 级断点续跑、磁盘余量保护和 `STOP`。例如需要停止
quick screen 时，在 `<base-root>/quick-screen/STOP` 建空文件；当前 GPU 任务结束后不再派发。
删除 `STOP` 后原命令重跑即可续跑。消费者只有在矩阵的每个 `experiment × seed` 都存在
`oof_summary.csv` 后才会启动，所以 STOP 或磁盘保护不会把局部结果误当作完整结果。

整个 overnight 编排也支持 `<base-root>/STOP`：如果启动前或 GPU 等待时存在该文件，入口会
写入 `state=stopped`、`resume_available=true` 并直接返回，既不查询/占用 GPU，也不启动任何
新阶段。删除该文件后用原命令即可恢复。

总状态写在 `<base-root>/overnight_status.json`，逐阶段记录参数数组、开始/结束时间、
return code、验证结论和每次尝试；文本输出在 `<base-root>/orchestrator_logs/`。重复执行时，
已完成且收据仍有效的阶段会安全跳过，失败或收据缺失的阶段会从原 run root 继续。

确认准备阶段会生成 `preconfirmation_evidence.json`、
`confirmation_selection_receipt.json` 和 `m8_blocked_receipt.json`。fair/quick/research
只能提名 M7 正式确认，不能写晋级；M8 在当前 verified semantic coverage 为 0 时不会进入
`confirm_matrix.yaml`，也不会启动 GPU。pair 网格没有公平 joint-finetune 消融，因此自动波次
只提名 M7.4 frozen，不会擅自运行 M7.4-JOINT。确认训练完成后，
`promotion_batch_receipt.json` 汇总每个候选的严格晋级/阻断收据；统计不达标的 blocked 是
正常完成，哈希、fold、seed、训练收据不完整仍会使流程失败。所有结果继续标记为
`LOCAL_STRICT_OOF_NOT_OFFICIAL`。

scale 接口保留 fold-local inner-scale receipt 字段。全局 discovery scale 只用于提名；正式
无偏确认应由每个 outer fold 自己的 train-only inner selection 物化，不能把全局 scale 当成
官方或无选择偏差的确认结果。

每个 outer fold 独立选择：R10 枚举 strain，R01 枚举 chemical，R11 枚举
strain×chemical，RT 枚举 strain×chemical×pair；R00 没有可识别的专家 gate，不调
scale。目标是 inner-OOF FC PCC；相应 context/drug residual 相对全零专家不得下降，
high-effect PCC/F1 均不得下降超过 0.005，FC 完全相同时选择较小 scale。每折保存 inner
assignments、所有候选指标、fit support/source hashes 和选中权重；断点续跑会重新核验整条
哈希链。任何全局 scale、outer validation label 或被篡改收据都会使 formal
confirmation/promotion 失败。
