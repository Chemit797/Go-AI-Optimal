# 外部资源、预训练与许可披露

本文件区分“当前 M12.0 实际使用”“仅用于研究消融”“仅用于文档生成”三类资源。
仓库不包含任何外部训练原始矩阵；为保证实体身份审计可追溯，仓库保留少量 PubChem、
ChEBI、ChEMBL 公共 API 证据快照。为保证最终推理可复现，仓库还包含项目训练产生的
M2/M6/M9 checkpoint 和一份 OP3 chemical encoder，完整哈希见
`weights/manifest.json`。

## 1. 当前 M12.0 实际使用

| 资源 | 用途 | 来源与版本 | 许可/合规状态 | 是否打包 |
|---|---|---|---|---|
| GOAI 官方虚拟酵母数据 | M6/M9.6 下游训练、OOF 与测试条件 | 组委会发布的 train_val/test 文件，输入哈希记录在运行 manifest | 仅按赛事授权使用；不得随附件再分发 | 否 |
| Open Problems - Single-Cell Perturbations 2023 pseudobulk differential expression | 训练 M9.6 的 RNA perturbation chemical encoder；使用严格 compound 排除和 shuffled 负对照 | [Open Problems schema](https://github.com/openproblems-bio/task_perturbation_prediction)；数据文件 `2023-09-12_de_by_cell_type_train.h5ad` | 数据标注为 CC BY 4.0；需保留署名。赛事是否允许外部数据仍应以 GOAI 最新规则为准 | 原始数据否；训练后的 encoder 是 |
| RDKit Morgan fingerprint | 将 canonical SMILES 转为 2,048 位分子指纹 | RDKit `2024.03.5` | BSD 3-Clause | 仅源码依赖，不含二进制包 |
| PubChem PUG REST | 化合物名称、CID、InChIKey、isomeric SMILES 的身份核对 | [PubChem](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)；映射记录含 CID 和来源 | NCBI/PubChem 公共数据使用条款；需引用来源 | 不含下载缓存 |

OP3 encoder 在本地记录中的 SHA256 为
`c8d9091bbdd6f5d4eeae85106a9db4b773425da0544a9d73fae05ce9dbb7c996`，仓库路径为
`weights/pretrained/op3_rna_chemical_encoder.pt`。源码同时提供从公开 OP3 数据重新训练
encoder 的入口。

## 2. 仅用于研究消融，未进入 M12.0

| 资源 | 实验用途 | 最终决策 | 许可/来源 |
|---|---|---|---|
| L1000FWD signatures | RNA-to-protein transfer 的 L1000 real/shuffled 对照 | 未进入 M12.0 | 提供方说明学术/非营利用途免费；奖金赛事或商业用途需额外核实：[下载页](https://maayanlab.cloud/l1000fwd/download_page) |
| Peter et al. 2018 1,011 yeast genomes | 菌株身份核验、SNP-distance MDS | M12.1 的真实语义优于 shuffled，但叠加主模型最佳权重为 0，未进入 M12.0 | Nature 论文补充数据及 1,011 genomes 公开资源；引用原论文与资源条款 |
| STRING v12 physical network | PPI real-vs-rewired 图消融 | 真实图未击败随机重连，未进入 M12.0 | [STRING](https://string-db.org/) 许可与引用要求 |
| ChemBERTa | frozen molecular embedding 消融 | 直接拼接未晋级 | Hugging Face 模型卡对应许可；本包不含权重 |

## 3. 方法参考，不含复制代码

- Kaggle / NeurIPS 2023 Open Problems - Single-Cell Perturbations：借鉴 chemical-held-out
  验证、response statistics、低秩输出和药物/细胞交互的公开经验。GOAI 的 test-only
  chemical 与原比赛“药物在其他细胞已见”不同，因此没有照搬 target statistics。
- LiSH MoA：借鉴 drug-aware validation、输出相关性和异构模型融合。
- OpenVaccine：借鉴 mask-aware loss 与可靠位置加权。
- CHAMPS scalar coupling：借鉴 molecule-group validation 与结构语义负对照。

提交源码未复制上述竞赛方案代码；所有模型实现均为本项目重新编写。

## 4. Python 依赖

记录训练环境：Python `3.8.20`、NumPy `1.24.4`、Pandas `2.0.3`、PyTorch
`2.4.1+cu124`、RDKit `2024.03.5`、scikit-learn `1.3.2`、SciPy `1.10.1`、
PyYAML `6.0.3`、h5py `3.11.0`。完整版本见根目录 `requirements.txt` 和
`environment.lock.txt`。各依赖遵循其上游开源许可证。

## 5. 文档生成资源

PDF 使用 Noto Sans CJK SC 字体生成，字体项目采用 SIL Open Font License 1.1。
字体仅嵌入 PDF，不作为单独文件分发。文档生成工具为 `python-docx 1.1.2` 和
`reportlab 4.3.1`，不属于模型运行依赖。

## 6. 明确未使用

- 没有商业 API、闭源大模型或付费推理服务参与 M12.0 训练与预测。
- 没有使用测试真值、私有湿实验结果或人工标注测试响应。
- test metadata 只用于产生预测顺序与逐行 support 路由，不参与 OOF 选参。
