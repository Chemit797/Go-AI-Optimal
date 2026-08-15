# GOAI 初赛提交附件 / Preliminary Submission Attachment

本目录保存按算法赛页面要求组织的单 ZIP 初赛附件：

```text
AI4R_ALG_AIVC_队伍名待替换.zip
```

ZIP 内含初赛方案 PDF/DOCX、源码、环境说明、主要结果表、关键结果图、外部资源披露和
包级 manifest。它不含 GOAI 官方数据、模型权重、OOF/test 预测、`prediction.csv`、
凭据或运行缓存。完整模型权重已单独保存在本仓库根目录的 `weights/` 中。

校验信息：

```text
size:   769258 bytes
sha256: 92342e5a3f0e16a6909fb791fd600844c2ef44891141b466885dff36adc93e70
```

## 上传前必须完成

当前版本还不是可直接上传的最终命名版本。必须先把真实队伍名补入：

1. ZIP 内 `README.md`；
2. `初赛方案说明.docx` 的队伍信息；
3. 重新导出的 `初赛方案说明.pdf`；
4. 顶层目录名；
5. ZIP 文件名，最终格式为 `AI4R_ALG_AIVC_实际队伍名.zip`。

页面每阶段最多提交 3 次，并以截止前最后一次成功提交为准。改名和补信息后应重新计算
SHA256，并再次执行 ZIP 完整性与敏感文件检查。

---

This directory contains the single-ZIP preliminary algorithm-track attachment.
It includes the proposal in PDF and DOCX formats, source code, result tables,
figures, dependency information, external-resource disclosure, and a package
manifest. It excludes official GOAI data, checkpoints, OOF/test predictions,
submission CSV files, credentials, and caches.

The current archive is a staging artifact. Replace every
`队伍名待替换` placeholder with the actual team name, regenerate the PDF, rename
the top-level directory and archive to `AI4R_ALG_AIVC_<team-name>.zip`, and rerun
the integrity audit before uploading it to the competition portal.
