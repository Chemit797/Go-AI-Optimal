# Released weights / 发布权重

This directory contains every checkpoint required by `scripts/predict_m12.py`.
It does not contain official GOAI data, OOF predictions, test predictions, or
submission files. Run `python scripts/verify_release.py` before inference.

本目录包含 `scripts/predict_m12.py` 完整推理所需的全部 checkpoint，不含 GOAI
官方原始数据、OOF 预测、测试预测或提交文件。推理前请执行
`python scripts/verify_release.py` 核对大小和 SHA256。

| Directory | Role | Seeds |
|---|---|---|
| `m2/mse/` | M2.0 learned-rank-64 fallback | 42, 43, 2026 |
| `m2/huber/` | M2.31 robust auxiliary fallback | 42, 43, 2026 |
| `m6/concat256/` | M6.11 cell-conditioned background/response | 42, 43, 2026 |
| `m6/film256/` | M6.21 time-extrapolation route | 42, 43, 2026 |
| `m9/op3_residual/` | M9.6 OP3 chemical-response transfer | 42, 43, 2026 |
| `pretrained/` | Frozen OP3 RNA chemical encoder provenance artifact | one |

The Python source is MIT licensed. Checkpoint use remains subject to the GOAI
official-data terms and the external OP3 CC BY 4.0 attribution described in
`EXTERNAL_RESOURCES.md`.

源码采用 MIT License。权重使用仍须遵守 GOAI 官方数据条款，以及
`EXTERNAL_RESOURCES.md` 中记录的 OP3 CC BY 4.0 署名要求。
