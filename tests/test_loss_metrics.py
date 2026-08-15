from __future__ import annotations

import numpy as np
import torch

from goai_baseline.loss import masked_mse
from goai_baseline.metrics import masked_global_r2, masked_rmse, protein_r2


def test_masked_mse_ignores_filled_missing_targets():
    prediction = torch.tensor([[1.0, 2.0]])
    target_a = torch.tensor([[1.0, 0.0]])
    target_b = torch.tensor([[1.0, 999.0]])
    mask = torch.tensor([[1.0, 0.0]])
    assert masked_mse(prediction, target_a, mask).item() == 0.0
    assert masked_mse(prediction, target_b, mask).item() == 0.0


def test_mask_aware_metrics():
    truth = np.array([[1.0, np.nan], [3.0, 2.0]])
    prediction = np.array([[1.0, 100.0], [3.0, 2.0]])
    mask = ~np.isnan(truth)
    assert masked_rmse(prediction, truth, mask) == 0.0
    assert masked_global_r2(prediction, truth, mask) == 1.0
    values = protein_r2(prediction, truth, mask)
    assert values[0] == 1.0
    assert np.isnan(values[1])
