"""Loss functions for partially observed protein matrices."""

from __future__ import annotations

import torch


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    denominator = mask.sum()
    if denominator.item() <= 0:
        raise ValueError("masked_mse requires at least one observed target")
    return (((prediction - target).square()) * mask).sum() / denominator
