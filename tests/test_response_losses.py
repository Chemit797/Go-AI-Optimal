from __future__ import annotations

import numpy as np
import torch

from goai_response.train import (
    _fit_fixed_svd_basis,
    _masked_correlation_loss,
    _masked_or_zero,
)


def test_masked_robust_losses_ignore_unobserved_values():
    prediction = torch.tensor([[1.0, 100.0]])
    target = torch.tensor([[0.0, -100.0]])
    mask = torch.tensor([[1.0, 0.0]])
    assert torch.isclose(_masked_or_zero(prediction, target, mask, "huber", 1.0), torch.tensor(0.5))
    assert torch.isclose(_masked_or_zero(prediction, target, mask, "mse_mae"), torch.tensor(1.0))


def test_masked_correlation_loss_handles_perfect_and_empty_masks():
    prediction = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[2.0, 4.0, 6.0]])
    mask = torch.ones_like(prediction)
    assert torch.isclose(_masked_correlation_loss(prediction, target, mask), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(_masked_correlation_loss(prediction, target, torch.zeros_like(mask)), torch.tensor(0.0))


def test_fixed_svd_basis_has_fold_local_center_and_requested_shape():
    fc = np.asarray([[1.0, 4.0, 0.0], [3.0, 8.0, 0.0], [0.0, 12.0, 0.0]], dtype=np.float32)
    mask = np.asarray([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    center, basis, summary = _fit_fixed_svd_basis(fc, mask, rank=4, device=torch.device("cpu"))
    assert torch.allclose(center, torch.tensor([2.0, 8.0, 0.0]))
    assert basis.shape == (4, 3)
    assert summary["training_response_rows"] == 3
    assert summary["effective_rank"] == 3
    assert 0.0 <= summary["explained_energy_ratio"] <= 1.00001
    gram = basis[:3] @ basis[:3].T
    assert torch.allclose(gram, torch.eye(3), atol=1e-5)
