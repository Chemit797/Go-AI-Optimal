from __future__ import annotations

import torch
import pytest

from goai_response.model import ResponseDecompositionRegressor


def test_response_model_returns_three_aligned_heads():
    model = ResponseDecompositionRegressor(5, 4, 3, 7, 8, 2, 2, 0.0)
    absolute, background, response = model(torch.randn(3, 5), torch.randn(3, 4), torch.randn(3, 3), torch.tensor([[0.0], [1.0], [1.0]]))
    assert absolute.shape == background.shape == response.shape == (3, 7)
    assert torch.allclose(absolute[0], background[0])


def test_response_model_can_disable_calibration():
    model = ResponseDecompositionRegressor(5, 4, 3, 7, 8, 2, 2, 0.0, calibration_enabled=False)
    absolute, background, _ = model(torch.randn(2, 5), torch.randn(2, 4), torch.randn(2, 3), torch.zeros(2, 1))
    assert torch.allclose(absolute, background)


def test_fixed_svd_basis_is_frozen_and_centered():
    model = ResponseDecompositionRegressor(
        5, 4, 3, 7, 8, 2, 2, 0.0, response_basis="fixed_svd"
    )
    center = torch.arange(7, dtype=torch.float32)
    basis = torch.zeros(2, 7)
    model.set_fixed_response_basis(center, basis)
    _, _, response = model(
        torch.randn(3, 5),
        torch.randn(3, 4),
        torch.randn(3, 3),
        torch.ones(3, 1),
    )
    assert torch.allclose(response, center.expand_as(response))
    assert "response_proteins" not in dict(model.named_parameters())
    assert "response_proteins" in dict(model.named_buffers())


def test_fixed_basis_rejects_wrong_shape():
    model = ResponseDecompositionRegressor(
        5, 4, 3, 7, 8, 2, 2, 0.0, response_basis="fixed_svd"
    )
    with pytest.raises(ValueError):
        model.set_fixed_response_basis(torch.zeros(6), torch.zeros(2, 7))


@pytest.mark.parametrize("mode", ["shared_concat", "shared_gate", "shared_film"])
def test_shared_cell_modes_produce_aligned_outputs(mode):
    model = ResponseDecompositionRegressor(
        9, 4, 3, 7, 8, 2, 2, 0.0,
        cell_input_dim=4,
        perturbation_input_dim=5,
        interaction_mode=mode,
    )
    absolute, background, response = model(
        torch.randn(3, 9),
        torch.randn(3, 4),
        torch.randn(3, 3),
        torch.tensor([[0.0], [1.0], [1.0]]),
        torch.randn(3, 4),
        torch.randn(3, 5),
        torch.zeros(3, 7),
    )
    assert absolute.shape == background.shape == response.shape == (3, 7)
    assert torch.allclose(absolute[0], background[0])


def test_response_prior_is_added_exactly_when_residual_is_zero():
    model = ResponseDecompositionRegressor(3, 2, 2, 4, 3, 2, 2, 0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    prior = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    _, _, response = model(
        torch.zeros(2, 3), torch.zeros(2, 2), torch.zeros(2, 2),
        torch.ones(2, 1), response_prior=prior,
    )
    assert torch.allclose(response, prior)


def test_independent_legacy_path_matches_original_equation_exactly():
    torch.manual_seed(9)
    model = ResponseDecompositionRegressor(5, 4, 3, 7, 8, 2, 2, 0.0)
    response_inputs = torch.randn(3, 5)
    background_inputs = torch.randn(3, 4)
    observation_inputs = torch.randn(3, 3)
    treatment = torch.tensor([[0.0], [1.0], [1.0]])
    expected_background = model.background_encoder(background_inputs)
    expected_response = model.response_center + model.response_encoder(response_inputs) @ model.response_proteins
    expected_calibration = model.calibration_encoder(observation_inputs) @ model.calibration_proteins
    expected_absolute = expected_background + expected_calibration + treatment * expected_response
    absolute, background, response = model(response_inputs, background_inputs, observation_inputs, treatment)
    assert torch.equal(absolute, expected_absolute)
    assert torch.equal(background, expected_background + expected_calibration)
    assert torch.equal(response, expected_response)


def test_calibration_is_hard_centered_on_fold_fit_observations():
    model = ResponseDecompositionRegressor(5, 4, 3, 7, 8, 2, 2, 0.0)
    observations = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    model.set_calibration_input_center(observations.mean(dim=0))
    model.eval()
    output = model.forward_named_components(
        torch.zeros(4, 5),
        torch.zeros(4, 4),
        observations,
        torch.zeros(4, 1),
    )
    assert torch.allclose(output.calibration.mean(dim=0), torch.zeros(7), atol=1e-7)

    # Calibration remains a function of observation metadata only: changing
    # biological inputs with the same observation vector cannot change C_obs.
    changed = model.forward_named_components(
        torch.randn(4, 5) * 100.0,
        torch.randn(4, 4) * 100.0,
        observations,
        torch.zeros(4, 1),
    )
    assert torch.equal(output.calibration, changed.calibration)


def test_plate_dropout_masks_centered_deviation_not_fold_mean():
    model = ResponseDecompositionRegressor(
        3, 2, 2, 4, 3, 2, 2, 0.0,
        calibration_plate_start=0,
        calibration_plate_end=2,
        calibration_plate_dropout=0.999,
    )
    center = torch.tensor([0.25, 0.75])
    model.set_calibration_input_center(center)
    with torch.no_grad():
        model.calibration_encoder.weight.fill_(1.0)
        model.calibration_proteins.fill_(1.0)
    model.train()
    output = model.forward_named_components(
        torch.zeros(2, 3),
        torch.zeros(2, 2),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.zeros(2, 1),
    )
    # With both centered plate deviations dropped, the calibration is exactly
    # zero. Dropping raw one-hot entries before centering would be non-zero.
    assert torch.allclose(output.calibration, torch.zeros(2, 4))
