from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from goai_response.nested_scale import (
    PROTOCOL,
    active_expert_axes,
    compose_scaled_prediction,
    select_nested_scales,
    validate_receipt,
    write_receipt,
)


def _model(**changes):
    values = {
        "background_strain_expert_enabled": True,
        "response_strain_expert_enabled": True,
        "response_chemical_expert_enabled": True,
        "response_pair_expert_enabled": True,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _candidate(
    strain: float,
    chemical: float,
    pair: float,
    fc: float,
    *,
    context: float = 0.20,
    drug: float = 0.21,
    high_pcc: float = 0.50,
    high_f1: float = 0.30,
) -> dict[str, float]:
    return {
        "strain_scale": strain,
        "chemical_scale": chemical,
        "pair_scale": pair,
        "fc_pcc": fc,
        "context_residual_pcc": context,
        "drug_residual_pcc": drug,
        "high_effect_pcc": high_pcc,
        "high_effect_f1": high_f1,
    }


def test_outer_validation_labels_cannot_enter_pure_inner_selection() -> None:
    # The selector receives only already aggregated inner-OOF metrics.  An
    # arbitrary mutation of a separate outer-label object has no data path to
    # the selected scale.
    candidates = pd.DataFrame(
        [_candidate(0.0, 0.0, 0.0, 0.30), _candidate(0.5, 0.0, 0.0, 0.34)]
    )
    outer_labels = np.asarray([1.0, 2.0])
    first, _ = select_nested_scales(candidates, "R10", ("strain",))
    outer_labels[:] = [-999.0, 999.0]
    second, _ = select_nested_scales(candidates, "R10", ("strain",))
    assert first == second == {"strain": 0.5, "chemical": 0.0, "pair": 0.0}


def test_inner_oof_metric_change_can_change_selected_scale() -> None:
    candidates = pd.DataFrame(
        [_candidate(0.0, 0.0, 0.0, 0.30), _candidate(0.5, 0.0, 0.0, 0.34)]
    )
    first, _ = select_nested_scales(candidates, "R10", ("strain",))
    # This represents an inner-label-dependent score change; no outer score is
    # accepted by the API.
    changed = candidates.copy()
    changed.loc[changed["strain_scale"].eq(0.5), "fc_pcc"] = 0.29
    second, _ = select_nested_scales(changed, "R10", ("strain",))
    assert first["strain"] == 0.5
    assert second["strain"] == 0.0


def test_guardrails_and_lower_scale_tie_break_are_locked() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(0.0, 0.0, 0.0, 0.30),
            _candidate(0.25, 0.0, 0.0, 0.34),
            _candidate(0.50, 0.0, 0.0, 0.34),
            # Best FC, but the relevant residual regresses.
            _candidate(0.75, 0.0, 0.0, 0.40, context=0.19),
        ]
    )
    selected, audited = select_nested_scales(candidates, "R10", ("strain",))
    assert selected["strain"] == 0.25
    assert not bool(
        audited.loc[audited["strain_scale"].eq(0.75), "guardrail_pass"].iloc[0]
    )


def test_r00_has_no_identifiable_expert_scale_and_zero_gate_is_invariant() -> None:
    assert active_expert_axes(_model(), "R00") == ()
    shape = (2, 3)
    components = {
        "B_U": np.ones(shape, dtype=np.float32),
        "B_s": np.full(shape, 1000.0, dtype=np.float32),
        "C_obs": np.full(shape, 2.0, dtype=np.float32),
        "R_U": np.full(shape, 3.0, dtype=np.float32),
        "R_s": np.full(shape, 1000.0, dtype=np.float32),
        "R_c": np.full(shape, 1000.0, dtype=np.float32),
        "R_sc": np.full(shape, 1000.0, dtype=np.float32),
    }
    result = compose_scaled_prediction(
        components,
        np.ones(2, dtype=np.float32),
        {"strain": 0.0, "chemical": 0.0, "pair": 0.0},
    )
    np.testing.assert_allclose(result, 6.0)


def test_nested_receipt_resume_rejects_tampered_evidence(tmp_path) -> None:
    directory = tmp_path / "nested"
    payload = {
        "protocol": PROTOCOL,
        "status": "not_applicable",
        "scenario": "R00",
        "outer_fold": 3,
        "outer_train_ids_sha256": "train",
        "outer_validation_ids_sha256": "validation",
        "source_contract_fingerprint_sha256": "source",
        "global_scale_used": False,
        "outer_validation_labels_used": False,
        "canonical_training_scales": {
            "strain": 1.0,
            "chemical": 1.0,
            "pair": 1.0,
        },
        "active_axes": [],
        "inner_n_folds": 2,
        "selected_scales": {"strain": 0.0, "chemical": 0.0, "pair": 0.0},
    }
    _, receipt_hash = write_receipt(
        directory,
        payload=payload,
        assignments=pd.DataFrame(
            columns=["scenario", "fold", "sample_id", "eligible"]
        ),
        candidates=pd.DataFrame(
            [{"scenario": "R00", "status": "not_applicable"}]
        ),
        fit_receipts=[],
    )
    validate_receipt(
        directory,
        expected_sha256=receipt_hash,
        expected_scenario="R00",
        expected_fold=3,
        expected_train_ids_sha256="train",
        expected_validation_ids_sha256="validation",
        expected_source_contract_sha256="source",
    )
    with (directory / "candidate_metrics.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="tampered"):
        validate_receipt(directory, expected_sha256=receipt_hash)
