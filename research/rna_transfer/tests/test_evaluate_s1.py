from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluate_s1 import (
    PredictionPayload,
    PredictionRequest,
    build_bootstrap_sufficient_statistics,
    build_fold_train_context_reference,
    evaluate_prediction,
    load_aligned_prediction,
    load_s1_cache,
    paired_cluster_bootstrap,
    summarize_folds,
)


def _write_synthetic_contract(root: Path):
    sample_ids = np.asarray([f"s{index}" for index in range(8)])
    chemicals = np.asarray([f"drug{index}" for index in range(8)])
    proteins = np.asarray(["p0", "p1", "p2", "p3"])
    folds = np.repeat(np.arange(4), 2)
    true_delta = np.asarray(
        [
            [1.5 + 0.1 * index, -1.4 - 0.1 * index, 0.20 * index, -0.15 * index]
            for index in range(8)
        ],
        dtype=np.float32,
    )
    matched = np.tile(np.asarray([10.0, 11.0, 12.0, 13.0], dtype=np.float32), (8, 1))
    truth = matched + true_delta
    mask = np.ones_like(true_delta, dtype=bool)
    cache_path = root / "cache.npz"
    np.savez_compressed(
        cache_path,
        sample_ids=sample_ids,
        chemicals=chemicals,
        proteins=proteins,
        folds=folds,
        delta=true_delta,
        mask=mask,
        matched_control=matched,
        matched_control_mask=mask,
        treatment_truth=truth,
        truth_mask=mask,
        context_keys=np.asarray(["shared"] * 8),
    )

    fold_dir = root / "delta_folds"
    fold_dir.mkdir()
    for fold in range(4):
        rows = folds == fold
        np.savez_compressed(
            fold_dir / f"fold_{fold}.npz",
            sample_ids=sample_ids[rows],
            chemicals=chemicals[rows],
            proteins=proteins,
            pred_delta=true_delta[rows],
            fold=np.asarray(fold),
        )
    absolute_path = root / "legacy_S1.npz"
    order = np.asarray([7, 5, 3, 1, 6, 4, 2, 0])
    np.savez_compressed(
        absolute_path,
        sample_ids=sample_ids[order],
        protein_ids=proteins,
        values=truth[order],
    )
    return cache_path, fold_dir, absolute_path


def test_delta_folds_and_legacy_absolute_score_identically(tmp_path):
    cache_path, fold_dir, absolute_path = _write_synthetic_contract(tmp_path)
    cache = load_s1_cache(cache_path)
    context = build_fold_train_context_reference(cache)

    delta = load_aligned_prediction(
        PredictionRequest("delta", "delta", [fold_dir]), cache
    )
    absolute = load_aligned_prediction(
        PredictionRequest("absolute", "absolute", [absolute_path]), cache
    )
    delta_folds, delta_clusters = evaluate_prediction("delta", delta, cache, context)
    absolute_folds, _ = evaluate_prediction("absolute", absolute, cache, context)

    for metric in (
        "fc_pcc",
        "context_residual_pcc",
        "high_effect_pcc",
        "high_effect_f1",
    ):
        np.testing.assert_allclose(delta_folds[metric], 1.0, atol=1e-6)
        np.testing.assert_allclose(absolute_folds[metric], 1.0, atol=1e-6)
    # A delta-only model has no independently predicted absolute background;
    # using the observed validation control here would be an oracle protocol.
    assert delta_folds["absolute_sample_r2_median"].isna().all()
    np.testing.assert_allclose(
        absolute_folds["absolute_sample_r2_median"], 1.0, atol=1e-6
    )
    summary = summarize_folds(pd.concat([delta_folds, absolute_folds], ignore_index=True))
    assert summary["n_scored_folds"].tolist() == [4, 4]
    assert len(delta_clusters) == 8


def test_repeated_full_predictions_are_seed_averaged(tmp_path):
    cache_path, _, absolute_path = _write_synthetic_contract(tmp_path)
    cache = load_s1_cache(cache_path)
    payload = load_aligned_prediction(
        PredictionRequest("bag", "absolute", [absolute_path, absolute_path]), cache
    )
    expected = cache.truth_absolute
    np.testing.assert_allclose(payload.values, expected, atol=1e-7)


def test_repeated_label_rejects_complementary_partial_seed_paths(tmp_path):
    cache_path, fold_dir, _ = _write_synthetic_contract(tmp_path)
    cache = load_s1_cache(cache_path)
    with pytest.raises(ValueError, match="every path entry to cover all S1 rows"):
        load_aligned_prediction(
            PredictionRequest(
                "invalid_bag",
                "delta",
                [fold_dir / "fold_0.npz", fold_dir / "fold_1.npz"],
            ),
            cache,
        )


def test_paired_bootstrap_recomputes_fold_metrics_from_cluster_moments(tmp_path):
    cache_path, fold_dir, _ = _write_synthetic_contract(tmp_path)
    cache = load_s1_cache(cache_path)
    context = build_fold_train_context_reference(cache)
    candidate = load_aligned_prediction(
        PredictionRequest("candidate", "delta", [fold_dir]), cache
    )
    control_values = candidate.values.copy()
    control_values[:, 0] *= -0.8
    control_values[:, 2] += 0.5 * candidate.values[:, 0]
    control = PredictionPayload(
        sample_ids=candidate.sample_ids,
        proteins=candidate.proteins,
        values=control_values,
        kind="delta",
        source_files=[],
    )

    candidate_folds, candidate_clusters = evaluate_prediction(
        "candidate", candidate, cache, context
    )
    control_folds, control_clusters = evaluate_prediction(
        "control", control, cache, context
    )
    statistics = {
        "candidate": build_bootstrap_sufficient_statistics(
            "candidate", candidate, cache, context
        ),
        "control": build_bootstrap_sufficient_statistics(
            "control", control, cache, context
        ),
    }
    result = paired_cluster_bootstrap(
        statistics, [("candidate", "control")], draws=200, seed=7
    )
    assert result["n_clusters"].eq(8).all()
    assert result["metric"].tolist() == [
        "fc_pcc",
        "context_residual_pcc",
        "high_effect_pcc",
        "high_effect_f1",
    ]

    candidate_summary = summarize_folds(candidate_folds).iloc[0]
    control_summary = summarize_folds(control_folds).iloc[0]
    for metric in result["metric"]:
        expected = (
            candidate_summary[metric + "_mean"]
            - control_summary[metric + "_mean"]
        )
        observed = result.loc[result["metric"].eq(metric), "mean_delta"].iloc[0]
        np.testing.assert_allclose(observed, expected, atol=1e-10)

    # The previous implementation averaged per-cluster correlations.  Its FC
    # estimate is deliberately different from the recomputed fold-macro PCC.
    old_candidate = candidate_clusters.set_index("chemical_cluster")["fc_pcc"]
    old_control = control_clusters.set_index("chemical_cluster")["fc_pcc"]
    old_delta = float((old_candidate - old_control).mean())
    new_delta = float(result.loc[result["metric"].eq("fc_pcc"), "mean_delta"].iloc[0])
    assert not np.isclose(old_delta, new_delta, atol=1e-5)
