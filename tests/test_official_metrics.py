from __future__ import annotations

import numpy as np

from goai_baseline.config import load_config
from goai_baseline.official_metrics import absolute_fidelity, evaluate_official_proxy
from goai_baseline.preprocess import prepare_data

from .conftest import write_config


def test_absolute_fidelity_is_perfect_for_observed_truth(tmp_path):
    data = prepare_data(load_config(write_config(tmp_path)))
    truth = data.y_log2.loc[data.train_ids]
    report = absolute_fidelity(truth.fillna(0.0), truth)
    assert np.isclose(report["absolute_sample_pcc_median"], 1.0)
    assert np.isclose(report["absolute_sample_r2_median"], 1.0)
    assert np.isclose(report["absolute_protein_pcc_median"], 1.0)
    assert np.isclose(report["absolute_protein_r2_median"], 1.0)


def test_proxy_reports_all_frozen_splits(tmp_path):
    data = prepare_data(load_config(write_config(tmp_path)))

    def perfect_predictor(ids):
        return data.y_log2.loc[ids].fillna(0.0)

    report = evaluate_official_proxy(data, perfect_predictor)
    assert set(report["split"]) == {"val_chem_only", "val_strain_only", "val_both", "val_time"}
    assert np.isfinite(report["absolute_sample_r2_median"]).all()
    assert report.loc[report["split"].eq("val_chem_only"), "response_n_samples"].item() == 1
    assert (report.loc[~report["split"].eq("val_chem_only"), "response_n_samples"] == 0).all()
