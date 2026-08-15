from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goai_baseline.csv_scorer import load_csv_inputs, score_files


def _write_pair(tmp_path, prediction_values=None):
    log2_truth = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [11.0, 13.0, 15.0],
            [13.0, 16.0, 19.0],
        ]
    )
    ids = ["s1", "s2", "s3"]
    truth = pd.DataFrame(np.exp2(log2_truth), columns=["P1", "P2", "P3"])
    truth.insert(0, "sample_ID", ids)
    truth["split_final"] = ["test", "test", "test"]
    prediction = pd.DataFrame(
        log2_truth if prediction_values is None else prediction_values,
        columns=["P1", "P2", "P3"],
    )
    prediction.insert(0, "sample_ID", list(reversed(ids)))
    prediction.loc[:, ["P1", "P2", "P3"]] = prediction.loc[::-1, ["P1", "P2", "P3"]].to_numpy()
    truth_path = tmp_path / "test.csv"
    prediction_path = tmp_path / "prediction.csv"
    truth.to_csv(truth_path, index=False)
    prediction.to_csv(prediction_path, index=False)
    return truth_path, prediction_path


def test_perfect_prediction_aligns_ids_and_converts_raw_truth(tmp_path):
    truth_path, prediction_path = _write_pair(tmp_path)

    report = score_files(truth_path, prediction_path)

    assert report["truth_input_scale"] == "raw"
    assert report["prediction_order_matched_truth"] is False
    assert report["ignored_truth_columns"] == ["split_final"]
    assert report["published_modules_status"] == "support_files_not_found"
    absolute = report["absolute_fidelity"]
    assert absolute["absolute_n_samples"] == 3
    assert absolute["absolute_n_observed_values"] == 9
    for name in (
        "absolute_sample_pcc_median",
        "absolute_sample_r2_median",
        "absolute_protein_pcc_median",
        "absolute_protein_r2_median",
    ):
        assert np.isclose(absolute[name], 1.0)


def test_log2_scale_can_be_forced(tmp_path):
    truth_path, prediction_path = _write_pair(tmp_path)
    truth = pd.read_csv(truth_path)
    truth.loc[:, ["P1", "P2", "P3"]] = np.log2(truth.loc[:, ["P1", "P2", "P3"]])
    truth.to_csv(truth_path, index=False)

    inputs = load_csv_inputs(truth_path, prediction_path, truth_scale="log2")

    assert inputs.truth_scale == "log2"
    np.testing.assert_allclose(inputs.truth.to_numpy(), inputs.prediction.to_numpy())


def test_mismatched_sample_ids_are_rejected(tmp_path):
    truth_path, prediction_path = _write_pair(tmp_path)
    prediction = pd.read_csv(prediction_path)
    prediction.loc[0, "sample_ID"] = "unexpected"
    prediction.to_csv(prediction_path, index=False)

    with pytest.raises(ValueError, match="sample_ID"):
        load_csv_inputs(truth_path, prediction_path)


def test_prediction_missing_answer_protein_column_is_rejected(tmp_path):
    truth_path, prediction_path = _write_pair(tmp_path)
    prediction = pd.read_csv(prediction_path).drop(columns="P3")
    prediction.to_csv(prediction_path, index=False)

    with pytest.raises(ValueError, match="预测缺少"):
        load_csv_inputs(truth_path, prediction_path)


def test_prediction_extra_column_is_rejected(tmp_path):
    truth_path, prediction_path = _write_pair(tmp_path)
    prediction = pd.read_csv(prediction_path)
    prediction["not_a_protein"] = 1.0
    prediction.to_csv(prediction_path, index=False)

    with pytest.raises(ValueError, match="预测多出"):
        load_csv_inputs(truth_path, prediction_path)


def test_nonfinite_prediction_is_rejected(tmp_path):
    truth_path, prediction_path = _write_pair(tmp_path)
    prediction = pd.read_csv(prediction_path)
    prediction.loc[0, "P1"] = np.nan
    prediction.to_csv(prediction_path, index=False)

    with pytest.raises(ValueError, match="不能包含 NaN"):
        load_csv_inputs(truth_path, prediction_path)


def _metadata_row(sample_id, chemical, split, plate="plate1"):
    return {
        "sample_ID": sample_id,
        "data_source": "source1",
        "Strains": "strain1",
        "Medium": "glucose",
        "Temperature": 30,
        "pert_time": 60,
        "pert_time_unit": "min",
        "pert_id": f"#{sample_id}",
        "perturbation_no_concentration": chemical,
        "instrument": "instrument1",
        "Yeast_cell_plate": plate,
        "protein_well": "A1",
        "split_final": split,
        "strain_role": "train" if split == "train" else "test",
        "chemical_role": "train" if chemical in {"Water", "DrugA", "DrugB"} else "test",
    }


def test_full_public_modules_are_computed_with_support_files(tmp_path):
    proteins = ["P1", "P2", "P3"]
    control = np.asarray([10.0, 11.0, 12.0])
    train_rows = [
        _metadata_row("tr_ctrl", "Water", "train"),
        _metadata_row("tr_a", "DrugA", "train"),
        _metadata_row("tr_b", "DrugB", "train"),
    ]
    train_log2 = np.vstack(
        [
            control,
            control + np.asarray([1.0, -1.0, 0.5]),
            control + np.asarray([-0.5, 1.0, -1.0]),
        ]
    )

    test_rows = []
    test_log2 = []
    split_chemicals = {
        "test_chem_only": "DrugNew",
        "test_strain_only": "DrugA",
        "test_both": "DrugNew",
        "test_time": "DrugA",
    }
    deltas = {
        "test_chem_only": np.asarray([2.0, -2.0, 3.0]),
        "test_strain_only": np.asarray([3.0, -2.0, 2.0]),
        "test_both": np.asarray([2.0, -3.0, 4.0]),
        "test_time": np.asarray([4.0, -2.0, 3.0]),
    }
    for index, (split, chemical) in enumerate(split_chemicals.items()):
        ctrl_id = f"te_ctrl_{index}"
        treat_id = f"te_treat_{index}"
        test_rows.extend(
            [
                _metadata_row(ctrl_id, "Water", split),
                _metadata_row(treat_id, chemical, split),
            ]
        )
        test_log2.extend([control, control + deltas[split]])

    train_metadata_path = tmp_path / "train_metadata.csv"
    train_proteome_path = tmp_path / "train_proteome.csv"
    test_metadata_path = tmp_path / "test_metadata.csv"
    truth_path = tmp_path / "test.csv"
    prediction_path = tmp_path / "prediction.csv"
    pd.DataFrame(train_rows).to_csv(train_metadata_path, index=False)
    train_raw = pd.DataFrame(np.exp2(train_log2), columns=proteins)
    train_raw.insert(0, "sample_ID", [row["sample_ID"] for row in train_rows])
    train_raw.to_csv(train_proteome_path, index=False)
    pd.DataFrame(test_rows).to_csv(test_metadata_path, index=False)
    truth_raw = pd.DataFrame(np.exp2(np.asarray(test_log2)), columns=proteins)
    truth_raw.insert(0, "sample_ID", [row["sample_ID"] for row in test_rows])
    truth_raw.to_csv(truth_path, index=False)
    prediction = pd.DataFrame(test_log2, columns=proteins)
    prediction.insert(0, "sample_ID", [row["sample_ID"] for row in test_rows])
    prediction.to_csv(prediction_path, index=False)

    report = score_files(
        truth_path,
        prediction_path,
        metadata_test_path=test_metadata_path,
        metadata_train_path=train_metadata_path,
        proteome_train_path=train_proteome_path,
    )

    assert report["published_modules_status"] == "computed_with_public_handbook_proxy"
    assert len(report["split_metrics"]) == 4
    assert np.isclose(report["overall_metrics"]["fc_pcc"], 1.0)
    assert np.isclose(report["weighted_proxy"]["score"], 100.0)

    references_path = tmp_path / "goai_scoring_references.npz"
    np.savez_compressed(
        references_path,
        proteins=np.asarray(proteins),
        context_keys=np.asarray(
            [["source1", "instrument1", "plate1", "strain1", "glucose", "30", "60", "min"]]
        ),
        context_values=np.asarray([[0.25, 0.0, -0.25]], dtype=np.float32),
        drug_keys=np.asarray(["DrugA", "DrugB"]),
        drug_values=np.asarray(
            [[1.0, -1.0, 0.5], [-0.5, 1.0, -1.0]],
            dtype=np.float32,
        ),
    )
    frozen_report = score_files(
        truth_path,
        prediction_path,
        metadata_test_path=test_metadata_path,
        references_path=references_path,
    )

    assert frozen_report["support_files"]["frozen_references"] == str(references_path)
    assert np.isclose(frozen_report["weighted_proxy"]["score"], 100.0)
