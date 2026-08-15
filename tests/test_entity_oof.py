from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import yaml

from goai_baseline.config import load_config
from goai_baseline.official_metrics import evaluate_prediction_set
from goai_baseline.preprocess import prepare_data
from goai_response.config import load_response_config
from goai_response.entities import manifest_sha256
from goai_response.nested_scale import validate_receipt as validate_nested_scale_receipt
from goai_response.oof import SCENARIOS, assert_fold_isolation, make_fold_slices, run_entity_oof
import goai_response.oof as oof_module
from goai_response.predict import load_response_checkpoint
from goai_response.train import train_response_model

from .conftest import METADATA_COLUMNS, metadata_row


def _oof_files(root):
    rows = []
    values = []
    counter = 0
    for strain in ("S1", "S2", "S3", "S4"):
        for time in (15, 30):
            sample_id = f"control_{strain}_{time}"
            rows.append(metadata_row(sample_id, "Water", "train", strain=strain, time=time))
            values.append([8.0 + counter, 16.0 + counter])
            counter += 1
            for chemical in ("DrugA", "DrugB", "DrugC", "DrugD"):
                sample_id = f"treat_{strain}_{chemical}_{time}"
                rows.append(metadata_row(sample_id, chemical, "train", strain=strain, time=time))
                values.append([9.0 + counter, 17.0 + counter])
                counter += 1
    rows.extend(
        [
            metadata_row("outer_s1", "DrugZ", "val_chem_only", strain="S1"),
            metadata_row("outer_s2", "DrugA", "val_strain_only", strain="S9"),
            metadata_row("outer_s3", "DrugZ", "val_both", strain="S9"),
            metadata_row("outer_time", "DrugA", "val_time", strain="S1", time=60),
        ]
    )
    values.extend([[12.0, 20.0]] * 4)
    metadata = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    proteome = pd.DataFrame(values, columns=["P1", "P2"])
    proteome.insert(0, "sample_ID", metadata["sample_ID"])
    test = pd.DataFrame([metadata_row("test_1", "DrugZ", "test_chem_only")], columns=METADATA_COLUMNS)
    metadata.to_csv(root / "metadata.csv", index=False)
    proteome.to_csv(root / "proteome.csv", index=False)
    test.to_csv(root / "WAYB_WAYC_metadata_test.csv", index=False)
    baseline = {
        "data": {
            "metadata_train_val": "metadata.csv",
            "proteome_train_val": "proteome.csv",
            "metadata_test": "WAYB_WAYC_metadata_test.csv",
            "missing_rate_threshold": 0.8,
        },
        "model": {"hidden_dim": 4, "dropout": 0.0, "learning_rate": 0.001, "epochs": 1, "seed": 17, "device": "cpu"},
        "features": {"chemical_hash_dim": 4},
        "runtime": {"runs_dir": "runs"},
    }
    response = {
        "baseline_config": "baseline.yaml",
        "model": {
            "hidden_dim": 4,
            "response_rank": 2,
            "calibration_rank": 2,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "epochs": 1,
            "batch_size": 64,
            "seed": 17,
            "device": "cpu",
            "absolute_weight": 0.25,
            "background_weight": 1.0,
            "fc_weight": 1.0,
            "target_scale_floor": 0.1,
            "calibration_enabled": True,
        },
        "entity": {"chemical_map": None, "strain_features": None, "chemical_bits": 8},
        "graph": {"variant": "none", "artifact": None, "weight": 0.0},
        "runtime": {"runs_dir": "runs"},
    }
    with (root / "baseline.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(baseline, handle)
    with (root / "response.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(response, handle)
    return root / "response.yaml"


def _attach_authoritative_registries(response_path):
    root = response_path.parent
    metadata = pd.concat(
        [
            pd.read_csv(root / "metadata.csv"),
            pd.read_csv(root / "WAYB_WAYC_metadata_test.csv"),
        ],
        ignore_index=True,
    )
    chemicals = sorted(metadata["perturbation_no_concentration"].astype(str).unique())
    strains = sorted(metadata["Strains"].astype(str).unique())
    chemical_registry = root / "chemical_registry.tsv"
    strain_registry = root / "strain_registry.tsv"
    pd.DataFrame(
        [
            {
                "raw_name": name,
                "canonical_id": f"test-chemical:{name}",
                "canonical_name": name,
                "mapping_status": "high_confidence_candidate",
                "evidence_tier": "B_primary_candidate",
                "proxy_target": "",
                "is_control": name == "Water",
                "is_quality_control": False,
            }
            for name in chemicals
        ]
    ).to_csv(chemical_registry, sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "strain_code": name,
                "canonical_id": f"test-strain:{name}",
                "canonical_name": name,
                "mapping_status": "high_confidence_candidate",
                "evidence_tier": "B_primary_candidate",
                "proxy_target": "",
                "is_control": False,
                "is_quality_control": False,
            }
            for name in strains
        ]
    ).to_csv(strain_registry, sep="\t", index=False)
    with response_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["entity"]["chemical_registry"] = chemical_registry.name
    payload["entity"]["strain_registry"] = strain_registry.name
    payload["entity"]["allow_proxy_semantics"] = False
    payload["entity"]["chemical_structure_view"] = "exact"
    with response_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)
    return response_path


def test_entity_folds_are_deterministic_and_isolated(tmp_path):
    config = load_response_config(_oof_files(tmp_path))
    data = prepare_data(config.baseline)
    first, first_assignments = make_fold_slices(data.metadata, data.train_ids, n_folds=2, seed=11)
    second, second_assignments = make_fold_slices(data.metadata, data.train_ids, n_folds=2, seed=11)

    pd.testing.assert_frame_equal(first_assignments, second_assignments)
    assert [(item.scenario, item.fold, item.validation_ids.tolist()) for item in first] == [
        (item.scenario, item.fold, item.validation_ids.tolist()) for item in second
    ]
    assert set(first_assignments["scenario"]) == set(SCENARIOS)
    assert not set(first_assignments["sample_ID"]) & {"outer_s1", "outer_s2", "outer_s3", "outer_time"}
    for fold in first:
        assert_fold_isolation(data.metadata, fold)
    s3 = first_assignments.loc[first_assignments["scenario"].eq("S3") & first_assignments["eligible"]]
    treatments = data.train_ids[data.metadata.loc[data.train_ids, "perturbation_no_concentration"].ne("Water").to_numpy()]
    assert set(s3["sample_ID"]) == set(treatments)
    forward = first_assignments.loc[
        first_assignments["scenario"].eq("time_forward") & first_assignments["eligible"]
    ]
    assert set(forward["pert_time"]) == {30}
    assert len(forward) == 16


def test_plate_diagnostic_holds_out_complete_plates(tmp_path):
    config = load_response_config(_oof_files(tmp_path))
    data = prepare_data(config.baseline)
    metadata = data.metadata.copy()
    metadata.loc[data.train_ids, "Yeast_cell_plate"] = metadata.loc[data.train_ids, "Strains"].astype(str).map(
        {"S1": "P1", "S2": "P2", "S3": "P3", "S4": "P4"}
    )
    slices, assignments = make_fold_slices(metadata, data.train_ids, n_folds=2, seed=19, scenarios=("plate",))
    assert set(assignments["scenario"]) == {"plate"}
    assert assignments["sample_ID"].nunique() == len(data.train_ids)
    for fold in slices:
        assert not set(metadata.loc[fold.train_ids, "Yeast_cell_plate"]) & set(metadata.loc[fold.validation_ids, "Yeast_cell_plate"])


def test_oof_proxy_cannot_use_validation_control(tmp_path):
    config = load_response_config(_oof_files(tmp_path))
    data = prepare_data(config.baseline)
    target_id = pd.Index(["treat_S1_DrugA_15"])
    fold_train = data.train_ids.difference(pd.Index(["control_S1_15", *target_id]))
    fold_data = replace(data, train_ids=fold_train)
    prediction = data.y_log2.loc[target_id].fillna(0.0)
    result = evaluate_prediction_set(fold_data, prediction, target_id, "unit", control_pool_ids=fold_train)
    assert result["response_n_samples"] == 0


def test_entity_oof_end_to_end_writes_reproducible_artifacts(tmp_path):
    config = load_response_config(_oof_files(tmp_path))
    output = run_entity_oof(config, tmp_path / "oof-run", n_folds=2, seed=23, scenarios=("S1", "S2", "S3"))
    for name in (
        "fold_assignments.csv",
        "oof_metrics.csv",
        "oof_summary.csv",
        "oof_official_proxy_metrics.csv",
        "oof_manifest.json",
        "run_contract.json",
    ):
        assert (output / name).is_file()
    assignments = pd.read_csv(output / "fold_assignments.csv")
    summary = pd.read_csv(output / "oof_summary.csv")
    assert set(assignments["scenario"]) == {"S1", "S2", "S3"}
    assert set(summary["scenario"]) == {"S1", "S2", "S3"}
    assert "fc_pcc_mean" in summary.columns
    assert (output / "training_histories" / "S1_fold_0.csv").is_file()
    for scenario in ("S1", "S2", "S3"):
        prediction = pd.read_csv(output / "oof_predictions" / f"{scenario}.csv")
        assert prediction["sample_ID"].is_unique
        assert np.isfinite(prediction[["P1", "P2"]].to_numpy()).all()
        with np.load(output / "oof_predictions" / f"{scenario}.npz", allow_pickle=False) as payload:
            assert payload["values"].shape[1] == 2
            assert payload["protein_ids"].astype(str).tolist() == ["P1", "P2"]
    with (output / "oof_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with (output / "run_contract.json").open("r", encoding="utf-8") as handle:
        run_contract = json.load(handle)
    source = run_contract["source_fingerprint"]
    assert source["sha256"]
    assert source["file_count"] == len(source["files"])
    source_paths = [item["path"] for item in source["files"]]
    assert source_paths == sorted(source_paths)
    assert "src/goai_response/oof.py" in source_paths
    assert "src/goai_baseline/preprocess.py" in source_paths
    assert any(path.startswith("src/goai_graph/") for path in source_paths)
    assert "scripts/nightly/run_matrix.py" in source_paths
    assert manifest["outer_validation_used_for_fitting_or_tuning"] is False
    assert manifest["protocol"] == "entity_oof_v1"
    assert manifest["seed"] == 23
    assert manifest["model_seed"] == 23
    assert manifest["scorer_only_controls_used_for_training"] is False
    s2_folds = [fold for fold in manifest["folds"] if fold["scenario"] == "S2"]
    assert all(fold["scorer_only_control_count"] > 0 for fold in s2_folds)
    official = pd.read_csv(output / "oof_official_proxy_metrics.csv")
    assert (official.loc[official["scenario"].eq("S2"), "response_n_samples"] > 0).all()


def test_standard_response_training_still_writes_checkpoint(tmp_path):
    config = load_response_config(_oof_files(tmp_path))
    output = train_response_model(config, tmp_path / "standard-run")
    assert (output / "checkpoint.pt").is_file()
    assert (output / "metrics.csv").is_file()


def test_final_refit_uses_all_released_labels_without_outer_rescoring(tmp_path):
    config = load_response_config(_oof_files(tmp_path))
    output = train_response_model(config, tmp_path / "final-refit", fit_all_labeled=True)
    with (output / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["fit_scope"] == "all_released_labeled_rows"
    assert manifest["fit_sample_count"] == 44
    assert manifest["outer_metrics_valid"] is False
    checkpoint = __import__("torch").load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["fit_scope"] == "all_released_labeled_rows"
    assert checkpoint["fit_sample_count"] == 44


def test_entity_oof_resume_reuses_completed_folds(tmp_path):
    config = load_response_config(_oof_files(tmp_path))
    output = run_entity_oof(
        config,
        tmp_path / "resumed-run",
        n_folds=2,
        seed=29,
        scenarios=("S1",),
    )
    marker = output / "folds" / "S1_fold_0" / "completed.json"
    first_mtime = marker.stat().st_mtime_ns
    resumed = run_entity_oof(
        config,
        output,
        n_folds=2,
        seed=29,
        scenarios=("S1",),
        resume=True,
    )
    assert resumed == output
    assert marker.stat().st_mtime_ns == first_mtime
    prediction = pd.read_csv(output / "oof_predictions" / "S1.csv")
    assert prediction["sample_ID"].is_unique


def test_formal_nested_scale_oof_is_fold_local_and_resume_validated(
    tmp_path, monkeypatch
):
    response_path = _attach_authoritative_registries(_oof_files(tmp_path))
    payload = yaml.safe_load(response_path.read_text(encoding="utf-8"))
    payload["model"].update(
        {
            "interaction_mode": "shared_general_experts",
            "background_strain_expert_enabled": True,
            "response_strain_expert_enabled": True,
            "response_chemical_expert_enabled": True,
            "response_pair_expert_enabled": False,
            "strain_expert_scale": 1.0,
            "chemical_expert_scale": 1.0,
            "pair_expert_scale": 1.0,
            "nested_expert_scale_selection": True,
            "nested_expert_scale_inner_folds": 2,
        }
    )
    response_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_response_config(response_path)
    # The two-protein toy fixture has no stable high-effect subset.  Supply a
    # finite deterministic scorer here so this test targets the orchestration,
    # component grid, evidence chain, and resume contract rather than metric
    # edge cases already covered by the pure selector tests.
    def finite_metrics(_data, prediction, _ids, label, control_pool_ids=None):
        del control_pool_ids
        return {
            "split": label,
            "fc_pcc": float(prediction.to_numpy().mean()),
            "context_residual_pcc": 0.20,
            "drug_residual_pcc": 0.21,
            "high_effect_pcc": 0.50,
            "high_effect_f1": 0.30,
        }

    monkeypatch.setattr(oof_module, "evaluate_prediction_set", finite_metrics)
    output = run_entity_oof(
        config,
        tmp_path / "nested-scale-run",
        n_folds=2,
        seed=43,
        scenarios=("R10",),
    )
    contract = json.loads((output / "run_contract.json").read_text(encoding="utf-8"))
    for fold in (0, 1):
        fold_dir = output / "folds" / f"R10_fold_{fold}"
        completion = json.loads(
            (fold_dir / "completed.json").read_text(encoding="utf-8")
        )
        receipt = validate_nested_scale_receipt(
            fold_dir / "nested_expert_scale",
            expected_sha256=completion["nested_expert_scale_receipt_sha256"],
            expected_scenario="R10",
            expected_fold=fold,
            expected_train_ids_sha256=completion["train_ids_sha256"],
            expected_validation_ids_sha256=completion["validation_ids_sha256"],
            expected_source_contract_sha256=contract["fingerprint_sha256"],
        )
        assert receipt["status"] == "selected"
        assert receipt["active_axes"] == ["strain"]
        assert receipt["scored_inner_fits"] >= 2
        assert receipt["global_scale_used"] is False
        assert receipt["outer_validation_labels_used"] is False
    resumed = run_entity_oof(
        config,
        output,
        n_folds=2,
        seed=43,
        scenarios=("R10",),
        resume=True,
    )
    assert resumed == output


def test_entity_oof_resume_rejects_changed_effective_config(tmp_path):
    config = load_response_config(_oof_files(tmp_path))
    output = run_entity_oof(
        config,
        tmp_path / "contract-run",
        n_folds=2,
        seed=31,
        scenarios=("S1",),
        audit_only=True,
    )
    changed = replace(config, model=replace(config.model, epochs=config.model.epochs + 1))
    with np.testing.assert_raises_regex(ValueError, "run contract does not match"):
        run_entity_oof(
            changed,
            output,
            n_folds=2,
            seed=31,
            scenarios=("S1",),
            audit_only=True,
            resume=True,
        )


def test_entity_oof_resume_rejects_changed_source_fingerprint(tmp_path, monkeypatch):
    config = load_response_config(_oof_files(tmp_path))
    output = run_entity_oof(
        config,
        tmp_path / "source-contract-run",
        n_folds=2,
        seed=37,
        scenarios=("S1",),
        audit_only=True,
    )
    original = oof_module._source_fingerprint()
    changed = {**original, "sha256": "f" * 64}
    monkeypatch.setattr(oof_module, "_source_fingerprint", lambda: changed)
    with np.testing.assert_raises_regex(ValueError, "run contract does not match source"):
        run_entity_oof(
            config,
            output,
            n_folds=2,
            seed=37,
            scenarios=("S1",),
            audit_only=True,
            resume=True,
        )


def test_authoritative_support_manifest_is_chained_into_run_and_checkpoint(tmp_path):
    config = load_response_config(_attach_authoritative_registries(_oof_files(tmp_path)))
    output = train_response_model(config, tmp_path / "support-run")
    support_path = output / "support_manifest.json"
    assert support_path.is_file()
    assert (output / "support_manifest.json.sha256").is_file()
    with support_path.open("r", encoding="utf-8") as handle:
        support = json.load(handle)
    checkpoint = __import__("torch").load(
        output / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["support_manifest"] == support
    assert checkpoint["support_manifest_sha256"] == manifest_sha256(support)
    assert checkpoint["artifact_hashes"]["chemical_registry"]["sha256"]
    assert checkpoint["artifact_hashes"]["strain_registry"]["sha256"]
    assert checkpoint["artifact_chain_sha256"]
    builder_state = checkpoint["feature_state"]
    assert sorted(builder_state["strain_entity_keys"]) == sorted(
        support["seen_support_keys"]["strain"]
    )
    assert sorted(builder_state["chemical_entity_keys"]) == sorted(
        support["seen_support_keys"]["chemical"]
    )
    assert sorted(map(tuple, builder_state["pair_entity_keys"])) == sorted(
        map(tuple, support["seen_support_keys"]["pair"])
    )
    assert sorted(map(tuple, builder_state["context_entity_keys"])) == sorted(
        map(tuple, support["seen_support_keys"]["context_time"])
    )
    assert sorted(builder_state["strain_raw_entity_keys"]) == sorted(
        support["seen_raw_keys"]["strain"]
    )
    assert sorted(builder_state["chemical_raw_entity_keys"]) == sorted(
        support["seen_raw_keys"]["chemical"]
    )
    source = pd.read_csv(tmp_path / "metadata.csv")
    expected_treatment = source["split_final"].eq("train") & source[
        "perturbation_no_concentration"
    ].ne("Water")
    assert sum(row["count"] for row in support["pair_counts"]) == int(
        expected_treatment.sum()
    )
    assert sum(row["count"] for row in support["context_time_counts"]) == sum(
        row["count"] for row in support["pair_counts"]
    )
    load_response_checkpoint(output / "checkpoint.pt", __import__("torch").device("cpu"), config)


def test_oof_fold_support_manifests_are_fit_local_and_hash_checked(tmp_path):
    config = load_response_config(_attach_authoritative_registries(_oof_files(tmp_path)))
    output = run_entity_oof(
        config,
        tmp_path / "support-oof",
        n_folds=2,
        seed=37,
        scenarios=("S1",),
    )
    manifests = []
    for fold in (0, 1):
        fold_dir = output / "folds" / f"S1_fold_{fold}"
        with (fold_dir / "completed.json").open("r", encoding="utf-8") as handle:
            completion = json.load(handle)
        with (fold_dir / "support_manifest.json").open("r", encoding="utf-8") as handle:
            support = json.load(handle)
        assert completion["support_manifest_sha256"] == manifest_sha256(support)
        assert completion["support_manifest_file_sha256"] == (
            fold_dir / "support_manifest.json.sha256"
        ).read_text(encoding="ascii").strip()
        assert support["fit_row_count"] == completion["train_count"]
        manifests.append(support)
    assert manifests[0]["seen_raw_keys"]["chemical"] != manifests[1]["seen_raw_keys"]["chemical"]


def test_checkpoint_rejects_support_vocabulary_tampering(tmp_path):
    config = load_response_config(_attach_authoritative_registries(_oof_files(tmp_path)))
    output = train_response_model(config, tmp_path / "tamper-run")
    checkpoint_path = output / "checkpoint.pt"
    checkpoint = __import__("torch").load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    checkpoint["feature_state"]["chemical_entity_keys"] = ["not-the-fit-vocabulary"]
    tampered = output / "tampered.pt"
    __import__("torch").save(checkpoint, tampered)
    with np.testing.assert_raises_regex(ValueError, "chemical expert vocabulary"):
        load_response_checkpoint(tampered, __import__("torch").device("cpu"))


def test_checkpoint_rejects_artifact_chain_tampering(tmp_path):
    config = load_response_config(_attach_authoritative_registries(_oof_files(tmp_path)))
    output = train_response_model(config, tmp_path / "artifact-tamper-run")
    checkpoint = __import__("torch").load(
        output / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    checkpoint["artifact_hashes"]["chemical_registry"]["sha256"] = "0" * 64
    tampered = output / "artifact-tampered.pt"
    __import__("torch").save(checkpoint, tampered)
    with np.testing.assert_raises_regex(ValueError, "artifact hash chain"):
        load_response_checkpoint(tampered, __import__("torch").device("cpu"), config)
