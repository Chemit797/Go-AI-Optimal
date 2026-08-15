from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from goai_response.config import load_response_config
from goai_response.oof import (
    REGIME_SCENARIOS,
    FoldSlice,
    assert_fold_isolation,
    make_fold_slices,
    run_entity_oof,
)

from .conftest import METADATA_COLUMNS, metadata_row
from .test_entity_oof import _oof_files
from goai_response.entities import registry_from_frame


def _factorial_metadata() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strain in ("S1", "S2", "S3", "S4"):
        for time in (15, 30):
            rows.append(metadata_row(f"ctrl_{strain}_{time}", "Water", "train", strain=strain, time=time))
            for chemical in ("DrugA", "DrugB", "DrugC", "DrugD"):
                rows.append(
                    metadata_row(
                        f"tx_{strain}_{chemical}_{time}",
                        chemical,
                        "train",
                        strain=strain,
                        time=time,
                    )
                )
    metadata = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    return metadata.set_index("sample_ID", verify_integrity=True)


def test_support_regimes_have_expected_fold_membership_without_leakage():
    metadata = _factorial_metadata()
    train_ids = metadata.index
    slices, assignments = make_fold_slices(
        metadata,
        train_ids,
        n_folds=2,
        seed=71,
        scenarios=REGIME_SCENARIOS,
    )

    assert set(assignments["scenario"]) == set(REGIME_SCENARIOS)
    assert not assignments.duplicated(["sample_ID", "scenario"]).any()
    for fold in slices:
        assert_fold_isolation(metadata, fold)

    expected = {
        "R00": (False, False, False, False),
        "R10": (True, False, False, False),
        "R01": (False, True, False, False),
        "R11": (True, True, False, False),
        "RT": (True, True, True, False),
    }
    for regime, flags in expected.items():
        eligible = assignments.loc[
            assignments["scenario"].eq(regime) & assignments["eligible"]
        ]
        assert len(eligible) == 32
        observed = set(
            eligible[
                [
                    "strain_seen_in_fold",
                    "chemical_seen_in_fold",
                    "pair_seen_in_fold",
                    "time_group_seen_in_fold",
                ]
            ].itertuples(index=False, name=None)
        )
        assert observed == {flags}


def test_each_entity_regime_scores_every_treatment_once():
    metadata = _factorial_metadata()
    slices, assignments = make_fold_slices(
        metadata,
        metadata.index,
        n_folds=2,
        seed=83,
        scenarios=("R00", "R10", "R01", "R11"),
    )
    treatment_ids = {index for index in metadata.index if index.startswith("tx_")}
    for regime in ("R00", "R10", "R01", "R11"):
        eligible = assignments.loc[
            assignments["scenario"].eq(regime) & assignments["eligible"],
            "sample_ID",
        ]
        assert set(eligible) == treatment_ids
        assert eligible.is_unique
    assert all(len(fold.validation_ids) > 0 for fold in slices)


def test_single_axis_regimes_do_not_repeat_identical_fits():
    metadata = _factorial_metadata()
    n_folds = 2
    slices, assignments = make_fold_slices(
        metadata,
        metadata.index,
        n_folds=n_folds,
        seed=89,
        scenarios=("R00", "R10", "R01", "R11", "RT"),
    )
    by_scenario = {
        scenario: [fold for fold in slices if fold.scenario == scenario]
        for scenario in ("R00", "R10", "R01", "R11", "RT")
    }
    assert len(by_scenario["R00"]) == n_folds ** 2
    assert len(by_scenario["R10"]) == n_folds
    assert len(by_scenario["R01"]) == n_folds
    assert len(by_scenario["R11"]) == n_folds ** 2
    assert len(by_scenario["RT"]) == n_folds

    for scenario in ("R10", "R01"):
        train_signatures = {
            tuple(fold.train_ids.astype(str)) for fold in by_scenario[scenario]
        }
        assert len(train_signatures) == n_folds
        eligible = assignments.loc[
            assignments["scenario"].eq(scenario) & assignments["eligible"]
        ]
        treatment_ids = {
            sample_id for sample_id in metadata.index if sample_id.startswith("tx_")
        }
        assert set(eligible["sample_ID"]) == treatment_ids
        assert eligible["sample_ID"].is_unique
        for fold in by_scenario[scenario]:
            assert_fold_isolation(metadata, fold)
            if scenario == "R10":
                assert not fold.heldout_strains
            else:
                assert not fold.heldout_chemicals


def test_regime_audit_writes_versioned_protocol_and_support_flags(tmp_path):
    config = load_response_config(_oof_files(tmp_path))
    output = run_entity_oof(
        config,
        tmp_path / "regime-audit",
        n_folds=2,
        seed=97,
        scenarios=REGIME_SCENARIOS,
        audit_only=True,
    )
    assignments = pd.read_csv(output / "fold_assignments.csv")
    assert {
        "strain_seen_in_fold",
        "chemical_seen_in_fold",
        "pair_seen_in_fold",
        "time_group_seen_in_fold",
    } <= set(assignments.columns)
    with (output / "oof_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["protocol"] == "support_regime_oof_v2"
    assert set(manifest["support_regime_definition"]) == set(REGIME_SCENARIOS)


def test_isolation_uses_the_same_normalized_raw_key_as_expert_vocab():
    metadata = pd.DataFrame(
        [
            metadata_row("fit", "Drug A", "train", strain="S 1"),
            metadata_row("valid", "  drug   a  ", "train", strain="s 1"),
        ],
        columns=METADATA_COLUMNS,
    ).set_index("sample_ID")
    fold = FoldSlice(
        scenario="R00",
        fold=0,
        train_ids=pd.Index(["fit"]),
        validation_ids=pd.Index(["valid"]),
    )
    with pytest.raises(AssertionError, match="R00"):
        assert_fold_isolation(metadata, fold)


def test_registry_aliases_share_one_canonical_oof_fold_and_cannot_leak(tmp_path):
    metadata = _factorial_metadata().copy()
    alias_rows = metadata.loc[metadata["perturbation_no_concentration"].eq("DrugA")].copy()
    alias_rows.index = pd.Index([f"alias_{value}" for value in alias_rows.index])
    alias_rows["perturbation_no_concentration"] = "Drug A reviewed alias"
    alias_rows["pert_id"] = [f"#alias-{index}" for index in range(len(alias_rows))]
    metadata = pd.concat([metadata, alias_rows])

    chemical_names = sorted(metadata["perturbation_no_concentration"].unique())
    chemical_rows = []
    for name in chemical_names:
        canonical = "chemical:DrugA" if name in {"DrugA", "Drug A reviewed alias"} else f"chemical:{name}"
        chemical_rows.append(
            {
                "raw_name": name,
                "canonical_id": canonical,
                "canonical_name": name,
                "mapping_status": "verified",
                "evidence_tier": "A_verified",
                "proxy_target": "",
                "is_control": name == "Water",
                "is_quality_control": False,
            }
        )
    strain_rows = [
        {
            "strain_code": name,
            "canonical_id": f"strain:{name}",
            "canonical_name": name,
            "mapping_status": "verified",
            "evidence_tier": "A_verified",
            "proxy_target": "",
            "is_control": False,
            "is_quality_control": False,
        }
        for name in sorted(metadata["Strains"].unique())
    ]
    chemical = tmp_path / "chemical.tsv"
    strain = tmp_path / "strain.tsv"
    registry_from_frame(pd.DataFrame(chemical_rows), "chemical").table.to_csv(
        chemical, sep="\t", index=False
    )
    registry_from_frame(pd.DataFrame(strain_rows), "strain").table.to_csv(
        strain, sep="\t", index=False
    )
    config_path = _oof_files(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["entity"]["chemical_registry"] = str(chemical)
    payload["entity"]["strain_registry"] = str(strain)
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_response_config(config_path)

    slices, assignments = make_fold_slices(
        metadata,
        metadata.index,
        n_folds=2,
        seed=101,
        scenarios=("R10",),
        config=config,
    )
    drug_a = assignments.loc[
        assignments["perturbation_no_concentration"].isin(
            ["DrugA", "Drug A reviewed alias"]
        )
    ]
    assert drug_a["chemical_support_key"].unique().tolist() == ["chemical:DrugA"]
    assert drug_a["fold"].nunique() == 1
    for fold in slices:
        assert_fold_isolation(metadata, fold, config)
