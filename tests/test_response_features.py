from __future__ import annotations

import numpy as np
import pandas as pd

from goai_response.features import ResponseFeatureBuilder


def _write_registry(path, entity_type, rows):
    key = "raw_name" if entity_type == "chemical" else "strain_code"
    records = []
    for value, status in rows:
        is_proxy = status == "proxy"
        records.append(
            {
                key: value,
                "canonical_id": f"test:{value}",
                "canonical_name": value,
                "mapping_status": status,
                "evidence_tier": (
                    "D_proxy_assumption"
                    if is_proxy
                    else "E_unresolved"
                    if status == "unresolved"
                    else "B_primary_candidate"
                ),
                "proxy_target": "test:parent" if is_proxy else "",
                "is_control": False,
                "is_quality_control": value == "QC",
            }
        )
    pd.DataFrame(records).to_csv(path, sep="\t", index=False)


def test_response_features_separate_unseen_observation_categories(tmp_path):
    metadata = pd.DataFrame({
        "Strains": ["A", "B", "A"], "perturbation_no_concentration": ["Water", "Drug", "Drug"],
        "Medium": ["M", "M", "M"], "Temperature": ["30", "30", "30"], "pert_time": [10, 10, 20],
        "data_source": ["S", "S", "NEW"], "instrument": ["I", "I", "NEW"], "Yeast_cell_plate": ["P", "P", "NEW"],
    }, index=["x", "y", "z"])
    builder = ResponseFeatureBuilder().fit(metadata, pd.Index(["x", "y"]))
    values = builder.transform(metadata)
    assert values.response.shape[0] == 3
    assert values.background.shape[0] == 3
    assert values.observation.shape[0] == 3
    assert values.is_treatment[:, 0].tolist() == [0.0, 1.0, 1.0]
    assert np.all(values.observation[2] == 0.0)


def test_unseen_chemical_bits_do_not_explode_after_train_scaling(tmp_path):
    mapping = tmp_path / "chemicals.tsv"
    pd.DataFrame({
        "raw_name": ["Water", "DrugA", "DrugB"], "status": ["resolved"] * 3,
        "is_control": [True, False, False], "isomeric_smiles": ["O", "CC", "c1ccccc1"],
    }).to_csv(mapping, sep="\t", index=False)
    metadata = pd.DataFrame({
        "Strains": ["A", "A", "A"], "perturbation_no_concentration": ["Water", "DrugA", "DrugB"],
        "Medium": ["M"] * 3, "Temperature": ["30"] * 3, "pert_time": [10] * 3,
        "data_source": ["S"] * 3, "instrument": ["I"] * 3, "Yeast_cell_plate": ["P"] * 3,
    }, index=["x", "y", "z"])
    builder = ResponseFeatureBuilder(chemical_map=mapping, chemical_bits=16).fit(metadata, pd.Index(["x", "y"]))
    values = builder.transform(metadata)
    assert np.isfinite(values.response).all()
    assert np.abs(values.response[2]).max() < 1_000.0


def test_plate_can_be_removed_or_corrupted_only_for_training_rows():
    metadata = pd.DataFrame({
        "Strains": ["A", "A", "A"], "perturbation_no_concentration": ["Water", "Drug", "Drug"],
        "Medium": ["M"] * 3, "Temperature": ["30"] * 3, "pert_time": [10] * 3,
        "data_source": ["S"] * 3, "instrument": ["I"] * 3, "Yeast_cell_plate": ["P1", "P2", "P3"],
    }, index=["train_a", "train_b", "valid"])
    no_plate = ResponseFeatureBuilder(calibration_use_plate=False).fit(metadata, pd.Index(["train_a", "train_b"]))
    assert "Yeast_cell_plate" not in no_plate.observation_slices
    shuffled = ResponseFeatureBuilder(
        calibration_plate_shuffle=True, calibration_shuffle_seed=7
    ).fit(metadata, pd.Index(["train_a", "train_b"]))
    assert set(shuffled.plate_training_assignments) == {"train_a", "train_b"}
    assert "valid" not in shuffled.plate_training_assignments


def test_response_prototype_is_leave_one_row_out_and_unseen_safe():
    metadata = pd.DataFrame({
        "Strains": ["A", "A", "B"],
        "perturbation_no_concentration": ["Drug1", "Drug1", "Drug2"],
    }, index=["a", "b", "c"])
    fc = np.asarray([[1.0, 10.0], [3.0, 30.0], [9.0, 90.0]], dtype=np.float32)
    mask = np.ones_like(fc)
    builder = ResponseFeatureBuilder().fit_response_priors(metadata, fc, mask, "chemical", alpha=1.0)
    train_prior = builder.response_prior(metadata, leave_one_out_fc=fc, leave_one_out_mask=mask)
    # Rows a and b see only the other Drug1 row; neither can copy its own target.
    assert not np.allclose(train_prior[0], train_prior[1])
    unseen = pd.DataFrame({"Strains": ["Z"], "perturbation_no_concentration": ["NeverSeen"]})
    assert np.allclose(builder.response_prior(unseen), 0.0)


def test_missing_strain_semantics_are_zero_after_resolved_only_scaling(tmp_path):
    strain_table = tmp_path / "strain.tsv"
    pd.DataFrame(
        {
            "strain_code": ["A", "B", "Missing"],
            "genome_axis": [10.0, 20.0, 0.0],
            "resolved": [1, 1, 0],
            "missing": [0, 0, 1],
            "proxy": [0, 0, 0],
        }
    ).to_csv(strain_table, sep="\t", index=False)
    chemical_registry = tmp_path / "chem_registry.tsv"
    strain_registry = tmp_path / "strain_registry.tsv"
    _write_registry(chemical_registry, "chemical", [("Drug", "high_confidence_candidate")])
    _write_registry(
        strain_registry,
        "strain",
        [
            ("A", "high_confidence_candidate"),
            ("B", "high_confidence_candidate"),
            ("Missing", "unresolved"),
        ],
    )
    metadata = pd.DataFrame(
        {
            "Strains": ["A", "B", "Missing", "NeverSeen"],
            "perturbation_no_concentration": ["Drug"] * 4,
            "Medium": ["M"] * 4,
            "Temperature": ["30"] * 4,
            "pert_time": [15] * 4,
            "data_source": ["S"] * 4,
            "pert_id": ["1"] * 4,
            "instrument": ["I"] * 4,
            "Yeast_cell_plate": ["P"] * 4,
        },
        index=["a", "b", "missing", "unknown"],
    )
    builder = ResponseFeatureBuilder(
        strain_features_path=strain_table,
        chemical_registry_path=chemical_registry,
        strain_registry_path=strain_registry,
    ).fit(metadata, pd.Index(["a", "b", "missing"]))
    assert np.allclose(builder.strain_mean, [15.0])
    assert np.allclose(builder.strain_scale, [5.0])
    values = builder.transform(metadata)
    # General cell is medium one-hot + temperature one-hot + four time values,
    # followed by one scaled genome coordinate and three unscaled status flags.
    strain_block = values.general_cell[:, -4:]
    assert np.allclose(strain_block[0], [-1.0, 1.0, 0.0, 0.0])
    assert np.allclose(strain_block[1], [1.0, 1.0, 0.0, 0.0])
    assert np.allclose(strain_block[2], [0.0, 0.0, 1.0, 0.0])
    assert np.allclose(strain_block[3], [0.0, 0.0, 1.0, 0.0])


def test_strain_scaler_uses_unique_fold_supported_entities(tmp_path):
    strain_table = tmp_path / "strain.tsv"
    pd.DataFrame(
        {
            "strain_code": ["A", "B"],
            "axis": [10.0, 20.0],
            "resolved": [1, 1],
            "missing": [0, 0],
            "proxy": [0, 0],
        }
    ).to_csv(strain_table, sep="\t", index=False)
    metadata = pd.DataFrame(
        {
            "Strains": ["A", "A", "A", "B"],
            "perturbation_no_concentration": ["Drug"] * 4,
            "Medium": ["M"] * 4,
            "Temperature": ["30"] * 4,
            "pert_time": [15] * 4,
            "data_source": ["S"] * 4,
            "instrument": ["I"] * 4,
            "Yeast_cell_plate": ["P"] * 4,
        }
    )
    builder = ResponseFeatureBuilder(strain_features_path=strain_table).fit(
        metadata, metadata.index
    )
    assert np.allclose(builder.strain_mean, [15.0])
    assert np.allclose(builder.strain_scale, [5.0])


def test_constant_fold_strain_feature_does_not_explode_for_heldout_value(tmp_path):
    strain_table = tmp_path / "strain.tsv"
    pd.DataFrame(
        {
            "strain_code": ["A", "B", "Heldout"],
            "constant_binary": [0.0, 0.0, 1.0],
            "large_count": [100000.0, 120000.0, 300000.0],
            "resolved": [1, 1, 1],
            "missing": [0, 0, 0],
            "proxy": [0, 0, 0],
        }
    ).to_csv(strain_table, sep="\t", index=False)
    metadata = pd.DataFrame(
        {
            "Strains": ["A", "B", "Heldout"],
            "perturbation_no_concentration": ["Drug"] * 3,
            "Medium": ["M"] * 3,
            "Temperature": ["30"] * 3,
            "pert_time": [15] * 3,
            "data_source": ["S"] * 3,
            "instrument": ["I"] * 3,
            "Yeast_cell_plate": ["P"] * 3,
        }
    )
    builder = ResponseFeatureBuilder(strain_features_path=strain_table).fit(
        metadata, metadata.index[:2]
    )
    block = builder._strain_block(metadata.loc[metadata.index[2:], "Strains"])
    assert np.isfinite(block).all()
    assert float(np.max(np.abs(block[:, :2]))) <= 5.0


def test_strain_feature_columns_select_low_dimensional_semantics(tmp_path):
    strain_table = tmp_path / "strain.tsv"
    pd.DataFrame(
        {
            "strain_code": ["A", "B", "Heldout"],
            "snp_mds_001": [1.0, 2.0, 3.0],
            "snp_mds_002": [4.0, 5.0, 6.0],
            "metadata_noise": [100.0, 200.0, 300.0],
            "resolved": [1, 1, 1],
            "missing": [0, 0, 0],
            "proxy": [0, 0, 0],
        }
    ).to_csv(strain_table, sep="\t", index=False)
    metadata = pd.DataFrame(
        {
            "Strains": ["A", "B", "Heldout"],
            "perturbation_no_concentration": ["Water", "Drug", "Drug"],
            "Medium": ["M"] * 3,
            "Temperature": ["30"] * 3,
            "pert_time": [10] * 3,
            "data_source": ["S"] * 3,
            "instrument": ["I"] * 3,
            "Yeast_cell_plate": ["P"] * 3,
        },
        index=["a", "b", "heldout"],
    )
    builder = ResponseFeatureBuilder(
        strain_features_path=strain_table,
        strain_feature_columns=("snp_mds_001", "snp_mds_002"),
    ).fit(metadata, pd.Index(["a", "b"]))
    assert builder.strain_semantic_columns == ["snp_mds_001", "snp_mds_002"]
    assert builder.strain_columns == [
        "snp_mds_001",
        "snp_mds_002",
        "resolved",
        "missing",
        "proxy",
    ]
    restored = ResponseFeatureBuilder.from_state_dict(builder.state_dict())
    assert restored.strain_feature_columns == (
        "snp_mds_001",
        "snp_mds_002",
    )
    assert restored.transform(metadata).general_cell.shape == (3, 11)


def test_strain_rbf_is_fold_fit_unseen_safe_and_serializable(tmp_path):
    strain_table = tmp_path / "strain.tsv"
    pd.DataFrame(
        {
            "strain_code": ["A", "B", "Heldout", "Missing", "Proxy"],
            "snp_mds_001": [0.0, 2.0, 1.0, 0.0, 1.5],
            "snp_mds_002": [0.0, 0.0, 0.5, 0.0, 0.0],
            "resolved": [1, 1, 1, 0, 0],
            "missing": [0, 0, 0, 1, 0],
            "proxy": [0, 0, 0, 0, 1],
        }
    ).to_csv(strain_table, sep="\t", index=False)
    metadata = pd.DataFrame(
        {
            "Strains": ["A", "B", "Heldout", "Missing", "Proxy"],
            "perturbation_no_concentration": ["Drug"] * 5,
            "Medium": ["M"] * 5,
            "Temperature": ["30"] * 5,
            "pert_time": [15] * 5,
            "data_source": ["S"] * 5,
            "instrument": ["I"] * 5,
            "Yeast_cell_plate": ["P"] * 5,
        },
        index=["a", "b", "heldout", "missing", "proxy"],
    )
    builder = ResponseFeatureBuilder(
        strain_features_path=strain_table,
        strain_feature_columns=("snp_mds_001", "snp_mds_002"),
        strain_feature_transform="rbf",
    ).fit(metadata, pd.Index(["a", "b"]))

    assert builder.strain_kernel_centers.shape == (2, 2)
    assert np.isclose(builder.strain_kernel_bandwidth, 2.0)
    assert builder.summary()["strain_semantic_dim"] == 2
    assert builder.summary()["strain_encoded_semantic_dim"] == 2
    assert builder.summary()["strain_kernel_center_count"] == 2
    block = builder._strain_block(metadata["Strains"])
    # The fold-held-out resolved strain receives similarities to the two
    # training-only centers; missing and proxy entities receive no semantics.
    assert np.isfinite(block).all()
    assert np.all(block[2, :2] > 0.0)
    assert np.allclose(block[3, :2], 0.0)
    assert np.allclose(block[4, :2], 0.0)

    restored = ResponseFeatureBuilder.from_state_dict(builder.state_dict())
    assert restored.strain_feature_transform == "rbf"
    assert np.allclose(restored.strain_kernel_centers, builder.strain_kernel_centers)
    assert np.isclose(restored.strain_kernel_bandwidth, builder.strain_kernel_bandwidth)
    assert np.allclose(restored._strain_block(metadata["Strains"]), block)


def test_strain_nearest_is_fold_fit_and_blocks_missing_semantics(tmp_path):
    strain_table = tmp_path / "strain.tsv"
    pd.DataFrame(
        {
            "strain_code": ["A", "B", "NearB", "Missing"],
            "snp_mds_001": [0.0, 10.0, 9.0, 0.0],
            "resolved": [1, 1, 1, 0],
            "missing": [0, 0, 0, 1],
            "proxy": [0, 0, 0, 0],
        }
    ).to_csv(strain_table, sep="\t", index=False)
    metadata = pd.DataFrame(
        {
            "Strains": ["A", "B", "NearB", "Missing"],
            "perturbation_no_concentration": ["Drug"] * 4,
            "Medium": ["M"] * 4,
            "Temperature": ["30"] * 4,
            "pert_time": [15] * 4,
            "data_source": ["S"] * 4,
            "instrument": ["I"] * 4,
            "Yeast_cell_plate": ["P"] * 4,
        },
        index=["a", "b", "near_b", "missing"],
    )
    builder = ResponseFeatureBuilder(
        strain_features_path=strain_table,
        strain_feature_columns=("snp_mds_001",),
        strain_feature_transform="nearest",
    ).fit(metadata, pd.Index(["a", "b"]))
    block = builder._strain_block(metadata["Strains"])
    assert np.allclose(block[0, :2], [1.0, 0.0])
    assert np.allclose(block[1, :2], [0.0, 1.0])
    assert np.allclose(block[2, :2], [0.0, 1.0])
    assert np.allclose(block[3, :2], [0.0, 0.0])
    assert builder.summary()["strain_feature_transform"] == "nearest"
    assert builder.summary()["strain_encoded_semantic_dim"] == 2
    restored = ResponseFeatureBuilder.from_state_dict(builder.state_dict())
    assert np.allclose(restored._strain_block(metadata["Strains"]), block)


def test_chemical_registry_blocks_proxy_and_unresolved_vectors_by_default(tmp_path):
    chemical_map = tmp_path / "chemical_map.tsv"
    pd.DataFrame(
        {
            "raw_name": ["ExactA", "ExactB", "Proxy", "QC"],
            "status": ["resolved", "resolved", "resolved", "unresolved"],
            "is_control": [False, False, False, True],
            "isomeric_smiles": ["CC", "c1ccccc1", "CCC", ""],
        }
    ).to_csv(chemical_map, sep="\t", index=False)
    chemical_registry = tmp_path / "chem_registry.tsv"
    strain_registry = tmp_path / "strain_registry.tsv"
    _write_registry(
        chemical_registry,
        "chemical",
        [
            ("ExactA", "high_confidence_candidate"),
            ("ExactB", "high_confidence_candidate"),
            ("Proxy", "proxy"),
            ("QC", "unresolved"),
        ],
    )
    _write_registry(strain_registry, "strain", [("A", "high_confidence_candidate")])
    metadata = pd.DataFrame(
        {
            "Strains": ["A"] * 4,
            "perturbation_no_concentration": ["ExactA", "ExactB", "Proxy", "QC"],
            "Medium": ["M"] * 4,
            "Temperature": ["30"] * 4,
            "pert_time": [15] * 4,
            "data_source": ["S"] * 4,
            "pert_id": ["1", "2", "3", "4"],
            "instrument": ["I"] * 4,
            "Yeast_cell_plate": ["P"] * 4,
        },
        index=["a", "b", "proxy", "qc"],
    )
    builder = ResponseFeatureBuilder(
        chemical_map=chemical_map,
        chemical_bits=16,
        chemical_registry_path=chemical_registry,
        strain_registry_path=strain_registry,
    ).fit(metadata, metadata.index)
    block = builder.transform(metadata).general_perturbation
    assert np.any(np.abs(block[0, :-3]) > 0)
    assert np.allclose(block[2, :-3], 0.0)
    assert np.allclose(block[3, :-3], 0.0)
    assert np.allclose(block[2, -3:], [0.0, 1.0, 1.0])
    assert np.allclose(block[3, -3:], [0.0, 1.0, 0.0])

    parent = ResponseFeatureBuilder(
        chemical_map=chemical_map,
        chemical_bits=16,
        chemical_registry_path=chemical_registry,
        strain_registry_path=strain_registry,
        chemical_parent_views_path=(tmp_path / "parent_views.tsv"),
        chemical_structure_view="parent",
        allow_proxy_semantics=True,
    )
    pd.DataFrame(
        {
            "raw_name": ["Proxy"],
            "parent_canonical_id": ["test:parent"],
            "parent_name": ["Parent"],
            "parent_inchikey": [""],
            "parent_isomeric_smiles": ["CCC"],
            "view_status": ["parent_normalized_ablation"],
        }
    ).to_csv(parent.chemical_parent_views_path, sep="\t", index=False)
    parent = parent.fit(metadata, metadata.index)
    parent_block = parent.transform(metadata).general_perturbation
    # Proxy is the only enabled parent view in this tiny fit, so train-fitted
    # centering makes its continuous representation zero; the explicit proxy
    # flag still proves the reviewed parent route was enabled.
    assert np.isfinite(parent_block[2, :-3]).all()
    assert np.allclose(parent_block[2, -3:], [0.0, 0.0, 1.0])


def test_production_exact_doxycycline_and_proxy_oligomycin_gate():
    project = __import__("pathlib").Path(__file__).resolve().parents[1]
    chemical_map = project / "data/processed/chemical_entity_map.tsv"
    chemical_registry = project / "data/processed/entities/chemical_registry.tsv"
    strain_registry = project / "data/processed/entities/strain_registry.tsv"
    metadata = pd.DataFrame(
        {
            "Strains": ["BAH", "BAH", "BAH"],
            "perturbation_no_concentration": [
                "Doxycycline hyclate",
                "Abietic acid",
                "Oligomycin",
            ],
            "Medium": ["M"] * 3,
            "Temperature": ["30"] * 3,
            "pert_time": [15] * 3,
            "data_source": ["S"] * 3,
            "pert_id": ["1", "2", "3"],
            "instrument": ["I"] * 3,
            "Yeast_cell_plate": ["P"] * 3,
        },
        index=["doxy", "exact", "proxy"],
    )
    builder = ResponseFeatureBuilder(
        chemical_map=chemical_map,
        chemical_bits=32,
        chemical_registry_path=chemical_registry,
        strain_registry_path=strain_registry,
    ).fit(metadata, metadata.index)
    block = builder.transform(metadata).general_perturbation
    assert np.allclose(block[0, -3:], [1.0, 0.0, 0.0])
    assert np.any(np.abs(block[0, :-3]) > 0)
    assert np.allclose(block[2, :-3], 0.0)
    assert np.allclose(block[2, -3:], [0.0, 1.0, 1.0])


def test_exact_structure_view_rejects_parent_identity_drift(tmp_path):
    chemical_map = tmp_path / "chemical_map.tsv"
    pd.DataFrame(
        {
            "raw_name": ["Salt"],
            "status": ["resolved"],
            "is_control": [False],
            "cid": ["parent-cid"],
            "inchikey": ["PARENT-KEY"],
            "isomeric_smiles": ["CC"],
        }
    ).to_csv(chemical_map, sep="\t", index=False)
    chemical_registry = tmp_path / "chem_registry.tsv"
    strain_registry = tmp_path / "strain_registry.tsv"
    pd.DataFrame(
        [
            {
                "raw_name": "Salt",
                "canonical_id": "test:exact-salt",
                "canonical_name": "Salt",
                "mapping_status": "high_confidence_candidate",
                "evidence_tier": "B_primary_candidate",
                "proxy_target": "",
                "is_control": False,
                "is_quality_control": False,
                "pubchem_cid": "exact-cid",
                "inchikey": "EXACT-KEY",
            }
        ]
    ).to_csv(chemical_registry, sep="\t", index=False)
    _write_registry(strain_registry, "strain", [("A", "high_confidence_candidate")])
    metadata = pd.DataFrame(
        {
            "Strains": ["A"],
            "perturbation_no_concentration": ["Salt"],
            "Medium": ["M"],
            "Temperature": ["30"],
            "pert_time": [15],
            "data_source": ["S"],
            "pert_id": ["1"],
            "instrument": ["I"],
            "Yeast_cell_plate": ["P"],
        },
        index=["salt"],
    )
    with np.testing.assert_raises_regex(ValueError, "disagrees with registry"):
        ResponseFeatureBuilder(
            chemical_map=chemical_map,
            chemical_registry_path=chemical_registry,
            strain_registry_path=strain_registry,
        ).fit(metadata, metadata.index)


def test_reviewed_parent_view_accepts_doxy_parent_and_marks_proxy():
    project = __import__("pathlib").Path(__file__).resolve().parents[1]
    metadata = pd.DataFrame(
        {
            "Strains": ["BAH", "BAH"],
            "perturbation_no_concentration": ["Doxycycline hyclate", "Abietic acid"],
            "Medium": ["M", "M"],
            "Temperature": ["30", "30"],
            "pert_time": [15, 15],
            "data_source": ["S", "S"],
            "pert_id": ["1", "2"],
            "instrument": ["I", "I"],
            "Yeast_cell_plate": ["P", "P"],
        },
        index=["doxy", "ordinary-exact"],
    )
    builder = ResponseFeatureBuilder(
        chemical_map=project / "data/processed/chemical_views/chemical_entity_map_parent_normalized.tsv",
        chemical_bits=32,
        chemical_registry_path=project / "data/processed/entities/chemical_registry.tsv",
        strain_registry_path=project / "data/processed/entities/strain_registry.tsv",
        chemical_parent_views_path=project / "data/processed/entities/chemical_parent_normalized_views.tsv",
        chemical_structure_view="parent",
        allow_proxy_semantics=True,
    ).fit(metadata, metadata.index)
    block = builder.transform(metadata).general_perturbation
    assert np.any(np.abs(block[0, :-3]) > 0)
    assert np.any(np.abs(block[1, :-3]) > 0)
    assert np.allclose(block[0, -3:], [0.0, 0.0, 1.0])
    assert np.allclose(block[1, -3:], [1.0, 0.0, 0.0])


def test_parent_view_keeps_unreviewed_proxy_zero(tmp_path):
    project = __import__("pathlib").Path(__file__).resolve().parents[1]
    parent_contract = pd.read_csv(
        project / "data/processed/entities/chemical_parent_normalized_views.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    # Deliberately omit Oligomycin: a proxy row in the identity registry is
    # not enough to opt it into this particular reviewed parent experiment.
    parent_contract = parent_contract.loc[parent_contract["raw_name"].ne("Oligomycin")]
    contract_path = tmp_path / "partial_parent_views.tsv"
    parent_contract.to_csv(contract_path, sep="\t", index=False)
    metadata = pd.DataFrame(
        {
            "Strains": ["BAH", "BAH"],
            "perturbation_no_concentration": ["Doxycycline hyclate", "Oligomycin"],
            "Medium": ["M", "M"],
            "Temperature": ["30", "30"],
            "pert_time": [15, 15],
            "data_source": ["S", "S"],
            "pert_id": ["1", "2"],
            "instrument": ["I", "I"],
            "Yeast_cell_plate": ["P", "P"],
        },
        index=["reviewed", "not-reviewed"],
    )
    builder = ResponseFeatureBuilder(
        chemical_map=project / "data/processed/chemical_views/chemical_entity_map_parent_normalized.tsv",
        chemical_bits=32,
        chemical_registry_path=project / "data/processed/entities/chemical_registry.tsv",
        strain_registry_path=project / "data/processed/entities/strain_registry.tsv",
        chemical_parent_views_path=contract_path,
        chemical_structure_view="parent",
        allow_proxy_semantics=True,
    ).fit(metadata, metadata.index)
    block = builder.transform(metadata).general_perturbation
    assert np.allclose(block[1, :-3], 0.0)
    assert np.allclose(block[1, -3:], [0.0, 1.0, 1.0])


def test_zero_structure_view_preserves_identity_flags_without_a_vector():
    project = __import__("pathlib").Path(__file__).resolve().parents[1]
    metadata = pd.DataFrame(
        {
            "Strains": ["BAH", "BAH"],
            "perturbation_no_concentration": ["Abietic acid", "Oligomycin"],
            "Medium": ["M", "M"],
            "Temperature": ["30", "30"],
            "pert_time": [15, 15],
            "data_source": ["S", "S"],
            "pert_id": ["1", "2"],
            "instrument": ["I", "I"],
            "Yeast_cell_plate": ["P", "P"],
        },
        index=["exact", "proxy"],
    )
    builder = ResponseFeatureBuilder(
        chemical_map=project / "data/processed/chemical_views/chemical_entity_map_zero_risky.tsv",
        chemical_registry_path=project / "data/processed/entities/chemical_registry.tsv",
        strain_registry_path=project / "data/processed/entities/strain_registry.tsv",
        chemical_structure_view="zero",
    ).fit(metadata, metadata.index)
    block = builder.transform(metadata).general_perturbation
    assert block.shape == (2, 3)
    assert np.allclose(block[0], [1.0, 0.0, 0.0])
    assert np.allclose(block[1], [0.0, 1.0, 1.0])


def test_zero_risky_view_only_zeros_seven_reviewed_identity_risk_entities():
    project = __import__("pathlib").Path(__file__).resolve().parents[1]
    risk_table = pd.read_csv(
        project / "resources/entities/chemical_identity_risk_review.tsv",
        sep="\t",
    )
    risky = risk_table.loc[
        risk_table["zero_risky"].astype(str).str.casefold().eq("true"), "raw_name"
    ].astype(str).tolist()
    assert len(risky) == 7
    chemicals = ["Abietic acid", "Artemisinin", *risky]
    metadata = pd.DataFrame(
        {
            "Strains": ["BAH"] * len(chemicals),
            "perturbation_no_concentration": chemicals,
            "Medium": ["M"] * len(chemicals),
            "Temperature": ["30"] * len(chemicals),
            "pert_time": [15] * len(chemicals),
            "data_source": ["S"] * len(chemicals),
            "pert_id": [str(index) for index in range(len(chemicals))],
            "instrument": ["I"] * len(chemicals),
            "Yeast_cell_plate": ["P"] * len(chemicals),
        },
        index=[f"row-{index}" for index in range(len(chemicals))],
    )
    builder = ResponseFeatureBuilder(
        chemical_map=project / "data/processed/chemical_views/chemical_entity_map_zero_risky.tsv",
        chemical_bits=32,
        chemical_registry_path=project / "data/processed/entities/chemical_registry.tsv",
        strain_registry_path=project / "data/processed/entities/strain_registry.tsv",
        chemical_identity_risks_path=project / "resources/entities/chemical_identity_risk_review.tsv",
        chemical_structure_view="zero_risky",
    ).fit(metadata, metadata.index)
    block = builder.transform(metadata).general_perturbation
    assert np.any(np.abs(block[0, :-3]) > 0)
    assert np.any(np.abs(block[1, :-3]) > 0)
    assert np.allclose(block[2:, :-3], 0.0)
    # Four candidate exact identities remain resolved even though the
    # row-level identity-risk ablation suppresses their continuous vectors.
    for name in {
        "Doxycycline hyclate",
        "G418",
        "Hygromycin B",
        "LY 294002 hydrochloride",
    }:
        assert np.allclose(block[chemicals.index(name), -3:], [1.0, 0.0, 0.0])
    for name in set(risky) - {
        "Doxycycline hyclate",
        "G418",
        "Hygromycin B",
        "LY 294002 hydrochloride",
    }:
        assert np.allclose(block[chemicals.index(name), -3:], [0.0, 1.0, 1.0])
