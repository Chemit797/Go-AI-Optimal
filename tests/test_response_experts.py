from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import yaml

from goai_baseline.preprocess import prepare_data
from goai_response.config import load_response_config
from goai_response.features import ResponseFeatureBuilder
from goai_response.model import ResponseDecompositionRegressor
from goai_response.predict import (
    COMPONENT_NAMES,
    _natural_scale_components,
    predict_response_components,
)
from goai_response.train import _predict, fit_response_model, train_response_model

from .conftest import write_config


def _metadata(strains: list[str], chemicals: list[str]) -> pd.DataFrame:
    size = len(strains)
    return pd.DataFrame(
        {
            "Strains": strains,
            "perturbation_no_concentration": chemicals,
            "Medium": ["M"] * size,
            "Temperature": ["30"] * size,
            "pert_time": [15] * size,
            "data_source": ["S"] * size,
            "instrument": ["I"] * size,
            "Yeast_cell_plate": ["P"] * size,
        },
        index=[f"row_{index}" for index in range(size)],
    )


def test_entity_indices_are_fold_fitted_canonical_and_unknown_safe():
    train = _metadata([" Strain A ", "Strain B"], ["Drug A", "Ｄｒｕｇ B"])
    builder = ResponseFeatureBuilder().fit(train, train.index)
    query = _metadata(
        ["strain   a", "STRAIN B", "never seen"],
        ["DRUG A", "Drug B", "new drug"],
    )
    features = builder.transform(query)

    assert features.strain_indices.tolist() == [1, 2, 0]
    assert features.chemical_indices.tolist() == [1, 2, 0]
    assert features.strain_seen[:, 0].tolist() == [1.0, 1.0, 0.0]
    assert features.chemical_seen[:, 0].tolist() == [1.0, 1.0, 0.0]
    assert features.pair_seen[:, 0].tolist() == [1.0, 1.0, 0.0]
    # With no external strain representation, ID changes cannot leak into the
    # universal cell tensor.  They are available only to the gated expert.
    assert np.array_equal(features.general_cell[0], features.general_cell[1])
    assert features.general_perturbation.shape == (3, 0)


def test_registry_aliases_share_expert_indices_and_proxy_parent_does_not(tmp_path):
    chemical_registry = tmp_path / "chemical_registry.tsv"
    strain_registry = tmp_path / "strain_registry.tsv"
    pd.DataFrame(
        [
            {
                "raw_name": "Drug alias A",
                "canonical_id": "pubchem:123",
                "canonical_name": "Drug",
                "mapping_status": "verified",
                "evidence_tier": "A_verified",
                "proxy_target": "",
                "is_control": False,
                "is_quality_control": False,
            },
            {
                "raw_name": "Drug alias B",
                "canonical_id": "pubchem:123",
                "canonical_name": "Drug",
                "mapping_status": "high_confidence_candidate",
                "evidence_tier": "B_primary_candidate",
                "proxy_target": "",
                "is_control": False,
                "is_quality_control": False,
            },
            {
                "raw_name": "Drug proxy",
                "canonical_id": "goai-chemical:drug-proxy",
                "canonical_name": "Drug proxy",
                "mapping_status": "proxy",
                "evidence_tier": "D_proxy_assumption",
                "proxy_target": "pubchem:123",
                "is_control": False,
                "is_quality_control": False,
            },
        ]
    ).to_csv(chemical_registry, sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "strain_code": alias,
                "canonical_id": "goai-strain:DHY210",
                "canonical_name": "DHY210",
                "mapping_status": "unresolved",
                "evidence_tier": "E_unresolved",
                "proxy_target": "",
                "is_control": False,
                "is_quality_control": False,
            }
            for alias in ("DHY210 alias A", "DHY210 alias B")
        ]
    ).to_csv(strain_registry, sep="\t", index=False)
    all_metadata = _metadata(
        ["DHY210 alias A", "DHY210 alias B", "DHY210 alias B"],
        ["Drug alias A", "Drug alias B", "Drug proxy"],
    )
    builder = ResponseFeatureBuilder(
        chemical_registry_path=chemical_registry,
        strain_registry_path=strain_registry,
    ).fit(all_metadata, pd.Index(["row_0"]))

    features = builder.transform(all_metadata)
    assert features.strain_indices.tolist() == [1, 1, 1]
    assert features.chemical_indices.tolist() == [1, 1, 0]
    assert features.strain_seen[:, 0].tolist() == [1.0, 1.0, 1.0]
    assert features.chemical_seen[:, 0].tolist() == [1.0, 1.0, 0.0]
    assert features.pair_indices.tolist() == [1, 1, 0]
    assert features.pair_seen[:, 0].tolist() == [1.0, 1.0, 0.0]
    assert builder.chemical_entity_keys == ["pubchem:123"]
    assert builder.strain_entity_keys == ["goai-strain:DHY210"]
    assert builder.chemical_raw_entity_keys == ["drug alias a"]


def _expert_model() -> ResponseDecompositionRegressor:
    model = ResponseDecompositionRegressor(
        response_input_dim=4,
        background_input_dim=3,
        observation_input_dim=2,
        n_proteins=3,
        hidden_dim=2,
        response_rank=2,
        calibration_rank=2,
        dropout=0.0,
        calibration_enabled=False,
        interaction_mode="shared_general_experts",
        general_cell_input_dim=2,
        general_perturbation_input_dim=1,
        n_strain_entities=2,
        n_chemical_entities=2,
        n_pair_entities=2,
        response_pair_expert_enabled=True,
        entity_dropout=0.0,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.response_proteins.copy_(
            torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        )
        model.background_strain_embeddings.weight[1].copy_(torch.tensor([2.0, 3.0]))
        model.background_strain_decoder.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        )
        model.response_strain_embeddings.weight[1].copy_(torch.tensor([1.0, 0.0]))
        model.response_chemical_embeddings.weight[1].copy_(torch.tensor([0.0, 2.0]))
        # Pair (strain index 1, chemical index 1) maps to padding-safe index 1.
        model.response_pair_embeddings.weight[1].copy_(torch.tensor([3.0, 4.0]))
    model.eval()
    return model


def test_m7_hard_gates_implement_universal_plus_entity_experts_exactly():
    model = _expert_model()
    batch = 4
    result = model.forward_named_components(
        torch.zeros(batch, 4),
        torch.zeros(batch, 3),
        torch.zeros(batch, 2),
        torch.ones(batch, 1),
        general_cell_inputs=torch.zeros(batch, 2),
        general_perturbation_inputs=torch.zeros(batch, 1),
        strain_indices=torch.tensor([0, 1, 0, 1]),
        chemical_indices=torch.tensor([0, 0, 1, 1]),
        strain_seen=torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
        chemical_seen=torch.tensor([[0.0], [0.0], [1.0], [1.0]]),
        pair_indices=torch.tensor([0, 0, 0, 1]),
        pair_seen=torch.tensor([[0.0], [0.0], [0.0], [1.0]]),
    )

    strain_response = torch.tensor([1.0, 0.0, 1.0])
    chemical_response = torch.tensor([0.0, 2.0, 2.0])
    pair_response = torch.tensor([3.0, 4.0, 7.0])
    assert torch.allclose(result.response[0], torch.zeros(3))
    assert torch.allclose(result.response[1], strain_response)
    assert torch.allclose(result.response[2], chemical_response)
    assert torch.allclose(
        result.response[3], strain_response + chemical_response + pair_response
    )
    assert torch.allclose(result.background_strain[1], torch.tensor([2.0, 3.0, 5.0]))
    assert torch.allclose(result.background_strain[2], torch.zeros(3))
    assert torch.allclose(
        result.absolute,
        result.background_universal
        + result.background_strain
        + result.response_universal
        + result.response_strain
        + result.response_chemical
        + result.response_pair,
    )

    scaled = _natural_scale_components(
        result,
        target_mean=np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
        target_scale=np.asarray([2.0, 3.0, 4.0], dtype=np.float32),
    )
    assert tuple(scaled) == COMPONENT_NAMES
    reconstructed = (
        scaled["B_U"]
        + scaled["B_s"]
        + scaled["C_obs"]
        + scaled["R_U"]
        + scaled["R_s"]
        + scaled["R_c"]
        + scaled["R_sc"]
    )
    assert np.allclose(reconstructed, scaled["final"], atol=1e-6)


def test_m7_gate_not_index_controls_expert_eligibility():
    model = _expert_model()
    result = model.forward_named_components(
        torch.zeros(1, 4),
        torch.zeros(1, 3),
        torch.zeros(1, 2),
        torch.ones(1, 1),
        general_cell_inputs=torch.zeros(1, 2),
        general_perturbation_inputs=torch.zeros(1, 1),
        strain_indices=torch.tensor([1]),
        chemical_indices=torch.tensor([1]),
        strain_seen=torch.zeros(1, 1),
        chemical_seen=torch.zeros(1, 1),
        pair_indices=torch.tensor([1]),
        pair_seen=torch.zeros(1, 1),
    )
    assert torch.allclose(result.background_strain, torch.zeros(1, 3))
    assert torch.allclose(result.response_strain, torch.zeros(1, 3))
    assert torch.allclose(result.response_chemical, torch.zeros(1, 3))
    assert torch.allclose(result.response_pair, torch.zeros(1, 3))


def test_pair_expert_requires_explicit_fold_fit_pair_gate():
    model = _expert_model()
    result = model.forward_named_components(
        torch.zeros(1, 4),
        torch.zeros(1, 3),
        torch.zeros(1, 2),
        torch.ones(1, 1),
        general_cell_inputs=torch.zeros(1, 2),
        general_perturbation_inputs=torch.zeros(1, 1),
        strain_indices=torch.tensor([1]),
        chemical_indices=torch.tensor([1]),
        strain_seen=torch.ones(1, 1),
        chemical_seen=torch.ones(1, 1),
        pair_indices=torch.tensor([0]),
        pair_seen=torch.zeros(1, 1),
    )
    assert torch.allclose(result.response_pair, torch.zeros(1, 3))
    assert result.pair_gate.item() == 0.0


def test_pair_expert_is_context_modulated_but_unknown_pair_stays_zero():
    model = _expert_model()
    with torch.no_grad():
        # Make the shared cell trunk expose a different first latent coordinate
        # for two otherwise identical pair rows.
        model.general_cell_encoder[0].weight.copy_(torch.eye(2))
        model.general_cell_encoder[0].bias.zero_()
        model.response_pair_context.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        )
    result = model.forward_named_components(
        torch.zeros(3, 4),
        torch.zeros(3, 3),
        torch.zeros(3, 2),
        torch.ones(3, 1),
        general_cell_inputs=torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 0.0]]),
        general_perturbation_inputs=torch.zeros(3, 1),
        strain_indices=torch.tensor([1, 1, 1]),
        chemical_indices=torch.tensor([1, 1, 1]),
        strain_seen=torch.ones(3, 1),
        chemical_seen=torch.ones(3, 1),
        pair_indices=torch.tensor([1, 1, 0]),
        pair_seen=torch.tensor([[1.0], [1.0], [0.0]]),
    )
    assert not torch.allclose(result.response_pair[0], result.response_pair[1])
    assert torch.allclose(result.response_pair[2], torch.zeros(3))


def test_balanced_entity_dropout_covers_four_regimes_nearly_equally():
    torch.manual_seed(41)
    strain, chemical = ResponseDecompositionRegressor.balanced_entity_regime_gates(
        torch.ones(19, 1), torch.ones(19, 1)
    )
    regimes = 2 * strain[:, 0].long() + chemical[:, 0].long()
    counts = torch.bincount(regimes, minlength=4)
    assert int(counts.max() - counts.min()) <= 1
    assert set(regimes.tolist()) == {0, 1, 2, 3}


def test_training_stages_freeze_only_target_components():
    model = _expert_model()
    expected_prefixes = {
        "strain": ("background_strain_", "response_strain_"),
        "chemical": ("response_chemical_",),
        "pair": ("response_pair_",),
    }
    for stage, prefixes in expected_prefixes.items():
        model.set_training_stage(stage)
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        assert trainable
        assert all(name.startswith(prefixes) for name in trainable)
    model.set_training_stage("universal")
    assert all(
        not name.startswith(("background_strain_", "response_strain_", "response_chemical_", "response_pair_"))
        for name, value in model.named_parameters()
        if value.requires_grad
    )
    model.set_training_stage("joint")
    assert all(value.requires_grad for value in model.parameters())


def test_pair_frozen_stage_learns_on_top_of_frozen_entity_experts():
    """Pair warmup must use the same R_U+R_s+R_c baseline as inference."""
    model = _expert_model()
    model.train()
    model.set_training_stage("pair")

    result = model.forward_named_components(
        torch.zeros(1, 4),
        torch.zeros(1, 3),
        torch.zeros(1, 2),
        torch.ones(1, 1),
        general_cell_inputs=torch.zeros(1, 2),
        general_perturbation_inputs=torch.zeros(1, 1),
        strain_indices=torch.tensor([1]),
        chemical_indices=torch.tensor([1]),
        strain_seen=torch.ones(1, 1),
        chemical_seen=torch.ones(1, 1),
        pair_indices=torch.tensor([1]),
        pair_seen=torch.ones(1, 1),
    )

    assert result.strain_gate.item() == 1.0
    assert result.chemical_gate.item() == 1.0
    assert result.pair_gate.item() == 1.0
    assert torch.allclose(result.response_strain, torch.tensor([[1.0, 0.0, 1.0]]))
    assert torch.allclose(result.response_chemical, torch.tensor([[0.0, 2.0, 2.0]]))
    assert torch.allclose(result.response_pair, torch.tensor([[3.0, 4.0, 7.0]]))
    assert torch.allclose(
        result.response,
        result.response_universal
        + result.response_strain
        + result.response_chemical
        + result.response_pair,
    )
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    assert trainable
    assert all(name.startswith("response_pair_") for name in trainable)


def test_m7_configs_expose_general_and_pair_ablation():
    general = load_response_config("configs/experiments/m7_general_only.yaml")
    experts = load_response_config("configs/experiments/m7_entity_experts.yaml")
    pair = load_response_config("configs/experiments/m7_entity_pair_expert.yaml")
    assert general.model.interaction_mode == "shared_general_experts"
    assert not general.model.response_chemical_expert_enabled
    assert experts.model.response_chemical_expert_enabled
    assert not experts.model.response_pair_expert_enabled
    assert pair.model.response_pair_expert_enabled
    assert 0.0 < pair.model.entity_dropout < 1.0


def test_m7_training_and_unknown_entity_inference_are_connected(tmp_path):
    baseline = write_config(tmp_path, epochs=1)
    response = {
        "baseline_config": baseline.name,
        "model": {
            "hidden_dim": 4,
            "response_rank": 2,
            "calibration_rank": 2,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "epochs": 5,
            "batch_size": 8,
            "seed": 11,
            "device": "cpu",
            "absolute_weight": 0.25,
            "background_weight": 1.0,
            "fc_weight": 1.0,
            "target_scale_floor": 0.1,
            "calibration_enabled": True,
            "interaction_mode": "shared_general_experts",
            "response_pair_expert_enabled": True,
            "entity_dropout": 0.25,
            "universal_epochs": 1,
            "strain_expert_epochs": 1,
            "chemical_expert_epochs": 1,
            "pair_expert_epochs": 1,
            "joint_learning_rate_scale": 0.2,
        },
        "entity": {"chemical_map": None, "strain_features": None, "chemical_bits": 8},
        "graph": {"variant": "none", "artifact": None, "weight": 0.0},
        "runtime": {"runs_dir": "runs"},
    }
    response_path = tmp_path / "response.yaml"
    with response_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(response, handle)
    config = load_response_config(response_path)
    data = prepare_data(config.baseline)
    fit = fit_response_model(config, data, data.train_ids)
    ids = data.metadata.index[data.metadata["split_final"].eq("val_both")]
    prediction = _predict(
        fit.model,
        fit.builder,
        data.metadata,
        ids,
        data.proteins,
        fit.target_mean,
        fit.target_scale,
        fit.device,
        config.model.batch_size,
    )
    assert prediction.shape == (1, len(data.proteins))
    assert np.isfinite(prediction.to_numpy()).all()
    unknown = fit.builder.transform(data.metadata.loc[ids])
    assert unknown.strain_seen[:, 0].tolist() == [0.0]
    assert unknown.chemical_seen[:, 0].tolist() == [0.0]
    assert [row["training_stage"] for row in fit.history] == [
        "universal", "strain", "chemical", "pair", "joint"
    ]
    assert [row["learning_rate"] for row in fit.history] == [
        0.001, 0.001, 0.001, 0.001, 0.0002
    ]

    run = train_response_model(config, tmp_path / "m7-run")
    manifest_path = predict_response_components(
        response_path, run, tmp_path / "components"
    )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    assert manifest["components"] == {name: f"{name}.npy" for name in COMPONENT_NAMES}
    assert manifest["max_abs_reconstruction_error"] <= 1e-4
    arrays = {
        name: np.load(manifest_path.parent / filename)
        for name, filename in manifest["components"].items()
    }
    reconstructed = (
        arrays["B_U"]
        + arrays["B_s"]
        + arrays["C_obs"]
        + arrays["R_U"]
        + arrays["R_s"]
        + arrays["R_c"]
        + arrays["R_sc"]
    )
    assert np.allclose(reconstructed, arrays["final"], atol=1e-4)


def test_fold_matched_warm_start_preserves_r00_and_writes_receipt(tmp_path):
    """Frozen residual experts must begin from the exact M7.0 common state."""
    baseline = write_config(tmp_path, epochs=1)

    def write_response(name: str, *, experts: bool) -> str:
        response = {
            "baseline_config": baseline.name,
            "model": {
                "hidden_dim": 4,
                "response_rank": 2,
                "calibration_rank": 2,
                "dropout": 0.0,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "epochs": 3 if experts else 1,
                "batch_size": 8,
                "seed": 17,
                "device": "cpu",
                "absolute_weight": 0.25,
                "background_weight": 1.0,
                "fc_weight": 1.0,
                "target_scale_floor": 0.1,
                "calibration_enabled": True,
                "interaction_mode": "shared_general_experts",
                "background_strain_expert_enabled": experts,
                "response_strain_expert_enabled": experts,
                "response_chemical_expert_enabled": experts,
                "response_pair_expert_enabled": False,
                "entity_dropout": 0.25 if experts else 0.0,
                "universal_epochs": 1,
                "strain_expert_epochs": 1 if experts else 0,
                "chemical_expert_epochs": 1 if experts else 0,
                "pair_expert_epochs": 0,
                "joint_learning_rate_scale": 0.2,
                "fold_matched_universal_warm_start": True,
            },
            "entity": {
                "chemical_map": None,
                "strain_features": None,
                "chemical_bits": 8,
            },
            "graph": {"variant": "none", "artifact": None, "weight": 0.0},
            "runtime": {"runs_dir": "runs"},
        }
        path = tmp_path / name
        path.write_text(yaml.safe_dump(response), encoding="utf-8")
        return str(path)

    control_config = load_response_config(
        write_response("m70-fold-matched.yaml", experts=False)
    )
    expert_config = load_response_config(
        write_response("m73-fold-matched.yaml", experts=True)
    )
    data = prepare_data(control_config.baseline)
    control = fit_response_model(control_config, data, data.train_ids)
    expert = fit_response_model(expert_config, data, data.train_ids)

    control_hash = control.training_receipt["universal_state_sha256"]
    receipt = expert.training_receipt
    assert receipt["enabled"] is True
    assert receipt["expanded_after_universal"] is True
    assert receipt["optimizer_reset_on_expert_expansion"] is True
    assert receipt["copied_universal_state_sha256"] == control_hash
    assert receipt["post_frozen_expert_universal_state_sha256"] == control_hash
    assert receipt["final_universal_state_sha256"] == control_hash
    assert receipt["common_state_unchanged_during_frozen_experts"] is True

    r00_ids = data.metadata.index[data.metadata["split_final"].eq("val_both")]
    control_prediction = _predict(
        control.model,
        control.builder,
        data.metadata,
        r00_ids,
        data.proteins,
        control.target_mean,
        control.target_scale,
        control.device,
        control_config.model.batch_size,
    )
    expert_prediction = _predict(
        expert.model,
        expert.builder,
        data.metadata,
        r00_ids,
        data.proteins,
        expert.target_mean,
        expert.target_scale,
        expert.device,
        expert_config.model.batch_size,
    )
    assert np.array_equal(
        control_prediction.to_numpy(), expert_prediction.to_numpy()
    )
