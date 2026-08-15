"""Configuration for response-aligned experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from goai_baseline.config import BaselineConfig, load_config


LOCKED_EXPERT_SCALES = frozenset({0.0, 0.25, 0.5, 0.75, 1.0})


@dataclass(frozen=True)
class ResponseModelConfig:
    hidden_dim: int
    response_rank: int
    calibration_rank: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int
    seed: int
    device: str
    absolute_weight: float
    background_weight: float
    fc_weight: float
    target_scale_floor: float
    calibration_enabled: bool
    response_basis: str
    absolute_loss: str
    background_loss: str
    fc_loss: str
    huber_delta: float
    fc_correlation_weight: float
    interaction_mode: str
    calibration_use_plate: bool
    calibration_plate_dropout: float
    calibration_plate_shuffle: bool
    calibration_shuffle_seed: int
    calibration_center_weight: float
    calibration_l2_weight: float
    qc_policy: str
    response_prior_mode: str
    response_prior_alpha: float
    response_prior_learnable_scale: bool
    background_strain_expert_enabled: bool
    response_strain_expert_enabled: bool
    response_chemical_expert_enabled: bool
    response_pair_expert_enabled: bool
    strain_expert_scale: float
    chemical_expert_scale: float
    pair_expert_scale: float
    entity_dropout: float
    universal_epochs: int
    strain_expert_epochs: int
    chemical_expert_epochs: int
    pair_expert_epochs: int
    joint_learning_rate_scale: float
    fold_matched_universal_warm_start: bool
    allow_research_expert_scale_override: bool
    nested_expert_scale_selection: bool
    nested_expert_scale_inner_folds: int


@dataclass(frozen=True)
class EntityConfig:
    chemical_map: Path | None
    chemical_structure_manifest_required: bool
    chemical_features: Path | None
    chemical_features_manifest_required: bool
    strain_features: Path | None
    strain_features_manifest_required: bool
    strain_feature_columns: tuple[str, ...] | None
    strain_feature_transform: str
    chemical_registry: Path | None
    strain_registry: Path | None
    chemical_parent_views: Path | None
    chemical_identity_risks: Path | None
    chemical_bits: int
    allow_proxy_semantics: bool
    chemical_structure_view: str
    semantic_identity_policy: str
    semantic_training_coverage_required: bool


@dataclass(frozen=True)
class GraphRegularizationConfig:
    artifact: Path | None
    variant: str
    weight: float


@dataclass(frozen=True)
class ResponseConfig:
    path: Path
    baseline: BaselineConfig
    model: ResponseModelConfig
    entity: EntityConfig
    graph: GraphRegularizationConfig
    runs_dir: Path


def _resolve(root: Path, value: object | None) -> Path | None:
    if value in (None, "", False):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping")
    return value


def load_response_config(path: str | Path) -> ResponseConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Response config root must be a mapping")
    root = config_path.parent
    baseline_value = payload.get("baseline_config")
    if not isinstance(baseline_value, str):
        raise ValueError("baseline_config must be a path")
    model = _mapping(payload, "model")
    entity = _mapping(payload, "entity")
    runtime = _mapping(payload, "runtime")
    graph = payload.get("graph", {})
    if not isinstance(graph, dict):
        raise ValueError("graph must be a mapping when supplied")
    result = ResponseModelConfig(
        hidden_dim=int(model["hidden_dim"]),
        response_rank=int(model["response_rank"]),
        calibration_rank=int(model["calibration_rank"]),
        dropout=float(model["dropout"]),
        learning_rate=float(model["learning_rate"]),
        weight_decay=float(model["weight_decay"]),
        epochs=int(model["epochs"]),
        batch_size=int(model["batch_size"]),
        seed=int(model["seed"]),
        device=str(model["device"]),
        absolute_weight=float(model["absolute_weight"]),
        background_weight=float(model["background_weight"]),
        fc_weight=float(model["fc_weight"]),
        target_scale_floor=float(model["target_scale_floor"]),
        calibration_enabled=bool(model.get("calibration_enabled", True)),
        response_basis=str(model.get("response_basis", "learned")),
        absolute_loss=str(model.get("absolute_loss", "mse")),
        background_loss=str(model.get("background_loss", "mse")),
        fc_loss=str(model.get("fc_loss", "mse")),
        huber_delta=float(model.get("huber_delta", 1.0)),
        fc_correlation_weight=float(model.get("fc_correlation_weight", 0.0)),
        interaction_mode=str(model.get("interaction_mode", "independent_legacy")),
        calibration_use_plate=bool(model.get("calibration_use_plate", True)),
        calibration_plate_dropout=float(model.get("calibration_plate_dropout", 0.0)),
        calibration_plate_shuffle=bool(model.get("calibration_plate_shuffle", False)),
        calibration_shuffle_seed=int(model.get("calibration_shuffle_seed", model["seed"])),
        calibration_center_weight=float(model.get("calibration_center_weight", 0.0)),
        calibration_l2_weight=float(model.get("calibration_l2_weight", 0.0)),
        qc_policy=str(model.get("qc_policy", "legacy")),
        response_prior_mode=str(model.get("response_prior_mode", "none")),
        response_prior_alpha=float(model.get("response_prior_alpha", 4.0)),
        response_prior_learnable_scale=bool(model.get("response_prior_learnable_scale", False)),
        background_strain_expert_enabled=bool(model.get("background_strain_expert_enabled", True)),
        response_strain_expert_enabled=bool(model.get("response_strain_expert_enabled", True)),
        response_chemical_expert_enabled=bool(model.get("response_chemical_expert_enabled", True)),
        response_pair_expert_enabled=bool(model.get("response_pair_expert_enabled", False)),
        strain_expert_scale=float(model.get("strain_expert_scale", 1.0)),
        chemical_expert_scale=float(model.get("chemical_expert_scale", 1.0)),
        pair_expert_scale=float(model.get("pair_expert_scale", 1.0)),
        entity_dropout=float(model.get("entity_dropout", 0.0)),
        universal_epochs=int(model.get("universal_epochs", 0)),
        strain_expert_epochs=int(model.get("strain_expert_epochs", 0)),
        chemical_expert_epochs=int(model.get("chemical_expert_epochs", 0)),
        pair_expert_epochs=int(model.get("pair_expert_epochs", 0)),
        joint_learning_rate_scale=float(model.get("joint_learning_rate_scale", 1.0)),
        fold_matched_universal_warm_start=bool(
            model.get("fold_matched_universal_warm_start", False)
        ),
        allow_research_expert_scale_override=bool(
            model.get("allow_research_expert_scale_override", False)
        ),
        nested_expert_scale_selection=bool(
            model.get("nested_expert_scale_selection", False)
        ),
        nested_expert_scale_inner_folds=int(
            model.get("nested_expert_scale_inner_folds", 2)
        ),
    )
    if min(result.hidden_dim, result.response_rank, result.calibration_rank, result.epochs, result.batch_size) <= 0:
        raise ValueError("Model dimensions, epochs, and batch size must be positive")
    if not 0 <= result.dropout < 1:
        raise ValueError("model.dropout must be in [0, 1)")
    if min(result.absolute_weight, result.background_weight, result.fc_weight) < 0:
        raise ValueError("Loss weights cannot be negative")
    if result.target_scale_floor <= 0:
        raise ValueError("target_scale_floor must be positive")
    if result.response_basis not in {"learned", "fixed_svd"}:
        raise ValueError("model.response_basis must be learned or fixed_svd")
    valid_losses = {"mse", "huber", "mse_mae"}
    for name, value in (
        ("absolute_loss", result.absolute_loss),
        ("background_loss", result.background_loss),
        ("fc_loss", result.fc_loss),
    ):
        if value not in valid_losses:
            raise ValueError(f"model.{name} must be one of {sorted(valid_losses)}")
    if result.huber_delta <= 0:
        raise ValueError("model.huber_delta must be positive")
    if result.fc_correlation_weight < 0:
        raise ValueError("model.fc_correlation_weight cannot be negative")
    if result.interaction_mode not in {"independent_legacy", "shared_concat", "shared_gate", "shared_film", "shared_general_experts"}:
        raise ValueError("model.interaction_mode is invalid")
    if not 0 <= result.calibration_plate_dropout < 1:
        raise ValueError("model.calibration_plate_dropout must be in [0, 1)")
    if min(result.calibration_center_weight, result.calibration_l2_weight) < 0:
        raise ValueError("calibration penalties cannot be negative")
    if result.qc_policy not in {"legacy", "controls_only"}:
        raise ValueError("model.qc_policy must be legacy or controls_only")
    if result.response_prior_mode not in {"none", "chemical", "strain", "both"}:
        raise ValueError("model.response_prior_mode is invalid")
    if result.response_prior_alpha <= 0:
        raise ValueError("model.response_prior_alpha must be positive")
    if not 0 <= result.entity_dropout < 1:
        raise ValueError("model.entity_dropout must be in [0, 1)")
    if min(result.strain_expert_scale, result.chemical_expert_scale, result.pair_expert_scale) < 0:
        raise ValueError("model expert scales cannot be negative")
    scales = {
        result.strain_expert_scale,
        result.chemical_expert_scale,
        result.pair_expert_scale,
    }
    if not result.allow_research_expert_scale_override and not scales <= LOCKED_EXPERT_SCALES:
        raise ValueError(
            "model expert scales must be one of 0, 0.25, 0.5, 0.75, 1; "
            "set allow_research_expert_scale_override=true only for labelled research"
        )
    if result.nested_expert_scale_inner_folds < 2:
        raise ValueError("model.nested_expert_scale_inner_folds must be at least 2")
    if result.nested_expert_scale_selection:
        if result.interaction_mode != "shared_general_experts":
            raise ValueError(
                "nested expert-scale selection requires shared_general_experts"
            )
        if scales != {1.0}:
            raise ValueError(
                "nested expert-scale models must train every expert at canonical "
                "scale 1; inner OOF chooses inference scales"
            )
    staged_epochs = (
        result.universal_epochs
        + result.strain_expert_epochs
        + result.chemical_expert_epochs
        + result.pair_expert_epochs
    )
    if min(
        result.universal_epochs,
        result.strain_expert_epochs,
        result.chemical_expert_epochs,
        result.pair_expert_epochs,
    ) < 0:
        raise ValueError("model staged epoch counts cannot be negative")
    if staged_epochs > result.epochs:
        raise ValueError("model staged epoch counts cannot exceed total epochs")
    if staged_epochs and result.interaction_mode != "shared_general_experts":
        raise ValueError("model staged training requires shared_general_experts")
    if result.strain_expert_epochs and not (
        result.background_strain_expert_enabled or result.response_strain_expert_enabled
    ):
        raise ValueError("strain expert warmup requires an enabled strain expert")
    if result.chemical_expert_epochs and not result.response_chemical_expert_enabled:
        raise ValueError("chemical expert warmup requires an enabled chemical expert")
    if result.pair_expert_epochs and not result.response_pair_expert_enabled:
        raise ValueError("pair expert warmup requires an enabled pair expert")
    if result.joint_learning_rate_scale <= 0:
        raise ValueError("model.joint_learning_rate_scale must be positive")
    if result.fold_matched_universal_warm_start:
        if result.interaction_mode != "shared_general_experts":
            raise ValueError(
                "fold-matched universal warm start requires shared_general_experts"
            )
        if result.universal_epochs <= 0:
            raise ValueError(
                "fold-matched universal warm start requires universal_epochs > 0"
            )
        experts_enabled = any(
            (
                result.background_strain_expert_enabled,
                result.response_strain_expert_enabled,
                result.response_chemical_expert_enabled,
                result.response_pair_expert_enabled,
            )
        )
        if experts_enabled and staged_epochs == result.universal_epochs:
            raise ValueError(
                "fold-matched expert config must allocate a frozen residual stage"
            )
    runs = _resolve(root, runtime.get("runs_dir"))
    if runs is None:
        raise ValueError("runtime.runs_dir is required")
    chemical_structure_view = str(entity.get("chemical_structure_view", "exact"))
    if chemical_structure_view not in {"exact", "parent", "zero_risky", "zero", "shuffled"}:
        raise ValueError(
            "entity.chemical_structure_view must be exact, parent, zero_risky, zero, or shuffled"
        )
    chemical_registry = _resolve(root, entity.get("chemical_registry"))
    strain_registry = _resolve(root, entity.get("strain_registry"))
    chemical_parent_views = _resolve(root, entity.get("chemical_parent_views"))
    chemical_identity_risks = _resolve(root, entity.get("chemical_identity_risks"))
    if (chemical_registry is None) != (strain_registry is None):
        raise ValueError(
            "entity.chemical_registry and entity.strain_registry must be configured together"
        )
    if chemical_structure_view == "parent" and chemical_parent_views is None:
        raise ValueError(
            "entity.chemical_parent_views is required for the explicit parent view"
        )
    if chemical_structure_view == "zero_risky" and chemical_identity_risks is None:
        raise ValueError(
            "entity.chemical_identity_risks is required for the reviewed zero_risky view"
        )
    allow_proxy_semantics = bool(entity.get("allow_proxy_semantics", False))
    semantic_identity_policy = str(
        entity.get("semantic_identity_policy", "verified_only")
    )
    if semantic_identity_policy not in {
        "verified_only",
        "research_allow_candidate",
    }:
        raise ValueError(
            "entity.semantic_identity_policy must be verified_only or "
            "research_allow_candidate"
        )
    if allow_proxy_semantics and semantic_identity_policy != "research_allow_candidate":
        raise ValueError(
            "Proxy/parent semantics are research-only: set "
            "entity.semantic_identity_policy=research_allow_candidate explicitly"
        )
    chemical_map = _resolve(root, entity.get("chemical_map"))
    chemical_structure_manifest_required = bool(
        entity.get("chemical_structure_manifest_required", False)
    )
    if chemical_structure_manifest_required and chemical_map is None:
        raise ValueError(
            "entity.chemical_structure_manifest_required=true requires chemical_map"
        )
    chemical_features = _resolve(root, entity.get("chemical_features"))
    chemical_features_manifest_required = bool(
        entity.get("chemical_features_manifest_required", False)
    )
    if chemical_features_manifest_required and chemical_features is None:
        raise ValueError(
            "entity.chemical_features_manifest_required=true requires chemical_features"
        )
    strain_features = _resolve(root, entity.get("strain_features"))
    strain_features_manifest_required = bool(
        entity.get("strain_features_manifest_required", False)
    )
    if strain_features_manifest_required and strain_features is None:
        raise ValueError(
            "entity.strain_features_manifest_required=true requires strain_features"
        )
    raw_strain_feature_columns = entity.get("strain_feature_columns")
    if raw_strain_feature_columns is None:
        strain_feature_columns = None
    elif isinstance(raw_strain_feature_columns, list) and all(
        isinstance(value, str) and value.strip()
        for value in raw_strain_feature_columns
    ):
        strain_feature_columns = tuple(
            str(value).strip() for value in raw_strain_feature_columns
        )
        if len(set(strain_feature_columns)) != len(strain_feature_columns):
            raise ValueError("entity.strain_feature_columns contains duplicates")
    else:
        raise ValueError("entity.strain_feature_columns must be a list of column names")
    if strain_feature_columns is not None and strain_features is None:
        raise ValueError(
            "entity.strain_feature_columns requires entity.strain_features"
        )
    strain_feature_transform = str(
        entity.get("strain_feature_transform", "scaled")
    )
    if strain_feature_transform not in {"scaled", "rbf", "nearest"}:
        raise ValueError(
            "entity.strain_feature_transform must be scaled, rbf, or nearest"
        )
    if strain_feature_transform in {"rbf", "nearest"} and strain_features is None:
        raise ValueError(
            f"entity.strain_feature_transform={strain_feature_transform} requires strain_features"
        )
    if chemical_structure_view == "parent" and not allow_proxy_semantics:
        raise ValueError(
            "entity.chemical_structure_view=parent requires allow_proxy_semantics=true"
        )
    return ResponseConfig(
        path=config_path,
        baseline=load_config(_resolve(root, baseline_value)),  # type: ignore[arg-type]
        model=result,
        entity=EntityConfig(
            chemical_map=chemical_map,
            chemical_structure_manifest_required=chemical_structure_manifest_required,
            chemical_features=chemical_features,
            chemical_features_manifest_required=chemical_features_manifest_required,
            strain_features=strain_features,
            strain_features_manifest_required=strain_features_manifest_required,
            strain_feature_columns=strain_feature_columns,
            strain_feature_transform=strain_feature_transform,
            chemical_registry=chemical_registry,
            strain_registry=strain_registry,
            chemical_parent_views=chemical_parent_views,
            chemical_identity_risks=chemical_identity_risks,
            chemical_bits=int(entity.get("chemical_bits", 512)),
            allow_proxy_semantics=allow_proxy_semantics,
            chemical_structure_view=chemical_structure_view,
            semantic_identity_policy=semantic_identity_policy,
            semantic_training_coverage_required=bool(
                entity.get("semantic_training_coverage_required", False)
            ),
        ),
        graph=GraphRegularizationConfig(
            artifact=_resolve(root, graph.get("artifact")),
            variant=str(graph.get("variant", "none")),
            weight=float(graph.get("weight", 0.0)),
        ),
        runs_dir=runs,
    )
