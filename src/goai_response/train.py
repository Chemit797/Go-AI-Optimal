"""Train and evaluate the response-aligned non-graph MVP."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from goai_baseline.audit import audit_inputs
from goai_baseline.controls import exact_control_predictions
from goai_baseline.evaluate import evaluate_predictor, write_evaluation
from goai_baseline.loss import masked_mse
from goai_baseline.manifest import write_manifest
from goai_baseline.official_metrics import evaluate_official_proxy
from .artifacts import (
    validate_chemical_feature_artifact,
    validate_chemical_structure_artifact,
    validate_strain_feature_artifact,
)
from goai_baseline.preprocess import prepare_data
from goai_baseline.schema import control_mask, treatment_mask
from goai_baseline.train import resolve_device, set_seed

from .config import ResponseConfig, load_response_config
from .entities import (
    build_support_manifest,
    manifest_sha256,
    sha256_file,
    stable_json_dumps,
    write_json_with_hash,
)
from .features import ResponseFeatureBuilder, ResponseFeatures
from .model import ResponseDecompositionRegressor


@dataclass
class ResponseFit:
    """Fold-fitted response model and all training-derived state."""

    model: ResponseDecompositionRegressor
    builder: ResponseFeatureBuilder
    target_mean: np.ndarray
    target_scale: np.ndarray
    history: list[dict[str, float | int | str]]
    fc_pair_count: int
    fc_observed_count: int
    basis_summary: dict[str, float | int | str]
    support_manifest: dict[str, object] | None
    support_manifest_sha256: str
    artifact_hashes: dict[str, object]
    artifact_chain_sha256: str
    training_receipt: dict[str, object]
    device: torch.device


def _artifact_hash_chain(config: ResponseConfig) -> tuple[dict[str, object], str]:
    """Hash the effective response configuration and every declared model artifact."""

    def record(path: Path | None) -> dict[str, str] | None:
        if path is None:
            return None
        if not path.is_file():
            raise FileNotFoundError(f"Declared model artifact does not exist: {path}")
        return {"path": str(path.resolve()), "sha256": sha256_file(path)}

    effective = {
        "model": asdict(config.model),
        "entity": asdict(config.entity),
        "graph": asdict(config.graph),
        "baseline_config": str(config.baseline.path),
    }
    effective_hash = hashlib.sha256(
        stable_json_dumps(effective).encode("utf-8")
    ).hexdigest()
    chemical_structure_manifest = validate_chemical_structure_artifact(
        config.entity.chemical_map,
        manifest_required=config.entity.chemical_structure_manifest_required,
    )
    if (
        isinstance(chemical_structure_manifest, dict)
        and chemical_structure_manifest.get("status") == "verified_manifest"
    ):
        expected_structure_kind = {
            "exact": "exact",
            "parent": "parent_normalized",
            "zero_risky": "zero_risky",
            "shuffled": "exact_shuffled",
        }.get(config.entity.chemical_structure_view)
        if expected_structure_kind is not None and str(
            chemical_structure_manifest.get("selected_kind", "")
        ) != expected_structure_kind:
            raise ValueError(
                "Chemical structure view label does not match its declared "
                f"manifest member: expected {expected_structure_kind!r}"
            )
    artifacts: dict[str, object] = {
        "response_config": record(config.path),
        "effective_response_config_sha256": effective_hash,
        "baseline_config": record(config.baseline.path),
        "chemical_map": record(config.entity.chemical_map),
        "chemical_structure_manifest": chemical_structure_manifest,
        "chemical_features": record(config.entity.chemical_features),
        "chemical_features_manifest": validate_chemical_feature_artifact(
            config.entity.chemical_features,
            manifest_required=config.entity.chemical_features_manifest_required,
        ),
        "strain_features": record(config.entity.strain_features),
        "strain_features_manifest": validate_strain_feature_artifact(
            config.entity.strain_features,
            manifest_required=config.entity.strain_features_manifest_required,
        ),
        "chemical_registry": record(config.entity.chemical_registry),
        "strain_registry": record(config.entity.strain_registry),
        "chemical_parent_views": record(config.entity.chemical_parent_views),
        "chemical_identity_risks": record(config.entity.chemical_identity_risks),
        "graph": record(config.graph.artifact),
    }
    chain = hashlib.sha256(stable_json_dumps(artifacts).encode("utf-8")).hexdigest()
    return artifacts, chain


def _fold_support_manifest(
    config: ResponseConfig,
    metadata: pd.DataFrame,
    builder: ResponseFeatureBuilder,
) -> tuple[dict[str, object] | None, str]:
    """Build the authoritative fold-local support snapshot when registries are configured."""
    if config.entity.chemical_registry is None and config.entity.strain_registry is None:
        return None, ""
    if config.entity.chemical_registry is None or config.entity.strain_registry is None:
        raise ValueError("Chemical and strain registries must be configured together")
    manifest = build_support_manifest(
        metadata,
        {
            "chemical": config.entity.chemical_registry,
            "strain": config.entity.strain_registry,
        },
    )
    builder.validate_support_manifest(manifest)
    return manifest, manifest_sha256(manifest)


def _load_response_edges(config: ResponseConfig, proteins: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load only a declared topology control for response smoothness."""
    if config.graph.variant == "none" or config.graph.weight == 0:
        return None
    if config.graph.variant not in {"real_ppi", "rewired_ppi"}:
        raise ValueError("graph.variant must be none, real_ppi, or rewired_ppi")
    if config.graph.artifact is None:
        raise ValueError("graph.artifact is required when PPI regularisation is enabled")
    from goai_graph.graph import load_graph_bundle

    bundle = load_graph_bundle(config.graph.artifact)
    if bundle.proteins != proteins:
        raise ValueError("PPI artifact protein order does not match response targets")
    edges = bundle.edge_index if config.graph.variant == "real_ppi" else bundle.rewired_edge_index
    weights = bundle.edge_weight if config.graph.variant == "real_ppi" else bundle.rewired_edge_weight
    return torch.from_numpy(edges).long().to(device), torch.from_numpy(weights).float().to(device)


def _ppi_smoothness(response: torch.Tensor, graph_edges: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
    """Weakly encourage related proteins to share response components.

    This is deliberately a regulariser, not a claim that PPI edges are active
    causal transmission channels in every perturbation condition.
    """
    if graph_edges is None:
        return response.sum() * 0.0
    edges, weights = graph_edges
    difference = response[:, edges[0]] - response[:, edges[1]]
    return (difference.square() * weights.unsqueeze(0)).mean()


def _target_statistics(targets: pd.DataFrame, floor: float) -> tuple[np.ndarray, np.ndarray]:
    values = targets.to_numpy(dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(values, axis=0).astype(np.float32)
        scale = np.maximum(np.nanstd(values, axis=0).astype(np.float32), floor)
    # The global protein contract is fixed before folding, so a rare protein
    # may have no observation in one fold's training rows.
    mean = np.nan_to_num(mean, nan=0.0)
    scale = np.nan_to_num(scale, nan=floor)
    return mean, scale


def _training_fc_targets(data, target_mean: np.ndarray, target_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = data.train_ids
    values = np.zeros((len(ids), len(data.proteins)), dtype=np.float32)
    mask = np.zeros_like(values)
    treatment_ids = ids[treatment_mask(data.metadata.loc[ids]).to_numpy()]
    matched = exact_control_predictions(data.metadata, data.y_log2, treatment_ids, ids)
    usable = treatment_ids[matched.has_exact_match.to_numpy()]
    if len(usable):
        delta = data.y_log2.loc[usable].to_numpy(dtype=np.float32) - matched.predictions.loc[usable].to_numpy(dtype=np.float32)
        positions = ids.get_indexer(usable)
        finite = np.isfinite(delta)
        values[positions] = np.nan_to_num(delta / target_scale[None, :], nan=0.0)
        mask[positions] = finite.astype(np.float32)
    return values, mask


def _as_tensors(
    features: ResponseFeatures,
    response_prior: np.ndarray,
    background_selector: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    fc: np.ndarray,
    fc_mask: np.ndarray,
):
    return TensorDataset(
        torch.from_numpy(features.response), torch.from_numpy(features.background), torch.from_numpy(features.cell),
        torch.from_numpy(features.perturbation), torch.from_numpy(features.observation), torch.from_numpy(features.is_treatment),
        torch.from_numpy(features.general_cell), torch.from_numpy(features.general_perturbation),
        torch.from_numpy(features.strain_indices), torch.from_numpy(features.chemical_indices),
        torch.from_numpy(features.strain_seen), torch.from_numpy(features.chemical_seen),
        torch.from_numpy(features.pair_indices), torch.from_numpy(features.pair_seen),
        torch.from_numpy(response_prior), torch.from_numpy(background_selector), torch.from_numpy(targets), torch.from_numpy(masks), torch.from_numpy(fc), torch.from_numpy(fc_mask),
    )


def _masked_or_zero(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    kind: str = "mse",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Keep shuffled mini-batches valid when they contain no paired controls."""
    if float(mask.sum().detach().cpu()) <= 0:
        return prediction.sum() * 0.0
    if kind == "mse":
        return masked_mse(prediction, target, mask)
    error = prediction - target
    if kind == "huber":
        absolute = error.abs()
        elementwise = torch.where(
            absolute <= huber_delta,
            0.5 * error.square(),
            huber_delta * (absolute - 0.5 * huber_delta),
        )
    elif kind == "mse_mae":
        elementwise = 0.5 * error.square() + 0.5 * error.abs()
    else:
        raise ValueError(f"Unknown masked loss: {kind}")
    return (elementwise * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_correlation_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """One minus Pearson correlation over observed response values."""
    observed = mask.bool()
    if int(observed.sum().detach().cpu()) < 2:
        return prediction.sum() * 0.0
    pred = prediction[observed]
    truth = target[observed]
    pred = pred - pred.mean()
    truth = truth - truth.mean()
    denominator = pred.square().sum().sqrt() * truth.square().sum().sqrt()
    if float(denominator.detach().cpu()) <= 1e-12:
        return prediction.sum() * 0.0
    return 1.0 - (pred * truth).sum() / denominator.clamp_min(1e-12)


def _fit_fixed_svd_basis(
    fc: np.ndarray,
    fc_mask: np.ndarray,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
    """Fit a response PCA basis using only fold-training matched-control deltas."""
    usable = fc_mask.any(axis=1)
    if not usable.any():
        raise ValueError("fixed_svd requires at least one exact matched-control response pair")
    observed = fc_mask[usable].astype(np.float32)
    values = fc[usable].astype(np.float32)
    counts = observed.sum(axis=0)
    center = np.divide(
        (values * observed).sum(axis=0),
        counts,
        out=np.zeros(values.shape[1], dtype=np.float32),
        where=counts > 0,
    )
    centered = (values - center[None, :]) * observed
    matrix = torch.from_numpy(centered).to(device)
    effective_rank = min(rank, matrix.shape[0], matrix.shape[1])
    _, singular, right = torch.pca_lowrank(matrix, q=effective_rank, center=False, niter=4)
    basis = torch.zeros((rank, matrix.shape[1]), dtype=matrix.dtype, device=device)
    basis[:effective_rank] = right[:, :effective_rank].T
    total_energy = float(matrix.square().sum().detach().cpu())
    explained = float(singular.square().sum().detach().cpu()) / total_energy if total_energy > 0 else 0.0
    summary: dict[str, float | int | str] = {
        "mode": "fixed_svd",
        "requested_rank": rank,
        "effective_rank": effective_rank,
        "training_response_rows": int(usable.sum()),
        "explained_energy_ratio": explained,
    }
    return torch.from_numpy(center).to(device), basis, summary


def _predict(
    model: ResponseDecompositionRegressor,
    builder: ResponseFeatureBuilder,
    metadata: pd.DataFrame,
    ids: pd.Index,
    proteins: list[str],
    target_mean: np.ndarray,
    target_scale: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    features = builder.transform(metadata.loc[ids])
    response_prior = builder.response_prior(metadata.loc[ids])
    model.eval()
    result: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            absolute, _, _ = model(
                torch.from_numpy(features.response[start:end]).to(device),
                torch.from_numpy(features.background[start:end]).to(device),
                torch.from_numpy(features.observation[start:end]).to(device),
                torch.from_numpy(features.is_treatment[start:end]).to(device),
                torch.from_numpy(features.cell[start:end]).to(device),
                torch.from_numpy(features.perturbation[start:end]).to(device),
                torch.from_numpy(response_prior[start:end]).to(device),
                torch.from_numpy(features.general_cell[start:end]).to(device),
                torch.from_numpy(features.general_perturbation[start:end]).to(device),
                torch.from_numpy(features.strain_indices[start:end]).to(device),
                torch.from_numpy(features.chemical_indices[start:end]).to(device),
                torch.from_numpy(features.strain_seen[start:end]).to(device),
                torch.from_numpy(features.chemical_seen[start:end]).to(device),
                torch.from_numpy(features.pair_indices[start:end]).to(device),
                torch.from_numpy(features.pair_seen[start:end]).to(device),
            )
            result.append(absolute.cpu().numpy())
    values = np.concatenate(result, axis=0) * target_scale[None, :] + target_mean[None, :]
    return pd.DataFrame(values, index=ids, columns=proteins)


def _predict_core_components(
    model: ResponseDecompositionRegressor,
    builder: ResponseFeatureBuilder,
    metadata: pd.DataFrame,
    ids: pd.Index,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Export the natural-log2 base, response, and reconstructed prediction."""
    features = builder.transform(metadata.loc[ids])
    response_prior = builder.response_prior(metadata.loc[ids])
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        "background_plus_calibration": [],
        "response": [],
        "final": [],
    }
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            absolute, background_plus_calibration, response = model(
                torch.from_numpy(features.response[start:end]).to(device),
                torch.from_numpy(features.background[start:end]).to(device),
                torch.from_numpy(features.observation[start:end]).to(device),
                torch.from_numpy(features.is_treatment[start:end]).to(device),
                torch.from_numpy(features.cell[start:end]).to(device),
                torch.from_numpy(features.perturbation[start:end]).to(device),
                torch.from_numpy(response_prior[start:end]).to(device),
                torch.from_numpy(features.general_cell[start:end]).to(device),
                torch.from_numpy(features.general_perturbation[start:end]).to(device),
                torch.from_numpy(features.strain_indices[start:end]).to(device),
                torch.from_numpy(features.chemical_indices[start:end]).to(device),
                torch.from_numpy(features.strain_seen[start:end]).to(device),
                torch.from_numpy(features.chemical_seen[start:end]).to(device),
                torch.from_numpy(features.pair_indices[start:end]).to(device),
                torch.from_numpy(features.pair_seen[start:end]).to(device),
            )
            scale = torch.as_tensor(
                target_scale, dtype=absolute.dtype, device=absolute.device
            )
            mean = torch.as_tensor(
                target_mean, dtype=absolute.dtype, device=absolute.device
            )
            values = {
                "background_plus_calibration": background_plus_calibration * scale
                + mean,
                "response": response * scale,
                "final": absolute * scale + mean,
            }
            for name, value in values.items():
                collected[name].append(
                    value.detach().cpu().numpy().astype(np.float32, copy=False)
                )
    result = {
        name: np.concatenate(parts, axis=0) for name, parts in collected.items()
    }
    result["is_treatment"] = features.is_treatment.astype(np.float32, copy=True)
    return result


def fit_response_model(config: ResponseConfig, data, fit_ids: pd.Index, seed: int | None = None) -> ResponseFit:
    """Fit the unchanged response model using only explicit fold-training IDs."""
    fold_seed = config.model.seed if seed is None else seed
    set_seed(fold_seed)
    device = resolve_device(config.model.device)
    fold_data = replace(data, train_ids=pd.Index(fit_ids))
    builder = ResponseFeatureBuilder(
        chemical_map=config.entity.chemical_map,
        strain_features_path=config.entity.strain_features,
        strain_feature_columns=config.entity.strain_feature_columns,
        strain_feature_transform=config.entity.strain_feature_transform,
        chemical_bits=config.entity.chemical_bits,
        chemical_features_path=config.entity.chemical_features,
        chemical_registry_path=config.entity.chemical_registry,
        strain_registry_path=config.entity.strain_registry,
        chemical_parent_views_path=config.entity.chemical_parent_views,
        chemical_identity_risks_path=config.entity.chemical_identity_risks,
        allow_proxy_semantics=config.entity.allow_proxy_semantics,
        chemical_structure_view=config.entity.chemical_structure_view,
        semantic_identity_policy=config.entity.semantic_identity_policy,
        semantic_training_coverage_required=(
            config.entity.semantic_training_coverage_required
        ),
        calibration_use_plate=config.model.calibration_use_plate,
        calibration_plate_shuffle=config.model.calibration_plate_shuffle,
        calibration_shuffle_seed=config.model.calibration_shuffle_seed,
    ).fit(data.metadata, fold_data.train_ids)
    support_manifest, support_manifest_hash = _fold_support_manifest(
        config, data.metadata.loc[fold_data.train_ids], builder
    )
    artifact_hashes, artifact_chain_hash = _artifact_hash_chain(config)
    train_features = builder.transform(data.metadata.loc[fold_data.train_ids])
    target_mean, target_scale = _target_statistics(
        data.y_log2.loc[fold_data.train_ids], config.model.target_scale_floor
    )
    raw = data.y_log2.loc[fold_data.train_ids].to_numpy(dtype=np.float32)
    targets = np.nan_to_num((raw - target_mean[None, :]) / target_scale[None, :], nan=0.0)
    masks = np.isfinite(raw).astype(np.float32)
    fc, fc_mask = _training_fc_targets(fold_data, target_mean, target_scale)
    builder.fit_response_priors(
        data.metadata.loc[fold_data.train_ids],
        fc,
        fc_mask,
        config.model.response_prior_mode,
        config.model.response_prior_alpha,
    )
    train_response_prior = builder.response_prior(
        data.metadata.loc[fold_data.train_ids],
        leave_one_out_fc=fc,
        leave_one_out_mask=fc_mask,
    )
    if config.model.qc_policy == "controls_only":
        background_selector = control_mask(data.metadata.loc[fold_data.train_ids]).to_numpy(dtype=np.float32).reshape(-1, 1)
    else:
        background_selector = 1.0 - train_features.is_treatment
    dataset = _as_tensors(train_features, train_response_prior, background_selector, targets, masks, fc, fc_mask)
    loader = DataLoader(
        dataset,
        batch_size=config.model.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(fold_seed),
    )
    full_model_kwargs = dict(
        response_input_dim=train_features.response.shape[1],
        background_input_dim=train_features.background.shape[1],
        observation_input_dim=train_features.observation.shape[1],
        n_proteins=len(data.proteins),
        hidden_dim=config.model.hidden_dim,
        response_rank=config.model.response_rank,
        calibration_rank=config.model.calibration_rank,
        dropout=config.model.dropout,
        calibration_enabled=config.model.calibration_enabled,
        response_basis=config.model.response_basis,
        cell_input_dim=train_features.cell.shape[1],
        perturbation_input_dim=train_features.perturbation.shape[1],
        interaction_mode=config.model.interaction_mode,
        calibration_plate_start=builder.observation_slices.get("Yeast_cell_plate", (-1, -1))[0],
        calibration_plate_end=builder.observation_slices.get("Yeast_cell_plate", (-1, -1))[1],
        calibration_plate_dropout=config.model.calibration_plate_dropout,
        response_prior_learnable_scale=config.model.response_prior_learnable_scale,
        general_cell_input_dim=train_features.general_cell.shape[1],
        general_perturbation_input_dim=train_features.general_perturbation.shape[1],
        n_strain_entities=len(builder.strain_entity_keys),
        n_chemical_entities=len(builder.chemical_entity_keys),
        n_pair_entities=len(builder.pair_entity_keys),
        background_strain_expert_enabled=config.model.background_strain_expert_enabled,
        response_strain_expert_enabled=config.model.response_strain_expert_enabled,
        response_chemical_expert_enabled=config.model.response_chemical_expert_enabled,
        response_pair_expert_enabled=config.model.response_pair_expert_enabled,
        strain_expert_scale=config.model.strain_expert_scale,
        chemical_expert_scale=config.model.chemical_expert_scale,
        pair_expert_scale=config.model.pair_expert_scale,
        entity_dropout=config.model.entity_dropout,
        allow_research_expert_scale_override=config.model.allow_research_expert_scale_override,
    )
    experts_enabled = any(
        (
            config.model.background_strain_expert_enabled,
            config.model.response_strain_expert_enabled,
            config.model.response_chemical_expert_enabled,
            config.model.response_pair_expert_enabled,
        )
    )
    strict_expansion = bool(
        config.model.fold_matched_universal_warm_start and experts_enabled
    )

    def build_model(*, universal_only: bool) -> ResponseDecompositionRegressor:
        kwargs = dict(full_model_kwargs)
        if universal_only:
            kwargs.update(
                background_strain_expert_enabled=False,
                response_strain_expert_enabled=False,
                response_chemical_expert_enabled=False,
                response_pair_expert_enabled=False,
            )
        return ResponseDecompositionRegressor(**kwargs).to(device)

    model = build_model(universal_only=strict_expansion)
    model.set_calibration_input_center(
        torch.from_numpy(train_features.observation.mean(axis=0)).to(device)
    )
    fixed_response_center: torch.Tensor | None = None
    fixed_response_basis: torch.Tensor | None = None
    if config.model.response_basis == "fixed_svd":
        response_center, response_basis, basis_summary = _fit_fixed_svd_basis(
            fc, fc_mask, config.model.response_rank, device
        )
        model.set_fixed_response_basis(response_center, response_basis)
        fixed_response_center = response_center
        fixed_response_basis = response_basis
    else:
        basis_summary = {
            "mode": "learned",
            "requested_rank": config.model.response_rank,
            "effective_rank": config.model.response_rank,
            "training_response_rows": int(fc_mask.any(axis=1).sum()),
            "explained_energy_ratio": 0.0,
        }
    graph_edges = _load_response_edges(config, data.proteins, device)
    def build_optimizer(current: ResponseDecompositionRegressor) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            current.parameters(),
            lr=config.model.learning_rate,
            weight_decay=config.model.weight_decay,
        )

    optimizer = build_optimizer(model)
    training_receipt: dict[str, object] = {
        "schema": "fold_matched_universal_warm_start_v1",
        "enabled": bool(config.model.fold_matched_universal_warm_start),
        "expert_expansion_required": strict_expansion,
        "expanded_after_universal": False,
        "optimizer_reset_on_expert_expansion": False,
        "universal_epochs": int(config.model.universal_epochs),
        "universal_state_sha256": "",
        "copied_universal_state_sha256": "",
        "post_frozen_expert_universal_state_sha256": "",
        "final_universal_state_sha256": "",
        "common_state_unchanged_during_frozen_experts": None,
        # Fold-local receipt: this is computed only from the actual fit IDs.
        # Identity flags and ID-expert support are deliberately excluded from
        # semantic coverage, so a formal M8 run cannot silently train on an
        # all-zero semantic modality.
        "semantic_training_coverage": deepcopy(
            builder.semantic_training_coverage
        ),
    }

    def expand_experts_after_universal() -> None:
        nonlocal model, optimizer
        if not strict_expansion or training_receipt["expanded_after_universal"]:
            return
        source = model
        source_hash = source.universal_state_sha256()
        expected_hash = str(training_receipt["universal_state_sha256"])
        if expected_hash and source_hash != expected_hash:
            raise RuntimeError("Universal state changed before expert expansion")
        expanded = build_model(universal_only=False)
        expanded.set_calibration_input_center(
            torch.from_numpy(train_features.observation.mean(axis=0)).to(device)
        )
        if fixed_response_center is not None and fixed_response_basis is not None:
            expanded.set_fixed_response_basis(
                fixed_response_center, fixed_response_basis
            )
        copied_hash = expanded.copy_universal_state_from(source)
        model = expanded
        # The residual phases deliberately start with a fresh optimizer.  This
        # prevents hidden universal Adam moments from updating frozen tensors
        # and makes the transfer receipt independent of optimizer internals.
        optimizer = build_optimizer(model)
        training_receipt["expanded_after_universal"] = True
        training_receipt["optimizer_reset_on_expert_expansion"] = True
        training_receipt["universal_state_sha256"] = source_hash
        training_receipt["copied_universal_state_sha256"] = copied_hash

    history: list[dict[str, float | int | str]] = []
    for epoch in range(1, config.model.epochs + 1):
        boundary_universal = config.model.universal_epochs
        boundary_strain = boundary_universal + config.model.strain_expert_epochs
        boundary_chemical = boundary_strain + config.model.chemical_expert_epochs
        boundary_pair = boundary_chemical + config.model.pair_expert_epochs
        if epoch <= boundary_universal:
            training_stage = "universal"
        elif epoch <= boundary_strain:
            training_stage = "strain"
        elif epoch <= boundary_chemical:
            training_stage = "chemical"
        elif epoch <= boundary_pair:
            training_stage = "pair"
        else:
            training_stage = "joint"
        if strict_expansion and epoch > boundary_universal:
            expand_experts_after_universal()
        if training_stage == "joint" and not training_receipt[
            "post_frozen_expert_universal_state_sha256"
        ]:
            training_receipt[
                "post_frozen_expert_universal_state_sha256"
            ] = model.universal_state_sha256()
        model.set_training_stage(training_stage)
        epoch_learning_rate = config.model.learning_rate * (
            config.model.joint_learning_rate_scale if training_stage == "joint" else 1.0
        )
        for group in optimizer.param_groups:
            group["lr"] = epoch_learning_rate
        model.train()
        totals = {"absolute": 0.0, "background": 0.0, "fc": 0.0, "fc_corr": 0.0, "ppi": 0.0, "cal_center": 0.0, "cal_l2": 0.0, "loss": 0.0}
        batches = 0
        for response, background, cell, perturbation, observation, treatment, general_cell, general_perturbation, strain_indices, chemical_indices, strain_seen, chemical_seen, pair_indices, pair_seen, response_prior, background_selector, target, mask, delta, delta_mask in loader:
            response, background, cell, perturbation, observation, treatment, general_cell, general_perturbation, strain_indices, chemical_indices, strain_seen, chemical_seen, pair_indices, pair_seen, response_prior, background_selector, target, mask, delta, delta_mask = [
                value.to(device)
                for value in (response, background, cell, perturbation, observation, treatment, general_cell, general_perturbation, strain_indices, chemical_indices, strain_seen, chemical_seen, pair_indices, pair_seen, response_prior, background_selector, target, mask, delta, delta_mask)
            ]
            optimizer.zero_grad(set_to_none=True)
            absolute, background_prediction, response_prediction, _, calibration_prediction = model.forward_components(
                response, background, observation, treatment, cell, perturbation, response_prior,
                general_cell, general_perturbation, strain_indices, chemical_indices,
                strain_seen, chemical_seen, pair_indices, pair_seen,
            )
            absolute_loss = _masked_or_zero(absolute, target, mask, config.model.absolute_loss, config.model.huber_delta)
            background_loss = _masked_or_zero(
                background_prediction,
                target,
                mask * background_selector,
                config.model.background_loss,
                config.model.huber_delta,
            )
            fc_loss = _masked_or_zero(response_prediction, delta, delta_mask, config.model.fc_loss, config.model.huber_delta)
            fc_corr_loss = _masked_correlation_loss(response_prediction, delta, delta_mask)
            ppi_loss = _ppi_smoothness(response_prediction, graph_edges)
            calibration_center_loss = calibration_prediction.mean(dim=0).square().mean()
            calibration_l2_loss = calibration_prediction.square().mean()
            loss = (
                config.model.absolute_weight * absolute_loss
                + config.model.background_weight * background_loss
                + config.model.fc_weight * fc_loss
                + config.model.fc_correlation_weight * fc_corr_loss
                + config.graph.weight * ppi_loss
                + config.model.calibration_center_weight * calibration_center_loss
                + config.model.calibration_l2_weight * calibration_l2_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for name, value in (("absolute", absolute_loss), ("background", background_loss), ("fc", fc_loss), ("fc_corr", fc_corr_loss), ("ppi", ppi_loss), ("cal_center", calibration_center_loss), ("cal_l2", calibration_l2_loss), ("loss", loss)):
                totals[name] += float(value.detach().cpu())
            batches += 1
        row = {
            "epoch": epoch,
            "training_stage": training_stage,
            "learning_rate": epoch_learning_rate,
            **{name: value / batches for name, value in totals.items()},
        }
        history.append(row)
        if epoch == boundary_universal:
            training_receipt["universal_state_sha256"] = (
                model.universal_state_sha256()
            )
        if epoch == boundary_pair and boundary_pair > boundary_universal:
            training_receipt[
                "post_frozen_expert_universal_state_sha256"
            ] = model.universal_state_sha256()

    # Even a receipt-only config that stops exactly at the universal boundary
    # must serialize the requested full expert architecture with zero experts.
    expand_experts_after_universal()
    if not training_receipt["universal_state_sha256"]:
        training_receipt["universal_state_sha256"] = model.universal_state_sha256()
    if not training_receipt["copied_universal_state_sha256"]:
        training_receipt["copied_universal_state_sha256"] = str(
            training_receipt["universal_state_sha256"]
        )
    if not training_receipt["post_frozen_expert_universal_state_sha256"]:
        training_receipt[
            "post_frozen_expert_universal_state_sha256"
        ] = model.universal_state_sha256()
    training_receipt["final_universal_state_sha256"] = (
        model.universal_state_sha256()
    )
    training_receipt["common_state_unchanged_during_frozen_experts"] = (
        str(training_receipt["universal_state_sha256"])
        == str(training_receipt["post_frozen_expert_universal_state_sha256"])
    )
    return ResponseFit(
        model=model,
        builder=builder,
        target_mean=target_mean,
        target_scale=target_scale,
        history=history,
        fc_pair_count=int(fc_mask.any(axis=1).sum()),
        fc_observed_count=int(fc_mask.sum()),
        basis_summary=basis_summary,
        support_manifest=support_manifest,
        support_manifest_sha256=support_manifest_hash,
        artifact_hashes=artifact_hashes,
        artifact_chain_sha256=artifact_chain_hash,
        training_receipt=training_receipt,
        device=device,
    )


def train_response_model(
    config: ResponseConfig,
    run_dir: str | Path | None = None,
    seed: int | None = None,
    fit_all_labeled: bool = False,
) -> Path:
    if config.model.nested_expert_scale_selection:
        raise ValueError(
            "nested_expert_scale_selection is an outer-OOF confirmation policy; "
            "a single full-data refit cannot silently substitute scale 1"
        )
    if seed is not None:
        config = replace(config, model=replace(config.model, seed=seed))
    audit_inputs(config.baseline)
    data = prepare_data(config.baseline)
    output = Path(run_dir) if run_dir else config.runs_dir / f"response-mvp-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)

    fit_ids = data.metadata.index if fit_all_labeled else data.train_ids
    fit_scope = "all_released_labeled_rows" if fit_all_labeled else "outer_train_only"
    fit = fit_response_model(config, data, fit_ids)
    model, builder = fit.model, fit.builder
    target_mean, target_scale = fit.target_mean, fit.target_scale
    history, device = fit.history, fit.device
    fitted_features = builder.transform(data.metadata.loc[fit_ids])
    support_manifest_file_sha256 = ""
    if fit.support_manifest is not None:
        support_manifest_file_sha256 = write_json_with_hash(
            output / "support_manifest.json", fit.support_manifest
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_kwargs": {
            "response_input_dim": fitted_features.response.shape[1],
            "background_input_dim": fitted_features.background.shape[1],
            "observation_input_dim": fitted_features.observation.shape[1],
            "n_proteins": len(data.proteins),
            "hidden_dim": config.model.hidden_dim,
            "response_rank": config.model.response_rank,
            "calibration_rank": config.model.calibration_rank,
            "dropout": config.model.dropout,
            "calibration_enabled": config.model.calibration_enabled,
            "response_basis": config.model.response_basis,
            "cell_input_dim": fitted_features.cell.shape[1],
            "perturbation_input_dim": fitted_features.perturbation.shape[1],
            "interaction_mode": config.model.interaction_mode,
            "calibration_plate_start": builder.observation_slices.get("Yeast_cell_plate", (-1, -1))[0],
            "calibration_plate_end": builder.observation_slices.get("Yeast_cell_plate", (-1, -1))[1],
            "calibration_plate_dropout": config.model.calibration_plate_dropout,
            "response_prior_learnable_scale": config.model.response_prior_learnable_scale,
            "general_cell_input_dim": fitted_features.general_cell.shape[1],
            "general_perturbation_input_dim": fitted_features.general_perturbation.shape[1],
            "n_strain_entities": len(builder.strain_entity_keys),
            "n_chemical_entities": len(builder.chemical_entity_keys),
            "n_pair_entities": len(builder.pair_entity_keys),
            "background_strain_expert_enabled": config.model.background_strain_expert_enabled,
            "response_strain_expert_enabled": config.model.response_strain_expert_enabled,
            "response_chemical_expert_enabled": config.model.response_chemical_expert_enabled,
            "response_pair_expert_enabled": config.model.response_pair_expert_enabled,
            "strain_expert_scale": config.model.strain_expert_scale,
            "chemical_expert_scale": config.model.chemical_expert_scale,
            "pair_expert_scale": config.model.pair_expert_scale,
            "entity_dropout": config.model.entity_dropout,
            "allow_research_expert_scale_override": config.model.allow_research_expert_scale_override,
        },
        "feature_state": builder.state_dict(), "proteins": data.proteins, "target_mean": target_mean, "target_scale": target_scale,
        "target_scale_name": "per-protein train-standardised log2; response target is matched-control FC on this scale",
        "fit_scope": fit_scope,
        "fit_sample_count": int(len(fit_ids)),
        "support_manifest": fit.support_manifest,
        "support_manifest_sha256": fit.support_manifest_sha256,
        "support_manifest_file_sha256": support_manifest_file_sha256,
        "artifact_hashes": fit.artifact_hashes,
        "artifact_chain_sha256": fit.artifact_chain_sha256,
        "training_schedule": {
            "universal_epochs": config.model.universal_epochs,
            "strain_expert_epochs": config.model.strain_expert_epochs,
            "chemical_expert_epochs": config.model.chemical_expert_epochs,
            "pair_expert_epochs": config.model.pair_expert_epochs,
            "joint_epochs": config.model.epochs
            - config.model.universal_epochs
            - config.model.strain_expert_epochs
            - config.model.chemical_expert_epochs
            - config.model.pair_expert_epochs,
            "joint_learning_rate_scale": config.model.joint_learning_rate_scale,
            "fold_matched_universal_warm_start": config.model.fold_matched_universal_warm_start,
        },
        "training_receipt": fit.training_receipt,
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    with (output / "feature_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({**builder.summary(), "train_fc_pairs": fit.fc_pair_count, "train_fc_observed_values": fit.fc_observed_count, "response_basis": fit.basis_summary}, handle, ensure_ascii=False, indent=2)

    predictor = lambda ids: _predict(model, builder, data.metadata, ids, data.proteins, target_mean, target_scale, device, config.model.batch_size)
    if fit_all_labeled:
        pd.DataFrame().to_csv(output / "metrics.csv", index=False)
        pd.DataFrame().to_csv(output / "official_proxy_metrics.csv", index=False)
    else:
        metrics, protein_metrics = evaluate_predictor(data, predictor, "response_mvp")
        write_evaluation(output, metrics, protein_metrics)
        evaluate_official_proxy(data, predictor).to_csv(output / "official_proxy_metrics.csv", index=False)
        print(metrics.to_string(index=False))
    write_manifest(output / "manifest.json", config.baseline, {"experiment": "response_calibration_mvp", "response_config": str(config.path), "device": str(device), "model_parameter_count": int(sum(p.numel() for p in model.parameters())), "fit_scope": fit_scope, "fit_sample_count": int(len(fit_ids)), "outer_metrics_valid": not fit_all_labeled, "loss_weights": {"absolute": config.model.absolute_weight, "background": config.model.background_weight, "fc": config.model.fc_weight, "fc_correlation": config.model.fc_correlation_weight, "ppi": config.graph.weight}, "loss_types": {"absolute": config.model.absolute_loss, "background": config.model.background_loss, "fc": config.model.fc_loss}, "response_basis": fit.basis_summary, "graph_variant": config.graph.variant, "fc_training_pairs": fit.fc_pair_count, "support_manifest": "support_manifest.json" if fit.support_manifest is not None else None, "support_manifest_sha256": fit.support_manifest_sha256, "support_manifest_file_sha256": support_manifest_file_sha256, "artifact_hashes": fit.artifact_hashes, "artifact_chain_sha256": fit.artifact_chain_sha256, "training_receipt": fit.training_receipt})
    print(f"Wrote run: {output.resolve()}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train response-aligned virtual-cell MVP")
    parser.add_argument("--config", required=True); parser.add_argument("--run-dir", default=None); parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fit-all-labeled", action="store_true")
    args = parser.parse_args()
    train_response_model(load_response_config(args.config), args.run_dir, args.seed, args.fit_all_labeled)


if __name__ == "__main__":
    main()
