"""Inference for the response-decomposition checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from goai_baseline.schema import SAMPLE_ID, require_metadata_columns, require_unique_sample_ids
from goai_baseline.submission import verify_submission
from goai_baseline.train import resolve_device

from .artifacts import (
    validate_chemical_feature_artifact,
    validate_chemical_structure_artifact,
    validate_strain_feature_artifact,
)
from .config import ResponseConfig, load_response_config
from .entities import (
    load_json_with_hash,
    load_registry,
    manifest_sha256,
    sha256_file,
    stable_json_dumps,
)
from .features import ResponseFeatureBuilder
from .model import ResponseDecompositionRegressor


COMPONENT_NAMES = ("B_U", "B_s", "C_obs", "R_U", "R_s", "R_c", "R_sc", "final")


def _natural_scale_components(
    output,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert named standardized heads into an exactly reconstructable log2 decomposition."""
    scale = torch.as_tensor(target_scale, dtype=output.absolute.dtype, device=output.absolute.device)
    mean = torch.as_tensor(target_mean, dtype=output.absolute.dtype, device=output.absolute.device)
    tensors = {
        # The per-protein target mean is assigned to B_U so the natural-scale
        # components reconstruct final without an unlabelled intercept term.
        "B_U": output.background_universal * scale + mean,
        "B_s": output.background_strain * scale,
        "C_obs": output.calibration * scale,
        # Fold-safe response priors are part of the universal response rather
        # than an entity-ID expert and are folded into R_U for this contract.
        "R_U": (output.response_universal + output.response_prior) * scale,
        "R_s": output.response_strain * scale,
        "R_c": output.response_chemical * scale,
        "R_sc": output.response_pair * scale,
        "final": output.absolute * scale + mean,
    }
    return {name: value.detach().cpu().numpy().astype(np.float32) for name, value in tensors.items()}


def load_response_checkpoint(
    path: str | Path,
    device: torch.device,
    config: ResponseConfig | None = None,
) -> tuple[ResponseDecompositionRegressor, ResponseFeatureBuilder, list[str], np.ndarray, np.ndarray]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {"model_state_dict", "model_kwargs", "feature_state", "proteins", "target_mean", "target_scale"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Response checkpoint is missing fields: {missing}")

    model = ResponseDecompositionRegressor(**payload["model_kwargs"]).to(device)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    # response_center was added after the first MVP checkpoints.  Its all-zero
    # default exactly reproduces those learned-basis models.
    # New buffers default to the legacy computation when absent.  In
    # particular, a zero calibration_input_center exactly preserves historical
    # M2/M6 predictions.
    allowed_missing = {
        "response_center",
        "response_prior_scale",
        "calibration_input_center",
    }
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise ValueError(
            "Checkpoint state does not match the declared response model: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    builder = ResponseFeatureBuilder.from_state_dict(payload["feature_state"])
    support_manifest = payload.get("support_manifest")
    if support_manifest is not None:
        if not isinstance(support_manifest, dict):
            raise ValueError("Checkpoint support_manifest must be a mapping")
        actual_manifest_hash = manifest_sha256(support_manifest)
        expected_manifest_hash = str(payload.get("support_manifest_sha256", ""))
        if not expected_manifest_hash or actual_manifest_hash != expected_manifest_hash:
            raise ValueError("Checkpoint support manifest hash is missing or invalid")
        builder.validate_support_manifest(support_manifest)
        sidecar = checkpoint_path.parent / "support_manifest.json"
        if sidecar.is_file():
            sidecar_manifest = load_json_with_hash(
                sidecar,
                str(payload.get("support_manifest_file_sha256", "")) or None,
            )
            if manifest_sha256(sidecar_manifest) != actual_manifest_hash:
                raise ValueError("Checkpoint and support_manifest.json do not match")
        if config is not None:
            if config.entity.chemical_registry is None or config.entity.strain_registry is None:
                raise ValueError(
                    "This checkpoint requires its authoritative chemical and strain registries"
                )
            current_registry_hashes = {
                "chemical": load_registry(config.entity.chemical_registry, "chemical").sha256,
                "strain": load_registry(config.entity.strain_registry, "strain").sha256,
            }
            if current_registry_hashes != support_manifest.get("registry_sha256"):
                raise ValueError("Configured entity registries do not match the checkpoint")
    artifact_hashes = payload.get("artifact_hashes")
    if config is not None and not isinstance(artifact_hashes, dict):
        if support_manifest is not None:
            raise ValueError(
                "Registry-aware checkpoint is missing its artifact hash chain"
            )
    if config is not None and isinstance(artifact_hashes, dict):
        expected_chain = str(payload.get("artifact_chain_sha256", ""))
        actual_chain = hashlib.sha256(
            stable_json_dumps(artifact_hashes).encode("utf-8")
        ).hexdigest()
        if not expected_chain or actual_chain != expected_chain:
            raise ValueError("Checkpoint artifact hash chain is missing or invalid")
        if "chemical_features_manifest" in artifact_hashes:
            expected_feature_manifest = artifact_hashes["chemical_features_manifest"]
            actual_feature_manifest = validate_chemical_feature_artifact(
                config.entity.chemical_features,
                manifest_required=config.entity.chemical_features_manifest_required,
            )
            if stable_json_dumps(actual_feature_manifest) != stable_json_dumps(
                expected_feature_manifest
            ):
                raise ValueError(
                    "Chemical feature manifest/source chain does not match checkpoint"
                )
        if "chemical_structure_manifest" in artifact_hashes:
            expected_structure_manifest = artifact_hashes["chemical_structure_manifest"]
            actual_structure_manifest = validate_chemical_structure_artifact(
                config.entity.chemical_map,
                manifest_required=config.entity.chemical_structure_manifest_required,
            )
            if stable_json_dumps(actual_structure_manifest) != stable_json_dumps(
                expected_structure_manifest
            ):
                raise ValueError(
                    "Chemical structure manifest/source chain does not match checkpoint"
                )
        if "strain_features_manifest" in artifact_hashes:
            expected_strain_manifest = artifact_hashes["strain_features_manifest"]
            actual_strain_manifest = validate_strain_feature_artifact(
                config.entity.strain_features,
                manifest_required=config.entity.strain_features_manifest_required,
            )
            if stable_json_dumps(actual_strain_manifest) != stable_json_dumps(
                expected_strain_manifest
            ):
                raise ValueError(
                    "Strain feature manifest/source chain does not match checkpoint"
                )
        declared = {
            "response_config": config.path,
            "baseline_config": config.baseline.path,
            "chemical_map": config.entity.chemical_map,
            "chemical_features": config.entity.chemical_features,
            "strain_features": config.entity.strain_features,
            "chemical_registry": config.entity.chemical_registry,
            "strain_registry": config.entity.strain_registry,
            "chemical_parent_views": config.entity.chemical_parent_views,
            "chemical_identity_risks": config.entity.chemical_identity_risks,
            "graph": config.graph.artifact,
        }
        for name, current_path in declared.items():
            expected = artifact_hashes.get(name)
            if expected is None:
                if current_path is not None:
                    raise ValueError(f"Configured {name} was absent from checkpoint artifact chain")
                continue
            if current_path is None or not current_path.is_file():
                raise ValueError(f"Checkpoint artifact {name} is unavailable at prediction time")
            if not isinstance(expected, dict) or sha256_file(current_path) != expected.get("sha256"):
                raise ValueError(f"Configured {name} hash does not match checkpoint artifact chain")
    proteins = [str(protein) for protein in payload["proteins"]]
    target_mean = np.asarray(payload["target_mean"], dtype=np.float32)
    target_scale = np.asarray(payload["target_scale"], dtype=np.float32)
    expected = len(proteins)
    if target_mean.shape != (expected,) or target_scale.shape != (expected,):
        raise ValueError("Checkpoint target statistics do not match its protein list")
    if not np.isfinite(target_mean).all() or not np.isfinite(target_scale).all() or np.any(target_scale <= 0):
        raise ValueError("Checkpoint target statistics are invalid")
    return model, builder, proteins, target_mean, target_scale


def predict_response_test(
    config_path: str | Path,
    run_dir: str | Path,
    output_csv: str | Path | None = None,
) -> Path:
    config = load_response_config(config_path)
    device = resolve_device(config.model.device)
    run = Path(run_dir)
    checkpoint_path = run / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, builder, proteins, target_mean, target_scale = load_response_checkpoint(
        checkpoint_path, device, config
    )
    metadata_path = config.baseline.data.metadata_test
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Test metadata not found: {metadata_path}")
    metadata = pd.read_csv(metadata_path, low_memory=False)
    require_metadata_columns(metadata)
    require_unique_sample_ids(metadata, "test metadata")
    metadata = metadata.set_index(SAMPLE_ID, verify_integrity=True)
    features = builder.transform(metadata)
    response_prior = builder.response_prior(metadata)

    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(metadata), config.model.batch_size):
            end = min(start + config.model.batch_size, len(metadata))
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
            batches.append(absolute.cpu().numpy())
    standardised = np.concatenate(batches, axis=0)
    prediction = standardised * target_scale[None, :] + target_mean[None, :]
    if not np.isfinite(prediction).all():
        raise ValueError("Response model produced non-finite predictions")

    output = Path(output_csv) if output_csv is not None else run / "prediction.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(prediction, index=metadata.index, columns=proteins)
    submission.index.name = SAMPLE_ID
    submission.to_csv(output)
    report = verify_submission(output, metadata_path, proteins)
    report["model"] = "response_decomposition_mvp"
    report["checkpoint"] = str(checkpoint_path.resolve())
    with (output.parent / "prediction_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote submission: {output.resolve()}")
    return output


def predict_response_components(
    config_path: str | Path,
    run_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Stream the M7 named decomposition to NPY files plus a reconstruction manifest."""
    config = load_response_config(config_path)
    device = resolve_device(config.model.device)
    run = Path(run_dir)
    checkpoint_path = run / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model, builder, proteins, target_mean, target_scale = load_response_checkpoint(
        checkpoint_path, device, config
    )
    if model.interaction_mode != "shared_general_experts":
        raise ValueError("Named expert export is available only for shared_general_experts checkpoints")

    metadata_path = config.baseline.data.metadata_test
    metadata = pd.read_csv(metadata_path, low_memory=False)
    require_metadata_columns(metadata)
    require_unique_sample_ids(metadata, "test metadata")
    metadata = metadata.set_index(SAMPLE_ID, verify_integrity=True)
    features = builder.transform(metadata)
    response_prior = builder.response_prior(metadata)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    shape = (len(metadata), len(proteins))
    component_maps = {
        name: np.lib.format.open_memmap(
            destination / f"{name}.npy", mode="w+", dtype=np.float32, shape=shape
        )
        for name in COMPONENT_NAMES
    }
    max_reconstruction_error = 0.0
    with torch.no_grad():
        for start in range(0, len(metadata), config.model.batch_size):
            end = min(start + config.model.batch_size, len(metadata))
            output = model.forward_named_components(
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
            values = _natural_scale_components(output, target_mean, target_scale)
            for name in COMPONENT_NAMES:
                component_maps[name][start:end] = values[name]
            treatment = features.is_treatment[start:end]
            reconstructed = (
                values["B_U"]
                + values["B_s"]
                + values["C_obs"]
                + treatment
                * (
                    values["R_U"]
                    + values["R_s"]
                    + values["R_c"]
                    + values["R_sc"]
                )
            )
            max_reconstruction_error = max(
                max_reconstruction_error,
                float(np.max(np.abs(reconstructed - values["final"]))),
            )
    for values in component_maps.values():
        values.flush()
    if max_reconstruction_error > 1e-4:
        raise RuntimeError(
            f"Named components do not reconstruct final prediction: {max_reconstruction_error}"
        )

    pd.DataFrame({SAMPLE_ID: metadata.index.astype(str)}).to_csv(
        destination / "sample_ids.csv", index=False
    )
    pd.DataFrame({"protein": proteins}).to_csv(destination / "proteins.csv", index=False)
    manifest = {
        "protocol": "m7_named_component_export_v1",
        "checkpoint": str(checkpoint_path.resolve()),
        "interaction_mode": model.interaction_mode,
        "rows": shape[0],
        "proteins": shape[1],
        "dtype": "float32",
        "scale": "natural log2 proteome units",
        "components": {name: f"{name}.npy" for name in COMPONENT_NAMES},
        "response_prior_policy": "included in R_U",
        "target_mean_policy": "included in B_U",
        "reconstruction": "final = B_U + B_s + C_obs + is_treatment * (R_U + R_s + R_c + R_sc)",
        "max_abs_reconstruction_error": max_reconstruction_error,
        "sample_ids": "sample_ids.csv",
        "protein_order": "proteins.csv",
    }
    manifest_path = destination / "component_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GOAI predictions from a response-model checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--components-output-dir", default=None)
    args = parser.parse_args()
    predict_response_test(args.config, args.run_dir, args.output_csv)
    if args.components_output_dir is not None:
        predict_response_components(args.config, args.run_dir, args.components_output_dir)


if __name__ == "__main__":
    main()
