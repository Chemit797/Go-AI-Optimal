"""Replace selected test scenarios in a verified base prediction with seed ensembles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from goai_baseline.schema import (
    SAMPLE_ID,
    SPLIT,
    require_metadata_columns,
    require_unique_sample_ids,
    treatment_mask,
)
from goai_baseline.submission import verify_submission
from goai_response.config import load_response_config
from goai_response.predict import load_response_checkpoint
from goai_response.routing import CONTROL_ROUTE, SUPPORT_REGIMES, support_route_audit
from goai_response.train import _predict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _components(spec: dict[str, object]) -> list[dict[str, object]]:
    raw = spec.get("components")
    if raw is None:
        raw = [{"config": spec["config"], "runs": spec["runs"], "weight": 1.0}]
    if not isinstance(raw, list) or not raw or not all(isinstance(item, dict) for item in raw):
        raise ValueError("Route components must be a non-empty list of objects")
    for component in raw:
        if "config" not in component:
            raise ValueError("Every route component requires config")
        runs = component.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError("Every route component requires a non-empty runs list")
    return raw


def _read_support_manifest(value: object, route_path: Path) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        path = Path(value)
        if not path.is_absolute():
            path = route_path.parent / path
        from goai_response.entities import load_json_with_hash

        return load_json_with_hash(path.resolve())
    raise ValueError("support_manifest must be an object or JSON path")


def _fit_support_manifest(
    route_manifest: dict[str, object],
    route_path: Path,
    routes: dict[str, object],
) -> dict[str, object]:
    """Load and cross-check manifests tied to the actual routed refits."""

    candidates: list[tuple[str, dict[str, object]]] = []
    explicit = route_manifest.get("support_manifest", route_manifest.get("fit_support_manifest"))
    if explicit is not None:
        candidates.append(("route_manifest", _read_support_manifest(explicit, route_path)))
    checkpoint_count = 0
    for route_name, raw_spec in routes.items():
        if not isinstance(raw_spec, dict):
            continue
        for component in _components(raw_spec):
            for raw_run in component["runs"]:
                run = Path(str(raw_run)).resolve()
                sidecar = run / "support_manifest.json"
                if sidecar.is_file():
                    candidates.append((str(sidecar), _read_support_manifest(str(sidecar), route_path)))
                checkpoint = run / "checkpoint.pt"
                if not checkpoint.is_file():
                    # Manifest consistency is checked first so a corrupted
                    # multi-run route cannot be obscured by a later missing
                    # checkpoint. Strict checkpoint/config checks still occur
                    # below after the manifest comparison.
                    continue
                checkpoint_count += 1
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                embedded = payload.get("support_manifest", payload.get("fit_support_manifest"))
                if isinstance(embedded, dict):
                    candidates.append((str(checkpoint), embedded))
    if not candidates:
        raise ValueError(
            "Support-regime routes require the actual fit support manifest at the route-manifest, "
            "run sidecar, or checkpoint level"
        )
    from goai_response.entities import manifest_sha256

    hashes = {manifest_sha256(manifest) for _, manifest in candidates}
    if len(hashes) != 1:
        sources = [source for source, _ in candidates]
        raise ValueError(f"Routed refits have different fit support manifests: {sources}")
    # Only after support consistency is established do we enforce that every
    # production component supplies its exact config and chained checkpoint.
    for route_name, raw_spec in routes.items():
        if not isinstance(raw_spec, dict):
            continue
        for component in _components(raw_spec):
            config_path = Path(str(component["config"])).resolve()
            if not config_path.is_file():
                raise ValueError(
                    f"Routed component config is missing for {route_name}: {config_path}"
                )
            for raw_run in component["runs"]:
                checkpoint = Path(str(raw_run)).resolve() / "checkpoint.pt"
                if not checkpoint.is_file():
                    raise ValueError(f"Routed checkpoint is missing: {checkpoint}")
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                embedded = payload.get("support_manifest", payload.get("fit_support_manifest"))
                if not isinstance(embedded, dict):
                    raise ValueError(
                        f"Every support-routed checkpoint must embed its fit support manifest: {checkpoint}"
                    )
                from goai_response.entities import manifest_sha256

                if manifest_sha256(embedded) != str(
                    payload.get("support_manifest_sha256", "")
                ):
                    raise ValueError(
                        f"Routed checkpoint support manifest hash is missing or invalid: {checkpoint}"
                    )
                if not isinstance(payload.get("artifact_hashes"), dict) or not str(
                    payload.get("artifact_chain_sha256", "")
                ):
                    raise ValueError(
                        f"Routed checkpoint lacks its strict artifact chain: {checkpoint}"
                    )
    return candidates[0][1]


def _route_average(
    config_path: Path,
    runs: list[Path],
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if not runs:
        raise ValueError("A routed family needs at least one all-labeled refit")
    config = load_response_config(config_path)
    device = torch.device(config.model.device)
    total: np.ndarray | None = None
    proteins: list[str] | None = None
    records: list[dict[str, object]] = []
    for run in runs:
        checkpoint = run / "checkpoint.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("fit_scope") != "all_released_labeled_rows":
            raise ValueError(f"Routed checkpoint is not an all-labeled refit: {checkpoint}")
        if not isinstance(payload.get("support_manifest"), dict):
            raise ValueError(
                f"Every routed checkpoint must embed a support manifest: {checkpoint}"
            )
        if not isinstance(payload.get("artifact_hashes"), dict):
            raise ValueError(
                f"Every routed checkpoint must carry an artifact chain: {checkpoint}"
            )
        model, builder, current_proteins, mean, scale = load_response_checkpoint(
            checkpoint,
            device,
            config,
        )
        if proteins is None:
            proteins = current_proteins
        elif current_proteins != proteins:
            raise ValueError("Routed seed checkpoints have different protein contracts")
        prediction = _predict(
            model,
            builder,
            metadata,
            metadata.index,
            current_proteins,
            mean,
            scale,
            device,
            config.model.batch_size,
        ).to_numpy(dtype=np.float64)
        total = prediction if total is None else total + prediction
        records.append(
            {
                "run": str(run.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "fit_scope": payload.get("fit_scope"),
                "fit_sample_count": int(payload.get("fit_sample_count", 0)),
                "config": str(config_path),
                "config_sha256": _sha256(config_path),
                "support_manifest_sha256": payload.get("support_manifest_sha256"),
                "artifact_chain_sha256": payload.get("artifact_chain_sha256"),
            }
        )
        del model, prediction
    assert total is not None and proteins is not None
    frame = pd.DataFrame(total / len(runs), index=metadata.index, columns=proteins)
    return frame, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-prediction", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--route-manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    base_path = Path(args.base_prediction).resolve()
    metadata_path = Path(args.metadata).resolve()
    route_path = Path(args.route_manifest).resolve()
    output_path = Path(args.output_csv).resolve()
    with route_path.open("r", encoding="utf-8") as handle:
        route_manifest = json.load(handle)
    routes = route_manifest.get("routes")
    if not isinstance(routes, dict) or not routes:
        raise ValueError("route manifest must contain a non-empty routes object")

    metadata = pd.read_csv(metadata_path, low_memory=False)
    require_metadata_columns(metadata)
    require_unique_sample_ids(metadata, "test metadata")
    metadata = metadata.set_index(SAMPLE_ID, verify_integrity=True)
    base = pd.read_csv(base_path).set_index(SAMPLE_ID, verify_integrity=True)
    if not base.index.equals(metadata.index):
        raise ValueError("Base prediction sample order does not match test metadata")
    values = base.to_numpy(dtype=np.float64, copy=True)
    route_records: dict[str, object] = {}
    used_rows = np.zeros(len(metadata), dtype=bool)
    support_keys = set(routes) & set(SUPPORT_REGIMES)
    fit_support = _fit_support_manifest(route_manifest, route_path, routes) if support_keys else None
    support_manifest_hash: str | None = None
    if fit_support is None:
        route_audit = pd.DataFrame(
            {
                SAMPLE_ID: metadata.index.astype(str),
                SPLIT: metadata[SPLIT].astype(str).to_numpy(),
                "is_treatment": treatment_mask(metadata).to_numpy(dtype=bool),
                "support_regime": "legacy_split_route",
            },
            index=metadata.index,
        )
    else:
        route_audit = support_route_audit(metadata, fit_support)
        support_manifest_hash = str(route_audit["fit_support_manifest_sha256"].iloc[0])
    route_audit["selected_route"] = "base"
    route_audit["route_source"] = "base"
    route_audit["route_weight"] = 0.0
    for route_name, spec in routes.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid route spec for {route_name}")
        if route_name in SUPPORT_REGIMES:
            row_mask = route_audit["support_regime"].astype(str).eq(str(route_name)).to_numpy()
            route_source = "fit_support"
        else:
            row_mask = metadata[SPLIT].astype(str).eq(str(route_name)).to_numpy()
            route_source = "legacy_split"
        if not row_mask.any():
            raise ValueError(f"Route has no test rows: {route_name}")
        if np.any(used_rows & row_mask):
            raise ValueError(f"Overlapping routed test rows: {route_name}")
        used_rows |= row_mask
        route_metadata = metadata.loc[row_mask]
        weight = float(spec.get("weight", 1.0))
        if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError(f"Route weight must be in [0,1]: {route_name}")
        raw_components = _components(spec)
        component_weights = np.asarray(
            [float(component.get("weight", 1.0)) for component in raw_components],
            dtype=np.float64,
        )
        if (
            not np.isfinite(component_weights).all()
            or np.any(component_weights < 0.0)
            or not np.isclose(component_weights.sum(), 1.0, atol=1e-8)
        ):
            raise ValueError(f"Route component weights must be finite, non-negative, and sum to one: {route_name}")
        routed_values: np.ndarray | None = None
        component_records: list[dict[str, object]] = []
        for component, component_weight in zip(raw_components, component_weights):
            if not isinstance(component, dict):
                raise ValueError(f"Invalid route component for {route_name}")
            config_path = Path(component["config"]).resolve()
            runs = [Path(item).resolve() for item in component["runs"]]
            routed, checkpoint_records = _route_average(config_path, runs, route_metadata)
            if routed.columns.tolist() != base.columns.tolist():
                raise ValueError(f"Route protein contract differs from base: {route_name}")
            current = float(component_weight) * routed.to_numpy(dtype=np.float64)
            routed_values = current if routed_values is None else routed_values + current
            component_records.append(
                {
                    "weight": float(component_weight),
                    "config": str(config_path),
                    "config_sha256": _sha256(config_path),
                    "checkpoints": checkpoint_records,
                }
            )
        assert routed_values is not None
        values[row_mask] = (1.0 - weight) * values[row_mask] + weight * routed_values
        route_audit.loc[row_mask, "selected_route"] = str(route_name)
        route_audit.loc[row_mask, "route_source"] = route_source
        route_audit.loc[row_mask, "route_weight"] = weight
        route_records[str(route_name)] = {
            "rows": int(row_mask.sum()),
            "route_weight": weight,
            "components": component_records,
        }

    if not np.isfinite(values).all():
        raise ValueError("Routed ensemble produced NaN or infinity")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(values.astype(np.float32), index=metadata.index, columns=base.columns)
    result.index.name = SAMPLE_ID
    result.to_csv(output_path)
    route_audit.reset_index(drop=True).to_csv(output_path.parent / "route_audit.csv", index=False)
    report = verify_submission(output_path, metadata_path, base.columns.tolist())
    report.update(
        {
            "protocol": "support_routed_seed_ensemble_v3" if support_keys else "scenario_routed_seed_ensemble_v2",
            "model": route_manifest.get("model", "unregistered_candidate"),
            "base_prediction": str(base_path),
            "base_prediction_sha256": _sha256(base_path),
            "route_manifest": str(route_path),
            "route_manifest_sha256": _sha256(route_path),
            "routes": route_records,
            "fit_support_routing": bool(support_keys),
            "fit_support_manifest_sha256": support_manifest_hash,
            "support_route_counts": route_audit["support_regime"].value_counts().sort_index().to_dict(),
            "control_route": CONTROL_ROUTE,
            "route_audit": str((output_path.parent / "route_audit.csv").resolve()),
            "unrouted_rows_from_base": int((~used_rows).sum()),
            "prediction_sha256": _sha256(output_path),
            "official_submission_not_performed": True,
        }
    )
    with (output_path.parent / "prediction_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
