from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import load_config, morgan_fingerprint, sha256, write_json


SAMPLE_ID = "sample_ID"
CHEMICAL = "perturbation_no_concentration"
SPLIT = "split_final"
MATCH_FIELDS = (
    "data_source",
    "instrument",
    "Yeast_cell_plate",
    "Strains",
    "Medium",
    "Temperature",
    "pert_time",
    "pert_time_unit",
)
CONTROLS = frozenset({"water", "dmso"})
QUALITY_CONTROL = "quality control"


def normalized_name(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.casefold()


def fit_context_vocab(metadata: pd.DataFrame, fields: list[str]) -> dict[str, list[str]]:
    return {field: sorted(metadata[field].astype(str).unique()) for field in fields}


def encode_context(metadata: pd.DataFrame, fields: list[str], vocabulary: dict[str, list[str]]) -> np.ndarray:
    columns: list[np.ndarray] = []
    for field in fields:
        lookup = {value: index + 1 for index, value in enumerate(vocabulary[field])}
        columns.append(metadata[field].astype(str).map(lookup).fillna(0).to_numpy(dtype=np.int64))
    return np.column_stack(columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    metadata_path = Path(paths["goai_metadata"])
    proteome_path = Path(paths["goai_proteome"])
    chemical_path = Path(paths["goai_chemical_map"])
    folds_path = Path(paths["goai_s1_folds"])
    cache = Path(paths["private_cache"])
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)

    metadata = pd.read_csv(metadata_path, keep_default_na=False).set_index(SAMPLE_ID, drop=False)
    raw = pd.read_csv(proteome_path).set_index(SAMPLE_ID)
    raw = raw.reindex(metadata.index)
    train_ids = metadata.index[metadata[SPLIT].eq("train")]
    missing = raw.loc[train_ids].isna().mean(axis=0)
    proteins = missing.index[missing.lt(0.80)].astype(str).to_numpy()
    selected = raw.loc[:, proteins].to_numpy(dtype=np.float64)
    mask = np.isfinite(selected) & (selected > 0)
    log2 = np.full(selected.shape, np.nan, dtype=np.float32)
    log2[mask] = np.log2(selected[mask]).astype(np.float32)
    del raw, selected

    train_metadata = metadata.loc[train_ids]
    names = normalized_name(train_metadata[CHEMICAL])
    treatment_ids = train_ids[~names.isin(CONTROLS | {QUALITY_CONTROL}).to_numpy()]
    control_ids = train_ids[names.isin(CONTROLS).to_numpy()]
    control_frame = metadata.loc[control_ids, list(MATCH_FIELDS)].astype(str)
    control_keys = pd.MultiIndex.from_frame(control_frame)
    control_values = pd.DataFrame(log2[metadata.index.get_indexer(control_ids)], index=control_keys, columns=proteins)
    control_mean = control_values.groupby(level=list(range(len(MATCH_FIELDS))), sort=False).mean()
    treatment_keys = pd.MultiIndex.from_frame(metadata.loc[treatment_ids, list(MATCH_FIELDS)].astype(str))
    matched = control_mean.reindex(treatment_keys)
    matched.index = treatment_ids
    treatment_y = pd.DataFrame(log2[metadata.index.get_indexer(treatment_ids)], index=treatment_ids, columns=proteins)
    delta = (treatment_y - matched).to_numpy(dtype=np.float32)
    delta_mask = np.isfinite(delta)

    fold_table = pd.read_csv(folds_path, keep_default_na=False)
    eligible = fold_table["eligible"]
    if eligible.dtype != bool:
        eligible = eligible.astype(str).str.strip().str.casefold().eq("true")
    fold_table = fold_table[(fold_table["scenario"] == "S1") & eligible]
    fold_lookup = fold_table.set_index(SAMPLE_ID)["fold"]
    folds = fold_lookup.reindex(treatment_ids).to_numpy()
    if pd.isna(folds).any():
        missing_ids = treatment_ids[pd.isna(folds)][:10].tolist()
        raise ValueError(f"Missing S1 fold assignments: {missing_ids}")
    folds = folds.astype(np.int64)

    chemical_map = pd.read_csv(chemical_path, sep="\t", keep_default_na=False).set_index("raw_name")
    smiles_by_name = chemical_map["canonical_smiles"].to_dict()
    unique_names = sorted(metadata.loc[treatment_ids, CHEMICAL].unique())
    invalid = [name for name in unique_names if not smiles_by_name.get(name)]
    if invalid:
        raise ValueError(f"Missing GOAI chemical structures: {invalid}")
    fp_by_name = {
        name: morgan_fingerprint(
            smiles_by_name[name], int(config["fingerprint"]["radius"]), int(config["fingerprint"]["n_bits"])
        )
        for name in unique_names
    }
    fingerprints = np.stack(metadata.loc[treatment_ids, CHEMICAL].map(fp_by_name).to_numpy()).astype(np.float32)
    fields = list(config["goai_s1"]["context_fields"])
    vocabulary = fit_context_vocab(metadata.loc[train_ids], fields)
    contexts = encode_context(metadata.loc[treatment_ids], fields, vocabulary)
    context_keys = np.asarray(
        metadata.loc[treatment_ids, list(MATCH_FIELDS)]
        .astype(str)
        .agg("\x1f".join, axis=1)
        .tolist(),
        dtype=str,
    )
    matched_values = matched.to_numpy(dtype=np.float32)
    treatment_values = treatment_y.to_numpy(dtype=np.float32)

    output = cache / "goai_s1_delta.npz"
    np.savez_compressed(
        output,
        sample_ids=np.asarray(treatment_ids.astype(str).tolist(), dtype=str),
        chemicals=np.asarray(
            metadata.loc[treatment_ids, CHEMICAL].astype(str).tolist(), dtype=str
        ),
        fingerprints=fingerprints,
        contexts=contexts,
        delta=np.nan_to_num(delta, nan=0.0),
        mask=delta_mask,
        folds=folds,
        proteins=np.asarray(proteins.tolist(), dtype=str),
        context_keys=context_keys,
        context_cardinalities=np.asarray(
            [len(vocabulary[field]) + 1 for field in fields], dtype=np.int64
        ),
        treatment_truth=np.nan_to_num(treatment_values, nan=0.0),
        treatment_truth_mask=np.isfinite(treatment_values),
        matched_control=np.nan_to_num(matched_values, nan=0.0),
        matched_control_mask=np.isfinite(matched_values),
    )
    write_json(
        cache / "goai_s1_delta_manifest.json",
        {
            "status": "complete",
            "private_competition_derived": True,
            "must_not_copy_to_shared_mount": True,
            "metadata": str(metadata_path),
            "metadata_sha256": sha256(metadata_path),
            "proteome": str(proteome_path),
            "proteome_sha256": sha256(proteome_path),
            "chemical_map": str(chemical_path),
            "chemical_map_sha256": sha256(chemical_path),
            "fold_assignments": str(folds_path),
            "fold_assignments_sha256": sha256(folds_path),
            "n_treatments": len(treatment_ids),
            "n_exact_control": int(np.isfinite(matched.to_numpy()).any(axis=1).sum()),
            "n_proteins": len(proteins),
            "fold_counts": {str(key): int(value) for key, value in pd.Series(folds).value_counts().sort_index().items()},
            "context_fields": fields,
            "context_vocabulary": vocabulary,
            "output": str(output),
        },
    )
    print(json.dumps({"status": "complete", "output": str(output), "n_treatments": len(treatment_ids), "n_proteins": len(proteins)}, indent=2))


if __name__ == "__main__":
    main()
