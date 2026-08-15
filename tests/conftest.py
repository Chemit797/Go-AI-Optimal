from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


METADATA_COLUMNS = [
    "sample_ID",
    "data_source",
    "Strains",
    "Medium",
    "Temperature",
    "pert_time",
    "pert_time_unit",
    "pert_id",
    "perturbation_no_concentration",
    "instrument",
    "Yeast_cell_plate",
    "protein_well",
    "split_final",
    "strain_role",
    "chemical_role",
]


def metadata_row(sample_id: str, chemical: str, split: str, strain: str = "S1", time: int = 15) -> dict[str, object]:
    role = "train" if split == "train" else "val"
    chemical_role = "train" if chemical in {"Water", "DrugA"} else "val"
    return {
        "sample_ID": sample_id,
        "data_source": "D1",
        "Strains": strain,
        "Medium": "M1",
        "Temperature": 30,
        "pert_time": time,
        "pert_time_unit": "min",
        "pert_id": f"#{sample_id}",
        "perturbation_no_concentration": chemical,
        "instrument": "I1",
        "Yeast_cell_plate": "P1",
        "protein_well": "A1",
        "split_final": split,
        "strain_role": role,
        "chemical_role": chemical_role,
    }


def make_tiny_files(root: Path) -> tuple[Path, Path, Path]:
    train_rows = [
        metadata_row("tr_ctrl", "Water", "train"),
        metadata_row("tr_a", "DrugA", "train"),
        metadata_row("tr_b", "DrugB", "train", strain="S2", time=30),
        metadata_row("val_chem", "DrugC", "val_chem_only"),
        metadata_row("val_strain", "DrugA", "val_strain_only", strain="S3"),
        metadata_row("val_both", "DrugC", "val_both", strain="S3"),
        metadata_row("val_time", "DrugA", "val_time", time=60),
    ]
    metadata = pd.DataFrame(train_rows, columns=METADATA_COLUMNS)
    proteome = pd.DataFrame(
        {
            "sample_ID": metadata["sample_ID"],
            "P1": [8.0, 16.0, 32.0, 12.0, 14.0, 11.0, 15.0],
            "P2": [np.nan, 4.0, 8.0, 6.0, 7.0, 5.0, 9.0],
            "P3": [np.nan, np.nan, np.nan, 2.0, 2.0, 2.0, 2.0],
        }
    )
    test_metadata = pd.DataFrame(
        [
            metadata_row("test_1", "DrugZ", "test_chem_only", strain="S1"),
            metadata_row("test_2", "DrugA", "test_strain_only", strain="S9"),
        ],
        columns=METADATA_COLUMNS,
    )
    train_path = root / "metadata_train_val.csv"
    protein_path = root / "proteome_train_val.csv"
    test_path = root / "WAYB_WAYC_metadata_test.csv"
    metadata.to_csv(train_path, index=False)
    proteome.to_csv(protein_path, index=False)
    test_metadata.to_csv(test_path, index=False)
    return train_path, protein_path, test_path


def write_config(root: Path, epochs: int = 2) -> Path:
    train_path, protein_path, test_path = make_tiny_files(root)
    config = {
        "data": {
            "metadata_train_val": train_path.name,
            "proteome_train_val": protein_path.name,
            "metadata_test": test_path.name,
            "missing_rate_threshold": 0.80,
        },
        "model": {
            "hidden_dim": 8,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": epochs,
            "seed": 42,
            "device": "cpu",
        },
        "features": {"chemical_hash_dim": 8},
        "runtime": {"runs_dir": "runs"},
    }
    path = root / "baseline.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return path
