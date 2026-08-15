"""Auditable chemical identities, structural features, and node-target evidence.

The competition metadata contains display names rather than machine-readable
chemical identifiers.  This module resolves those names through PubChem and
stores the returned identifiers locally.  Missing or ambiguous records remain
explicitly unresolved; they are never substituted with a guessed molecule.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from goai_baseline.config import load_config
from goai_baseline.preprocess import prepare_data
from goai_baseline.schema import CHEMICAL


PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    "{name}/property/Title,IsomericSMILES,CanonicalSMILES,InChIKey/JSON"
)

# These are nomenclature normalisations only.  They deliberately do not encode
# target information or biological assumptions.
NAME_ALIASES = {
    "CHX": "Cycloheximide",
    "FCCP": "Carbonyl cyanide 4-(trifluoromethoxy)phenylhydrazone",
    "G418": "Geneticin",
    "H2O2": "Hydrogen peroxide",
    "MMS": "Methyl methanesulfonate",
    "1-10 Phenanthroline monohydrate": "1,10-Phenanthroline monohydrate",
    "LY 294002 hydrochloride": "LY294002",
    "(1R, 2S, 5R) - (-) - Menthol": "L-menthol",
    # PubChem has an exact formulation entry (CID 54705095).  Do not collapse
    # the hyclate/ethanol/hydrate salt to parent doxycycline.
    "Doxycycline hyclate": "Doxycycline hyclate",
    "Hoechst 33258": "Bisbenzimide",
    "SDS": "Sodium dodecyl sulfate",
    "Tunicamycin": "Tunicamycin A1",
    "U-73122": "U73122",
    "Oligomycin": "Oligomycin A",
}

CONTROL_NAMES = {"Water", "DMSO", "Quality Control"}


@dataclass(frozen=True)
class ChemicalFeatures:
    names: list[str]
    matrix: np.ndarray
    resolved: dict[str, bool]
    feature_names: list[str]


def _get_json(name: str) -> dict[str, object]:
    url = PUBCHEM_URL.format(name=urllib.parse.quote(name, safe=""))
    request = urllib.request.Request(url, headers={"User-Agent": "GOAI-AIVC/0.2"})
    with urllib.request.urlopen(request, timeout=8) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_compounds(names: list[str]) -> pd.DataFrame:
    unique_names = sorted(set(map(str, names)))

    def resolve_one(raw_name: str) -> dict[str, object]:
        query_name = NAME_ALIASES.get(raw_name, raw_name)
        row: dict[str, object] = {
            "raw_name": raw_name,
            "query_name": query_name,
            "is_control": raw_name in CONTROL_NAMES,
            "cid": "",
            "title": "",
            "isomeric_smiles": "",
            "canonical_smiles": "",
            "inchikey": "",
            "status": "unresolved",
            "error": "",
            "source": "PubChem PUG REST",
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            payload = _get_json(query_name)
            properties = payload["PropertyTable"]["Properties"]  # type: ignore[index]
            if len(properties) != 1:
                raise ValueError(f"expected one PubChem record, got {len(properties)}")
            record = properties[0]
            row.update(
                {
                    "cid": int(record["CID"]),
                    "title": str(record.get("Title", "")),
                    "isomeric_smiles": str(record.get("SMILES", record.get("IsomericSMILES", ""))),
                    "canonical_smiles": str(record.get("ConnectivitySMILES", record.get("CanonicalSMILES", ""))),
                    "inchikey": str(record.get("InChIKey", "")),
                    "status": "resolved",
                }
            )
        except Exception as error:  # network and name-resolution failures are data, not guesses
            row["error"] = f"{type(error).__name__}: {error}"
        return row

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(resolve_one, name): name for name in unique_names}
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows).sort_values("raw_name").reset_index(drop=True)


def build_entity_map(config_path: str | Path, output: str | Path) -> Path:
    config = load_config(config_path)
    data = prepare_data(config)
    metadata_test = pd.read_csv(config.data.metadata_test, low_memory=False)
    names = data.metadata[CHEMICAL].astype(str).tolist() + metadata_test[CHEMICAL].astype(str).tolist()
    mapping = resolve_compounds(names)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(destination, sep="\t", index=False)
    print(
        f"Resolved {(mapping.status == 'resolved').sum()}/{len(mapping)} compounds; "
        f"wrote {destination.resolve()}"
    )
    return destination


def morgan_features(mapping: pd.DataFrame, n_bits: int = 512, radius: int = 2) -> ChemicalFeatures:
    """Build RDKit Morgan bits plus compact physicochemical descriptors.

    Controls and unresolved identities produce explicit all-zero structure rows.
    A separate resolved flag lets the model distinguish this from a molecule
    whose valid descriptor happens to be zero.
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import Crippen, Descriptors, Lipinski
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    except ImportError as error:  # pragma: no cover - environment guard
        raise RuntimeError("RDKit is required: install with `python -m pip install rdkit`") from error

    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    descriptor_names = ["mol_wt", "logp", "tpsa", "h_donors", "h_acceptors", "rot_bonds", "ring_count"]
    matrix = np.zeros((len(mapping), n_bits + len(descriptor_names) + 2), dtype=np.float32)
    resolved: dict[str, bool] = {}
    for row_index, row in mapping.reset_index(drop=True).iterrows():
        name = str(row["raw_name"])
        smiles = str(row.get("isomeric_smiles", ""))
        molecule = Chem.MolFromSmiles(smiles) if row["status"] == "resolved" and smiles else None
        ok = molecule is not None and not bool(row["is_control"])
        resolved[name] = bool(ok)
        if not ok:
            continue
        fingerprint = generator.GetFingerprint(molecule)
        bits = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, bits)
        matrix[row_index, :n_bits] = bits
        matrix[row_index, n_bits : n_bits + len(descriptor_names)] = [
            Descriptors.MolWt(molecule),
            Crippen.MolLogP(molecule),
            Descriptors.TPSA(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            Lipinski.NumRotatableBonds(molecule),
            Lipinski.RingCount(molecule),
        ]
        matrix[row_index, -2] = 1.0  # valid chemical structure
        matrix[row_index, -1] = 0.0  # control indicator
    controls = mapping["is_control"].to_numpy(dtype=bool)
    matrix[controls, -1] = 1.0
    return ChemicalFeatures(
        names=mapping["raw_name"].astype(str).tolist(),
        matrix=matrix,
        resolved=resolved,
        feature_names=[f"morgan_{index}" for index in range(n_bits)] + descriptor_names + ["structure_resolved", "is_control"],
    )


def load_chemical_features(path: str | Path, n_bits: int = 512) -> ChemicalFeatures:
    mapping = pd.read_csv(path, sep="\t", keep_default_na=False)
    required = {"raw_name", "status", "is_control", "isomeric_smiles"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Chemical map is missing columns: {sorted(missing)}")
    return morgan_features(mapping, n_bits=n_bits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an auditable PubChem chemical identity table")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output", default="data/processed/chemical_entity_map.tsv")
    args = parser.parse_args()
    build_entity_map(args.config, args.output)


if __name__ == "__main__":
    main()
