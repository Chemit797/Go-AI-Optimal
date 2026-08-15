from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]


def test_exact_doxycycline_hyclate_and_parent_ablation_are_distinct():
    exact = pd.read_csv(
        ROOT / "data/processed/chemical_views/chemical_entity_map_exact.tsv",
        sep="\t", keep_default_na=False,
    ).set_index("raw_name")
    parent = pd.read_csv(
        ROOT / "data/processed/chemical_views/chemical_entity_map_parent_normalized.tsv",
        sep="\t", keep_default_na=False,
    ).set_index("raw_name")
    name = "Doxycycline hyclate"
    assert str(exact.loc[name, "cid"]) == "54705095"
    assert exact.loc[name, "inchikey"] == "DWBSXBGBGHKWOT-WBYAVNBMSA-N"
    assert str(parent.loc[name, "cid"]) == "54671203"
    assert parent.loc[name, "inchikey"] == "SGKRLCUYIXIAHR-AKNGSSGZSA-N"
    assert exact.loc[name, "isomeric_smiles"] != parent.loc[name, "isomeric_smiles"]
    assert Chem.MolFromSmiles(exact.loc[name, "isomeric_smiles"]) is not None
    assert Chem.MolFromSmiles(parent.loc[name, "isomeric_smiles"]) is not None


def test_ly_salt_is_exact_and_hoechst_rejects_ethoxy_analogue():
    exact = pd.read_csv(
        ROOT / "data/processed/chemical_views/chemical_entity_map_exact.tsv",
        sep="\t", keep_default_na=False,
    ).set_index("raw_name")
    parent = pd.read_csv(
        ROOT / "data/processed/chemical_views/chemical_entity_map_parent_normalized.tsv",
        sep="\t", keep_default_na=False,
    ).set_index("raw_name")

    ly = "LY 294002 hydrochloride"
    assert str(exact.loc[ly, "cid"]) == "11957589"
    assert exact.loc[ly, "inchikey"] == "OQZQSRICUOWBLW-UHFFFAOYSA-N"
    assert str(parent.loc[ly, "cid"]) == "3973"
    assert parent.loc[ly, "inchikey"] == "CZQHHVNHHHRRDU-UHFFFAOYSA-N"
    assert exact.loc[ly, "isomeric_smiles"] != parent.loc[ly, "isomeric_smiles"]

    hoechst = "Hoechst 33258"
    assert str(exact.loc[hoechst, "cid"]) == "2392"
    assert exact.loc[hoechst, "inchikey"] == "INAAIJLSXJJHOZ-UHFFFAOYSA-N"
    assert exact.loc[hoechst, "inchikey"] != "PRDFBSVERLRRMY-UHFFFAOYSA-N"


def test_zero_risky_view_never_contains_proxy_structure():
    zero = pd.read_csv(
        ROOT / "data/processed/chemical_views/chemical_entity_map_zero_risky.tsv",
        sep="\t", keep_default_na=False,
    ).set_index("raw_name")
    risk_table = pd.read_csv(
        ROOT / "resources/entities/chemical_identity_risk_review.tsv",
        sep="\t", keep_default_na=False,
    )
    risky = risk_table.loc[
        risk_table["zero_risky"].astype(str).str.casefold().eq("true"), "raw_name"
    ]
    assert set(risky) == {
        "Doxycycline hyclate",
        "G418",
        "Hoechst 33258",
        "Hygromycin B",
        "LY 294002 hydrochloride",
        "Oligomycin",
        "Tunicamycin",
    }
    assert zero.loc[risky, "status"].eq("unresolved").all()
    assert zero.loc[risky, "isomeric_smiles"].eq("").all()


def test_parent_and_identity_risk_contracts_are_not_conflated():
    parents = pd.read_csv(
        ROOT / "data/processed/entities/chemical_parent_normalized_views.tsv",
        sep="\t", keep_default_na=False,
    )["raw_name"]
    risk_table = pd.read_csv(
        ROOT / "resources/entities/chemical_identity_risk_review.tsv",
        sep="\t", keep_default_na=False,
    )
    risks = risk_table.loc[
        risk_table["zero_risky"].astype(str).str.casefold().eq("true"), "raw_name"
    ]
    assert len(parents) == 5
    assert len(risks) == 7
    assert set(risks) - set(parents) == {"G418", "Hygromycin B"}
