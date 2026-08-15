"""Build auditable GOAI strain semantics from the Peter et al. 1,011-isolate resources.

The script deliberately keeps identity/provenance rows separate from the purely
numeric matrix consumed by the response model.  Five GOAI codes exactly match
standardised isolate codes in Peter et al. (2018); they remain labelled as
high-confidence candidates rather than organizer-verified identities.  DHY210
is never substituted with S288C and receives an explicit missing view.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd


GOAI_STRAINS = ("BAH", "BAI", "CEK", "CGD", "CRD", "DHY210")
CANDIDATE_ISOLATES = {
    "BAH": "SX3",
    "BAI": "BJ6",
    "CEK": "JCM_2985-4B",
    "CGD": "UCD_09-448",
    "CRD": "FIMA_3",
}
PETER_TABLE_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fs41586-018-0030-5/"
    "MediaObjects/41586_2018_30_MOESM3_ESM.xls"
)
PETER_DISTANCE_URL = "http://1002genomes.u-strasbg.fr/files/1011DistanceMatrixBasedOnSNPs.tab.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")
    return text or "unknown"


def load_peter_table(path: Path) -> pd.DataFrame:
    """Read Table S1 with the old-XLS reader only at artifact-build time."""
    try:
        import xlrd
    except ImportError as error:  # pragma: no cover - environment-specific message
        raise RuntimeError(
            "Building strain semantics requires xlrd>=2.0; install the optional entity dependency"
        ) from error
    workbook = xlrd.open_workbook(str(path))
    sheet = workbook.sheet_by_name("Table S1")
    header = [str(sheet.cell_value(3, column)).strip() for column in range(sheet.ncols)]
    rows = [
        [sheet.cell_value(row, column) for column in range(sheet.ncols)]
        for row in range(4, sheet.nrows)
    ]
    table = pd.DataFrame(rows, columns=header)
    table["Standardized name"] = table["Standardized name"].astype(str).str.strip()
    table["Isolate name"] = table["Isolate name"].astype(str).str.strip()
    table = table.loc[table["Standardized name"].ne("")].copy()
    if table["Standardized name"].duplicated().any():
        duplicates = table.loc[table["Standardized name"].duplicated(), "Standardized name"].tolist()
        raise ValueError(f"Peter Table S1 has duplicate standardised codes: {duplicates[:5]}")
    return table.set_index("Standardized name", verify_integrity=True)


def load_distance_matrix(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n\r").split("\t")
    if not header or header[0] != "STD":
        raise ValueError("SNP distance matrix must start with an STD header")
    matrix = pd.read_csv(path, sep="\t", compression="infer", index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    if matrix.index.tolist() != matrix.columns.tolist():
        raise ValueError("SNP distance matrix row and column contracts differ")
    values = matrix.to_numpy(dtype=np.float64)
    if values.shape[0] != values.shape[1] or not np.isfinite(values).all():
        raise ValueError("SNP distance matrix is not finite and square")
    if not np.allclose(values, values.T, atol=1e-7) or not np.allclose(np.diag(values), 0.0):
        raise ValueError("SNP distance matrix is not symmetric with a zero diagonal")
    return matrix


def classical_mds(distance: np.ndarray, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic classical-MDS coordinates and retained eigenvalues."""
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    distance = np.asarray(distance, dtype=np.float64)
    squared = distance.square() if hasattr(distance, "square") else np.square(distance)
    gram = -0.5 * (
        squared
        - squared.mean(axis=1, keepdims=True)
        - squared.mean(axis=0, keepdims=True)
        + squared.mean()
    )
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > max(float(eigenvalues[0]) * 1e-12, 1e-12)
    retained = min(dimensions, int(positive.sum()))
    coordinates = np.zeros((distance.shape[0], dimensions), dtype=np.float64)
    if retained:
        values = eigenvalues[:retained]
        vectors = eigenvectors[:, :retained]
        # Eigenvector signs are arbitrary.  Anchor each dimension at its
        # largest-magnitude element so repeated builds have identical signs.
        anchors = np.argmax(np.abs(vectors), axis=0)
        signs = np.sign(vectors[anchors, np.arange(retained)])
        signs[signs == 0] = 1.0
        coordinates[:, :retained] = vectors * signs * np.sqrt(values)[None, :]
    return coordinates, eigenvalues[:retained]


def _one_hot_rows(records: pd.DataFrame, field: str, prefix: str) -> dict[str, np.ndarray]:
    categories = sorted({_slug(value) for value in records[field]})
    result: dict[str, np.ndarray] = {}
    for code, value in records[field].items():
        current = _slug(value)
        result[str(code)] = np.asarray([float(current == category) for category in categories], dtype=np.float32)
    result["__columns__"] = np.asarray([f"{prefix}_{category}" for category in categories], dtype=object)
    return result


def build_outputs(
    table_path: Path,
    distance_path: Path,
    dimensions: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    table = load_peter_table(table_path)
    distances = load_distance_matrix(distance_path)
    missing_codes = sorted(set(CANDIDATE_ISOLATES) - set(table.index))
    missing_distances = sorted(set(CANDIDATE_ISOLATES) - set(distances.index))
    if missing_codes or missing_distances:
        raise ValueError(
            f"Candidate codes missing from sources: table={missing_codes}, distance={missing_distances}"
        )
    for code, isolate in CANDIDATE_ISOLATES.items():
        observed = str(table.loc[code, "Isolate name"]).strip()
        if observed != isolate:
            raise ValueError(f"Candidate isolate mismatch for {code}: expected {isolate}, found {observed}")

    coordinates, eigenvalues = classical_mds(distances.to_numpy(dtype=np.float64), dimensions)
    coordinate_frame = pd.DataFrame(
        coordinates,
        index=distances.index,
        columns=[f"snp_mds_{index + 1:03d}" for index in range(dimensions)],
    )
    candidate_records = table.loc[list(CANDIDATE_ISOLATES)].copy()
    clade_hot = _one_hot_rows(candidate_records, "Clades", "clade")
    ecology_hot = _one_hot_rows(candidate_records, "Ecological origins", "ecology")
    geography_hot = _one_hot_rows(candidate_records, "Geographical origins", "geography")
    category_columns = [
        *clade_hot.pop("__columns__").tolist(),
        *ecology_hot.pop("__columns__").tolist(),
        *geography_hot.pop("__columns__").tolist(),
    ]

    registry_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for code in GOAI_STRAINS:
        if code in CANDIDATE_ISOLATES:
            row = table.loc[code]
            registry_rows.append(
                {
                    "strain_code": code,
                    "canonical_strain_id": f"peter2018:{code}",
                    "isolate_name": str(row["Isolate name"]).strip(),
                    "standardized_name": code,
                    "isolation": str(row["Isolation"]).strip(),
                    "ecological_origin": str(row["Ecological origins"]).strip(),
                    "geographical_origin": str(row["Geographical origins"]).strip(),
                    "ploidy": float(row["Ploidy"]),
                    "aneuploidy": str(row["Aneuploidies"]).strip(),
                    "zygosity": str(row["Zygosity"]).strip(),
                    "clade": str(row["Clades"]).strip(),
                    "evidence_tier": "HIGH_CONFIDENCE_CANDIDATE",
                    "identity_status": "candidate",
                    "is_proxy": False,
                    "source_table_url": PETER_TABLE_URL,
                    "distance_matrix_url": PETER_DISTANCE_URL,
                    "source_table_sha256": sha256(table_path),
                    "distance_matrix_sha256": sha256(distance_path),
                    "notes": "GOAI code string exactly matches the Peter et al. standardized isolate code; organizer identity not asserted",
                }
            )
            values: dict[str, object] = {
                "strain_code": code,
                **coordinate_frame.loc[code].to_dict(),
                "ploidy": float(row["Ploidy"]),
                "heterozygous_snp_fraction": float(row["Proportion of clean heterozygous SNPs (whole dataset)"]),
                "aneuploid": float(str(row["Aneuploidies"]).strip().casefold() != "euploid"),
                "homozygous": float(str(row["Zygosity"]).strip().casefold() == "homozygous"),
                "total_snps": float(row["Total number of SNPs"]),
                "singletons": float(row["Number of singletons"]),
                "mean_coverage": float(row["Mean coverage"]),
            }
            category_values = np.concatenate([clade_hot[code], ecology_hot[code], geography_hot[code]])
            values.update(dict(zip(category_columns, category_values.tolist())))
            values.update({"resolved": 1.0, "missing": 0.0, "proxy": 0.0})
            feature_rows.append(values)
        else:
            registry_rows.append(
                {
                    "strain_code": code,
                    "canonical_strain_id": f"goai:{code}",
                    "isolate_name": "",
                    "standardized_name": "",
                    "isolation": "",
                    "ecological_origin": "",
                    "geographical_origin": "",
                    "ploidy": "",
                    "aneuploidy": "",
                    "zygosity": "",
                    "clade": "",
                    "evidence_tier": "UNRESOLVED",
                    "identity_status": "missing",
                    "is_proxy": False,
                    "source_table_url": PETER_TABLE_URL,
                    "distance_matrix_url": PETER_DISTANCE_URL,
                    "source_table_sha256": sha256(table_path),
                    "distance_matrix_sha256": sha256(distance_path),
                    "notes": "No Peter 2018 standardized-code match; S288C substitution is forbidden",
                }
            )
            zero_columns = [*coordinate_frame.columns, "ploidy", "heterozygous_snp_fraction", "aneuploid", "homozygous", "total_snps", "singletons", "mean_coverage", *category_columns]
            values = {"strain_code": code, **{column: 0.0 for column in zero_columns}}
            values.update({"resolved": 0.0, "missing": 1.0, "proxy": 0.0})
            feature_rows.append(values)

    registry = pd.DataFrame(registry_rows)
    features = pd.DataFrame(feature_rows)
    numeric = features.drop(columns=["strain_code"]).to_numpy(dtype=np.float64)
    if registry["strain_code"].tolist() != list(GOAI_STRAINS):
        raise AssertionError("Registry GOAI strain order changed")
    if not np.isfinite(numeric).all() or features["strain_code"].duplicated().any():
        raise AssertionError("Numeric strain feature contract is invalid")
    manifest = {
        "protocol": "goai_peter2018_strain_semantics_v1",
        "dimensions": dimensions,
        "target_strains": list(GOAI_STRAINS),
        "candidate_strains": sorted(CANDIDATE_ISOLATES),
        "unresolved_strains": ["DHY210"],
        "feature_columns": features.columns[1:].tolist(),
        "positive_eigenvalues_retained": len(eigenvalues),
        "retained_eigenvalues": eigenvalues.tolist(),
        "source_table": str(table_path.resolve()),
        "source_table_url": PETER_TABLE_URL,
        "source_table_sha256": sha256(table_path),
        "distance_matrix": str(distance_path.resolve()),
        "distance_matrix_url": PETER_DISTANCE_URL,
        "distance_matrix_sha256": sha256(distance_path),
        "identity_evidence_registry": str(
            (Path(__file__).resolve().parents[1] / "resources/entities/strain_identity_candidates.tsv").resolve()
        ),
        "identity_evidence_registry_sha256": sha256(
            Path(__file__).resolve().parents[1] / "resources/entities/strain_identity_candidates.tsv"
        ),
        "identity_evidence_manifest": str(
            (Path(__file__).resolve().parents[1] / "data/external/strain_identity_evidence/manifest.json").resolve()
        ),
        "identity_evidence_manifest_sha256": sha256(
            Path(__file__).resolve().parents[1] / "data/external/strain_identity_evidence/manifest.json"
        ),
        "identity_warning": "Five mappings are high-confidence public candidates, not organizer-verified identities; DHY210 is missing, never S288C",
    }
    return registry, features, manifest


def write_outputs(
    registry: pd.DataFrame,
    features: pd.DataFrame,
    manifest: dict[str, object],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "strain_identity_registry.tsv"
    features_path = output_dir / "strain_semantics_numeric.tsv"
    shuffled_path = output_dir / "strain_semantics_shuffled.tsv"
    manifest_path = output_dir / "strain_semantics_manifest.json"
    registry.to_csv(registry_path, sep="\t", index=False)
    features.to_csv(features_path, sep="\t", index=False, float_format="%.10g")
    shuffled = features.copy()
    semantic_columns = [
        column for column in features.columns
        if column not in {"strain_code", "resolved", "missing", "proxy"}
    ]
    resolved_rows = np.flatnonzero(features["resolved"].to_numpy(dtype=float) > 0.5)
    rng = np.random.default_rng(991)
    # A derangement is a stronger negative control than a raw permutation,
    # which can accidentally leave one of only five strains unchanged.
    shift = int(rng.integers(1, len(resolved_rows)))
    permutation = np.roll(resolved_rows, shift)
    shuffled.loc[resolved_rows, semantic_columns] = features.loc[permutation, semantic_columns].to_numpy()
    shuffled.to_csv(shuffled_path, sep="\t", index=False, float_format="%.10g")
    manifest.update(
        {
            "real_path": str(features_path.resolve()),
            "real_sha256": sha256(features_path),
            "shuffled_path": str(shuffled_path.resolve()),
            "shuffled_sha256": sha256(shuffled_path),
            "shuffle_seed": 991,
            "shuffle_rotation": shift,
            "resolved_row_permutation": permutation.tolist(),
        }
    )
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peter-table", required=True, type=Path)
    parser.add_argument("--distance-matrix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dimensions", type=int, default=32)
    args = parser.parse_args()
    registry, features, manifest = build_outputs(
        args.peter_table.resolve(), args.distance_matrix.resolve(), args.dimensions
    )
    write_outputs(registry, features, manifest, args.output_dir.resolve())
    print(json.dumps({**manifest, "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
