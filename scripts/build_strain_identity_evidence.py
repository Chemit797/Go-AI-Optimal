#!/usr/bin/env python3
"""Build the small, reviewable GOAI strain-identity evidence snapshots.

Network access is deliberately outside the registry builder.  This script
downloads only the public records required by the six-strain audit, writes
content hashes, and never promotes a competition code to a public isolate.
Promotion is a separate, organizer-evidence decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENA_PROJECT = "ERP014555"
ENA_API = "https://www.ebi.ac.uk/ena/portal/api"
TARGET_ENA_SAMPLES = {
    "SX3": "SAMEA3895227",
    "BJ6": "SAMEA3895228",
    "JCM_2985-4B": "SAMEA3895619",
    "UCD_09-448": "SAMEA3895648",
    "FIMA_3": "SAMEA3895807",
}
TARGET_NCBI_ASSEMBLIES = {
    "GCA_003276965.1",  # BJ6
    "GCA_003277085.1",  # SX3
    "GCA_947370195.1",  # BAH/SX3
    "GCA_947370255.1",  # BAI/BJ6
    "GCA_949124515.1",  # BAI/BJ6 ScRAP
    "GCA_949124635.1",  # BAH/SX3 ScRAP
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "goai-strain-audit/1.0"})
    last_error: Exception | None = None
    for _ in range(5):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:  # pragma: no cover - transient network path
            last_error = error
    assert last_error is not None
    raise last_error


def build_ena(output_dir: Path) -> None:
    fields = (
        "study_accession,sample_accession,sample_alias,sample_title,"
        "scientific_name,run_accession,experiment_accession"
    )
    query = urllib.parse.urlencode(
        {
            "accession": ENA_PROJECT,
            "result": "read_run",
            "fields": fields,
            "format": "tsv",
            "download": "true",
        }
    )
    project_url = f"{ENA_API}/filereport?{query}"
    rows = list(csv.DictReader(fetch(project_url).decode("utf-8").splitlines(), delimiter="\t"))
    selected = {
        row["sample_title"].rsplit("sample ", 1)[-1]: row
        for row in rows
        if row["sample_title"].rsplit("sample ", 1)[-1] in TARGET_ENA_SAMPLES
    }
    if set(selected) != set(TARGET_ENA_SAMPLES):
        raise ValueError(f"ENA target mismatch: {sorted(selected)}")
    output = output_dir / "ena_erp014555_target_runs.tsv"
    fieldnames = list(rows[0])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected[key] for key in sorted(selected))

    for isolate, accession in TARGET_ENA_SAMPLES.items():
        payload = fetch(f"https://www.ebi.ac.uk/ena/browser/api/xml/{accession}")
        root = ET.fromstring(payload)
        if root.findtext(".//PRIMARY_ID") != accession:
            raise ValueError(f"ENA sample identity mismatch for {isolate}/{accession}")
        (output_dir / f"ena_{accession}.xml").write_bytes(payload)


def build_ncbi(raw_pages: list[Path], output_dir: Path) -> None:
    reports: list[dict[str, object]] = []
    for index, source in enumerate(raw_pages, start=1):
        target = output_dir / f"ncbi_datasets_taxon4932_page{index}_raw.json"
        shutil.copyfile(source, target)
        reports.extend(json.loads(source.read_text(encoding="utf-8"))["reports"])
    selected = [row for row in reports if row.get("accession") in TARGET_NCBI_ASSEMBLIES]
    if {str(row["accession"]) for row in selected} != TARGET_NCBI_ASSEMBLIES:
        raise ValueError("NCBI Datasets target-assembly set is incomplete")
    payload = {
        "schema_version": "goai.ncbi-strain-evidence.v1",
        "retrieved_utc": "2026-08-13",
        "source": "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/taxon/4932/dataset_report",
        "records": sorted(selected, key=lambda row: str(row["accession"])),
    }
    (output_dir / "ncbi_datasets_target_assemblies.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data/external/strain_identity_evidence",
    )
    parser.add_argument("--ncbi-page", action="append", type=Path, default=[])
    parser.add_argument("--skip-ena", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_ena:
        build_ena(args.output_dir)
    if args.ncbi_page:
        build_ncbi(args.ncbi_page, args.output_dir)
    files = sorted(
        path for path in args.output_dir.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema_version": "goai.strain-identity-evidence.v1",
        "retrieved_utc": "2026-08-13",
        "files": {path.name: sha256(path) for path in files},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
