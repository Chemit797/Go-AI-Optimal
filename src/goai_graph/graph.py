"""Protein identifier mapping, PPI graph construction, and graph controls."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
import shutil
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class SGDRecord:
    systematic_name: str
    standard_name: str
    sgd_id: str
    feature_type: str


@dataclass(frozen=True)
class GraphBundle:
    proteins: list[str]
    systematic_ids: list[str]
    edge_index: np.ndarray
    edge_weight: np.ndarray
    rewired_edge_index: np.ndarray
    rewired_edge_weight: np.ndarray
    min_score: int


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_if_missing(url: str, path: str | Path, attempts: int = 3) -> Path:
    destination = Path(path)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "GOAI-PPI-Graph/1.0"})
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if partial.stat().st_size == 0:
                raise RuntimeError(f"Downloaded empty resource from {url}")
            partial.replace(destination)
            return destination
        except Exception as error:  # pragma: no cover - exercised only on network failure
            last_error = error
            if partial.exists():
                partial.unlink()
            if attempt < attempts:
                time.sleep(float(attempt))
    raise RuntimeError(f"Failed to download {url} after {attempts} attempts") from last_error


def _unique_records(records: list[SGDRecord]) -> list[SGDRecord]:
    unique = {(record.systematic_name, record.sgd_id): record for record in records}
    return list(unique.values())


def map_proteins_to_sgd(proteins: list[str], sgd_features_path: str | Path) -> pd.DataFrame:
    by_systematic: dict[str, list[SGDRecord]] = defaultdict(list)
    by_standard: dict[str, list[SGDRecord]] = defaultdict(list)
    by_alias: dict[str, list[SGDRecord]] = defaultdict(list)
    with Path(sgd_features_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 6:
                continue
            sgd_id, feature_type, _, systematic, standard, aliases = row[:6]
            if not systematic:
                continue
            record = SGDRecord(systematic, standard, sgd_id, feature_type)
            by_systematic[systematic.casefold()].append(record)
            if standard:
                by_standard[standard.casefold()].append(record)
            for alias in aliases.split("|") if aliases else []:
                if alias.strip():
                    by_alias[alias.strip().casefold()].append(record)

    rows: list[dict[str, object]] = []
    for protein_index, raw_name in enumerate(proteins):
        candidates: list[SGDRecord] = []
        mapping_method = "unresolved"
        for method, lookup in (
            ("systematic", by_systematic),
            ("standard", by_standard),
            ("alias", by_alias),
        ):
            candidates = _unique_records(lookup.get(raw_name.casefold(), []))
            if candidates:
                mapping_method = method if len(candidates) == 1 else f"ambiguous_{method}"
                break
        record = candidates[0] if len(candidates) == 1 else None
        rows.append(
            {
                "protein_index": protein_index,
                "raw_name": raw_name,
                "systematic_name": "" if record is None else record.systematic_name,
                "standard_name": "" if record is None else record.standard_name,
                "sgd_id": "" if record is None else record.sgd_id,
                "feature_type": "" if record is None else record.feature_type,
                "mapping_method": mapping_method,
                "mapping_status": "mapped" if record is not None else mapping_method,
            }
        )
    return pd.DataFrame(rows)


def read_string_edges(
    path: str | Path,
    systematic_to_index: dict[str, int],
    min_score: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= min_score <= 1000:
        raise ValueError("STRING min_score must be in [0, 1000]")
    edges: dict[tuple[int, int], int] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().split()
        if header[:3] != ["protein1", "protein2", "combined_score"]:
            raise ValueError(f"Unexpected STRING header: {header}")
        for line in handle:
            protein_a, protein_b, score_text = line.split()
            score = int(score_text)
            if score < min_score:
                continue
            systematic_a = protein_a.split(".", 1)[-1]
            systematic_b = protein_b.split(".", 1)[-1]
            if systematic_a not in systematic_to_index or systematic_b not in systematic_to_index:
                continue
            a = systematic_to_index[systematic_a]
            b = systematic_to_index[systematic_b]
            if a == b:
                continue
            edge = (a, b) if a < b else (b, a)
            edges[edge] = max(score, edges.get(edge, 0))
    ordered = sorted(edges)
    edge_index = np.asarray(ordered, dtype=np.int64).T if ordered else np.empty((2, 0), dtype=np.int64)
    edge_weight = np.asarray([edges[edge] / 1000.0 for edge in ordered], dtype=np.float32)
    return edge_index, edge_weight


def degree_preserving_rewire(
    edge_index: np.ndarray,
    seed: int,
    swaps_per_edge: int,
) -> np.ndarray:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, n_edges]")
    if swaps_per_edge < 0:
        raise ValueError("swaps_per_edge must be non-negative")
    edges = [tuple(sorted((int(a), int(b)))) for a, b in edge_index.T]
    edge_set = set(edges)
    if len(edge_set) != len(edges):
        raise ValueError("edge_index contains duplicate undirected edges")
    rng = random.Random(seed)
    target_swaps = swaps_per_edge * len(edges)
    successful = 0
    attempts = 0
    max_attempts = max(100, target_swaps * 20)
    while successful < target_swaps and attempts < max_attempts and len(edges) >= 2:
        attempts += 1
        first, second = rng.sample(range(len(edges)), 2)
        a, b = edges[first]
        c, d = edges[second]
        if len({a, b, c, d}) < 4:
            continue
        if rng.random() < 0.5:
            candidate_one = tuple(sorted((a, d)))
            candidate_two = tuple(sorted((c, b)))
        else:
            candidate_one = tuple(sorted((a, c)))
            candidate_two = tuple(sorted((b, d)))
        if candidate_one == candidate_two or candidate_one in edge_set or candidate_two in edge_set:
            continue
        edge_set.remove(edges[first])
        edge_set.remove(edges[second])
        edges[first] = candidate_one
        edges[second] = candidate_two
        edge_set.add(candidate_one)
        edge_set.add(candidate_two)
        successful += 1
    if target_swaps and successful < target_swaps:
        raise RuntimeError(f"Only completed {successful:,}/{target_swaps:,} requested edge swaps")
    return np.asarray(sorted(edges), dtype=np.int64).T


def graph_statistics(n_nodes: int, edge_index: np.ndarray) -> dict[str, object]:
    adjacency: list[list[int]] = [[] for _ in range(n_nodes)]
    for a, b in edge_index.T:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    degrees = np.asarray([len(neighbors) for neighbors in adjacency], dtype=np.int64)
    covered = int(np.sum(degrees > 0))
    seen: set[int] = set()
    largest_component = 0
    for node in np.flatnonzero(degrees > 0):
        start = int(node)
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        largest_component = max(largest_component, size)
    nonzero = degrees[degrees > 0]
    return {
        "n_nodes": n_nodes,
        "n_edges": int(edge_index.shape[1]),
        "covered_nodes": covered,
        "coverage": covered / n_nodes if n_nodes else 0.0,
        "isolated_nodes": n_nodes - covered,
        "largest_connected_component": largest_component,
        "median_nonzero_degree": float(np.median(nonzero)) if len(nonzero) else 0.0,
        "max_degree": int(degrees.max()) if len(degrees) else 0,
    }


def save_graph_bundle(
    artifact_path: str | Path,
    proteins: list[str],
    mapping: pd.DataFrame,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    rewired_edge_index: np.ndarray,
    rewired_edge_weight: np.ndarray,
    min_score: int,
) -> None:
    output = Path(artifact_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    systematic_ids = mapping["systematic_name"].fillna("").astype(str).tolist()
    np.savez_compressed(
        output,
        proteins=np.asarray(proteins, dtype=str),
        systematic_ids=np.asarray(systematic_ids, dtype=str),
        edge_index=edge_index,
        edge_weight=edge_weight,
        rewired_edge_index=rewired_edge_index,
        rewired_edge_weight=rewired_edge_weight,
        min_score=np.asarray(min_score, dtype=np.int64),
    )


def load_graph_bundle(path: str | Path) -> GraphBundle:
    with np.load(path, allow_pickle=False) as payload:
        return GraphBundle(
            proteins=payload["proteins"].astype(str).tolist(),
            systematic_ids=payload["systematic_ids"].astype(str).tolist(),
            edge_index=payload["edge_index"].astype(np.int64),
            edge_weight=payload["edge_weight"].astype(np.float32),
            rewired_edge_index=payload["rewired_edge_index"].astype(np.int64),
            rewired_edge_weight=payload["rewired_edge_weight"].astype(np.float32),
            min_score=int(payload["min_score"]),
        )


def normalized_adjacency(
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    if edge_index.shape != (2, len(edge_weight)):
        raise ValueError("edge_index and edge_weight shapes are inconsistent")
    nodes = np.arange(n_nodes, dtype=np.int64)
    rows = np.concatenate([edge_index[0], edge_index[1], nodes])
    cols = np.concatenate([edge_index[1], edge_index[0], nodes])
    weights = np.concatenate([edge_weight, edge_weight, np.ones(n_nodes, dtype=np.float32)])
    degree = np.zeros(n_nodes, dtype=np.float32)
    np.add.at(degree, rows, weights)
    normalized = weights / np.sqrt(degree[rows] * degree[cols])
    indices = torch.from_numpy(np.stack([rows, cols])).long().to(device)
    values = torch.from_numpy(normalized.astype(np.float32)).to(device)
    return torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes), device=device).coalesce()


def identity_adjacency(n_nodes: int, device: torch.device) -> torch.Tensor:
    nodes = torch.arange(n_nodes, device=device)
    indices = torch.stack([nodes, nodes])
    return torch.sparse_coo_tensor(indices, torch.ones(n_nodes, device=device), (n_nodes, n_nodes)).coalesce()


def write_graph_manifest(
    path: str | Path,
    *,
    resources: dict[str, dict[str, object]],
    mapping: pd.DataFrame,
    min_score: int,
    rewire_seed: int,
    rewire_swaps_per_edge: int,
    real_stats: dict[str, object],
    rewired_stats: dict[str, object],
) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "species": "Saccharomyces cerevisiae (NCBI taxonomy 4932)",
        "graph_type": "STRING v12.0 physical protein interaction network",
        "min_combined_score": min_score,
        "resources": resources,
        "mapping": {
            "total": int(len(mapping)),
            "mapped": int(mapping["mapping_status"].eq("mapped").sum()),
            "unresolved": mapping.loc[mapping["mapping_status"].ne("mapped"), "raw_name"].tolist(),
            "method_counts": mapping["mapping_method"].value_counts().to_dict(),
        },
        "real_graph": real_stats,
        "rewired_graph": {
            **rewired_stats,
            "seed": rewire_seed,
            "swaps_per_edge": rewire_swaps_per_edge,
            "control": "undirected double-edge swaps; unweighted node degree preserved",
        },
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
