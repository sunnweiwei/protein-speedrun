#!/usr/bin/env python
"""Build the fixed packed corpus from synthetic data or gold_small/v1."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

TRACK_ROOT = Path(__file__).resolve().parents[1]
if str(TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_ROOT))

from speedrun.corpus import encode_sequence, save_corpus, sha256_file

SPLIT_SALT = "protein-pretrain-speedrun-gold-small-v1"


def _split_for_cluster(cluster_id: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}:{cluster_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "probe_train"
    return "probe_eval"


def _synthetic_coords(length: int, phase: float) -> np.ndarray:
    residue = np.arange(length, dtype=np.float32)
    angle = (2.0 * np.pi / 24.0) * residue + phase
    x = 4.0 * np.cos(angle)
    y = 4.0 * np.sin(angle)
    z = 1.2 * np.mod(residue, 12.0)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def build_synthetic(output: Path) -> dict[str, Any]:
    rng = np.random.default_rng(20260729)

    def make_sequence(index: int) -> np.ndarray:
        length = int(rng.integers(56, 81))
        base = (np.arange(length) * (index % 7 + 1) + index) % 20
        noise = rng.random(length) < 0.1
        base[noise] = rng.integers(0, 20, size=int(noise.sum()))
        return base.astype(np.uint8)

    train = [make_sequence(index) for index in range(48)]
    probe_train = []
    probe_eval = []
    for split, records, count in (
        ("probe_train", probe_train, 12),
        ("probe_eval", probe_eval, 12),
    ):
        phase_offset = 0.2 if split == "probe_train" else 0.5
        for index in range(count):
            sequence = make_sequence(100 + index)
            coords = _synthetic_coords(len(sequence), phase_offset + 0.03 * index)
            records.append((sequence, coords, np.ones(len(sequence), dtype=np.bool_)))

    save_corpus(
        output,
        train_sequences=train,
        probe_train=probe_train,
        probe_eval=probe_eval,
    )
    return {
        "schema_version": 1,
        "kind": "synthetic_smoke",
        "corpus_sha256": sha256_file(output),
        "counts": {
            "train": len(train),
            "probe_train": len(probe_train),
            "probe_eval": len(probe_eval),
        },
    }


def _load_source_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "pdb_id",
                "entity_id_l",
                "entity_id_r",
                "sequence_cluster_l",
                "sequence_cluster_r",
            }
            if not required.issubset(row):
                raise ValueError(f"{path}:{line_number}: missing required fields")
            rows.append(row)
    if not rows:
        raise ValueError("source manifest is empty")
    return rows


def _entity_candidates(rows: list[dict[str, Any]]) -> dict[str, list[tuple[str, str]]]:
    candidates: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        for side in ("l", "r"):
            cluster = str(row[f"sequence_cluster_{side}"])
            entity = str(row[f"entity_id_{side}"])
            candidates[cluster].add((str(row["pdb_id"]).lower(), entity))
    return {
        cluster: sorted(values)
        for cluster, values in sorted(candidates.items())
    }


def _integer_sequence_ids(values: np.ndarray) -> np.ndarray:
    """Normalize mmCIF label_seq_id strings while preserving unresolved blanks."""
    strings = np.asarray(values).astype(str)
    output = np.full(strings.shape, -1, dtype=np.int32)
    numeric = np.char.isdigit(strings)
    output[numeric] = strings[numeric].astype(np.int32)
    return output


def _representative_coords(
    payload: dict[str, Any], entity_id: str, sequence: str
) -> tuple[np.ndarray, np.ndarray]:
    atom_array = payload["atom_array"]
    entity_values = np.asarray(atom_array.label_entity_id).astype(str)
    entity_mask = entity_values == str(entity_id)
    if not entity_mask.any():
        raise ValueError(f"entity {entity_id} has no atoms")
    asym_values = np.asarray(atom_array.label_asym_id).astype(str)
    asym_ids, counts = np.unique(asym_values[entity_mask], return_counts=True)
    asym_id = str(asym_ids[int(np.argmax(counts))])
    chain_mask = entity_mask & (asym_values == asym_id)

    atom_names = np.asarray(atom_array.atom_name).astype(str)
    sequence_ids = _integer_sequence_ids(atom_array.label_seq_id)
    if hasattr(atom_array, "is_resolved"):
        resolved_atoms = np.asarray(atom_array.is_resolved).astype(bool)
    else:
        resolved_atoms = np.ones(len(atom_array), dtype=bool)
    coordinates = np.asarray(atom_array.coord, dtype=np.float32)

    output = np.full((len(sequence), 3), np.nan, dtype=np.float32)
    resolved = np.zeros(len(sequence), dtype=np.bool_)
    for position, amino_acid in enumerate(sequence, 1):
        residue_mask = (
            chain_mask
            & resolved_atoms
            & (sequence_ids == position)
        )
        preferred = "CA" if amino_acid == "G" else "CB"
        indices = np.flatnonzero(residue_mask & (atom_names == preferred))
        if not len(indices) and preferred == "CB":
            indices = np.flatnonzero(residue_mask & (atom_names == "CA"))
        if len(indices):
            coordinate = coordinates[int(indices[0])]
            if np.isfinite(coordinate).all():
                output[position - 1] = coordinate
                resolved[position - 1] = True
    return output, resolved


def _load_entity(
    structure_dir: Path, pdb_id: str, entity_id: str, *, require_structure: bool
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    path = structure_dir / f"{pdb_id}.pkl.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)  # trusted, operator-pinned Protenix artifact
    sequence = str(payload["sequences"][str(entity_id)])
    tokens = encode_sequence(sequence)
    if len(tokens) < 32:
        raise ValueError("sequence is too short for the long-range contact protocol")
    if not require_structure:
        return tokens, None, None
    coords, resolved = _representative_coords(payload, entity_id, sequence)
    if float(resolved.mean()) < 0.8:
        raise ValueError("fewer than 80% representative coordinates are resolved")
    return tokens, coords, resolved


def build_gold_small(
    source_manifest: Path,
    structure_dir: Path,
    output: Path,
    metadata_output: Path,
    materializer_image: str,
) -> dict[str, Any]:
    candidates = _entity_candidates(_load_source_rows(source_manifest))
    train: list[np.ndarray] = []
    probe_train: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    probe_eval: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    failures: dict[str, int] = defaultdict(int)
    split_clusters: dict[str, list[str]] = defaultdict(list)

    for cluster, cluster_candidates in candidates.items():
        split = _split_for_cluster(cluster)
        require_structure = split != "train"
        selected = None
        for pdb_id, entity_id in cluster_candidates:
            try:
                selected = _load_entity(
                    structure_dir,
                    pdb_id,
                    entity_id,
                    require_structure=require_structure,
                )
                break
            except (FileNotFoundError, KeyError, ValueError):
                failures[split] += 1
        if selected is None:
            continue
        tokens, coords, resolved = selected
        split_clusters[split].append(cluster)
        if split == "train":
            train.append(tokens)
        else:
            if coords is None or resolved is None:
                raise AssertionError("probe split is missing structure")
            record = (tokens, coords, resolved)
            if split == "probe_train":
                probe_train.append(record)
            else:
                probe_eval.append(record)

    if len(train) < 100 or len(probe_train) < 10 or len(probe_eval) < 10:
        raise ValueError(
            "materialized corpus is unexpectedly small: "
            f"{len(train)}/{len(probe_train)}/{len(probe_eval)}"
        )
    overlap = (
        set(split_clusters["train"]) & set(split_clusters["probe_train"])
        | set(split_clusters["train"]) & set(split_clusters["probe_eval"])
        | set(split_clusters["probe_train"]) & set(split_clusters["probe_eval"])
    )
    if overlap:
        raise AssertionError("sequence-cluster split overlap")

    save_corpus(
        output,
        train_sequences=train,
        probe_train=probe_train,
        probe_eval=probe_eval,
    )
    metadata = {
        "schema_version": 1,
        "kind": "gold_small_v1_protein_pretraining",
        "split_salt": SPLIT_SALT,
        "source_manifest_sha256": sha256_file(source_manifest),
        "materializer_image": materializer_image,
        "corpus_sha256": sha256_file(output),
        "counts": {
            "train": len(train),
            "probe_train": len(probe_train),
            "probe_eval": len(probe_eval),
        },
        "cluster_counts": {
            key: len(value) for key, value in sorted(split_clusters.items())
        },
        "failed_candidates_before_representative": dict(sorted(failures.items())),
        "cluster_overlap": 0,
    }
    temporary = metadata_output.with_name(metadata_output.name + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary.replace(metadata_output)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic")
    synthetic.add_argument("--output", required=True, type=Path)
    synthetic.add_argument("--metadata-output", type=Path)

    gold = subparsers.add_parser("gold-small")
    gold.add_argument("--source-manifest", required=True, type=Path)
    gold.add_argument("--structure-dir", required=True, type=Path)
    gold.add_argument("--materializer-image", required=True)
    gold.add_argument("--output", required=True, type=Path)
    gold.add_argument("--metadata-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "synthetic":
        metadata = build_synthetic(args.output)
        metadata_output = args.metadata_output or args.output.with_suffix(".manifest.json")
        metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    else:
        metadata = build_gold_small(
            args.source_manifest,
            args.structure_dir,
            args.output,
            args.metadata_output,
            args.materializer_image,
        )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
