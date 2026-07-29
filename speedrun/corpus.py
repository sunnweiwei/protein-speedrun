"""Immutable packed-array corpus contract."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: index for index, aa in enumerate(AMINO_ACIDS)}
UNKNOWN_TOKEN_ID = 20
PAD_TOKEN_ID = 21
MASK_TOKEN_ID = 22
VOCAB_SIZE = 23


def encode_sequence(sequence: str) -> np.ndarray:
    return np.fromiter(
        (AA_TO_ID.get(amino_acid.upper(), UNKNOWN_TOKEN_ID) for amino_acid in sequence),
        dtype=np.uint8,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PackedSequences:
    tokens: np.ndarray
    offsets: np.ndarray
    coords: np.ndarray | None = None
    resolved: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.tokens.dtype != np.uint8 or self.tokens.ndim != 1:
            raise ValueError("packed tokens must be one-dimensional uint8")
        if self.offsets.dtype != np.int64 or self.offsets.ndim != 1:
            raise ValueError("offsets must be one-dimensional int64")
        if len(self.offsets) < 2 or self.offsets[0] != 0:
            raise ValueError("offsets must begin at zero and contain a sequence")
        if np.any(np.diff(self.offsets) <= 0) or self.offsets[-1] != len(self.tokens):
            raise ValueError("offsets must be strictly increasing through packed tokens")
        if self.tokens.size and int(self.tokens.max()) > UNKNOWN_TOKEN_ID:
            raise ValueError("corpus tokens may only contain amino acids and unknown")
        if (self.coords is None) != (self.resolved is None):
            raise ValueError("coordinates and resolved mask must be supplied together")
        if self.coords is not None:
            if self.coords.shape != (len(self.tokens), 3):
                raise ValueError("packed coordinates must have shape [num_tokens, 3]")
            if self.coords.dtype != np.float32:
                raise ValueError("coordinates must be float32")
            if self.resolved is None or self.resolved.shape != (len(self.tokens),):
                raise ValueError("resolved mask must align with tokens")
            if self.resolved.dtype != np.bool_:
                raise ValueError("resolved mask must be boolean")
            if not np.isfinite(self.coords[self.resolved]).all():
                raise ValueError("resolved coordinates must be finite")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def sequence(self, index: int) -> np.ndarray:
        start, end = self.offsets[index : index + 2]
        return self.tokens[start:end]

    def structure(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if self.coords is None or self.resolved is None:
            raise ValueError("split has no structures")
        start, end = self.offsets[index : index + 2]
        return self.coords[start:end], self.resolved[start:end]


@dataclass(frozen=True)
class SpeedrunCorpus:
    path: Path
    sha256: str
    train: PackedSequences
    probe_train: PackedSequences
    probe_eval: PackedSequences


def _split_from_payload(
    payload: np.lib.npyio.NpzFile, prefix: str, *, structures: bool
) -> PackedSequences:
    required = {f"{prefix}_tokens", f"{prefix}_offsets"}
    if structures:
        required.update({f"{prefix}_coords", f"{prefix}_resolved"})
    missing = required.difference(payload.files)
    if missing:
        raise ValueError(f"corpus is missing arrays: {sorted(missing)}")
    return PackedSequences(
        tokens=np.asarray(payload[f"{prefix}_tokens"]),
        offsets=np.asarray(payload[f"{prefix}_offsets"]),
        coords=(
            np.asarray(payload[f"{prefix}_coords"])
            if structures
            else None
        ),
        resolved=(
            np.asarray(payload[f"{prefix}_resolved"])
            if structures
            else None
        ),
    )


def load_corpus(path: Path) -> SpeedrunCorpus:
    resolved_path = path.resolve()
    if not resolved_path.is_file():
        raise ValueError(f"corpus does not exist: {resolved_path}")
    with np.load(resolved_path, allow_pickle=False) as payload:
        allowed = {
            "schema_version",
            "train_tokens",
            "train_offsets",
            "probe_train_tokens",
            "probe_train_offsets",
            "probe_train_coords",
            "probe_train_resolved",
            "probe_eval_tokens",
            "probe_eval_offsets",
            "probe_eval_coords",
            "probe_eval_resolved",
        }
        if set(payload.files) != allowed:
            raise ValueError(
                "corpus array names do not match schema: "
                f"{sorted(set(payload.files).symmetric_difference(allowed))}"
            )
        schema = np.asarray(payload["schema_version"])
        if schema.shape != () or int(schema) != 1:
            raise ValueError("unsupported corpus schema")
        train = _split_from_payload(payload, "train", structures=False)
        probe_train = _split_from_payload(payload, "probe_train", structures=True)
        probe_eval = _split_from_payload(payload, "probe_eval", structures=True)
    return SpeedrunCorpus(
        path=resolved_path,
        sha256=sha256_file(resolved_path),
        train=train,
        probe_train=probe_train,
        probe_eval=probe_eval,
    )


def _pack_sequences(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not sequences:
        raise ValueError("cannot pack an empty sequence split")
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(sequence) for sequence in sequences])
    return np.concatenate(sequences).astype(np.uint8), offsets


def _pack_structures(
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("cannot pack an empty structure split")
    sequences = [record[0] for record in records]
    tokens, offsets = _pack_sequences(sequences)
    coords = np.concatenate([record[1] for record in records]).astype(np.float32)
    resolved = np.concatenate([record[2] for record in records]).astype(np.bool_)
    return tokens, offsets, coords, resolved


def save_corpus(
    path: Path,
    *,
    train_sequences: list[np.ndarray],
    probe_train: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    probe_eval: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable corpus: {output}")
    train_tokens, train_offsets = _pack_sequences(train_sequences)
    probe_train_arrays = _pack_structures(probe_train)
    probe_eval_arrays = _pack_structures(probe_eval)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(1, dtype=np.int64),
            train_tokens=train_tokens,
            train_offsets=train_offsets,
            probe_train_tokens=probe_train_arrays[0],
            probe_train_offsets=probe_train_arrays[1],
            probe_train_coords=probe_train_arrays[2],
            probe_train_resolved=probe_train_arrays[3],
            probe_eval_tokens=probe_eval_arrays[0],
            probe_eval_offsets=probe_eval_arrays[1],
            probe_eval_coords=probe_eval_arrays[2],
            probe_eval_resolved=probe_eval_arrays[3],
        )
    temporary.replace(output)

