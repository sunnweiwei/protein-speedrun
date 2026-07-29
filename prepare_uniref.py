#!/usr/bin/env python
"""Download historical UniRef and materialize a memory-mapped training corpus."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np

UNREF_2021_04_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/"
    "release-2021_04/uniref/uniref2021_04.tar.gz"
)
UNREF_2021_04_BYTES = 169_188_630_923
UNREF_2021_04_MD5 = "444f0a7062a65a988ba1d5949d1d6419"
RESTYPES = "ARNDCQEGHILKMFPSTWYV"
TOKEN = {residue: index for index, residue in enumerate(RESTYPES)}


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > UNREF_2021_04_BYTES:
        raise ValueError(f"{partial}: partial file is larger than the source")
    request = urllib.request.Request(UNREF_2021_04_URL)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=120) as response:
        if offset and response.status != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        session_start_bytes = offset
        started = time.monotonic()
        last_report = started
        with partial.open(mode) as target:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                offset += len(chunk)
                now = time.monotonic()
                if now - last_report >= 10:
                    elapsed = max(now - started, 1e-6)
                    print(
                        json.dumps(
                            {
                                "bytes": offset,
                                "fraction": offset / UNREF_2021_04_BYTES,
                                "session_mib_s": (offset - session_start_bytes)
                                / elapsed
                                / 2**20,
                            }
                        ),
                        flush=True,
                    )
                    last_report = now
    actual_size = partial.stat().st_size
    if actual_size != UNREF_2021_04_BYTES:
        raise RuntimeError(
            f"incomplete download: {actual_size} of {UNREF_2021_04_BYTES} bytes"
        )
    actual_md5 = file_digest(partial, "md5")
    if actual_md5 != UNREF_2021_04_MD5:
        raise RuntimeError(f"MD5 mismatch: {actual_md5}")
    partial.replace(output)
    print(json.dumps({"status": "complete", "path": str(output), "md5": actual_md5}))


def extract_uniref50(archive: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    with partial.open("wb") as target:
        outer = subprocess.Popen(
            ["tar", "-xOzf", str(archive), "uniref50.tar"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if outer.stdout is None:
            raise RuntimeError("failed to open the outer tar stream")
        inner = subprocess.Popen(
            ["tar", "-xOf", "-", "uniref50.xml.gz"],
            stdin=outer.stdout,
            stdout=target,
            stderr=subprocess.PIPE,
        )
        outer.stdout.close()
        _, inner_error = inner.communicate()
        outer_error = outer.stderr.read() if outer.stderr else b""
        outer_status = outer.wait()
    if outer_status or inner.returncode:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            "UniRef50 extraction failed:\n"
            + outer_error.decode(errors="replace")[-2000:]
            + inner_error.decode(errors="replace")[-2000:]
        )
    if partial.stat().st_size == 0:
        partial.unlink()
        raise RuntimeError("UniRef50 extraction produced an empty XML archive")
    partial.replace(output)
    print(
        json.dumps(
            {
                "status": "complete",
                "path": str(output),
                "bytes": output.stat().st_size,
                "sha256": file_digest(output),
            }
        )
    )


def fasta_records(path: Path):
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt") as handle:
        identifier = None
        sequence = []
        for line in handle:
            if line.startswith(">"):
                if sequence:
                    yield identifier, "".join(sequence)
                identifier = line[1:].strip().split()[0]
                sequence = []
            else:
                sequence.append(line.strip().upper())
        if sequence:
            yield identifier, "".join(sequence)


def uniref_xml_records(path: Path):
    """Yield representative cluster IDs and sequences from the UniRef XML dump."""
    entry_prefix = b'<entry id="'
    in_representative = False
    identifier = None
    with gzip.open(path, "rb") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith(entry_prefix):
                identifier = stripped[len(entry_prefix) :].split(b'"', 1)[0].decode()
            elif stripped == b"<representativeMember>":
                in_representative = True
            elif stripped == b"</representativeMember>":
                in_representative = False
            elif in_representative and stripped.startswith(b"<sequence "):
                sequence = stripped.split(b">", 1)[1]
                while b"</sequence>" not in sequence:
                    sequence += next(handle).strip()
                value = sequence.split(b"</sequence>", 1)[0].decode().upper()
                if identifier is None:
                    raise ValueError(f"{path}: sequence appears before an entry ID")
                yield identifier, value


def probe_sequences(path: Path) -> set[str]:
    with np.load(path, allow_pickle=False) as data:
        sequences = set()
        for split in ("probe_train", "probe_eval"):
            tokens = data[f"{split}_tokens"]
            offsets = data[f"{split}_offsets"]
            for start, end in zip(offsets[:-1], offsets[1:]):
                sequences.add("".join(RESTYPES[int(value)] for value in tokens[start:end]))
    return sequences


def write_npy(path: Path, values: np.ndarray) -> dict[str, object]:
    np.save(path, values, allow_pickle=False)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_digest(path),
    }


def materialize(
    source: Path,
    source_kind: str,
    probe_corpus: Path,
    output: Path,
    sequences: int,
    shard_sequences: int,
    seed: int,
    min_length: int,
    max_length: int,
) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    excluded = probe_sequences(probe_corpus)
    rng = random.Random(seed)
    reservoir: list[tuple[str, str]] = []
    seen = 0
    rejected_noncanonical = 0
    rejected_length = 0
    rejected_probe_exact = 0
    records = (
        uniref_xml_records(source)
        if source_kind == "uniref_xml"
        else fasta_records(source)
    )
    for identifier, sequence in records:
        if not min_length <= len(sequence) <= max_length:
            rejected_length += 1
            continue
        if any(residue not in TOKEN for residue in sequence):
            rejected_noncanonical += 1
            continue
        if sequence in excluded:
            rejected_probe_exact += 1
            continue
        seen += 1
        if len(reservoir) < sequences:
            reservoir.append((identifier, sequence))
        else:
            replacement = rng.randrange(seen)
            if replacement < sequences:
                reservoir[replacement] = (identifier, sequence)
        if seen % 1_000_000 == 0:
            print(json.dumps({"eligible_sequences_seen": seen}), flush=True)
    if len(reservoir) != sequences:
        raise ValueError(f"requested {sequences} sequences but found {len(reservoir)}")
    reservoir.sort(
        key=lambda value: hashlib.sha256(value[1].encode()).digest()
    )

    rows = []
    total_residues = 0
    for shard_index, start in enumerate(range(0, sequences, shard_sequences)):
        selected = reservoir[start : start + shard_sequences]
        offsets = np.zeros(len(selected) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(sequence) for _, sequence in selected])
        tokens = np.fromiter(
            (
                TOKEN[residue]
                for _, sequence in selected
                for residue in sequence
            ),
            dtype=np.uint8,
            count=int(offsets[-1]),
        )
        token_path = output / f"train-{shard_index:05d}-tokens.npy"
        offset_path = output / f"train-{shard_index:05d}-offsets.npy"
        id_path = output / f"train-{shard_index:05d}-ids.txt"
        token_record = write_npy(token_path, tokens)
        offset_record = write_npy(offset_path, offsets)
        id_path.write_text("".join(f"{identifier}\n" for identifier, _ in selected))
        rows.append(
            {
                "tokens": token_record["path"],
                "tokens_bytes": token_record["bytes"],
                "tokens_sha256": token_record["sha256"],
                "offsets": offset_record["path"],
                "offsets_bytes": offset_record["bytes"],
                "offsets_sha256": offset_record["sha256"],
                "ids": id_path.name,
                "ids_bytes": id_path.stat().st_size,
                "ids_sha256": file_digest(id_path),
                "sequences": len(selected),
                "residues": len(tokens),
            }
        )
        total_residues += len(tokens)

    probe_copy = output / "probe.npz"
    shutil.copyfile(probe_corpus, probe_copy)
    probe_sha = file_digest(probe_copy)
    identity = {
        "format": "protein-speedrun-sharded-v1",
        "source": {
            "name": "UniRef50",
            "release": "2021_04",
            "kind": source_kind,
            "sha256": file_digest(source),
        },
        "selection": {
            "method": "uniform_reservoir",
            "seed": seed,
            "min_length": min_length,
            "max_length": max_length,
            "canonical_residues_only": True,
            "probe_exact_matches_removed": rejected_probe_exact,
        },
        "sequences": sequences,
        "residues": total_residues,
        "train_shards": rows,
        "probe_corpus": probe_copy.name,
        "probe_sha256": probe_sha,
    }
    content = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    manifest = {**identity, "content_sha256": hashlib.sha256(content).hexdigest()}
    (output / "corpus.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    report = {
        "eligible_sequences_seen": seen,
        "rejected_length": rejected_length,
        "rejected_noncanonical": rejected_noncanonical,
        "rejected_probe_exact": rejected_probe_exact,
        "output_sequences": sequences,
        "output_residues": total_residues,
        "content_sha256": manifest["content_sha256"],
    }
    (output / "materialization.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--output", type=Path, required=True)
    extract_parser = subparsers.add_parser("extract-uniref50")
    extract_parser.add_argument("--archive", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    materialize_parser = subparsers.add_parser("materialize")
    source = materialize_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fasta", type=Path)
    source.add_argument("--uniref-xml", type=Path)
    materialize_parser.add_argument("--probe-corpus", type=Path, required=True)
    materialize_parser.add_argument("--output", type=Path, required=True)
    materialize_parser.add_argument("--sequences", type=positive, default=1_000_000)
    materialize_parser.add_argument(
        "--shard-sequences", type=positive, default=50_000
    )
    materialize_parser.add_argument("--seed", type=int, default=20260729)
    materialize_parser.add_argument("--min-length", type=positive, default=32)
    materialize_parser.add_argument("--max-length", type=positive, default=1024)
    arguments = parser.parse_args()
    if arguments.command == "download":
        download(arguments.output)
        return
    if arguments.command == "extract-uniref50":
        extract_uniref50(arguments.archive, arguments.output)
        return
    materialize(
        arguments.uniref_xml or arguments.fasta,
        "uniref_xml" if arguments.uniref_xml else "fasta",
        arguments.probe_corpus,
        arguments.output,
        arguments.sequences,
        arguments.shard_sequences,
        arguments.seed,
        arguments.min_length,
        arguments.max_length,
    )


if __name__ == "__main__":
    main()
