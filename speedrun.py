#!/usr/bin/env python
"""Minimal protein pretraining speedrun: train, checkpoint, and Contact P@L."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parent
PAD_TOKEN = 21
CONTACT_SEPARATION = 24
CONTACT_DISTANCE = 8.0
PROBE_SEED = 20260729


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


class Packed:
    def __init__(self, tokens, offsets, coords=None, resolved=None):
        self.tokens = tokens
        self.offsets = offsets
        self.coords = coords
        self.resolved = resolved

    def __len__(self):
        return len(self.offsets) - 1

    def sequence(self, index):
        start, end = self.offsets[index : index + 2]
        return self.tokens[start:end]

    def structure(self, index):
        start, end = self.offsets[index : index + 2]
        return self.coords[start:end], self.resolved[start:end]


class ShardedPacked:
    """Memory-map a large training split without copying it into every rank."""

    def __init__(self, shards):
        self.shards = shards
        self.ends = []
        total = 0
        for shard in shards:
            total += len(shard)
            self.ends.append(total)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def sequence(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.ends, index)
        start = self.ends[shard_index - 1] if shard_index else 0
        return self.shards[shard_index].sequence(index - start)


def _load_npz_corpus(path: Path) -> tuple[str, Packed, Packed, Packed]:
    with np.load(path, allow_pickle=False) as data:
        def split(name, structured):
            return Packed(
                np.asarray(data[f"{name}_tokens"]),
                np.asarray(data[f"{name}_offsets"]),
                np.asarray(data[f"{name}_coords"]) if structured else None,
                np.asarray(data[f"{name}_resolved"]) if structured else None,
            )

        train = split("train", False)
        probe_train = split("probe_train", True)
        probe_eval = split("probe_eval", True)
    return sha256(path), train, probe_train, probe_eval


def load_corpus(path: Path) -> tuple[str, Packed, Packed, Packed]:
    if path.suffix == ".npz":
        return _load_npz_corpus(path)
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "protein-speedrun-sharded-v1":
        raise ValueError(f"{path}: unsupported corpus format")
    root = path.parent.resolve()

    def resolve(relative):
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"{path}: corpus path escapes its directory")
        return resolved

    shards = []
    for row in manifest["train_shards"]:
        tokens_path = resolve(row["tokens"])
        offsets_path = resolve(row["offsets"])
        if tokens_path.stat().st_size != int(row["tokens_bytes"]):
            raise ValueError(f"{tokens_path}: size does not match manifest")
        if offsets_path.stat().st_size != int(row["offsets_bytes"]):
            raise ValueError(f"{offsets_path}: size does not match manifest")
        tokens = np.load(tokens_path, mmap_mode="r", allow_pickle=False)
        offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        if tokens.dtype != np.uint8 or offsets.dtype != np.int64:
            raise ValueError(f"{path}: invalid shard dtypes")
        if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0:
            raise ValueError(f"{path}: invalid shard offsets")
        if int(offsets[-1]) != len(tokens):
            raise ValueError(f"{path}: shard offsets do not cover its tokens")
        if len(offsets) - 1 != int(row["sequences"]):
            raise ValueError(f"{path}: shard sequence count mismatch")
        shards.append(Packed(tokens, offsets))
    if not shards:
        raise ValueError(f"{path}: corpus has no training shards")
    probe_path = resolve(manifest["probe_corpus"])
    if sha256(probe_path) != manifest["probe_sha256"]:
        raise ValueError(f"{probe_path}: probe digest does not match manifest")
    _, _, probe_train, probe_eval = _load_npz_corpus(probe_path)
    return (
        manifest["content_sha256"],
        ShardedPacked(shards),
        probe_train,
        probe_eval,
    )


def pack(sequences):
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(sequence) for sequence in sequences])
    return np.concatenate(sequences).astype(np.uint8), offsets


def make_synthetic_corpus(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260729)

    def sequence(index):
        length = int(rng.integers(56, 81))
        values = (np.arange(length) * (index % 7 + 1) + index) % 20
        return values.astype(np.uint8)

    def structures(offset):
        records = []
        for index in range(12):
            tokens = sequence(100 + index)
            residue = np.arange(len(tokens), dtype=np.float32)
            angle = 2 * np.pi * residue / 24 + offset + 0.03 * index
            coords = np.stack(
                [4 * np.cos(angle), 4 * np.sin(angle), 1.2 * (residue % 12)],
                axis=-1,
            ).astype(np.float32)
            records.append((tokens, coords, np.ones(len(tokens), dtype=np.bool_)))
        return records

    train_tokens, train_offsets = pack([sequence(i) for i in range(48)])
    probe_train = structures(0.2)
    probe_eval = structures(0.5)
    pt, po = pack([item[0] for item in probe_train])
    et, eo = pack([item[0] for item in probe_eval])
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            train_tokens=train_tokens,
            train_offsets=train_offsets,
            probe_train_tokens=pt,
            probe_train_offsets=po,
            probe_train_coords=np.concatenate([item[1] for item in probe_train]),
            probe_train_resolved=np.concatenate([item[2] for item in probe_train]),
            probe_eval_tokens=et,
            probe_eval_offsets=eo,
            probe_eval_coords=np.concatenate([item[1] for item in probe_eval]),
            probe_eval_resolved=np.concatenate([item[2] for item in probe_eval]),
        )


def load_candidate(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_checkpoint(
    directory: Path,
    model: nn.Module,
    model_config: dict[str, Any],
    candidate_path: Path,
    metadata: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    weights = directory / "weights.pt"
    candidate = directory / "candidate.py"
    torch.save(
        {name: value.detach().cpu() for name, value in model.state_dict().items()},
        weights,
    )
    candidate.write_bytes(candidate_path.read_bytes())
    write_json(directory / "model.json", model_config)
    write_json(
        directory / "checkpoint.json",
        {
            **metadata,
            "weights_sha256": sha256(weights),
            "candidate_sha256": sha256(candidate),
        },
    )


def load_checkpoint(directory: Path, device: torch.device):
    metadata = json.loads((directory / "checkpoint.json").read_text())
    candidate_path = directory / "candidate.py"
    weights_path = directory / "weights.pt"
    if sha256(candidate_path) != metadata["candidate_sha256"]:
        raise ValueError("checkpoint candidate code digest mismatch")
    if sha256(weights_path) != metadata["weights_sha256"]:
        raise ValueError("checkpoint weights digest mismatch")
    module = load_candidate(candidate_path, "checkpoint_candidate")
    model = module.build_model(json.loads((directory / "model.json").read_text()))
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    return model.to(device).eval(), metadata


def encode(model, sequence, device):
    tokens = torch.from_numpy(sequence.astype(np.int64))[None].to(device)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    with torch.inference_mode():
        embeddings = model.encode(tokens, mask)[0].float()
    if embeddings.shape[0] != len(sequence) or not torch.isfinite(embeddings).all():
        raise ValueError("encode() must return finite [B, L, D] embeddings")
    return nn.functional.normalize(embeddings, dim=-1)


def pairs(length, resolved, device):
    left, right = torch.triu_indices(
        length, length, offset=CONTACT_SEPARATION, device=device
    )
    valid = resolved[left] & resolved[right]
    return left[valid], right[valid]


def labels(coords, left, right):
    return torch.linalg.vector_norm(coords[left] - coords[right], dim=-1) < (
        CONTACT_DISTANCE
    )


def fit_contact_probe(x, y, width, device):
    torch.manual_seed(PROBE_SEED)
    probe = nn.Linear(int(width), 1).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=0.03, weight_decay=1e-4
    )
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(
            probe(x).squeeze(-1), y
        )
        loss.backward()
        optimizer.step()
    return probe.eval()


def train_contact_probe(model, split, device):
    generator = torch.Generator().manual_seed(PROBE_SEED)
    features, targets = [], []
    width = None
    for index in range(len(split)):
        sequence = split.sequence(index)
        coords, resolved = split.structure(index)
        embeddings = encode(model, sequence, device).cpu()
        width = embeddings.shape[-1]
        left, right = pairs(
            len(sequence), torch.from_numpy(resolved), torch.device("cpu")
        )
        target = labels(torch.from_numpy(coords), left, right)
        positive = torch.nonzero(target).flatten()
        negative = torch.nonzero(~target).flatten()
        if not len(positive) or not len(negative):
            continue
        count = min(len(positive), len(negative), 1024)
        positive = positive[torch.randperm(len(positive), generator=generator)[:count]]
        negative = negative[torch.randperm(len(negative), generator=generator)[:count]]
        selected = torch.cat([positive, negative])
        features.append(embeddings[left[selected]] * embeddings[right[selected]])
        targets.append(target[selected].float())
    x = torch.cat(features).to(device)
    y = torch.cat(targets).to(device)
    return fit_contact_probe(x, y, width, device)


def contact_p_at_l(model, probe, split, device):
    scores = []
    with torch.inference_mode():
        for index in range(len(split)):
            sequence = split.sequence(index)
            coords, resolved = split.structure(index)
            embeddings = encode(model, sequence, device)
            coords = torch.from_numpy(coords).to(device)
            resolved = torch.from_numpy(resolved).to(device)
            left, right = pairs(len(sequence), resolved, device)
            target = labels(coords, left, right)
            if int(target.sum()) < math.ceil(0.25 * len(sequence)):
                continue
            logits = probe(embeddings[left] * embeddings[right]).squeeze(-1)
            top = torch.topk(logits, min(len(logits), len(sequence))).indices
            scores.append(float(target[top].float().mean().cpu()))
    return float(np.mean(scores)), len(scores)


def distributed_probe_rows(model, split, device, rank, world_size):
    """Build an exact shard while advancing the canonical sampler on all ranks."""
    generator = torch.Generator().manual_seed(PROBE_SEED)
    features, targets, positions = [], [], []
    width = 0
    offset = 0
    for index in range(len(split)):
        sequence = split.sequence(index)
        coords, resolved = split.structure(index)
        left, right = pairs(
            len(sequence), torch.from_numpy(resolved), torch.device("cpu")
        )
        target = labels(torch.from_numpy(coords), left, right)
        positive = torch.nonzero(target).flatten()
        negative = torch.nonzero(~target).flatten()
        if not len(positive) or not len(negative):
            continue
        count = min(len(positive), len(negative), 1024)
        positive = positive[
            torch.randperm(len(positive), generator=generator)[:count]
        ]
        negative = negative[
            torch.randperm(len(negative), generator=generator)[:count]
        ]
        selected = torch.cat([positive, negative])
        rows = len(selected)
        if index % world_size == rank:
            embeddings = encode(model, sequence, device).cpu()
            width = embeddings.shape[-1]
            features.append(
                embeddings[left[selected]] * embeddings[right[selected]]
            )
            targets.append(target[selected].float())
            positions.append(torch.arange(offset, offset + rows))
        offset += rows
    width_tensor = torch.tensor(width, dtype=torch.int64, device=device)
    dist.all_reduce(width_tensor, op=dist.ReduceOp.MAX)
    width = int(width_tensor.item())
    if features:
        return (
            torch.cat(features),
            torch.cat(targets),
            torch.cat(positions),
            offset,
            width,
        )
    return (
        torch.empty((0, width), dtype=torch.float32),
        torch.empty(0, dtype=torch.float32),
        torch.empty(0, dtype=torch.int64),
        offset,
        width,
    )


def gather_probe_rows(
    local_x,
    local_y,
    local_positions,
    total_rows,
    width,
    device,
    rank,
    world_size,
):
    local_rows = len(local_y)
    size = torch.tensor(local_rows, dtype=torch.int64, device=device)
    sizes = [torch.empty_like(size) for _ in range(world_size)]
    dist.all_gather(sizes, size)
    sizes = [int(value.item()) for value in sizes]
    padded_rows = max(sizes)

    padded_x = torch.zeros((padded_rows, width), device=device)
    padded_y = torch.zeros(padded_rows, device=device)
    padded_positions = torch.zeros(
        padded_rows, dtype=torch.int64, device=device
    )
    if local_rows:
        padded_x[:local_rows] = local_x.to(device)
        padded_y[:local_rows] = local_y.to(device)
        padded_positions[:local_rows] = local_positions.to(device)

    gathered_x = (
        [torch.empty_like(padded_x) for _ in range(world_size)]
        if rank == 0
        else None
    )
    gathered_y = (
        [torch.empty_like(padded_y) for _ in range(world_size)]
        if rank == 0
        else None
    )
    gathered_positions = (
        [torch.empty_like(padded_positions) for _ in range(world_size)]
        if rank == 0
        else None
    )
    dist.gather(padded_x, gathered_x, dst=0)
    dist.gather(padded_y, gathered_y, dst=0)
    dist.gather(padded_positions, gathered_positions, dst=0)
    if rank != 0:
        return None, None

    x = torch.empty((total_rows, width), device=device)
    y = torch.empty(total_rows, device=device)
    for source_rank, rows in enumerate(sizes):
        position = gathered_positions[source_rank][:rows]
        x[position] = gathered_x[source_rank][:rows]
        y[position] = gathered_y[source_rank][:rows]
    return x, y


def distributed_contact_scores(model, probe, split, device, rank, world_size):
    indices, scores = [], []
    with torch.inference_mode():
        for index in range(rank, len(split), world_size):
            sequence = split.sequence(index)
            coords, resolved = split.structure(index)
            embeddings = encode(model, sequence, device)
            coords = torch.from_numpy(coords).to(device)
            resolved = torch.from_numpy(resolved).to(device)
            left, right = pairs(len(sequence), resolved, device)
            target = labels(coords, left, right)
            if int(target.sum()) < math.ceil(0.25 * len(sequence)):
                continue
            logits = probe(embeddings[left] * embeddings[right]).squeeze(-1)
            top = torch.topk(logits, min(len(logits), len(sequence))).indices
            indices.append(index)
            scores.append(float(target[top].float().mean().cpu()))

    local_size = len(scores)
    size = torch.tensor(local_size, dtype=torch.int64, device=device)
    sizes = [torch.empty_like(size) for _ in range(world_size)]
    dist.all_gather(sizes, size)
    sizes = [int(value.item()) for value in sizes]
    padded_size = max(sizes)
    padded_indices = torch.zeros(
        padded_size, dtype=torch.int64, device=device
    )
    padded_scores = torch.zeros(
        padded_size, dtype=torch.float64, device=device
    )
    if local_size:
        padded_indices[:local_size] = torch.tensor(indices, device=device)
        padded_scores[:local_size] = torch.tensor(
            scores, dtype=torch.float64, device=device
        )
    gathered_indices = (
        [torch.empty_like(padded_indices) for _ in range(world_size)]
        if rank == 0
        else None
    )
    gathered_scores = (
        [torch.empty_like(padded_scores) for _ in range(world_size)]
        if rank == 0
        else None
    )
    dist.gather(padded_indices, gathered_indices, dst=0)
    dist.gather(padded_scores, gathered_scores, dst=0)
    if rank != 0:
        return None
    ordered = []
    for source_rank, rows in enumerate(sizes):
        for row in range(rows):
            ordered.append(
                (
                    int(gathered_indices[source_rank][row].item()),
                    float(gathered_scores[source_rank][row].item()),
                )
            )
    ordered.sort()
    values = [value for _, value in ordered]
    return float(np.mean(values)), len(values)


def evaluate_distributed(
    checkpoint, corpus_path, output, device, rank, world_size
):
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device)
    try:
        corpus_sha, _, probe_train, probe_eval = load_corpus(corpus_path)
        model, metadata = load_checkpoint(checkpoint, device)
        if corpus_sha != metadata["corpus_sha256"]:
            raise ValueError("checkpoint was trained on a different corpus")
        local_x, local_y, local_positions, total_rows, width = (
            distributed_probe_rows(
                model, probe_train, device, rank, world_size
            )
        )
        x, y = gather_probe_rows(
            local_x,
            local_y,
            local_positions,
            total_rows,
            width,
            device,
            rank,
            world_size,
        )
        if rank == 0:
            probe = fit_contact_probe(x, y, width, device)
        else:
            probe = nn.Linear(width, 1).to(device)
        dist.broadcast(probe.weight, src=0)
        dist.broadcast(probe.bias, src=0)
        result = distributed_contact_scores(
            model, probe.eval(), probe_eval, device, rank, world_size
        )
        payload = torch.zeros(2, dtype=torch.float64, device=device)
        if rank == 0:
            payload[0], payload[1] = result
        dist.broadcast(payload, src=0)
        result = {
            "contact_p_at_l": float(payload[0].item()),
            "eligible_proteins": int(payload[1].item()),
        }
        if rank == 0:
            write_json(output, result)
        return result
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state, device)


def evaluate(checkpoint: Path, corpus_path: Path, output: Path) -> dict[str, Any]:
    device = torch.device("cuda")
    corpus_sha, _, probe_train, probe_eval = load_corpus(corpus_path)
    model, metadata = load_checkpoint(checkpoint, device)
    if corpus_sha != metadata["corpus_sha256"]:
        raise ValueError("checkpoint was trained on a different corpus")
    probe = train_contact_probe(model, probe_train, device)
    score, eligible = contact_p_at_l(model, probe, probe_eval, device)
    result = {"contact_p_at_l": score, "eligible_proteins": eligible}
    write_json(output, result)
    return result


def batch(split, batch_size, max_length, rng, device):
    indices = rng.integers(0, len(split), size=batch_size)
    sequences = []
    for index in indices:
        sequence = split.sequence(int(index))
        start = (
            int(rng.integers(0, len(sequence) - max_length + 1))
            if len(sequence) > max_length
            else 0
        )
        sequences.append(sequence[start : start + max_length])
    length = max(len(sequence) for sequence in sequences)
    tokens = torch.full((batch_size, length), PAD_TOKEN, device=device, dtype=torch.long)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        value = torch.from_numpy(sequence.astype(np.int64)).to(device)
        tokens[row, : len(value)] = value
        mask[row, : len(value)] = True
    return tokens, mask


def run_evaluator(checkpoint, corpus_path, output):
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "eval",
            "--checkpoint",
            str(checkpoint),
            "--corpus",
            str(corpus_path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr[-4000:])
    return json.loads(output.read_text())


def evaluator_worker(rank, world_size, init_file):
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_file.resolve()}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request["command"] == "stop":
            print(json.dumps({"stopped": True}), flush=True)
            break
        try:
            result = evaluate_distributed(
                Path(request["checkpoint"]),
                Path(request["corpus"]),
                Path(request["output"]),
                device,
                rank,
                world_size,
            )
            print(json.dumps({"ok": True, "result": result}), flush=True)
        except Exception as error:
            print(json.dumps({"ok": False, "error": repr(error)}), flush=True)
            raise
    dist.destroy_process_group()


class DistributedEvaluatorClient:
    """Persistent isolated evaluator process, one per training rank."""

    def __init__(self, output, rank, world_size, local_rank):
        self.log_path = output / f"evaluator-rank-{rank}.log"
        self.log = self.log_path.open("w")
        environment = os.environ.copy()
        visible = environment.get("CUDA_VISIBLE_DEVICES")
        device = visible.split(",")[local_rank] if visible else str(local_rank)
        environment["CUDA_VISIBLE_DEVICES"] = device
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "eval-worker",
            "--rank",
            str(rank),
            "--world-size",
            str(world_size),
            "--init-file",
            str(output / "distributed-evaluator.init"),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            env=environment,
        )
        response = self._response()
        if not response.get("ready"):
            raise RuntimeError(f"evaluator worker did not start: {response}")

    def _response(self):
        line = self.process.stdout.readline()
        if not line:
            self.log.flush()
            detail = self.log_path.read_text()[-4000:]
            raise RuntimeError(f"evaluator worker exited early:\n{detail}")
        return json.loads(line)

    def evaluate(self, checkpoint, corpus, output):
        request = {
            "command": "evaluate",
            "checkpoint": str(checkpoint),
            "corpus": str(corpus),
            "output": str(output),
        }
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response = self._response()
        if not response.get("ok"):
            raise RuntimeError(f"distributed evaluator failed: {response}")
        return response["result"]

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write(json.dumps({"command": "stop"}) + "\n")
            self.process.stdin.flush()
            response = self._response()
            if not response.get("stopped"):
                raise RuntimeError(f"evaluator worker did not stop: {response}")
            self.process.wait(timeout=30)
        self.log.close()


class DistributedLogits(nn.Module):
    """Give DDP a normal forward while preserving the candidate logits contract."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, padding_mask):
        return self.model.logits(tokens, padding_mask)


class LogitsAdapter:
    def __init__(self, distributed_model):
        self.distributed_model = distributed_model

    def logits(self, tokens, padding_mask):
        return self.distributed_model(tokens, padding_mask)


def train(
    config_path: Path, corpus_path: Path, output: Path, seed_override: int | None
):
    config = json.loads(config_path.read_text())
    if seed_override is not None:
        config["seed"] = seed_override
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("NCCL_ALGO", "Ring")
    os.environ.setdefault("NCCL_PROTO", "Simple")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", device_id=device)
    else:
        device = torch.device("cuda", local_rank)
    if distributed and local_world_size != torch.cuda.device_count():
        raise RuntimeError("DDP must launch one process per visible GPU")
    if not distributed and torch.cuda.device_count() != 1:
        raise RuntimeError("single-process v0 requires exactly one visible GPU")
    if "H100 80GB" not in torch.cuda.get_device_name(device):
        raise RuntimeError("the v0 test requires H100 80GB GPUs")
    global_batch_size = int(config["batch_size"])
    if global_batch_size % world_size:
        raise ValueError("batch_size must be divisible by the DDP world size")
    local_batch_size = global_batch_size // world_size
    corpus_sha, train_split, _, _ = load_corpus(corpus_path)
    if rank == 0:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"run output is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "config.json", config)
    if distributed:
        dist.barrier()
    candidate_path = ROOT / "candidate.py"
    schedule = [0, *range(config["checkpoint_every"], config["steps"] + 1, config["checkpoint_every"])]
    if schedule[-1] != config["steps"]:
        schedule.append(config["steps"])
    evaluator = None
    if distributed:
        evaluator = DistributedEvaluatorClient(
            output, rank, world_size, local_rank
        )

    timer = time.perf_counter
    started = timer()
    candidate = load_candidate(candidate_path, "training_candidate")
    torch.manual_seed(config["seed"])
    model = candidate.build_model(config["model"]).to(device)
    optimizer = candidate.build_optimizer(model, config)
    training_model = model
    if distributed:
        training_model = LogitsAdapter(
            DistributedDataParallel(
                DistributedLogits(model),
                device_ids=[local_rank],
                broadcast_buffers=False,
            )
        )
    generator = torch.Generator(device=device).manual_seed(
        config["seed"] + 1 + rank * 100003
    )
    rng = np.random.default_rng(config["seed"] + rank * 100003)
    excluded = 0.0
    local_tokens_seen = 0
    tokens_seen = 0
    history = []
    streak = 0

    def checkpoint(step):
        nonlocal excluded, streak, tokens_seen
        if distributed:
            dist.barrier()
        torch.cuda.synchronize(device)
        callback_started = timer()
        training_seconds = callback_started - started - excluded
        count = torch.tensor(local_tokens_seen, dtype=torch.int64, device=device)
        if distributed:
            dist.all_reduce(count)
        tokens_seen = int(count.item())
        directory = output / f"step-{step:08d}"
        if rank == 0:
            save_checkpoint(
                directory,
                model,
                config["model"],
                candidate_path,
                {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "training_seconds": training_seconds,
                    "world_size": world_size,
                    "local_world_size": local_world_size,
                    "global_batch_size": global_batch_size,
                    "corpus_sha256": corpus_sha,
                },
            )
        if distributed:
            dist.barrier()
        evaluation_started = timer()
        if distributed:
            result = evaluator.evaluate(
                directory,
                corpus_path,
                directory / "evaluation.json",
            )
        else:
            result = run_evaluator(
                directory, corpus_path, directory / "evaluation.json"
            )
        evaluation_seconds = timer() - evaluation_started
        history.append(
            {
                "step": step,
                "training_seconds": training_seconds,
                "evaluation_seconds": evaluation_seconds,
                "tokens_seen": tokens_seen,
                **result,
            }
        )
        target = config["target_contact_p_at_l"]
        streak = streak + 1 if target is not None and result["contact_p_at_l"] >= target else 0
        if distributed:
            dist.barrier()
        torch.cuda.synchronize(device)
        excluded += timer() - callback_started
        return streak >= 2

    stopped = checkpoint(0)
    torch.cuda.reset_peak_memory_stats(device)
    loss_value = None
    for step in range(1, config["steps"] + 1):
        if stopped:
            break
        model.train()
        tokens, mask = batch(
            train_split,
            local_batch_size,
            config["train_max_length"],
            rng,
            device,
        )
        if hasattr(candidate, "set_optimizer_step"):
            candidate.set_optimizer_step(optimizer, step, config)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=config.get("precision", "fp32") == "bf16",
        ):
            loss = candidate.training_loss(
                training_model, tokens, mask, config, generator
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        local_tokens_seen += int(mask.sum())
        loss_value = float(loss.detach().cpu())
        if step in schedule[1:]:
            stopped = checkpoint(step)
    torch.cuda.synchronize(device)
    if distributed and loss_value is not None:
        mean_loss = torch.tensor(loss_value, device=device)
        dist.all_reduce(mean_loss)
        loss_value = float((mean_loss / world_size).cpu())
    memory = torch.tensor(
        [
            torch.cuda.max_memory_allocated(device),
            torch.cuda.max_memory_reserved(device),
        ],
        dtype=torch.int64,
        device=device,
    )
    if distributed:
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    result = {
        "status": "confirmed" if stopped else "calibration",
        "seed": config["seed"],
        "contact_p_at_l": history[-1]["contact_p_at_l"],
        "training_seconds": history[-1]["training_seconds"],
        "step": history[-1]["step"],
        "tokens_seen": tokens_seen,
        "last_loss": loss_value,
        "last_learning_rate": optimizer.param_groups[0]["lr"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "world_size": world_size,
        "local_world_size": local_world_size,
        "global_batch_size": global_batch_size,
        "peak_memory_allocated_bytes": int(memory[0].item()),
        "peak_memory_reserved_bytes": int(memory[1].item()),
        "corpus_sha256": corpus_sha,
        "history": history,
    }
    if rank == 0:
        write_json(output / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
    if distributed:
        evaluator.close()
        dist.destroy_process_group()
    return result


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--corpus", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seed", type=int)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--corpus", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("eval")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--corpus", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    worker = subparsers.add_parser("eval-worker")
    worker.add_argument("--rank", type=int, required=True)
    worker.add_argument("--world-size", type=int, required=True)
    worker.add_argument("--init-file", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "eval":
        evaluate(args.checkpoint, args.corpus, args.output)
        return
    if args.command == "eval-worker":
        evaluator_worker(args.rank, args.world_size, args.init_file)
        return
    if args.command == "smoke":
        make_synthetic_corpus(args.corpus)
    result = train(
        args.config,
        args.corpus,
        args.output,
        getattr(args, "seed", None),
    )
    if args.command == "smoke":
        checkpoint = args.output / f"step-{result['step']:08d}"
        repeated = run_evaluator(
            checkpoint, args.corpus, checkpoint / "repeat-evaluation.json"
        )
        delta = abs(repeated["contact_p_at_l"] - result["contact_p_at_l"])
        if delta > 1e-7:
            raise RuntimeError(f"evaluator jitter is too high: {delta}")
        print(f"smoke passed; evaluator max delta={delta}")


if __name__ == "__main__":
    main()
