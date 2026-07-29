#!/usr/bin/env python
"""Minimal protein pretraining speedrun: train, checkpoint, and Contact P@L."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

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


def load_corpus(path: Path) -> tuple[str, Packed, Packed, Packed]:
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
    torch.manual_seed(PROBE_SEED)
    probe = nn.Linear(int(width), 1).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.03, weight_decay=1e-4)
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(
            probe(x).squeeze(-1), y
        )
        loss.backward()
        optimizer.step()
    return probe.eval()


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
    sequences = [split.sequence(int(index))[:max_length] for index in indices]
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


def train(
    config_path: Path, corpus_path: Path, output: Path, seed_override: int | None
):
    config = json.loads(config_path.read_text())
    if seed_override is not None:
        config["seed"] = seed_override
    if torch.cuda.device_count() != 1 or "H100 80GB" not in torch.cuda.get_device_name(0):
        raise RuntimeError("the v0 test requires exactly one visible H100 80GB")
    corpus_sha, train_split, _, _ = load_corpus(corpus_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"run output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = ROOT / "candidate.py"
    schedule = [0, *range(config["checkpoint_every"], config["steps"] + 1, config["checkpoint_every"])]
    if schedule[-1] != config["steps"]:
        schedule.append(config["steps"])

    timer = time.perf_counter
    started = timer()
    candidate = load_candidate(candidate_path, "training_candidate")
    device = torch.device("cuda")
    torch.manual_seed(config["seed"])
    model = candidate.build_model(config["model"]).to(device)
    optimizer = candidate.build_optimizer(model, config)
    generator = torch.Generator(device=device).manual_seed(config["seed"] + 1)
    rng = np.random.default_rng(config["seed"])
    excluded = 0.0
    tokens_seen = 0
    history = []
    streak = 0

    def checkpoint(step):
        nonlocal excluded, streak
        torch.cuda.synchronize()
        callback_started = timer()
        training_seconds = callback_started - started - excluded
        directory = output / f"step-{step:08d}"
        save_checkpoint(
            directory,
            model,
            config["model"],
            candidate_path,
            {
                "step": step,
                "tokens_seen": tokens_seen,
                "training_seconds": training_seconds,
                "corpus_sha256": corpus_sha,
            },
        )
        result = run_evaluator(
            directory, corpus_path, directory / "evaluation.json"
        )
        history.append(
            {
                "step": step,
                "training_seconds": training_seconds,
                "tokens_seen": tokens_seen,
                **result,
            }
        )
        target = config["target_contact_p_at_l"]
        streak = streak + 1 if target is not None and result["contact_p_at_l"] >= target else 0
        torch.cuda.synchronize()
        excluded += timer() - callback_started
        return streak >= 2

    stopped = checkpoint(0)
    loss_value = None
    for step in range(1, config["steps"] + 1):
        if stopped:
            break
        model.train()
        tokens, mask = batch(
            train_split,
            config["batch_size"],
            config["train_max_length"],
            rng,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = candidate.training_loss(model, tokens, mask, config, generator)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens_seen += int(mask.sum())
        loss_value = float(loss.detach().cpu())
        if step in schedule[1:]:
            stopped = checkpoint(step)
    torch.cuda.synchronize()
    result = {
        "status": "confirmed" if stopped else "calibration",
        "seed": config["seed"],
        "contact_p_at_l": history[-1]["contact_p_at_l"],
        "training_seconds": history[-1]["training_seconds"],
        "step": history[-1]["step"],
        "tokens_seen": tokens_seen,
        "last_loss": loss_value,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "corpus_sha256": corpus_sha,
        "history": history,
    }
    write_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
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
    args = parser.parse_args()

    if args.command == "eval":
        evaluate(args.checkpoint, args.corpus, args.output)
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
