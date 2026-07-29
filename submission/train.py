"""Editable reference training algorithm and objectives."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn

from speedrun.corpus import MASK_TOKEN_ID, PAD_TOKEN_ID, PackedSequences
from submission.model import build_model

CheckpointCallback = Callable[[nn.Module, int, int], bool]


def _batch(
    split: PackedSequences,
    *,
    batch_size: int,
    max_length: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = rng.integers(0, len(split), size=batch_size)
    sequences = [split.sequence(int(index))[:max_length] for index in indices]
    length = max(len(sequence) for sequence in sequences)
    tokens = torch.full(
        (batch_size, length),
        PAD_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    padding_mask = torch.zeros(
        (batch_size, length), dtype=torch.bool, device=device
    )
    for row, sequence in enumerate(sequences):
        sequence_tensor = torch.from_numpy(sequence.astype(np.int64)).to(device)
        tokens[row, : len(sequence)] = sequence_tensor
        padding_mask[row, : len(sequence)] = True
    return tokens, padding_mask


def _mlm_loss(
    model: nn.Module,
    tokens: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    probability: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    selected = (
        torch.rand(tokens.shape, device=tokens.device, generator=generator)
        < probability
    ) & padding_mask
    empty_rows = ~selected.any(dim=1)
    if empty_rows.any():
        selected[empty_rows, 0] = True
    corrupted = tokens.clone()
    corrupted[selected] = MASK_TOKEN_ID
    logits = model.logits(corrupted, padding_mask)
    loss = nn.functional.cross_entropy(logits[selected], tokens[selected])
    return loss, int(padding_mask.sum().item())


def _causal_loss(
    model: nn.Module,
    tokens: torch.Tensor,
    padding_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    logits = model.logits(tokens[:, :-1], padding_mask[:, :-1])
    target = tokens[:, 1:]
    valid = padding_mask[:, 1:]
    loss = nn.functional.cross_entropy(logits[valid], target[valid])
    return loss, int(padding_mask.sum().item())


def train(
    config: dict[str, Any],
    train_split: PackedSequences,
    checkpoint_callback: CheckpointCallback,
) -> dict[str, Any]:
    seed = int(config["seed"])
    training = config["training"]
    requested_device = str(training["device"])
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but no GPU is assigned")
    device = torch.device(requested_device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 1)

    model = build_model(config["model"]).to(device)
    optimizer_name = str(training["optimizer"])
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            betas=(0.9, 0.95),
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            momentum=0.9,
        )
    else:
        raise ValueError(f"unsupported reference optimizer: {optimizer_name}")

    objective = str(config["objective"])
    steps = int(training["steps"])
    checkpoint_every = int(training["checkpoint_every"])
    tokens_seen = 0
    last_loss = math.nan
    if checkpoint_callback(model, 0, 0):
        return {
            "last_step": 0,
            "tokens_seen": 0,
            "last_training_loss": None,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        }
    for step in range(1, steps + 1):
        model.train()
        tokens, padding_mask = _batch(
            train_split,
            batch_size=int(training["batch_size"]),
            max_length=int(training["max_length"]),
            rng=rng,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        if objective == "mlm":
            loss, batch_tokens = _mlm_loss(
                model,
                tokens,
                padding_mask,
                probability=float(training["mask_probability"]),
                generator=generator,
            )
        elif objective == "causal":
            loss, batch_tokens = _causal_loss(model, tokens, padding_mask)
        else:
            raise ValueError(f"unsupported reference objective: {objective}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["grad_clip_norm"])
        )
        optimizer.step()
        tokens_seen += batch_tokens
        last_loss = float(loss.detach().cpu())

        if step % checkpoint_every == 0 or step == steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            should_stop = checkpoint_callback(model, step, tokens_seen)
            if should_stop:
                break
    return {
        "last_step": step,
        "tokens_seen": tokens_seen,
        "last_training_loss": last_loss,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
