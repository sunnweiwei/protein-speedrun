"""The sole pretraining metric: long-range contact precision at L."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from speedrun.checkpoint import load_model
from speedrun.corpus import PAD_TOKEN_ID, PackedSequences


def _encode_one(
    model: nn.Module,
    sequence: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    tokens = torch.from_numpy(sequence.astype(np.int64))[None].to(device)
    padding_mask = torch.ones_like(tokens, dtype=torch.bool)
    with torch.inference_mode():
        embeddings = model.encode(tokens, padding_mask)
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 3
        or embeddings.shape[:2] != tokens.shape
        or embeddings.shape[2] < 1
        or not torch.isfinite(embeddings).all()
    ):
        raise ValueError("encode() must return finite [B, L, D] embeddings")
    embeddings = embeddings[0].float()
    return nn.functional.normalize(embeddings, dim=-1)


def _eligible_pairs(
    length: int, resolved: torch.Tensor, separation: int
) -> tuple[torch.Tensor, torch.Tensor]:
    left, right = torch.triu_indices(
        length, length, offset=separation, device=resolved.device
    )
    valid = resolved[left] & resolved[right]
    return left[valid], right[valid]


def _labels(
    coords: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    distance_threshold: float,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(coords[left] - coords[right], dim=-1)
    return distance < distance_threshold


def _sample_probe_pairs(
    embeddings: torch.Tensor,
    coords: np.ndarray,
    resolved: np.ndarray,
    *,
    separation: int,
    distance_threshold: float,
    max_pairs: int,
    negative_to_positive_ratio: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    coordinate_tensor = torch.from_numpy(coords).to(embeddings.device)
    resolved_tensor = torch.from_numpy(resolved).to(embeddings.device)
    left, right = _eligible_pairs(len(embeddings), resolved_tensor, separation)
    if not len(left):
        return None
    labels = _labels(
        coordinate_tensor, left, right, distance_threshold
    )
    positives = torch.nonzero(labels, as_tuple=False).flatten()
    negatives = torch.nonzero(~labels, as_tuple=False).flatten()
    if not len(positives) or not len(negatives):
        return None
    positive_limit = max(
        1, int(max_pairs / (1.0 + negative_to_positive_ratio))
    )
    positive_count = min(len(positives), positive_limit)
    negative_count = min(
        len(negatives),
        int(math.ceil(positive_count * negative_to_positive_ratio)),
    )
    positives = positives[
        torch.randperm(len(positives), generator=generator)[:positive_count]
    ]
    negatives = negatives[
        torch.randperm(len(negatives), generator=generator)[:negative_count]
    ]
    selected = torch.cat([positives, negatives])
    pair_left = left[selected]
    pair_right = right[selected]
    features = embeddings[pair_left] * embeddings[pair_right]
    targets = labels[selected].float()
    return features, targets


def _train_probe(
    model: nn.Module,
    split: PackedSequences,
    protocol: dict[str, Any],
    *,
    device: torch.device,
) -> nn.Linear:
    metric = protocol["metric"]
    probe_config = protocol["probe"]
    cpu_generator = torch.Generator(device="cpu").manual_seed(
        int(probe_config["seed"])
    )
    features = []
    targets = []
    embedding_dimension = None
    for index in range(len(split)):
        sequence = split.sequence(index)
        coords, resolved = split.structure(index)
        embeddings = _encode_one(model, sequence, device=device)
        embedding_dimension = int(embeddings.shape[-1])
        sampled = _sample_probe_pairs(
            embeddings.cpu(),
            coords,
            resolved,
            separation=int(metric["sequence_separation_min"]),
            distance_threshold=float(metric["contact_distance_angstrom"]),
            max_pairs=int(probe_config["max_pairs_per_protein"]),
            negative_to_positive_ratio=float(
                probe_config["negative_to_positive_ratio"]
            ),
            generator=cpu_generator,
        )
        if sampled is not None:
            features.append(sampled[0])
            targets.append(sampled[1])
    if embedding_dimension is None or not features:
        raise ValueError("contact probe training split has no usable pairs")
    feature_tensor = torch.cat(features).to(device)
    target_tensor = torch.cat(targets).to(device)

    torch.manual_seed(int(probe_config["seed"]))
    probe = nn.Linear(embedding_dimension, 1).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(probe_config["learning_rate"]),
        weight_decay=float(probe_config["weight_decay"]),
    )
    for _ in range(int(probe_config["steps"])):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(feature_tensor).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target_tensor)
        loss.backward()
        optimizer.step()
    probe.eval()
    return probe


def _pair_scores(
    embeddings: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    probe: nn.Linear,
    *,
    batch_size: int = 65536,
) -> torch.Tensor:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(left), batch_size):
            end = min(start + batch_size, len(left))
            features = embeddings[left[start:end]] * embeddings[right[start:end]]
            outputs.append(probe(features).squeeze(-1))
    return torch.cat(outputs)


def _evaluate_probe(
    model: nn.Module,
    probe: nn.Linear,
    split: PackedSequences,
    protocol: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[float, int]:
    metric = protocol["metric"]
    values = []
    for index in range(len(split)):
        sequence = split.sequence(index)
        coords, resolved = split.structure(index)
        embeddings = _encode_one(model, sequence, device=device)
        coords_tensor = torch.from_numpy(coords).to(device)
        resolved_tensor = torch.from_numpy(resolved).to(device)
        left, right = _eligible_pairs(
            len(sequence),
            resolved_tensor,
            int(metric["sequence_separation_min"]),
        )
        if not len(left):
            continue
        labels = _labels(
            coords_tensor,
            left,
            right,
            float(metric["contact_distance_angstrom"]),
        )
        minimum_true = math.ceil(
            len(sequence) * float(metric["minimum_true_contacts_per_residue"])
        )
        if int(labels.sum()) < minimum_true:
            continue
        scores = _pair_scores(embeddings, left, right, probe)
        top_count = min(
            len(scores),
            max(
                1,
                int(
                    round(
                        len(sequence)
                        * float(metric["top_predictions_per_residue"])
                    )
                ),
            ),
        )
        top_indices = torch.topk(scores, top_count, sorted=False).indices
        values.append(float(labels[top_indices].float().mean().cpu()))
    if not values:
        raise ValueError("contact evaluation split has no eligible proteins")
    return float(np.mean(values)), len(values)


def evaluate_checkpoint(
    checkpoint_dir: Path,
    *,
    corpus,
    protocol: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model, metadata = load_model(checkpoint_dir, device)
    if metadata["corpus_sha256"] != corpus.sha256:
        raise ValueError("checkpoint was trained on a different corpus")
    probe = _train_probe(
        model,
        corpus.probe_train,
        protocol,
        device=device,
    )
    score, proteins = _evaluate_probe(
        model,
        probe,
        corpus.probe_eval,
        protocol,
        device=device,
    )
    result = {
        "schema_version": 1,
        "metric": protocol["metric"]["name"],
        "value": score,
        "higher_is_better": True,
        "eligible_proteins": proteins,
        "checkpoint_step": metadata["step"],
        "checkpoint_training_seconds": metadata["training_seconds"],
    }
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("invalid Contact P@L")
    return result


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported protocol schema")
    if payload.get("metric", {}).get("name") != "long_range_contact_p_at_l":
        raise ValueError("protocol must define the sole Contact P@L metric")
    target = payload.get("target", {})
    target_status = target.get("status")
    target_value = target.get("value")
    if target_status == "calibration_required":
        if target_value is not None:
            raise ValueError("an uncalibrated protocol target must be null")
    elif target_status == "frozen":
        if (
            not isinstance(target_value, (int, float))
            or isinstance(target_value, bool)
            or not math.isfinite(target_value)
            or not 0.0 <= target_value <= 1.0
        ):
            raise ValueError("a frozen protocol target must be a finite fraction")
    else:
        raise ValueError("protocol target status must be calibration_required or frozen")
    stability = payload.get("stability", {})
    consecutive = stability.get("consecutive_checkpoint_passes")
    if not isinstance(consecutive, int) or isinstance(consecutive, bool) or consecutive < 2:
        raise ValueError("protocol must require at least two consecutive passes")
    hardware = payload.get("hardware", {})
    if hardware.get("accelerators_per_run") != 1:
        raise ValueError("v0 protocol must assign exactly one accelerator per run")
    if not isinstance(hardware.get("accelerator_name_contains"), str):
        raise ValueError("protocol must pin an accelerator name")
    runtime = payload.get("runtime", {})
    if not isinstance(runtime.get("torch_version"), str):
        raise ValueError("protocol must pin the PyTorch version")
    if not isinstance(runtime.get("cuda_version"), str):
        raise ValueError("protocol must pin the CUDA version")
    if not isinstance(runtime.get("cudnn_version"), int):
        raise ValueError("protocol must pin the cuDNN version")
    if not str(runtime.get("base_image_digest", "")).startswith("sha256:"):
        raise ValueError("protocol must pin the base image digest")
    if not str(runtime.get("container_image_id", "")).startswith("sha256:"):
        raise ValueError("protocol must pin the built container image")
    return payload
