"""Strict, code-plus-weights checkpoint contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

REQUIRED_METADATA_FIELDS = {
    "schema_version",
    "candidate_id",
    "seed",
    "step",
    "tokens_seen",
    "training_seconds",
    "objective",
    "corpus_sha256",
    "model_config_sha256",
    "weights_sha256",
    "code_sha256",
}


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def write_checkpoint(
    directory: Path,
    *,
    model: torch.nn.Module,
    model_config: dict[str, Any],
    candidate_id: str,
    seed: int,
    step: int,
    tokens_seen: int,
    training_seconds: float,
    objective: str,
    corpus_sha256: str,
    model_code_path: Path,
) -> Path:
    output = directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    weights_path = output / "weights.pt"
    model_config_path = output / "model_config.json"
    model_code_snapshot = output / "model.py"
    metadata_path = output / "checkpoint.json"

    state_dict = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    torch.save(state_dict, weights_path)
    _atomic_json(model_config_path, model_config)
    model_code_snapshot.write_bytes(model_code_path.read_bytes())
    metadata = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "seed": seed,
        "step": step,
        "tokens_seen": tokens_seen,
        "training_seconds": training_seconds,
        "objective": objective,
        "corpus_sha256": corpus_sha256,
        "model_config_sha256": canonical_json_sha256(model_config),
        "weights_sha256": sha256_file(weights_path),
        "code_sha256": sha256_file(model_code_snapshot),
    }
    _atomic_json(metadata_path, metadata)
    return output


def load_metadata(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = directory / "checkpoint.json"
    model_config_path = directory / "model_config.json"
    if not metadata_path.is_file() or not model_config_path.is_file():
        raise ValueError(f"{directory}: missing checkpoint metadata")
    metadata = json.loads(metadata_path.read_text())
    model_config = json.loads(model_config_path.read_text())
    if not isinstance(metadata, dict) or set(metadata) != REQUIRED_METADATA_FIELDS:
        raise ValueError(f"{directory}: checkpoint metadata fields do not match schema")
    if metadata["schema_version"] != 1:
        raise ValueError(f"{directory}: unsupported checkpoint schema")
    for name in ("seed", "step", "tokens_seen"):
        value = metadata[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{directory}: invalid {name}")
    training_seconds = metadata["training_seconds"]
    if (
        not isinstance(training_seconds, (int, float))
        or isinstance(training_seconds, bool)
        or not math.isfinite(training_seconds)
        or training_seconds < 0
    ):
        raise ValueError(f"{directory}: invalid training_seconds")
    for name in (
        "candidate_id",
        "objective",
        "corpus_sha256",
        "model_config_sha256",
        "weights_sha256",
        "code_sha256",
    ):
        if not isinstance(metadata[name], str) or not metadata[name]:
            raise ValueError(f"{directory}: invalid {name}")
    if canonical_json_sha256(model_config) != metadata["model_config_sha256"]:
        raise ValueError(f"{directory}: model config digest mismatch")
    weights_path = directory / "weights.pt"
    if not weights_path.is_file() or sha256_file(weights_path) != metadata["weights_sha256"]:
        raise ValueError(f"{directory}: weights digest mismatch")
    return metadata, model_config


def load_submission_module(path: Path) -> ModuleType:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"submission model code does not exist: {resolved}")
    spec = importlib.util.spec_from_file_location("speedrun_submission_model", resolved)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load submission module: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build_model", None)):
        raise ValueError("submission/model.py must export build_model(config)")
    return module


def load_model(
    directory: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    metadata, model_config = load_metadata(directory)
    model_code_path = directory / "model.py"
    if (
        not model_code_path.is_file()
        or sha256_file(model_code_path) != metadata["code_sha256"]
    ):
        raise ValueError(f"{directory}: checkpoint model code digest mismatch")
    module = load_submission_module(model_code_path)
    model = module.build_model(model_config)
    if not isinstance(model, torch.nn.Module):
        raise ValueError("build_model must return torch.nn.Module")
    if not callable(getattr(model, "encode", None)):
        raise ValueError("submission model must implement encode(tokens, padding_mask)")
    state_dict = torch.load(
        directory / "weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state_dict, dict):
        raise ValueError("weights.pt must contain a state_dict")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, metadata
