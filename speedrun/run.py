#!/usr/bin/env python
"""Trusted v0 timed runner around an editable submission."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

TRACK_ROOT = Path(__file__).resolve().parents[1]
if str(TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_ROOT))

from speedrun.checkpoint import (
    canonical_json_sha256,
    sha256_file,
    write_checkpoint,
)
from speedrun.contact_eval import load_protocol
from speedrun.corpus import load_corpus
from speedrun.stability import confirmed_time_to_target


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "candidate_id",
        "seed",
        "objective",
        "model",
        "training",
        "evaluation",
    }
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError("run config fields do not match schema v1")
    if not isinstance(config["candidate_id"], str) or not config["candidate_id"]:
        raise ValueError("candidate_id must be a non-empty string")
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    training = config["training"]
    for name in ("steps", "checkpoint_every", "batch_size", "max_length"):
        if not isinstance(training.get(name), int) or training[name] < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    if training["checkpoint_every"] > training["steps"]:
        raise ValueError("checkpoint_every cannot exceed steps")
    target = config["evaluation"].get("target_contact_p_at_l")
    if target is not None and (
        not isinstance(target, (int, float))
        or isinstance(target, bool)
        or not math.isfinite(target)
        or not 0.0 <= target <= 1.0
    ):
        raise ValueError("target_contact_p_at_l must be null or a fraction")
    consecutive = config["evaluation"].get("consecutive_checkpoint_passes")
    if (
        not isinstance(consecutive, int)
        or isinstance(consecutive, bool)
        or consecutive < 2
    ):
        raise ValueError("consecutive_checkpoint_passes must be at least two")


def _load_train_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("speedrun_submission_train", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load submission training module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "train", None)):
        raise ValueError("submission/train.py must export train()")
    return module


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def run(
    *,
    config_path: Path,
    protocol_path: Path,
    corpus_path: Path,
    output: Path,
    seed_override: int | None = None,
) -> dict[str, Any]:
    config = _load_json(config_path)
    if seed_override is not None:
        config["seed"] = seed_override
    _validate_config(config)
    protocol = load_protocol(protocol_path)
    corpus = load_corpus(corpus_path)
    if (
        config["training"]["device"] != "cuda"
        or config["evaluation"]["device"] != "cuda"
    ):
        raise ValueError("official v0 training and evaluation both require CUDA")
    expected_accelerators = int(protocol["hardware"]["accelerators_per_run"])
    if torch.cuda.device_count() != expected_accelerators:
        raise RuntimeError(
            f"protocol requires {expected_accelerators} visible accelerator"
        )
    accelerator_name = torch.cuda.get_device_name(0)
    if protocol["hardware"]["accelerator_name_contains"] not in accelerator_name:
        raise RuntimeError(
            f"protocol requires {protocol['hardware']['accelerator_name_contains']}, "
            f"found {accelerator_name}"
        )
    runtime = protocol["runtime"]
    actual_runtime = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "base_image_digest": runtime["base_image_digest"],
        "container_image_id": os.environ.get("SPEEDRUN_IMAGE_ID"),
    }
    for field in (
        "torch_version",
        "cuda_version",
        "cudnn_version",
        "container_image_id",
    ):
        if actual_runtime[field] != runtime[field]:
            raise RuntimeError(
                f"protocol requires runtime {field}={runtime[field]}, "
                f"found {actual_runtime[field]}"
            )
    protocol_target = protocol["target"]["value"]
    if config["evaluation"]["target_contact_p_at_l"] != protocol_target:
        raise ValueError("run target must exactly match the trusted protocol")
    protocol_consecutive = int(
        protocol["stability"]["consecutive_checkpoint_passes"]
    )
    if (
        int(config["evaluation"]["consecutive_checkpoint_passes"])
        != protocol_consecutive
    ):
        raise ValueError(
            "target confirmation rule must exactly match the trusted protocol"
        )

    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to mix with existing run output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "run_config.json", config)
    _atomic_json(output / "protocol.json", protocol)
    checkpoint_root = output / "checkpoints"
    checkpoint_root.mkdir()

    model_code_path = TRACK_ROOT / "submission" / "model.py"
    train_code_path = TRACK_ROOT / "submission" / "train.py"
    train_snapshot_path = output / "train.py"
    train_snapshot_path.write_bytes(train_code_path.read_bytes())
    evaluation_device_name = str(config["evaluation"]["device"])
    if evaluation_device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but no GPU is assigned")
    evaluation_device = torch.device(evaluation_device_name)
    target = protocol_target
    consecutive = protocol_consecutive
    steps = int(config["training"]["steps"])
    checkpoint_every = int(config["training"]["checkpoint_every"])
    checkpoint_schedule = [0]
    checkpoint_schedule.extend(range(checkpoint_every, steps + 1, checkpoint_every))
    if checkpoint_schedule[-1] != steps:
        checkpoint_schedule.append(steps)

    history: list[dict[str, Any]] = []
    excluded_seconds = 0.0
    trusted_timer = time.perf_counter
    trusted_subprocess_run = subprocess.run
    trusted_python = sys.executable
    stop_requested = False

    def checkpoint_callback(
        model: torch.nn.Module, step: int, tokens_seen: int
    ) -> bool:
        nonlocal excluded_seconds, stop_requested
        if stop_requested:
            raise RuntimeError("submission continued after target confirmation")
        schedule_index = len(history)
        if (
            schedule_index >= len(checkpoint_schedule)
            or step != checkpoint_schedule[schedule_index]
        ):
            expected = (
                checkpoint_schedule[schedule_index]
                if schedule_index < len(checkpoint_schedule)
                else None
            )
            raise ValueError(
                f"checkpoint step {step} violates trusted schedule; expected {expected}"
            )
        if (
            not isinstance(tokens_seen, int)
            or isinstance(tokens_seen, bool)
            or tokens_seen < 0
            or (history and tokens_seen < history[-1]["tokens_seen"])
        ):
            raise ValueError("tokens_seen must be a monotonic non-negative integer")
        torch.cuda.synchronize()
        callback_started = trusted_timer()
        training_seconds = callback_started - run_started - excluded_seconds
        checkpoint_dir = checkpoint_root / f"step-{step:08d}"
        write_checkpoint(
            checkpoint_dir,
            model=model,
            model_config=config["model"],
            candidate_id=config["candidate_id"],
            seed=config["seed"],
            step=step,
            tokens_seen=tokens_seen,
            training_seconds=training_seconds,
            objective=config["objective"],
            corpus_sha256=corpus.sha256,
            model_code_path=model_code_path,
        )
        evaluation_started = trusted_timer()
        evaluation_path = checkpoint_dir / "evaluation.json"
        completed = trusted_subprocess_run(
            [
                trusted_python,
                str(TRACK_ROOT / "speedrun" / "evaluate_once.py"),
                "--checkpoint",
                str(checkpoint_dir),
                "--corpus",
                str(corpus.path),
                "--protocol",
                str(protocol_path),
                "--output",
                str(evaluation_path),
                "--device",
                str(evaluation_device),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                "external evaluator failed:\n"
                + completed.stderr[-4000:]
            )
        evaluation = json.loads(evaluation_path.read_text())
        evaluation_seconds = trusted_timer() - evaluation_started
        history.append(
            {
                "step": step,
                "tokens_seen": tokens_seen,
                "training_seconds": training_seconds,
                "evaluation_seconds": evaluation_seconds,
                "contact_p_at_l": evaluation["value"],
                "eligible_proteins": evaluation["eligible_proteins"],
                "checkpoint": str(checkpoint_dir),
            }
        )
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(history[-1], sort_keys=True) + "\n")
        confirmed = confirmed_time_to_target(
            history,
            target=target,
            consecutive_passes=consecutive,
        )
        stop_requested = confirmed is not None
        torch.cuda.synchronize()
        excluded_seconds += trusted_timer() - callback_started
        return confirmed is not None

    run_started = trusted_timer()
    train_module = _load_train_module(train_code_path)
    training_summary = train_module.train(
        config,
        corpus.train,
        checkpoint_callback,
    )
    torch.cuda.synchronize()
    total_training_seconds = trusted_timer() - run_started - excluded_seconds
    confirmed = confirmed_time_to_target(
        history,
        target=target,
        consecutive_passes=consecutive,
    )
    if confirmed is None and len(history) != len(checkpoint_schedule):
        raise RuntimeError("submission returned before completing checkpoint schedule")
    if not isinstance(training_summary, dict):
        raise ValueError("submission train() must return a summary object")
    if training_summary.get("last_step") != history[-1]["step"]:
        raise ValueError("training summary last_step does not match final checkpoint")
    if training_summary.get("tokens_seen") != history[-1]["tokens_seen"]:
        raise ValueError("training summary tokens_seen does not match final checkpoint")
    result = {
        "schema_version": 1,
        "candidate_id": config["candidate_id"],
        "seed": config["seed"],
        "objective": config["objective"],
        "corpus_sha256": corpus.sha256,
        "target_contact_p_at_l": target,
        "consecutive_checkpoint_passes": consecutive,
        "status": (
            "calibration"
            if target is None
            else "confirmed"
            if confirmed is not None
            else "not_reached"
        ),
        "confirmed_seconds_to_target": confirmed,
        "total_training_seconds": total_training_seconds,
        "final_contact_p_at_l": history[-1]["contact_p_at_l"],
        "checkpoint_count": len(history),
        "training_summary": training_summary,
        "config_sha256": canonical_json_sha256(config),
        "candidate_config_sha256": canonical_json_sha256(
            {key: value for key, value in config.items() if key != "seed"}
        ),
        "protocol_sha256": canonical_json_sha256(protocol),
        "train_code_sha256": sha256_file(train_snapshot_path),
        "model_code_sha256": sha256_file(model_code_path),
        "hardware": {
            "accelerator_count": torch.cuda.device_count(),
            "accelerator_name": accelerator_name,
        },
        "runtime": actual_runtime,
    }
    _atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        config_path=arguments.config,
        protocol_path=arguments.protocol,
        corpus_path=arguments.corpus,
        output=arguments.output,
        seed_override=arguments.seed,
    )
