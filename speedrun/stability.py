"""Predeclared stable target-crossing and multi-seed rules."""

from __future__ import annotations

import math
import statistics
from typing import Any


def confirmed_time_to_target(
    history: list[dict[str, Any]],
    *,
    target: float | None,
    consecutive_passes: int,
) -> float | None:
    if target is None:
        return None
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("target must be null or a finite fraction")
    if consecutive_passes < 1:
        raise ValueError("consecutive_passes must be positive")
    streak = 0
    previous_step = -1
    for item in history:
        step = item.get("step")
        value = item.get("contact_p_at_l")
        seconds = item.get("training_seconds")
        if (
            not isinstance(step, int)
            or step <= previous_step
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError("invalid checkpoint history")
        previous_step = step
        streak = streak + 1 if value >= target else 0
        if streak >= consecutive_passes:
            return float(seconds)
    return None


def summarize_seed_results(
    results: list[dict[str, Any]],
    *,
    official_seeds: list[int],
    required_seed_passes: int,
) -> dict[str, Any]:
    by_seed = {}
    for result in results:
        seed = result.get("seed")
        if seed in by_seed:
            raise ValueError(f"duplicate result for seed {seed}")
        by_seed[seed] = result
    selected = [by_seed[seed] for seed in official_seeds if seed in by_seed]
    confirmed = [
        float(result["confirmed_seconds_to_target"])
        for result in selected
        if isinstance(result.get("confirmed_seconds_to_target"), (int, float))
        and math.isfinite(result["confirmed_seconds_to_target"])
    ]
    qualified = (
        len(selected) == len(official_seeds)
        and len(confirmed) >= required_seed_passes
    )
    return {
        "schema_version": 1,
        "official_seed_count": len(official_seeds),
        "submitted_seed_count": len(selected),
        "confirmed_seed_count": len(confirmed),
        "required_seed_passes": required_seed_passes,
        "qualified": qualified,
        "median_confirmed_seconds_to_target": (
            statistics.median(confirmed) if qualified else None
        ),
    }


def summarize_official_results(
    results: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    """Reject mixed experiments before emitting the one leaderboard number."""
    target = protocol["target"]["value"]
    if target is None:
        raise ValueError("cannot create an official record before target calibration")
    identity_fields = (
        "candidate_id",
        "objective",
        "corpus_sha256",
        "target_contact_p_at_l",
        "consecutive_checkpoint_passes",
        "protocol_sha256",
        "candidate_config_sha256",
        "train_code_sha256",
        "model_code_sha256",
    )
    identities = set()
    for result in results:
        try:
            identity = tuple(result[field] for field in identity_fields)
        except KeyError as error:
            raise ValueError(f"result is missing identity field {error.args[0]}") from error
        identities.add(identity)
        if result["target_contact_p_at_l"] != target:
            raise ValueError("result target does not match the frozen protocol")
        if result["consecutive_checkpoint_passes"] != protocol["stability"][
            "consecutive_checkpoint_passes"
        ]:
            raise ValueError("result confirmation rule does not match the protocol")
        hardware = result.get("hardware", {})
        if (
            hardware.get("accelerator_count")
            != protocol["hardware"]["accelerators_per_run"]
            or protocol["hardware"]["accelerator_name_contains"]
            not in str(hardware.get("accelerator_name", ""))
        ):
            raise ValueError("result hardware does not match the protocol")
    if len(identities) != 1:
        raise ValueError("official record cannot mix candidates or corpora")

    summary = summarize_seed_results(
        results,
        official_seeds=protocol["stability"]["official_seeds"],
        required_seed_passes=int(
            protocol["stability"]["required_seed_passes"]
        ),
    )
    identity = next(iter(identities))
    summary.update(dict(zip(identity_fields, identity, strict=True)))
    return summary
