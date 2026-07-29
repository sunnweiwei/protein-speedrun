#!/usr/bin/env python
"""Recommend, but never silently freeze, a stable time-to-target threshold."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

TRACK_ROOT = Path(__file__).resolve().parents[1]
if str(TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_ROOT))

from speedrun.checkpoint import canonical_json_sha256
from speedrun.contact_eval import load_protocol
from speedrun.stability import confirmed_time_to_target


def recommend_target(
    histories: dict[int, list[dict[str, Any]]],
    *,
    consecutive_passes: int,
    initial_margin: float,
    final_margin: float,
    quantum: float,
) -> dict[str, Any]:
    for name, value in (
        ("initial_margin", initial_margin),
        ("final_margin", final_margin),
        ("quantum", quantum),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not histories:
        raise ValueError("no calibration histories")
    for seed, history in histories.items():
        if not history or history[0].get("step") != 0:
            raise ValueError(f"seed {seed} is missing the step-0 baseline")

    initial_max = max(
        float(history[0]["contact_p_at_l"]) for history in histories.values()
    )
    final_min = min(
        float(history[-1]["contact_p_at_l"]) for history in histories.values()
    )
    lower_bound = initial_max + initial_margin
    upper_bound = final_min - final_margin
    maximum_tick = math.floor((upper_bound + 1e-12) / quantum)
    minimum_tick = math.floor(lower_bound / quantum) + 1

    selected = None
    confirmations = None
    for tick in range(maximum_tick, minimum_tick - 1, -1):
        candidate = round(tick * quantum, 12)
        candidate_confirmations = {
            seed: confirmed_time_to_target(
                history,
                target=candidate,
                consecutive_passes=consecutive_passes,
            )
            for seed, history in histories.items()
        }
        if all(value is not None for value in candidate_confirmations.values()):
            selected = candidate
            confirmations = candidate_confirmations
            break
    if selected is None or confirmations is None:
        raise ValueError(
            "no target satisfies initial/final margins and all-seed confirmation"
        )
    return {
        "recommended_target_contact_p_at_l": selected,
        "initial_score_max": initial_max,
        "required_initial_margin": initial_margin,
        "final_score_min": final_min,
        "required_final_margin": final_margin,
        "quantum": quantum,
        "confirmation_seconds_by_seed": {
            str(seed): confirmations[seed] for seed in sorted(confirmations)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--initial-margin", required=True, type=float)
    parser.add_argument("--final-margin", required=True, type=float)
    parser.add_argument("--quantum", type=float, default=0.0001)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if protocol["target"]["status"] != "calibration_required":
        raise ValueError("target calibration requires an unfrozen protocol")
    expected_protocol_sha = canonical_json_sha256(protocol)
    official_seeds = protocol["stability"]["official_seeds"]

    results = []
    histories: dict[int, list[dict[str, Any]]] = {}
    for result_path in sorted(args.runs.rglob("result.json")):
        result = json.loads(result_path.read_text())
        seed = result.get("seed")
        if seed not in official_seeds:
            continue
        if seed in histories:
            raise ValueError(f"duplicate calibration result for seed {seed}")
        if result.get("status") != "calibration":
            raise ValueError(f"{result_path}: expected calibration status")
        if result.get("protocol_sha256") != expected_protocol_sha:
            raise ValueError(f"{result_path}: protocol digest mismatch")
        metrics_path = result_path.with_name("metrics.jsonl")
        history = [
            json.loads(line)
            for line in metrics_path.read_text().splitlines()
            if line.strip()
        ]
        histories[int(seed)] = history
        results.append(result)
    if set(histories) != set(official_seeds):
        missing = sorted(set(official_seeds) - set(histories))
        raise ValueError(f"missing official calibration seeds: {missing}")
    identity_fields = (
        "candidate_id",
        "corpus_sha256",
        "candidate_config_sha256",
        "train_code_sha256",
        "model_code_sha256",
    )
    identity = {
        tuple(result[field] for field in identity_fields)
        for result in results
    }
    if len(identity) != 1:
        raise ValueError("calibration cannot mix candidates, corpora, or training code")

    recommendation = recommend_target(
        histories,
        consecutive_passes=int(
            protocol["stability"]["consecutive_checkpoint_passes"]
        ),
        initial_margin=args.initial_margin,
        final_margin=args.final_margin,
        quantum=args.quantum,
    )
    identity_values = next(iter(identity))
    output = {
        "schema_version": 1,
        "status": "recommendation_only",
        **dict(zip(identity_fields, identity_values, strict=True)),
        "official_seeds": official_seeds,
        **recommendation,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
