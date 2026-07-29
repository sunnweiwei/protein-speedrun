#!/usr/bin/env python
"""Run the trusted external metric repeatedly to measure evaluator jitter."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

TRACK_ROOT = Path(__file__).resolve().parents[1]
if str(TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_ROOT))

from speedrun.contact_eval import evaluate_checkpoint, load_protocol
from speedrun.corpus import load_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeat < 2:
        raise ValueError("repeat must be at least two")
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but no GPU is assigned")
    corpus = load_corpus(args.corpus)
    protocol = load_protocol(args.protocol)
    evaluations = [
        evaluate_checkpoint(
            args.checkpoint,
            corpus=corpus,
            protocol=protocol,
            device=device,
        )
        for _ in range(args.repeat)
    ]
    scores = [float(item["value"]) for item in evaluations]
    spread = max(scores) - min(scores)
    result = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "metric": protocol["metric"]["name"],
        "scores": scores,
        "max_minus_min": spread,
        "tolerance": args.tolerance,
        "stable": spread <= args.tolerance,
        "eligible_proteins": evaluations[0]["eligible_proteins"],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(
                f"refusing to overwrite verification result: {args.output}"
            )
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(rendered)
        temporary.replace(args.output)
    print(rendered, end="")
    if not result["stable"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
