#!/usr/bin/env python
"""Isolated one-checkpoint evaluator used by the trusted timed runner."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but no GPU is assigned")
    result = evaluate_checkpoint(
        args.checkpoint,
        corpus=load_corpus(args.corpus),
        protocol=load_protocol(args.protocol),
        device=device,
    )
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
