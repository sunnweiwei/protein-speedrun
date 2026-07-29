#!/usr/bin/env python
"""Aggregate official seeds into the single leaderboard record time."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRACK_ROOT = Path(__file__).resolve().parents[1]
if str(TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_ROOT))

from speedrun.contact_eval import load_protocol
from speedrun.stability import summarize_official_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    paths = sorted(args.runs.rglob("result.json"))
    results = [json.loads(path.read_text()) for path in paths]
    summary = summarize_official_results(results, protocol)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(rendered)
        temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
