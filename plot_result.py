#!/usr/bin/env python3
"""Render the one-number speedrun curve without a plotting dependency."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    points = [
        (row["tokens_seen"] / 1_000_000, row["contact_p_at_l"])
        for row in result["history"]
    ]
    width, height = 1000, 620
    left, right, top, bottom = 110, 40, 85, 90
    chart_width = width - left - right
    chart_height = height - top - bottom
    x_max = max(x for x, _ in points) or 1
    values = [y for _, y in points]
    margin = max((max(values) - min(values)) * 0.15, 0.001)
    y_min = max(0, min(values) - margin)
    y_max = max(values) + margin

    def xy(x: float, y: float) -> tuple[float, float]:
        px = left + chart_width * x / x_max
        py = top + chart_height * (y_max - y) / (y_max - y_min)
        return px, py

    image = Image.new("RGB", (width, height), "#fbfcfe")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    bold = ImageFont.load_default(size=22)

    for tick in range(6):
        fraction = tick / 5
        x = x_max * fraction
        px, _ = xy(x, y_min)
        draw.line((px, top, px, height - bottom), fill="#e3e8ef", width=1)
        draw.text((px, height - bottom + 14), f"{x:.0f}", fill="#44546a", font=small, anchor="ma")
        y = y_min + (y_max - y_min) * fraction
        _, py = xy(0, y)
        draw.line((left, py, width - right, py), fill="#e3e8ef", width=1)
        draw.text((left - 15, py), f"{y:.3f}", fill="#44546a", font=small, anchor="rm")

    draw.line((left, top, left, height - bottom), fill="#263238", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="#263238", width=2)
    draw.line([xy(x, y) for x, y in points], fill="#2563eb", width=4, joint="curve")
    for x, y in points:
        px, py = xy(x, y)
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill="#2563eb", outline="white", width=2)

    start, final = values[0], values[-1]
    best_index = max(range(len(points)), key=lambda index: values[index])
    best_x, best_y = xy(*points[best_index])
    draw.ellipse(
        (best_x - 8, best_y - 8, best_x + 8, best_y + 8),
        outline="#0f8a4b",
        width=4,
    )
    draw.text((width / 2, 25), "150M Protein Pretraining Budget Curve", fill="#172033", font=bold, anchor="ma")
    draw.text((width / 2, height - 40), "Training tokens seen (millions)", fill="#263238", font=font, anchor="ma")
    draw.text((left, top - 14), "Contact P@L", fill="#263238", font=font, anchor="lb")
    draw.text(
        (best_x + 12, best_y - 12),
        f"best {values[best_index]:.4f}",
        fill="#096536",
        font=small,
        anchor="lb",
    )
    final_x, final_y = xy(*points[-1])
    draw.text(
        (final_x - 8, final_y - 18),
        f"{final:.4f}  (+{(final / start - 1) * 100:.1f}%)",
        fill="#133f92",
        font=small,
        anchor="rb",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)


if __name__ == "__main__":
    main()
