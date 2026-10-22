"""Create a short animated result walkthrough from a demo metrics.json file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def frame(title: str, subtitle: str, values: list[tuple[str, float, str]], highlight: int) -> Image.Image:
    image = Image.new("RGB", (900, 500), "#f5f8fa")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((35, 35, 865, 465), radius=18, fill="#ffffff", outline="#d6e1e6", width=2)
    draw.text((70, 70), title, font=font(34), fill="#183746")
    draw.text((70, 122), subtitle, font=font(18), fill="#58707d")
    max_value = max(value for _, value, _ in values) * 1.15
    for i, (name, value, color) in enumerate(values):
        y = 195 + i * 75
        draw.text((75, y), name, font=font(24), fill="#183746")
        width = int(520 * value / max_value)
        draw.rounded_rectangle((230, y - 2, 230 + width, y + 36), radius=7, fill=color)
        draw.text((760, y), f"{value:.4f}", font=font(24), fill="#183746")
        if i == highlight:
            draw.rounded_rectangle((60, y - 12, 830, y + 48), radius=10, outline="#0c9a91", width=3)
    draw.text((75, 435), "NLL lower is better  |  same held-out test set", font=font(15), fill="#58707d")
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    scores = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))
    names = list(scores)
    colors = ["#748894", "#397b9d", "#0c9a91"]
    values = [(name, scores[name]["nll"], colors[i]) for i, name in enumerate(names)]
    frames = [
        frame("LAPE on real model outputs", "Anchor calibration followed by local refinement", values, -1),
        frame("Step 1: base classifier", "Frozen features and logits; no test-label fitting", values, 0),
        frame("Step 2: fit the anchor", "Calibration labels fit the global anchor", values, 1),
        frame("Step 3: local refinement", "OOF anchor scores and reference neighbors feed LAPE", values, 2),
    ]
    path = args.output_dir / "walkthrough.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[1100, 1300, 1300, 2400], loop=0, optimize=True)
    print(path)


if __name__ == "__main__":
    main()
