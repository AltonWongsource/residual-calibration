#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the DTD base-to-new manifest from split_zhou_DescribableTextures.json."
    )
    p.add_argument("--split-json", required=True)
    p.add_argument("--image-dir", required=True, help="DTD images/ directory")
    p.add_argument("--output", required=True)
    p.add_argument("--shots", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def sample_per_class(items, shots: int, rng: random.Random):
    grouped = defaultdict(list)
    for rel_path, label, class_name in items:
        grouped[int(label)].append((rel_path, int(label), str(class_name)))
    out = []
    for label in sorted(grouped):
        candidates = grouped[label]
        out.extend(rng.sample(candidates, min(shots, len(candidates))))
    return out


def main() -> None:
    args = parse_args()
    split_path = Path(args.split_json).resolve()
    image_dir = Path(args.image_dir).resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    for key in ["train", "val", "test"]:
        if key not in split:
            raise ValueError(f"split JSON missing {key!r}")
    all_items = split["train"] + split["val"] + split["test"]
    labels = sorted({int(item[1]) for item in all_items})
    if labels != list(range(max(labels) + 1)):
        raise ValueError("split labels must be contiguous from zero")
    n_base = math.ceil(len(labels) / 2)
    base_labels = set(labels[:n_base])
    novel_labels = set(labels[n_base:])
    rng = random.Random(args.seed)
    base_train_all = [x for x in split["train"] if int(x[1]) in base_labels]
    base_val_all = [x for x in split["val"] if int(x[1]) in base_labels]
    novel_test_all = [x for x in split["test"] if int(x[1]) in novel_labels]
    base_train = sample_per_class(base_train_all, args.shots, rng)
    base_val = sample_per_class(base_val_all, min(args.shots, 4), rng)

    rows = []
    for partition, items in [
        ("base_train", base_train),
        ("base_calibration", base_val),
        ("novel_pool", novel_test_all),
    ]:
        for rel_path, label, class_name in items:
            path = Path(str(rel_path))
            if not path.is_absolute():
                path = image_dir / path
            rows.append(
                {
                    "path": str(path),
                    "global_label": int(label),
                    "class_name": str(class_name),
                    "partition": partition,
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "global_label", "class_name", "partition"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "output": str(output),
                "n_classes": len(labels),
                "n_base_classes": len(base_labels),
                "n_novel_classes": len(novel_labels),
                "base_train": len(base_train),
                "base_calibration": len(base_val),
                "novel_pool": len(novel_test_all),
                "seed": args.seed,
                "shots": args.shots,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
