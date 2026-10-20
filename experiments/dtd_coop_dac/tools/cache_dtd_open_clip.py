#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from cached_coop.clip_adapter import load_open_clip
from cached_coop.cache_io import validate_feature_cache


PARTITIONS = {"base_train", "base_calibration", "novel_pool"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a DTD feature cache for CPU-efficient CoOp from a manifest CSV."
    )
    p.add_argument("--manifest", required=True, help="CSV: path,global_label,class_name,partition")
    p.add_argument("--output", required=True)
    p.add_argument("--model-name", default="ViT-B-32")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--device", default="cpu")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--base-train-views", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"path", "global_label", "class_name", "partition"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifest must contain columns {sorted(required)}")
    for row in rows:
        if row["partition"] not in PARTITIONS:
            raise ValueError(f"unknown partition: {row['partition']}")
    return rows


def deterministic_transform(transform, image: Image.Image, seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    return transform(image)


def encode_batches(model, tensors: list[torch.Tensor], batch_size: int, device: str) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start : start + batch_size]).to(device)
            feat = model.encode_image(batch).float()
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            outputs.append(feat.cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    rows = read_manifest(manifest_path)
    model, _, preprocess_train, preprocess_val = load_open_clip(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    labels = sorted({int(r["global_label"]) for r in rows})
    if labels != list(range(max(labels) + 1)):
        raise ValueError("global_label values must be contiguous from 0")
    class_names = [""] * len(labels)
    for row in rows:
        idx = int(row["global_label"])
        if class_names[idx] and class_names[idx] != row["class_name"]:
            raise ValueError("inconsistent class_name for a global label")
        class_names[idx] = row["class_name"]
    base_mask = np.zeros(len(class_names), dtype=bool)
    for row in rows:
        if row["partition"].startswith("base_"):
            base_mask[int(row["global_label"])] = True

    arrays: dict[str, np.ndarray] = {
        "class_names": np.asarray(class_names, dtype=object),
        "base_class_mask": base_mask,
    }
    root = manifest_path.parent
    for partition in ["base_train", "base_calibration", "novel_pool"]:
        selected = [r for r in rows if r["partition"] == partition]
        if not selected:
            raise ValueError(f"manifest contains no rows for {partition}")
        labels_part = np.asarray([int(r["global_label"]) for r in selected], dtype=np.int64)
        if partition == "base_train":
            all_views = []
            for view in range(max(1, args.base_train_views)):
                tensors = []
                for i, row in enumerate(selected):
                    image_path = Path(row["path"])
                    if not image_path.is_absolute():
                        image_path = root / image_path
                    image = Image.open(image_path).convert("RGB")
                    tensors.append(
                        deterministic_transform(
                            preprocess_train,
                            image,
                            args.seed + 100000 * view + i,
                        )
                    )
                all_views.append(encode_batches(model, tensors, args.batch_size, args.device))
            features = np.stack(all_views, axis=1)
        else:
            tensors = []
            for row in selected:
                image_path = Path(row["path"])
                if not image_path.is_absolute():
                    image_path = root / image_path
                tensors.append(preprocess_val(Image.open(image_path).convert("RGB")))
            features = encode_batches(model, tensors, args.batch_size, args.device)
        arrays[f"{partition}_features"] = features
        arrays[f"{partition}_labels"] = labels_part
        arrays[f"{partition}_paths"] = np.asarray([r["path"] for r in selected], dtype=object)

    validate_feature_cache(arrays)
    arrays["metadata_json"] = np.asarray(
        [
            str(
                {
                    "model_name": args.model_name,
                    "pretrained": args.pretrained,
                    "base_train_views": args.base_train_views,
                    "manifest": str(manifest_path),
                }
            )
        ],
        dtype=object,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(output)


if __name__ == "__main__":
    main()
