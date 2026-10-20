#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from cached_coop.cache_io import load_feature_cache, validate_feature_cache
from cached_coop.trainer import collapse_views


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def remap(labels_global: np.ndarray, class_indices: np.ndarray) -> np.ndarray:
    mapping = {int(g): i for i, g in enumerate(class_indices.tolist())}
    try:
        return np.asarray([mapping[int(y)] for y in labels_global], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"label is not represented by logit columns: {exc}") from exc


def logits(features: np.ndarray, text: np.ndarray, scale: float) -> np.ndarray:
    f = normalize(collapse_views(features))
    t = normalize(text)
    return (float(scale) * f @ t.T).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a strict base-to-new DAC/LAPE master bundle.")
    p.add_argument("--cache", required=True)
    p.add_argument("--coop-text-outputs", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache = load_feature_cache(args.cache)
    validate_feature_cache(cache)
    with np.load(args.coop_text_outputs, allow_pickle=True) as data:
        pre = np.asarray(data["pretrained_text_features"], dtype=np.float32)
        tuned = np.asarray(data["tuned_text_features"], dtype=np.float32)
        scale = float(np.asarray(data["logit_scale"]).reshape(()))
    base_mask = np.asarray(cache["base_class_mask"], dtype=bool).reshape(-1)
    base_idx = np.flatnonzero(base_mask).astype(np.int64)
    novel_idx = np.flatnonzero(~base_mask).astype(np.int64)
    base_y_global = np.asarray(cache["base_calibration_labels"], dtype=np.int64).reshape(-1)
    novel_y_global = np.asarray(cache["novel_pool_labels"], dtype=np.int64).reshape(-1)
    base_logits = logits(cache["base_calibration_features"], tuned[base_idx], scale)
    novel_logits = logits(cache["novel_pool_features"], tuned[novel_idx], scale)
    novel_features = collapse_views(cache["novel_pool_features"])
    arrays = {
        "pretrained_text_features": pre,
        "tuned_text_features": tuned,
        "base_class_mask": base_mask,
        "base_calibration_logits": base_logits,
        "base_calibration_labels": remap(base_y_global, base_idx),
        "base_logit_class_indices": base_idx,
        "novel_pool_logits": novel_logits,
        "novel_pool_features": novel_features,
        "novel_pool_labels": remap(novel_y_global, novel_idx),
        "novel_logit_class_indices": novel_idx,
        "class_names": np.asarray(cache["class_names"], dtype=object),
        "metadata_json": np.asarray(
            [
                json.dumps(
                    {
                        "protocol": "feature-cached CoOp; official base/new class split supplied by manifest",
                        "logits": "base and novel classifiers are evaluated on their respective class subsets",
                        "feature_cache": str(Path(args.cache).resolve()),
                        "coop_text_outputs": str(Path(args.coop_text_outputs).resolve()),
                    }
                )
            ],
            dtype=object,
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(output)


if __name__ == "__main__":
    main()
