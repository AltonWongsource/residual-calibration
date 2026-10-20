from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_array(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return np.load(p, allow_pickle=True)
    if p.suffix.lower() == ".npz":
        with np.load(p, allow_pickle=True) as data:
            if len(data.files) != 1:
                raise ValueError(f"{p} contains multiple arrays; export the intended array as .npy")
            return data[data.files[0]]
    raise ValueError(f"unsupported array file: {p}")


def normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the master NPZ consumed by run_fast_rebuttal.py.")
    p.add_argument("--pretrained-text", required=True)
    p.add_argument("--tuned-text", required=True)
    p.add_argument("--base-mask", required=True)
    p.add_argument("--base-cal-labels", required=True)
    p.add_argument("--novel-pool-labels", required=True)
    p.add_argument("--novel-pool-features", required=True)
    p.add_argument("--base-cal-logits")
    p.add_argument("--novel-pool-logits")
    p.add_argument("--base-cal-features")
    p.add_argument("--logit-scale", type=float, default=100.0, help="Used only when logits are computed from normalized image/text features.")
    p.add_argument("--class-names")
    p.add_argument("--metadata-json")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pre_text = normalize_rows(load_array(args.pretrained_text))
    tuned_text = normalize_rows(load_array(args.tuned_text))
    if pre_text.shape != tuned_text.shape:
        raise ValueError("pretrained and tuned text features must have the same shape")
    base_mask = np.asarray(load_array(args.base_mask), dtype=bool).reshape(-1)
    if base_mask.size != pre_text.shape[0]:
        raise ValueError("base mask size must equal the number of classes")
    base_labels = np.asarray(load_array(args.base_cal_labels), dtype=np.int64).reshape(-1)
    novel_labels = np.asarray(load_array(args.novel_pool_labels), dtype=np.int64).reshape(-1)
    novel_features = normalize_rows(load_array(args.novel_pool_features))
    if novel_features.shape[0] != novel_labels.size:
        raise ValueError("novel pool features and labels are incompatible")

    if args.novel_pool_logits:
        novel_logits = np.asarray(load_array(args.novel_pool_logits), dtype=np.float32)
    else:
        novel_logits = args.logit_scale * (novel_features @ tuned_text.T)
    if novel_logits.shape != (novel_labels.size, pre_text.shape[0]):
        raise ValueError("novel logits have an invalid shape")

    if args.base_cal_logits:
        base_logits = np.asarray(load_array(args.base_cal_logits), dtype=np.float32)
    elif args.base_cal_features:
        base_features = normalize_rows(load_array(args.base_cal_features))
        base_logits = args.logit_scale * (base_features @ tuned_text.T)
    else:
        raise ValueError("provide either --base-cal-logits or --base-cal-features")
    if base_logits.shape != (base_labels.size, pre_text.shape[0]):
        raise ValueError("base calibration logits have an invalid shape")

    arrays: dict[str, np.ndarray] = {
        "pretrained_text_features": pre_text,
        "tuned_text_features": tuned_text,
        "base_class_mask": base_mask,
        "base_calibration_logits": base_logits.astype(np.float32),
        "base_calibration_labels": base_labels,
        "novel_pool_logits": novel_logits.astype(np.float32),
        "novel_pool_features": novel_features.astype(np.float32),
        "novel_pool_labels": novel_labels,
    }
    if args.class_names:
        arrays["class_names"] = np.asarray(load_array(args.class_names), dtype=object)
    metadata = {
        "logit_scale": args.logit_scale,
        "logits_computed_from_features": not (args.base_cal_logits and args.novel_pool_logits),
    }
    if args.metadata_json:
        metadata.update(json.loads(Path(args.metadata_json).read_text(encoding="utf-8")))
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=object)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
