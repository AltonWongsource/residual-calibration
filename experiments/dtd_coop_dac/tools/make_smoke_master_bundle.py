from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    c, d_text, d_img = 8, 16, 12
    base_mask = np.zeros(c, dtype=bool)
    base_mask[:4] = True
    pre = rng.normal(size=(c, d_text))
    pre /= np.linalg.norm(pre, axis=1, keepdims=True)
    tuned = pre + rng.normal(scale=0.08, size=pre.shape)
    tuned /= np.linalg.norm(tuned, axis=1, keepdims=True)
    centers = rng.normal(size=(c, d_img))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    projection = rng.normal(size=(d_img, c))

    base_labels = np.repeat(np.arange(4), 20)
    base_features = centers[base_labels] + rng.normal(scale=0.25, size=(base_labels.size, d_img))
    base_logits = base_features @ projection + rng.normal(scale=0.2, size=(base_labels.size, c))

    novel_labels = np.repeat(np.arange(4, 8), 30)
    novel_features = centers[novel_labels] + rng.normal(scale=0.25, size=(novel_labels.size, d_img))
    novel_logits = novel_features @ projection + rng.normal(scale=0.2, size=(novel_labels.size, c))
    # Give the synthetic correct classes a modest signal.
    novel_logits[np.arange(novel_labels.size), novel_labels] += 2.0
    base_logits[np.arange(base_labels.size), base_labels] += 2.0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        pretrained_text_features=pre.astype(np.float32),
        tuned_text_features=tuned.astype(np.float32),
        base_class_mask=base_mask,
        base_calibration_logits=base_logits.astype(np.float32),
        base_calibration_labels=base_labels.astype(np.int64),
        novel_pool_logits=novel_logits.astype(np.float32),
        novel_pool_features=novel_features.astype(np.float32),
        novel_pool_labels=novel_labels.astype(np.int64),
        class_names=np.asarray([f"class_{i}" for i in range(c)], dtype=object),
    )
    print(output)


if __name__ == "__main__":
    main()
