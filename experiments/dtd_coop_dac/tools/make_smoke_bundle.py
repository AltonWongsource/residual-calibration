from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="./example/smoke_bundle.npz")
    args = p.parse_args()
    rng = np.random.default_rng(7)
    c, d_text, d_img = 8, 12, 10
    base_mask = np.zeros(c, dtype=bool)
    base_mask[:4] = True
    pre = rng.normal(size=(c, d_text))
    pre /= np.linalg.norm(pre, axis=1, keepdims=True)
    tuned = pre.copy()
    tuned[:4] += 0.05 * rng.normal(size=(4, d_text))
    tuned[4:] += 0.45 * rng.normal(size=(4, d_text))
    tuned /= np.linalg.norm(tuned, axis=1, keepdims=True)

    def make(n: int, allowed: np.ndarray):
        labels = rng.choice(allowed, size=n)
        features = rng.normal(size=(n, d_img))
        centers = rng.normal(size=(c, d_img))
        features += centers[labels]
        logits = rng.normal(scale=0.7, size=(n, c))
        logits[np.arange(n), labels] += 2.5
        return logits.astype(np.float32), features.astype(np.float32), labels.astype(np.int64)

    base_logits, base_feat, base_y = make(80, np.arange(4))
    ref_logits, ref_feat, ref_y = make(80, np.arange(4, 8))
    test_logits, test_feat, test_y = make(120, np.arange(4, 8))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pretrained_text_features=pre.astype(np.float32),
        tuned_text_features=tuned.astype(np.float32),
        base_class_mask=base_mask,
        base_calibration_logits=base_logits,
        base_calibration_features=base_feat,
        base_calibration_labels=base_y,
        novel_reference_logits=ref_logits,
        novel_reference_features=ref_feat,
        novel_reference_labels=ref_y,
        novel_test_logits=test_logits,
        novel_test_features=test_feat,
        novel_test_labels=test_y,
        class_names=np.asarray([f"class_{i}" for i in range(c)], dtype=object),
    )
    print(path)


if __name__ == "__main__":
    main()
