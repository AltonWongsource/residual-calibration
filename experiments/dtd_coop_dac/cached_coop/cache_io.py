from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = {
    "class_names",
    "base_class_mask",
    "base_train_features",
    "base_train_labels",
    "base_calibration_features",
    "base_calibration_labels",
    "novel_pool_features",
    "novel_pool_labels",
}


def load_feature_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _feature_matrix(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim not in {2, 3} or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite [N,D] or [N,V,D] array")
    return arr


def validate_feature_cache(cache: dict[str, np.ndarray]) -> None:
    missing = REQUIRED_KEYS - set(cache)
    if missing:
        raise ValueError(f"feature cache missing keys: {sorted(missing)}")
    names = np.asarray(cache["class_names"]).astype(str).reshape(-1)
    mask = np.asarray(cache["base_class_mask"], dtype=bool).reshape(-1)
    if names.size < 2 or mask.size != names.size or not np.any(mask) or np.all(mask):
        raise ValueError("class_names/base_class_mask must define non-empty base and novel classes")
    feature_dim: int | None = None
    for prefix in ["base_train", "base_calibration", "novel_pool"]:
        feat = _feature_matrix(cache[f"{prefix}_features"], f"{prefix}_features")
        labels = np.asarray(cache[f"{prefix}_labels"], dtype=np.int64).reshape(-1)
        if feat.shape[0] != labels.size or labels.size == 0:
            raise ValueError(f"{prefix} features/labels are incompatible or empty")
        if labels.min() < 0 or labels.max() >= names.size:
            raise ValueError(f"{prefix} labels outside global class range")
        dim = int(feat.shape[-1])
        feature_dim = dim if feature_dim is None else feature_dim
        if dim != feature_dim:
            raise ValueError("all cached feature arrays must have the same last dimension")
    base_train_y = np.asarray(cache["base_train_labels"], dtype=np.int64).reshape(-1)
    base_cal_y = np.asarray(cache["base_calibration_labels"], dtype=np.int64).reshape(-1)
    novel_y = np.asarray(cache["novel_pool_labels"], dtype=np.int64).reshape(-1)
    if np.any(~mask[base_train_y]) or np.any(~mask[base_cal_y]):
        raise ValueError("base_train/base_calibration must contain base-class labels only")
    if np.any(mask[novel_y]):
        raise ValueError("novel_pool must contain novel-class labels only")


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
