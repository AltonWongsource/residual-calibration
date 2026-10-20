from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


STRICT_REQUIRED = {
    "pretrained_text_features",
    "tuned_text_features",
    "base_class_mask",
    "base_calibration_logits",
    "base_calibration_labels",
    "novel_test_logits",
    "novel_test_labels",
}
TARGET_REQUIRED = {
    "novel_reference_logits",
    "novel_reference_features",
    "novel_reference_labels",
    "novel_test_features",
}


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _validate_logits_labels(
    bundle: dict[str, np.ndarray],
    prefix: str,
    *,
    global_class_count: int,
    mapping_key: str | None = None,
) -> int:
    logits = np.asarray(bundle[f"{prefix}_logits"])
    labels = np.asarray(bundle[f"{prefix}_labels"]).reshape(-1)
    if logits.ndim != 2 or logits.shape[0] != labels.size or labels.size == 0:
        raise ValueError(f"{prefix} logits/labels have incompatible shapes")
    if labels.min() < 0 or labels.max() >= logits.shape[1]:
        raise ValueError(f"{prefix} labels are outside local logit-column range")
    if mapping_key and mapping_key in bundle:
        mapping = np.asarray(bundle[mapping_key], dtype=np.int64).reshape(-1)
        if mapping.size != logits.shape[1]:
            raise ValueError(f"{mapping_key} must match the number of logit columns")
        if mapping.min() < 0 or mapping.max() >= global_class_count:
            raise ValueError(f"{mapping_key} contains invalid global class indices")
        if np.unique(mapping).size != mapping.size:
            raise ValueError(f"{mapping_key} must contain unique global class indices")
    elif logits.shape[1] != global_class_count:
        raise ValueError(
            f"{prefix} has {logits.shape[1]} columns but no {mapping_key}; "
            f"either provide the mapping or use all {global_class_count} global classes"
        )
    return int(logits.shape[1])


def validate_bundle(bundle: dict[str, np.ndarray], *, require_target_reference: bool) -> None:
    missing = STRICT_REQUIRED - set(bundle)
    if require_target_reference:
        missing |= TARGET_REQUIRED - set(bundle)
    if missing:
        raise ValueError(f"bundle is missing keys: {sorted(missing)}")

    pre = np.asarray(bundle["pretrained_text_features"])
    tuned = np.asarray(bundle["tuned_text_features"])
    mask = np.asarray(bundle["base_class_mask"]).reshape(-1)
    if pre.ndim != 2 or tuned.shape != pre.shape or mask.size != pre.shape[0]:
        raise ValueError("text features/base mask have incompatible shapes")
    c_global = pre.shape[0]
    _validate_logits_labels(
        bundle,
        "base_calibration",
        global_class_count=c_global,
        mapping_key="base_logit_class_indices",
    )
    novel_c = _validate_logits_labels(
        bundle,
        "novel_test",
        global_class_count=c_global,
        mapping_key="novel_logit_class_indices",
    )
    if require_target_reference:
        ref_c = _validate_logits_labels(
            bundle,
            "novel_reference",
            global_class_count=c_global,
            mapping_key="novel_logit_class_indices",
        )
        if ref_c != novel_c:
            raise ValueError("novel reference/test logits must use the same class columns")
        features = np.asarray(bundle["novel_reference_features"])
        labels = np.asarray(bundle["novel_reference_labels"]).reshape(-1)
        test_features = np.asarray(bundle["novel_test_features"])
        if features.ndim != 2 or features.shape[0] != labels.size:
            raise ValueError("novel reference features/labels are incompatible")
        if test_features.ndim != 2 or test_features.shape[0] != np.asarray(bundle["novel_test_labels"]).size:
            raise ValueError("novel test features/labels are incompatible")


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
