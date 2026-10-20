from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .repro import ensure_dir, jsonable


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, sort_keys=True)


def save_prediction_bundle(
    path: str | Path,
    *,
    logits: np.ndarray,
    probs: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    class_names: list[str],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save an LAPE-ready prediction/reference bundle.

    The `.npz` intentionally stores both logits and probabilities. Existing
    post-hoc anchor code can consume logits, while LAPE can use features,
    labels, and out-of-fold anchor probabilities generated downstream.
    """
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        logits=np.asarray(logits, dtype=np.float32),
        probs=np.asarray(probs, dtype=np.float32),
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int64),
        class_names=np.asarray(class_names, dtype=object),
        metadata_json=np.asarray(json.dumps(jsonable(metadata or {})), dtype=object),
    )
