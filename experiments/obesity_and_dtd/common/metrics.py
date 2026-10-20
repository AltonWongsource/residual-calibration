from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

_EPS = 1e-12


def _validate_probabilities(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if probs.ndim != 2:
        raise ValueError(f"probs must have shape [N, K], got {probs.shape}")
    if probs.shape[0] != labels.shape[0]:
        raise ValueError("probs and labels have inconsistent sample counts")
    if probs.shape[0] == 0:
        raise ValueError("empty prediction array")
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= probs.shape[1]:
        raise ValueError("labels contain an invalid class index")
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("probability rows must have positive mass")
    probs = np.clip(probs / row_sums, _EPS, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, labels


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    probs, labels = _validate_probabilities(probs, labels)
    return float(np.mean(np.argmax(probs, axis=1) == labels))


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    probs, labels = _validate_probabilities(probs, labels)
    return float(-np.mean(np.log(probs[np.arange(labels.size), labels])))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probs, labels = _validate_probabilities(probs, labels)
    one_hot = np.eye(probs.shape[1], dtype=np.float64)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Top-label ECE with equal-width bins."""
    probs, labels = _validate_probabilities(probs, labels)
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    correct = (pred == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        left, right = edges[b], edges[b + 1]
        mask = (conf >= left) & (conf <= right) if b == 0 else (conf > left) & (conf <= right)
        count = int(mask.sum())
        if count:
            ece += (count / labels.size) * abs(float(correct[mask].mean() - conf[mask].mean()))
    return float(ece)


def adaptive_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Classwise ACE used in the paper.

    For each class, sort examples by the class probability, form equal-frequency
    bins, compare class frequency with mean class probability, and average over
    classes.
    """
    probs, labels = _validate_probabilities(probs, labels)
    n, k = probs.shape
    total = 0.0
    for cls in range(k):
        order = np.argsort(probs[:, cls], kind="mergesort")
        bins = np.array_split(order, min(n_bins, n))
        cls_error = 0.0
        target = (labels == cls).astype(np.float64)
        for idx in bins:
            if idx.size:
                cls_error += (idx.size / n) * abs(
                    float(target[idx].mean() - probs[idx, cls].mean())
                )
        total += cls_error
    return float(total / k)


def static_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Classwise/static calibration error over the full probability vector."""
    probs, labels = _validate_probabilities(probs, labels)
    n, k = probs.shape
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for cls in range(k):
        conf = probs[:, cls]
        target = (labels == cls).astype(np.float64)
        cls_error = 0.0
        for b in range(n_bins):
            left, right = edges[b], edges[b + 1]
            mask = (conf >= left) & (conf <= right) if b == 0 else (conf > left) & (conf <= right)
            count = int(mask.sum())
            if count:
                cls_error += (count / n) * abs(float(target[mask].mean() - conf[mask].mean()))
        total += cls_error
    return float(total / k)


def evaluate_probabilities(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, float]:
    return {
        "nll": negative_log_likelihood(probs, labels),
        "accuracy": accuracy(probs, labels),
        "ece": expected_calibration_error(probs, labels, n_bins=n_bins),
        "ace": adaptive_calibration_error(probs, labels, n_bins=n_bins),
        "sce": static_calibration_error(probs, labels, n_bins=n_bins),
        "brier": brier_score(probs, labels),
    }


@dataclass(frozen=True)
class FlipStatistics:
    preservation_rate: float
    change_rate: float
    wrong_to_correct: float
    correct_to_wrong: float
    always_correct: float
    always_wrong: float


def prediction_flip_statistics(
    before_probs: np.ndarray,
    after_probs: np.ndarray,
    labels: np.ndarray,
) -> FlipStatistics:
    before_probs, labels = _validate_probabilities(before_probs, labels)
    after_probs, labels_after = _validate_probabilities(after_probs, labels)
    if not np.array_equal(labels, labels_after):
        raise ValueError("labels changed between before and after predictions")
    if before_probs.shape != after_probs.shape:
        raise ValueError("before and after probability arrays have different shapes")

    before = before_probs.argmax(axis=1)
    after = after_probs.argmax(axis=1)
    before_ok = before == labels
    after_ok = after == labels
    same = before == after

    return FlipStatistics(
        preservation_rate=float(same.mean()),
        change_rate=float((~same).mean()),
        wrong_to_correct=float((~before_ok & after_ok).mean()),
        correct_to_wrong=float((before_ok & ~after_ok).mean()),
        always_correct=float((before_ok & after_ok).mean()),
        always_wrong=float((~before_ok & ~after_ok).mean()),
    )
