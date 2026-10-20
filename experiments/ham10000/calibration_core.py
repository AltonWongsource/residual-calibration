"""Selected original calibration-runner definitions.

Mechanically extracted; see the source/protocol note in README.md.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, recall_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
EPS = 1e-12

def ensure_dense_array(x: Any) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x)

def clip_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, EPS, 1.0)
    row_sums = probs.sum(axis=1, keepdims=True)
    return probs / np.clip(row_sums, EPS, None)

def get_deep_features(model: MLPClassifier, x: np.ndarray) -> np.ndarray:
    activations = ensure_dense_array(x).astype(np.float32, copy=False)
    try:
        for idx in range(len(model.coefs_) - 1):
            activations = np.dot(activations, model.coefs_[idx]) + model.intercepts_[idx]
            activations = np.maximum(0.0, activations)
        return activations
    except Exception:
        return activations

def compute_top_label_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    probs = clip_probs(probs)
    y_true = np.asarray(y_true)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == y_true
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        in_bin = (confidences > bin_edges[idx]) & (confidences <= bin_edges[idx + 1])
        if np.any(in_bin):
            ece += abs(confidences[in_bin].mean() - accuracies[in_bin].mean()) * in_bin.mean()
    return float(ece)

def compute_binary_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    probs = clip_probs(probs)
    y_true = np.asarray(y_true)
    if probs.shape[1] != 2:
        return float("nan")
    p_pos = probs[:, 1]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        in_bin = (p_pos > bin_edges[idx]) & (p_pos <= bin_edges[idx + 1])
        if np.any(in_bin):
            ece += abs(p_pos[in_bin].mean() - y_true[in_bin].mean()) * in_bin.mean()
    return float(ece)

def compute_ace(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    probs = clip_probs(probs)
    y_true = np.asarray(y_true)
    n_classes = probs.shape[1]
    per_class_scores: list[float] = []
    for cls in range(n_classes):
        cls_probs = probs[:, cls]
        cls_true = (y_true == cls).astype(float)
        order = np.argsort(cls_probs)
        cls_probs_sorted = cls_probs[order]
        cls_true_sorted = cls_true[order]
        bins = np.array_split(np.arange(len(cls_probs_sorted)), n_bins)
        cls_ace = 0.0
        non_empty = 0
        for bin_idx in bins:
            if len(bin_idx) == 0:
                continue
            conf = float(np.mean(cls_probs_sorted[bin_idx]))
            acc = float(np.mean(cls_true_sorted[bin_idx]))
            cls_ace += abs(conf - acc)
            non_empty += 1
        if non_empty > 0:
            per_class_scores.append(cls_ace / non_empty)
    if not per_class_scores:
        return float("nan")
    return float(np.mean(per_class_scores))

class VectorScaling:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.labels = list(range(n_classes))
        self.weights = np.ones(n_classes, dtype=np.float64)
        self.bias = np.zeros(n_classes, dtype=np.float64)

    def fit(self, logits: np.ndarray, y: np.ndarray) -> None:
        def loss(params: np.ndarray) -> float:
            weights = params[: self.n_classes]
            bias = params[self.n_classes :]
            preds = softmax(logits * weights + bias, axis=1)
            return log_loss(y, preds, labels=self.labels)

        init = np.concatenate([self.weights, self.bias])
        result = minimize(loss, init, method="L-BFGS-B")
        self.weights = result.x[: self.n_classes]
        self.bias = result.x[self.n_classes :]

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return clip_probs(softmax(logits * self.weights + self.bias, axis=1))

class LAPECalibrator:
    def __init__(
        self,
        n_classes: int,
        metric: str = "euclidean",
        candidate_alpha: np.ndarray | None = None,
        m_anchor_floor: int = 5,
        k_min_floor: int = 20,
        k_min_divisor: int = 5,
        k_max_multiplier: int = 4,
        k_max_cap: int = 600,
    ):
        self.n_classes = n_classes
        self.metric = metric
        self.candidate_alpha = (
            np.asarray(candidate_alpha, dtype=np.float64)
            if candidate_alpha is not None
            else np.linspace(0.0, 1.0, 11)
        )
        self.m_anchor_floor = m_anchor_floor
        self.k_min_floor = k_min_floor
        self.k_min_divisor = k_min_divisor
        self.k_max_multiplier = k_max_multiplier
        self.k_max_cap = k_max_cap

        self.y_ref: np.ndarray | None = None
        self.pi_ref: np.ndarray | None = None
        self.knn: NearestNeighbors | None = None
        self.m_anchor = 5
        self.k_min = 20
        self.k_max = 21

    def fit(self, x_ref: np.ndarray, y_ref: np.ndarray, base_probs_ref: np.ndarray) -> None:
        x_ref = np.asarray(x_ref, dtype=np.float32)
        self.y_ref = np.asarray(y_ref, dtype=int).ravel()
        self.pi_ref = clip_probs(base_probs_ref)
        n_ref = len(x_ref)
        sqrt_n = np.sqrt(max(n_ref, 1))

        self.m_anchor = max(self.m_anchor_floor, int(np.floor(sqrt_n / 2)))
        self.k_min = max(self.k_min_floor, int(sqrt_n / self.k_min_divisor))
        self.k_max = min(int(self.m_anchor * self.k_max_multiplier), self.k_max_cap, n_ref - 1)
        if self.k_max <= self.k_min:
            self.k_max = min(n_ref - 1, self.k_min + 10)
        if self.m_anchor >= self.k_max:
            self.m_anchor = max(1, self.k_max - 1)

        self.knn = NearestNeighbors(n_neighbors=self.k_max + 1, metric=self.metric, n_jobs=-1)
        self.knn.fit(x_ref)

    def predict(self, x_test: np.ndarray, base_probs_test: np.ndarray) -> np.ndarray:
        assert self.knn is not None
        assert self.y_ref is not None
        assert self.pi_ref is not None

        x_test = np.asarray(x_test, dtype=np.float32)
        base_probs_test = clip_probs(base_probs_test)
        distances_all, indices_all = self.knn.kneighbors(x_test)
        final_probs = np.zeros((len(x_test), self.n_classes), dtype=np.float64)
        step = max(1, (self.k_max - self.k_min) // 10)

        for row_idx in range(len(x_test)):
            pi_base = base_probs_test[row_idx]
            distances = distances_all[row_idx]
            indices = indices_all[row_idx]

            sigma = max(float(distances[self.m_anchor]), 1e-9)
            weights_all = np.exp(-((distances[: self.k_max] ** 2) / (sigma**2)))
            y_nb_all = self.y_ref[indices[: self.k_max]]
            pi_neigh_all = self.pi_ref[indices[: self.k_max]]

            best_loss = np.inf
            best_k = self.k_min
            best_alpha = 0.0

            for k in range(self.k_min, self.k_max + 1, step):
                y_est = y_nb_all[:k]
                base_counts = np.bincount(y_est, minlength=self.n_classes)
                val_counts = np.tile(base_counts, (k, 1))
                val_counts[np.arange(k), y_est] -= 1
                p_emp = clip_probs((val_counts + 1.0 / self.n_classes) / k)

                w_k = weights_all[:k]
                w_k = w_k / np.clip(w_k.sum(), EPS, None)
                pi_neigh_k = pi_neigh_all[:k]

                for alpha in self.candidate_alpha:
                    p_mix = (1.0 - alpha) * pi_neigh_k + alpha * p_emp
                    probs_at_true = p_mix[np.arange(k), y_est]
                    loss = -np.sum(w_k * np.log(np.clip(probs_at_true, EPS, 1.0)))
                    if loss < best_loss:
                        best_loss = loss
                        best_k = k
                        best_alpha = float(alpha)

            final_counts = np.bincount(y_nb_all[:best_k], minlength=self.n_classes)
            p_local = (final_counts + 1.0 / self.n_classes) / (best_k + 1.0)
            p_local = p_local / np.clip(p_local.sum(), EPS, None)
            final_probs[row_idx] = (1.0 - best_alpha) * pi_base + best_alpha * p_local

        return clip_probs(final_probs)

def evaluate_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    base_predictions: np.ndarray,
    minority_classes: list[int] | None = None,
) -> dict[str, Any]:
    probs = clip_probs(probs)
    predictions = np.argmax(probs, axis=1)
    metrics: dict[str, Any] = {
        "nll": float(log_loss(y_true, probs, labels=list(range(probs.shape[1])))),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "ece_toplabel": compute_top_label_ece(probs, y_true),
        "ace": compute_ace(probs, y_true),
        "pred_change_rate_vs_base": float(np.mean(predictions != base_predictions)),
    }

    if probs.shape[1] == 2:
        metrics["auc"] = float(roc_auc_score(y_true, probs[:, 1]))
        metrics["ece_binary"] = compute_binary_ece(probs, y_true)
    else:
        metrics["auc"] = float("nan")
        metrics["ece_binary"] = float("nan")

    if minority_classes:
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, predictions))
        metrics["rare_recall_macro"] = float(
            recall_score(y_true, predictions, labels=minority_classes, average="macro", zero_division=0)
        )
        metrics["f1_macro"] = float(f1_score(y_true, predictions, average="macro", zero_division=0))
    else:
        metrics["balanced_accuracy"] = float("nan")
        metrics["rare_recall_macro"] = float("nan")
        metrics["f1_macro"] = float("nan")

    return metrics
