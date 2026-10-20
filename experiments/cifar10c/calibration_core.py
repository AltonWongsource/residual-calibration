"""Selected original calibration-runner definitions for CIFAR-10-C.

Mechanically extracted; see the source/protocol note in README.md.
"""

from __future__ import annotations
import time
from typing import Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, recall_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
EPS = 1e-12

def clip_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, EPS, 1.0)
    row_sums = probs.sum(axis=1, keepdims=True)
    return probs / np.clip(row_sums, EPS, None)

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

class TemperatureScaling:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.temperature = 1.0
        self.labels = list(range(n_classes))

    def fit(self, logits: np.ndarray, y: np.ndarray) -> None:
        def loss(t: np.ndarray) -> float:
            if t[0] <= 0.0:
                return 1e9
            return log_loss(y, softmax(logits / t[0], axis=1), labels=self.labels)

        result = minimize(loss, np.array([1.0]), bounds=[(0.01, 20.0)], method="L-BFGS-B")
        self.temperature = float(result.x[0])

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return clip_probs(softmax(logits / self.temperature, axis=1))

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

class DirichletCalibration:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.labels = list(range(n_classes))
        self.weights = np.eye(n_classes, dtype=np.float64)
        self.bias = np.zeros(n_classes, dtype=np.float64)

    def fit(self, probs: np.ndarray, y: np.ndarray) -> None:
        log_probs = np.log(clip_probs(probs))

        def loss(params: np.ndarray) -> float:
            weights = params[: self.n_classes * self.n_classes].reshape(self.n_classes, self.n_classes)
            bias = params[self.n_classes * self.n_classes :]
            preds = softmax(np.dot(log_probs, weights.T) + bias, axis=1)
            return log_loss(y, preds, labels=self.labels)

        init = np.concatenate([self.weights.flatten(), self.bias])
        result = minimize(loss, init, method="L-BFGS-B", options={"maxiter": 50})
        self.weights = result.x[: self.n_classes * self.n_classes].reshape(self.n_classes, self.n_classes)
        self.bias = result.x[self.n_classes * self.n_classes :]

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        log_probs = np.log(clip_probs(probs))
        return clip_probs(softmax(np.dot(log_probs, self.weights.T) + self.bias, axis=1))

class LocalTemperatureScaling:
    def __init__(self, n_classes: int, k_neighbors: int = 100):
        self.n_classes = n_classes
        self.k_neighbors = k_neighbors
        self.t_grid = np.linspace(0.1, 5.0, 50)
        self.knn: NearestNeighbors | None = None
        self.y_ref: np.ndarray | None = None
        self.logits_ref: np.ndarray | None = None

    def fit(self, x_ref: np.ndarray, y_ref: np.ndarray, logits_ref: np.ndarray) -> None:
        x_ref = np.asarray(x_ref, dtype=np.float32)
        n_neighbors = min(max(2, self.k_neighbors), len(x_ref))
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
        self.knn.fit(x_ref)
        self.y_ref = np.asarray(y_ref, dtype=int)
        self.logits_ref = np.asarray(logits_ref, dtype=np.float64)

    def predict_proba(self, x_test: np.ndarray, logits_test: np.ndarray) -> np.ndarray:
        assert self.knn is not None
        assert self.y_ref is not None
        assert self.logits_ref is not None
        _, indices = self.knn.kneighbors(np.asarray(x_test, dtype=np.float32))
        final_probs = np.zeros_like(logits_test, dtype=np.float64)
        for row_idx, nb_idx in enumerate(indices):
            nb_logits = self.logits_ref[nb_idx]
            nb_y = self.y_ref[nb_idx].astype(int)
            scaled = softmax(nb_logits[None, :, :] / self.t_grid[:, None, None], axis=2)
            true_probs = np.take_along_axis(scaled, nb_y[None, :, None], axis=2).squeeze(-1)
            losses = -np.mean(np.log(np.clip(true_probs, EPS, 1.0)), axis=1)
            best_t = float(self.t_grid[int(np.argmin(losses))])
            final_probs[row_idx] = softmax(logits_test[row_idx] / best_t, axis=0)
        return clip_probs(final_probs)

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

def fit_predict_array_method(
    method_name: str,
    n_classes: int,
    x_cal: np.ndarray,
    x_test: np.ndarray,
    logits_cal: np.ndarray,
    logits_test: np.ndarray,
    probs_cal: np.ndarray,
    y_cal: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    fit_start = time.perf_counter()
    if method_name == "TS":
        model = TemperatureScaling(n_classes)
        model.fit(logits_cal, y_cal)
        fit_seconds = time.perf_counter() - fit_start
        pred_start = time.perf_counter()
        probs = model.predict_proba(logits_test)
        return probs, fit_seconds, time.perf_counter() - pred_start
    if method_name == "VS":
        model = VectorScaling(n_classes)
        model.fit(logits_cal, y_cal)
        fit_seconds = time.perf_counter() - fit_start
        pred_start = time.perf_counter()
        probs = model.predict_proba(logits_test)
        return probs, fit_seconds, time.perf_counter() - pred_start
    if method_name == "Dir":
        model = DirichletCalibration(n_classes)
        model.fit(probs_cal, y_cal)
        fit_seconds = time.perf_counter() - fit_start
        pred_start = time.perf_counter()
        probs = model.predict_proba(clip_probs(softmax(logits_test, axis=1)))
        return probs, fit_seconds, time.perf_counter() - pred_start
    if method_name == "LTS":
        model = LocalTemperatureScaling(n_classes)
        model.fit(x_cal, y_cal, logits_cal)
        fit_seconds = time.perf_counter() - fit_start
        pred_start = time.perf_counter()
        probs = model.predict_proba(x_test, logits_test)
        return probs, fit_seconds, time.perf_counter() - pred_start
    if method_name == "ETS":
        probs = fit_ets_probs(logits_cal, y_cal, logits_test)
        fit_seconds = time.perf_counter() - fit_start
        return probs, fit_seconds, 0.0
    if method_name == "IROvA":
        probs = fit_irova_probs(logits_cal, y_cal, logits_test)
        fit_seconds = time.perf_counter() - fit_start
        return probs, fit_seconds, 0.0
    if method_name == "MIR":
        probs = fit_mir_probs(logits_cal, y_cal, logits_test)
        fit_seconds = time.perf_counter() - fit_start
        return probs, fit_seconds, 0.0
    raise ValueError(f"Unsupported method_name={method_name}")

def build_cv_reference_probs_arrays(
    method_name: str,
    n_classes: int,
    x_cal: np.ndarray,
    logits_cal: np.ndarray,
    probs_cal: np.ndarray,
    y_cal: np.ndarray,
    seed: int,
    n_splits: int = 5,
    cv_kind: str = "kfold",
) -> np.ndarray:
    splitter = (
        KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if cv_kind == "kfold"
        else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )
    split_iter = splitter.split(logits_cal) if cv_kind == "kfold" else splitter.split(logits_cal, y_cal)
    ref_probs = np.zeros_like(probs_cal, dtype=np.float64)

    for train_idx, val_idx in split_iter:
        x_tr = x_cal[train_idx]
        x_val = x_cal[val_idx]
        logits_tr = logits_cal[train_idx]
        logits_val = logits_cal[val_idx]
        probs_tr = probs_cal[train_idx]
        y_tr = y_cal[train_idx]

        if method_name == "TS":
            model = TemperatureScaling(n_classes)
            model.fit(logits_tr, y_tr)
            ref_probs[val_idx] = model.predict_proba(logits_val)
        elif method_name == "VS":
            model = VectorScaling(n_classes)
            model.fit(logits_tr, y_tr)
            ref_probs[val_idx] = model.predict_proba(logits_val)
        elif method_name == "Dir":
            model = DirichletCalibration(n_classes)
            model.fit(probs_tr, y_tr)
            ref_probs[val_idx] = model.predict_proba(probs_cal[val_idx])
        elif method_name == "LTS":
            model = LocalTemperatureScaling(n_classes)
            model.fit(x_tr, y_tr, logits_tr)
            ref_probs[val_idx] = model.predict_proba(x_val, logits_val)
        elif method_name == "ETS":
            ref_probs[val_idx] = fit_ets_probs(logits_tr, y_tr, logits_val)
        elif method_name == "IROvA":
            ref_probs[val_idx] = fit_irova_probs(logits_tr, y_tr, logits_val)
        elif method_name == "MIR":
            ref_probs[val_idx] = fit_mir_probs(logits_tr, y_tr, logits_val)
        else:
            raise ValueError(f"Unsupported method_name={method_name}")

    return clip_probs(ref_probs)

def run_array_calibration_suite(
    suite_name: str,
    y_cal: np.ndarray,
    y_test: np.ndarray,
    x_cal: np.ndarray,
    x_test: np.ndarray,
    logits_cal: np.ndarray,
    logits_test: np.ndarray,
    base_probs_cal: np.ndarray,
    base_probs_test: np.ndarray,
    seed: int,
    standard_methods: list[str],
    cascade_methods: list[str],
    lape_params: dict[str, Any] | None = None,
    minority_classes: list[int] | None = None,
    cv_kind: str = "kfold",
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]]]:
    n_classes = base_probs_test.shape[1]
    params = lape_params or {}
    rows: list[dict[str, Any]] = []
    outputs: dict[str, dict[str, Any]] = {}
    base_predictions = np.argmax(base_probs_test, axis=1)

    rows.append(
        {
            "dataset": suite_name,
            "method": "Base",
            "family": "base",
            "fit_seconds": 0.0,
            "predict_seconds": 0.0,
            "predict_ms_per_sample": 0.0,
            **evaluate_predictions(y_test, base_probs_test, base_predictions, minority_classes),
        }
    )
    outputs["Base"] = {"probs": base_probs_test, "fit_seconds": 0.0, "predict_seconds": 0.0}

    global_like = {"TS", "VS", "Dir", "ETS", "IROvA", "MIR"}
    for method_name in standard_methods:
        probs, fit_seconds, predict_seconds = fit_predict_array_method(
            method_name,
            n_classes,
            x_cal,
            x_test,
            logits_cal,
            logits_test,
            base_probs_cal,
            y_cal,
        )
        family = "global" if method_name in global_like else "local_baseline"
        outputs[method_name] = {
            "probs": probs,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
        }
        rows.append(
            {
                "dataset": suite_name,
                "method": method_name,
                "family": family,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "predict_ms_per_sample": 1000.0 * predict_seconds / max(len(y_test), 1),
                **evaluate_predictions(y_test, probs, base_predictions, minority_classes),
            }
        )

    pure_fit_start = time.perf_counter()
    pure_lape = LAPECalibrator(n_classes, **params)
    pure_lape.fit(x_cal, y_cal, base_probs_cal)
    pure_fit_seconds = time.perf_counter() - pure_fit_start
    pure_pred_start = time.perf_counter()
    pure_probs = pure_lape.predict(x_test, base_probs_test)
    pure_pred_seconds = time.perf_counter() - pure_pred_start
    outputs["Pure LAPE"] = {
        "probs": pure_probs,
        "fit_seconds": pure_fit_seconds,
        "predict_seconds": pure_pred_seconds,
    }
    rows.append(
        {
            "dataset": suite_name,
            "method": "Pure LAPE",
            "family": "pure_lape",
            "fit_seconds": pure_fit_seconds,
            "predict_seconds": pure_pred_seconds,
            "predict_ms_per_sample": 1000.0 * pure_pred_seconds / max(len(y_test), 1),
            **evaluate_predictions(y_test, pure_probs, base_predictions, minority_classes),
        }
    )

    for method_name in cascade_methods:
        ref_probs = build_cv_reference_probs_arrays(
            method_name,
            n_classes,
            x_cal,
            logits_cal,
            base_probs_cal,
            y_cal,
            seed,
            cv_kind=cv_kind,
        )
        cascade_fit_start = time.perf_counter()
        cascade_lape = LAPECalibrator(n_classes, **params)
        cascade_lape.fit(x_cal, y_cal, ref_probs)
        cascade_fit_seconds = time.perf_counter() - cascade_fit_start + outputs[method_name]["fit_seconds"]
        cascade_pred_start = time.perf_counter()
        cascade_probs = cascade_lape.predict(x_test, outputs[method_name]["probs"])
        cascade_pred_seconds = time.perf_counter() - cascade_pred_start + outputs[method_name]["predict_seconds"]
        outputs[f"{method_name}+LAPE"] = {
            "probs": cascade_probs,
            "fit_seconds": cascade_fit_seconds,
            "predict_seconds": cascade_pred_seconds,
        }
        rows.append(
            {
                "dataset": suite_name,
                "method": f"{method_name}+LAPE",
                "family": "cascade",
                "fit_seconds": cascade_fit_seconds,
                "predict_seconds": cascade_pred_seconds,
                "predict_ms_per_sample": 1000.0 * cascade_pred_seconds / max(len(y_test), 1),
                **evaluate_predictions(y_test, cascade_probs, base_predictions, minority_classes),
            }
        )

    df = pd.DataFrame(rows).sort_values(by=["nll", "method"]).reset_index(drop=True)
    return df, outputs, {"base_predictions": base_predictions}

def _temperature_scaling_objective(t: np.ndarray, logits: np.ndarray, y: np.ndarray) -> float:
    if t[0] <= 0:
        return 1e9
    probs = clip_probs(softmax(logits / t[0], axis=1))
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], EPS, 1.0))))

def fit_ets_probs(cal_logits: np.ndarray, y_cal: np.ndarray, eval_logits: np.ndarray) -> np.ndarray:
    n_classes = cal_logits.shape[1]
    y_cal = np.asarray(y_cal, dtype=int)
    result_t = minimize(
        _temperature_scaling_objective,
        np.array([1.0], dtype=np.float64),
        args=(np.asarray(cal_logits, dtype=np.float64), y_cal),
        bounds=[(0.05, 5.0)],
        method="L-BFGS-B",
    )
    t = float(result_t.x[0])
    p_raw = clip_probs(softmax(cal_logits, axis=1))
    p_temp = clip_probs(softmax(cal_logits / t, axis=1))
    p_uniform = np.full_like(p_raw, 1.0 / n_classes)

    def weight_objective(w: np.ndarray) -> float:
        probs = clip_probs(w[0] * p_temp + w[1] * p_raw + w[2] * p_uniform)
        return float(-np.mean(np.log(np.clip(probs[np.arange(len(y_cal)), y_cal], EPS, 1.0))))

    cons = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    res_w = minimize(
        weight_objective,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        constraints=cons,
        method="SLSQP",
        options={"maxiter": 200, "disp": False},
    )
    w = np.asarray(res_w.x, dtype=np.float64)

    p_raw_eval = clip_probs(softmax(eval_logits, axis=1))
    p_temp_eval = clip_probs(softmax(eval_logits / t, axis=1))
    p_uniform_eval = np.full_like(p_raw_eval, 1.0 / n_classes)
    return clip_probs(w[0] * p_temp_eval + w[1] * p_raw_eval + w[2] * p_uniform_eval)

def fit_irova_probs(cal_logits: np.ndarray, y_cal: np.ndarray, eval_logits: np.ndarray) -> np.ndarray:
    p_cal = clip_probs(softmax(cal_logits, axis=1))
    p_eval = clip_probs(softmax(eval_logits, axis=1))
    n_classes = p_cal.shape[1]
    calibrated = np.zeros_like(p_eval, dtype=np.float64)
    for cls in range(n_classes):
        ir = IsotonicRegression(out_of_bounds="clip")
        cls_target = (np.asarray(y_cal) == cls).astype(float)
        ir.fit(p_cal[:, cls], cls_target)
        calibrated[:, cls] = ir.predict(p_eval[:, cls]) + 1e-9 * p_eval[:, cls]
    return clip_probs(calibrated)

def fit_mir_probs(cal_logits: np.ndarray, y_cal: np.ndarray, eval_logits: np.ndarray) -> np.ndarray:
    y_onehot = np.eye(cal_logits.shape[1], dtype=np.float64)[np.asarray(y_cal, dtype=int)]
    p_cal = clip_probs(softmax(cal_logits, axis=1))
    p_eval = clip_probs(softmax(eval_logits, axis=1))
    ir = IsotonicRegression(out_of_bounds="clip")
    calibrated = ir.fit_transform(p_cal.flatten(), y_onehot.flatten())
    _ = calibrated  # keep fit object consistent with transform usage
    p_out = ir.predict(p_eval.flatten()).reshape(eval_logits.shape) + 1e-9 * p_eval
    return clip_probs(p_out)
