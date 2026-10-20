from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

_EPS = 1e-12


@dataclass(frozen=True)
class DACResult:
    logits: np.ndarray
    probs: np.ndarray
    gamma_by_class: np.ndarray
    predicted_class_before_dac: np.ndarray
    temperature: float


def _as_2d_finite(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 2D array")
    return arr


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = _as_2d_finite(x, "features")
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError("feature rows must have non-zero norm")
    return arr / norm


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = _as_2d_finite(logits, "logits")
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return (exp_x / exp_x.sum(axis=1, keepdims=True)).astype(np.float32)


def textual_proximity(
    query_text_features: np.ndarray,
    base_text_features: np.ndarray,
    *,
    k: int = 5,
) -> np.ndarray:
    """Equation (6) of Wang et al. (ICML 2024).

    P(w_i, W) = exp(- mean_{w_j in KNN(w_i, W)} ||w_i-w_j||_2).
    Both query and base text features are L2 normalized before distances.
    """
    query = l2_normalize_rows(query_text_features)
    base = l2_normalize_rows(base_text_features)
    if base.shape[0] < 1:
        raise ValueError("at least one base-class text feature is required")
    k_eff = min(max(int(k), 1), base.shape[0])
    # Number of classes is small, so an exact dense distance matrix is clearer
    # and fully deterministic.
    distances = np.linalg.norm(query[:, None, :] - base[None, :, :], axis=2)
    nearest = np.partition(distances, kth=k_eff - 1, axis=1)[:, :k_eff]
    return np.exp(-nearest.mean(axis=1)).astype(np.float64)


def compute_textual_deviation(
    *,
    pretrained_text_features: np.ndarray,
    tuned_text_features: np.ndarray,
    base_class_mask: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Equation (7) of Wang et al. (ICML 2024).

    gamma(c_i) = P(w'_i, W') / P(w_i, W).
    Following the paper, gamma is set to one for base classes.
    """
    pre = l2_normalize_rows(pretrained_text_features)
    tuned = l2_normalize_rows(tuned_text_features)
    if pre.shape != tuned.shape:
        raise ValueError("pretrained and tuned text features must have equal shape")
    mask = np.asarray(base_class_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != pre.shape[0] or not np.any(mask):
        raise ValueError("base_class_mask must select at least one class")
    p_pre = textual_proximity(pre, pre[mask], k=k)
    p_tuned = textual_proximity(tuned, tuned[mask], k=k)
    gamma = p_tuned / np.clip(p_pre, _EPS, None)
    gamma[mask] = 1.0
    if np.any(~np.isfinite(gamma)) or np.any(gamma <= 0):
        raise ValueError("invalid textual deviation scores")
    return gamma.astype(np.float32)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    max_iter: int = 200,
    init_temperature: float = 1.0,
) -> float:
    """Fit a positive scalar temperature by NLL minimization.

    Returned T is used as logits / T. This is equivalent to learning a
    logit-scale multiplier 1/T.
    """
    x_np = _as_2d_finite(logits, "logits")
    y_np = np.asarray(labels, dtype=np.int64).reshape(-1)
    if y_np.shape[0] != x_np.shape[0] or y_np.size == 0:
        raise ValueError("labels must match non-empty logits")
    if y_np.min() < 0 or y_np.max() >= x_np.shape[1]:
        raise ValueError("labels outside logit class range")

    x = torch.tensor(x_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.long)
    log_t = torch.tensor(
        float(np.log(max(init_temperature, 1e-3))),
        dtype=torch.float64,
        requires_grad=True,
    )
    optimizer = torch.optim.LBFGS(
        [log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_t).clamp(1e-3, 1e3)
        loss = F.cross_entropy(x / temperature, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_t.detach()).clamp(1e-3, 1e3))
    if not np.isfinite(temperature):
        raise RuntimeError("temperature optimization produced a non-finite value")
    return temperature


def apply_dac(
    *,
    raw_logits: np.ndarray,
    gamma_by_class: np.ndarray,
    temperature: float,
) -> DACResult:
    """Apply Equation (8) of Wang et al. (ICML 2024).

    The tuned model first predicts c_hat using its original logits. All class
    logits for that sample are then scaled by gamma(c_hat), after the base
    temperature scaling. As a scalar row-wise rescaling, DAC preserves argmax.
    """
    raw = _as_2d_finite(raw_logits, "raw_logits")
    gamma = np.asarray(gamma_by_class, dtype=np.float64).reshape(-1)
    if gamma.shape[0] != raw.shape[1] or np.any(gamma <= 0):
        raise ValueError("gamma_by_class has invalid shape or values")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    predicted = raw.argmax(axis=1)
    row_scale = gamma[predicted] / float(temperature)
    dac_logits = raw * row_scale[:, None]
    return DACResult(
        logits=dac_logits.astype(np.float32),
        probs=softmax_np(dac_logits),
        gamma_by_class=gamma.astype(np.float32),
        predicted_class_before_dac=predicted.astype(np.int64),
        temperature=float(temperature),
    )
