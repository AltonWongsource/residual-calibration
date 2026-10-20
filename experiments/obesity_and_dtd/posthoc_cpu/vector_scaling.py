from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold


@dataclass(frozen=True)
class VectorScalingFit:
    scale: np.ndarray
    bias: np.ndarray
    final_nll: float
    converged: bool
    iterations: int

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim != 2 or logits.shape[1] != self.scale.size:
            raise ValueError("logits shape is incompatible with the fitted vector scaler")
        return logits * self.scale[None, :] + self.bias[None, :]

    def transform_probs(self, logits: np.ndarray) -> np.ndarray:
        calibrated = self.transform_logits(logits)
        shifted = calibrated - calibrated.max(axis=1, keepdims=True)
        exp = np.exp(shifted, dtype=np.float64)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs.astype(np.float32)


class _VectorScaler(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(num_classes, dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.zeros(num_classes, dtype=torch.float64))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.scale + self.bias


def fit_vector_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    max_iter: int = 150,
    l2: float = 1e-6,
    tolerance_grad: float = 1e-8,
    tolerance_change: float = 1e-10,
) -> VectorScalingFit:
    """Fit multiclass Vector Scaling by minimizing calibration-set NLL.

    The calibrated logits are ``a_k * logit_k + b_k``.  Parameters are
    unconstrained, matching the standard diagonal affine formulation.
    """
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("invalid logits/labels shapes")
    if y.size == 0:
        raise ValueError("cannot fit Vector Scaling on an empty set")
    if y.min() < 0 or y.max() >= x.shape[1]:
        raise ValueError("labels contain invalid class indices")

    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    model = _VectorScaler(x.shape[1])
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=0.5,
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn="strong_wolfe",
    )
    evaluations = 0

    def closure() -> torch.Tensor:
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        output = model(xt)
        loss = F.cross_entropy(output, yt)
        if l2 > 0:
            loss = loss + l2 * (
                (model.scale - 1.0).pow(2).mean() + model.bias.pow(2).mean()
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Vector Scaling objective")
        loss.backward()
        evaluations += 1
        return loss

    converged = True
    try:
        optimizer.step(closure)
    except (RuntimeError, FloatingPointError):
        # Conservative Adam fallback for unusually ill-conditioned logits.
        converged = False
        model = _VectorScaler(x.shape[1])
        adam = torch.optim.Adam(model.parameters(), lr=0.03)
        evaluations = 0
        for _ in range(max_iter * 3):
            adam.zero_grad(set_to_none=True)
            output = model(xt)
            loss = F.cross_entropy(output, yt)
            if l2 > 0:
                loss = loss + l2 * (
                    (model.scale - 1.0).pow(2).mean() + model.bias.pow(2).mean()
                )
            loss.backward()
            adam.step()
            evaluations += 1

    with torch.inference_mode():
        final_nll = float(F.cross_entropy(model(xt), yt))
        scale = model.scale.detach().cpu().numpy().astype(np.float32)
        bias = model.bias.detach().cpu().numpy().astype(np.float32)
    return VectorScalingFit(
        scale=scale,
        bias=bias,
        final_nll=final_nll,
        converged=converged,
        iterations=evaluations,
    )


def _splitter(labels: np.ndarray, n_splits: int, seed: int):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels)
    positive = counts[counts > 0]
    if positive.size and positive.min() >= 2:
        folds = min(n_splits, int(positive.min()))
        folds = max(folds, 2)
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed), True
    folds = min(n_splits, labels.size)
    if folds < 2:
        raise ValueError("at least two calibration examples are required for cross-fitting")
    return KFold(n_splits=folds, shuffle=True, random_state=seed), False


def cross_fitted_vector_scaling(
    calibration_logits: np.ndarray,
    calibration_labels: np.ndarray,
    test_logits: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 1,
    max_iter: int = 150,
    l2: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, VectorScalingFit, dict[str, object]]:
    """Return OOF reference probabilities and full-fit calibration/test probabilities."""
    x = np.asarray(calibration_logits, dtype=np.float32)
    y = np.asarray(calibration_labels, dtype=np.int64).reshape(-1)
    test = np.asarray(test_logits, dtype=np.float32)
    if x.shape[0] != y.size:
        raise ValueError("calibration logits/labels size mismatch")

    splitter, stratified = _splitter(y, n_splits=n_splits, seed=seed)
    oof = np.empty_like(x, dtype=np.float32)
    fold_records: list[dict[str, object]] = []
    split_iter = splitter.split(x, y) if stratified else splitter.split(x)
    for fold, (train_idx, valid_idx) in enumerate(split_iter):
        fit = fit_vector_scaling(
            x[train_idx], y[train_idx], max_iter=max_iter, l2=l2
        )
        oof[valid_idx] = fit.transform_probs(x[valid_idx])
        fold_records.append(
            {
                "fold": fold,
                "train_size": int(train_idx.size),
                "valid_size": int(valid_idx.size),
                "fit_nll": fit.final_nll,
                "converged_lbfgs": fit.converged,
                "iterations": fit.iterations,
            }
        )

    full = fit_vector_scaling(x, y, max_iter=max_iter, l2=l2)
    calibration_in_sample = full.transform_probs(x)
    test_probs = full.transform_probs(test)
    metadata = {
        "requested_folds": int(n_splits),
        "actual_folds": len(fold_records),
        "stratified": stratified,
        "folds": fold_records,
        "full_fit_nll": full.final_nll,
        "full_fit_converged_lbfgs": full.converged,
        "full_fit_iterations": full.iterations,
        "scale": full.scale.tolist(),
        "bias": full.bias.tolist(),
    }
    return oof, calibration_in_sample, test_probs, full, metadata
