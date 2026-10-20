from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .model import CachedCoOpModel


@dataclass(frozen=True)
class CachedCoOpConfig:
    n_ctx: int = 16
    ctx_init: str | None = None
    epochs: int = 100
    learning_rate: float = 0.002
    momentum: float = 0.9
    weight_decay: float = 0.0
    seed: int = 1
    patience: int = 25
    min_delta: float = 1e-5
    full_batch: bool = True
    batch_size: int = 256
    view_sampling: str = "random"  # random | mean
    restore_best_validation: bool = False


@dataclass
class CachedCoOpResult:
    model: CachedCoOpModel
    tuned_text_features: np.ndarray
    history: list[dict[str, float]]
    best_epoch: int
    best_validation_nll: float
    config: CachedCoOpConfig


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collapse_views(features: np.ndarray, *, mode: str = "mean") -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError("features must be [N,D] or [N,V,D]")
    if mode != "mean":
        raise ValueError("collapse mode must be 'mean'")
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    normalized = arr / np.maximum(norm, 1e-12)
    out = normalized.mean(axis=1)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out.astype(np.float32)


def _epoch_features(features: torch.Tensor, mode: str) -> torch.Tensor:
    if features.ndim == 2:
        return features
    if features.ndim != 3:
        raise ValueError("training features must be [N,D] or [N,V,D]")
    if mode == "mean":
        return F.normalize(F.normalize(features, dim=-1).mean(dim=1), dim=-1)
    if mode == "random":
        n, v, _ = features.shape
        choice = torch.randint(v, (n,), device=features.device)
        return features[torch.arange(n, device=features.device), choice]
    raise ValueError("view_sampling must be random or mean")


def _local_labels(labels: np.ndarray, base_indices: np.ndarray) -> np.ndarray:
    mapping = np.full(int(base_indices.max()) + 1, -1, dtype=np.int64)
    mapping[base_indices] = np.arange(base_indices.size, dtype=np.int64)
    if labels.max(initial=0) >= mapping.size or np.any(mapping[labels] < 0):
        raise ValueError("training/validation labels must be base-class global labels")
    return mapping[labels]


def _nll(
    model: CachedCoOpModel,
    features: torch.Tensor,
    labels_local: torch.Tensor,
    base_indices_t: torch.Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        if features.ndim == 3:
            features = F.normalize(F.normalize(features, dim=-1).mean(dim=1), dim=-1)
        logits = model(features)[:, base_indices_t]
        return float(F.cross_entropy(logits, labels_local).item())


def train_cached_coop(
    *,
    clip_model: nn.Module,
    tokenizer: Callable[[Sequence[str]], torch.Tensor],
    class_names: Sequence[str],
    base_class_mask: np.ndarray,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    config: CachedCoOpConfig | None = None,
    device: str | torch.device = "cpu",
) -> CachedCoOpResult:
    cfg = config or CachedCoOpConfig()
    _set_seed(cfg.seed)
    dev = torch.device(device)
    base_mask = np.asarray(base_class_mask, dtype=bool).reshape(-1)
    if base_mask.size != len(class_names) or not np.any(base_mask):
        raise ValueError("invalid base_class_mask")
    base_indices = np.flatnonzero(base_mask).astype(np.int64)
    y_train_global = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    y_val_global = np.asarray(validation_labels, dtype=np.int64).reshape(-1)
    y_train = _local_labels(y_train_global, base_indices)
    y_val = _local_labels(y_val_global, base_indices)

    model = CachedCoOpModel(
        class_names=class_names,
        clip_model=clip_model,
        tokenizer=tokenizer,
        n_ctx=cfg.n_ctx,
        ctx_init=cfg.ctx_init,
    ).to(dev)
    train_x = torch.as_tensor(np.asarray(train_features, dtype=np.float32), device=dev)
    val_x = torch.as_tensor(np.asarray(validation_features, dtype=np.float32), device=dev)
    train_y = torch.as_tensor(y_train, dtype=torch.long, device=dev)
    val_y = torch.as_tensor(y_val, dtype=torch.long, device=dev)
    base_indices_t = torch.as_tensor(base_indices, dtype=torch.long, device=dev)

    optimizer = torch.optim.SGD(
        [model.prompt_learner.ctx],
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs, 1)
    )
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.prompt_learner.state_dict())
    best_val = math.inf
    best_epoch = 0
    stale = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_x = _epoch_features(train_x, cfg.view_sampling)
        if cfg.full_batch or epoch_x.shape[0] <= cfg.batch_size:
            batches = [(epoch_x, train_y)]
        else:
            order = torch.randperm(epoch_x.shape[0], device=dev)
            batches = [
                (epoch_x[idx], train_y[idx])
                for idx in order.split(cfg.batch_size)
            ]
        weighted_loss = 0.0
        seen = 0
        for xb, yb in batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)[:, base_indices_t]
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.item()) * int(yb.numel())
            seen += int(yb.numel())
        scheduler.step()
        train_loss = weighted_loss / max(seen, 1)
        val_loss = _nll(model, val_x, val_y, base_indices_t)
        history.append(
            {
                "epoch": float(epoch),
                "train_nll": train_loss,
                "validation_nll": val_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.prompt_learner.state_dict())
            stale = 0
        else:
            stale += 1
        if cfg.patience > 0 and stale >= cfg.patience:
            break

    if cfg.restore_best_validation:
        model.prompt_learner.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tuned = model.text_features().cpu().numpy().astype(np.float32)
    return CachedCoOpResult(
        model=model,
        tuned_text_features=tuned,
        history=history,
        best_epoch=best_epoch,
        best_validation_nll=float(best_val),
        config=cfg,
    )


def save_cached_coop_result(
    result: CachedCoOpResult,
    *,
    checkpoint_path: str | Path,
    history_path: str | Path,
    metadata: dict[str, object] | None = None,
) -> None:
    checkpoint = {
        "prompt_learner": result.model.prompt_learner.state_dict(),
        "config": asdict(result.config),
        "best_epoch": result.best_epoch,
        "best_validation_nll": result.best_validation_nll,
        "metadata": metadata or {},
    }
    p = Path(checkpoint_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, p)
    h = Path(history_path)
    h.parent.mkdir(parents=True, exist_ok=True)
    h.write_text(json.dumps(result.history, indent=2), encoding="utf-8")
