from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class MDCALoss(nn.Module):
    """Cross-entropy plus Multi-class Difference in Confidence and Accuracy.

    MDCA is computed per mini-batch as the mean absolute difference between
    the classwise average predicted probability and the class frequency.

    Reference:
      Hebbalaguppe et al., CVPR 2022.
    """

    def __init__(self, num_classes: int, beta: float = 1.0) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        self.num_classes = int(num_classes)
        self.beta = float(beta)

    def mdca_term(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"expected logits [B, {self.num_classes}], got {tuple(logits.shape)}"
            )
        probs = F.softmax(logits, dim=1)
        target_freq = F.one_hot(targets, num_classes=self.num_classes).to(probs.dtype).mean(dim=0)
        pred_mass = probs.mean(dim=0)
        return torch.abs(pred_mass - target_freq).mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets)
        return ce + self.beta * self.mdca_term(logits, targets)


class MbLSLoss(nn.Module):
    """Cross-entropy plus margin-based logit-distance smoothing.

    For each sample, let z_max be the largest logit. MbLS penalizes only
    distances z_max - z_k that exceed `margin`:

        CE + lambda * mean_k ReLU(z_max - z_k - margin).

    The winning class contributes zero automatically. The original paper fixes
    lambda=0.1 and tunes the margin on a validation split; margin=10 is a
    reasonable pre-specified default for harder/many-class image tasks.

    Reference:
      Liu et al., CVPR 2022.
    """

    def __init__(self, margin: float = 10.0, penalty_weight: float = 0.1) -> None:
        super().__init__()
        if margin < 0:
            raise ValueError("margin must be non-negative")
        if penalty_weight < 0:
            raise ValueError("penalty_weight must be non-negative")
        self.margin = float(margin)
        self.penalty_weight = float(penalty_weight)

    def penalty(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"expected logits [B, K], got {tuple(logits.shape)}")
        winner = logits.max(dim=1, keepdim=True).values
        distances = winner - logits
        return F.relu(distances - self.margin).mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets)
        return ce + self.penalty_weight * self.penalty(logits)


@dataclass(frozen=True)
class LossConfig:
    name: str
    num_classes: int
    mdca_beta: float = 1.0
    mbls_margin: float = 10.0
    mbls_lambda: float = 0.1


def build_loss(cfg: LossConfig) -> nn.Module:
    name = cfg.name.lower()
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "mdca":
        return MDCALoss(num_classes=cfg.num_classes, beta=cfg.mdca_beta)
    if name == "mbls":
        return MbLSLoss(margin=cfg.mbls_margin, penalty_weight=cfg.mbls_lambda)
    raise ValueError(f"unknown loss: {cfg.name}")
