from __future__ import annotations

import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    """Small MLP that exposes its final hidden representation for LAPE."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if input_dim < 1 or num_classes < 2:
            raise ValueError("invalid input_dim or num_classes")
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev = hidden
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_classes)
        self.feature_dim = prev

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class FrozenFeatureHead(nn.Module):
    """Trainable head on cached frozen-backbone features."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
