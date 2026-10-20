from __future__ import annotations

import numpy as np
import torch
from torch import nn

from cached_coop.trainer import CachedCoOpConfig, train_cached_coop


class ToyTransformer(nn.Module):
    def forward(self, x, attn_mask=None):
        # x is [L, B, D]. Make every token, including EOT, depend on context.
        return x + 0.2 * x.mean(dim=0, keepdim=True)


class ToyClip(nn.Module):
    def __init__(self, vocab=128, width=8, length=12):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, width)
        self.positional_embedding = nn.Parameter(torch.zeros(length, width), requires_grad=False)
        self.transformer = ToyTransformer()
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.eye(width), requires_grad=False)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(5.0), dtype=torch.float32), requires_grad=False)
        self.dtype = torch.float32


def tokenizer(texts):
    out = torch.zeros((len(texts), 12), dtype=torch.long)
    for i, text in enumerate(texts):
        words = text.lower().replace('.', '').split()
        out[i, 0] = 1
        for j, word in enumerate(words[:9], start=1):
            out[i, j] = 2 + (sum(map(ord, word)) % 100)
        out[i, min(len(words) + 1, 10)] = 127
    return out


def test_cached_coop_training_smoke():
    rng = np.random.default_rng(4)
    class_names = ["class_a", "class_b", "class_c", "class_d"]
    base_mask = np.array([True, True, False, False])
    train_y = np.repeat([0, 1], 8)
    val_y = np.repeat([0, 1], 4)
    centers = np.eye(8, dtype=np.float32)[:2]
    train_x = centers[train_y] + 0.05 * rng.normal(size=(16, 8))
    train_x = np.stack([train_x, train_x + 0.01 * rng.normal(size=train_x.shape)], axis=1).astype(np.float32)
    val_x = (centers[val_y] + 0.05 * rng.normal(size=(8, 8))).astype(np.float32)
    result = train_cached_coop(
        clip_model=ToyClip(),
        tokenizer=tokenizer,
        class_names=class_names,
        base_class_mask=base_mask,
        train_features=train_x,
        train_labels=train_y,
        validation_features=val_x,
        validation_labels=val_y,
        config=CachedCoOpConfig(epochs=4, n_ctx=2, patience=0, seed=3),
    )
    assert result.tuned_text_features.shape == (4, 8)
    assert np.all(np.isfinite(result.tuned_text_features))
    assert len(result.history) == 4
    assert result.best_epoch >= 1
