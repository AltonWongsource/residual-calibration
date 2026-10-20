from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def load_open_clip(
    *,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cpu",
    cache_dir: str | None = None,
):
    """Load open_clip lazily so tests do not require the optional package."""
    try:
        import open_clip  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "open_clip_torch is required for real CLIP execution. Install it "
            "or run inside the environment used by the existing frozen-CLIP experiment."
        ) from exc
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device, **kwargs
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    model.float()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer, preprocess_train, preprocess_val


def encode_text_templates(
    *,
    model,
    tokenizer,
    class_names: Sequence[str],
    templates: Sequence[str],
    device: str = "cpu",
) -> np.ndarray:
    if not templates:
        raise ValueError("templates must be non-empty")
    outputs = []
    with torch.no_grad():
        for name in class_names:
            texts = [template.format(str(name).replace("_", " ")) for template in templates]
            tokens = tokenizer(texts).to(device)
            feats = F.normalize(model.encode_text(tokens).float(), dim=-1)
            feat = F.normalize(feats.mean(dim=0), dim=-1)
            outputs.append(feat.cpu())
    return torch.stack(outputs).numpy().astype(np.float32)
