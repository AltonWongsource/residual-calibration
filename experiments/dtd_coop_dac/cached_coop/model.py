from __future__ import annotations

import inspect
from typing import Callable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


class PromptLearner(nn.Module):
    """Shared-context CoOp prompt learner.

    This follows the standard CoOp parameterization: frozen CLIP encoders and
    a trainable sequence of continuous context tokens shared across classes.
    """

    def __init__(
        self,
        *,
        class_names: Sequence[str],
        clip_model: nn.Module,
        tokenizer: Callable[[Sequence[str]], torch.Tensor],
        n_ctx: int = 16,
        ctx_init: str | None = None,
        class_token_position: str = "end",
    ) -> None:
        super().__init__()
        if not class_names:
            raise ValueError("class_names must be non-empty")
        if class_token_position not in {"end"}:
            raise ValueError("only class_token_position='end' is supported")
        self.class_names = [str(name).replace("_", " ") for name in class_names]
        self.n_cls = len(self.class_names)
        self.class_token_position = class_token_position
        dtype = getattr(clip_model, "dtype", torch.float32)
        ctx_dim = int(clip_model.token_embedding.weight.shape[1])

        if ctx_init:
            clean = ctx_init.replace("_", " ").strip()
            n_ctx = len(clean.split())
            tokenized = tokenizer([clean])
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized).to(dtype=dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :].clone()
            prompt_prefix = clean
        else:
            if n_ctx < 1:
                raise ValueError("n_ctx must be positive")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.n_ctx = int(n_ctx)
        self.ctx = nn.Parameter(ctx_vectors)
        prompts = [f"{prompt_prefix} {name}." for name in self.class_names]
        tokenized_prompts = tokenizer(prompts)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).to(dtype=dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :], persistent=False)
        self.register_buffer(
            "token_suffix", embedding[:, 1 + self.n_ctx :, :], persistent=False
        )
        self.register_buffer("tokenized_prompts", tokenized_prompts, persistent=True)

    def forward(self) -> torch.Tensor:
        ctx = self.ctx
        if ctx.ndim == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


class TextEncoder(nn.Module):
    """Thin adapter around OpenAI/open_clip style CLIP text components."""

    def __init__(self, clip_model: nn.Module) -> None:
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.attn_mask = getattr(clip_model, "attn_mask", None)
        self.dtype = getattr(clip_model, "dtype", torch.float32)
        self.batch_first = bool(getattr(self.transformer, "batch_first", False))

    def _run_transformer(self, x: torch.Tensor) -> torch.Tensor:
        # OpenAI CLIP accepts transformer(x). open_clip commonly accepts
        # transformer(x, attn_mask=...). Try the explicit mask first only when
        # the signature supports it, then fall back cleanly.
        try:
            sig = inspect.signature(self.transformer.forward)
            if "attn_mask" in sig.parameters:
                return self.transformer(x, attn_mask=self.attn_mask)
        except (TypeError, ValueError):
            pass
        return self.transformer(x)

    def forward(
        self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor
    ) -> torch.Tensor:
        x = prompts.to(dtype=self.dtype)
        pos = self.positional_embedding[: x.shape[1]].to(dtype=self.dtype)
        x = x + pos
        if not self.batch_first:
            x = x.permute(1, 0, 2)
        x = self._run_transformer(x)
        if not self.batch_first:
            x = x.permute(1, 0, 2)
        x = self.ln_final(x).to(dtype=self.dtype)
        eot = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), eot]
        projection = self.text_projection
        if isinstance(projection, nn.Module):
            x = projection(x)
        else:
            x = x @ projection
        return x


class CachedCoOpModel(nn.Module):
    """CoOp classifier operating on frozen cached CLIP image features."""

    def __init__(
        self,
        *,
        class_names: Sequence[str],
        clip_model: nn.Module,
        tokenizer: Callable[[Sequence[str]], torch.Tensor],
        n_ctx: int = 16,
        ctx_init: str | None = None,
    ) -> None:
        super().__init__()
        self.prompt_learner = PromptLearner(
            class_names=class_names,
            clip_model=clip_model,
            tokenizer=tokenizer,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
        )
        self.text_encoder = TextEncoder(clip_model)
        raw_scale = getattr(clip_model, "logit_scale", None)
        if raw_scale is None:
            self.register_buffer("logit_scale", torch.tensor(1.0), persistent=False)
        else:
            scale = raw_scale.detach().float().exp() if raw_scale.ndim == 0 else raw_scale.detach().float()
            self.register_buffer("logit_scale", scale.reshape(()), persistent=False)

        # Freeze every CLIP component. Only context vectors remain trainable.
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def text_features(self) -> torch.Tensor:
        prompts = self.prompt_learner()
        tokens = self.prompt_learner.tokenized_prompts
        features = self.text_encoder(prompts, tokens)
        return F.normalize(features.float(), dim=-1)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        image_features = F.normalize(image_features.float(), dim=-1)
        text_features = self.text_features()
        return self.logit_scale.float() * image_features @ text_features.t()
