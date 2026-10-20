#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from cached_coop.cache_io import load_feature_cache, validate_feature_cache
from cached_coop.clip_adapter import encode_text_templates, load_open_clip
from cached_coop.trainer import (
    CachedCoOpConfig,
    save_cached_coop_result,
    train_cached_coop,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one CPU-efficient feature-cached CoOp checkpoint.")
    p.add_argument("--cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="ViT-B-32")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--clip-cache-dir", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-ctx", type=int, default=16)
    p.add_argument("--ctx-init", default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=0.002)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--view-sampling", choices=["random", "mean"], default="random")
    p.add_argument("--restore-best-validation", action="store_true", help="Optional; default uses the fixed final epoch to avoid calibration-set model selection.")
    p.add_argument(
        "--pretrained-template",
        action="append",
        default=None,
        help="May be passed multiple times; default: 'a photo of a {}.'",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache = load_feature_cache(args.cache)
    validate_feature_cache(cache)
    model, tokenizer, _, _ = load_open_clip(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
        cache_dir=args.clip_cache_dir,
    )
    class_names = np.asarray(cache["class_names"]).astype(str).tolist()
    templates = args.pretrained_template or ["a photo of a {}."]
    pretrained_text = encode_text_templates(
        model=model,
        tokenizer=tokenizer,
        class_names=class_names,
        templates=templates,
        device=args.device,
    )
    cfg = CachedCoOpConfig(
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
        view_sampling=args.view_sampling,
        restore_best_validation=args.restore_best_validation,
    )
    result = train_cached_coop(
        clip_model=model,
        tokenizer=tokenizer,
        class_names=class_names,
        base_class_mask=cache["base_class_mask"],
        train_features=cache["base_train_features"],
        train_labels=cache["base_train_labels"],
        validation_features=cache["base_calibration_features"],
        validation_labels=cache["base_calibration_labels"],
        config=cfg,
        device=args.device,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_cached_coop_result(
        result,
        checkpoint_path=output / "cached_coop_checkpoint.pt",
        history_path=output / "training_history.json",
        metadata={
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "feature_cache": str(Path(args.cache).resolve()),
            "pretrained_templates": templates,
            "protocol_name": "feature-cached standard CoOp objective",
        },
    )
    logit_scale = float(result.model.logit_scale.detach().cpu())
    np.savez_compressed(
        output / "coop_text_outputs.npz",
        pretrained_text_features=pretrained_text,
        tuned_text_features=result.tuned_text_features,
        logit_scale=np.asarray(logit_scale, dtype=np.float32),
        class_names=np.asarray(class_names, dtype=object),
        base_class_mask=np.asarray(cache["base_class_mask"], dtype=bool),
    )
    summary = {
        "best_epoch": result.best_epoch,
        "best_validation_nll": result.best_validation_nll,
        "epochs_completed": len(result.history),
        "logit_scale": logit_scale,
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
