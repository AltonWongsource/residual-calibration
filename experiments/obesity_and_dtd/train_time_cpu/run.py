from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from common.io import save_json, save_prediction_bundle
from common.metrics import evaluate_probabilities
from common.repro import ensure_dir, seed_everything
from train_time_cpu.data import (
    ArraySplits,
    build_breastmnist_splits,
    build_obesity_splits,
    make_loader,
)
from train_time_cpu.losses import LossConfig, build_loss
from train_time_cpu.models import FrozenFeatureHead, TabularMLP


@dataclass(frozen=True)
class Candidate:
    mdca_beta: float = 1.0
    mbls_margin: float = 5.0
    mbls_lambda: float = 0.1

    @property
    def name(self) -> str:
        return (
            f"beta_{self.mdca_beta:g}"
            if self.mdca_beta != 1.0
            else f"margin_{self.mbls_margin:g}_lambda_{self.mbls_lambda:g}"
        )


def parse_float_list(text: str) -> list[float]:
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one number")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CPU-first CE/MDCA/MbLS comparison on real classification data."
    )
    p.add_argument("--dataset", choices=["obesity", "breastmnist"], required=True)
    p.add_argument("--loss", choices=["ce", "mdca", "mbls"], required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--obesity-csv")
    p.add_argument("--feature-cache", default="./cache/breastmnist_resnet18.npz")
    p.add_argument("--force-rebuild-cache", action="store_true")
    p.add_argument("--output-root", default="./outputs_cpu/train_time")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--feature-batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cpu-threads", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--mdca-betas", type=parse_float_list, default=[0.1, 0.5, 1.0])
    p.add_argument("--mbls-margins", type=parse_float_list, default=[1.0, 3.0, 5.0])
    p.add_argument("--mbls-lambda", type=float, default=0.1)
    p.add_argument("--ece-bins", type=int, default=15)
    p.add_argument("--skip-posthoc", action="store_true")
    p.add_argument("--posthoc-folds", type=int, default=5)
    p.add_argument("--posthoc-alpha-step", type=float, default=0.05)
    p.add_argument("--posthoc-feature-mode", choices=["none", "zscore", "l2"], default="none")
    p.add_argument("--posthoc-pca-dim", type=int, default=0)
    p.add_argument("--include-direct-reuse", action="store_true")
    p.add_argument("--quick", action="store_true", help="Two-epoch smoke test with one candidate.")
    return p.parse_args()


def candidate_grid(args: argparse.Namespace) -> list[Candidate]:
    if args.loss == "ce":
        return [Candidate()]
    if args.loss == "mdca":
        return [Candidate(mdca_beta=value) for value in args.mdca_betas]
    return [
        Candidate(mbls_margin=value, mbls_lambda=args.mbls_lambda)
        for value in args.mbls_margins
    ]


def build_model(args: argparse.Namespace, splits: ArraySplits) -> torch.nn.Module:
    input_dim = int(splits.train_x.shape[1])
    num_classes = len(splits.class_names)
    if args.dataset == "obesity":
        return TabularMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=(args.hidden_dim, max(args.hidden_dim // 2, 16)),
            dropout=args.dropout,
        )
    return FrozenFeatureHead(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )


def evaluate_loader(
    model: torch.nn.Module,
    loader: Iterable,
    *,
    device: torch.device,
    n_bins: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    indices_all: list[np.ndarray] = []
    input_features_all: list[np.ndarray] = []
    hidden_features_all: list[np.ndarray] = []
    with torch.inference_mode():
        for inputs, labels, indices in loader:
            inputs = inputs.to(device)
            if isinstance(model, TabularMLP):
                logits, hidden = model(inputs, return_features=True)
                exported_features = hidden
            else:
                logits = model(inputs)
                exported_features = inputs
            logits_all.append(logits.cpu().numpy())
            labels_all.append(labels.numpy())
            indices_all.append(indices.numpy())
            input_features_all.append(inputs.cpu().numpy())
            hidden_features_all.append(exported_features.cpu().numpy())
    logits = np.concatenate(logits_all, axis=0).astype(np.float32)
    labels = np.concatenate(labels_all, axis=0).astype(np.int64)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy().astype(np.float32)
    payload = {
        "logits": logits,
        "probs": probs,
        "features": np.concatenate(hidden_features_all, axis=0).astype(np.float32),
        "labels": labels,
        "indices": np.concatenate(indices_all, axis=0).astype(np.int64),
        "model_inputs": np.concatenate(input_features_all, axis=0).astype(np.float32),
    }
    return evaluate_probabilities(probs, labels, n_bins=n_bins), payload


def train_candidate(
    *,
    args: argparse.Namespace,
    splits: ArraySplits,
    candidate: Candidate,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    # Reset to the same initialization for fair candidate comparison.
    seed_everything(args.seed, deterministic=True)
    model = build_model(args, splits).to(device)
    criterion = build_loss(
        LossConfig(
            name=args.loss,
            num_classes=len(splits.class_names),
            mdca_beta=candidate.mdca_beta,
            mbls_margin=candidate.mbls_margin,
            mbls_lambda=candidate.mbls_lambda,
        )
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    train_loader = make_loader(
        splits.train_x,
        splits.train_y,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.workers,
    )
    selection_loader = make_loader(
        splits.selection_x,
        splits.selection_y,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.workers,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_selection_nll = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, float]] = []
    max_epochs = 2 if args.quick else args.max_epochs
    patience = 2 if args.quick else args.patience
    min_epochs = 1 if args.quick else args.min_epochs

    for epoch in range(1, max_epochs + 1):
        start = time.time()
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for inputs, labels, _ in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch = labels.numel()
            total_loss += float(loss.detach()) * batch
            total_correct += int((logits.argmax(dim=1) == labels).sum())
            total_count += batch

        selection_metrics, _ = evaluate_loader(
            model, selection_loader, device=device, n_bins=args.ece_bins
        )
        scheduler.step(selection_metrics["nll"])
        row = {
            "epoch": float(epoch),
            "train_loss": total_loss / max(total_count, 1),
            "train_accuracy": total_correct / max(total_count, 1),
            "selection_nll": selection_metrics["nll"],
            "selection_accuracy": selection_metrics["accuracy"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.time() - start,
        }
        history.append(row)
        print(
            f"dataset={args.dataset} loss={args.loss} candidate={candidate} "
            f"epoch={epoch:03d} train_loss={row['train_loss']:.5f} "
            f"selection_nll={row['selection_nll']:.5f}"
        )
        if selection_metrics["nll"] < best_selection_nll - 1e-6:
            best_selection_nll = selection_metrics["nll"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch >= min_epochs and bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    return model, {
        "candidate": candidate.__dict__,
        "best_selection_nll": best_selection_nll,
        "best_epoch": best_epoch,
        "history": history,
    }


def append_csv(path: Path, row: dict[str, object]) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, args.cpu_threads)))
        except RuntimeError:
            pass
    seed_everything(args.seed, deterministic=True)
    device = torch.device("cpu")

    if args.dataset == "obesity":
        splits = build_obesity_splits(seed=args.seed, csv_path=args.obesity_csv)
    else:
        splits = build_breastmnist_splits(
            seed=args.seed,
            data_root=args.data_root,
            cache_path=args.feature_cache,
            feature_batch_size=args.feature_batch_size,
            workers=args.workers,
            force_rebuild_cache=args.force_rebuild_cache,
        )

    records: list[tuple[torch.nn.Module, dict[str, object]]] = []
    grid = candidate_grid(args)
    if args.quick:
        grid = grid[:1]
    for candidate in grid:
        records.append(
            train_candidate(args=args, splits=splits, candidate=candidate, device=device)
        )
    best_model, best_record = min(records, key=lambda item: item[1]["best_selection_nll"])

    calibration_loader = make_loader(
        splits.calibration_x,
        splits.calibration_y,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.workers,
    )
    test_loader = make_loader(
        splits.test_x,
        splits.test_y,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.workers,
    )
    calibration_metrics, calibration = evaluate_loader(
        best_model, calibration_loader, device=device, n_bins=args.ece_bins
    )
    test_metrics, test = evaluate_loader(
        best_model, test_loader, device=device, n_bins=args.ece_bins
    )

    run_dir = ensure_dir(
        Path(args.output_root) / args.dataset / args.loss / f"seed_{args.seed}"
    )
    torch.save(best_model.state_dict(), run_dir / "model_best.pt")
    config = vars(args) | {
        "device": "cpu",
        "num_classes": len(splits.class_names),
        "class_names": splits.class_names,
        "selected_candidate": best_record["candidate"],
        "selected_epoch": best_record["best_epoch"],
        "selection_nll": best_record["best_selection_nll"],
        "data_metadata": splits.metadata,
        "scope": (
            "Obesity uses full MLP training. BreastMNIST uses a frozen ImageNet ResNet-18 "
            "and trains only the prediction head under CE/MDCA/MbLS."
        ),
    }
    save_json(run_dir / "config.json", config)
    save_json(
        run_dir / "candidate_search.json",
        [record for _, record in records],
    )
    metadata = {
        "dataset": args.dataset,
        "train_time_loss": args.loss,
        "seed": args.seed,
        "selected_candidate": best_record["candidate"],
        "lape_note": (
            "Fit the post-hoc anchor on calibration logits. If calibration points are reused as "
            "LAPE references, construct out-of-fold anchor probabilities for those points."
        ),
    }
    save_prediction_bundle(
        run_dir / "calibration.npz",
        logits=calibration["logits"],
        probs=calibration["probs"],
        features=calibration["features"],
        labels=calibration["labels"],
        indices=calibration["indices"],
        class_names=splits.class_names,
        metadata=metadata | {"split": "calibration"},
    )
    save_prediction_bundle(
        run_dir / "test.npz",
        logits=test["logits"],
        probs=test["probs"],
        features=test["features"],
        labels=test["labels"],
        indices=test["indices"],
        class_names=splits.class_names,
        metadata=metadata | {"split": "test"},
    )
    save_json(
        run_dir / "metrics.json",
        {"calibration": calibration_metrics, "test": test_metrics},
    )
    row: dict[str, object] = {
        "dataset": args.dataset,
        "loss": args.loss,
        "seed": args.seed,
        "selected_candidate": json.dumps(best_record["candidate"], sort_keys=True),
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"calibration_{k}": v for k, v in calibration_metrics.items()},
        "run_dir": str(run_dir),
    }
    append_csv(Path(args.output_root) / "summary.csv", row)

    if not args.skip_posthoc:
        from posthoc_cpu.pipeline import PosthocConfig, run_posthoc_comparison

        posthoc = run_posthoc_comparison(
            calibration_path=run_dir / "calibration.npz",
            test_path=run_dir / "test.npz",
            output_dir=run_dir / "posthoc",
            config=PosthocConfig(
                folds=args.posthoc_folds,
                seed=args.seed,
                ece_bins=args.ece_bins,
                alpha_step=args.posthoc_alpha_step,
                feature_mode=args.posthoc_feature_mode,
                pca_dim=args.posthoc_pca_dim,
                include_direct_reuse=args.include_direct_reuse,
            ),
        )
        row["posthoc_dir"] = posthoc["output_dir"]

    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
