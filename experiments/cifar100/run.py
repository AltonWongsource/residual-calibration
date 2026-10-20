from __future__ import annotations

import argparse
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models


WORKSPACE_ROOT = Path(__file__).resolve().parent
import calibration_core as runner


DATA_ROOT = WORKSPACE_ROOT / "data"
OUT_ROOT = WORKSPACE_ROOT / "outputs" / "vision_cifar10_resnet18"
STATUS_PATH = OUT_ROOT / "status.json"
METHODS = runner.SINGLE_VIEW_STANDARD_METHODS
CASCADE_METHODS = runner.SINGLE_VIEW_CASCADE_METHODS


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_status() -> dict[str, Any]:
    if not STATUS_PATH.exists():
        return {"created_at": now_iso(), "stages": {}, "methods": {}}
    return json.loads(STATUS_PATH.read_text(encoding="utf-8"))


def save_status(status: dict[str, Any]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")


def mark_stage(status: dict[str, Any], stage: str, state: str, **extra: Any) -> None:
    status.setdefault("stages", {}).setdefault(stage, {}).update({"state": state, "time": now_iso(), **extra})
    save_status(status)


def make_backbone(backbone_name: str) -> tuple[torch.nn.Module, Any, int, str]:
    if backbone_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        feature_dim = 512
        label = "ImageNet-pretrained ResNet18, frozen feature extractor"
    elif backbone_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        feature_dim = 2048
        label = "ImageNet-pretrained ResNet50, frozen feature extractor"
    elif backbone_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        model.classifier = torch.nn.Identity()
        feature_dim = 1280
        label = "ImageNet-pretrained EfficientNet-B0, frozen feature extractor"
    elif backbone_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = torch.nn.Identity()
        feature_dim = 768
        label = "ImageNet-pretrained ConvNeXt-Tiny, frozen feature extractor"
    elif backbone_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
        model.heads = torch.nn.Identity()
        feature_dim = 768
        label = "ImageNet-pretrained ViT-B/16, frozen feature extractor"
    elif backbone_name == "swin_t":
        weights = models.Swin_T_Weights.DEFAULT
        model = models.swin_t(weights=weights)
        model.head = torch.nn.Identity()
        feature_dim = 768
        label = "ImageNet-pretrained Swin-T, frozen feature extractor"
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    model.eval()
    return model, weights.transforms(), feature_dim, label


def stratified_subset(indices: np.ndarray, labels: np.ndarray, n_take: int, seed: int) -> np.ndarray:
    if n_take >= len(indices):
        return indices
    keep, _ = train_test_split(indices, train_size=n_take, random_state=seed, stratify=labels[indices])
    return np.asarray(keep, dtype=int)


def configure_output(dataset_name: str, backbone_name: str) -> None:
    global OUT_ROOT, STATUS_PATH
    OUT_ROOT = runner.OUTPUT_DIR / f"vision_{dataset_name}_{backbone_name}"
    STATUS_PATH = OUT_ROOT / "status.json"


def prepare_cifar_splits(
    dataset_name: str,
    backbone_name: str,
    seed: int,
    quick: bool,
) -> tuple[dict[str, Subset], dict[str, np.ndarray], dict[str, Any]]:
    _, preprocess, feature_dim, backbone_label = make_backbone(backbone_name)
    dataset_cls = datasets.CIFAR100 if dataset_name == "cifar100" else datasets.CIFAR10
    train_dataset = dataset_cls(root=str(DATA_ROOT), train=True, transform=preprocess, download=True)
    test_dataset = dataset_cls(root=str(DATA_ROOT), train=False, transform=preprocess, download=True)
    train_labels_all = np.asarray(train_dataset.targets, dtype=int)
    test_labels_all = np.asarray(test_dataset.targets, dtype=int)
    n_classes = 100 if dataset_name == "cifar100" else 10

    all_train_idx = np.arange(len(train_dataset))
    train_idx, cal_idx = train_test_split(
        all_train_idx,
        test_size=10000,
        random_state=seed,
        stratify=train_labels_all,
    )
    test_idx = np.arange(len(test_dataset))

    if quick:
        train_idx = stratified_subset(np.asarray(train_idx), train_labels_all, 1200, seed)
        cal_idx = stratified_subset(np.asarray(cal_idx), train_labels_all, 600, seed + 1)
        test_idx = stratified_subset(test_idx, test_labels_all, 600, seed + 2)

    splits = {
        "train": Subset(train_dataset, train_idx.tolist()),
        "cal": Subset(train_dataset, cal_idx.tolist()),
        "test": Subset(test_dataset, test_idx.tolist()),
    }
    labels = {
        "train": train_labels_all[train_idx],
        "cal": train_labels_all[cal_idx],
        "test": test_labels_all[test_idx],
    }
    meta = {
        "dataset": "CIFAR-100" if dataset_name == "cifar100" else "CIFAR-10",
        "backbone": backbone_label,
        "head": "multinomial logistic-regression linear probe on frozen CNN features",
        "split_scheme": "CIFAR train split -> 40k train / 10k calibration; CIFAR test split -> 10k test",
        "quick": quick,
        "feature_dim": int(feature_dim),
        "n_train": int(len(labels["train"])),
        "n_cal": int(len(labels["cal"])),
        "n_test": int(len(labels["test"])),
        "n_classes": n_classes,
    }
    return splits, labels, meta


@torch.inference_mode()
def extract_split_features(
    split_name: str,
    dataset: Subset,
    labels: np.ndarray,
    backbone_name: str,
    batch_size: int,
    status: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    combined_path = OUT_ROOT / "features" / f"{split_name}.npz"
    if combined_path.exists():
        data = np.load(combined_path)
        return np.asarray(data["features"], dtype=np.float32), np.asarray(data["labels"], dtype=int)

    feature_root = OUT_ROOT / "features" / split_name
    feature_root.mkdir(parents=True, exist_ok=True)
    model, _, _, _ = make_backbone(backbone_name)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_batches = len(loader)
    for batch_idx, (images, y_batch) in enumerate(loader):
        chunk_path = feature_root / f"chunk_{batch_idx:05d}.npz"
        if chunk_path.exists():
            continue
        feats = model(images).detach().cpu().numpy().astype(np.float32)
        np.savez_compressed(chunk_path, features=feats, labels=y_batch.numpy().astype(int))
        mark_stage(
            status,
            f"feature_{split_name}",
            "running",
            completed_batches=batch_idx + 1,
            total_batches=total_batches,
        )

    chunks = []
    chunk_labels = []
    for path in sorted(feature_root.glob("chunk_*.npz")):
        data = np.load(path)
        chunks.append(np.asarray(data["features"], dtype=np.float32))
        chunk_labels.append(np.asarray(data["labels"], dtype=int))
    features = np.vstack(chunks)
    y = np.concatenate(chunk_labels)
    if len(y) != len(labels):
        raise RuntimeError(f"{split_name} feature count mismatch: got {len(y)} expected {len(labels)}")
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(combined_path, features=features, labels=y)
    mark_stage(status, f"feature_{split_name}", "completed", n_samples=int(len(y)))
    return features, y


def linear_logits(head: LogisticRegression, features: np.ndarray) -> np.ndarray:
    logits = np.asarray(features, dtype=np.float64) @ head.coef_.T + head.intercept_
    if logits.ndim == 1:
        logits = logits[:, None]
    if logits.shape[1] == 1:
        logits = np.concatenate([-logits, logits], axis=1)
    return logits


def choose_linear_fc_clip(features: np.ndarray, labels: np.ndarray, head: LogisticRegression) -> float:
    base_probs = runner.clip_probs(softmax(linear_logits(head, features), axis=1))
    base_acc = float(np.mean(np.argmax(base_probs, axis=1) == labels))
    high = float(np.quantile(np.abs(features).ravel(), 0.995))
    low = max(0.01, high / 100.0)
    candidates = np.unique(np.concatenate([np.linspace(low, high, 80), [1.25 * high, 1.5 * high]]))
    best_c = high
    best_nll = float("inf")
    for c in candidates:
        probs = runner.clip_probs(softmax(linear_logits(head, np.clip(features, -c, c)), axis=1))
        acc = float(np.mean(np.argmax(probs, axis=1) == labels))
        if acc < 0.99 * base_acc:
            continue
        nll = float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], runner.EPS, 1.0))))
        if nll < best_nll:
            best_nll = nll
            best_c = float(c)
    return float(best_c)


def apply_linear_fc(head: LogisticRegression, features: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clipped = np.clip(features, -c, c)
    logits = linear_logits(head, clipped)
    probs = runner.clip_probs(softmax(logits, axis=1))
    return clipped, logits, probs


def load_or_train_head(x_train: np.ndarray, y_train: np.ndarray, x_cal: np.ndarray, x_test: np.ndarray, status: dict[str, Any]):
    head_path = OUT_ROOT / "head.pkl"
    if head_path.exists():
        with head_path.open("rb") as f:
            return pickle.load(f)
    mark_stage(status, "linear_head", "running")
    scaler = StandardScaler()
    z_train = scaler.fit_transform(x_train).astype(np.float32)
    z_cal = scaler.transform(x_cal).astype(np.float32)
    z_test = scaler.transform(x_test).astype(np.float32)
    head = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1, C=1.0)
    start = time.perf_counter()
    head.fit(z_train, y_train)
    train_seconds = time.perf_counter() - start
    artifact = {
        "scaler": scaler,
        "head": head,
        "z_train": z_train,
        "z_cal": z_cal,
        "z_test": z_test,
        "train_seconds": train_seconds,
    }
    with head_path.open("wb") as f:
        pickle.dump(artifact, f)
    mark_stage(status, "linear_head", "completed", train_seconds=train_seconds)
    return artifact


def fit_predict_method(method_name: str, n_classes: int, z_cal: np.ndarray, z_test: np.ndarray, logits_cal: np.ndarray, logits_test: np.ndarray, probs_cal: np.ndarray, y_cal: np.ndarray, head: LogisticRegression, seed: int):
    start = time.perf_counter()
    if method_name in runner.STANDARD_METHODS:
        return runner.fit_predict_array_method(method_name, n_classes, z_cal, z_test, logits_cal, logits_test, probs_cal, y_cal)
    if method_name == "FC":
        c = choose_linear_fc_clip(z_cal, y_cal, head)
        fit_seconds = time.perf_counter() - start
        pred_start = time.perf_counter()
        _, _, probs = apply_linear_fc(head, z_test, c)
        return probs, fit_seconds, time.perf_counter() - pred_start
    if method_name in {"FC+TS", "FC+ETS"}:
        c = choose_linear_fc_clip(z_cal, y_cal, head)
        _, logits_cal_fc, _ = apply_linear_fc(head, z_cal, c)
        _, logits_test_fc, _ = apply_linear_fc(head, z_test, c)
        if method_name == "FC+TS":
            model = runner.TemperatureScaling(n_classes)
            model.fit(logits_cal_fc, y_cal)
            fit_seconds = time.perf_counter() - start
            pred_start = time.perf_counter()
            probs = model.predict_proba(logits_test_fc)
            return probs, fit_seconds, time.perf_counter() - pred_start
        probs = runner.fit_ets_probs(logits_cal_fc, y_cal, logits_test_fc)
        return probs, time.perf_counter() - start, 0.0
    if method_name == "SAG":
        probs, _ = runner.apply_sag(z_cal, logits_cal, y_cal, z_test, logits_test, seed=seed)
        return probs, time.perf_counter() - start, 0.0
    raise ValueError(method_name)


def build_cv_reference(method_name: str, n_classes: int, z_cal: np.ndarray, logits_cal: np.ndarray, probs_cal: np.ndarray, y_cal: np.ndarray, head: LogisticRegression, seed: int) -> np.ndarray:
    if method_name in runner.STANDARD_METHODS:
        return runner.build_cv_reference_probs_arrays(method_name, n_classes, z_cal, logits_cal, probs_cal, y_cal, seed, cv_kind="stratified")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    ref = np.zeros_like(probs_cal, dtype=np.float64)
    for train_idx, val_idx in splitter.split(z_cal, y_cal):
        if method_name == "FC":
            c = choose_linear_fc_clip(z_cal[train_idx], y_cal[train_idx], head)
            _, _, ref[val_idx] = apply_linear_fc(head, z_cal[val_idx], c)
        elif method_name in {"FC+TS", "FC+ETS"}:
            c = choose_linear_fc_clip(z_cal[train_idx], y_cal[train_idx], head)
            _, logits_train_fc, _ = apply_linear_fc(head, z_cal[train_idx], c)
            _, logits_val_fc, _ = apply_linear_fc(head, z_cal[val_idx], c)
            if method_name == "FC+TS":
                model = runner.TemperatureScaling(n_classes)
                model.fit(logits_train_fc, y_cal[train_idx])
                ref[val_idx] = model.predict_proba(logits_val_fc)
            else:
                ref[val_idx] = runner.fit_ets_probs(logits_train_fc, y_cal[train_idx], logits_val_fc)
        elif method_name == "SAG":
            ref[val_idx] = runner.apply_sag(
                z_cal[train_idx],
                logits_cal[train_idx],
                y_cal[train_idx],
                z_cal[val_idx],
                logits_cal[val_idx],
                seed=seed,
            )[0]
        else:
            raise ValueError(method_name)
    return runner.clip_probs(ref)


def checkpoint_row(rows_path: Path, row: dict[str, Any]) -> None:
    if rows_path.exists():
        rows = pd.read_csv(rows_path)
        rows = rows[rows["method"] != row["method"]]
        rows = pd.concat([rows, pd.DataFrame([row])], ignore_index=True)
    else:
        rows = pd.DataFrame([row])
    rows.to_csv(rows_path, index=False)


def run_experiment(dataset_name: str, backbone_name: str, seed: int, quick: bool, batch_size: int) -> None:
    configure_output(dataset_name, backbone_name)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    status = load_status()
    status["last_invoked_at"] = now_iso()
    status["seed"] = seed
    status["quick"] = quick
    save_status(status)

    splits, labels, meta = prepare_cifar_splits(dataset_name, backbone_name, seed, quick)
    n_classes = int(meta["n_classes"])
    (OUT_ROOT / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    x_train, y_train = extract_split_features("train", splits["train"], labels["train"], backbone_name, batch_size, status)
    x_cal, y_cal = extract_split_features("cal", splits["cal"], labels["cal"], backbone_name, batch_size, status)
    x_test, y_test = extract_split_features("test", splits["test"], labels["test"], backbone_name, batch_size, status)
    artifact = load_or_train_head(x_train, y_train, x_cal, x_test, status)
    z_cal = artifact["z_cal"]
    z_test = artifact["z_test"]
    head = artifact["head"]

    logits_cal = linear_logits(head, z_cal)
    logits_test = linear_logits(head, z_test)
    probs_cal = runner.clip_probs(softmax(logits_cal, axis=1))
    probs_test = runner.clip_probs(softmax(logits_test, axis=1))
    base_predictions = np.argmax(probs_test, axis=1)

    rows_path = OUT_ROOT / f"{dataset_name}_{backbone_name}_detailed.csv"
    probs_dir = OUT_ROOT / "method_probs"
    probs_dir.mkdir(parents=True, exist_ok=True)
    method_status = status.setdefault("methods", {})

    def method_done(name: str) -> bool:
        return method_status.get(name, {}).get("state") == "completed" and (probs_dir / f"{name.replace('+', '_')}.npz").exists()

    def save_method(name: str, family: str, probs: np.ndarray, fit_seconds: float, predict_seconds: float) -> None:
        np.savez_compressed(probs_dir / f"{name.replace('+', '_')}.npz", probs=runner.clip_probs(probs))
        row = {
            "dataset": dataset_name,
            "backbone": f"{backbone_name}_imagenet_frozen",
            "method": name,
            "family": family,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "predict_ms_per_sample": 1000.0 * predict_seconds / max(len(y_test), 1),
            **runner.evaluate_predictions(y_test, probs, base_predictions),
        }
        checkpoint_row(rows_path, row)
        method_status[name] = {"state": "completed", "time": now_iso()}
        save_status(status)

    if not method_done("Base"):
        save_method("Base", "base", probs_test, float(artifact["train_seconds"]), 0.0)

    full_outputs: dict[str, np.ndarray] = {"Base": probs_test}
    for method in METHODS:
        path = probs_dir / f"{method.replace('+', '_')}.npz"
        if method_done(method):
            full_outputs[method] = runner.clip_probs(np.load(path)["probs"])
            continue
        method_status[method] = {"state": "running", "time": now_iso()}
        save_status(status)
        probs, fit_seconds, predict_seconds = fit_predict_method(
            method, n_classes, z_cal, z_test, logits_cal, logits_test, probs_cal, y_cal, head, seed
        )
        full_outputs[method] = probs
        save_method(method, "global" if method in {"TS", "VS", "Dir", "ETS", "IROvA", "MIR"} else "local_baseline", probs, fit_seconds, predict_seconds)

    if not method_done("Pure LAPE"):
        start = time.perf_counter()
        lape = runner.LAPECalibrator(n_classes, k_min_floor=50, k_min_divisor=10, k_max_multiplier=4, k_max_cap=1000)
        lape.fit(z_cal, y_cal, probs_cal)
        fit_seconds = time.perf_counter() - start
        pred_start = time.perf_counter()
        probs = lape.predict(z_test, probs_test)
        save_method("Pure LAPE", "pure_lape", probs, fit_seconds, time.perf_counter() - pred_start)

    for method in CASCADE_METHODS:
        name = f"{method}+LAPE"
        if method_done(name):
            continue
        method_status[name] = {"state": "running", "time": now_iso()}
        save_status(status)
        start = time.perf_counter()
        ref_probs = build_cv_reference(method, n_classes, z_cal, logits_cal, probs_cal, y_cal, head, seed)
        lape = runner.LAPECalibrator(n_classes, k_min_floor=50, k_min_divisor=10, k_max_multiplier=4, k_max_cap=1000)
        lape.fit(z_cal, y_cal, ref_probs)
        fit_seconds = time.perf_counter() - start
        pred_start = time.perf_counter()
        probs = lape.predict(z_test, full_outputs[method])
        save_method(name, "cascade", probs, fit_seconds, time.perf_counter() - pred_start)

    df = pd.read_csv(rows_path).sort_values(["nll", "method"]).reset_index(drop=True)
    df.to_csv(OUT_ROOT / f"{dataset_name}_{backbone_name}_detailed_sorted.csv", index=False)
    summary = {
        **meta,
        "seed": seed,
        "metrics_by_method": {
            str(row["method"]): {
                "nll": float(row["nll"]),
                "accuracy": float(row["accuracy"]),
                "ece_toplabel": float(row["ece_toplabel"]),
                "ace": float(row["ace"]),
            }
            for _, row in df.sort_values("method").iterrows()
        },
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(runner.make_json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    mark_stage(status, "experiment", "completed", summary_path="summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR frozen CNN all-baseline segmented calibration experiment.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument(
        "--backbone",
        choices=["resnet18", "resnet50", "efficientnet_b0", "convnext_tiny", "vit_b_16", "swin_t"],
        default="resnet18",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quick", action="store_true", help="Run a small stratified subset for smoke testing.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        dataset_name=args.dataset,
        backbone_name=args.backbone,
        seed=args.seed,
        quick=args.quick,
        batch_size=args.batch_size,
    )
