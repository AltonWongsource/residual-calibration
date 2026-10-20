"""Target-domain CIFAR-10-C calibration experiment with resumable cached features.

The source classifier is a frozen ImageNet-pretrained ResNet18 followed by a
multinomial linear probe fit only on clean CIFAR-10 source images. For each
pre-registered target condition, labelled target examples are split into a
reference/calibration set and an untouched target test set. Every post-hoc
method receives exactly the same reference examples.
"""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "feature_cache"
RESULTS = ROOT / "results"
ARCHIVE = DATA / "CIFAR-10-C.tar"
C10C_DIR = DATA / "CIFAR-10-C"
import calibration_core as lr

CONDITIONS = [(c, s) for c in ("gaussian_noise", "motion_blur", "snow") for s in (1, 3, 5)]
SPLIT_SEEDS = (42, 43, 44, 45, 46)
CAL_RATIO = 0.20
METHODS = ["TS", "VS", "Dir", "ETS", "LTS"]
DOWNLOAD_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"


class ArrayImages(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        from PIL import Image

        return self.transform(Image.fromarray(self.images[idx])), int(self.labels[idx])


def ensure_cifar10c() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if C10C_DIR.exists() and (C10C_DIR / "labels.npy").exists():
        return
    if not ARCHIVE.exists() or ARCHIVE.stat().st_size < 2_000_000_000:
        print("Downloading official CIFAR-10-C archive (about 2.9 GB)...", flush=True)
        part = ARCHIVE.with_suffix(".tar.part")
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with requests.get(DOWNLOAD_URL, stream=True, timeout=60, headers=headers) as response:
            response.raise_for_status()
            mode = "ab" if offset and response.status_code == 206 else "wb"
            with part.open(mode) as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        handle.write(block)
        part.replace(ARCHIVE)
    print("Extracting CIFAR-10-C...", flush=True)
    with tarfile.open(ARCHIVE, "r") as tar:
        tar.extractall(DATA, filter="data")


def extract_features(images: np.ndarray, labels: np.ndarray, model, transform, name: str) -> tuple[np.ndarray, np.ndarray]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.npz"
    if path.exists():
        saved = np.load(path)
        return saved["x"], saved["y"]
    loader = DataLoader(ArrayImages(images, labels, transform), batch_size=64, shuffle=False, num_workers=0)
    chunks: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    with torch.inference_mode():
        for index, (batch, y) in enumerate(loader, start=1):
            chunks.append(model(batch).numpy().astype(np.float32))
            ys.append(y.numpy())
            if index % 100 == 0:
                print(f"{name}: {index}/{len(loader)} batches", flush=True)
    x = np.concatenate(chunks)
    y = np.concatenate(ys)
    np.savez_compressed(path, x=x, y=y)
    return x, y


def get_backbone():
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    backbone = models.resnet18(weights=weights)
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    return backbone, weights.transforms()


def get_clean_source(model, transform):
    raw = datasets.CIFAR10(root=str(DATA), train=True, download=True)
    images = np.asarray(raw.data)
    labels = np.asarray(raw.targets, dtype=np.int64)
    return extract_features(images, labels, model, transform, "cifar10_clean_train")


def get_target_features(corruption: str, severity: int, model, transform):
    cached = CACHE / f"{corruption}_s{severity}.npz"
    if cached.exists():
        with np.load(cached) as saved:
            return saved["x"], saved["y"]
    if model is None or transform is None:
        raise FileNotFoundError(f"required cached target features are missing: {cached}")
    labels = np.load(C10C_DIR / "labels.npy").astype(np.int64)
    images = np.load(C10C_DIR / f"{corruption}.npy")
    start = (severity - 1) * 10_000
    stop = severity * 10_000
    return extract_features(images[start:stop], labels[start:stop], model, transform, f"{corruption}_s{severity}")


def train_probe(x_source: np.ndarray, y_source: np.ndarray):
    path = CACHE / "clean_source_probe.npz"
    if path.exists():
        obj = np.load(path)
        return obj["mean"], obj["scale"], obj["coef"], obj["intercept"]
    scaler = StandardScaler().fit(x_source)
    probe = LogisticRegression(C=2.0, max_iter=500, solver="lbfgs", n_jobs=1, random_state=2026)
    probe.fit(scaler.transform(x_source), y_source)
    np.savez_compressed(path, mean=scaler.mean_, scale=scaler.scale_, coef=probe.coef_, intercept=probe.intercept_)
    return scaler.mean_, scaler.scale_, probe.coef_, probe.intercept_


def logits(x: np.ndarray, mean: np.ndarray, scale: np.ndarray, coef: np.ndarray, intercept: np.ndarray):
    return ((x - mean) / scale) @ coef.T + intercept


def standardize_reference(x_cal: np.ndarray, x_test: np.ndarray):
    scaler = StandardScaler().fit(x_cal)
    return scaler.transform(x_cal).astype(np.float32), scaler.transform(x_test).astype(np.float32)


def target_split(y: np.ndarray, seed: int):
    split = StratifiedShuffleSplit(n_splits=1, train_size=CAL_RATIO, random_state=seed)
    cal, test = next(split.split(np.zeros(len(y)), y))
    return cal, test


def argmax_diagnostics(y: np.ndarray, anchor: np.ndarray, refined: np.ndarray):
    a = np.argmax(anchor, axis=1)
    r = np.argmax(refined, axis=1)
    changed = a != r
    return {
        "argmax_changed_rate": float(np.mean(changed)),
        "argmax_preserved_rate": float(np.mean(~changed)),
        "wrong_to_correct_rate": float(np.mean((a != y) & (r == y))),
        "correct_to_wrong_rate": float(np.mean((a == y) & (r != y))),
    }


def run_condition(corruption: str, severity: int, seed: int, target_x: np.ndarray, target_y: np.ndarray, probe):
    out = RESULTS / f"{corruption}_s{severity}_seed{seed}.csv"
    if out.exists():
        print(f"Skipping complete {out.stem}", flush=True)
        return
    cal_idx, test_idx = target_split(target_y, seed)
    x_cal_raw, x_test_raw = target_x[cal_idx], target_x[test_idx]
    y_cal, y_test = target_y[cal_idx], target_y[test_idx]
    x_cal, x_test = standardize_reference(x_cal_raw, x_test_raw)
    mean, scale, coef, intercept = probe
    l_cal, l_test = logits(x_cal_raw, mean, scale, coef, intercept), logits(x_test_raw, mean, scale, coef, intercept)
    p_cal, p_test = lr.softmax(l_cal, axis=1), lr.softmax(l_test, axis=1)
    suite, outputs, meta = lr.run_array_calibration_suite(
        suite_name=f"CIFAR-10-C/{corruption}/severity-{severity}",
        y_cal=y_cal, y_test=y_test, x_cal=x_cal, x_test=x_test,
        logits_cal=l_cal, logits_test=l_test, base_probs_cal=p_cal, base_probs_test=p_test,
        seed=seed, standard_methods=METHODS, cascade_methods=METHODS,
        lape_params={"metric": "cosine", "k_min_floor": 20, "k_min_divisor": 8, "k_max_multiplier": 4, "k_max_cap": 250},
        cv_kind="stratified",
    )
    diagnostics = []
    for anchor in ["TS", "VS", "Dir", "ETS", "LTS"]:
        diag = argmax_diagnostics(y_test, outputs[anchor]["probs"], outputs[f"{anchor}+LAPE"]["probs"])
        diag.update({"method": f"{anchor}+LAPE"})
        diagnostics.append(diag)
    diag_df = pd.DataFrame(diagnostics)
    suite = suite.merge(diag_df, on="method", how="left")
    suite.insert(0, "split_seed", seed)
    suite.insert(1, "corruption", corruption)
    suite.insert(2, "severity", severity)
    RESULTS.mkdir(parents=True, exist_ok=True)
    suite.to_csv(out, index=False)
    print(f"Completed {out.stem}", flush=True)


def summarize():
    files = sorted(RESULTS.glob("*_seed*.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    metrics = ["nll", "ece_toplabel", "ace", "accuracy", "argmax_changed_rate", "argmax_preserved_rate", "wrong_to_correct_rate", "correct_to_wrong_rate"]
    present = [m for m in metrics if m in df]
    grouped = df.groupby(["corruption", "severity", "method"], as_index=False)[present].agg(["mean", "std"])
    grouped.columns = ["_".join(c).strip("_") for c in grouped.columns]
    grouped.to_csv(RESULTS / "summary_by_condition.csv", index=False)
    overall = df.groupby("method", as_index=False)[present].agg(["mean", "std"])
    overall.columns = ["_".join(c).strip("_") for c in overall.columns]
    overall.to_csv(RESULTS / "summary_overall.csv", index=False)


def main():
    global DATA, CACHE, RESULTS, ARCHIVE, C10C_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--corruption", choices=("gaussian_noise", "motion_blur", "snow", "all"), default="gaussian_noise", help="The paper's Table 9 uses Gaussian noise; other corruptions are optional extensions.")
    parser.add_argument("--severity", type=int, choices=(1, 3, 5))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SPLIT_SEEDS))
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--cache-only", action="store_true", help="Use existing real-image feature/probe caches without downloading or encoding images.")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    DATA, CACHE, RESULTS = args.data_dir, args.cache_dir, args.output_dir
    ARCHIVE, C10C_DIR = DATA / "CIFAR-10-C.tar", DATA / "CIFAR-10-C"
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize()
        return
    selected_conditions = [
        (corruption, severity)
        for corruption, severity in CONDITIONS
        if (args.corruption == "all" or corruption == args.corruption)
        and (args.severity is None or severity == args.severity)
    ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "manifest.json").write_text(json.dumps({
        "protocol": "target-domain labelled reference calibration on CIFAR-10-C",
        "conditions": selected_conditions, "split_seeds": args.seeds, "calibration_ratio": CAL_RATIO,
        "backbone": "frozen ImageNet-pretrained ResNet18", "source_head": "multinomial linear probe on clean CIFAR-10 source",
        "cache_only": args.cache_only,
        "methods": ["Base", *METHODS, "Pure LAPE", *[f"{m}+LAPE" for m in METHODS]],
    }, indent=2), encoding="utf-8")
    if args.cache_only:
        probe_path = CACHE / "clean_source_probe.npz"
        if not probe_path.exists():
            raise FileNotFoundError(f"required real-image source probe cache is missing: {probe_path}")
        with np.load(probe_path) as saved:
            probe = tuple(saved[key] for key in ("mean", "scale", "coef", "intercept"))
        model, transform = None, None
    else:
        ensure_cifar10c()
        model, transform = get_backbone()
        source_x, source_y = get_clean_source(model, transform)
        probe = train_probe(source_x, source_y)
    for corruption, severity in selected_conditions:
        tx, ty = get_target_features(corruption, severity, model, transform)
        for seed in args.seeds:
            run_condition(corruption, severity, seed, tx, ty, probe)
            if not args.skip_summary:
                summarize()


if __name__ == "__main__":
    main()
