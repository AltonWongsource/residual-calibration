from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from calibration_core import (
    LAPECalibrator,
    VectorScaling,
    clip_probs,
    evaluate_predictions,
    get_deep_features,
)


WORK_ROOT = Path(__file__).resolve().parent
DATA_ROOT = WORK_ROOT / "data" / "ham10000"
OUTPUT_ROOT = WORK_ROOT / "outputs"
MANIFEST_ROOT = WORK_ROOT / "manifests"
STATUS_PATH = MANIFEST_ROOT / "pipeline_status.json"
PAPER_CFAF_METHOD = "CFAF-LAPE (concat/std/cosine, alpha<=0.5)"
PAPER_PROTOCOL = "paper_fixed_v1"


TASKS: dict[str, dict[str, Any]] = {
    "ham10000_malignant_benign": {
        "label": "HAM10000 Malignant vs Benign",
        "target_kind": "diagnosis_1",
        "positive_value": "Malignant",
        "valid_values": {"Benign", "Malignant"},
        "train_ratio": 0.60,
        "cal_ratio": 0.20,
        "test_ratio": 0.20,
        "head_epochs": 50,
        "batch_size": 96,
    },
    "ham10000_melanoma_nevus": {
        "label": "HAM10000 Melanoma vs Nevus",
        "target_kind": "diagnosis_3",
        "positive_value": "Melanoma, NOS",
        "valid_values": {"Melanoma, NOS", "Nevus"},
        "train_ratio": 0.60,
        "cal_ratio": 0.20,
        "test_ratio": 0.20,
        "head_epochs": 50,
        "batch_size": 96,
    },
}


FAIR_METADATA_COLUMNS = [
    "age_approx",
    "sex",
    "anatom_site_1",
    "anatom_site_2",
    "anatom_site_3",
    "anatom_site_general",
    "anatom_site_special",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ISICImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: Any) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return int(len(self.frame))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[int(idx)]
        path = DATA_ROOT / f"{row['isic_id']}.jpg"
        image = Image.open(path).convert("RGB")
        return self.transform(image), int(row["label"])


class FrozenResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = backbone.fc.in_features
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


def make_transform() -> transforms.Compose:
    weights = ResNet18_Weights.IMAGENET1K_V1
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
        ]
    )


def make_loader(frame: pd.DataFrame, batch_size: int) -> DataLoader:
    return DataLoader(
        ISICImageDataset(frame, make_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def extract_features(encoder: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    feats_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = encoder(images)
            feats_all.append(feats.cpu().numpy().astype(np.float32))
            y_all.append(labels.numpy().astype(int))
    return np.concatenate(feats_all, axis=0), np.concatenate(y_all, axis=0)


def train_linear_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    n_classes: int,
    device: torch.device,
    epochs: int,
    batch_size: int = 256,
) -> tuple[nn.Linear, float]:
    head = nn.Linear(x_train.shape[1], n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_cal = float("inf")
    start = time.perf_counter()
    n_train = len(x_train)
    for _ in range(epochs):
        head.train()
        perm = np.random.permutation(n_train)
        for offset in range(0, n_train, batch_size):
            idx = perm[offset : offset + batch_size]
            xb = torch.from_numpy(x_train[idx]).to(device)
            yb = torch.from_numpy(y_train[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            cal_logits = head(torch.from_numpy(x_cal).to(device))
            cal_loss = criterion(cal_logits, torch.from_numpy(y_cal).to(device)).item()
        if cal_loss < best_cal:
            best_cal = cal_loss
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    return head, time.perf_counter() - start


def collect_head_outputs(head: nn.Linear, x: np.ndarray, y: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(x).to(device))
        probs = torch.softmax(logits, dim=1)
    return {
        "logits": logits.cpu().numpy().astype(np.float64),
        "probs": probs.cpu().numpy().astype(np.float64),
        "features": x.astype(np.float32),
        "labels": y.astype(int),
    }


def build_metadata_preprocessor(train_df: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in FAIR_METADATA_COLUMNS if c in train_df.columns and train_df[c].dtype == object]
    num_cols = [c for c in FAIR_METADATA_COLUMNS if c in train_df.columns and c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def fit_vs_cv(logits_cal: np.ndarray, y_cal: np.ndarray, n_classes: int, seed: int) -> np.ndarray:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    probs = np.zeros((len(y_cal), n_classes), dtype=np.float64)
    for tr, va in skf.split(logits_cal, y_cal):
        vs = VectorScaling(n_classes)
        vs.fit(logits_cal[tr], y_cal[tr])
        probs[va] = vs.predict_proba(logits_cal[va])
    return clip_probs(probs)


def fit_vs_full(logits_cal: np.ndarray, y_cal: np.ndarray, logits_eval: np.ndarray, n_classes: int) -> np.ndarray:
    vs = VectorScaling(n_classes)
    vs.fit(logits_cal, y_cal)
    return clip_probs(vs.predict_proba(logits_eval))


def optimize_two_view_fusion(cal_probs_list: list[np.ndarray], y_cal: np.ndarray) -> np.ndarray:
    def objective(raw_w: np.ndarray) -> float:
        w = np.exp(raw_w)
        w = w / np.sum(w)
        fused = clip_probs(sum(alpha * p for alpha, p in zip(w, cal_probs_list)))
        return log_loss(y_cal, fused, labels=list(range(fused.shape[1])))

    init = np.zeros(len(cal_probs_list), dtype=np.float64)
    result = minimize(objective, init, method="L-BFGS-B")
    w = np.exp(result.x)
    return w / np.sum(w)


def ensure_status() -> dict[str, Any]:
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    status = {"tasks": {}}
    STATUS_PATH.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return status


def update_status(task_name: str, payload: dict[str, Any]) -> None:
    status = ensure_status()
    status["tasks"][task_name] = payload
    STATUS_PATH.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")


def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(DATA_ROOT / "metadata.csv")
    return df


def prepare_task_frame(task_name: str) -> pd.DataFrame:
    cfg = TASKS[task_name]
    df = load_metadata()
    df = df[df[cfg["target_kind"]].isin(cfg["valid_values"])].copy()
    df["label"] = (df[cfg["target_kind"]] == cfg["positive_value"]).astype(int)
    return df


def stratified_group_split(frame: pd.DataFrame, seed: int, train_ratio: float, cal_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lesions = (
        frame.groupby("lesion_id", as_index=False)
        .agg(label=("label", "first"))
        .reset_index(drop=True)
    )
    lesion_train, lesion_temp = train_test_split(
        lesions,
        test_size=1.0 - train_ratio,
        random_state=seed,
        stratify=lesions["label"],
    )
    relative_cal = cal_ratio / (1.0 - train_ratio)
    lesion_cal, lesion_test = train_test_split(
        lesion_temp,
        test_size=1.0 - relative_cal,
        random_state=seed,
        stratify=lesion_temp["label"],
    )
    train_ids = set(lesion_train["lesion_id"])
    cal_ids = set(lesion_cal["lesion_id"])
    test_ids = set(lesion_test["lesion_id"])
    return (
        frame[frame["lesion_id"].isin(train_ids)].copy(),
        frame[frame["lesion_id"].isin(cal_ids)].copy(),
        frame[frame["lesion_id"].isin(test_ids)].copy(),
    )


def task_outdir(task_name: str) -> Path:
    out = OUTPUT_ROOT / task_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_task(task_name: str, seed: int, force: bool = False) -> dict[str, Any]:
    cfg = TASKS[task_name]
    outdir = task_outdir(task_name)
    summary_path = outdir / f"{task_name}_summary.json"
    if summary_path.exists() and not force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("protocol") != PAPER_PROTOCOL:
            raise RuntimeError("Existing summary uses a different protocol; rerun explicitly with --force.")
        return summary

    update_status(task_name, {"status": "running", "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_df = prepare_task_frame(task_name)
    train_df, cal_df, test_df = stratified_group_split(full_df, seed, cfg["train_ratio"], cfg["cal_ratio"])

    # Image view
    encoder = FrozenResNet18().to(device)
    batch_size = int(cfg["batch_size"])
    img_train_raw, y_train_img = extract_features(encoder, make_loader(train_df, batch_size), device)
    img_cal_raw, y_cal = extract_features(encoder, make_loader(cal_df, batch_size), device)
    img_test_raw, y_test = extract_features(encoder, make_loader(test_df, batch_size), device)
    img_scaler = StandardScaler()
    img_train = img_scaler.fit_transform(img_train_raw).astype(np.float32)
    img_cal = img_scaler.transform(img_cal_raw).astype(np.float32)
    img_test = img_scaler.transform(img_test_raw).astype(np.float32)
    img_head, img_train_seconds = train_linear_head(
        img_train, y_train_img, img_cal, y_cal, 2, device, int(cfg["head_epochs"])
    )
    img_cal_out = collect_head_outputs(img_head, img_cal, y_cal, device)
    img_test_out = collect_head_outputs(img_head, img_test, y_test, device)

    # Metadata view
    metadata_pre = build_metadata_preprocessor(train_df[FAIR_METADATA_COLUMNS])
    meta_train_x = metadata_pre.fit_transform(train_df[FAIR_METADATA_COLUMNS]).astype(np.float32)
    meta_cal_x = metadata_pre.transform(cal_df[FAIR_METADATA_COLUMNS]).astype(np.float32)
    meta_test_x = metadata_pre.transform(test_df[FAIR_METADATA_COLUMNS]).astype(np.float32)
    meta_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed, early_stopping=True)
    meta_start = time.perf_counter()
    meta_model.fit(meta_train_x, train_df["label"].to_numpy(dtype=int))
    meta_train_seconds = time.perf_counter() - meta_start
    meta_cal_probs = clip_probs(meta_model.predict_proba(meta_cal_x))
    meta_test_probs = clip_probs(meta_model.predict_proba(meta_test_x))
    meta_cal_logits = np.log(meta_cal_probs)
    meta_test_logits = np.log(meta_test_probs)
    meta_feat_scaler = StandardScaler()
    meta_train_feats = meta_feat_scaler.fit_transform(get_deep_features(meta_model, meta_train_x)).astype(np.float32)
    meta_cal_feats = meta_feat_scaler.transform(get_deep_features(meta_model, meta_cal_x)).astype(np.float32)
    meta_test_feats = meta_feat_scaler.transform(get_deep_features(meta_model, meta_test_x)).astype(np.float32)

    # Per-view calibrated anchors
    img_cv = fit_vs_cv(img_cal_out["logits"], y_cal, 2, seed)
    meta_cv = fit_vs_cv(meta_cal_logits, y_cal, 2, seed)
    img_test_vs = fit_vs_full(img_cal_out["logits"], y_cal, img_test_out["logits"], 2)
    meta_test_vs = fit_vs_full(meta_cal_logits, y_cal, meta_test_logits, 2)

    weights = optimize_two_view_fusion([img_cv, meta_cv], y_cal)
    fused_test = clip_probs(weights[0] * img_test_vs + weights[1] * meta_test_vs)

    joint_train = np.hstack([img_cal_out["features"], meta_cal_feats]).astype(np.float32)
    joint_test = np.hstack([img_test_out["features"], meta_test_feats]).astype(np.float32)

    # Train the joint concat anchor used by the paper-reported CFAF row.
    joint_train_supervised = np.hstack([img_train, meta_train_feats]).astype(np.float32)
    joint_hold = joint_train[: min(len(joint_train), max(128, len(joint_train) // 5))]
    joint_hold_y = y_cal[: len(joint_hold)]
    concat_head, _concat_train_seconds = train_linear_head(
        joint_train_supervised,
        y_train_img,
        joint_hold,
        joint_hold_y,
        2,
        device,
        epochs=max(20, int(cfg["head_epochs"]) // 2),
    )
    concat_logits_cal = concat_head(torch.from_numpy(joint_train).to(device)).detach().cpu().numpy().astype(np.float64)
    concat_logits_test = concat_head(torch.from_numpy(joint_test).to(device)).detach().cpu().numpy().astype(np.float64)
    concat_cv = fit_vs_cv(concat_logits_cal, y_cal, 2, seed)
    concat_test_probs = fit_vs_full(concat_logits_cal, y_cal, concat_logits_test, 2)

    rows = []
    base_predictions = np.argmax(img_test_vs, axis=1)
    method_specs = [
        ("Image-only (ResNet18+VS)", "image_only", img_test_vs, img_train_seconds, 0.0),
        ("Metadata-only (MLP+VS)", "meta_only", meta_test_vs, meta_train_seconds, 0.0),
        ("Fused (VS+Weight)", "fusion", fused_test, 0.0, 0.0),
    ]
    for method, family, probs, fit_seconds, pred_seconds in method_specs:
        rows.append(
            {
                "task": task_name,
                "method": method,
                "family": family,
                "fit_seconds": float(fit_seconds),
                "predict_seconds": float(pred_seconds),
                "predict_ms_per_sample": 1000.0 * float(pred_seconds) / max(len(y_test), 1),
                **evaluate_predictions(y_test, probs, base_predictions),
            }
        )

    lape = LAPECalibrator(
        2,
        metric="cosine",
        candidate_alpha=np.linspace(0.0, 0.5, 11),
        k_min_floor=10,
        k_min_divisor=8,
        k_max_cap=min(300, max(len(y_cal) // 3, 30)),
        m_anchor_floor=3,
    )
    fit_start = time.perf_counter()
    lape.fit(joint_train, y_cal, concat_cv)
    lape_fit_seconds = time.perf_counter() - fit_start
    pred_start = time.perf_counter()
    paper_probs = lape.predict(joint_test, concat_test_probs)
    lape_pred_seconds = time.perf_counter() - pred_start
    paper_cfaf = {
        "task": task_name,
        "method": PAPER_CFAF_METHOD,
        "family": "cfaf_lape",
        "fit_seconds": float(lape_fit_seconds),
        "predict_seconds": float(lape_pred_seconds),
        "predict_ms_per_sample": 1000.0 * float(lape_pred_seconds) / max(len(y_test), 1),
        "rep_key": "std",
        "anchor_key": "concat",
        "metric_name": "cosine",
        **evaluate_predictions(y_test, paper_probs, base_predictions),
    }
    rows.append(paper_cfaf)
    paper_late_fusion = next(row for row in rows if row["family"] == "fusion")
    df = pd.DataFrame(rows)

    summary = {
        "task": task_name,
        "protocol": PAPER_PROTOCOL,
        "label": cfg["label"],
        "seed": seed,
        "image_view": "ResNet18 frozen features + linear head + VS",
        "metadata_view": "MLP(hidden=(64,32)) on clinical metadata + VS",
        "joint_manifold": "Concatenated standardized image+metadata features",
        "n_classes": 2,
        "n_train": int(len(train_df)),
        "n_cal": int(len(cal_df)),
        "n_test": int(len(test_df)),
        "n_train_lesions": int(train_df["lesion_id"].nunique()),
        "n_cal_lesions": int(cal_df["lesion_id"].nunique()),
        "n_test_lesions": int(test_df["lesion_id"].nunique()),
        "positive_rate_train": float(train_df["label"].mean()),
        "positive_rate_cal": float(cal_df["label"].mean()),
        "positive_rate_test": float(test_df["label"].mean()),
        "fusion_weights": [float(x) for x in weights],
        "paper_late_fusion": paper_late_fusion,
        "paper_cfaf": paper_cfaf,
        "notes": [
            "Images were split at the lesion level to avoid leakage across multiple photographs of the same lesion.",
            "The metadata view uses only age, sex, and anatomical-location fields available before diagnosis.",
            "Each view is calibrated with VS; weighted late fusion is optimized on the calibration split.",
            "CFAF-LAPE uses concat/std/cosine with alpha <= 0.5 for both tasks.",
        ],
    }
    df.to_csv(outdir / f"{task_name}_detailed.csv", index=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    update_status(
        task_name,
        {
            "status": "completed",
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary_path": str(summary_path),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task", choices=sorted(TASKS), default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_status()

    tasks = [args.task] if args.task else list(TASKS)
    summaries = [run_task(task_name, args.seed, force=args.force) for task_name in tasks]
    pd.DataFrame(
        [
            {
                "task": s["task"],
                "label": s["label"],
                "late_fusion_method": s["paper_late_fusion"]["method"],
                "late_fusion_nll": s["paper_late_fusion"]["nll"],
                "late_fusion_accuracy": s["paper_late_fusion"]["accuracy"],
                "late_fusion_ece": s["paper_late_fusion"]["ece_toplabel"],
                "late_fusion_ace": s["paper_late_fusion"]["ace"],
                "cfaf_method": s["paper_cfaf"]["method"],
                "cfaf_nll": s["paper_cfaf"]["nll"],
                "cfaf_accuracy": s["paper_cfaf"]["accuracy"],
                "cfaf_ece": s["paper_cfaf"]["ece_toplabel"],
                "cfaf_ace": s["paper_cfaf"]["ace"],
            }
            for s in summaries
        ]
    ).to_csv(OUTPUT_ROOT / "cfaf_ham10000_multimodal_summary.csv", index=False)


if __name__ == "__main__":
    main()
