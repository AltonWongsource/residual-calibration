"""Real OpenML UCI Multiple Features: two-view VS fusion and CFAF-LAPE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import fetch_openml
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lape_core import LAPECalibrator, VectorScaling, clip_probs  # noqa: E402


def hidden(model: MLPClassifier, x: np.ndarray) -> np.ndarray:
    h = np.asarray(x, dtype=np.float32)
    for weight, bias in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        h = np.maximum(0.0, h @ weight + bias)
    return h


def train_view(x_train: np.ndarray, x_cal: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    scaler = StandardScaler().fit(x_train)
    x_train, x_cal, x_test = (scaler.transform(x) for x in (x_train, x_cal, x_test))
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed).fit(x_train, y_train)
    p_cal, p_test = model.predict_proba(x_cal), model.predict_proba(x_test)
    return {
        "logits_cal": np.log(clip_probs(p_cal)), "logits_test": np.log(clip_probs(p_test)),
        "features_cal": hidden(model, x_cal), "features_test": hidden(model, x_test),
    }


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    prediction = np.argmax(p, axis=1)
    confidence = np.max(p, axis=1)
    correct = (prediction == y).astype(float)
    edges = np.linspace(0.0, 1.0, 16)
    ece = 0.0
    for index in range(15):
        mask = (confidence >= edges[index]) & (confidence <= edges[index + 1] if index == 14 else confidence < edges[index + 1])
        if np.any(mask):
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return {"nll": float(log_loss(y, p, labels=[0, 1])), "ece": ece, "accuracy": float(np.mean(correct))}


def main() -> None:
    parser = argparse.ArgumentParser(description="CFAF-LAPE on real UCI Multiple Features (digits 0–4 vs 5–9)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs" / "metrics.json")
    args = parser.parse_args()
    data_a = fetch_openml(name="mfeat-fourier", version=1, parser="auto")
    data_b = fetch_openml(name="mfeat-zernike", version=1, parser="auto")
    raw_a = np.asarray(data_a.target).astype(int)
    raw_b = np.asarray(data_b.target).astype(int)
    if raw_a.shape != (2000,) or not np.array_equal(raw_a, raw_b):
        raise ValueError("OpenML views have unexpected row count or label alignment; aborting")
    y = (raw_a >= 5).astype(int)
    # Preserve OpenML's float64 input through the train-split scaler, as in the paper runner.
    x_a = data_a.data.to_numpy()
    x_b = data_b.data.to_numpy()
    indices = np.arange(len(y))
    train_idx, other_idx = train_test_split(indices, test_size=0.5, random_state=args.seed, stratify=y)
    cal_idx, test_idx = train_test_split(other_idx, test_size=0.5, random_state=args.seed, stratify=y[other_idx])
    y_cal, y_test = y[cal_idx], y[test_idx]
    a = train_view(x_a[train_idx], x_a[cal_idx], x_a[test_idx], y[train_idx], args.seed)
    b = train_view(x_b[train_idx], x_b[cal_idx], x_b[test_idx], y[train_idx], args.seed)

    cv = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof_a, oof_b = np.zeros((len(cal_idx), 2)), np.zeros((len(cal_idx), 2))
    for fit_idx, val_idx in cv.split(cal_idx):
        for view, oof in ((a, oof_a), (b, oof_b)):
            anchor = VectorScaling(2)
            anchor.fit(view["logits_cal"][fit_idx], y_cal[fit_idx])
            oof[val_idx] = anchor.predict_proba(view["logits_cal"][val_idx])
    test_probs = []
    for view in (a, b):
        anchor = VectorScaling(2)
        anchor.fit(view["logits_cal"], y_cal)
        test_probs.append(anchor.predict_proba(view["logits_test"]))
    p_a, p_b = test_probs

    # Select the single-view comparison and fit fusion weight on calibration OOF scores only.
    cv_nll = [score(y_cal, oof_a)["nll"], score(y_cal, oof_b)["nll"]]
    selected_view = int(np.argmin(cv_nll))
    def fusion_loss(value: np.ndarray) -> float:
        return score(y_cal, clip_probs(value[0] * oof_a + (1 - value[0]) * oof_b))["nll"]
    weight = float(minimize(fusion_loss, np.array([0.5]), bounds=[(0.0, 1.0)], method="L-BFGS-B").x[0])
    fused_cal = clip_probs(weight * oof_a + (1 - weight) * oof_b)
    fused_test = clip_probs(weight * p_a + (1 - weight) * p_b)
    joint_cal = np.hstack([a["features_cal"], b["features_cal"]])
    joint_test = np.hstack([a["features_test"], b["features_test"]])
    joint_scaler = StandardScaler().fit(joint_cal)
    lape = LAPECalibrator(2, k_min_floor=5, k_min_divisor=8, k_max_multiplier=4, k_max_cap=200, m_anchor_floor=3)
    lape.fit(joint_scaler.transform(joint_cal), y_cal, fused_cal)
    refined = lape.predict(joint_scaler.transform(joint_test), fused_test)
    result = {
        "source": "OpenML mfeat-fourier + mfeat-zernike, real 2,000-row UCI Multiple Features",
        "seed": args.seed, "split_sizes": {"train": len(train_idx), "calibration": len(cal_idx), "test": len(test_idx)},
        "selection": "five-fold calibration OOF NLL; held-out test labels used for final metrics only",
        "selected_single_view": ["fourier", "zernike"][selected_view], "fusion_weight_fourier": weight,
        "metrics": {"view_fourier": score(y_test, p_a), "view_zernike": score(y_test, p_b),
                    "selected_single_view": score(y_test, test_probs[selected_view]),
                    "late_fusion": score(y_test, fused_test), "cfaf_lape": score(y_test, refined)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
