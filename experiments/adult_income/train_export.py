"""Real OpenML Adult Income MLP export; no synthetic-data fallback."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def hidden_features(model: MLPClassifier, x: np.ndarray) -> np.ndarray:
    h = np.asarray(x, dtype=np.float32)
    for weights, bias in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        h = np.maximum(0.0, h @ weights + bias)
    return h


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the real OpenML Adult Income MLP and export model outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs" / "adult_model_outputs.npz")
    args = parser.parse_args()
    data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    x_raw = data.data
    y = pd.Series(data.target).astype(str).str.contains(">50K", regex=False).astype(int).to_numpy()
    if x_raw.shape != (48_842, 14) or len(np.unique(y)) != 2:
        raise ValueError(f"Unexpected OpenML Adult dataset shape/classes: {x_raw.shape}")

    x_train, x_other, y_train, y_other = train_test_split(
        x_raw, y, test_size=0.5, random_state=args.seed, stratify=y
    )
    x_cal, x_test, y_cal, y_test = train_test_split(
        x_other, y_other, test_size=0.5, random_state=args.seed, stratify=y_other
    )
    categorical = list(x_train.select_dtypes(include=["category", "object"]).columns)
    numeric = list(x_train.select_dtypes(include=["int64", "float64"]).columns)
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    train = preprocessor.fit_transform(x_train).astype(np.float32)
    cal = preprocessor.transform(x_cal).astype(np.float32)
    test = preprocessor.transform(x_test).astype(np.float32)
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=args.seed)
    model.fit(train, y_train)
    h_cal, h_test = hidden_features(model, cal), hidden_features(model, test)
    h_scaler = StandardScaler().fit(h_cal)
    h_cal, h_test = h_scaler.transform(h_cal), h_scaler.transform(h_test)
    logits_cal = np.log(np.clip(model.predict_proba(cal), 1e-12, 1.0))
    logits_test = np.log(np.clip(model.predict_proba(test), 1e-12, 1.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, features_cal=h_cal.astype(np.float32), features_test=h_test.astype(np.float32),
        logits_cal=logits_cal, logits_test=logits_test,
        labels_cal=y_cal.astype(np.int64), labels_test=y_test.astype(np.int64),
    )
    print(f"Saved {len(y_cal)} calibration and {len(y_test)} test examples to {args.output}")


if __name__ == "__main__":
    main()
