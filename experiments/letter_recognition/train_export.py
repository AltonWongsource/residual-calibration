"""Export the real UCI Letter Recognition MLP protocol for post-hoc calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


def hidden_features(model: MLPClassifier, x: np.ndarray) -> np.ndarray:
    h = np.asarray(x, dtype=np.float32)
    for weights, bias in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        h = np.maximum(0.0, h @ weights + bias)
    return h


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on real UCI Letter Recognition data and export calibration/test model outputs.")
    parser.add_argument("--data", type=Path, required=True, help="Official letter-recognition.data CSV (20,000 rows, no header)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs" / "letter_model_outputs.npz")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Real UCI data file is required: {args.data}. No synthetic fallback is used.")

    frame = pd.read_csv(args.data, header=None)
    if frame.shape != (20_000, 17):
        raise ValueError(f"Expected the official 20,000-row, 17-column Letter Recognition file; got {frame.shape}")
    labels = LabelEncoder().fit_transform(frame.iloc[:, 0].astype(str))
    if len(np.unique(labels)) != 26:
        raise ValueError("Expected all 26 letter classes")
    features = frame.iloc[:, 1:].to_numpy(dtype=np.float32)

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=args.seed, stratify=labels
    )
    x_train, x_cal, y_train, y_cal = train_test_split(
        x_train_full, y_train_full, test_size=0.5, random_state=args.seed, stratify=y_train_full
    )
    scaler = StandardScaler().fit(x_train)
    x_train, x_cal, x_test = (scaler.transform(x) for x in (x_train, x_cal, x_test))
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=args.seed)
    model.fit(x_train, y_train)
    h_cal, h_test = hidden_features(model, x_cal), hidden_features(model, x_test)
    feature_scaler = StandardScaler().fit(h_cal)
    h_cal, h_test = feature_scaler.transform(h_cal), feature_scaler.transform(h_test)
    # The original tabular runner uses log(predict_proba) as anchor scores.
    logits_cal = np.log(np.clip(model.predict_proba(x_cal), 1e-12, 1.0))
    logits_test = np.log(np.clip(model.predict_proba(x_test), 1e-12, 1.0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features_cal=h_cal.astype(np.float32), features_test=h_test.astype(np.float32),
        logits_cal=logits_cal.astype(np.float64), logits_test=logits_test.astype(np.float64),
        labels_cal=y_cal.astype(np.int64), labels_test=y_test.astype(np.int64),
    )
    print(f"Saved {len(y_cal)} calibration and {len(y_test)} test examples to {args.output}")


if __name__ == "__main__":
    main()
