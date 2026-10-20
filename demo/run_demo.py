"""Try the original anchor and LAPE functions on a real dataset or an NPZ bundle."""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lape_core import LAPECalibrator, TemperatureScaling, VectorScaling  # noqa: E402


def built_in_data(seed: int) -> dict[str, np.ndarray]:
    data = load_breast_cancer()
    x_train, x_other, y_train, y_other = train_test_split(data.data, data.target, test_size=0.60, stratify=data.target, random_state=seed)
    x_cal, x_test, y_cal, y_test = train_test_split(x_other, y_other, test_size=0.50, stratify=y_other, random_state=seed + 1)
    scaler = StandardScaler().fit(x_train)
    train, cal, test = (scaler.transform(x) for x in (x_train, x_cal, x_test))
    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=seed).fit(train, y_train)
    def features(x: np.ndarray) -> np.ndarray:
        h = x
        for weights, bias in zip(model.coefs_[:-1], model.intercepts_[:-1]):
            h = np.maximum(0.0, h @ weights + bias)
        return h
    def logits(x: np.ndarray) -> np.ndarray:
        score = (features(x) @ model.coefs_[-1] + model.intercepts_[-1]).ravel()
        return np.column_stack([np.zeros_like(score), score])
    return {"features_cal": features(cal), "features_test": features(test), "logits_cal": logits(cal), "logits_test": logits(test), "labels_cal": y_cal, "labels_test": y_test}


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        aliases = {"features_cal": ("features_cal", "h_cal"), "features_test": ("features_test", "h_test"), "logits_cal": ("logits_cal", "l_cal"), "logits_test": ("logits_test", "l_test"), "labels_cal": ("labels_cal", "y_cal"), "labels_test": ("labels_test", "y_test")}
        bundle = {key: np.asarray(data[next(name for name in names if name in data)]) for key, names in aliases.items()}
    n_cal, n_test = len(bundle["labels_cal"]), len(bundle["labels_test"])
    if bundle["features_cal"].shape[0] != n_cal or bundle["logits_cal"].shape[0] != n_cal or bundle["features_test"].shape[0] != n_test or bundle["logits_test"].shape[0] != n_test:
        raise ValueError("Feature/logit rows must match corresponding label counts")
    if bundle["features_cal"].shape[1] != bundle["features_test"].shape[1] or bundle["logits_cal"].shape[1] != bundle["logits_test"].shape[1]:
        raise ValueError("Calibration and test feature/logit dimensions must match")
    return bundle


def load_split_bundles(calibration: Path, test: Path) -> dict[str, np.ndarray]:
    """Load the two-file exports produced by the selected experiment scripts."""
    bundle: dict[str, np.ndarray] = {}
    for suffix, path in (("cal", calibration), ("test", test)):
        with np.load(path, allow_pickle=False) as data:
            for name in ("features", "logits", "labels"):
                bundle[f"{name}_{suffix}"] = np.asarray(data[name])
    if any(bundle[f"features_{part}"].shape[0] != len(bundle[f"labels_{part}"]) or
           bundle[f"logits_{part}"].shape[0] != len(bundle[f"labels_{part}"])
           for part in ("cal", "test")):
        raise ValueError("Feature/logit rows must match corresponding label counts")
    if bundle["features_cal"].shape[1] != bundle["features_test"].shape[1] or bundle["logits_cal"].shape[1] != bundle["logits_test"].shape[1]:
        raise ValueError("Calibration and test feature/logit dimensions must match")
    return bundle


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"nll": float(log_loss(y, p, labels=list(range(p.shape[1])))), "accuracy": float(accuracy_score(y, p.argmax(axis=1)))}


def main() -> None:
    parser = argparse.ArgumentParser(description="LAPE anchor/refinement demonstration")
    parser.add_argument("--npz", type=Path, help="Optional real-model feature/logit bundle; see README")
    parser.add_argument("--calibration", type=Path, help="Calibration NPZ exported by an experiment track")
    parser.add_argument("--test", type=Path, help="Test NPZ exported by the same experiment track")
    parser.add_argument("--anchor", choices=("TS", "VS"), default="VS")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "demo" / "output")
    args = parser.parse_args()
    if args.npz and (args.calibration or args.test):
        parser.error("Choose either --npz or the --calibration/--test pair")
    if bool(args.calibration) != bool(args.test):
        parser.error("--calibration and --test must be provided together")
    bundle = (load_bundle(args.npz) if args.npz else
              load_split_bundles(args.calibration, args.test) if args.calibration else
              built_in_data(args.seed))
    y_cal = bundle["labels_cal"].astype(int)
    y_test = bundle["labels_test"].astype(int)
    logits_cal, logits_test = bundle["logits_cal"], bundle["logits_test"]
    n_classes = logits_cal.shape[1]
    anchor_class = TemperatureScaling if args.anchor == "TS" else VectorScaling
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(y_cal), n_classes))
    for train_idx, val_idx in cv.split(logits_cal, y_cal):
        fold_anchor = anchor_class(n_classes)
        fold_anchor.fit(logits_cal[train_idx], y_cal[train_idx])
        oof[val_idx] = fold_anchor.predict_proba(logits_cal[val_idx])
    anchor = anchor_class(n_classes)
    anchor.fit(logits_cal, y_cal)
    base_probs = softmax(logits_test, axis=1)
    anchor_probs = anchor.predict_proba(logits_test)
    lape = LAPECalibrator(n_classes)
    lape.fit(bundle["features_cal"], y_cal, oof)
    refined_probs = lape.predict(bundle["features_test"], anchor_probs)
    scores = {"base": evaluate(y_test, base_probs), args.anchor: evaluate(y_test, anchor_probs), args.anchor + "+LAPE": evaluate(y_test, refined_probs)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), constrained_layout=True)
    names = list(scores)
    axes[0].bar(names, [scores[n]["nll"] for n in names], color=["#64748b", "#0f766e", "#f59e0b"])
    axes[0].set_title("NLL (lower is better)")
    axes[1].bar(names, [scores[n]["accuracy"] for n in names], color=["#64748b", "#0f766e", "#f59e0b"])
    axes[1].set_title("Accuracy (higher is better)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.savefig(args.output / "comparison.png", dpi=180)
    plt.close(fig)
    rows = "".join(f"<tr><th>{html.escape(name)}</th><td>{value['nll']:.4f}</td><td>{value['accuracy']:.2%}</td></tr>" for name, value in scores.items())
    image_data = base64.b64encode((args.output / "comparison.png").read_bytes()).decode("ascii")
    origin = "exported model outputs" if args.npz or args.calibration else "scikit-learn breast-cancer benchmark (illustrative, not a paper result)"
    report = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LAPE demo</title><style>body{{font:16px system-ui,sans-serif;background:#f7f8f6;color:#19302d;max-width:900px;margin:4rem auto;padding:0 1.5rem}}h1{{font-size:2rem}}.card{{background:white;border:1px solid #d9e5df;border-radius:16px;padding:1.5rem;box-shadow:0 8px 24px #163d3010}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:.7rem;border-bottom:1px solid #e1e9e5}}.note{{color:#52645e}}</style><h1>LAPE calibration demo</h1><p class="note">Source: {html.escape(origin)}. The same held-out test set is used for all three methods; this demo does not claim improvement on every dataset.</p><div class="card"><img alt="NLL and accuracy comparison" src="data:image/png;base64,{image_data}"><table><tr><th>Method</th><th>NLL ↓</th><th>Accuracy ↑</th></tr>{rows}</table></div><p class="note">Anchor: {args.anchor}. Five-fold out-of-fold anchor probabilities are used for the LAPE reference set, and the anchor is refit on the full calibration set for test predictions.</p></html>"""
    (args.output / "report.html").write_text(report, encoding="utf-8")
    print(json.dumps(scores, indent=2))
    print(f"Saved demo output to {args.output}")


if __name__ == "__main__":
    main()
