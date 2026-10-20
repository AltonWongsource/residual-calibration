from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.io import save_json
from common.metrics import (
    evaluate_probabilities,
    prediction_flip_statistics,
)
from common.repro import ensure_dir
from posthoc_cpu.lape import LAPEConfig, fit_predict_lape, pure_local_config
from posthoc_cpu.vector_scaling import cross_fitted_vector_scaling


@dataclass(frozen=True)
class PosthocConfig:
    folds: int = 5
    seed: int = 1
    vs_max_iter: int = 150
    vs_l2: float = 1e-6
    ece_bins: int = 15
    alpha_step: float = 0.05
    feature_mode: str = "none"
    pca_dim: int = 0
    include_direct_reuse: bool = False


def _load_bundle(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _metric_row(
    method: str,
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    config: PosthocConfig,
) -> dict[str, Any]:
    metrics = evaluate_probabilities(
        probs,
        labels,
        n_bins=config.ece_bins,
    )
    return {"method": method, **metrics}


def _save_method_predictions(
    path: Path,
    *,
    probs: np.ndarray,
    diagnostics: dict[str, np.ndarray] | None = None,
) -> None:
    payload: dict[str, np.ndarray] = {"probs": np.asarray(probs, dtype=np.float32)}
    for key, value in (diagnostics or {}).items():
        payload[key] = np.asarray(value)
    np.savez_compressed(path, **payload)


def _relative_improvement(anchor: dict[str, Any], refined: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in ("nll", "ece", "ace", "brier"):
        base = float(anchor[metric])
        out[f"delta_{metric}_percent"] = (
            100.0 * (base - float(refined[metric])) / base if abs(base) > 1e-15 else float("nan")
        )
    acc = float(anchor["accuracy"])
    out["delta_accuracy_percent"] = (
        100.0 * (float(refined["accuracy"]) - acc) / acc if abs(acc) > 1e-15 else float("nan")
    )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["method", "nll", "ece", "ace", "accuracy", "brier"]
    lines = [
        "| Method | NLL | ECE | ACE | Accuracy | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {nll:.6f} | {ece:.6f} | {ace:.6f} | {accuracy:.6f} | {brier:.6f} |".format(
                **{key: row[key] for key in columns}
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_posthoc_comparison(
    *,
    calibration_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    config: PosthocConfig | None = None,
) -> dict[str, Any]:
    cfg = config or PosthocConfig()
    output = ensure_dir(output_dir)
    predictions_dir = ensure_dir(output / "predictions")
    calibration = _load_bundle(calibration_path)
    test = _load_bundle(test_path)

    cal_logits = np.asarray(calibration["logits"], dtype=np.float32)
    cal_probs = np.asarray(calibration["probs"], dtype=np.float32)
    cal_features = np.asarray(calibration["features"], dtype=np.float32)
    cal_labels = np.asarray(calibration["labels"], dtype=np.int64)
    test_logits = np.asarray(test["logits"], dtype=np.float32)
    test_probs = np.asarray(test["probs"], dtype=np.float32)
    test_features = np.asarray(test["features"], dtype=np.float32)
    test_labels = np.asarray(test["labels"], dtype=np.int64)
    if cal_logits.shape[1] != test_logits.shape[1]:
        raise ValueError("calibration and test bundles have different class counts")

    alpha_grid = tuple(np.round(np.arange(0.0, 1.0 + cfg.alpha_step / 2.0, cfg.alpha_step), 10).tolist())
    lape_cfg = LAPEConfig(
        alpha_grid=alpha_grid,
        feature_mode=cfg.feature_mode,
        pca_dim=cfg.pca_dim,
    )

    rows: list[dict[str, Any]] = []
    rows.append(_metric_row("Raw", test_probs, test_labels, config=cfg))
    _save_method_predictions(predictions_dir / "raw.npz", probs=test_probs)

    # Directly refine the train-time model. Its calibration/reference predictions
    # are already out-of-sample with respect to model training.
    raw_lape, raw_lape_meta = fit_predict_lape(
        reference_features=cal_features,
        reference_labels=cal_labels,
        reference_anchor_probs=cal_probs,
        query_features=test_features,
        query_anchor_probs=test_probs,
        config=lape_cfg,
    )
    rows.append(_metric_row("Raw+LAPE", raw_lape.probs, test_labels, config=cfg))
    _save_method_predictions(
        predictions_dir / "raw_lape.npz",
        probs=raw_lape.probs,
        diagnostics={
            "selected_m": raw_lape.selected_m,
            "selected_alpha": raw_lape.selected_alpha,
            "objective": raw_lape.objective,
            "bandwidth": raw_lape.bandwidth,
            "nearest_distance": raw_lape.nearest_distance,
        },
    )

    vs_oof, vs_in_sample, vs_test, _, vs_meta = cross_fitted_vector_scaling(
        cal_logits,
        cal_labels,
        test_logits,
        n_splits=cfg.folds,
        seed=cfg.seed,
        max_iter=cfg.vs_max_iter,
        l2=cfg.vs_l2,
    )
    rows.append(_metric_row("VS", vs_test, test_labels, config=cfg))
    _save_method_predictions(predictions_dir / "vs.npz", probs=vs_test)

    vs_lape, vs_lape_meta = fit_predict_lape(
        reference_features=cal_features,
        reference_labels=cal_labels,
        reference_anchor_probs=vs_oof,
        query_features=test_features,
        query_anchor_probs=vs_test,
        config=lape_cfg,
    )
    rows.append(_metric_row("VS+LAPE", vs_lape.probs, test_labels, config=cfg))
    _save_method_predictions(
        predictions_dir / "vs_lape.npz",
        probs=vs_lape.probs,
        diagnostics={
            "selected_m": vs_lape.selected_m,
            "selected_alpha": vs_lape.selected_alpha,
            "objective": vs_lape.objective,
            "bandwidth": vs_lape.bandwidth,
            "nearest_distance": vs_lape.nearest_distance,
        },
    )

    pure, pure_meta = fit_predict_lape(
        reference_features=cal_features,
        reference_labels=cal_labels,
        reference_anchor_probs=cal_probs,
        query_features=test_features,
        query_anchor_probs=test_probs,
        config=pure_local_config(lape_cfg),
    )
    rows.append(_metric_row("Pure LAPE", pure.probs, test_labels, config=cfg))
    _save_method_predictions(
        predictions_dir / "pure_lape.npz",
        probs=pure.probs,
        diagnostics={
            "selected_m": pure.selected_m,
            "selected_alpha": pure.selected_alpha,
            "objective": pure.objective,
        },
    )

    direct_meta: dict[str, object] | None = None
    if cfg.include_direct_reuse:
        direct, direct_meta = fit_predict_lape(
            reference_features=cal_features,
            reference_labels=cal_labels,
            reference_anchor_probs=vs_in_sample,
            query_features=test_features,
            query_anchor_probs=vs_test,
            config=lape_cfg,
        )
        rows.append(
            _metric_row("VS+LAPE (direct reuse)", direct.probs, test_labels, config=cfg)
        )
        _save_method_predictions(
            predictions_dir / "vs_lape_direct_reuse.npz",
            probs=direct.probs,
            diagnostics={
                "selected_m": direct.selected_m,
                "selected_alpha": direct.selected_alpha,
                "objective": direct.objective,
            },
        )

    by_method = {row["method"]: row for row in rows}
    comparisons = {
        "Raw_to_Raw+LAPE": {
            **_relative_improvement(by_method["Raw"], by_method["Raw+LAPE"]),
            "flip_statistics": asdict(
                prediction_flip_statistics(test_probs, raw_lape.probs, test_labels)
            ),
        },
        "VS_to_VS+LAPE": {
            **_relative_improvement(by_method["VS"], by_method["VS+LAPE"]),
            "flip_statistics": asdict(
                prediction_flip_statistics(vs_test, vs_lape.probs, test_labels)
            ),
        },
    }
    for refined_name, anchor_name, comparison_key in (
        ("Raw+LAPE", "Raw", "Raw_to_Raw+LAPE"),
        ("VS+LAPE", "VS", "VS_to_VS+LAPE"),
    ):
        refined_row = by_method[refined_name]
        refined_row["anchor"] = anchor_name
        refined_row.update(
            {
                key: value
                for key, value in comparisons[comparison_key].items()
                if key != "flip_statistics"
            }
        )
        refined_row.update(
            {
                f"flip_{key}": value
                for key, value in comparisons[comparison_key]["flip_statistics"].items()
            }
        )

    if cfg.include_direct_reuse:
        direct_row = by_method["VS+LAPE (direct reuse)"]
        direct_row["anchor"] = "VS"
        direct_row.update(_relative_improvement(by_method["VS"], direct_row))

    _write_csv(output / "comparison.csv", rows)
    _write_markdown(output / "comparison.md", rows)
    save_json(output / "comparison.json", {"metrics": rows, "comparisons": comparisons})
    save_json(
        output / "posthoc_config.json",
        {
            "config": asdict(cfg),
            "lape_grid": lape_cfg.to_dict(),
            "vector_scaling": vs_meta,
            "raw_lape": raw_lape_meta,
            "vs_lape": vs_lape_meta,
            "pure_lape": pure_meta,
            "direct_reuse_lape": direct_meta,
            "calibration_path": str(calibration_path),
            "test_path": str(test_path),
        },
    )
    return {"metrics": rows, "comparisons": comparisons, "output_dir": str(output)}
