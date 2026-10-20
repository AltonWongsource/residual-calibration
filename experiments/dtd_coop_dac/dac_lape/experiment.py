from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dac import apply_dac, compute_textual_deviation, fit_temperature, softmax_np
from .io import load_npz, save_json, validate_bundle
from .lape import LAPEConfig, fit_predict_lape, pure_local_config
from .metrics import evaluate_probabilities


@dataclass(frozen=True)
class ExperimentConfig:
    k_text: int = 5
    ece_bins: int = 15
    alpha_step: float = 0.05
    feature_mode: str = "l2"
    pca_dim: int = 0
    run_target_labeled_extension: bool = True
    run_strict_lape_diagnostic: bool = False


def _evaluate(probs: np.ndarray, labels: np.ndarray, cfg: ExperimentConfig) -> dict[str, float]:
    return evaluate_probabilities(
        probs,
        labels,
        n_bins=cfg.ece_bins,
    )


def _append_method(
    rows: list[dict[str, object]],
    *,
    protocol: str,
    method: str,
    probs: np.ndarray,
    labels: np.ndarray,
    cfg: ExperimentConfig,
) -> dict[str, float]:
    metrics = _evaluate(probs, labels, cfg)
    rows.append({"protocol": protocol, "method": method, **metrics})
    return metrics


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["protocol", "method"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    *,
    bundle_path: str | Path,
    output_dir: str | Path,
    config: ExperimentConfig | None = None,
) -> dict[str, object]:
    cfg = config or ExperimentConfig()
    bundle = load_npz(bundle_path)
    validate_bundle(bundle, require_target_reference=cfg.run_target_labeled_extension)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pre_text = np.asarray(bundle["pretrained_text_features"], dtype=np.float32)
    tuned_text = np.asarray(bundle["tuned_text_features"], dtype=np.float32)
    base_mask = np.asarray(bundle["base_class_mask"], dtype=bool).reshape(-1)
    gamma = compute_textual_deviation(
        pretrained_text_features=pre_text,
        tuned_text_features=tuned_text,
        base_class_mask=base_mask,
        k=cfg.k_text,
    )
    base_cal_logits = np.asarray(bundle["base_calibration_logits"], dtype=np.float32)
    base_cal_labels = np.asarray(bundle["base_calibration_labels"], dtype=np.int64)
    temperature = fit_temperature(base_cal_logits, base_cal_labels)

    test_logits = np.asarray(bundle["novel_test_logits"], dtype=np.float32)
    test_labels = np.asarray(bundle["novel_test_labels"], dtype=np.int64)
    novel_class_indices = np.asarray(
        bundle.get("novel_logit_class_indices", np.arange(test_logits.shape[1])),
        dtype=np.int64,
    ).reshape(-1)
    if novel_class_indices.size != test_logits.shape[1]:
        raise ValueError("novel_logit_class_indices must match novel logit columns")
    novel_gamma = gamma[novel_class_indices]
    raw_probs = softmax_np(test_logits)
    ts_probs = softmax_np(test_logits / temperature)
    dac = apply_dac(raw_logits=test_logits, gamma_by_class=novel_gamma, temperature=temperature)

    rows: list[dict[str, object]] = []
    strict_metrics = {
        "Tuned": _append_method(rows, protocol="strict_base_to_new", method="Tuned", probs=raw_probs, labels=test_labels, cfg=cfg),
        "TS_base": _append_method(rows, protocol="strict_base_to_new", method="TS_base", probs=ts_probs, labels=test_labels, cfg=cfg),
        "DAC": _append_method(rows, protocol="strict_base_to_new", method="DAC", probs=dac.probs, labels=test_labels, cfg=cfg),
    }

    lape_metadata: dict[str, object] = {}
    if cfg.run_target_labeled_extension:
        ref_logits = np.asarray(bundle["novel_reference_logits"], dtype=np.float32)
        ref_features = np.asarray(bundle["novel_reference_features"], dtype=np.float32)
        ref_labels = np.asarray(bundle["novel_reference_labels"], dtype=np.int64)
        test_features = np.asarray(bundle["novel_test_features"], dtype=np.float32)
        ref_raw_probs = softmax_np(ref_logits)
        ref_dac = apply_dac(
            raw_logits=ref_logits, gamma_by_class=novel_gamma, temperature=temperature
        )
        alpha_grid = tuple(np.arange(0.0, 1.0 + cfg.alpha_step / 2, cfg.alpha_step).tolist())
        lcfg = LAPEConfig(
            alpha_grid=alpha_grid,
            feature_mode=cfg.feature_mode,
            pca_dim=cfg.pca_dim,
        )
        raw_lape, raw_meta = fit_predict_lape(
            reference_features=ref_features,
            reference_labels=ref_labels,
            reference_anchor_probs=ref_raw_probs,
            query_features=test_features,
            query_anchor_probs=raw_probs,
            config=lcfg,
        )
        dac_lape, dac_meta = fit_predict_lape(
            reference_features=ref_features,
            reference_labels=ref_labels,
            reference_anchor_probs=ref_dac.probs,
            query_features=test_features,
            query_anchor_probs=dac.probs,
            config=lcfg,
        )
        pure_lape, pure_meta = fit_predict_lape(
            reference_features=ref_features,
            reference_labels=ref_labels,
            reference_anchor_probs=ref_raw_probs,
            query_features=test_features,
            query_anchor_probs=raw_probs,
            config=pure_local_config(lcfg),
        )
        _append_method(rows, protocol="novel_labeled_reference_extension", method="Tuned", probs=raw_probs, labels=test_labels, cfg=cfg)
        _append_method(rows, protocol="novel_labeled_reference_extension", method="DAC", probs=dac.probs, labels=test_labels, cfg=cfg)
        _append_method(rows, protocol="novel_labeled_reference_extension", method="Tuned+LAPE", probs=raw_lape.probs, labels=test_labels, cfg=cfg)
        _append_method(rows, protocol="novel_labeled_reference_extension", method="DAC+LAPE", probs=dac_lape.probs, labels=test_labels, cfg=cfg)
        _append_method(rows, protocol="novel_labeled_reference_extension", method="Pure LAPE", probs=pure_lape.probs, labels=test_labels, cfg=cfg)
        lape_metadata = {
            "Tuned+LAPE": raw_meta,
            "DAC+LAPE": dac_meta,
            "Pure LAPE": pure_meta,
        }
        np.savez_compressed(
            out / "predictions_target_labeled.npz",
            labels=test_labels,
            tuned_probs=raw_probs,
            dac_probs=dac.probs,
            tuned_lape_probs=raw_lape.probs,
            dac_lape_probs=dac_lape.probs,
            pure_lape_probs=pure_lape.probs,
        )

    if cfg.run_strict_lape_diagnostic:
        # This intentionally uses base-only references on novel-class test data.
        # It is a failure-mode diagnostic, not a valid claim of open-vocabulary LAPE.
        needed = {"base_calibration_features"}
        if not needed.issubset(bundle):
            raise ValueError("strict LAPE diagnostic requires base_calibration_features")
        if base_cal_logits.shape[1] != test_logits.shape[1]:
            raise ValueError(
                "base-only strict LAPE diagnostic is undefined when base and novel classifiers use different class subsets"
            )
        base_features = np.asarray(bundle["base_calibration_features"], dtype=np.float32)
        test_features = np.asarray(bundle["novel_test_features"], dtype=np.float32)
        base_probs = softmax_np(base_cal_logits)
        alpha_grid = tuple(np.arange(0.0, 1.0 + cfg.alpha_step / 2, cfg.alpha_step).tolist())
        strict_lape, meta = fit_predict_lape(
            reference_features=base_features,
            reference_labels=base_cal_labels,
            reference_anchor_probs=base_probs,
            query_features=test_features,
            query_anchor_probs=raw_probs,
            config=LAPEConfig(alpha_grid=alpha_grid, feature_mode=cfg.feature_mode, pca_dim=cfg.pca_dim),
        )
        _append_method(rows, protocol="strict_base_to_new_failure_diagnostic", method="Base-only LAPE", probs=strict_lape.probs, labels=test_labels, cfg=cfg)
        lape_metadata["Base-only LAPE diagnostic"] = meta

    class_names = bundle.get("class_names", np.asarray([str(i) for i in range(pre_text.shape[0])], dtype=object))
    gamma_rows = []
    for i, value in enumerate(gamma):
        gamma_rows.append({
            "class_index": i,
            "class_name": str(class_names[i]),
            "is_base": bool(base_mask[i]),
            "gamma": float(value),
        })
    _write_csv(out / "metrics.csv", rows)
    _write_csv(out / "textual_deviation_scores.csv", gamma_rows)
    np.savez_compressed(
        out / "predictions_strict.npz",
        labels=test_labels,
        tuned_probs=raw_probs,
        ts_base_probs=ts_probs,
        dac_probs=dac.probs,
        gamma_global=gamma,
        novel_gamma=novel_gamma,
        novel_logit_class_indices=novel_class_indices,
    )
    summary = {
        "temperature": temperature,
        "k_text": cfg.k_text,
        "strict_metrics": strict_metrics,
        "lape_metadata": lape_metadata,
        "scope": {
            "strict_base_to_new": "Faithful DAC protocol: tuned on/calibrated with base classes and evaluated on novel classes.",
            "novel_labeled_reference_extension": "Separate extension with labelled novel-class reference samples; not the strict unlabeled-novel protocol.",
        },
    }
    save_json(out / "summary.json", summary)
    return {"output_dir": str(out), **summary}
