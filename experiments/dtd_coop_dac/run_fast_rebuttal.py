from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from dac_lape.experiment import ExperimentConfig, run_experiment
from dac_lape.io import load_npz, save_json

MASTER_REQUIRED = {
    "pretrained_text_features",
    "tuned_text_features",
    "base_class_mask",
    "base_calibration_logits",
    "base_calibration_labels",
    "novel_pool_logits",
    "novel_pool_features",
    "novel_pool_labels",
}


def _parse_seeds(text: str) -> list[int]:
    seeds = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one split seed is required")
    return seeds


def _validate_master(bundle: dict[str, np.ndarray]) -> None:
    missing = MASTER_REQUIRED - set(bundle)
    if missing:
        raise ValueError(f"master bundle is missing keys: {sorted(missing)}")
    pre = np.asarray(bundle["pretrained_text_features"])
    tuned = np.asarray(bundle["tuned_text_features"])
    mask = np.asarray(bundle["base_class_mask"]).reshape(-1)
    if pre.ndim != 2 or tuned.shape != pre.shape or mask.size != pre.shape[0]:
        raise ValueError("text features and base mask have incompatible shapes")
    c = pre.shape[0]
    base_logits = np.asarray(bundle["base_calibration_logits"])
    base_labels = np.asarray(bundle["base_calibration_labels"]).reshape(-1)
    pool_logits = np.asarray(bundle["novel_pool_logits"])
    pool_features = np.asarray(bundle["novel_pool_features"])
    pool_labels = np.asarray(bundle["novel_pool_labels"]).reshape(-1)
    if base_logits.ndim != 2 or base_logits.shape[0] != base_labels.size:
        raise ValueError("base calibration logits/labels have incompatible shapes")
    if base_labels.min() < 0 or base_labels.max() >= base_logits.shape[1]:
        raise ValueError("base calibration labels outside local logit-column range")
    if "base_logit_class_indices" in bundle:
        mapping = np.asarray(bundle["base_logit_class_indices"]).reshape(-1)
        if mapping.size != base_logits.shape[1] or mapping.min() < 0 or mapping.max() >= c:
            raise ValueError("invalid base_logit_class_indices")
    elif base_logits.shape[1] != c:
        raise ValueError("base_logit_class_indices is required for subset-class base logits")
    if pool_logits.ndim != 2 or pool_logits.shape[0] != pool_labels.size:
        raise ValueError("novel pool logits/labels have incompatible shapes")
    if pool_labels.min() < 0 or pool_labels.max() >= pool_logits.shape[1]:
        raise ValueError("novel pool labels outside local logit-column range")
    if "novel_logit_class_indices" in bundle:
        mapping = np.asarray(bundle["novel_logit_class_indices"]).reshape(-1)
        if mapping.size != pool_logits.shape[1] or mapping.min() < 0 or mapping.max() >= c:
            raise ValueError("invalid novel_logit_class_indices")
    elif pool_logits.shape[1] != c:
        raise ValueError("novel_logit_class_indices is required for subset-class novel logits")
    if pool_features.ndim != 2 or pool_features.shape[0] != pool_labels.size:
        raise ValueError("novel pool features/labels have incompatible shapes")
    if np.unique(pool_labels).size < 2:
        raise ValueError("novel pool must contain at least two classes")
    counts = np.unique(pool_labels, return_counts=True)[1]
    if counts.min() < 2:
        raise ValueError("every novel class needs at least two examples")


def _common(bundle: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = [
        "pretrained_text_features",
        "tuned_text_features",
        "base_class_mask",
        "base_calibration_logits",
        "base_calibration_labels",
    ]
    out = {k: np.asarray(bundle[k]) for k in keys}
    for optional in ["class_names", "base_logit_class_indices", "novel_logit_class_indices"]:
        if optional in bundle:
            out[optional] = np.asarray(bundle[optional])
    if "metadata_json" in bundle:
        out["metadata_json"] = np.asarray(bundle["metadata_json"])
    return out


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _read_metrics(path: Path, split_seed: int | None, model_seed: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.insert(0, "model_seed", model_seed)
    df.insert(1, "split_seed", "strict" if split_seed is None else split_seed)
    return df


def _aggregate(extension: pd.DataFrame, output_dir: Path) -> None:
    metric_cols = [
        c
        for c in extension.columns
        if c not in {"model_seed", "split_seed", "protocol", "method"}
        and pd.api.types.is_numeric_dtype(extension[c])
    ]
    rows: list[dict[str, object]] = []
    for method, group in extension.groupby("method", sort=False):
        row: dict[str, object] = {"method": method, "n_splits": int(group["split_seed"].nunique())}
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    agg = pd.DataFrame(rows)
    agg.to_csv(output_dir / "extension_aggregate.csv", index=False)

    display = [
        m
        for m in ["nll", "ece", "ace", "accuracy", "brier"]
        if f"{m}_mean" in agg
    ]
    pretty = {
        "nll": "NLL",
        "ece": "ECE",
        "ace": "ACE",
        "accuracy": "Accuracy",
        "brier": "Brier",
    }
    lines = ["# DAC + LAPE labelled-reference extension", "", "Mean +/- standard deviation over reference/test split seeds.", ""]
    header = ["Method"] + [pretty.get(m, m) for m in display]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for _, row in agg.iterrows():
        cells = [str(row["method"])]
        for metric in display:
            cells.append(f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    (output_dir / "extension_aggregate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    anchors = {"Tuned+LAPE": "Tuned", "DAC+LAPE": "DAC"}
    gain_rows: list[dict[str, object]] = []
    for refined, anchor in anchors.items():
        if refined not in set(extension["method"]) or anchor not in set(extension["method"]):
            continue
        for seed in sorted(extension["split_seed"].unique(), key=str):
            a = extension[(extension["method"] == anchor) & (extension["split_seed"] == seed)].iloc[0]
            r = extension[(extension["method"] == refined) & (extension["split_seed"] == seed)].iloc[0]
            rec: dict[str, object] = {"split_seed": seed, "anchor": anchor, "refined": refined}
            for metric in metric_cols:
                av = float(a[metric])
                rv = float(r[metric])
                if metric == "accuracy":
                    rec[f"delta_{metric}"] = rv - av
                else:
                    rec[f"relative_improvement_{metric}_pct"] = 100.0 * (av - rv) / max(abs(av), 1e-12)
            gain_rows.append(rec)
    if gain_rows:
        pd.DataFrame(gain_rows).to_csv(output_dir / "lape_gain_by_split.csv", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast CPU rebuttal runner: one CoOp checkpoint, strict DAC, and repeated novel-reference LAPE splits.")
    p.add_argument("--master-bundle", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split-seeds", type=_parse_seeds, default=[0, 1, 2])
    p.add_argument("--reference-fraction", type=float, default=0.20)
    p.add_argument("--model-seed", default="one_fixed_coop_checkpoint")
    p.add_argument("--k-text", type=int, default=5)
    p.add_argument("--alpha-step", type=float, default=0.05)
    p.add_argument("--feature-mode", choices=["none", "zscore", "l2"], default="l2")
    p.add_argument("--pca-dim", type=int, default=0)
    p.add_argument("--keep-split-bundles", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.05 <= args.reference_fraction <= 0.50:
        raise ValueError("reference_fraction must be between 0.05 and 0.50")
    start = time.perf_counter()
    master = load_npz(args.master_bundle)
    _validate_master(master)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle_dir = output / "generated_bundles"
    common = _common(master)
    pool_logits = np.asarray(master["novel_pool_logits"])
    pool_features = np.asarray(master["novel_pool_features"])
    pool_labels = np.asarray(master["novel_pool_labels"]).reshape(-1)

    strict_bundle = bundle_dir / "strict_full_novel_pool.npz"
    _write_npz(
        strict_bundle,
        {
            **common,
            "novel_test_logits": pool_logits,
            "novel_test_features": pool_features,
            "novel_test_labels": pool_labels,
        },
    )
    strict_out = output / "strict_dac"
    run_experiment(
        bundle_path=strict_bundle,
        output_dir=strict_out,
        config=ExperimentConfig(
            k_text=args.k_text,
            alpha_step=args.alpha_step,
            feature_mode=args.feature_mode,
            pca_dim=args.pca_dim,
            run_target_labeled_extension=False,
        ),
    )
    strict_df = _read_metrics(strict_out / "metrics.csv", None, args.model_seed)
    strict_df.to_csv(output / "strict_metrics.csv", index=False)

    extension_frames: list[pd.DataFrame] = []
    split_manifest: list[dict[str, object]] = []
    for seed in args.split_seeds:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=args.reference_fraction,
            random_state=seed,
        )
        reference_idx, test_idx = next(splitter.split(pool_features, pool_labels))
        split_bundle = bundle_dir / f"novel_reference_split_seed{seed}.npz"
        _write_npz(
            split_bundle,
            {
                **common,
                "novel_reference_logits": pool_logits[reference_idx],
                "novel_reference_features": pool_features[reference_idx],
                "novel_reference_labels": pool_labels[reference_idx],
                "novel_test_logits": pool_logits[test_idx],
                "novel_test_features": pool_features[test_idx],
                "novel_test_labels": pool_labels[test_idx],
            },
        )
        seed_out = output / "labelled_novel_extension" / f"split_seed_{seed}"
        run_experiment(
            bundle_path=split_bundle,
            output_dir=seed_out,
            config=ExperimentConfig(
                k_text=args.k_text,
                alpha_step=args.alpha_step,
                feature_mode=args.feature_mode,
                pca_dim=args.pca_dim,
                run_target_labeled_extension=True,
            ),
        )
        frame = _read_metrics(seed_out / "metrics.csv", seed, args.model_seed)
        frame = frame[frame["protocol"] == "novel_labeled_reference_extension"].copy()
        extension_frames.append(frame)
        split_manifest.append(
            {
                "split_seed": seed,
                "n_reference": int(reference_idx.size),
                "n_test": int(test_idx.size),
                "reference_fraction": float(reference_idx.size / pool_labels.size),
            }
        )

    extension_df = pd.concat(extension_frames, ignore_index=True)
    extension_df.to_csv(output / "extension_metrics_per_split.csv", index=False)
    _aggregate(extension_df, output)

    elapsed = time.perf_counter() - start
    manifest = {
        "model_seed": args.model_seed,
        "split_seeds": args.split_seeds,
        "reference_fraction_requested": args.reference_fraction,
        "strict_protocol": "Tuned / TS_base / DAC on the full official novel pool from one fixed CoOp checkpoint.",
        "lape_protocol": "Tuned / DAC / Tuned+LAPE / DAC+LAPE / Pure LAPE on labelled novel-reference splits.",
        "cross_fitting_note": "No novel-reference OOF anchor construction is needed because CoOp is trained on base classes and DAC is fitted on base calibration labels; neither uses novel-reference labels.",
        "split_manifest": split_manifest,
        "elapsed_seconds": elapsed,
        "master_bundle": str(Path(args.master_bundle).resolve()),
    }
    save_json(output / "run_manifest.json", manifest)
    if not args.keep_split_bundles:
        shutil.rmtree(bundle_dir, ignore_errors=True)
    print(json.dumps({"output_dir": str(output), "elapsed_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
