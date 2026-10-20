from __future__ import annotations

import argparse
import json

from dac_lape.experiment import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run official-formula DAC and the scope-separated DAC+LAPE comparison.")
    p.add_argument("--bundle", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--k-text", type=int, default=5)
    p.add_argument("--ece-bins", type=int, default=15)
    p.add_argument("--alpha-step", type=float, default=0.05)
    p.add_argument("--feature-mode", choices=["none", "zscore", "l2"], default="l2")
    p.add_argument("--pca-dim", type=int, default=0)
    p.add_argument("--strict-only", action="store_true", help="Do not run the labelled-novel-reference LAPE extension.")
    p.add_argument("--strict-lape-diagnostic", action="store_true", help="Run the intentionally out-of-scope base-only-reference LAPE diagnostic.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(
        bundle_path=args.bundle,
        output_dir=args.output_dir,
        config=ExperimentConfig(
            k_text=args.k_text,
            ece_bins=args.ece_bins,
            alpha_step=args.alpha_step,
            feature_mode=args.feature_mode,
            pca_dim=args.pca_dim,
            run_target_labeled_extension=not args.strict_only,
            run_strict_lape_diagnostic=args.strict_lape_diagnostic,
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
