from __future__ import annotations

import argparse
import json
from pathlib import Path

from posthoc_cpu.pipeline import PosthocConfig, run_posthoc_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Raw/VS/LAPE comparisons from exported bundles.")
    parser.add_argument("--run-dir", help="Directory containing calibration.npz and test.npz")
    parser.add_argument("--calibration")
    parser.add_argument("--test")
    parser.add_argument("--output-dir")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--vs-max-iter", type=int, default=150)
    parser.add_argument("--vs-l2", type=float, default=1e-6)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--feature-mode", choices=["none", "zscore", "l2"], default="none")
    parser.add_argument("--pca-dim", type=int, default=0)
    parser.add_argument("--include-direct-reuse", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir:
        run_dir = Path(args.run_dir)
        calibration = run_dir / "calibration.npz"
        test = run_dir / "test.npz"
        output = Path(args.output_dir) if args.output_dir else run_dir / "posthoc"
    else:
        if not args.calibration or not args.test:
            raise SystemExit("provide --run-dir or both --calibration and --test")
        calibration = Path(args.calibration)
        test = Path(args.test)
        output = Path(args.output_dir or "./posthoc_output")
    result = run_posthoc_comparison(
        calibration_path=calibration,
        test_path=test,
        output_dir=output,
        config=PosthocConfig(
            folds=args.folds,
            seed=args.seed,
            vs_max_iter=args.vs_max_iter,
            vs_l2=args.vs_l2,
            ece_bins=args.ece_bins,
            alpha_step=args.alpha_step,
            feature_mode=args.feature_mode,
            pca_dim=args.pca_dim,
            include_direct_reuse=args.include_direct_reuse,
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
