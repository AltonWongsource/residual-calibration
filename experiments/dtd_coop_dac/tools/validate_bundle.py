from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dac_lape.io import load_npz, validate_bundle


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--strict-only", action="store_true")
    args = p.parse_args()
    bundle = load_npz(args.bundle)
    validate_bundle(bundle, require_target_reference=not args.strict_only)
    print("Bundle validation passed.")
    for key in sorted(bundle):
        print(f"{key}: {bundle[key].shape} {bundle[key].dtype}")


if __name__ == "__main__":
    main()
