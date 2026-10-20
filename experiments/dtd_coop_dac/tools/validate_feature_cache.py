#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cached_coop.cache_io import load_feature_cache, validate_feature_cache


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    args = p.parse_args()
    cache = load_feature_cache(args.cache)
    validate_feature_cache(cache)
    print("OK", sorted(cache))


if __name__ == "__main__":
    main()
