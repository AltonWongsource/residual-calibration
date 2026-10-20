from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from run_fast_rebuttal import _validate_master


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    args = p.parse_args()
    with np.load(args.bundle, allow_pickle=True) as data:
        bundle = {k: data[k] for k in data.files}
    _validate_master(bundle)
    summary = {k: list(np.asarray(v).shape) for k, v in bundle.items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
