from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run import target_split


def test_target_split_is_reproducible_disjoint_and_stratified() -> None:
    y = np.repeat(np.arange(10), 100)
    cal_a, test_a = target_split(y, 42)
    cal_b, test_b = target_split(y, 42)
    assert np.array_equal(cal_a, cal_b)
    assert np.array_equal(test_a, test_b)
    assert len(cal_a) == 200 and len(test_a) == 800
    assert not set(cal_a).intersection(test_b)
    assert np.array_equal(np.bincount(y[cal_a]), np.full(10, 20))
    assert np.array_equal(np.bincount(y[test_a]), np.full(10, 80))
