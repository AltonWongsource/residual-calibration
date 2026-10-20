from __future__ import annotations

import numpy as np

from dac_lape.experiment import ExperimentConfig, run_experiment


def test_subset_class_logits_supported(tmp_path):
    rng = np.random.default_rng(0)
    c_global, d = 6, 5
    pre = rng.normal(size=(c_global, d)).astype(np.float32)
    tuned = (pre + 0.05 * rng.normal(size=pre.shape)).astype(np.float32)
    mask = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    base_logits = rng.normal(size=(30, 3)).astype(np.float32)
    base_labels = rng.integers(0, 3, size=30)
    test_logits = rng.normal(size=(40, 3)).astype(np.float32)
    test_labels = rng.integers(0, 3, size=40)
    path = tmp_path / "bundle.npz"
    np.savez_compressed(
        path,
        pretrained_text_features=pre,
        tuned_text_features=tuned,
        base_class_mask=mask,
        base_calibration_logits=base_logits,
        base_calibration_labels=base_labels,
        base_logit_class_indices=np.array([0, 1, 2]),
        novel_test_logits=test_logits,
        novel_test_labels=test_labels,
        novel_logit_class_indices=np.array([3, 4, 5]),
    )
    result = run_experiment(
        bundle_path=path,
        output_dir=tmp_path / "out",
        config=ExperimentConfig(run_target_labeled_extension=False),
    )
    assert "strict_metrics" in result
