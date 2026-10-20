from pathlib import Path

import numpy as np

from posthoc_cpu.lape import LAPEConfig, fit_predict_lape
from posthoc_cpu.pipeline import PosthocConfig, run_posthoc_comparison
from posthoc_cpu.vector_scaling import cross_fitted_vector_scaling


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def test_vector_scaling_oof_shapes_and_probabilities():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(60, 3)).astype(np.float32)
    labels = np.repeat(np.arange(3), 20)
    test_logits = rng.normal(size=(12, 3)).astype(np.float32)
    oof, insample, test, _, meta = cross_fitted_vector_scaling(
        logits, labels, test_logits, n_splits=3, max_iter=15
    )
    assert oof.shape == logits.shape
    assert insample.shape == logits.shape
    assert test.shape == test_logits.shape
    assert np.allclose(oof.sum(axis=1), 1.0, atol=1e-5)
    assert meta["actual_folds"] == 3


def test_lape_returns_valid_query_specific_predictions():
    rng = np.random.default_rng(3)
    labels = np.repeat(np.arange(2), 20)
    features = np.vstack(
        [rng.normal(loc=-2.0, scale=0.2, size=(20, 2)), rng.normal(loc=2.0, scale=0.2, size=(20, 2))]
    )
    query_labels = np.array([0, 1])
    query_features = np.array([[-2.0, -2.0], [2.0, 2.0]])
    anchor_ref = np.full((40, 2), 0.5)
    anchor_query = np.full((2, 2), 0.5)
    result, meta = fit_predict_lape(
        reference_features=features,
        reference_labels=labels,
        reference_anchor_probs=anchor_ref,
        query_features=query_features,
        query_anchor_probs=anchor_query,
        config=LAPEConfig(alpha_grid=(0.0, 0.5, 1.0), m_grid=(2, 3, 4)),
    )
    assert np.allclose(result.probs.sum(axis=1), 1.0)
    assert np.array_equal(result.probs.argmax(axis=1), query_labels)
    assert result.selected_m.shape == (2,)
    assert 0.0 <= meta["mean_selected_alpha"] <= 1.0


def test_end_to_end_posthoc_pipeline(tmp_path: Path):
    rng = np.random.default_rng(4)
    labels = np.repeat(np.arange(3), 12)
    test_labels = np.repeat(np.arange(3), 5)
    features = rng.normal(size=(36, 4)).astype(np.float32)
    test_features = rng.normal(size=(15, 4)).astype(np.float32)
    logits = rng.normal(size=(36, 3)).astype(np.float32)
    test_logits = rng.normal(size=(15, 3)).astype(np.float32)
    np.savez_compressed(
        tmp_path / "calibration.npz",
        logits=logits,
        probs=_softmax(logits).astype(np.float32),
        features=features,
        labels=labels,
        indices=np.arange(labels.size),
    )
    np.savez_compressed(
        tmp_path / "test.npz",
        logits=test_logits,
        probs=_softmax(test_logits).astype(np.float32),
        features=test_features,
        labels=test_labels,
        indices=np.arange(test_labels.size),
    )
    result = run_posthoc_comparison(
        calibration_path=tmp_path / "calibration.npz",
        test_path=tmp_path / "test.npz",
        output_dir=tmp_path / "posthoc",
        config=PosthocConfig(folds=3, vs_max_iter=10, alpha_step=0.5),
    )
    methods = {row["method"] for row in result["metrics"]}
    assert {"Raw", "Raw+LAPE", "VS", "VS+LAPE", "Pure LAPE"} <= methods
    assert (tmp_path / "posthoc" / "comparison.csv").exists()
    assert (tmp_path / "posthoc" / "comparison.md").exists()
