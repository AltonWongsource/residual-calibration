import numpy as np

from dac_lape.dac import apply_dac, compute_textual_deviation, softmax_np, textual_proximity


def test_base_gamma_is_one_and_positive():
    rng = np.random.default_rng(1)
    pre = rng.normal(size=(8, 5))
    tuned = pre + 0.1 * rng.normal(size=(8, 5))
    mask = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    gamma = compute_textual_deviation(
        pretrained_text_features=pre,
        tuned_text_features=tuned,
        base_class_mask=mask,
        k=3,
    )
    assert np.allclose(gamma[mask], 1.0)
    assert np.all(gamma > 0)


def test_dac_preserves_argmax():
    logits = np.array([[2.0, 1.0, -1.0], [0.1, 0.3, 0.2]])
    gamma = np.array([0.5, 0.8, 1.0])
    out = apply_dac(raw_logits=logits, gamma_by_class=gamma, temperature=1.4)
    assert np.array_equal(logits.argmax(1), out.probs.argmax(1))
    assert np.allclose(out.probs.sum(1), 1.0)


def test_proximity_decreases_with_distance():
    base = np.array([[1.0, 0.0], [0.9, 0.1]])
    near = np.array([[1.0, 0.0]])
    far = np.array([[-1.0, 0.0]])
    assert textual_proximity(near, base, k=2)[0] > textual_proximity(far, base, k=2)[0]
