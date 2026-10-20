import numpy as np

from common.metrics import evaluate_probabilities, prediction_flip_statistics


def test_metrics_and_flip_statistics():
    probs = np.array([[0.8, 0.2], [0.4, 0.6]], dtype=float)
    labels = np.array([0, 1])
    metrics = evaluate_probabilities(probs, labels)
    assert metrics["accuracy"] == 1.0
    after = np.array([[0.3, 0.7], [0.4, 0.6]], dtype=float)
    flips = prediction_flip_statistics(probs, after, labels)
    assert flips.change_rate == 0.5
    assert flips.correct_to_wrong == 0.5
