import numpy as np
from gae.primitives import compute_entropy


def test_entropy_uniform_distribution():
    """Uniform → maximum entropy for 4 classes ≈ log(4)."""
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    e = compute_entropy(probs)
    assert e > 0
    assert abs(e - np.log(4)) < 0.01


def test_entropy_degenerate_distribution():
    """Single class with all probability → near-zero entropy."""
    probs = np.array([1.0, 0.0, 0.0, 0.0])
    e = compute_entropy(probs)
    assert abs(e) < 0.01


def test_entropy_handles_zeros():
    """Zero probabilities don't cause log(0) error."""
    probs = np.array([0.5, 0.5, 0.0, 0.0])
    e = compute_entropy(probs)
    assert np.isfinite(e)
    assert e > 0
