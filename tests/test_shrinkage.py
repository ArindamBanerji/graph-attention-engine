"""Unit tests for gae.shrinkage."""

import numpy as np
import pytest

from gae.shrinkage import (
    FixedAlpha,
    LinearRampAlpha,
    compute_effective_weights,
)
from gae.two_phase import CategoryState, MEAN_CONVERGENCE, VARIANCE_LEARNING


def test_fixed_alpha_default():
    schedule = FixedAlpha()
    assert schedule.alpha == pytest.approx(0.5)


def test_fixed_alpha_returns_constant():
    schedule = FixedAlpha(alpha=0.3)
    state = CategoryState(
        phase=VARIANCE_LEARNING,
        n_decisions=123,
        freeze_point=40,
    )
    assert schedule.compute_alpha(state) == pytest.approx(0.3)


def test_fixed_alpha_rejects_invalid():
    with pytest.raises(ValueError):
        FixedAlpha(alpha=-0.1)
    with pytest.raises(ValueError):
        FixedAlpha(alpha=1.1)


def test_linear_ramp_before_freeze():
    schedule = LinearRampAlpha(start=0.1, end=0.5, ramp_decisions=1000)
    state = CategoryState(
        phase=MEAN_CONVERGENCE,
        n_decisions=50,
        freeze_point=None,
    )
    assert schedule.compute_alpha(state) == pytest.approx(0.1)


def test_linear_ramp_at_freeze():
    schedule = LinearRampAlpha(start=0.1, end=0.5, ramp_decisions=1000)
    state = CategoryState(
        phase=VARIANCE_LEARNING,
        n_decisions=200,
        freeze_point=200,
    )
    assert schedule.compute_alpha(state) == pytest.approx(0.1)


def test_linear_ramp_midway():
    schedule = LinearRampAlpha(start=0.1, end=0.5, ramp_decisions=1000)
    state = CategoryState(
        phase=VARIANCE_LEARNING,
        n_decisions=700,
        freeze_point=200,
    )
    assert schedule.compute_alpha(state) == pytest.approx(0.3)


def test_linear_ramp_at_end():
    schedule = LinearRampAlpha(start=0.1, end=0.5, ramp_decisions=1000)
    state = CategoryState(
        phase=VARIANCE_LEARNING,
        n_decisions=1200,
        freeze_point=200,
    )
    assert schedule.compute_alpha(state) == pytest.approx(0.5)


def test_linear_ramp_past_end():
    schedule = LinearRampAlpha(start=0.1, end=0.5, ramp_decisions=1000)
    state = CategoryState(
        phase=VARIANCE_LEARNING,
        n_decisions=1800,
        freeze_point=200,
    )
    assert schedule.compute_alpha(state) == pytest.approx(0.5)


def test_linear_ramp_rejects_invalid():
    with pytest.raises(ValueError):
        LinearRampAlpha(start=-0.1)
    with pytest.raises(ValueError):
        LinearRampAlpha(end=1.1)
    with pytest.raises(ValueError):
        LinearRampAlpha(ramp_decisions=0)


def test_compute_effective_weights_alpha_zero():
    weights = np.array([3.0, 0.5, 1.0], dtype=np.float64)
    effective = compute_effective_weights(weights, alpha=0.0)
    assert np.allclose(effective, np.array([1.0, 1.0, 1.0], dtype=np.float64))


def test_compute_effective_weights_alpha_one():
    weights = np.array([3.0, 0.5, 1.0], dtype=np.float64)
    effective = compute_effective_weights(weights, alpha=1.0)
    assert np.allclose(effective, weights)


def test_compute_effective_weights_alpha_half():
    weights = np.array([3.0, 0.5, 1.0], dtype=np.float64)
    effective = compute_effective_weights(weights, alpha=0.5)
    assert np.allclose(effective, np.array([2.0, 0.75, 1.0], dtype=np.float64))


def test_compute_effective_weights_2d():
    weights = np.array([[3.0, 0.5], [1.0, 2.0]], dtype=np.float64)
    effective = compute_effective_weights(weights, alpha=0.25)
    expected = np.array([[1.5, 0.875], [1.0, 1.25]], dtype=np.float64)
    assert effective.shape == weights.shape
    assert np.allclose(effective, expected)


def test_compute_effective_weights_rejects_invalid_alpha():
    weights = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError):
        compute_effective_weights(weights, alpha=-0.01)
    with pytest.raises(ValueError):
        compute_effective_weights(weights, alpha=1.01)


def test_compute_effective_weights_all_positive():
    weights = np.array([3.0, 0.5, 1.0, 4.0], dtype=np.float64)
    effective = compute_effective_weights(weights, alpha=0.6)
    assert np.all(effective > 0.0)
