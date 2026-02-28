"""Tests for gae.events dataclasses."""

import numpy as np
import pytest

from gae.events import FactorComputedEvent, WeightsUpdatedEvent, ConvergenceEvent


class TestFactorComputedEvent:
    def test_construction(self):
        v = np.array([0.1, 0.2, 0.3])
        evt = FactorComputedEvent(
            node_id="n1",
            factor_vector=v,
            factor_names=("a", "b", "c"),
        )
        assert evt.node_id == "n1"
        assert evt.factor_vector.shape == (3,)

    def test_frozen(self):
        v = np.array([0.5, 0.6])
        evt = FactorComputedEvent("n2", v, ("x", "y"))
        with pytest.raises((AttributeError, TypeError)):
            evt.node_id = "other"  # type: ignore[misc]

    def test_mismatched_names_raises(self):
        with pytest.raises(AssertionError):
            FactorComputedEvent(
                node_id="n3",
                factor_vector=np.array([1.0, 2.0]),
                factor_names=("only_one",),   # len 1 != 2
            )

    def test_non_1d_vector_raises(self):
        with pytest.raises(AssertionError):
            FactorComputedEvent(
                node_id="n4",
                factor_vector=np.array([[1.0, 2.0]]),  # 2-D
                factor_names=("a", "b"),
            )

    def test_non_array_raises(self):
        with pytest.raises(AssertionError):
            FactorComputedEvent(
                node_id="n5",
                factor_vector=[1.0, 2.0],  # type: ignore[arg-type]
                factor_names=("a", "b"),
            )


class TestWeightsUpdatedEvent:
    def _make(self, **kw):
        defaults = dict(
            weights_before=np.array([0.1, 0.2]),
            weights_after=np.array([0.15, 0.25]),
            delta_norm=0.07,
            step=1,
        )
        defaults.update(kw)
        return WeightsUpdatedEvent(**defaults)

    def test_construction(self):
        evt = self._make()
        assert evt.step == 1
        assert evt.delta_norm == pytest.approx(0.07)

    def test_shape_mismatch_raises(self):
        with pytest.raises(AssertionError):
            self._make(weights_after=np.array([0.1, 0.2, 0.3]))

    def test_negative_step_raises(self):
        with pytest.raises(AssertionError):
            self._make(step=-1)


class TestConvergenceEvent:
    def test_construction(self):
        evt = ConvergenceEvent(step=10, converged=True, delta_norm=1e-5, threshold=1e-4)
        assert evt.converged is True

    def test_zero_threshold_raises(self):
        with pytest.raises(AssertionError):
            ConvergenceEvent(step=0, converged=False, delta_norm=0.1, threshold=0.0)

    def test_negative_step_raises(self):
        with pytest.raises(AssertionError):
            ConvergenceEvent(step=-1, converged=False, delta_norm=0.1, threshold=0.01)
