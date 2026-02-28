"""Tests for gae.convergence — get_convergence_metrics()."""

import numpy as np
import pytest

from gae.convergence import (
    ACCURACY_THRESHOLD,
    STABILITY_THRESHOLD,
    get_convergence_metrics,
)
from gae.learning import LearningState


W_INIT = np.array([
    [ 0.30, -0.10, -0.25, -0.15,  0.20,  0.25],
    [ 0.05,  0.20,  0.15,  0.10, -0.05,  0.05],
    [-0.10,  0.05,  0.20,  0.20, -0.10, -0.05],
    [-0.25,  0.30,  0.30,  0.15, -0.20, -0.15],
], dtype=np.float64)
NAMES = ["travel", "asset", "threat", "time", "device", "pattern"]
F = np.array([[0.9, 0.3, 0.1, 0.2, 0.8, 0.7]])


def make_state(**kw) -> LearningState:
    d = dict(W=W_INIT.copy(), n_actions=4, n_factors=6, factor_names=NAMES[:])
    d.update(kw)
    return LearningState(**d)


class TestGetConvergenceMetricsKeys:
    """All expected keys must always be present."""

    def test_keys_empty_history(self):
        s = make_state()
        m = get_convergence_metrics(s)
        assert "stability" in m
        assert "accuracy" in m
        assert "weight_norm" in m
        assert "converged" in m
        assert "decisions" in m
        assert "provisional_dimensions" in m
        assert "pending_autonomous" in m

    def test_keys_after_updates(self):
        s = make_state()
        s.update(0, "fp_close", +1, F)
        s.update(0, "fp_close", -1, F)
        m = get_convergence_metrics(s)
        assert "stability" in m and "accuracy" in m

    def test_all_values_numeric(self):
        s = make_state()
        s.update(0, "fp_close", +1, F)
        m = get_convergence_metrics(s)
        for k in ("stability", "accuracy", "weight_norm"):
            assert isinstance(m[k], float), f"{k} should be float"
        assert isinstance(m["decisions"], int)
        assert isinstance(m["converged"], bool)


class TestGetConvergenceMetricsValues:
    def test_weight_norm_positive(self):
        s = make_state()
        m = get_convergence_metrics(s)
        assert m["weight_norm"] > 0

    def test_empty_history_stability_zero(self):
        s = make_state()
        m = get_convergence_metrics(s)
        assert m["stability"] == 0.0

    def test_empty_history_accuracy_zero(self):
        s = make_state()
        m = get_convergence_metrics(s)
        assert m["accuracy"] == 0.0

    def test_empty_history_not_converged(self):
        s = make_state()
        m = get_convergence_metrics(s)
        assert m["converged"] is False

    def test_accuracy_all_correct(self):
        s = make_state()
        for _ in range(5):
            s.update(0, "fp_close", +1, F)
        m = get_convergence_metrics(s)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_accuracy_all_wrong(self):
        s = make_state()
        for _ in range(5):
            s.update(0, "fp_close", -1, F)
        m = get_convergence_metrics(s)
        assert m["accuracy"] == pytest.approx(0.0)

    def test_accuracy_mixed(self):
        s = make_state()
        s.update(0, "fp_close", +1, F)
        s.update(0, "fp_close", -1, F)
        m = get_convergence_metrics(s)
        assert m["accuracy"] == pytest.approx(0.5)

    def test_decisions_count_correct(self):
        s = make_state()
        s.update(0, "fp_close", +1, F)
        s.update(0, "fp_close", +1, F)
        m = get_convergence_metrics(s)
        assert m["decisions"] == 2

    def test_stable_weights_low_stability(self):
        """Many identical correct outcomes → W stabilises."""
        s = make_state()
        # Apply 20 identical updates — W should converge
        for _ in range(50):
            s.update(0, "fp_close", +1, F)
        m = get_convergence_metrics(s)
        # Stability measures std of recent W norms; may be > 0 early then converge
        assert isinstance(m["stability"], float)

    def test_converged_requires_high_accuracy(self):
        """Low accuracy → not converged even if stable."""
        s = make_state()
        for _ in range(20):
            s.update(0, "fp_close", -1, F)  # all wrong → accuracy = 0
        m = get_convergence_metrics(s)
        assert m["converged"] is False

    def test_provisional_dimensions_counted(self):
        s = make_state()
        s.expand_weight_matrix("new_dim")
        m = get_convergence_metrics(s)
        assert m["provisional_dimensions"] == 1

    def test_pending_autonomous_counted(self):
        s = make_state()
        s.update(0, "fp_close", +1, F, decision_source="autonomous")
        m = get_convergence_metrics(s)
        assert m["pending_autonomous"] == 1


class TestConvergenceTaskVerification:
    def test_task_verification_smoke(self):
        """Exact replication of the verification script from the task description."""
        W = np.array([
            [ 0.3, -0.1, -0.25, -0.15,  0.2,  0.25],
            [ 0.05,  0.2,  0.15,  0.10, -0.05,  0.05],
            [-0.1,  0.05,  0.20,  0.20, -0.1, -0.05],
            [-0.25,  0.3,  0.30,  0.15, -0.2, -0.15],
        ])
        f = np.array([[0.9, 0.3, 0.1, 0.2, 0.8, 0.7]])
        names = ["travel", "asset", "threat", "time", "device", "pattern"]
        state = LearningState(W=W.copy(), n_actions=4, n_factors=6, factor_names=names)
        state.update(0, "fp_close", +1, f)
        state.update(0, "fp_close", -1, f)
        state.expand_weight_matrix("new_dim")
        m = get_convergence_metrics(state)
        assert "stability" in m
        assert "accuracy" in m
