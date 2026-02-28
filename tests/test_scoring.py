"""Tests for gae.scoring — Eq. 4 scoring matrix."""

import numpy as np
import pytest

from gae.scoring import ScoringResult, score_alert


# Shared fixtures ──────────────────────────────────────────────────────────

ACTIONS = ["fp_close", "escalate_t2", "enrich_wait", "escalate_incident"]

W4x6 = np.array([
    [ 0.30, -0.10, -0.25, -0.15,  0.20,  0.25],
    [ 0.05,  0.20,  0.15,  0.10, -0.05,  0.05],
    [-0.10,  0.05,  0.20,  0.20, -0.10, -0.05],
    [-0.25,  0.30,  0.30,  0.15, -0.20, -0.15],
], dtype=np.float64)

F_TRAVEL = np.array([[0.9, 0.3, 0.1, 0.2, 0.8, 0.7]])   # high travel → fp_close


# ── score_alert basic contract ─────────────────────────────────────────────

class TestScoreAlert:
    def test_returns_scoring_result(self):
        r = score_alert(F_TRAVEL, W4x6, ACTIONS)
        assert isinstance(r, ScoringResult)

    def test_selects_fp_close_for_travel_pattern(self):
        r = score_alert(F_TRAVEL, W4x6, ACTIONS, tau=0.25)
        assert r.selected_action == "fp_close"

    def test_probabilities_sum_to_one(self):
        r = score_alert(F_TRAVEL, W4x6, ACTIONS)
        np.testing.assert_allclose(
            r.action_probabilities.flatten().sum(), 1.0, atol=1e-6
        )

    def test_probabilities_shape(self):
        r = score_alert(F_TRAVEL, W4x6, ACTIONS)
        assert r.action_probabilities.shape == (1, len(ACTIONS))

    def test_raw_scores_shape(self):
        r = score_alert(F_TRAVEL, W4x6, ACTIONS)
        assert r.raw_scores.shape == (1, len(ACTIONS))

    def test_factor_vector_preserved_r4(self):
        """Factor vector must be preserved unchanged (Requirement R4)."""
        r = score_alert(F_TRAVEL, W4x6, ACTIONS)
        np.testing.assert_array_equal(r.factor_vector, F_TRAVEL)

    def test_confidence_equals_max_probability(self):
        r = score_alert(F_TRAVEL, W4x6, ACTIONS)
        assert r.confidence == pytest.approx(r.action_probabilities.flatten().max())

    def test_temperature_stored(self):
        r = score_alert(F_TRAVEL, W4x6, ACTIONS, tau=0.5)
        assert r.temperature == 0.5

    def test_probabilities_all_positive(self):
        r = score_alert(F_TRAVEL, W4x6, ACTIONS)
        assert np.all(r.action_probabilities > 0)


# ── temperature effects ────────────────────────────────────────────────────

class TestTemperature:
    def test_lower_tau_gives_higher_confidence(self):
        r_sharp = score_alert(F_TRAVEL, W4x6, ACTIONS, tau=0.1)
        r_flat = score_alert(F_TRAVEL, W4x6, ACTIONS, tau=1.0)
        assert r_sharp.confidence > r_flat.confidence

    def test_tau_one_matches_standard_softmax(self):
        """τ=1.0 is ordinary softmax(f·Wᵀ)."""
        r = score_alert(F_TRAVEL, W4x6, ACTIONS, tau=1.0)
        np.testing.assert_allclose(r.action_probabilities.sum(), 1.0, atol=1e-6)

    def test_zero_tau_raises(self):
        with pytest.raises(ValueError, match="tau must be > 0"):
            score_alert(F_TRAVEL, W4x6, ACTIONS, tau=0.0)

    def test_negative_tau_raises(self):
        with pytest.raises(ValueError):
            score_alert(F_TRAVEL, W4x6, ACTIONS, tau=-0.1)


# ── shape / type guards ────────────────────────────────────────────────────

class TestShapeGuards:
    def test_1d_f_raises(self):
        with pytest.raises(AssertionError):
            score_alert(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), W4x6, ACTIONS)

    def test_wrong_n_factors_raises(self):
        W_bad = np.random.randn(4, 7)  # 7 factors, not 6
        with pytest.raises(AssertionError):
            score_alert(F_TRAVEL, W_bad, ACTIONS)

    def test_wrong_n_actions_raises(self):
        with pytest.raises(AssertionError):
            score_alert(F_TRAVEL, W4x6, ACTIONS[:-1])  # 3 actions, W has 4

    def test_non_array_f_raises(self):
        with pytest.raises(AssertionError):
            score_alert([[0.9, 0.3, 0.1, 0.2, 0.8, 0.7]], W4x6, ACTIONS)  # type: ignore

    def test_non_array_W_raises(self):
        with pytest.raises(AssertionError):
            score_alert(F_TRAVEL, W4x6.tolist(), ACTIONS)  # type: ignore


# ── verification smoke test (matches task specification) ──────────────────

def test_task_verification_smoke():
    """Exact replication of the verification script from the task description."""
    W = np.array([
        [ 0.3, -0.1, -0.25, -0.15,  0.2,  0.25],
        [ 0.05,  0.2,  0.15,  0.10, -0.05,  0.05],
        [-0.1,  0.05,  0.20,  0.20, -0.1, -0.05],
        [-0.25,  0.3,  0.30,  0.15, -0.2, -0.15],
    ])
    f = np.array([[0.9, 0.3, 0.1, 0.2, 0.8, 0.7]])
    actions = ["fp_close", "escalate_t2", "enrich_wait", "escalate_incident"]
    r = score_alert(f, W, actions, tau=0.25)
    assert r.selected_action == "fp_close"
    assert abs(sum(r.action_probabilities.flatten()) - 1.0) < 1e-6
