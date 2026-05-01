"""Tests for gae.learning — Eq. 4b/4c weight updates + LearningState."""

import numpy as np
import pytest

from gae.calibration import CalibrationProfile
from gae.learning import (
    ALPHA,
    EPSILON,
    LAMBDA_NEG,
    W_CLAMP,
    DimensionMetadata,
    LearningState,
    PendingValidation,
    WeightUpdate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FACTOR_NAMES = ["travel", "asset", "threat", "time", "device", "pattern"]

W_INIT = np.array([
    [ 0.30, -0.10, -0.25, -0.15,  0.20,  0.25],
    [ 0.05,  0.20,  0.15,  0.10, -0.05,  0.05],
    [-0.10,  0.05,  0.20,  0.20, -0.10, -0.05],
    [-0.25,  0.30,  0.30,  0.15, -0.20, -0.15],
], dtype=np.float64)

F = np.array([[0.9, 0.3, 0.1, 0.2, 0.8, 0.7]])


def make_state(**kwargs) -> LearningState:
    defaults = dict(
        W=W_INIT.copy(),
        n_actions=4,
        n_factors=6,
        factor_names=FACTOR_NAMES[:],
        profile=CalibrationProfile(),
    )
    defaults.update(kwargs)
    return LearningState(**defaults)


# ---------------------------------------------------------------------------
# LearningState construction
# ---------------------------------------------------------------------------

class TestLearningStateConstruction:
    def test_basic_construction(self):
        s = make_state()
        expected_shape = W_INIT.shape
        assert s.n_actions == 4
        assert s.n_factors == expected_shape[1]
        assert s.W.shape == expected_shape

    def test_epsilon_vector_initialised(self):
        s = make_state()
        assert s.epsilon_vector is not None
        assert s.epsilon_vector.shape == (s.n_factors,)
        np.testing.assert_allclose(s.epsilon_vector, EPSILON)

    def test_no_op_defaults(self):
        s = make_state()
        assert s.discount_strength == 0.0
        assert s.dimension_metadata == []
        assert s.pending_validations == []

    def test_shape_mismatch_raises(self):
        with pytest.raises(AssertionError):
            LearningState(W=W_INIT.copy(), n_actions=3, n_factors=6,
                          factor_names=FACTOR_NAMES, profile=CalibrationProfile())

    def test_wrong_factor_names_length_raises(self):
        with pytest.raises(AssertionError):
            LearningState(W=W_INIT.copy(), n_actions=4, n_factors=6,
                          factor_names=["a", "b"], profile=CalibrationProfile())


# ---------------------------------------------------------------------------
# update — Eq. 4b/4c
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_returns_weight_update_for_analyst(self):
        s = make_state()
        u = s.update(0, "fp_close", +1, F)
        assert isinstance(u, WeightUpdate)

    def test_returns_none_for_autonomous(self):
        s = make_state()
        result = s.update(0, "fp_close", +1, F, decision_source="autonomous")
        assert result is None

    def test_pending_validation_added_for_autonomous(self):
        s = make_state()
        s.update(0, "fp_close", +1, F, decision_source="autonomous")
        assert len(s.pending_validations) == 1

    def test_decision_count_increments(self):
        s = make_state()
        s.update(0, "fp_close", +1, F)
        assert s.decision_count == 1
        s.update(0, "fp_close", -1, F)
        assert s.decision_count == 2

    def test_history_grows(self):
        s = make_state()
        s.update(0, "fp_close", +1, F)
        s.update(1, "escalate_t2", -1, F)
        assert len(s.history) == 2

    def test_bad_outcome_raises(self):
        s = make_state()
        with pytest.raises(AssertionError):
            s.update(0, "fp_close", 0, F)   # 0 is not ±1

    def test_wrong_f_shape_raises(self):
        s = make_state()
        with pytest.raises(AssertionError):
            s.update(0, "fp_close", +1, np.array([0.5] * 6))  # 1-D

    def test_w_shape_preserved_after_update(self):
        s = make_state()
        s.update(0, "fp_close", +1, F)
        assert s.W.shape == W_INIT.shape

    def test_eq4b_correct_outcome_delta_vector(self):
        """delta_applied for +1 outcome should be α · f."""
        s = make_state()
        u = s.update(0, "fp_close", +1, F)
        expected = ALPHA * 1.0 * F.flatten() * 1.0   # δ=1 for correct
        np.testing.assert_allclose(u.delta_applied, expected, rtol=1e-9)

    def test_eq4b_incorrect_outcome_delta_vector(self):
        """delta_applied for -1 outcome should be -α·LAMBDA_NEG·f."""
        s = make_state()
        u = s.update(0, "fp_close", -1, F)
        expected = ALPHA * (-1.0) * F.flatten() * LAMBDA_NEG
        np.testing.assert_allclose(u.delta_applied, expected, rtol=1e-9)

    def test_20x_asymmetry_ratio(self):
        """Requirement R4 asymmetry: one failure ≈ 20 successes in norm."""
        s = make_state()
        u1 = s.update(0, "fp_close", +1, F)
        # reset so u2 starts fresh
        s2 = make_state()
        u2 = s2.update(0, "fp_close", -1, F)
        ratio = np.linalg.norm(u2.delta_applied) / np.linalg.norm(u1.delta_applied)
        assert 19.0 < ratio < 21.0, f"Asymmetry ratio {ratio:.2f} not ≈ 20×"

    def test_eq4c_decay_applied(self):
        """W decays by (1 - ε) after each update."""
        s = make_state()
        W_before = s.W.copy()
        s.update(0, "fp_close", +1, F)
        # The WHOLE matrix is decayed (not just action row)
        # W_after = (W_before + delta on row 0) * (1 - ε), then clamped
        # Check at least one off-row decayed
        for row in range(1, 4):
            assert not np.allclose(s.W[row], W_before[row])

    def test_w_clamp_applied(self):
        """W values must stay within [-W_CLAMP, +W_CLAMP]."""
        # Use a large f to drive W toward saturation
        big_f = np.ones((1, 6)) * 1.0
        s = make_state()
        for _ in range(500):
            s.update(0, "fp_close", +1, big_f)
        assert np.all(s.W <= W_CLAMP + 1e-9)
        assert np.all(s.W >= -W_CLAMP - 1e-9)

    def test_w_before_and_w_after_in_record(self):
        s = make_state()
        u = s.update(0, "fp_close", +1, F)
        assert u.W_before.shape == W_INIT.shape
        assert u.W_after.shape == W_INIT.shape
        assert not np.allclose(u.W_before, u.W_after)

    def test_factor_vector_preserved_in_record_r4(self):
        """factor_vector in WeightUpdate is a copy, not a reference (R4)."""
        s = make_state()
        f_copy = F.copy()
        u = s.update(0, "fp_close", +1, f_copy)
        f_copy[:] = 0.0   # mutate original
        np.testing.assert_array_equal(u.factor_vector, F)

    def test_a1_discount_reduces_alpha(self):
        """A1: positive outcome with high confidence → smaller effective α."""
        s = make_state(discount_strength=0.5)
        u = s.update(0, "fp_close", +1, F, confidence_at_decision=0.9)
        assert u.alpha_effective < ALPHA

    def test_a1_inactive_when_discount_zero(self):
        """Default discount_strength=0 means A1 is a no-op."""
        s = make_state()
        u = s.update(0, "fp_close", +1, F, confidence_at_decision=0.99)
        assert u.alpha_effective == pytest.approx(ALPHA)

    def test_a1_does_not_discount_negative_outcome(self):
        """A1 only discounts confirmations (outcome=+1)."""
        s = make_state(discount_strength=0.5)
        u = s.update(0, "fp_close", -1, F, confidence_at_decision=0.99)
        assert u.alpha_effective == pytest.approx(ALPHA)


# ---------------------------------------------------------------------------
# expand_weight_matrix — R5 + A4
# ---------------------------------------------------------------------------

class TestExpandWeightMatrix:
    def test_shape_grows_by_one_column(self):
        s = make_state()
        s.expand_weight_matrix("new_dim")
        assert s.W.shape == (4, 7)
        assert s.n_factors == 7

    def test_factor_names_updated(self):
        s = make_state()
        s.expand_weight_matrix("new_dim")
        assert "new_dim" in s.factor_names
        assert len(s.factor_names) == 7

    def test_epsilon_vector_grows(self):
        s = make_state()
        s.expand_weight_matrix("new_dim")
        assert s.epsilon_vector.shape == (7,)

    def test_new_dimension_is_provisional(self):
        s = make_state()
        s.expand_weight_matrix("new_dim")
        assert len(s.dimension_metadata) == 1
        assert s.dimension_metadata[0].state == "provisional"
        assert s.dimension_metadata[0].factor_name == "new_dim"

    def test_provisional_decay_is_faster(self):
        s = make_state()
        s.expand_weight_matrix("new_dim")
        # New column's ε should be 10× standard
        assert s.epsilon_vector[-1] > EPSILON

    def test_col_index_stored_correctly(self):
        s = make_state()
        s.expand_weight_matrix("new_dim")
        assert s.dimension_metadata[0].col_index == 6  # was n_factors before expand

    def test_expansion_history_recorded(self):
        s = make_state()
        s.expand_weight_matrix("new_dim")
        assert len(s.expansion_history) == 1
        assert s.expansion_history[0]["new_factor"] == "new_dim"

    def test_duplicate_factor_name_raises(self):
        s = make_state()
        with pytest.raises(AssertionError):
            s.expand_weight_matrix("travel")  # already exists

    def test_existing_columns_unchanged(self):
        s = make_state()
        W_orig = s.W.copy()
        s.expand_weight_matrix("new_dim")
        np.testing.assert_array_equal(s.W[:, :6], W_orig)


# ---------------------------------------------------------------------------
# Task verification smoke test
# ---------------------------------------------------------------------------

def test_task_verification_smoke():
    """Exact replication of the verification script from the task description."""
    W = np.array([
        [ 0.3, -0.1, -0.25, -0.15,  0.2,  0.25],
        [ 0.05,  0.2,  0.15,  0.10, -0.05,  0.05],
        [-0.1,  0.05,  0.20,  0.20, -0.1, -0.05],
        [-0.25,  0.3,  0.30,  0.15, -0.2, -0.15],
    ])
    f = np.array([[0.9, 0.3, 0.1, 0.2, 0.8, 0.7]])
    names = ["travel", "asset", "threat", "time", "device", "pattern"]

    state = LearningState(W=W.copy(), n_actions=4, n_factors=6, factor_names=names,
                          profile=CalibrationProfile())
    u1 = state.update(0, "fp_close", +1, f)
    u2 = state.update(0, "fp_close", -1, f)
    ratio = np.linalg.norm(u2.delta_applied) / np.linalg.norm(u1.delta_applied)
    assert 19 < ratio < 21, f"Asymmetry ratio {ratio:.1f} not ~20×"

    state.expand_weight_matrix("new_dim")
    assert state.W.shape == (4, 7)


# ---------------------------------------------------------------------------
# GAE-PROF-3: ProfileScorer delegation tests
# ---------------------------------------------------------------------------

def test_learning_state_profile_scorer_field_defaults_none():
    """profile_scorer defaults to None; is_profile_mode is False."""
    state = make_state()
    assert state.profile_scorer is None
    assert state.is_profile_mode is False


def test_attach_profile_scorer_sets_profile_mode():
    """attach_profile_scorer() wires scorer and sets is_profile_mode True."""
    from gae.profile_scorer import ProfileScorer
    state = make_state()
    mu = np.full((2, 3, 4), 0.5)
    scorer = ProfileScorer(mu=mu, actions=["a0", "a1", "a2"])
    state.attach_profile_scorer(scorer)
    assert state.is_profile_mode is True
    assert state.profile_scorer is scorer


def test_update_delegates_to_profile_scorer_when_attached():
    """When profile_scorer is attached, update() calls scorer.update()
    and the scorer's counts increment."""
    from gae.profile_scorer import ProfileScorer
    mu = np.full((2, 3, 4), 0.5)
    scorer = ProfileScorer(mu=mu, actions=["a0", "a1", "a2"])

    state = LearningState(
        W=np.zeros((3, 4)),
        n_actions=3,
        n_factors=4,
        factor_names=["f0", "f1", "f2", "f3"],
        profile=CalibrationProfile(),
    )
    state.attach_profile_scorer(scorer)

    assert scorer.counts[0, 0] == 0

    f = np.ones((1, 4)) * 0.5
    state.update(
        action_index=0,
        action_name="a0",
        outcome=+1,
        f=f,
        category_index=0,
    )

    assert scorer.counts[0, 0] == 1, (
        "ProfileScorer.update() should have been called, incrementing count"
    )


def test_legacy_path_unchanged_without_profile_scorer():
    """Without a profile_scorer attached, update() behaves exactly as before."""
    state = make_state()
    assert state.is_profile_mode is False
    u = state.update(0, "fp_close", +1, F)
    assert isinstance(u, WeightUpdate)
    assert state.decision_count == 1
