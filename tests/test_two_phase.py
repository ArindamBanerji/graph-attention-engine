"""Unit tests for gae.two_phase."""

from gae.two_phase import (
    CategoryState,
    DecisionCountPolicy,
    MEAN_CONVERGENCE,
    ManualPolicy,
    RollingAccuracyDeltaPolicy,
    VARIANCE_LEARNING,
)


def test_category_state_record_decision_increments():
    state = CategoryState()
    state.record_decision()
    assert state.n_decisions == 1
    assert state.phase == MEAN_CONVERGENCE


def test_category_state_freeze_sets_phase_and_freeze_point():
    state = CategoryState()
    state.record_decision()
    state.record_decision()
    state.freeze()
    assert state.phase == VARIANCE_LEARNING
    assert state.freeze_point == 2


def test_category_state_freeze_idempotent():
    state = CategoryState(n_decisions=3)
    state.freeze()
    state.record_decision()
    state.freeze()
    assert state.phase == VARIANCE_LEARNING
    assert state.freeze_point == 3


def test_decision_count_policy_freezes_at_threshold():
    policy = DecisionCountPolicy(n=2)
    state = CategoryState(n_decisions=1)
    assert policy.should_freeze(state) is False
    state.record_decision()
    assert policy.should_freeze(state) is True


def test_manual_policy_never_freezes():
    policy = ManualPolicy()
    state = CategoryState(n_decisions=10, phase=VARIANCE_LEARNING, freeze_point=5)
    assert policy.should_freeze(state) is False


def test_rolling_accuracy_delta_policy_is_inert_placeholder():
    policy = RollingAccuracyDeltaPolicy(threshold_pp=0.25)
    state = CategoryState(n_decisions=500)
    assert policy.threshold_pp == 0.25
    assert policy.should_freeze(state) is False
