"""FW-05 Phase A gate tests."""

from __future__ import annotations

import numpy as np
import pytest

from gae.calibration import CalibrationProfile
from gae.learning import LearningState, WeightUpdate
from gae.profile_scorer import CentroidUpdate, ProfileScorer, ScoringResult
from gae.shrinkage import FixedAlpha
from gae.two_phase import DecisionCountPolicy, MEAN_CONVERGENCE, VARIANCE_LEARNING


class RecordingEstimator:
    """Deterministic estimator used to inspect reestimate_dk() inputs."""

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.calls: list[tuple[list[tuple[np.ndarray, int, int]], np.ndarray, int, int]] = []

    def estimate(self, decisions, centroids, n_categories, n_dims) -> np.ndarray:
        copied_decisions = [
            (np.array(factor_vector, copy=True), category_index, action_index)
            for factor_vector, category_index, action_index in decisions
        ]
        self.calls.append(
            (
                copied_decisions,
                np.array(centroids, copy=True),
                n_categories,
                n_dims,
            )
        )
        return self.weights.copy()


def make_mu_two_phase() -> np.ndarray:
    return np.array(
        [
            [
                [0.10, 0.90],
                [0.60, 0.00],
                [0.85, 0.25],
                [0.20, 0.30],
            ],
            [
                [0.20, 0.20],
                [0.85, 0.80],
                [0.15, 0.75],
                [0.70, 0.15],
            ],
        ],
        dtype=np.float64,
    )


def make_mu_binary() -> np.ndarray:
    return np.array(
        [
            [
                [0.10, 0.90],
                [0.60, 0.00],
            ]
        ],
        dtype=np.float64,
    )


def make_mu_single_action() -> np.ndarray:
    return np.array([[[0.25, 0.75, 0.40]]], dtype=np.float64)


def make_actions(n_actions: int) -> list[str]:
    return [f"a{i}" for i in range(n_actions)]


def make_two_phase_scorer(
    *,
    mu: np.ndarray | None = None,
    actions: list[str] | None = None,
    freeze_after: int = 5,
    alpha: float = 0.5,
    estimator=None,
) -> ProfileScorer:
    if mu is None:
        mu = make_mu_two_phase()
    if actions is None:
        actions = make_actions(mu.shape[1])
    return ProfileScorer.for_soc_twophase(
        mu=mu,
        actions=actions,
        phase_policy=DecisionCountPolicy(n=freeze_after),
        dk_estimator=estimator,
        shrinkage_schedule=FixedAlpha(alpha),
    )


def make_profile_learning_state(
    *,
    scorer: ProfileScorer,
    n_actions: int,
    n_factors: int,
) -> LearningState:
    state = LearningState(
        W=np.zeros((n_actions, n_factors), dtype=np.float64),
        n_actions=n_actions,
        n_factors=n_factors,
        factor_names=[f"f{i}" for i in range(n_factors)],
        profile=CalibrationProfile(),
    )
    state.attach_profile_scorer(scorer)
    return state


def drive_to_phase2(
    scorer: ProfileScorer,
    *,
    category_index: int = 0,
    action_index: int = 0,
    n_updates: int,
) -> None:
    f = scorer.mu[category_index, action_index].copy()
    for _ in range(n_updates):
        scorer.update(
            f=f,
            category_index=category_index,
            action_index=action_index,
            correct=True,
        )


# ---------------------------------------------------------------------------
# Section 1 - Factory and construction
# ---------------------------------------------------------------------------


def test_fw05_factory_uses_default_soc_actions():
    scorer = ProfileScorer.for_soc_twophase(mu=np.full((2, 4, 3), 0.5, dtype=np.float64))
    assert scorer.actions == ["escalate", "investigate", "suppress", "monitor"]


def test_fw05_factory_accepts_custom_actions_list():
    actions = ["allow", "review", "deny", "defer"]
    scorer = make_two_phase_scorer(actions=actions)
    assert scorer.actions == actions


def test_fw05_factory_installs_requested_phase_policy():
    scorer = make_two_phase_scorer(freeze_after=10)
    assert scorer._learning_strategy is not None
    assert isinstance(scorer._learning_strategy.phase_policy, DecisionCountPolicy)
    assert scorer._learning_strategy.phase_policy.n == 10


def test_fw05_factory_initializes_all_categories_in_phase1():
    scorer = make_two_phase_scorer()
    assert scorer.get_phase(0) == MEAN_CONVERGENCE
    assert scorer.get_phase(1) == MEAN_CONVERGENCE


def test_fw05_factory_exposes_fixed_alpha_before_freeze():
    scorer = make_two_phase_scorer(alpha=0.35)
    assert scorer.get_alpha(0) == pytest.approx(0.35)
    assert scorer.get_alpha(1) == pytest.approx(0.35)


def test_fw05_factory_has_no_dk_weights_initially():
    scorer = make_two_phase_scorer()
    assert scorer.get_dk_weights(0) is None
    assert scorer.get_dk_weights(1) is None


def test_fw05_score_returns_scoringresult_with_expected_shapes():
    scorer = make_two_phase_scorer()
    f = np.array([0.12, 0.88], dtype=np.float64)
    result = scorer.score(f, category_index=0)
    assert isinstance(result, ScoringResult)
    assert result.probabilities.shape == (4,)
    assert result.distances.shape == (4,)
    assert result.action_name == scorer.actions[result.action_index]
    assert result.confidence == pytest.approx(float(result.probabilities[result.action_index]))


# ---------------------------------------------------------------------------
# Section 2 - Phase lifecycle and buffering
# ---------------------------------------------------------------------------


def test_fw05_phase_stays_in_mean_convergence_before_threshold():
    scorer = make_two_phase_scorer(freeze_after=5)
    drive_to_phase2(scorer, n_updates=4)
    assert scorer.get_phase(0) == MEAN_CONVERGENCE


def test_fw05_phase_transitions_at_threshold():
    scorer = make_two_phase_scorer(freeze_after=5)
    drive_to_phase2(scorer, n_updates=5)
    assert scorer.get_phase(0) == VARIANCE_LEARNING


def test_fw05_phase_transition_is_category_local():
    scorer = make_two_phase_scorer(freeze_after=5)
    drive_to_phase2(scorer, category_index=0, n_updates=5)
    assert scorer.get_phase(0) == VARIANCE_LEARNING
    assert scorer.get_phase(1) == MEAN_CONVERGENCE


def test_fw05_phase2_update_buffers_correct_decisions():
    scorer = make_two_phase_scorer(freeze_after=5)
    drive_to_phase2(scorer, n_updates=5)
    result = scorer.update(
        f=np.array([0.11, 0.89], dtype=np.float64),
        category_index=0,
        action_index=0,
        correct=True,
    )
    assert isinstance(result, CentroidUpdate)
    assert scorer._decision_buffer is not None
    assert len(scorer._decision_buffer) == 1
    buffered_f, buffered_c, buffered_a, buffered_correct = scorer._decision_buffer[0]
    np.testing.assert_allclose(buffered_f, np.array([0.11, 0.89], dtype=np.float64))
    assert (buffered_c, buffered_a, buffered_correct) == (0, 0, True)


def test_fw05_phase2_update_buffers_incorrect_decisions():
    scorer = make_two_phase_scorer(freeze_after=5)
    drive_to_phase2(scorer, n_updates=5)
    scorer.update(
        f=np.array([0.58, 0.03], dtype=np.float64),
        category_index=0,
        action_index=1,
        correct=False,
        gt_action_index=0,
    )
    assert scorer._decision_buffer is not None
    assert len(scorer._decision_buffer) == 1
    _, buffered_c, buffered_a, buffered_correct = scorer._decision_buffer[0]
    assert (buffered_c, buffered_a, buffered_correct) == (0, 1, False)


def test_fw05_phase2_buffer_growth_matches_post_freeze_updates():
    scorer = make_two_phase_scorer(freeze_after=5)
    drive_to_phase2(scorer, n_updates=5)
    for action_index in (0, 1, 0):
        scorer.update(
            f=scorer.mu[0, action_index].copy(),
            category_index=0,
            action_index=action_index,
            correct=True,
        )
    assert scorer._decision_buffer is not None
    assert len(scorer._decision_buffer) == 3


def test_fw05_phase2_updates_do_not_mutate_centroids():
    scorer = make_two_phase_scorer(freeze_after=5)
    drive_to_phase2(scorer, n_updates=5)
    before = scorer.centroids.copy()
    scorer.update(
        f=np.array([0.59, 0.02], dtype=np.float64),
        category_index=0,
        action_index=1,
        correct=True,
    )
    np.testing.assert_allclose(scorer.centroids, before)


def test_fw05_phase2_updates_keep_return_type_centroidupdate():
    scorer = make_two_phase_scorer(freeze_after=5)
    drive_to_phase2(scorer, n_updates=5)
    result = scorer.update(
        f=np.array([0.59, 0.02], dtype=np.float64),
        category_index=0,
        action_index=1,
        correct=True,
    )
    assert isinstance(result, CentroidUpdate)
    assert result.outcome == "phase2_buffered"


def test_fw05_phase2_updates_continue_decision_count():
    scorer = make_two_phase_scorer(freeze_after=5)
    drive_to_phase2(scorer, n_updates=5)
    result = scorer.update(
        f=np.array([0.59, 0.02], dtype=np.float64),
        category_index=0,
        action_index=1,
        correct=True,
    )
    assert result.decision_count == 6
    assert scorer.decision_count == 6


# ---------------------------------------------------------------------------
# Section 3 - DK re-estimation and alpha behavior
# ---------------------------------------------------------------------------


def test_fw05_reestimate_dk_noop_with_fewer_than_two_correct_decisions():
    estimator = RecordingEstimator(weights=np.array([[3.0, 1.0]], dtype=np.float64))
    scorer = make_two_phase_scorer(mu=make_mu_binary(), estimator=estimator)
    scorer._decision_buffer = [(np.array([0.10, 0.90]), 0, 0, True)]
    scorer.reestimate_dk()
    assert scorer.get_dk_weights(0) is None
    assert estimator.calls == []


def test_fw05_reestimate_dk_filters_incorrect_buffered_decisions():
    estimator = RecordingEstimator(weights=np.array([[4.0, 0.2]], dtype=np.float64))
    scorer = make_two_phase_scorer(mu=make_mu_binary(), estimator=estimator)
    scorer._decision_buffer = [
        (np.array([0.10, 0.90]), 0, 0, True),
        (np.array([0.60, 0.00]), 0, 1, False),
        (np.array([0.12, 0.88]), 0, 0, True),
    ]
    scorer.reestimate_dk()
    recorded_decisions = estimator.calls[0][0]
    assert len(recorded_decisions) == 2
    assert all(decision[1:] == (0, 0) for decision in recorded_decisions)


def test_fw05_reestimate_dk_passes_expected_shape_arguments():
    estimator = RecordingEstimator(weights=np.array([[4.0, 0.2], [1.0, 1.0]], dtype=np.float64))
    scorer = make_two_phase_scorer(mu=make_mu_two_phase(), estimator=estimator)
    scorer._decision_buffer = [
        (np.array([0.10, 0.90]), 0, 0, True),
        (np.array([0.12, 0.88]), 0, 0, True),
        (np.array([0.82, 0.78]), 1, 1, True),
    ]
    scorer.reestimate_dk()
    _, _, n_categories, n_dims = estimator.calls[0]
    assert n_categories == 2
    assert n_dims == 2


def test_fw05_reestimate_dk_stores_estimated_weights():
    expected = np.array([[4.0, 0.2], [1.2, 2.5]], dtype=np.float64)
    estimator = RecordingEstimator(weights=expected)
    scorer = make_two_phase_scorer(mu=make_mu_two_phase(), estimator=estimator)
    scorer._decision_buffer = [
        (np.array([0.10, 0.90]), 0, 0, True),
        (np.array([0.12, 0.88]), 0, 0, True),
        (np.array([0.82, 0.78]), 1, 1, True),
    ]
    scorer.reestimate_dk()
    assert scorer._dk_weights is not None
    np.testing.assert_allclose(scorer._dk_weights, expected)


def test_fw05_get_dk_weights_returns_copy_after_reestimate():
    expected = np.array([[4.0, 0.2], [1.2, 2.5]], dtype=np.float64)
    estimator = RecordingEstimator(weights=expected)
    scorer = make_two_phase_scorer(mu=make_mu_two_phase(), estimator=estimator)
    scorer._decision_buffer = [
        (np.array([0.10, 0.90]), 0, 0, True),
        (np.array([0.12, 0.88]), 0, 0, True),
    ]
    scorer.reestimate_dk()
    weights = scorer.get_dk_weights(0)
    assert weights is not None
    weights[0] = 999.0
    assert scorer._dk_weights is not None
    assert scorer._dk_weights[0, 0] == pytest.approx(4.0)


def test_fw05_score_ignores_dk_weights_before_phase2():
    scorer = make_two_phase_scorer(mu=make_mu_binary(), alpha=1.0)
    scorer._dk_weights = np.array([[10.0, 0.1]], dtype=np.float64)
    f = np.array([0.50, 0.90], dtype=np.float64)
    result = scorer.score(f, 0)
    assert result.action_index == 0


def test_fw05_score_uses_dk_weights_after_phase2():
    scorer = make_two_phase_scorer(mu=make_mu_binary(), alpha=1.0, freeze_after=5)
    drive_to_phase2(scorer, n_updates=5)
    scorer._dk_weights = np.array([[10.0, 0.1]], dtype=np.float64)
    f = np.array([0.50, 0.90], dtype=np.float64)
    result = scorer.score(f, 0)
    assert result.action_index == 1


def test_fw05_action_probability_changes_monotonically_with_alpha():
    f = np.array([0.50, 0.90], dtype=np.float64)
    weights = np.array([[10.0, 0.1]], dtype=np.float64)

    scorer_alpha0 = make_two_phase_scorer(mu=make_mu_binary(), alpha=0.0)
    scorer_alpha05 = make_two_phase_scorer(mu=make_mu_binary(), alpha=0.5)
    scorer_alpha1 = make_two_phase_scorer(mu=make_mu_binary(), alpha=1.0)

    for scorer in (scorer_alpha0, scorer_alpha05, scorer_alpha1):
        drive_to_phase2(scorer, n_updates=5)
        scorer._dk_weights = weights.copy()

    p_action1_alpha0 = scorer_alpha0.score(f, 0).probabilities[1]
    p_action1_alpha05 = scorer_alpha05.score(f, 0).probabilities[1]
    p_action1_alpha1 = scorer_alpha1.score(f, 0).probabilities[1]

    assert p_action1_alpha0 < p_action1_alpha05 < p_action1_alpha1


# ---------------------------------------------------------------------------
# Section 4 - Score edge cases and accessors
# ---------------------------------------------------------------------------


def test_fw05_single_action_score_returns_certain_probability_in_phase1():
    scorer = make_two_phase_scorer(mu=make_mu_single_action(), actions=["only"])
    result = scorer.score(np.array([0.10, 0.80, 0.30], dtype=np.float64), 0)
    assert result.action_index == 0
    assert result.action_name == "only"
    np.testing.assert_allclose(result.probabilities, np.array([1.0], dtype=np.float64))
    assert result.confidence == pytest.approx(1.0)
    assert result.confidence_gap == pytest.approx(0.0)


def test_fw05_single_action_score_returns_certain_probability_in_phase2():
    scorer = make_two_phase_scorer(
        mu=make_mu_single_action(),
        actions=["only"],
        alpha=1.0,
        freeze_after=5,
    )
    drive_to_phase2(scorer, n_updates=5)
    scorer._dk_weights = np.array([[2.5, 0.7, 1.2]], dtype=np.float64)
    result = scorer.score(np.array([0.10, 0.80, 0.30], dtype=np.float64), 0)
    np.testing.assert_allclose(result.probabilities, np.array([1.0], dtype=np.float64))
    assert result.action_index == 0


def test_fw05_score_invalid_category_negative_raises_assertionerror():
    scorer = make_two_phase_scorer(mu=make_mu_binary())
    with pytest.raises(AssertionError):
        scorer.score(np.array([0.50, 0.90], dtype=np.float64), -1)


def test_fw05_score_invalid_category_too_large_raises_assertionerror():
    scorer = make_two_phase_scorer(mu=make_mu_binary())
    with pytest.raises(AssertionError):
        scorer.score(np.array([0.50, 0.90], dtype=np.float64), 1)


def test_fw05_phase2_probabilities_still_sum_to_one():
    scorer = make_two_phase_scorer(mu=make_mu_binary(), alpha=1.0, freeze_after=5)
    drive_to_phase2(scorer, n_updates=5)
    scorer._dk_weights = np.array([[10.0, 0.1]], dtype=np.float64)
    result = scorer.score(np.array([0.50, 0.90], dtype=np.float64), 0)
    assert result.probabilities.shape == (2,)
    assert float(result.probabilities.sum()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Section 5 - LearningState delegation
# ---------------------------------------------------------------------------


def test_fw05_learning_state_profile_mode_returns_weightupdate():
    scorer = make_two_phase_scorer(mu=make_mu_binary(), actions=["left", "right"])
    state = make_profile_learning_state(scorer=scorer, n_actions=2, n_factors=2)
    result = state.update(
        action_index=0,
        action_name="left",
        outcome=+1,
        f=np.array([[0.10, 0.90]], dtype=np.float64),
        category_index=0,
    )
    assert isinstance(result, WeightUpdate)


def test_fw05_learning_state_profile_mode_populates_centroid_update():
    scorer = make_two_phase_scorer(mu=make_mu_binary(), actions=["left", "right"])
    state = make_profile_learning_state(scorer=scorer, n_actions=2, n_factors=2)
    result = state.update(
        action_index=0,
        action_name="left",
        outcome=+1,
        f=np.array([[0.10, 0.90]], dtype=np.float64),
        category_index=0,
    )
    assert result is not None
    assert isinstance(result.centroid_update, CentroidUpdate)


def test_fw05_learning_state_profile_mode_drives_scorer_phase_transition():
    scorer = make_two_phase_scorer(mu=make_mu_binary(), actions=["left", "right"], freeze_after=5)
    state = make_profile_learning_state(scorer=scorer, n_actions=2, n_factors=2)
    for _ in range(5):
        state.update(
            action_index=0,
            action_name="left",
            outcome=+1,
            f=np.array([[0.10, 0.90]], dtype=np.float64),
            category_index=0,
        )
    assert scorer.get_phase(0) == VARIANCE_LEARNING


def test_fw05_learning_state_profile_mode_buffers_after_freeze():
    scorer = make_two_phase_scorer(mu=make_mu_binary(), actions=["left", "right"], freeze_after=5)
    state = make_profile_learning_state(scorer=scorer, n_actions=2, n_factors=2)
    for _ in range(5):
        state.update(
            action_index=0,
            action_name="left",
            outcome=+1,
            f=np.array([[0.10, 0.90]], dtype=np.float64),
            category_index=0,
        )
    state.update(
        action_index=0,
        action_name="left",
        outcome=+1,
        f=np.array([[0.11, 0.89]], dtype=np.float64),
        category_index=0,
    )
    assert scorer._decision_buffer is not None
    assert len(scorer._decision_buffer) == 1


def test_fw05_learning_state_autonomous_returns_none_and_leaves_scorer_untouched():
    scorer = make_two_phase_scorer(mu=make_mu_binary(), actions=["left", "right"])
    state = make_profile_learning_state(scorer=scorer, n_actions=2, n_factors=2)
    result = state.update(
        action_index=0,
        action_name="left",
        outcome=+1,
        f=np.array([[0.10, 0.90]], dtype=np.float64),
        decision_source="autonomous",
        category_index=0,
    )
    assert result is None
    assert scorer.decision_count == 0
    assert len(state.pending_validations) == 1
