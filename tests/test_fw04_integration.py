from __future__ import annotations

import numpy as np
import pytest

from gae import LearningStrategy
from gae.dk_estimator import CoordinateDescentEstimator
from gae.profile_scorer import CentroidUpdate, ProfileScorer
from gae.shrinkage import FixedAlpha
from gae.two_phase import DecisionCountPolicy, MEAN_CONVERGENCE, VARIANCE_LEARNING


class RecordingEstimator:
    def __init__(self, weights: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.calls = []

    def estimate(self, decisions, centroids, n_categories, n_dims) -> np.ndarray:
        self.calls.append((list(decisions), np.array(centroids, copy=True), n_categories, n_dims))
        return self.weights.copy()


def _make_strategy(
    *,
    freeze_after: int = 999,
    estimator=None,
    alpha: float = 0.5,
) -> LearningStrategy:
    if estimator is None:
        estimator = CoordinateDescentEstimator()
    return LearningStrategy(
        phase_policy=DecisionCountPolicy(n=freeze_after),
        dk_estimator=estimator,
        shrinkage_schedule=FixedAlpha(alpha),
    )


def _make_mu() -> np.ndarray:
    return np.array(
        [
            [[0.10, 0.90], [0.60, 0.00]],
            [[0.25, 0.25], [0.75, 0.75]],
        ],
        dtype=np.float64,
    )


def _make_actions() -> list[str]:
    return ["a0", "a1"]


def test_learning_strategy_positional_misuse_raises_typeerror():
    mu = _make_mu()
    actions = _make_actions()
    strategy = _make_strategy()
    with pytest.raises(TypeError):
        ProfileScorer(mu, actions, LearningStrategy(
            phase_policy=strategy.phase_policy,
            dk_estimator=strategy.dk_estimator,
            shrinkage_schedule=strategy.shrinkage_schedule,
        ))


def test_learning_strategy_keyword_argument_works():
    scorer = ProfileScorer(_make_mu(), _make_actions(), learning_strategy=_make_strategy())
    assert scorer.get_phase(0) == MEAN_CONVERGENCE


def test_learning_strategy_initializes_internal_state():
    scorer = ProfileScorer(_make_mu(), _make_actions(), learning_strategy=_make_strategy())
    assert scorer._category_states is not None
    assert len(scorer._category_states) == scorer.centroids.shape[0]
    assert scorer._dk_weights is None
    assert scorer._decision_buffer == []


def test_get_phase_defaults_to_mean_convergence_without_strategy():
    scorer = ProfileScorer(_make_mu(), _make_actions())
    assert scorer.get_phase(1) == MEAN_CONVERGENCE


def test_get_alpha_defaults_to_zero_without_strategy():
    scorer = ProfileScorer(_make_mu(), _make_actions())
    assert scorer.get_alpha(0) == 0.0


def test_get_dk_weights_returns_copy():
    scorer = ProfileScorer(_make_mu(), _make_actions(), learning_strategy=_make_strategy())
    scorer._dk_weights = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    weights = scorer.get_dk_weights(1)
    assert weights is not None
    np.testing.assert_array_equal(weights, np.array([3.0, 4.0]))
    weights[0] = 999.0
    assert scorer._dk_weights[1, 0] == 3.0


def test_phase1_update_freezes_category_when_policy_triggers():
    scorer = ProfileScorer(
        _make_mu(),
        _make_actions(),
        learning_strategy=_make_strategy(freeze_after=1),
    )
    scorer.update(np.array([0.15, 0.85]), category_index=0, action_index=0, correct=True)
    assert scorer.get_phase(0) == VARIANCE_LEARNING
    assert scorer._category_states[0].freeze_point == 1


def test_phase2_update_buffers_without_mutating_centroids():
    scorer = ProfileScorer(
        _make_mu(),
        _make_actions(),
        learning_strategy=_make_strategy(freeze_after=1),
    )
    scorer.update(np.array([0.15, 0.85]), category_index=0, action_index=0, correct=True)
    before = scorer.centroids.copy()
    result = scorer.update(np.array([0.20, 0.80]), category_index=0, action_index=0, correct=True)
    np.testing.assert_array_equal(scorer.centroids, before)
    assert scorer._decision_buffer is not None
    assert len(scorer._decision_buffer) == 1
    buffered_f, buffered_c, buffered_a, buffered_correct = scorer._decision_buffer[0]
    np.testing.assert_array_equal(buffered_f, np.array([0.20, 0.80]))
    assert (buffered_c, buffered_a, buffered_correct) == (0, 0, True)
    assert result.outcome == "phase2_buffered"


def test_phase2_update_return_matches_phase1():
    scorer_phase1 = ProfileScorer(_make_mu(), _make_actions())
    scorer_phase2 = ProfileScorer(
        _make_mu(),
        _make_actions(),
        learning_strategy=_make_strategy(freeze_after=1),
    )
    phase1_result = scorer_phase1.update(
        np.array([0.15, 0.85]), category_index=0, action_index=0, correct=True
    )
    scorer_phase2.update(np.array([0.15, 0.85]), category_index=0, action_index=0, correct=True)
    phase2_result = scorer_phase2.update(
        np.array([0.20, 0.80]), category_index=0, action_index=0, correct=True
    )
    assert isinstance(phase1_result, CentroidUpdate)
    assert isinstance(phase2_result, type(phase1_result))


def test_reestimate_dk_is_noop_without_strategy_or_buffer():
    scorer_plain = ProfileScorer(_make_mu(), _make_actions())
    scorer_plain.reestimate_dk()
    assert scorer_plain.get_dk_weights(0) is None

    scorer_twophase = ProfileScorer(
        _make_mu(),
        _make_actions(),
        learning_strategy=_make_strategy(),
    )
    scorer_twophase.reestimate_dk()
    assert scorer_twophase.get_dk_weights(0) is None


def test_reestimate_dk_requires_at_least_two_correct_decisions():
    scorer = ProfileScorer(
        _make_mu(),
        _make_actions(),
        learning_strategy=_make_strategy(),
    )
    scorer._decision_buffer = [(np.array([0.20, 0.80]), 0, 0, True)]
    scorer.reestimate_dk()
    assert scorer.get_dk_weights(0) is None


def test_reestimate_dk_uses_only_correct_buffered_decisions():
    estimator = RecordingEstimator(weights=np.array([[5.0, 1.0], [1.0, 5.0]]))
    scorer = ProfileScorer(
        _make_mu(),
        _make_actions(),
        learning_strategy=_make_strategy(estimator=estimator),
    )
    scorer._decision_buffer = [
        (np.array([0.20, 0.80]), 0, 0, True),
        (np.array([0.55, 0.10]), 0, 1, False),
        (np.array([0.30, 0.70]), 1, 1, True),
    ]
    scorer.reestimate_dk()
    np.testing.assert_array_equal(
        scorer._dk_weights,
        np.array([[5.0, 1.0], [1.0, 5.0]], dtype=np.float64),
    )
    recorded_decisions, _, n_categories, n_dims = estimator.calls[0]
    assert len(recorded_decisions) == 2
    assert recorded_decisions[0][1:] == (0, 0)
    assert recorded_decisions[1][1:] == (1, 1)
    assert n_categories == 2
    assert n_dims == 2


def test_phase2_score_override_uses_dk_weights():
    strategy = _make_strategy(alpha=1.0)
    scorer = ProfileScorer(_make_mu(), _make_actions(), learning_strategy=strategy)
    baseline = ProfileScorer(_make_mu(), _make_actions())

    scorer._category_states[0].freeze()
    scorer._dk_weights = np.array([[10.0, 0.1], [1.0, 1.0]], dtype=np.float64)

    f = np.array([0.50, 0.90], dtype=np.float64)
    baseline_result = baseline.score(f, 0)
    phase2_result = scorer.score(f, 0)

    assert baseline_result.action_index == 0
    assert phase2_result.action_index == 1


def test_for_soc_twophase_installs_expected_defaults():
    mu = np.full((2, 4, 3), 0.5, dtype=np.float64)
    scorer = ProfileScorer.for_soc_twophase(mu=mu)
    assert scorer.auto_pause_on_amber is True
    assert scorer.eta_override == pytest.approx(0.01)
    assert isinstance(scorer._learning_strategy.phase_policy, DecisionCountPolicy)
    assert scorer._learning_strategy.phase_policy.n == 200
    assert isinstance(scorer._learning_strategy.dk_estimator, CoordinateDescentEstimator)
    assert isinstance(scorer._learning_strategy.shrinkage_schedule, FixedAlpha)
    assert scorer._learning_strategy.shrinkage_schedule.alpha == pytest.approx(0.5)


def test_setstate_backfills_new_twophase_attributes():
    scorer = ProfileScorer(_make_mu(), _make_actions())
    state = scorer.__dict__.copy()
    for attr in ("_learning_strategy", "_category_states", "_dk_weights", "_decision_buffer"):
        state.pop(attr, None)

    restored = ProfileScorer.__new__(ProfileScorer)
    restored.__setstate__(state)

    assert restored._learning_strategy is None
    assert restored._category_states is None
    assert restored._dk_weights is None
    assert restored._decision_buffer is None
