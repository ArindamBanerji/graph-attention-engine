from __future__ import annotations

import pickle

import numpy as np

from gae.profile_scorer import LearningStrategy, ProfileScorer
from gae.shrinkage import FixedAlpha
from gae.two_phase import DecisionCountPolicy, MEAN_CONVERGENCE, VARIANCE_LEARNING


class FixedEstimator:
    def __init__(self, weights: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)

    def estimate(self, decisions, centroids, n_categories, n_dims) -> np.ndarray:
        return self.weights.copy()


def make_mu() -> np.ndarray:
    return np.array(
        [
            [[0.10, 0.90], [0.80, 0.20]],
            [[0.25, 0.30], [0.70, 0.75]],
        ],
        dtype=np.float64,
    )


def make_actions() -> list[str]:
    return ["a0", "a1"]


def make_strategy(
    *,
    freeze_after: int = 3,
    weights: np.ndarray | None = None,
) -> LearningStrategy:
    if weights is None:
        weights = np.array([[2.0, 0.5], [0.75, 1.5]], dtype=np.float64)
    return LearningStrategy(
        phase_policy=DecisionCountPolicy(n=freeze_after),
        dk_estimator=FixedEstimator(weights),
        shrinkage_schedule=FixedAlpha(1.0),
    )


def make_two_phase_scorer(*, freeze_after: int = 3) -> ProfileScorer:
    return ProfileScorer(
        make_mu(),
        make_actions(),
        learning_strategy=make_strategy(freeze_after=freeze_after),
    )


def drive_updates(
    scorer: ProfileScorer,
    *,
    category_index: int,
    action_index: int,
    n_updates: int,
) -> None:
    f = scorer.centroids[category_index, action_index].copy()
    for _ in range(n_updates):
        scorer.update(f, category_index=category_index, action_index=action_index, correct=True)


def drive_phase2_with_weights(scorer: ProfileScorer) -> None:
    drive_updates(scorer, category_index=0, action_index=0, n_updates=1)
    scorer.update(
        np.array([0.11, 0.89], dtype=np.float64),
        category_index=0,
        action_index=0,
        correct=True,
    )
    scorer.update(
        np.array([0.79, 0.21], dtype=np.float64),
        category_index=0,
        action_index=1,
        correct=True,
    )
    scorer.reestimate_dk()


def test_checkpoint_default_scorer():
    scorer = ProfileScorer(make_mu(), make_actions())

    checkpoint = scorer.get_checkpoint_state()

    np.testing.assert_allclose(checkpoint["centroids"], scorer.centroids)
    assert checkpoint["category_phases"] is None
    assert checkpoint["dk_weights"] is None
    assert checkpoint["decision_buffer_size"] == 0
    assert checkpoint["freeze_points"] is None
    assert checkpoint["decision_counts"] is None

    checkpoint["centroids"][0, 0, 0] = 0.99
    assert scorer.centroids[0, 0, 0] != 0.99


def test_checkpoint_phase1_state():
    scorer = make_two_phase_scorer(freeze_after=5)
    drive_updates(scorer, category_index=0, action_index=0, n_updates=2)
    drive_updates(scorer, category_index=1, action_index=1, n_updates=1)

    checkpoint = scorer.get_checkpoint_state()

    assert checkpoint["category_phases"] == [MEAN_CONVERGENCE, MEAN_CONVERGENCE]
    assert checkpoint["freeze_points"] == [None, None]
    assert checkpoint["decision_counts"] == [2, 1]
    assert checkpoint["decision_buffer_size"] == 0


def test_checkpoint_phase2_state():
    scorer = make_two_phase_scorer(freeze_after=1)
    drive_phase2_with_weights(scorer)

    checkpoint = scorer.get_checkpoint_state()

    assert checkpoint["category_phases"] == [VARIANCE_LEARNING, MEAN_CONVERGENCE]
    assert checkpoint["freeze_points"] == [1, None]
    assert checkpoint["decision_counts"] == [3, 0]
    assert checkpoint["decision_buffer_size"] == 2
    assert checkpoint["dk_weights"] is not None
    np.testing.assert_allclose(checkpoint["dk_weights"], scorer._dk_weights)
    checkpoint["dk_weights"][0, 0] = 99.0
    assert scorer._dk_weights[0, 0] != 99.0


def test_checkpoint_roundtrip():
    original = make_two_phase_scorer(freeze_after=1)
    drive_phase2_with_weights(original)
    checkpoint = original.get_checkpoint_state()
    restored = make_two_phase_scorer(freeze_after=1)

    restored.restore_checkpoint_state(checkpoint)

    np.testing.assert_allclose(restored.centroids, original.centroids)
    np.testing.assert_allclose(restored._dk_weights, original._dk_weights)
    assert restored._dk_weights is not checkpoint["dk_weights"]
    assert restored.centroids is not checkpoint["centroids"]
    assert [state.phase for state in restored._category_states] == [
        state.phase for state in original._category_states
    ]
    assert [state.freeze_point for state in restored._category_states] == [1, None]
    assert [state.n_decisions for state in restored._category_states] == [3, 0]


def test_checkpoint_restore_does_not_restore_buffer():
    original = make_two_phase_scorer(freeze_after=1)
    drive_phase2_with_weights(original)
    restored = make_two_phase_scorer(freeze_after=1)

    restored.restore_checkpoint_state(original.get_checkpoint_state())

    assert original._decision_buffer is not None
    assert len(original._decision_buffer) == 2
    assert restored._decision_buffer == []


def test_checkpoint_partial_restore():
    scorer = make_two_phase_scorer(freeze_after=1)
    drive_updates(scorer, category_index=0, action_index=0, n_updates=1)
    original_phase = scorer.get_phase(0)
    shifted = scorer.centroids.copy()
    shifted[0, 0, 0] = 0.33

    scorer.restore_checkpoint_state({"centroids": shifted, "decision_counts": [7, 0]})

    np.testing.assert_allclose(scorer.centroids, shifted)
    assert scorer._dk_weights is None
    assert scorer.get_phase(0) == original_phase
    assert scorer._category_states[0].n_decisions == 7


def test_pickle_full_twophase_roundtrip():
    scorer = make_two_phase_scorer(freeze_after=1)
    drive_phase2_with_weights(scorer)

    restored = pickle.loads(pickle.dumps(scorer))

    np.testing.assert_allclose(restored.centroids, scorer.centroids)
    np.testing.assert_allclose(restored._dk_weights, scorer._dk_weights)
    assert [state.phase for state in restored._category_states] == [
        state.phase for state in scorer._category_states
    ]
    assert [state.n_decisions for state in restored._category_states] == [3, 0]
    assert len(restored._decision_buffer) == len(scorer._decision_buffer)
    np.testing.assert_allclose(
        restored._decision_buffer[0][0],
        scorer._decision_buffer[0][0],
    )


def test_pickle_backward_compat_old_format():
    scorer = ProfileScorer(make_mu(), make_actions())
    state = scorer.__dict__.copy()
    for attr in ("_learning_strategy", "_category_states", "_dk_weights", "_decision_buffer"):
        state.pop(attr, None)

    restored = ProfileScorer.__new__(ProfileScorer)
    restored.__setstate__(state)

    assert restored._learning_strategy is None
    assert restored._category_states is None
    assert restored._dk_weights is None
    assert restored._decision_buffer is None
