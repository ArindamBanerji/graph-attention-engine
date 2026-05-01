from __future__ import annotations

import numpy as np

from gae.batch_pipeline import BatchHistory, DefaultPromotionGate, NoveltyThresholdPolicy
from gae.novelty import NearestNeighborNovelty
from gae.profile_scorer import LearningStrategy, ProfileScorer
from gae.shrinkage import FixedAlpha
from gae.two_phase import DecisionCountPolicy, MEAN_CONVERGENCE, VARIANCE_LEARNING


class FixedEstimator:
    def __init__(self, weights: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.calls = 0

    def estimate(self, decisions, centroids, n_categories, n_dims) -> np.ndarray:
        self.calls += 1
        assert centroids.shape == (n_categories, centroids.shape[1], n_dims)
        return self.weights.copy()


def make_centroids(
    *,
    n_categories: int = 2,
    n_actions: int = 2,
    n_dims: int = 3,
) -> np.ndarray:
    centroids = np.zeros((n_categories, n_actions, n_dims), dtype=np.float64)
    for category_index in range(n_categories):
        for action_index in range(n_actions):
            base = 0.15 + 0.65 * (action_index / max(n_actions - 1, 1))
            offsets = 0.01 * category_index + 0.005 * np.arange(n_dims)
            centroids[category_index, action_index] = np.clip(base + offsets, 0.0, 1.0)
    return centroids


def make_actions(n_actions: int) -> list[str]:
    return [f"a{i}" for i in range(n_actions)]


def make_scorer(
    *,
    n_categories: int = 2,
    n_actions: int = 2,
    n_dims: int = 3,
    freeze_after: int = 2,
    weights: np.ndarray | None = None,
) -> ProfileScorer:
    centroids = make_centroids(
        n_categories=n_categories,
        n_actions=n_actions,
        n_dims=n_dims,
    )
    if weights is None:
        weights = np.ones((n_categories, n_dims), dtype=np.float64)
        weights[:, 0] = 2.0
    strategy = LearningStrategy(
        phase_policy=DecisionCountPolicy(n=freeze_after),
        dk_estimator=FixedEstimator(weights),
        shrinkage_schedule=FixedAlpha(1.0),
    )
    return ProfileScorer(
        centroids,
        make_actions(n_actions),
        learning_strategy=strategy,
    )


def drive_to_phase2(scorer: ProfileScorer, *, category_index: int, n_updates: int = 2) -> None:
    for update_index in range(n_updates):
        action_index = update_index % scorer.n_actions
        scorer.update(
            scorer.centroids[category_index, action_index].copy(),
            category_index=category_index,
            action_index=action_index,
            correct=True,
        )


def buffer_decisions(
    scorer: ProfileScorer,
    tracker: NearestNeighborNovelty,
    *,
    category_index: int = 0,
    n_decisions: int = 4,
) -> list[tuple[np.ndarray, int, int]]:
    decisions = []
    for index in range(n_decisions):
        action_index = index % scorer.n_actions
        factor_vector = scorer.centroids[category_index, action_index].copy()
        factor_vector = np.clip(factor_vector + 0.001 * index, 0.0, 1.0)
        scorer.score(factor_vector, category_index)
        scorer.update(
            factor_vector,
            category_index=category_index,
            action_index=action_index,
            correct=True,
        )
        tracker.record(factor_vector, category_index)
        decisions.append((factor_vector, category_index, action_index))
    return decisions


def compute_accuracy(
    weights: np.ndarray,
    centroids: np.ndarray,
    decisions: list[tuple[np.ndarray, int, int]],
) -> float:
    correct = 0
    for factor_vector, category_index, action_index in decisions:
        category_weights = weights[category_index] if weights.ndim == 2 else weights
        category_centroids = centroids[category_index]
        distances = np.sum(
            ((factor_vector - category_centroids) ** 2) * category_weights,
            axis=1,
        )
        if int(np.argmin(distances)) == action_index:
            correct += 1
    return correct / len(decisions)


def test_e2e_trigger_estimate_gate_promote():
    candidate = np.array([[2.0, 0.5, 1.5], [1.0, 1.0, 1.0]], dtype=np.float64)
    scorer = make_scorer(weights=candidate, freeze_after=2)
    drive_to_phase2(scorer, category_index=0)
    tracker = NearestNeighborNovelty(threshold=0.0, n_categories=2)
    policy = NoveltyThresholdPolicy(threshold=0.0, min_decisions=2, cooldown=0)
    decisions = buffer_decisions(scorer, tracker, n_decisions=4)

    assert policy.should_trigger(tracker.get_accumulator(0), len(decisions), 0)
    scorer.reestimate_dk()
    gate = DefaultPromotionGate(superiority_margin=0.05, floor=0.50, max_variance_ratio=100.0)
    verdict = gate.evaluate(
        old_accuracy=0.50,
        new_accuracy=0.75,
        old_weights=np.array([0.9, 1.0, 1.1], dtype=np.float64),
        new_weights=scorer.get_dk_weights(0),
    )

    assert verdict.promoted
    np.testing.assert_allclose(scorer.get_dk_weights(0), candidate[0])


def test_e2e_gate_rejects_bad_estimation():
    scorer = make_scorer(weights=np.ones((2, 3), dtype=np.float64), freeze_after=2)
    drive_to_phase2(scorer, category_index=0)
    tracker = NearestNeighborNovelty(threshold=0.0)
    buffer_decisions(scorer, tracker, n_decisions=4)
    scorer.reestimate_dk()
    gate = DefaultPromotionGate(superiority_margin=0.10, floor=0.80, max_variance_ratio=10.0)

    verdict = gate.evaluate(
        old_accuracy=0.75,
        new_accuracy=0.76,
        old_weights=np.ones(3, dtype=np.float64),
        new_weights=scorer.get_dk_weights(0),
    )

    assert not verdict.promoted
    assert verdict.reason


def test_e2e_novelty_tracker_accumulates_during_scoring():
    scorer = make_scorer(freeze_after=1)
    drive_to_phase2(scorer, category_index=0, n_updates=1)
    tracker = NearestNeighborNovelty(threshold=0.01, n_categories=2)

    buffer_decisions(scorer, tracker, n_decisions=5)

    assert tracker.get_accumulator(0) > 0.0
    assert tracker.get_history_size(0) == 5
    assert 0.0 <= tracker.get_novelty_rate(0) <= 1.0


def test_e2e_batch_history_records_all_attempts():
    history = BatchHistory(max_records=5)
    gate = DefaultPromotionGate(superiority_margin=0.05, floor=0.50, max_variance_ratio=2.0)
    cases = [
        (0.50, 0.60, np.array([0.8, 1.0, 1.2]), np.array([0.9, 1.0, 1.1])),
        (0.70, 0.71, np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])),
        (0.55, 0.75, np.array([1.0, 1.1, 0.9]), np.array([0.2, 3.0, 5.0])),
    ]

    for old_accuracy, new_accuracy, old_weights, new_weights in cases:
        verdict = gate.evaluate(old_accuracy, new_accuracy, old_weights, new_weights)
        history.record(0, old_accuracy, new_accuracy, old_weights, new_weights, verdict)

    records = history.get_records()
    assert history.total_attempts() == 3
    assert len(records) == 3
    assert any(record.promoted for record in records)
    assert any(not record.promoted for record in records)
    assert all(record.reason for record in records)


def test_e2e_checkpoint_survives_batch_cycle():
    candidate = np.array([[1.5, 0.8, 2.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    scorer = make_scorer(weights=candidate, freeze_after=2)
    drive_to_phase2(scorer, category_index=0)
    tracker = NearestNeighborNovelty(threshold=0.0)
    buffer_decisions(scorer, tracker, n_decisions=4)
    scorer.reestimate_dk()
    gate = DefaultPromotionGate(superiority_margin=0.0, floor=0.0, max_variance_ratio=10.0)
    gate.evaluate(0.50, 0.60, np.ones(3, dtype=np.float64), scorer.get_dk_weights(0))

    restored = make_scorer(weights=candidate, freeze_after=2)
    restored.restore_checkpoint_state(scorer.get_checkpoint_state())

    assert restored.get_phase(0) == VARIANCE_LEARNING
    np.testing.assert_allclose(restored.get_dk_weights(0), scorer.get_dk_weights(0))


def test_stress_100_batch_cycles():
    scorer = make_scorer(freeze_after=1)
    drive_to_phase2(scorer, category_index=0, n_updates=1)
    tracker = NearestNeighborNovelty(threshold=0.0)
    policy = NoveltyThresholdPolicy(threshold=0.0, min_decisions=1, cooldown=0)
    gate = DefaultPromotionGate(superiority_margin=0.0, floor=0.0, max_variance_ratio=100.0)
    history = BatchHistory(max_records=100)

    for cycle in range(100):
        factor_vector = scorer.centroids[0, cycle % scorer.n_actions].copy()
        factor_vector = np.clip(factor_vector + cycle * 0.0001, 0.0, 1.0)
        scorer.update(factor_vector, 0, cycle % scorer.n_actions, correct=True)
        tracker.record(factor_vector, 0)
        assert policy.should_trigger(tracker.get_accumulator(0), cycle + 1, 0)
        scorer.reestimate_dk()
        new_weights = scorer.get_dk_weights(0)
        if new_weights is None:
            new_weights = np.ones(scorer.n_factors, dtype=np.float64)
        verdict = gate.evaluate(0.5, 0.5, np.ones_like(new_weights), new_weights)
        history.record(0, 0.5, 0.5, np.ones_like(new_weights), new_weights, verdict)
        policy.record_trigger(0, cycle + 1)

    assert history.total_attempts() == 100


def test_stress_d20_categories_10():
    weights = np.ones((10, 20), dtype=np.float64)
    weights[:, ::2] = 1.5
    scorer = make_scorer(n_categories=10, n_actions=4, n_dims=20, weights=weights, freeze_after=1)
    drive_to_phase2(scorer, category_index=3, n_updates=1)
    tracker = NearestNeighborNovelty(threshold=0.0, n_categories=10)
    decisions = buffer_decisions(scorer, tracker, category_index=3, n_decisions=4)

    scorer.reestimate_dk()
    new_weights = scorer.get_dk_weights(3)
    old_accuracy = compute_accuracy(np.ones((10, 20), dtype=np.float64), scorer.centroids, decisions)
    new_accuracy = compute_accuracy(scorer._dk_weights, scorer.centroids, decisions)
    verdict = DefaultPromotionGate(superiority_margin=0.0, floor=0.0, max_variance_ratio=10.0).evaluate(
        old_accuracy,
        new_accuracy,
        np.ones(20, dtype=np.float64),
        new_weights,
    )

    assert new_weights.shape == (20,)
    assert isinstance(verdict.promoted, bool)


def test_stress_concurrent_categories():
    scorer = make_scorer(n_categories=6, n_actions=2, n_dims=4, freeze_after=2)
    tracker = NearestNeighborNovelty(threshold=0.0, n_categories=6)

    for round_index in range(2):
        for category_index in range(6):
            action_index = round_index % 2
            factor_vector = scorer.centroids[category_index, action_index].copy()
            scorer.update(factor_vector, category_index, action_index, correct=True)
            tracker.record(factor_vector, category_index)

    assert all(scorer.get_phase(category_index) == VARIANCE_LEARNING for category_index in range(6))
    assert all(tracker.get_history_size(category_index) == 2 for category_index in range(6))


def test_stress_gate_with_extreme_weights():
    gate = DefaultPromotionGate(superiority_margin=0.0, floor=0.0, max_variance_ratio=1.2)

    verdict = gate.evaluate(
        old_accuracy=0.60,
        new_accuracy=0.90,
        old_weights=np.array([0.9, 1.0, 1.1], dtype=np.float64),
        new_weights=np.array([0.01, 50.0, 100.0], dtype=np.float64),
    )

    assert not verdict.promoted
    assert not verdict.variance_pass
    assert "variance_fail" in verdict.reason


def test_compat_no_strategy_ignores_batch():
    scorer = ProfileScorer(make_centroids(), make_actions(2))

    checkpoint = scorer.get_checkpoint_state()
    scorer.reestimate_dk()

    assert checkpoint["category_phases"] is None
    assert checkpoint["dk_weights"] is None
    assert checkpoint["decision_buffer_size"] == 0
    assert scorer.get_dk_weights(0) is None


def test_compat_phase1_scorer_no_batch_trigger():
    scorer = make_scorer(freeze_after=10)
    policy = NoveltyThresholdPolicy(threshold=1.0, min_decisions=1, cooldown=0)
    tracker = NearestNeighborNovelty(threshold=0.1)

    assert scorer.get_phase(0) == MEAN_CONVERGENCE
    assert not policy.should_trigger(tracker.get_accumulator(0), 0, 0)
