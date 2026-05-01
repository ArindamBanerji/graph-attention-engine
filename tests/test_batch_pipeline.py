"""
Tests for gae.batch_pipeline.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from gae.batch_pipeline import (
    BatchHistory,
    DefaultPromotionGate,
    FixedIntervalPolicy,
    NoveltyThresholdPolicy,
)
from gae.dk_estimator import CoordinateDescentEstimator
from gae.novelty import NearestNeighborNovelty


def make_category_centroids() -> np.ndarray:
    return np.array(
        [
            [
                [0.10, 0.85, 0.25],
                [0.82, 0.15, 0.70],
            ]
        ],
        dtype=np.float64,
    )


def build_estimator_decisions() -> list[tuple[np.ndarray, int, int]]:
    return [
        (np.array([0.12, 0.88, 0.22], dtype=np.float64), 0, 0),
        (np.array([0.15, 0.84, 0.26], dtype=np.float64), 0, 0),
        (np.array([0.11, 0.86, 0.24], dtype=np.float64), 0, 0),
        (np.array([0.80, 0.16, 0.68], dtype=np.float64), 0, 1),
        (np.array([0.83, 0.13, 0.73], dtype=np.float64), 0, 1),
        (np.array([0.81, 0.18, 0.69], dtype=np.float64), 0, 1),
        (np.array([0.14, 0.83, 0.23], dtype=np.float64), 0, 0),
        (np.array([0.84, 0.12, 0.71], dtype=np.float64), 0, 1),
        (np.array([0.13, 0.87, 0.25], dtype=np.float64), 0, 0),
        (np.array([0.79, 0.17, 0.72], dtype=np.float64), 0, 1),
    ]


def compute_accuracy(
    weights: np.ndarray,
    centroids: np.ndarray,
    decisions: list[tuple[np.ndarray, int, int]],
) -> float:
    correct = 0
    for factor_vector, category_index, action_index in decisions:
        action_centroids = centroids[category_index]
        distances = np.sum(((factor_vector - action_centroids) ** 2) * weights, axis=1)
        predicted = int(np.argmin(distances))
        if predicted == action_index:
            correct += 1
    return correct / len(decisions)


# SECTION 1 — NoveltyThresholdPolicy


def test_novelty_threshold_policy_rejects_negative_threshold():
    with pytest.raises(ValueError, match="threshold must be >= 0"):
        NoveltyThresholdPolicy(threshold=-0.1)


def test_novelty_threshold_policy_rejects_non_positive_min_decisions():
    with pytest.raises(ValueError, match="min_decisions must be >= 1"):
        NoveltyThresholdPolicy(threshold=1.0, min_decisions=0)


def test_novelty_threshold_policy_rejects_negative_cooldown():
    with pytest.raises(ValueError, match="cooldown must be >= 0"):
        NoveltyThresholdPolicy(threshold=1.0, cooldown=-1)


def test_novelty_threshold_policy_requires_min_decisions():
    policy = NoveltyThresholdPolicy(threshold=2.0, min_decisions=3, cooldown=0)

    assert not policy.should_trigger(2.5, n_verified_decisions=2, category_index=0)


def test_novelty_threshold_policy_triggers_at_threshold():
    policy = NoveltyThresholdPolicy(threshold=2.0, min_decisions=3, cooldown=0)

    assert policy.should_trigger(2.0, n_verified_decisions=3, category_index=0)


def test_novelty_threshold_policy_cooldown_blocks_until_elapsed():
    policy = NoveltyThresholdPolicy(threshold=1.0, min_decisions=1, cooldown=3)

    assert policy.should_trigger(1.0, n_verified_decisions=5, category_index=0)
    policy.record_trigger(category_index=0, n_verified_decisions=5)
    assert not policy.should_trigger(5.0, n_verified_decisions=6, category_index=0)
    assert not policy.should_trigger(5.0, n_verified_decisions=7, category_index=0)
    assert policy.should_trigger(5.0, n_verified_decisions=8, category_index=0)


# SECTION 2 — DefaultPromotionGate


def test_default_promotion_gate_rejects_negative_margin():
    with pytest.raises(ValueError, match="superiority_margin must be >= 0"):
        DefaultPromotionGate(superiority_margin=-0.01)


def test_default_promotion_gate_rejects_floor_out_of_range():
    with pytest.raises(ValueError, match="floor must be in \\[0, 1\\]"):
        DefaultPromotionGate(floor=1.1)


def test_default_promotion_gate_rejects_non_positive_variance_ratio():
    with pytest.raises(ValueError, match="max_variance_ratio must be > 0"):
        DefaultPromotionGate(max_variance_ratio=0.0)


def test_default_promotion_gate_promotes_at_exact_superiority_margin():
    gate = DefaultPromotionGate(superiority_margin=0.05, floor=0.60, max_variance_ratio=2.0)

    verdict = gate.evaluate(
        old_accuracy=0.70,
        new_accuracy=0.75,
        old_weights=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        new_weights=np.array([1.2, 2.2, 3.2], dtype=np.float64),
    )

    assert verdict.promoted
    assert verdict.superiority_pass
    assert "placeholder_always_pass" in verdict.reason


def test_default_promotion_gate_uses_var_ratio_one_for_first_estimation():
    gate = DefaultPromotionGate(superiority_margin=0.0, floor=0.50, max_variance_ratio=1.5)

    verdict = gate.evaluate(
        old_accuracy=0.0,
        new_accuracy=0.60,
        old_weights=None,
        new_weights=np.array([0.5, 1.0, 1.5], dtype=np.float64),
    )

    assert verdict.var_ratio == pytest.approx(1.0)
    assert verdict.variance_pass


def test_default_promotion_gate_rejects_when_floor_fails():
    gate = DefaultPromotionGate(superiority_margin=0.01, floor=0.80, max_variance_ratio=10.0)

    verdict = gate.evaluate(
        old_accuracy=0.75,
        new_accuracy=0.79,
        old_weights=np.array([1.0, 1.2, 1.4], dtype=np.float64),
        new_weights=np.array([1.0, 1.1, 1.4], dtype=np.float64),
    )

    assert not verdict.promoted
    assert not verdict.floor_pass
    assert "floor_fail" in verdict.reason


def test_default_promotion_gate_rejects_when_superiority_fails():
    gate = DefaultPromotionGate(superiority_margin=0.03, floor=0.60, max_variance_ratio=10.0)

    verdict = gate.evaluate(
        old_accuracy=0.70,
        new_accuracy=0.72,
        old_weights=np.array([1.0, 1.1, 1.2], dtype=np.float64),
        new_weights=np.array([1.0, 1.2, 1.3], dtype=np.float64),
    )

    assert not verdict.promoted
    assert not verdict.superiority_pass
    assert "superiority_fail" in verdict.reason


def test_default_promotion_gate_rejects_when_variance_fails():
    gate = DefaultPromotionGate(superiority_margin=0.0, floor=0.50, max_variance_ratio=1.2)

    verdict = gate.evaluate(
        old_accuracy=0.70,
        new_accuracy=0.80,
        old_weights=np.array([1.0, 1.1, 0.9], dtype=np.float64),
        new_weights=np.array([0.1, 1.0, 4.0], dtype=np.float64),
    )

    assert not verdict.promoted
    assert not verdict.variance_pass
    assert "variance_fail" in verdict.reason


def test_default_promotion_gate_reason_contains_multiple_failures():
    gate = DefaultPromotionGate(superiority_margin=0.10, floor=0.90, max_variance_ratio=1.1)

    verdict = gate.evaluate(
        old_accuracy=0.85,
        new_accuracy=0.86,
        old_weights=np.array([1.0, 1.2, 0.8], dtype=np.float64),
        new_weights=np.array([0.1, 2.0, 4.0], dtype=np.float64),
    )

    assert not verdict.promoted
    assert "superiority_fail" in verdict.reason
    assert "floor_fail" in verdict.reason
    assert "variance_fail" in verdict.reason
    assert "placeholder_always_pass" in verdict.reason


def test_default_promotion_gate_conservation_always_passes():
    gate = DefaultPromotionGate(superiority_margin=0.0, floor=0.0, max_variance_ratio=100.0)

    verdict = gate.evaluate(
        old_accuracy=0.1,
        new_accuracy=0.1,
        old_weights=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        new_weights=np.array([1.0, 1.0, 1.0], dtype=np.float64),
    )

    assert verdict.conservation_pass


def test_default_promotion_gate_spec_defaults():
    gate = DefaultPromotionGate()

    assert gate.superiority_margin == pytest.approx(0.05)
    assert gate.floor == pytest.approx(0.75)
    assert gate.max_variance_ratio == pytest.approx(2.0)


# SECTION 3 — BatchHistory


def test_batch_history_rejects_invalid_max_records():
    with pytest.raises(ValueError, match="max_records must be >= 1"):
        BatchHistory(max_records=0)


def test_batch_history_hash_is_deterministic_for_same_weights():
    history = BatchHistory(max_records=5)
    weights = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    assert history._hash(weights) == history._hash(weights.copy())


def test_batch_history_hash_changes_for_different_weights():
    history = BatchHistory(max_records=5)

    assert history._hash(np.array([1.0, 2.0], dtype=np.float64)) != history._hash(
        np.array([1.0, 2.1], dtype=np.float64)
    )


def test_batch_history_record_stores_hashes():
    history = BatchHistory(max_records=5)
    gate = DefaultPromotionGate()
    old_weights = np.array([1.0, 1.0], dtype=np.float64)
    new_weights = np.array([1.2, 1.1], dtype=np.float64)
    verdict = gate.evaluate(0.5, 0.6, old_weights, new_weights)

    record = history.record(0, 0.5, 0.6, old_weights, new_weights, verdict)

    assert record.old_weights_hash == history._hash(old_weights)
    assert record.new_weights_hash == history._hash(new_weights)


def test_batch_history_filters_records():
    history = BatchHistory(max_records=5)
    gate = DefaultPromotionGate(superiority_margin=0.0, floor=0.0, max_variance_ratio=10.0)
    verdict_a = gate.evaluate(0.5, 0.6, None, np.array([1.0, 1.0], dtype=np.float64))
    verdict_b = gate.evaluate(
        0.6,
        0.55,
        np.array([1.0, 1.0], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
    )
    history.record(0, 0.5, 0.6, None, np.array([1.0, 1.0], dtype=np.float64), verdict_a)
    history.record(
        1,
        0.6,
        0.55,
        np.array([1.0, 1.0], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
        verdict_b,
    )

    assert len(history.get_records(category_index=0)) == 1
    assert len(history.get_records(promoted_only=True)) == 1


def test_batch_history_totals_and_trimming_keep_most_recent_records():
    history = BatchHistory(max_records=2)
    gate = DefaultPromotionGate(superiority_margin=0.0, floor=0.0, max_variance_ratio=2.0)
    base_time = datetime(2024, 1, 1)
    for offset in range(3):
        verdict = gate.evaluate(0.5, 0.6 + 0.01 * offset, None, np.array([1.0, 1.0], dtype=np.float64))
        history.record(
            category_index=offset,
            old_accuracy=0.5,
            new_accuracy=0.6 + 0.01 * offset,
            old_weights=None,
            new_weights=np.array([1.0 + offset, 2.0 + offset], dtype=np.float64),
            verdict=verdict,
            attempted_at=base_time + timedelta(minutes=offset),
        )

    records = history.get_records()
    assert len(records) == 2
    assert [record.category_index for record in records] == [1, 2]
    assert history.total_attempts() == 2
    assert history.total_promotions() == 2


# SECTION 4 — Integration


def test_novelty_policy_with_real_tracker():
    tracker = NearestNeighborNovelty(threshold=0.2)
    policy = NoveltyThresholdPolicy(threshold=3.0, min_decisions=3, cooldown=0)
    decisions = [
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([2.0, 2.0], dtype=np.float64),
        np.array([4.0, 4.0], dtype=np.float64),
    ]
    for factor_vector in decisions:
        tracker.record(factor_vector, 0)

    assert tracker.get_accumulator(0) > 3.0
    assert policy.should_trigger(
        tracker.get_accumulator(0),
        n_verified_decisions=len(decisions),
        category_index=0,
    )


def test_gate_with_real_estimator_output():
    centroids = make_category_centroids()
    decisions = build_estimator_decisions()
    estimator = CoordinateDescentEstimator(seed=7, n_rounds=3, max_per_cat=10)
    new_weights = estimator.estimate(decisions, centroids, n_categories=1, n_dims=3)[0]
    old_weights = np.ones(3, dtype=np.float64)
    old_accuracy = compute_accuracy(old_weights, centroids, decisions)
    new_accuracy = compute_accuracy(new_weights, centroids, decisions)
    gate = DefaultPromotionGate(superiority_margin=0.0, floor=0.5, max_variance_ratio=20.0)

    verdict = gate.evaluate(old_accuracy, new_accuracy, old_weights, new_weights)

    assert isinstance(verdict.promoted, bool)
    assert np.isfinite(verdict.var_ratio)
    assert verdict.new_accuracy == pytest.approx(new_accuracy)


def test_full_pipeline_flow():
    centroids = make_category_centroids()
    decisions = build_estimator_decisions()
    tracker = NearestNeighborNovelty(threshold=0.2)
    for factor_vector, _, _ in decisions[:4]:
        tracker.record(factor_vector, 0)
    policy = NoveltyThresholdPolicy(threshold=2.0, min_decisions=4, cooldown=0)
    assert policy.should_trigger(tracker.get_accumulator(0), 4, 0)

    estimator = CoordinateDescentEstimator(seed=11, n_rounds=3, max_per_cat=10)
    new_weights = estimator.estimate(decisions, centroids, n_categories=1, n_dims=3)[0]
    old_weights = np.ones(3, dtype=np.float64)
    old_accuracy = compute_accuracy(old_weights, centroids, decisions)
    new_accuracy = compute_accuracy(new_weights, centroids, decisions)
    gate = DefaultPromotionGate(superiority_margin=0.0, floor=0.5, max_variance_ratio=20.0)
    verdict = gate.evaluate(old_accuracy, new_accuracy, old_weights, new_weights)
    history = BatchHistory(max_records=5)
    record = history.record(0, old_accuracy, new_accuracy, old_weights, new_weights, verdict)

    assert history.total_attempts() == 1
    assert record.promoted == verdict.promoted
    assert record.verdict.reason == verdict.reason


def test_pipeline_gate_rejects_bad_weights():
    history = BatchHistory(max_records=5)
    gate = DefaultPromotionGate(superiority_margin=0.05, floor=0.80, max_variance_ratio=10.0)
    old_weights = np.ones(3, dtype=np.float64)
    new_weights = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    verdict = gate.evaluate(
        old_accuracy=0.78,
        new_accuracy=0.79,
        old_weights=old_weights,
        new_weights=new_weights,
    )
    record = history.record(0, 0.78, 0.79, old_weights, new_weights, verdict)

    assert not verdict.promoted
    assert not record.promoted
    assert history.total_attempts() == 1
    assert history.total_promotions() == 0
