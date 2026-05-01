"""
Tests for gae.dk_estimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from gae.dk_estimator import CoordinateDescentEstimator, DKEstimator


def make_centroids(n_categories: int, n_actions: int, n_dims: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 0.9, size=(n_categories, n_actions, n_dims))


def build_decisions_for_category(
    centroids: np.ndarray,
    category_index: int,
    n_samples: int,
    signal_dim: int | None = None,
    noise_scale: float = 0.03,
    seed: int = 0,
) -> list[tuple[np.ndarray, int, int]]:
    rng = np.random.default_rng(seed)
    n_actions = centroids.shape[1]
    decisions: list[tuple[np.ndarray, int, int]] = []
    for i in range(n_samples):
        action = i % n_actions
        if signal_dim is None:
            factor_vector = centroids[category_index, action].copy()
        else:
            confound_action = (action + 1) % n_actions
            factor_vector = centroids[category_index, confound_action].copy()
            factor_vector += rng.normal(0.0, noise_scale, size=centroids.shape[2])
            factor_vector[signal_dim] = centroids[category_index, action, signal_dim]
            factor_vector[signal_dim] += rng.normal(0.0, noise_scale)
        factor_vector = np.clip(factor_vector, 0.0, 1.0)
        decisions.append((factor_vector, category_index, action))
    return decisions


class TestDKEstimatorProtocol:
    def test_coordinate_descent_is_protocol_instance(self):
        assert isinstance(CoordinateDescentEstimator(), DKEstimator)


class TestCoordinateDescentInit:
    def test_invalid_n_rounds_raises(self):
        with pytest.raises(ValueError, match="n_rounds must be >= 1"):
            CoordinateDescentEstimator(n_rounds=0)

    def test_invalid_max_per_cat_raises(self):
        with pytest.raises(ValueError, match="max_per_cat must be >= 10"):
            CoordinateDescentEstimator(max_per_cat=9)

    def test_invalid_candidates_raise(self):
        with pytest.raises(ValueError, match="all candidates must be > 0"):
            CoordinateDescentEstimator(candidates=[0.5, 0.0, 2.0])


class TestCoordinateDescentEstimate:
    def test_estimate_returns_shape_for_category_centroids_and_leaves_empty_category_uniform(self):
        centroids = make_centroids(n_categories=3, n_actions=4, n_dims=5, seed=11)
        decisions = build_decisions_for_category(centroids, category_index=0, n_samples=60, seed=12)
        estimator = CoordinateDescentEstimator(seed=7)

        weights = estimator.estimate(
            decisions=decisions,
            centroids=centroids,
            n_categories=3,
            n_dims=5,
        )

        assert weights.shape == (3, 5)
        np.testing.assert_allclose(weights[1], np.ones(5))
        np.testing.assert_allclose(weights[2], np.ones(5))

    def test_estimate_supports_shared_action_centroids(self):
        shared_centroids = make_centroids(n_categories=1, n_actions=4, n_dims=4, seed=21)[0]
        rng = np.random.default_rng(22)
        decisions: list[tuple[np.ndarray, int, int]] = []
        for category_index in range(2):
            for i in range(40):
                action = i % shared_centroids.shape[0]
                factor_vector = shared_centroids[action] + rng.normal(0.0, 0.02, size=4)
                decisions.append((np.clip(factor_vector, 0.0, 1.0), category_index, action))

        estimator = CoordinateDescentEstimator(seed=3)
        weights = estimator.estimate(
            decisions=decisions,
            centroids=shared_centroids,
            n_categories=2,
            n_dims=4,
        )

        assert weights.shape == (2, 4)
        assert np.all(weights > 0.0)

    def test_compute_accuracy_is_perfect_for_exact_centroid_matches(self):
        centroids = np.array(
            [
                [0.1, 0.9, 0.2],
                [0.9, 0.1, 0.8],
                [0.3, 0.3, 0.7],
            ],
            dtype=np.float64,
        )
        features = centroids.copy()
        correct_actions = np.array([0, 1, 2], dtype=np.int64)
        weights = np.ones(3, dtype=np.float64)
        estimator = CoordinateDescentEstimator()

        acc = estimator._compute_accuracy(features, correct_actions, centroids, weights)
        assert acc == pytest.approx(1.0)

    def test_compute_accuracy_is_near_chance_on_pure_noise(self):
        rng = np.random.default_rng(31)
        centroids = rng.uniform(0.0, 1.0, size=(4, 6))
        features = rng.uniform(0.0, 1.0, size=(4000, 6))
        correct_actions = rng.integers(0, 4, size=4000, dtype=np.int64)
        estimator = CoordinateDescentEstimator()

        acc = estimator._compute_accuracy(features, correct_actions, centroids, np.ones(6))
        assert 0.15 <= acc <= 0.35

    def test_uniform_data_keeps_weights_within_two_x(self):
        rng = np.random.default_rng(41)
        centroids = np.full((1, 4, 6), 0.5, dtype=np.float64)
        decisions = [
            (rng.uniform(0.45, 0.55, size=6), 0, int(rng.integers(0, 4)))
            for _ in range(200)
        ]
        estimator = CoordinateDescentEstimator(seed=5)

        weights = estimator.estimate(decisions, centroids, n_categories=1, n_dims=6)
        ratio = float(weights[0].max() / weights[0].min())
        assert ratio <= 2.0 + 1e-9

    def test_discriminative_dim_zero_is_strictly_maximum(self):
        centroids = np.array(
            [
                [
                    [0.1, 0.9, 0.8, 0.7],
                    [0.4, 0.7, 0.6, 0.5],
                    [0.7, 0.3, 0.4, 0.3],
                    [0.9, 0.1, 0.2, 0.1],
                ]
            ],
            dtype=np.float64,
        )
        decisions = build_decisions_for_category(
            centroids,
            category_index=0,
            n_samples=160,
            signal_dim=0,
            noise_scale=0.01,
            seed=51,
        )
        estimator = CoordinateDescentEstimator(seed=6)

        weights = estimator.estimate(decisions, centroids, n_categories=1, n_dims=4)
        assert weights[0, 0] == pytest.approx(np.max(weights[0]))
        assert np.argmax(weights[0]) == 0
        assert np.sum(weights[0] == weights[0, 0]) == 1

    def test_categories_learn_different_top_dimensions(self):
        centroids = np.array(
            [
                [
                    [0.1, 0.9, 0.8, 0.7],
                    [0.4, 0.7, 0.6, 0.5],
                    [0.7, 0.3, 0.4, 0.3],
                    [0.9, 0.1, 0.2, 0.1],
                ],
                [
                    [0.9, 0.1, 0.8, 0.7],
                    [0.7, 0.4, 0.6, 0.5],
                    [0.3, 0.7, 0.4, 0.3],
                    [0.1, 0.9, 0.2, 0.1],
                ],
            ],
            dtype=np.float64,
        )
        decisions = build_decisions_for_category(
            centroids,
            category_index=0,
            n_samples=160,
            signal_dim=0,
            noise_scale=0.01,
            seed=61,
        )
        decisions.extend(
            build_decisions_for_category(
                centroids,
                category_index=1,
                n_samples=160,
                signal_dim=1,
                noise_scale=0.01,
                seed=62,
            )
        )
        estimator = CoordinateDescentEstimator(seed=9)

        weights = estimator.estimate(decisions, centroids, n_categories=2, n_dims=4)
        assert np.argmax(weights[0]) == 0
        assert np.argmax(weights[1]) == 1

    def test_subsampling_is_deterministic_for_same_seed_and_can_change_for_different_seed(self):
        centroids = np.array(
            [
                [
                    [0.1, 0.8, 0.2],
                    [0.3, 0.6, 0.4],
                    [0.7, 0.4, 0.6],
                    [0.9, 0.2, 0.8],
                ]
            ],
            dtype=np.float64,
        )
        rng = np.random.default_rng(71)
        decisions: list[tuple[np.ndarray, int, int]] = []
        for i in range(800):
            action = int(rng.integers(0, 4))
            if i < 400:
                factor_vector = centroids[0, (action + 1) % 4].copy()
                factor_vector[2] = centroids[0, action, 2]
            else:
                factor_vector = centroids[0, (action + 2) % 4].copy()
                factor_vector[1] = centroids[0, action, 1]
            factor_vector += rng.normal(0.0, 0.04, size=3)
            decisions.append((np.clip(factor_vector, 0.0, 1.0), 0, action))

        estimator_a = CoordinateDescentEstimator(max_per_cat=40, seed=13)
        estimator_b = CoordinateDescentEstimator(max_per_cat=40, seed=13)
        estimator_c = CoordinateDescentEstimator(max_per_cat=40, seed=14)
        category_decisions = [(factor_vector, action) for factor_vector, _, action in decisions]

        weights_a = estimator_a.estimate(decisions, centroids, n_categories=1, n_dims=3)
        weights_b = estimator_b.estimate(decisions, centroids, n_categories=1, n_dims=3)
        weights_c = estimator_c.estimate(decisions, centroids, n_categories=1, n_dims=3)
        subset_a = estimator_a._subsample_category(category_decisions, category_index=0)
        subset_b = estimator_b._subsample_category(category_decisions, category_index=0)
        subset_c = estimator_c._subsample_category(category_decisions, category_index=0)

        np.testing.assert_array_equal(weights_a, weights_b)
        assert len(subset_a) == 40
        assert len(subset_c) == 40
        for (factor_a, action_a), (factor_b, action_b) in zip(subset_a, subset_b):
            np.testing.assert_array_equal(factor_a, factor_b)
            assert action_a == action_b
        assert any(
            (not np.array_equal(factor_a, factor_c)) or (action_a != action_c)
            for (factor_a, action_a), (factor_c, action_c) in zip(subset_a, subset_c)
        )
        assert np.all(weights_c > 0.0)

    def test_compute_accuracy_is_numerically_stable_for_large_logits(self):
        centroids = np.array(
            [
                [0.0, 0.0],
                [1000.0, 1000.0],
            ],
            dtype=np.float64,
        )
        features = np.array(
            [
                [0.0, 0.0],
                [1000.0, 1000.0],
            ],
            dtype=np.float64,
        )
        correct_actions = np.array([0, 1], dtype=np.int64)
        weights = np.array([4.0, 4.0], dtype=np.float64)
        estimator = CoordinateDescentEstimator(tau=0.1)

        acc = estimator._compute_accuracy(features, correct_actions, centroids, weights)
        assert np.isfinite(acc)
        assert acc == pytest.approx(1.0)
