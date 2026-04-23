"""
Shape contract tests for GAE.

Verifies that ProfileScorer, kernels, and all public functions
enforce correct input/output tensor shapes at every interface.
"""

from __future__ import annotations

import numpy as np
import pytest

from gae.profile_scorer import ProfileScorer, ScoringResult, CentroidUpdate
from gae.kernels import L2Kernel, DiagonalKernel
from gae.calibration import soc_calibration_profile, s2p_calibration_profile


def make_scorer(n_cat=3, n_act=4, n_fac=6, seed=0) -> ProfileScorer:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.1, 0.9, (n_cat, n_act, n_fac))
    return ProfileScorer(mu=mu, actions=[f"a{i}" for i in range(n_act)])


# ── Centroid shape contracts ──────────────────────────────────────────────────

class TestCentroidShapeContracts:
    def test_soc_config_shape(self):
        n_cat, n_act, n_fac = 6, 4, 6
        mu = np.full((n_cat, n_act, n_fac), 0.5)
        scorer = ProfileScorer(
            mu=mu,
            actions=[f"a{i}" for i in range(n_act)],
            profile=soc_calibration_profile(),
        )
        assert scorer.centroids.shape == (n_cat, n_act, n_fac)
        assert scorer.centroids.shape == (n_cat, n_act, n_fac)

    def test_s2p_config_shape(self):
        n_cat, n_act, n_fac = 5, 5, 8
        mu = np.full((n_cat, n_act, n_fac), 0.5)
        scorer = ProfileScorer(
            mu=mu,
            actions=[f"a{i}" for i in range(n_act)],
            profile=s2p_calibration_profile(),
        )
        assert scorer.centroids.shape == (n_cat, n_act, n_fac)

    def test_minimal_config_shape(self):
        mu = np.full((1, 1, 1), 0.5)
        scorer = ProfileScorer(mu=mu, actions=["a"])
        assert scorer.centroids.shape == (1, 1, 1)

    def test_large_config_shape(self):
        n_cat, n_act, n_fac = 20, 10, 50
        rng = np.random.default_rng(0)
        mu = rng.uniform(0.0, 1.0, (n_cat, n_act, n_fac))
        scorer = ProfileScorer(mu=mu, actions=[f"a{i}" for i in range(n_act)])
        assert scorer.centroids.shape == (n_cat, n_act, n_fac)

    def test_n_categories_property(self):
        scorer = make_scorer(n_cat=5, n_act=3, n_fac=4)
        assert scorer.n_categories == 5

    def test_n_actions_property(self):
        scorer = make_scorer(n_cat=2, n_act=7, n_fac=4)
        assert scorer.n_actions == 7

    def test_n_factors_property(self):
        scorer = make_scorer(n_cat=2, n_act=3, n_fac=9)
        assert scorer.n_factors == 9

    def test_centroids_property_returns_correct_shape(self):
        scorer = make_scorer()
        assert scorer.centroids is not None
        assert scorer.centroids.shape == (3, 4, 6)


# ── score() output shape contracts ───────────────────────────────────────────

class TestScoreOutputShapes:
    def test_score_probs_shape_default(self):
        scorer = make_scorer(n_cat=3, n_act=4, n_fac=6)
        result = scorer.score(np.full(6, 0.5), category_index=0)
        assert result.probabilities.shape == (4,)

    def test_score_distances_shape_default(self):
        scorer = make_scorer(n_cat=3, n_act=4, n_fac=6)
        result = scorer.score(np.full(6, 0.5), category_index=0)
        assert result.distances.shape == (4,)

    def test_score_shape_5_actions(self):
        scorer = make_scorer(n_cat=2, n_act=5, n_fac=6)
        result = scorer.score(np.zeros(6), category_index=1)
        assert result.probabilities.shape == (5,)
        assert result.distances.shape == (5,)

    def test_score_shape_1_action(self):
        scorer = make_scorer(n_cat=1, n_act=1, n_fac=4)
        result = scorer.score(np.full(4, 0.5), category_index=0)
        assert result.probabilities.shape == (1,)
        assert result.distances.shape == (1,)

    def test_score_result_type(self):
        scorer = make_scorer()
        result = scorer.score(np.full(6, 0.3), category_index=0)
        assert isinstance(result, ScoringResult)

    def test_score_across_all_categories(self):
        n_cat, n_act, n_fac = 4, 3, 5
        scorer = make_scorer(n_cat=n_cat, n_act=n_act, n_fac=n_fac)
        f = np.full(n_fac, 0.5)
        for c in range(n_cat):
            result = scorer.score(f, category_index=c)
            assert result.probabilities.shape == (n_act,)
            assert result.distances.shape == (n_act,)

    def test_score_probs_shape_after_updates(self):
        """Shape contracts hold even after 100 update() calls."""
        scorer = make_scorer(n_cat=2, n_act=3, n_fac=6)
        for _ in range(100):
            scorer.update(np.full(6, 0.5), 0, 0, correct=True)
        result = scorer.score(np.full(6, 0.5), category_index=0)
        assert result.probabilities.shape == (3,)


# ── update() shape contracts ─────────────────────────────────────────────────

class TestUpdateShapeContracts:
    def test_update_returns_centroid_update(self):
        scorer = make_scorer(n_cat=2, n_act=3, n_fac=6)
        result = scorer.update(
            np.full(6, 0.5), category_index=0, action_index=0, correct=True
        )
        assert isinstance(result, CentroidUpdate)

    def test_update_does_not_change_centroid_shape(self):
        scorer = make_scorer(n_cat=2, n_act=3, n_fac=6)
        shape_before = scorer.centroids.shape
        scorer.update(
            np.full(6, 0.5), category_index=0, action_index=1,
            correct=False, gt_action_index=2,
        )
        assert scorer.centroids.shape == shape_before

    def test_update_preserves_n_categories(self):
        scorer = make_scorer(n_cat=3, n_act=4, n_fac=6)
        for _ in range(50):
            scorer.update(np.full(6, 0.5), 0, 0, correct=True)
        assert scorer.n_categories == 3

    def test_update_preserves_n_factors(self):
        scorer = make_scorer(n_cat=2, n_act=2, n_fac=8)
        scorer.update(np.full(8, 0.5), 0, 0, correct=True)
        assert scorer.n_factors == 8


# ── Input shape validation ────────────────────────────────────────────────────

class TestInputShapeValidation:
    def test_wrong_factor_length_score_raises(self):
        scorer = make_scorer(n_fac=6)
        f_wrong = np.full(5, 0.5)  # should be 6
        with pytest.raises((AssertionError, ValueError, IndexError)):
            scorer.score(f_wrong, category_index=0)

    def test_wrong_factor_length_update_raises(self):
        scorer = make_scorer(n_fac=6)
        f_wrong = np.full(7, 0.5)  # should be 6
        with pytest.raises((AssertionError, ValueError, IndexError)):
            scorer.update(f_wrong, category_index=0, action_index=0, correct=True)

    def test_out_of_range_category_raises(self):
        scorer = make_scorer(n_cat=2)
        with pytest.raises((IndexError, AssertionError, ValueError)):
            scorer.score(np.full(6, 0.5), category_index=99)

    def test_out_of_range_action_raises(self):
        scorer = make_scorer(n_act=4)
        with pytest.raises((IndexError, AssertionError, ValueError)):
            scorer.update(
                np.full(6, 0.5), category_index=0, action_index=99, correct=True
            )

    def test_negative_category_raises(self):
        scorer = make_scorer(n_cat=2)
        with pytest.raises((IndexError, AssertionError, ValueError)):
            scorer.score(np.full(6, 0.5), category_index=-1)


# ── Kernel shape contracts ────────────────────────────────────────────────────

class TestDiagonalKernelShapeContracts:
    def test_sigma_must_be_1d(self):
        with pytest.raises((AssertionError, ValueError)):
            DiagonalKernel(np.full((2, 3), 0.5))

    def test_weights_shape_matches_sigma(self):
        sigma = np.array([0.1, 0.2, 0.3, 0.4])
        k = DiagonalKernel(sigma)
        assert k.weights.shape == (4,)

    def test_compute_distance_output_shape(self):
        sigma = np.array([0.5, 0.5, 0.5])
        k = DiagonalKernel(sigma)
        f = np.array([0.3, 0.5, 0.7])
        mu = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        d = k.compute_distance(f, mu)
        assert d.shape == (3,)

    def test_compute_gradient_output_shape(self):
        sigma = np.array([0.5, 0.5, 0.5])
        k = DiagonalKernel(sigma)
        g = k.compute_gradient(np.array([0.3, 0.5, 0.7]), np.array([0.1, 0.2, 0.3]))
        assert g.shape == (3,)

    def test_raw_weights_shape_matches_sigma(self):
        sigma = np.array([0.2, 0.4, 0.6, 0.8])
        k = DiagonalKernel(sigma)
        assert k.raw_weights.shape == (4,)


class TestL2KernelShapeContracts:
    def test_compute_distance_shape(self):
        k = L2Kernel()
        f = np.array([0.1, 0.2, 0.3, 0.4])
        mu = np.array([[0.5, 0.5, 0.5, 0.5], [0.1, 0.2, 0.3, 0.4]])
        d = k.compute_distance(f, mu)
        assert d.shape == (2,)

    def test_compute_gradient_shape(self):
        k = L2Kernel()
        g = k.compute_gradient(np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6]))
        assert g.shape == (3,)

    def test_compute_distance_single_action(self):
        k = L2Kernel()
        f = np.array([0.5, 0.5])
        mu = np.array([[0.3, 0.7]])  # 1 action
        d = k.compute_distance(f, mu)
        assert d.shape == (1,)
