"""
Adversarial input tests for GAE.

Verifies that the library raises ValueError on NaN/Inf inputs and
survives other hostile inputs without producing silent corruption.

All tests must pass — no skips, no xfail.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from gae.kernels import DiagonalKernel, L2Kernel
from gae.profile_scorer import ProfileScorer, ScoringResult


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_scorer(n_cat=2, n_act=3, n_fac=6, seed=0) -> ProfileScorer:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.1, 0.9, (n_cat, n_act, n_fac))
    return ProfileScorer(mu=mu, actions=[f"a{i}" for i in range(n_act)])


def make_profile(tau=0.1, eta=0.05, eta_neg=0.05, decay=0.001):
    return SimpleNamespace(
        temperature=tau,
        extensions={"eta": eta, "eta_neg": eta_neg, "count_decay": decay},
    )


# ── NaN inputs ────────────────────────────────────────────────────────────────

class TestNaNInputs:
    def test_score_nan_single_dim(self):
        """NaN in one factor dimension raises ValueError."""
        scorer = make_scorer()
        f = np.array([np.nan, 0.5, 0.5, 0.5, 0.5, 0.5])
        with pytest.raises(ValueError, match="NaN or Inf"):
            scorer.score(f, 0)

    def test_score_nan_all_dims(self):
        """All-NaN factor vector raises ValueError."""
        scorer = make_scorer()
        f = np.full(6, np.nan)
        with pytest.raises(ValueError, match="NaN or Inf"):
            scorer.score(f, 0)

    def test_score_mixed_nan_and_valid(self):
        """Partial NaN raises ValueError."""
        scorer = make_scorer()
        f = np.array([np.nan, np.nan, np.nan, 0.3, 0.6, 0.9])
        with pytest.raises(ValueError, match="NaN or Inf"):
            scorer.score(f, 0)

    def test_update_nan_factor_centralizes_cleanly(self):
        """update() with NaN f raises ValueError before touching centroids."""
        scorer = make_scorer(n_cat=1, n_act=2, n_fac=3)
        f_nan = np.full(3, np.nan)
        with pytest.raises(ValueError, match="NaN or Inf"):
            scorer.update(f_nan, 0, 0, correct=True)


# ── Inf inputs ────────────────────────────────────────────────────────────────

class TestInfInputs:
    def test_score_positive_inf(self):
        """Positive Inf in factor raises ValueError."""
        scorer = make_scorer()
        f = np.array([np.inf, 0.5, 0.5, 0.5, 0.5, 0.5])
        with pytest.raises(ValueError, match="NaN or Inf"):
            scorer.score(f, 0)

    def test_score_negative_inf(self):
        """Negative Inf in factor raises ValueError."""
        scorer = make_scorer()
        f = np.array([-np.inf, 0.5, 0.5, 0.5, 0.5, 0.5])
        with pytest.raises(ValueError, match="NaN or Inf"):
            scorer.score(f, 0)


# ── Boundary factor values ────────────────────────────────────────────────────

class TestBoundaryFactorValues:
    def test_score_all_zero(self):
        scorer = make_scorer()
        result = scorer.score(np.zeros(6), 0)
        assert isinstance(result, ScoringResult)
        assert abs(result.probabilities.sum() - 1.0) < 1e-9
        assert not np.any(np.isnan(result.probabilities))

    def test_score_all_ones(self):
        scorer = make_scorer()
        result = scorer.score(np.ones(6), 0)
        assert isinstance(result, ScoringResult)
        assert abs(result.probabilities.sum() - 1.0) < 1e-9

    def test_score_negative_values(self):
        """Factor values outside [0,1] are accepted; probabilities remain valid."""
        scorer = make_scorer()
        result = scorer.score(np.full(6, -0.5), 0)
        assert isinstance(result, ScoringResult)
        assert abs(result.probabilities.sum() - 1.0) < 1e-9

    def test_score_large_positive(self):
        """Factor values >> 1 (1e6): softmax stability check."""
        scorer = make_scorer()
        result = scorer.score(np.full(6, 1e6), 0)
        assert isinstance(result, ScoringResult)
        assert abs(result.probabilities.sum() - 1.0) < 1e-6

    def test_score_very_large(self):
        """Factor values >> 1 (1e12): softmax must not produce NaN/Inf."""
        scorer = make_scorer()
        result = scorer.score(np.full(6, 1e12), 0)
        # Numeric stability: probs should still sum to ~1 or be very close
        total = result.probabilities.sum()
        assert not np.isnan(total), "Probabilities must not be NaN for large inputs"

    def test_score_very_small_positive(self):
        """Factor values near zero (1e-15): no underflow corruption."""
        scorer = make_scorer()
        result = scorer.score(np.full(6, 1e-15), 0)
        assert isinstance(result, ScoringResult)
        assert abs(result.probabilities.sum() - 1.0) < 1e-9

    def test_update_boundary_f_all_zeros(self):
        """update() with f=zeros: centroids move toward zero, stay in [0,1]."""
        scorer = make_scorer(n_cat=1, n_act=2, n_fac=4)
        for _ in range(50):
            scorer.update(np.zeros(4), 0, 0, correct=True)
        assert scorer.centroids.min() >= 0.0
        assert scorer.centroids.max() <= 1.0

    def test_update_boundary_f_all_ones(self):
        """update() with f=ones: centroids move toward 1, stay in [0,1]."""
        scorer = make_scorer(n_cat=1, n_act=2, n_fac=4)
        for _ in range(50):
            scorer.update(np.ones(4), 0, 0, correct=True)
        assert scorer.centroids.min() >= 0.0
        assert scorer.centroids.max() <= 1.0


# ── Shape errors ──────────────────────────────────────────────────────────────

class TestShapeErrors:
    def test_score_wrong_factor_length_raises(self):
        scorer = make_scorer(n_fac=6)
        with pytest.raises((AssertionError, ValueError)):
            scorer.score(np.zeros(5), 0)  # wrong length

    def test_score_wrong_factor_length_too_long(self):
        scorer = make_scorer(n_fac=6)
        with pytest.raises((AssertionError, ValueError)):
            scorer.score(np.zeros(8), 0)

    def test_score_wrong_category_index_raises(self):
        scorer = make_scorer(n_cat=2)
        with pytest.raises((AssertionError, IndexError)):
            scorer.score(np.zeros(6), 99)

    def test_update_category_out_of_range_raises(self):
        scorer = make_scorer(n_cat=2)
        with pytest.raises((IndexError, AssertionError)):
            scorer.update(np.zeros(6), 99, 0, correct=True)

    def test_update_action_out_of_range_raises(self):
        scorer = make_scorer(n_act=3)
        with pytest.raises((IndexError, AssertionError)):
            scorer.update(np.zeros(6), 0, 99, correct=True)

    def test_constructor_wrong_mu_ndim_raises(self):
        with pytest.raises((AssertionError, ValueError)):
            ProfileScorer(mu=np.zeros((4, 6)), actions=["a", "b"])  # 2D, not 3D

    def test_constructor_actions_length_mismatch_raises(self):
        mu = np.zeros((2, 3, 4))
        with pytest.raises((AssertionError, ValueError)):
            ProfileScorer(mu=mu, actions=["a", "b"])  # need 3 actions


# ── Centroid initialization edge cases ───────────────────────────────────────

class TestCentroidEdgeCases:
    def test_centroids_all_zeros_score_works(self):
        mu = np.zeros((1, 3, 4))
        scorer = ProfileScorer(mu=mu, actions=["a", "b", "c"])
        result = scorer.score(np.full(4, 0.5), 0)
        assert abs(result.probabilities.sum() - 1.0) < 1e-9
        # All centroids equidistant from 0.5 → uniform probs
        np.testing.assert_allclose(
            result.probabilities,
            np.full(3, 1.0 / 3),
            atol=1e-9,
        )

    def test_centroids_all_ones_score_works(self):
        mu = np.ones((1, 2, 3))
        scorer = ProfileScorer(mu=mu, actions=["a", "b"])
        result = scorer.score(np.zeros(3), 0)
        assert abs(result.probabilities.sum() - 1.0) < 1e-9

    def test_centroids_identical_rows_give_uniform(self):
        """When all action centroids are identical, probabilities are uniform."""
        mu = np.full((1, 4, 5), 0.5)  # all same
        scorer = ProfileScorer(mu=mu, actions=["a", "b", "c", "d"])
        result = scorer.score(np.full(5, 0.3), 0)
        expected = np.full(4, 0.25)
        np.testing.assert_allclose(result.probabilities, expected, atol=1e-9)


# ── DiagonalKernel adversarial sigma ─────────────────────────────────────────

class TestDiagonalKernelAdversarial:
    def test_sigma_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="sigma values must be > 0"):
            DiagonalKernel(np.array([0.1, 0.0, 0.3]))

    def test_sigma_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="sigma values must be > 0"):
            DiagonalKernel(np.array([0.1, -0.5, 0.3]))

    def test_sigma_all_zeros_raises_value_error(self):
        with pytest.raises(ValueError):
            DiagonalKernel(np.zeros(5))

    def test_sigma_containing_inf_gives_zero_weight(self):
        """sigma=inf → W=1/inf²=0 → that dimension gets zero weight (effectively ignored)."""
        k = DiagonalKernel(np.array([0.1, float("inf")]))
        assert k.weights[0] == pytest.approx(1.0)
        assert k.weights[1] == pytest.approx(0.0, abs=1e-10)

    def test_sigma_inf_distance_ignores_inf_dim(self):
        """With inf sigma on dim 1, varying dim 1 doesn't affect distance."""
        k = DiagonalKernel(np.array([0.5, float("inf")]))
        mu = np.array([[0.0, 0.0]])
        d1 = k.compute_distance(np.array([1.0, 0.0]), mu)
        d2 = k.compute_distance(np.array([1.0, 9.9]), mu)
        assert d1[0] == pytest.approx(d2[0], rel=1e-9)

    def test_sigma_2d_raises_assertion(self):
        with pytest.raises(AssertionError):
            DiagonalKernel(np.ones((3, 3)))


# ── Learning rate edge cases ──────────────────────────────────────────────────

class TestLearningRateEdgeCases:
    def test_update_eta_zero_leaves_centroids_unchanged(self):
        """eta=0 → delta=0 → centroids must not move."""
        mu_init = np.full((1, 2, 4), 0.5)
        scorer = ProfileScorer(
            mu=mu_init.copy(),
            actions=["a", "b"],
            profile=make_profile(eta=0.0),
        )
        mu_before = scorer.centroids.copy()
        scorer.update(np.ones(4), 0, 0, correct=True)
        np.testing.assert_array_equal(scorer.centroids, mu_before)

    def test_update_eta_neg_ge_1_raises(self):
        """eta_neg >= 1.0 must be rejected at construction time."""
        mu = np.full((1, 2, 3), 0.5)
        with pytest.raises(ValueError, match="eta_neg"):
            ProfileScorer(
                mu=mu,
                actions=["a", "b"],
                profile=make_profile(eta_neg=1.0),
            )

    def test_update_eta_neg_exactly_1_raises(self):
        mu = np.full((1, 2, 3), 0.5)
        with pytest.raises(ValueError):
            ProfileScorer(
                mu=mu,
                actions=["a", "b"],
                profile=make_profile(eta_neg=1.0),
            )


# ── Temperature edge cases ────────────────────────────────────────────────────

class TestTemperatureEdgeCases:
    def test_score_near_zero_tau_produces_sharp_distribution(self):
        """Very small tau → almost all probability mass on nearest centroid."""
        mu = np.zeros((1, 3, 4))
        mu[0, 0, :] = [0.9, 0.9, 0.9, 0.9]  # centroid 0 nearest to f
        mu[0, 1, :] = [0.1, 0.1, 0.1, 0.1]
        mu[0, 2, :] = [0.5, 0.5, 0.5, 0.5]
        scorer = ProfileScorer(
            mu=mu,
            actions=["a", "b", "c"],
            profile=make_profile(tau=1e-10),
        )
        f = np.full(4, 0.95)
        result = scorer.score(f, 0)
        # With near-zero tau, the closest centroid dominates
        assert result.action_index == 0
        assert result.confidence > 0.99

    def test_score_high_tau_produces_near_uniform(self):
        """Very large tau → near-uniform distribution over all actions."""
        mu = np.zeros((1, 3, 4))
        mu[0, 0, :] = [0.9, 0.9, 0.9, 0.9]
        mu[0, 1, :] = [0.1, 0.1, 0.1, 0.1]
        mu[0, 2, :] = [0.5, 0.5, 0.5, 0.5]
        scorer = ProfileScorer(
            mu=mu,
            actions=["a", "b", "c"],
            profile=make_profile(tau=100.0),
        )
        result = scorer.score(np.full(4, 0.5), 0)
        # High temperature → all probabilities close to 1/3
        assert np.all(result.probabilities > 0.20)
        assert np.all(result.probabilities < 0.50)

    def test_score_tau_zero_raises(self):
        """tau=0 raises ValueError before computing distances."""
        mu = np.zeros((1, 2, 3))
        mu[0, 0, :] = [0.1, 0.2, 0.3]
        mu[0, 1, :] = [0.7, 0.8, 0.9]
        scorer = ProfileScorer(
            mu=mu,
            actions=["a", "b"],
            profile=make_profile(tau=0.0),
        )
        with pytest.raises(ValueError, match="tau"):
            scorer.score(np.array([0.5, 0.5, 0.5]), 0)


# ── Long-running stability ────────────────────────────────────────────────────

class TestLongRunningStability:
    def test_10000_sequential_updates_no_nan(self):
        """10000 updates must not produce any NaN in the centroid tensor."""
        rng = np.random.default_rng(7)
        scorer = make_scorer(n_cat=2, n_act=4, n_fac=6, seed=1)
        for i in range(10_000):
            f = rng.uniform(0.0, 1.0, 6)
            c = int(rng.integers(2))
            a = int(rng.integers(4))
            scorer.update(f, c, a, correct=bool(rng.integers(2)))
        assert not np.any(np.isnan(scorer.centroids)), "Centroid NaN after 10000 updates"
        assert not np.any(np.isinf(scorer.centroids)), "Centroid Inf after 10000 updates"

    def test_10000_sequential_updates_stay_in_bounds(self):
        """10000 updates must keep all centroids in [0.0, 1.0]."""
        rng = np.random.default_rng(13)
        scorer = make_scorer(n_cat=2, n_act=4, n_fac=6, seed=2)
        for i in range(10_000):
            f = rng.uniform(-0.5, 1.5, 6)  # adversarial: outside [0,1]
            c = int(rng.integers(2))
            a = int(rng.integers(4))
            scorer.update(f, c, a, correct=bool(rng.integers(2)))
        assert scorer.centroids.min() >= 0.0, (
            f"mu.min={scorer.centroids.min()} < 0"
        )
        assert scorer.centroids.max() <= 1.0, (
            f"mu.max={scorer.centroids.max()} > 1"
        )

    def test_score_after_many_updates_still_valid(self):
        """score() must return valid ScoringResult after 1000 updates."""
        rng = np.random.default_rng(21)
        scorer = make_scorer(n_cat=2, n_act=3, n_fac=6)
        for _ in range(1000):
            f = rng.uniform(0.0, 1.0, 6)
            scorer.update(f, 0, 0, correct=True)
        result = scorer.score(np.full(6, 0.5), 0)
        assert isinstance(result, ScoringResult)
        assert abs(result.probabilities.sum() - 1.0) < 1e-9
        assert not np.any(np.isnan(result.probabilities))

    def test_frozen_scorer_update_is_noop(self):
        """After freeze(), update() must not change centroids."""
        scorer = make_scorer(n_cat=1, n_act=2, n_fac=4)
        mu_before = scorer.centroids.copy()
        scorer.freeze()
        for _ in range(100):
            scorer.update(np.ones(4), 0, 0, correct=True)
        np.testing.assert_array_equal(scorer.centroids, mu_before)


def test_score_tau_negative_raises():
    """Negative tau raises ValueError."""
    scorer = make_scorer()
    scorer.tau = -0.1
    with pytest.raises(ValueError, match="tau"):
        scorer.score(np.random.rand(6), 0)
