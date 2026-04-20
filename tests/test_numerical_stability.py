"""
Numerical stability tests for GAE.

Verifies that long-running learning, extreme inputs, and edge cases
never produce NaN, Inf, or out-of-bounds centroids.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from gae.kernels import DiagonalKernel, L2Kernel
from gae.calibration import derive_theta_min, check_conservation
from gae.profile_scorer import ProfileScorer, ScoringResult, MAX_ETA_DELTA


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_scorer(n_cat=2, n_act=4, n_fac=6, seed=0) -> ProfileScorer:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.1, 0.9, (n_cat, n_act, n_fac))
    return ProfileScorer(mu=mu, actions=[f"a{i}" for i in range(n_act)])


# ── Sequential update stability ───────────────────────────────────────────────

class TestSequentialUpdateStability:
    def test_10k_correct_updates_centroids_stay_in_01(self):
        """10K correct updates with varying f: mu stays in [0,1]."""
        rng = np.random.default_rng(0)
        scorer = make_scorer(n_cat=1, n_act=3, n_fac=6)
        f_target = rng.uniform(0.0, 1.0, 6)
        for _ in range(10_000):
            scorer.update(f_target, 0, int(rng.integers(3)), correct=True)
        assert scorer.centroids.min() >= 0.0
        assert scorer.centroids.max() <= 1.0

    def test_10k_incorrect_updates_no_nan(self):
        """10K incorrect updates: centroids never go NaN."""
        rng = np.random.default_rng(1)
        scorer = make_scorer(n_cat=2, n_act=4, n_fac=6)
        for _ in range(10_000):
            f = rng.uniform(0.0, 1.0, 6)
            c = int(rng.integers(2))
            a = int(rng.integers(4))
            gt = int(rng.integers(4))
            with np.errstate(all="ignore"):
                scorer.update(f, c, a, correct=False, gt_action_index=gt)
        assert not np.any(np.isnan(scorer.centroids))
        assert not np.any(np.isinf(scorer.centroids))

    def test_10k_alternating_correct_incorrect_no_nan(self):
        """Alternating correct/incorrect updates: centroids stay finite."""
        rng = np.random.default_rng(2)
        scorer = make_scorer(n_cat=1, n_act=2, n_fac=4)
        for i in range(10_000):
            f = rng.uniform(0.0, 1.0, 4)
            correct = i % 2 == 0
            gt = 1 if correct else 0
            with np.errstate(all="ignore"):
                scorer.update(f, 0, 0, correct=correct,
                              gt_action_index=None if correct else gt)
        assert not np.any(np.isnan(scorer.centroids))
        assert scorer.centroids.min() >= 0.0
        assert scorer.centroids.max() <= 1.0

    def test_10k_rapid_alternation_between_two_f(self):
        """Rapidly alternate between two very different factor vectors."""
        scorer = make_scorer(n_cat=1, n_act=2, n_fac=6)
        f_a = np.zeros(6)    # one extreme
        f_b = np.ones(6)     # opposite extreme
        for i in range(10_000):
            f = f_a if i % 2 == 0 else f_b
            scorer.update(f, 0, 0, correct=True)
        assert scorer.centroids.min() >= 0.0
        assert scorer.centroids.max() <= 1.0
        assert not np.any(np.isnan(scorer.centroids))

    def test_centroid_converges_under_constant_input(self):
        """After many correct updates toward same f, centroid moves toward f."""
        scorer = make_scorer(n_cat=1, n_act=2, n_fac=6, seed=10)
        f_target = np.full(6, 0.9)
        dist_before = np.linalg.norm(scorer.centroids[0, 0, :] - f_target)
        for _ in range(500):
            scorer.update(f_target, 0, 0, correct=True)
        dist_after = np.linalg.norm(scorer.centroids[0, 0, :] - f_target)
        assert dist_after < dist_before, "Centroid must move toward f_target after 500 updates"


# ── Probability distribution stability ───────────────────────────────────────

class TestProbabilityStability:
    def test_probabilities_never_negative(self):
        """Probabilities must be >= 0 for all inputs."""
        rng = np.random.default_rng(3)
        scorer = make_scorer()
        for _ in range(200):
            f = rng.uniform(0.0, 1.0, 6)
            result = scorer.score(f, int(rng.integers(2)))
            assert np.all(result.probabilities >= 0.0), (
                f"Negative probability: {result.probabilities}"
            )

    def test_probabilities_never_greater_than_one(self):
        """Probabilities must be <= 1.0 for all inputs."""
        rng = np.random.default_rng(4)
        scorer = make_scorer()
        for _ in range(200):
            f = rng.uniform(0.0, 1.0, 6)
            result = scorer.score(f, int(rng.integers(2)))
            assert np.all(result.probabilities <= 1.0), (
                f"Probability > 1: {result.probabilities}"
            )

    def test_probabilities_sum_to_one_across_1000_calls(self):
        """Sum of probabilities is exactly 1.0 across 1000 random factor vectors."""
        rng = np.random.default_rng(5)
        scorer = make_scorer()
        for _ in range(1_000):
            f = rng.uniform(0.0, 1.0, 6)
            result = scorer.score(f, int(rng.integers(2)))
            assert abs(result.probabilities.sum() - 1.0) < 1e-9, (
                f"Sum={result.probabilities.sum()}"
            )

    def test_score_batch_determinism(self):
        """1000 scores with same f produce identical results (no hidden mutable state)."""
        scorer = make_scorer(seed=99)
        f = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        results = [scorer.score(f, 0) for _ in range(1_000)]
        for r in results[1:]:
            np.testing.assert_array_equal(
                r.probabilities,
                results[0].probabilities,
                err_msg="score() must not have hidden mutable state affecting output",
            )


# ── Softmax numerical stability ───────────────────────────────────────────────

class TestSoftmaxNumericalStability:
    def test_softmax_stable_with_very_large_distances(self):
        """Very large distances (small f, large centroid) must not overflow softmax."""
        mu = np.ones((1, 4, 6)) * 0.99
        mu[0, 0, :] = 0.01
        scorer = ProfileScorer(mu=mu, actions=["a", "b", "c", "d"])
        result = scorer.score(np.zeros(6), 0)
        assert not np.any(np.isnan(result.probabilities))
        assert not np.any(np.isinf(result.probabilities))
        assert abs(result.probabilities.sum() - 1.0) < 1e-9

    def test_softmax_stable_with_zero_distance(self):
        """f exactly equals a centroid (distance=0): softmax must not produce NaN."""
        mu = np.full((1, 3, 4), 0.5)
        mu[0, 0, :] = [0.2, 0.4, 0.6, 0.8]
        mu[0, 1, :] = [0.5, 0.5, 0.5, 0.5]
        mu[0, 2, :] = [0.8, 0.6, 0.4, 0.2]
        scorer = ProfileScorer(mu=mu, actions=["a", "b", "c"])
        f = np.array([0.2, 0.4, 0.6, 0.8])  # exactly equals centroid 0
        result = scorer.score(f, 0)
        assert not np.any(np.isnan(result.probabilities))
        assert abs(result.probabilities.sum() - 1.0) < 1e-9

    def test_two_identical_centroids_no_nan(self):
        """When two action centroids are identical, score remains valid."""
        mu = np.full((1, 3, 4), 0.5)
        mu[0, 0, :] = mu[0, 1, :] = [0.2, 0.3, 0.4, 0.5]  # identical
        mu[0, 2, :] = [0.8, 0.7, 0.6, 0.5]
        scorer = ProfileScorer(mu=mu, actions=["a", "b", "c"])
        result = scorer.score(np.full(4, 0.3), 0)
        assert abs(result.probabilities.sum() - 1.0) < 1e-9
        assert not np.any(np.isnan(result.probabilities))
        # The two identical centroids get equal probability
        assert result.probabilities[0] == pytest.approx(result.probabilities[1], rel=1e-9)


# ── Gradient magnitude bounds ─────────────────────────────────────────────────

class TestGradientBounds:
    def test_max_eta_delta_cap_enforced_correct(self):
        """Even with large f-μ gap, step size capped at MAX_ETA_DELTA per coordinate."""
        mu = np.zeros((1, 1, 6))  # centroid at 0
        scorer = ProfileScorer(mu=mu, actions=["a"])
        f = np.ones(6)            # f at 1 — max gap
        mu_before = scorer.centroids.copy()
        scorer.update(f, 0, 0, correct=True)
        delta = np.abs(scorer.centroids - mu_before)
        assert delta.max() <= MAX_ETA_DELTA + 1e-10, (
            f"Max delta {delta.max()} exceeds MAX_ETA_DELTA={MAX_ETA_DELTA}"
        )

    def test_max_eta_delta_cap_enforced_incorrect(self):
        """Incorrect update also respects MAX_ETA_DELTA cap."""
        mu = np.full((1, 2, 6), 0.5)
        scorer = ProfileScorer(mu=mu, actions=["a", "b"])
        mu_before = scorer.centroids.copy()
        scorer.update(np.zeros(6), 0, 0, correct=False, gt_action_index=1)
        delta = np.abs(scorer.centroids - mu_before)
        assert delta.max() <= MAX_ETA_DELTA + 1e-10

    def test_diagonal_kernel_gradient_bounded_by_unity(self):
        """DiagonalKernel gradient magnitude per component is at most 1 × (f-mu)."""
        sigma = np.array([0.05, 0.10, 0.20, 0.50, 1.00, 2.00])  # very unequal
        k = DiagonalKernel(sigma)
        f = np.ones(6)
        mu = np.zeros(6)
        g = k.compute_gradient(f, mu)
        l2_grad = f - mu
        # Each component: |g_i| <= |l2_grad_i|
        assert np.all(np.abs(g) <= np.abs(l2_grad) + 1e-10), (
            f"DiagonalKernel gradient {g} exceeds L2 gradient {l2_grad}"
        )

    def test_l2_kernel_distance_non_negative(self):
        """L2 distance must be non-negative for any f and mu."""
        rng = np.random.default_rng(6)
        k = L2Kernel()
        for _ in range(100):
            f = rng.uniform(-5, 5, 8)   # adversarial range
            mu = rng.uniform(-5, 5, (4, 8))
            d = k.compute_distance(f, mu)
            assert np.all(d >= 0.0), f"Negative L2 distance: {d}"


# ── Conservation arithmetic stability ────────────────────────────────────────

class TestConservationStability:
    def test_conservation_v_zero_no_crash(self):
        """V=0 (no verified decisions): check_conservation should not raise."""
        theta = derive_theta_min()
        cc = check_conservation(alpha=0.3, q=0.8, V=0.0, theta_min=theta)
        assert cc.status == "RED"  # signal = 0 < theta
        assert cc.passed is False

    def test_conservation_alpha_zero_no_crash(self):
        """alpha=0 (no overrides): must not divide by zero."""
        theta = derive_theta_min()
        cc = check_conservation(alpha=0.0, q=0.9, V=100.0, theta_min=theta)
        assert cc.signal == pytest.approx(0.0)
        assert cc.status == "RED"

    def test_conservation_extreme_values(self):
        """alpha=0.99, V=1e6: must not overflow."""
        theta = derive_theta_min()
        cc = check_conservation(alpha=0.99, q=0.99, V=1e6, theta_min=theta)
        assert cc.status in ("GREEN", "AMBER")
        assert not np.isinf(cc.signal)
        assert not np.isnan(cc.signal)

    def test_conservation_theta_zero_headroom_inf(self):
        """theta_min=0: headroom is defined as inf (not divide-by-zero crash)."""
        cc = check_conservation(alpha=0.5, q=0.8, V=10.0, theta_min=0.0)
        assert not np.isnan(cc.headroom)


# ── Large tensor performance ──────────────────────────────────────────────────

class TestLargeTensor:
    def test_large_tensor_score_completes_quickly(self):
        """50 categories × 20 actions × 100 factors: score completes in < 1 second."""
        rng = np.random.default_rng(100)
        n_cat, n_act, n_fac = 50, 20, 100
        mu = rng.uniform(0.0, 1.0, (n_cat, n_act, n_fac))
        scorer = ProfileScorer(mu=mu, actions=[f"a{i}" for i in range(n_act)])
        f = rng.uniform(0.0, 1.0, n_fac)
        t0 = time.time()
        result = scorer.score(f, 0)
        elapsed = time.time() - t0
        assert elapsed < 1.0, f"score() took {elapsed:.3f}s > 1.0s on large tensor"
        assert abs(result.probabilities.sum() - 1.0) < 1e-9

    def test_large_tensor_centroids_shape_correct(self):
        """Large ProfileScorer centroids property has correct shape."""
        rng = np.random.default_rng(101)
        n_cat, n_act, n_fac = 100, 50, 200
        mu = rng.uniform(0.0, 1.0, (n_cat, n_act, n_fac))
        scorer = ProfileScorer(mu=mu, actions=[f"a{i}" for i in range(n_act)])
        assert scorer.centroids.shape == (n_cat, n_act, n_fac)

    def test_score_stability_small_perturbation(self):
        """Small perturbation in f produces small perturbation in probabilities (Lipschitz)."""
        scorer = make_scorer(seed=50)
        f0 = np.array([0.5, 0.4, 0.3, 0.6, 0.7, 0.2])
        epsilon = 1e-4
        r0 = scorer.score(f0, 0)
        for i in range(6):
            f1 = f0.copy()
            f1[i] += epsilon
            r1 = scorer.score(f1, 0)
            # Probability change must be bounded (not discontinuous)
            max_delta_p = np.abs(r1.probabilities - r0.probabilities).max()
            assert max_delta_p < 0.5, (
                f"Large probability jump {max_delta_p} for dim {i} perturbation {epsilon}"
            )
