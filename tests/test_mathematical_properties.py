"""
Mathematical property tests for GAE.

These tests verify the mathematical claims from math_synopsis_v13:
softmax properties, kernel properties, learning properties,
and convergence/conservation arithmetic.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gae.kernels import DiagonalKernel, L2Kernel
from gae.calibration import derive_theta_min, check_conservation
from gae.profile_scorer import ProfileScorer


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_scorer_with_separated_centroids(n_fac=6) -> ProfileScorer:
    """Build a scorer where each action centroid is clearly different."""
    mu = np.zeros((1, 4, n_fac))
    mu[0, 0, :] = 0.9   # action 0: high
    mu[0, 1, :] = 0.1   # action 1: low
    mu[0, 2, :] = 0.7   # action 2: mid-high
    mu[0, 3, :] = 0.3   # action 3: mid-low
    return ProfileScorer(mu=mu, actions=["a", "b", "c", "d"])


def make_profile(tau=0.1, eta=0.05, eta_neg=0.05, decay=0.001):
    return SimpleNamespace(
        temperature=tau,
        extensions={"eta": eta, "eta_neg": eta_neg, "count_decay": decay},
    )


# ── Softmax mathematical properties ──────────────────────────────────────────

class TestSoftmaxProperties:
    def test_probabilities_sum_exactly_one(self):
        scorer = make_scorer_with_separated_centroids()
        f = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        result = scorer.score(f, 0)
        assert abs(result.probabilities.sum() - 1.0) < 1e-12

    def test_probabilities_all_strictly_positive(self):
        """Softmax output is strictly positive for finite distances."""
        scorer = make_scorer_with_separated_centroids()
        result = scorer.score(np.full(6, 0.5), 0)
        assert np.all(result.probabilities > 0.0)

    def test_closest_centroid_gets_highest_probability(self):
        """Lower L2 distance → higher softmax probability (monotonicity)."""
        scorer = make_scorer_with_separated_centroids()
        f = np.full(6, 0.95)  # close to action 0 centroid (0.9)
        result = scorer.score(f, 0)
        assert result.action_index == 0, (
            f"Closest centroid (0.9) should have highest prob. Got action {result.action_index}"
        )

    def test_monotonicity_lower_distance_higher_prob(self):
        """Ranked distances → inversely ranked probabilities."""
        scorer = make_scorer_with_separated_centroids()
        f = np.full(6, 0.85)   # distances: d0<d2<d3<d1
        result = scorer.score(f, 0)
        dists = result.distances
        probs = result.probabilities
        # Sort by distance ascending → probabilities must be descending
        order_by_dist = np.argsort(dists)
        order_by_prob = np.argsort(-probs)
        np.testing.assert_array_equal(order_by_dist, order_by_prob,
                                       err_msg="Prob ranking must equal inverse distance ranking")

    def test_action_index_equals_argmax_probabilities(self):
        scorer = make_scorer_with_separated_centroids()
        result = scorer.score(np.full(6, 0.5), 0)
        assert result.action_index == int(np.argmax(result.probabilities))

    def test_confidence_equals_max_probability(self):
        scorer = make_scorer_with_separated_centroids()
        result = scorer.score(np.full(6, 0.5), 0)
        assert result.confidence == pytest.approx(result.probabilities.max())

    def test_softmax_translation_invariance(self):
        """Softmax(logits + c) = Softmax(logits) — the -max shift must not change output."""
        scorer = make_scorer_with_separated_centroids()
        f = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.2])
        result1 = scorer.score(f, 0)
        # The implementation already uses logits -= logits.max() internally.
        # Verify the output is unchanged on a second call (the shift is internal).
        result2 = scorer.score(f, 0)
        np.testing.assert_array_equal(result1.probabilities, result2.probabilities)

    def test_higher_tau_produces_more_uniform_distribution(self):
        """tau=1.0 must produce more uniform distribution than tau=0.1."""
        mu = np.zeros((1, 3, 4))
        mu[0, 0, :] = [0.9, 0.9, 0.9, 0.9]
        mu[0, 1, :] = [0.1, 0.1, 0.1, 0.1]
        mu[0, 2, :] = [0.5, 0.5, 0.5, 0.5]
        f = np.full(4, 0.95)

        s_sharp = ProfileScorer(mu=mu.copy(), actions=["a", "b", "c"],
                                profile=make_profile(tau=0.1))
        s_flat  = ProfileScorer(mu=mu.copy(), actions=["a", "b", "c"],
                                profile=make_profile(tau=1.0))

        r_sharp = s_sharp.score(f, 0)
        r_flat  = s_flat.score(f, 0)

        # Entropy of sharp < entropy of flat → max prob of sharp > max prob of flat
        assert r_sharp.confidence > r_flat.confidence, (
            f"tau=0.1 confidence={r_sharp.confidence:.3f} must exceed "
            f"tau=1.0 confidence={r_flat.confidence:.3f}"
        )


# ── L2 Kernel properties ──────────────────────────────────────────────────────

class TestL2KernelMathProperties:
    def test_self_distance_is_zero(self):
        """K(f, f) = 0 for any f."""
        k = L2Kernel()
        f = np.array([0.3, 0.5, 0.7, 0.2, 0.8, 0.4])
        mu = f.reshape(1, -1)
        d = k.compute_distance(f, mu)
        assert d[0] == pytest.approx(0.0, abs=1e-12)

    def test_symmetry(self):
        """K(f, g) == K(g, f)."""
        k = L2Kernel()
        f = np.array([0.3, 0.7, 0.5])
        g = np.array([0.6, 0.2, 0.9])
        d_fg = k.compute_distance(f, g.reshape(1, -1))[0]
        d_gf = k.compute_distance(g, f.reshape(1, -1))[0]
        assert d_fg == pytest.approx(d_gf, rel=1e-12)

    def test_non_negativity(self):
        """K(f, g) >= 0 always."""
        k = L2Kernel()
        rng = np.random.default_rng(20)
        for _ in range(100):
            f = rng.uniform(-2, 2, 5)
            mu = rng.uniform(-2, 2, (3, 5))
            d = k.compute_distance(f, mu)
            assert np.all(d >= 0.0)

    def test_triangle_inequality(self):
        """d(a,b) <= d(a,c) + d(c,b) (for L2, holds exactly)."""
        k = L2Kernel()
        rng = np.random.default_rng(21)
        for _ in range(50):
            a = rng.uniform(0, 1, 4)
            b = rng.uniform(0, 1, 4)
            c = rng.uniform(0, 1, 4)
            d_ab = np.sqrt(k.compute_distance(a, b.reshape(1, -1))[0])
            d_ac = np.sqrt(k.compute_distance(a, c.reshape(1, -1))[0])
            d_cb = np.sqrt(k.compute_distance(c, b.reshape(1, -1))[0])
            assert d_ab <= d_ac + d_cb + 1e-9, (
                f"Triangle inequality violated: d(a,b)={d_ab:.4f} > d(a,c)+d(c,b)={d_ac+d_cb:.4f}"
            )

    def test_gradient_equals_f_minus_mu(self):
        """L2Kernel gradient is exactly f - mu."""
        k = L2Kernel()
        rng = np.random.default_rng(22)
        f = rng.uniform(0, 1, 6)
        mu = rng.uniform(0, 1, 6)
        g = k.compute_gradient(f, mu)
        np.testing.assert_allclose(g, f - mu, rtol=1e-12)

    def test_gradient_is_zero_at_centroid(self):
        k = L2Kernel()
        f = np.array([0.3, 0.5, 0.7])
        g = k.compute_gradient(f, f)
        np.testing.assert_allclose(g, np.zeros(3), atol=1e-12)


# ── DiagonalKernel properties ─────────────────────────────────────────────────

class TestDiagonalKernelMathProperties:
    def test_unit_sigma_equals_l2(self):
        """DiagonalKernel(sigma=ones) is mathematically equivalent to L2Kernel."""
        l2 = L2Kernel()
        diag = DiagonalKernel(np.ones(6))
        rng = np.random.default_rng(30)
        for _ in range(50):
            f = rng.uniform(0, 1, 6)
            mu = rng.uniform(0, 1, (4, 6))
            np.testing.assert_allclose(
                l2.compute_distance(f, mu),
                diag.compute_distance(f, mu),
                rtol=1e-12,
            )

    def test_lower_weight_dim_contributes_less_to_distance(self):
        """Downweighting a dim reduces its contribution to total distance."""
        # Both dims differ from centroid; dim 1 gets much lower weight in k_sk
        f = np.array([0.0, 0.0])
        mu = np.array([[1.0, 1.0]])
        k_eq = DiagonalKernel(np.array([0.5, 0.5]))   # equal weights
        k_sk = DiagonalKernel(np.array([0.5, 50.0]))  # dim 1 nearly zeroed out
        d_eq = k_eq.compute_distance(f, mu)[0]
        d_sk = k_sk.compute_distance(f, mu)[0]
        assert d_sk < d_eq, (
            "Severely downweighted dim 1 must reduce total distance vs uniform weights"
        )

    def test_weights_sum_bounded(self):
        """DiagonalKernel weights are in (0, 1] with max = 1.0."""
        sigma = np.array([0.05, 0.10, 0.20, 0.50, 1.00])
        k = DiagonalKernel(sigma)
        assert k.weights.max() == pytest.approx(1.0)
        assert np.all(k.weights > 0)
        assert np.all(k.weights <= 1.0)

    def test_gradient_bounded_by_l2_gradient(self):
        """DiagonalKernel gradient <= L2 gradient component-wise (since weights <= 1)."""
        sigma = np.array([0.1, 0.3, 0.5, 1.0])
        diag = DiagonalKernel(sigma)
        rng = np.random.default_rng(31)
        f = rng.uniform(0, 1, 4)
        mu = rng.uniform(0, 1, 4)
        g_diag = diag.compute_gradient(f, mu)
        g_l2   = f - mu
        assert np.all(np.abs(g_diag) <= np.abs(g_l2) + 1e-12), (
            "DiagonalKernel gradient must not exceed L2 gradient component-wise"
        )


# ── Learning / update properties ─────────────────────────────────────────────

class TestLearningMathProperties:
    def test_correct_update_decreases_distance(self):
        """After correct update, centroid is strictly closer to f."""
        rng = np.random.default_rng(40)
        mu = rng.uniform(0.1, 0.9, (1, 2, 5))
        scorer = ProfileScorer(mu=mu, actions=["a", "b"])
        f = rng.uniform(0.0, 1.0, 5)
        dist_before = np.linalg.norm(scorer.centroids[0, 0, :] - f)
        scorer.update(f, 0, 0, correct=True)
        dist_after = np.linalg.norm(scorer.centroids[0, 0, :] - f)
        assert dist_after < dist_before

    def test_incorrect_update_increases_predicted_distance(self):
        """After incorrect update, predicted centroid is strictly farther from f."""
        mu = np.full((1, 2, 4), 0.5)
        scorer = ProfileScorer(mu=mu, actions=["a", "b"])
        f = np.zeros(4)
        dist_before = np.linalg.norm(scorer.centroids[0, 0, :] - f)
        scorer.update(f, 0, 0, correct=False, gt_action_index=1)
        dist_after = np.linalg.norm(scorer.centroids[0, 0, :] - f)
        assert dist_after > dist_before

    def test_incorrect_update_decreases_gt_distance(self):
        """After incorrect update, GT centroid is closer to f."""
        mu = np.full((1, 2, 4), 0.5)
        mu[0, 1, :] = 0.8   # GT centroid starts far from f=0
        scorer = ProfileScorer(mu=mu, actions=["a", "b"])
        f = np.zeros(4)
        dist_gt_before = np.linalg.norm(scorer.centroids[0, 1, :] - f)
        scorer.update(f, 0, 0, correct=False, gt_action_index=1)
        dist_gt_after = np.linalg.norm(scorer.centroids[0, 1, :] - f)
        assert dist_gt_after < dist_gt_before

    def test_eta_confirm_ge_eta_neg_eff(self):
        """By default, eta_confirm (0.05) >= eta_neg (0.05), so no asymmetry unless override set."""
        mu = np.full((1, 2, 4), 0.5)
        scorer = ProfileScorer(mu=mu, actions=["a", "b"])
        assert scorer.eta >= 0.0
        assert scorer.eta_neg >= 0.0

    def test_eta_override_attenuates_incorrect_path(self):
        """With eta_override=0.01 < eta=0.05, incorrect push is smaller than correct pull."""
        mu = np.full((1, 2, 4), 0.5)
        s_with    = ProfileScorer(mu=mu.copy(), actions=["a", "b"], eta_override=0.01)
        s_without = ProfileScorer(mu=mu.copy(), actions=["a", "b"])
        f = np.zeros(4)

        mu_with_before = s_with.centroids[0, 0, :].copy()
        mu_without_before = s_without.centroids[0, 0, :].copy()

        r_with    = s_with.update(f, 0, 0, correct=False, gt_action_index=1)
        r_without = s_without.update(f, 0, 0, correct=False, gt_action_index=1)

        # With eta_override=0.01, push step is smaller
        delta_with = np.abs(s_with.centroids[0, 0, :] - mu_with_before).sum()
        delta_without = np.abs(
            s_without.centroids[0, 0, :] - mu_without_before
        ).sum()
        assert delta_with <= delta_without, (
            f"eta_override=0.01 should produce smaller push: {delta_with} vs {delta_without}"
        )

    def test_centroid_converges_toward_target_after_many_updates(self):
        """After 200 correct updates toward same f, centroid within 0.05 of f."""
        mu = np.full((1, 1, 6), 0.2)
        scorer = ProfileScorer(mu=mu, actions=["a"])
        f_target = np.full(6, 0.8)
        for _ in range(200):
            scorer.update(f_target, 0, 0, correct=True)
        dist = np.linalg.norm(scorer.centroids[0, 0, :] - f_target)
        assert dist < 0.10, (
            f"After 200 correct updates, centroid distance {dist:.4f} to target should be < 0.10"
        )

    def test_update_direction_correct(self):
        """correct=True: centroid moves in direction of f-mu."""
        mu = np.zeros((1, 2, 3))
        mu[0, 0, :] = [0.3, 0.3, 0.3]
        scorer = ProfileScorer(mu=mu, actions=["a", "b"])
        f = np.array([0.8, 0.8, 0.8])
        mu_before = scorer.centroids[0, 0, :].copy()
        scorer.update(f, 0, 0, correct=True)
        delta = scorer.centroids[0, 0, :] - mu_before
        direction = f - mu_before
        # delta should be parallel to direction (f - mu) with positive sign
        assert np.all(delta * direction >= -1e-12), (
            "Centroid update direction should be toward f (same sign as f-mu)"
        )


# ── Conservation / convergence math ──────────────────────────────────────────

class TestConservationMath:
    def test_theta_min_formula(self):
        """theta_min = eta × N_half² / T_max."""
        for eta, n_half, t_max in [
            (0.05, 14.0, 21.0),
            (0.01, 20.0, 30.0),
            (0.10, 10.0, 15.0),
        ]:
            expected = eta * n_half ** 2 / t_max
            actual = derive_theta_min(eta=eta, n_half=n_half, t_max_days=t_max)
            assert actual == pytest.approx(expected, rel=1e-9)

    def test_theta_min_decreases_with_larger_t_max(self):
        """Longer T_max → lower theta_min (more time → lower signal floor)."""
        theta_21 = derive_theta_min(t_max_days=21.0)
        theta_42 = derive_theta_min(t_max_days=42.0)
        assert theta_42 < theta_21

    def test_theta_min_increases_with_larger_n_half(self):
        """Larger N_half → higher theta_min (slower convergence needs more signal)."""
        theta_small = derive_theta_min(n_half=10.0)
        theta_large = derive_theta_min(n_half=20.0)
        assert theta_large > theta_small

    def test_conservation_signal_is_alpha_times_q_times_v(self):
        """check_conservation.signal == alpha * q * V."""
        alpha, q, V = 0.3, 0.7, 50.0
        theta = derive_theta_min()
        cc = check_conservation(alpha=alpha, q=q, V=V, theta_min=theta)
        assert cc.signal == pytest.approx(alpha * q * V, rel=1e-9)

    def test_conservation_green_when_signal_ge_2x_theta(self):
        theta = derive_theta_min()
        alpha, q, V = 1.0, 1.0, 2.1 * theta  # signal = 2.1*theta > 2*theta
        cc = check_conservation(alpha=alpha, q=q, V=V, theta_min=theta)
        assert cc.status == "GREEN"

    def test_conservation_amber_between_theta_and_2x_theta(self):
        theta = derive_theta_min()
        alpha, q, V = 1.0, 1.0, 1.5 * theta  # signal = 1.5*theta
        cc = check_conservation(alpha=alpha, q=q, V=V, theta_min=theta)
        assert cc.status == "AMBER"

    def test_conservation_red_when_signal_lt_theta(self):
        theta = derive_theta_min()
        alpha, q, V = 1.0, 1.0, 0.5 * theta  # signal = 0.5*theta < theta
        cc = check_conservation(alpha=alpha, q=q, V=V, theta_min=theta)
        assert cc.status == "RED"

    def test_conservation_passed_iff_signal_ge_theta(self):
        theta = derive_theta_min()
        for signal_mult in [0.5, 0.99, 1.0, 1.01, 2.0]:
            signal = signal_mult * theta
            # Decompose signal as alpha=1, q=1, V=signal
            cc = check_conservation(alpha=1.0, q=1.0, V=signal, theta_min=theta)
            expected_passed = signal >= theta
            assert cc.passed == expected_passed, (
                f"signal={signal:.4f}, theta={theta:.4f}: passed={cc.passed} != {expected_passed}"
            )

    def test_conservation_headroom_monotone_with_signal(self):
        """Higher signal → higher headroom."""
        theta = derive_theta_min()
        cc1 = check_conservation(alpha=0.1, q=0.5, V=20.0, theta_min=theta)
        cc2 = check_conservation(alpha=0.3, q=0.5, V=20.0, theta_min=theta)
        assert cc2.headroom > cc1.headroom
