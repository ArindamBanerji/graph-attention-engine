"""
Unit tests for gae/kernels.py — L2Kernel, DiagonalKernel, ScoringKernel protocol.

Reference: docs/gae_design_v5.md §9; v6.0 kernel roadmap.
"""

import numpy as np
import pytest

from gae.kernels import L2Kernel, DiagonalKernel, ScoringKernel


# ------------------------------------------------------------------ #
# ScoringKernel protocol                                              #
# ------------------------------------------------------------------ #

class TestScoringKernelProtocol:
    def test_l2kernel_is_protocol_instance(self):
        assert isinstance(L2Kernel(), ScoringKernel)

    def test_diagonal_kernel_is_protocol_instance(self):
        weights = np.ones(4)
        assert isinstance(DiagonalKernel(weights), ScoringKernel)

    def test_custom_class_satisfies_protocol(self):
        class MyKernel:
            def compute_distance(self, f, mu):
                return np.zeros(mu.shape[0])

            def compute_gradient(self, f, mu):
                return f - mu

        assert isinstance(MyKernel(), ScoringKernel)


# ------------------------------------------------------------------ #
# L2Kernel                                                            #
# ------------------------------------------------------------------ #

class TestL2KernelDistance:
    def test_zero_distance_at_centroid(self):
        k = L2Kernel()
        f = np.array([0.2, 0.5, 0.8])
        mu = np.array([[0.2, 0.5, 0.8], [0.0, 0.0, 0.0]])
        d = k.compute_distance(f, mu)
        assert d[0] == pytest.approx(0.0)

    def test_output_shape(self):
        k = L2Kernel()
        f = np.zeros(6)
        mu = np.zeros((4, 6))
        d = k.compute_distance(f, mu)
        assert d.shape == (4,)

    def test_positive_distances(self):
        k = L2Kernel()
        f = np.array([0.1, 0.9])
        mu = np.array([[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]])
        d = k.compute_distance(f, mu)
        assert np.all(d >= 0.0)

    def test_known_value(self):
        k = L2Kernel()
        f = np.array([1.0, 0.0])
        mu = np.array([[0.0, 1.0]])
        d = k.compute_distance(f, mu)
        # (1-0)^2 + (0-1)^2 = 2.0
        assert d[0] == pytest.approx(2.0)

    def test_single_action(self):
        k = L2Kernel()
        f = np.array([0.3, 0.7, 0.5])
        mu = np.array([[0.3, 0.7, 0.5]])
        d = k.compute_distance(f, mu)
        assert d.shape == (1,)
        assert d[0] == pytest.approx(0.0)

    def test_symmetry(self):
        k = L2Kernel()
        f = np.array([0.2, 0.4, 0.6])
        mu = np.array([[0.8, 0.6, 0.4]])
        d1 = k.compute_distance(f, mu)
        # Reverse: distance from mu[0] to f should be identical
        d2 = k.compute_distance(mu[0], f.reshape(1, -1))
        assert d1[0] == pytest.approx(d2[0])


class TestL2KernelGradient:
    def test_gradient_direction(self):
        k = L2Kernel()
        f = np.array([0.8, 0.2, 0.5])
        mu = np.array([0.3, 0.7, 0.5])
        g = k.compute_gradient(f, mu)
        expected = np.array([0.5, -0.5, 0.0])
        np.testing.assert_allclose(g, expected)

    def test_gradient_output_shape(self):
        k = L2Kernel()
        f = np.zeros(6)
        mu = np.ones(6)
        g = k.compute_gradient(f, mu)
        assert g.shape == (6,)

    def test_gradient_zero_at_centroid(self):
        k = L2Kernel()
        f = np.array([0.5, 0.5])
        mu = np.array([0.5, 0.5])
        g = k.compute_gradient(f, mu)
        np.testing.assert_allclose(g, np.zeros(2))

    def test_gradient_equals_f_minus_mu(self):
        k = L2Kernel()
        rng = np.random.default_rng(42)
        f = rng.random(8)
        mu = rng.random(8)
        g = k.compute_gradient(f, mu)
        np.testing.assert_allclose(g, f - mu)


# ------------------------------------------------------------------ #
# DiagonalKernel                                                      #
# ------------------------------------------------------------------ #

class TestDiagonalKernelInit:
    def test_weights_stored(self):
        w = np.array([1.0, 2.0, 0.5])
        k = DiagonalKernel(w)
        np.testing.assert_array_equal(k.weights, w)

    def test_weights_converted_to_float64(self):
        w = np.array([1, 2, 3], dtype=np.int32)
        k = DiagonalKernel(w)
        assert k.weights.dtype == np.float64

    def test_weights_must_be_1d(self):
        with pytest.raises(AssertionError):
            DiagonalKernel(np.ones((3, 3)))


class TestDiagonalKernelDistance:
    def test_unit_weights_equals_l2(self):
        l2 = L2Kernel()
        diag = DiagonalKernel(np.ones(4))
        f = np.array([0.1, 0.4, 0.7, 0.9])
        mu = np.array([[0.5, 0.5, 0.5, 0.5], [0.1, 0.4, 0.7, 0.9]])
        np.testing.assert_allclose(
            l2.compute_distance(f, mu),
            diag.compute_distance(f, mu),
        )

    def test_zero_weight_ignores_dimension(self):
        k = DiagonalKernel(np.array([1.0, 0.0]))
        f = np.array([0.0, 0.9])   # second dim differs but weight=0
        mu = np.array([[0.0, 0.1]])  # second dim would be large L2
        d = k.compute_distance(f, mu)
        assert d[0] == pytest.approx(0.0)

    def test_known_weighted_value(self):
        k = DiagonalKernel(np.array([2.0, 0.5]))
        f = np.array([1.0, 0.0])
        mu = np.array([[0.0, 1.0]])
        # 2*(1-0)^2 + 0.5*(0-1)^2 = 2 + 0.5 = 2.5
        d = k.compute_distance(f, mu)
        assert d[0] == pytest.approx(2.5)

    def test_output_shape(self):
        k = DiagonalKernel(np.ones(5))
        f = np.zeros(5)
        mu = np.zeros((3, 5))
        d = k.compute_distance(f, mu)
        assert d.shape == (3,)

    def test_higher_weight_increases_distance(self):
        f = np.array([1.0, 0.0])
        mu = np.array([[0.0, 0.0]])
        k1 = DiagonalKernel(np.array([1.0, 1.0]))
        k2 = DiagonalKernel(np.array([4.0, 1.0]))
        assert k2.compute_distance(f, mu)[0] > k1.compute_distance(f, mu)[0]


class TestDiagonalKernelGradient:
    def test_unit_weights_equals_l2_gradient(self):
        l2 = L2Kernel()
        diag = DiagonalKernel(np.ones(5))
        f = np.random.default_rng(7).random(5)
        mu = np.random.default_rng(8).random(5)
        np.testing.assert_allclose(
            l2.compute_gradient(f, mu),
            diag.compute_gradient(f, mu),
        )

    def test_zero_weight_zeros_gradient_component(self):
        k = DiagonalKernel(np.array([1.0, 0.0, 1.0]))
        f = np.array([0.5, 0.9, 0.3])
        mu = np.array([0.1, 0.1, 0.1])
        g = k.compute_gradient(f, mu)
        assert g[1] == pytest.approx(0.0)

    def test_gradient_scaled_by_weight(self):
        # After w_max normalisation: g = (W / w_max) * (f - mu).
        # w_max=3.0 → g[0]=(3/3)*1=1.0, g[1]=(1/3)*1=0.333.
        # Reliable factor (w=3) still learns faster than noisy factor (w=1).
        k = DiagonalKernel(np.array([3.0, 1.0]))
        f = np.array([1.0, 1.0])
        mu = np.array([0.0, 0.0])
        g = k.compute_gradient(f, mu)
        assert g[0] == pytest.approx(1.0)            # w_max factor: full step
        assert g[1] == pytest.approx(1.0 / 3.0)     # noisier factor: proportionally smaller
        assert g[0] > g[1]                           # reliable still learns faster

    def test_gradient_output_shape(self):
        k = DiagonalKernel(np.ones(6))
        g = k.compute_gradient(np.zeros(6), np.ones(6))
        assert g.shape == (6,)


# ------------------------------------------------------------------ #
# KernelWeightRefresh — DiagonalKernel + CovarianceEstimator + ProfileScorer
# ------------------------------------------------------------------ #

class TestKernelWeightRefresh:
    """Tests for the V-CGA-FROZEN gap closure: refresh_weights / kernel_weight_refresh."""

    def _make_scorer(self, use_diagonal=True):
        """Helper: ProfileScorer with DiagonalKernel (or L2Kernel) and 1 category, 3 actions, 4 factors."""
        mu = np.full((1, 3, 4), 0.5)
        actions = ["a0", "a1", "a2"]
        from gae.kernels import DiagonalKernel, L2Kernel
        from gae.profile_scorer import ProfileScorer
        kernel = DiagonalKernel(np.ones(4)) if use_diagonal else L2Kernel()
        return ProfileScorer(mu=mu, actions=actions, scoring_kernel=kernel)

    def _make_estimator_with_n(self, d, n):
        """Helper: CovarianceEstimator fed with n uniform observations."""
        from gae.covariance import CovarianceEstimator
        est = CovarianceEstimator(d=d)
        rng = np.random.default_rng(42)
        for _ in range(n):
            est.update(rng.uniform(0.1, 0.9, size=d))
        return est

    def test_kernel_weight_refresh_updates_weights(self):
        """refresh_weights() produces new DiagonalKernel with 1/σ² weights; original unchanged."""
        from gae.covariance import CovarianceEstimator
        est = self._make_estimator_with_n(d=4, n=100)
        sigma = est.get_per_factor_sigma()
        assert sigma is not None, "Expected sigma with 100 samples"
        assert sigma.shape == (4,)

        original_weights = np.ones(4)
        k_orig = DiagonalKernel(original_weights.copy())
        k_new = k_orig.refresh_weights(sigma)

        # New instance returned — not the same object
        assert k_new is not k_orig

        # Original weights unchanged (immutable design)
        np.testing.assert_array_equal(k_orig.weights, original_weights)

        # New weights = 1 / σ², clipped at 1e-6
        expected = 1.0 / np.maximum(sigma, 1e-6) ** 2
        np.testing.assert_allclose(k_new.weights, expected, rtol=1e-9)

    def test_kernel_weight_refresh_safe_during_freeze(self):
        """kernel_weight_refresh() returns True and updates kernel even when frozen."""
        scorer = self._make_scorer(use_diagonal=True)
        scorer.freeze()
        assert scorer._frozen is True

        mu_before = scorer.mu.copy()
        est = self._make_estimator_with_n(d=4, n=100)

        result = scorer.kernel_weight_refresh(est)
        assert result is True

        # Centroid tensor must not change — kernel refresh only
        np.testing.assert_array_equal(scorer.mu, mu_before)

        # Kernel weights must have changed (were all-ones, now 1/σ²)
        sigma = est.get_per_factor_sigma()
        expected_weights = 1.0 / np.maximum(sigma, 1e-6) ** 2
        np.testing.assert_allclose(scorer.scoring_kernel.weights, expected_weights, rtol=1e-9)

    def test_kernel_weight_refresh_insufficient_data(self):
        """get_per_factor_sigma() returns None with < 50 observations; refresh returns False."""
        from gae.covariance import CovarianceEstimator
        est = self._make_estimator_with_n(d=4, n=10)   # below MIN_SAMPLES_FOR_SIGMA

        sigma = est.get_per_factor_sigma()
        assert sigma is None, "Expected None for n=10 < 50"

        scorer = self._make_scorer(use_diagonal=True)
        original_weights = scorer.scoring_kernel.weights.copy()
        result = scorer.kernel_weight_refresh(est)

        assert result is False
        np.testing.assert_array_equal(scorer.scoring_kernel.weights, original_weights)

    def test_kernel_weight_refresh_l2_noop(self):
        """kernel_weight_refresh() returns False for L2Kernel without raising."""
        scorer = self._make_scorer(use_diagonal=False)
        est = self._make_estimator_with_n(d=4, n=100)

        result = scorer.kernel_weight_refresh(est)
        assert result is False  # L2Kernel has no weights to refresh


# ------------------------------------------------------------------ #
# CLAIM-60 integration: σ reduction → weight refresh → accuracy gain #
# ------------------------------------------------------------------ #

class TestClaim60SigmaReductionAccuracy:
    """
    Proves CLAIM-60: lower σ on informative factors → higher accuracy
    via DiagonalKernel automatic reweighting.

    Setup: 6 SOC factors, 4 actions, 1 category.
      disc_dims [2,4]   = threat_intel_enrichment, pattern_history (true signal)
      noise_dims [0,1,3,5] = travel_match, asset_criticality, time_anomaly, device_trust

    Each test alert: disc dims point to correct action, noise dims point to a
    different (confounding) action. With equal weights the noise dims (×4) outweigh
    the signal dims (×2). With W_low the signal dims are weighted 16× higher (400 vs 25)
    so they dominate.
    """

    # Factor layout (SOC canonical):
    #   0=travel_match, 1=asset_criticality, 2=threat_intel,
    #   3=time_anomaly,  4=pattern_history,   5=device_trust
    DISC_DIMS  = [2, 4]
    NOISE_DIMS = [0, 1, 3, 5]

    # Discriminating centroid values per action (indices into DISC_DIMS)
    DISC_VALS = {
        0: [0.9, 0.9],   # escalate:    high ti, high ph
        1: [0.9, 0.1],   # investigate: high ti, low  ph
        2: [0.1, 0.9],   # suppress:    low  ti, high ph
        3: [0.1, 0.1],   # monitor:     low  ti, low  ph
    }

    # Noise-dim centroid values per action (indices into NOISE_DIMS)
    # Chosen so each action has a distinct noise-dim "fingerprint"
    NOISE_VALS = {
        0: [0.9, 0.1, 0.1, 0.1],
        1: [0.1, 0.9, 0.9, 0.9],
        2: [0.9, 0.9, 0.1, 0.9],
        3: [0.1, 0.1, 0.1, 0.1],
    }

    def _build_mu(self):
        """Build centroid tensor (1, 4, 6)."""
        mu = np.zeros((1, 4, 6))
        for act in range(4):
            for i, d in enumerate(self.DISC_DIMS):
                mu[0, act, d] = self.DISC_VALS[act][i]
            for i, d in enumerate(self.NOISE_DIMS):
                mu[0, act, d] = self.NOISE_VALS[act][i]
        return mu

    def _build_scorer(self, sigma_array):
        from gae.profile_scorer import ProfileScorer
        W = 1.0 / sigma_array ** 2
        return ProfileScorer(
            mu=self._build_mu(),
            actions=["escalate", "investigate", "suppress", "monitor"],
            scoring_kernel=DiagonalKernel(W),
        )

    def test_sigma_reduction_improves_accuracy(self):
        """
        Proves CLAIM-60: lower σ on informative factors → higher triage accuracy.

        W_high_noise: all σ = 0.20 → all weights = 25 (uniform).
        W_low_noise:  threat_intel σ = 0.05, pattern_history σ = 0.05
                      → disc weights = 400, noise weights = 25 (ratio 16:1).

        100 alerts: disc dims point to true action, noise dims confound.
        Expected: accuracy(W_low_noise) > accuracy(W_high_noise) by >= 3pp.
        """
        n_factors = 6
        sigma_high = np.full(n_factors, 0.20)
        sigma_low  = np.full(n_factors, 0.20)
        sigma_low[2] = 0.05   # threat_intel
        sigma_low[4] = 0.05   # pattern_history

        scorer_high = self._build_scorer(sigma_high)
        scorer_low  = self._build_scorer(sigma_low)

        rng = np.random.default_rng(42)
        n_alerts = 100
        correct_high = 0
        correct_low  = 0

        for i in range(n_alerts):
            true_action     = i % 4
            confound_action = (true_action + 2) % 4   # maximally different action

            f = np.zeros(n_factors)
            # Disc dims: small noise around true action centroid
            for j, d in enumerate(self.DISC_DIMS):
                f[d] = np.clip(
                    self.DISC_VALS[true_action][j] + rng.normal(0, 0.05), 0.0, 1.0
                )
            # Noise dims: small noise around CONFOUNDING action centroid
            for j, d in enumerate(self.NOISE_DIMS):
                f[d] = np.clip(
                    self.NOISE_VALS[confound_action][j] + rng.normal(0, 0.05), 0.0, 1.0
                )

            if scorer_high.score(f, 0).action_index == true_action:
                correct_high += 1
            if scorer_low.score(f, 0).action_index == true_action:
                correct_low += 1

        acc_high = correct_high / n_alerts
        acc_low  = correct_low  / n_alerts
        delta_pp = (acc_low - acc_high) * 100.0

        assert acc_low > acc_high + 0.03, (
            f"CLAIM-60 not validated: acc_low_noise={acc_low:.3f}, "
            f"acc_high_noise={acc_high:.3f}, delta={delta_pp:.1f}pp (need >= 3pp). "
            f"DiagonalKernel σ-reweighting should favour informative factors."
        )


# ------------------------------------------------------------------ #
# Gradient normalisation fix — V-CLAIM60-S2P oscillation regression  #
# ------------------------------------------------------------------ #

class TestDiagonalGradientNormalisation:
    """
    Tests for the w_max normalisation fix in DiagonalKernel.compute_gradient().
    Before fix: return self.weights * (f - mu)  → η*W*(f-mu) up to 13.9*(f-mu)
    After fix:  return (self.weights/w_max) * (f - mu) → bounded by η*(f-mu)
    """

    def test_diagonal_gradient_bounded_by_eta(self):
        """
        Max step = η × max|G| must not exceed η regardless of weight magnitude.
        SOC asset_criticality W=277.8 was the worst offender before fix.
        """
        eta = 0.05
        kernel = DiagonalKernel(np.array([277.8, 44.4]))
        f  = np.array([0.9, 0.9])
        mu = np.array([0.1, 0.1])
        G = kernel.compute_gradient(f, mu)
        max_step = eta * np.max(np.abs(G))
        assert max_step <= eta + 1e-9, (
            f"Max step {max_step:.6f} exceeds η={eta}. "
            f"Gradient normalisation fix not applied. G={G}"
        )

    def test_diagonal_gradient_reliable_learns_faster(self):
        """
        After normalisation the gradient ratio equals the weight ratio.
        Reliable factor (high W) gets proportionally larger gradient than noisy factor.
        """
        kernel = DiagonalKernel(np.array([277.8, 44.4]))
        f  = np.array([0.8, 0.8])
        mu = np.array([0.5, 0.5])
        G = kernel.compute_gradient(f, mu)
        assert G[0] > G[1], (
            f"Reliable factor (W=277.8) gradient {G[0]:.6f} should exceed "
            f"noisy factor (W=44.4) gradient {G[1]:.6f}"
        )
        expected_ratio = 277.8 / 44.4
        actual_ratio   = G[0] / G[1]
        assert abs(actual_ratio - expected_ratio) < 1e-6, (
            f"Gradient ratio {actual_ratio:.6f} should equal weight ratio "
            f"{expected_ratio:.6f} (proportional learning speed preserved)"
        )

    def test_diagonal_no_oscillation_after_many_updates(self):
        """
        200 update() calls with SOC-like high weights must NOT produce centroid
        oscillation. Before fix: centroids slammed to 0/1 boundary every step.
        After fix: centroids converge smoothly.
        """
        from gae.profile_scorer import ProfileScorer

        sigma = np.array([0.06, 0.07, 0.20])   # SOC-like: asset_crit, time_anomaly, pattern
        W = 1.0 / sigma ** 2                    # [277.8, 204.1, 25.0]

        mu_init = np.full((1, 1, 3), 0.5)
        scorer = ProfileScorer(
            mu=mu_init,
            actions=["escalate"],
            scoring_kernel=DiagonalKernel(W),
        )

        f = np.array([0.8, 0.7, 0.6])
        centroid_history = []

        for _ in range(200):
            scorer.update(f, category_index=0, action_index=0, correct=True)
            centroid_history.append(scorer.mu[0, 0, :].copy())

        final_centroid = scorer.mu[0, 0, :]

        # No dimension stuck at hard boundary
        assert not np.any(final_centroid == 0.0), (
            f"Centroid dimension(s) stuck at 0.0 after 200 updates: {final_centroid}"
        )
        assert not np.any(final_centroid == 1.0), (
            f"Centroid dimension(s) stuck at 1.0 after 200 updates: {final_centroid}"
        )

        # Final 20 steps converged — variance < 0.01 (not oscillating)
        last_20 = np.array(centroid_history[-20:])  # shape (20, 3)
        var_per_dim = np.var(last_20, axis=0)
        assert np.all(var_per_dim < 0.01), (
            f"Centroid oscillating in final 20 steps. "
            f"Per-dim variance: {var_per_dim} (threshold 0.01). "
            f"Final centroid: {final_centroid}"
        )
