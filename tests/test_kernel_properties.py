"""
Kernel property tests for GAE.

Tests DiagonalKernel and L2Kernel mathematical properties, and
KernelSelector rule-based recommendation. Complementary to
test_kernels.py and test_kernel_selector.py, which cover integration
and empirical Phase 4 recommendation; these focus on mathematical
properties and Phase 2 rule boundaries.
"""

from __future__ import annotations

import numpy as np
import pytest

from gae.kernels import L2Kernel, DiagonalKernel
from gae.kernel_selector import KernelSelector


# ── DiagonalKernel mathematical properties ────────────────────────────────────

class TestDiagonalKernelMathProperties:
    def test_weights_in_zero_one(self):
        sigma = np.array([0.1, 0.3, 0.5, 1.0, 2.0])
        k = DiagonalKernel(sigma)
        assert k.weights.min() >= 0.0
        assert k.weights.max() <= 1.0 + 1e-10

    def test_max_weight_is_one(self):
        sigma = np.array([0.1, 0.3, 0.5])
        k = DiagonalKernel(sigma)
        assert k.weights.max() == pytest.approx(1.0, abs=1e-9)

    def test_lower_sigma_gives_higher_weight(self):
        """Dim 0 is 10× less noisy → should have higher weight."""
        sigma = np.array([0.1, 1.0])
        k = DiagonalKernel(sigma)
        assert k.weights[0] > k.weights[1]

    def test_equal_sigma_gives_equal_weights(self):
        sigma = np.full(4, 0.5)
        k = DiagonalKernel(sigma)
        np.testing.assert_allclose(k.weights, np.ones(4), atol=1e-9)

    def test_distance_non_negative(self):
        sigma = np.array([0.3, 0.5, 0.2])
        k = DiagonalKernel(sigma)
        rng = np.random.default_rng(0)
        for _ in range(50):
            f = rng.uniform(0, 1, 3)
            mu = rng.uniform(0, 1, (4, 3))
            d = k.compute_distance(f, mu)
            assert np.all(d >= 0.0)

    def test_distance_zero_when_f_equals_centroid(self):
        sigma = np.array([0.2, 0.4, 0.6])
        k = DiagonalKernel(sigma)
        f = np.array([0.3, 0.5, 0.7])
        mu = f.reshape(1, -1)
        d = k.compute_distance(f, mu)
        assert d[0] == pytest.approx(0.0, abs=1e-10)

    def test_gradient_formula_matches_weighted_residual(self):
        """gradient = weights * (f - mu) because max(weights) = 1."""
        sigma = np.array([0.2, 0.4, 0.6, 0.8])
        k = DiagonalKernel(sigma)
        f = np.array([0.7, 0.3, 0.5, 0.9])
        mu = np.array([0.1, 0.6, 0.2, 0.4])
        g = k.compute_gradient(f, mu)
        expected = k.weights * (f - mu)
        np.testing.assert_allclose(g, expected, rtol=1e-9)

    def test_uniform_sigma_gradient_equals_l2_gradient(self):
        """When sigma is uniform, DiagonalKernel gradient equals L2 gradient."""
        sigma = np.full(4, 0.5)
        diag = DiagonalKernel(sigma)
        l2 = L2Kernel()
        f = np.array([0.2, 0.8, 0.3, 0.7])
        mu = np.array([0.5, 0.5, 0.5, 0.5])
        g_diag = diag.compute_gradient(f, mu)
        g_l2 = l2.compute_gradient(f, mu)
        np.testing.assert_allclose(g_diag, g_l2, rtol=1e-9)

    def test_sigma_zero_raises_value_error(self):
        with pytest.raises(ValueError):
            DiagonalKernel(np.array([0.0, 0.5]))

    def test_sigma_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            DiagonalKernel(np.array([-0.1, 0.5]))

    def test_sigma_2d_raises(self):
        with pytest.raises((AssertionError, ValueError)):
            DiagonalKernel(np.array([[0.5, 0.5]]))

    def test_raw_weights_formula(self):
        """raw_weights = 1/sigma²."""
        sigma = np.array([0.5, 2.0, 1.0])
        k = DiagonalKernel(sigma)
        np.testing.assert_allclose(k.raw_weights, 1.0 / sigma**2, rtol=1e-9)

    def test_raw_weights_larger_than_normalized(self):
        """raw_weights >= weights: normalization only divides."""
        sigma = np.array([0.3, 0.6, 1.2])
        k = DiagonalKernel(sigma)
        assert k.raw_weights.max() >= k.weights.max() - 1e-12

    def test_noise_ratio_is_positive(self):
        """noise_ratio must be a positive finite float for any valid sigma."""
        sigma = np.full(4, 0.5)
        k = DiagonalKernel(sigma)
        assert k.noise_ratio > 0.0
        assert np.isfinite(k.noise_ratio)

    def test_noise_ratio_higher_for_more_heterogeneous_sigma(self):
        """More heterogeneous sigma → higher noise_ratio."""
        k_uniform = DiagonalKernel(np.array([0.5, 0.5]))
        k_hetero  = DiagonalKernel(np.array([0.1, 1.0]))
        assert k_hetero.noise_ratio > k_uniform.noise_ratio

    def test_refresh_weights_returns_new_instance(self):
        sigma = np.array([0.2, 0.4, 0.6])
        k = DiagonalKernel(sigma)
        k2 = k.refresh_weights(np.array([0.1, 0.3, 0.5]))
        assert k2 is not k

    def test_refresh_weights_does_not_mutate_original(self):
        sigma = np.array([0.2, 0.4, 0.6])
        k = DiagonalKernel(sigma)
        weights_orig = k.weights.copy()
        k.refresh_weights(np.array([0.1, 0.3, 0.5]))
        np.testing.assert_array_equal(k.weights, weights_orig)

    def test_refresh_weights_uniform_gives_equal_weights(self):
        sigma = np.array([0.2, 0.4])
        k = DiagonalKernel(sigma)
        k2 = k.refresh_weights(np.array([0.5, 0.5]))
        np.testing.assert_allclose(k2.weights, np.ones(2), atol=1e-9)

    def test_distance_higher_with_low_weight_dim_suppressed(self):
        """
        With unequal weights, the high-noise dimension contributes less.
        Compare distance with near-uniform weights vs skewed weights.
        """
        f = np.array([0.0, 0.0])
        mu = np.array([[1.0, 1.0]])
        # Equal sigma → both dims contribute
        k_eq = DiagonalKernel(np.array([0.5, 0.5]))
        d_eq = k_eq.compute_distance(f, mu)[0]
        # Skewed: dim 1 is 100× noisier → nearly zeroed out
        k_sk = DiagonalKernel(np.array([0.5, 50.0]))
        d_sk = k_sk.compute_distance(f, mu)[0]
        assert d_sk < d_eq


# ── L2Kernel mathematical properties ─────────────────────────────────────────

class TestL2KernelMathProperties:
    def test_self_distance_zero(self):
        k = L2Kernel()
        f = np.array([0.3, 0.5, 0.7])
        mu = f.reshape(1, -1)
        d = k.compute_distance(f, mu)
        assert d[0] == pytest.approx(0.0, abs=1e-12)

    def test_distance_symmetric(self):
        k = L2Kernel()
        a = np.array([0.1, 0.2, 0.3])
        b = np.array([0.4, 0.5, 0.6])
        d_ab = k.compute_distance(a, b.reshape(1, -1))[0]
        d_ba = k.compute_distance(b, a.reshape(1, -1))[0]
        assert d_ab == pytest.approx(d_ba, rel=1e-9)

    def test_gradient_is_f_minus_mu(self):
        k = L2Kernel()
        f = np.array([0.7, 0.3, 0.5])
        mu = np.array([0.2, 0.8, 0.1])
        g = k.compute_gradient(f, mu)
        np.testing.assert_allclose(g, f - mu, rtol=1e-12)

    def test_distance_scales_with_magnitude(self):
        k = L2Kernel()
        f = np.zeros(4)
        d1 = k.compute_distance(f, np.array([[1.0, 0.0, 0.0, 0.0]]))[0]
        d2 = k.compute_distance(f, np.array([[2.0, 0.0, 0.0, 0.0]]))[0]
        assert d2 > d1

    def test_gradient_points_toward_f(self):
        """Gradient f-mu points from mu toward f."""
        k = L2Kernel()
        f = np.array([0.9, 0.9])
        mu = np.array([0.1, 0.1])
        g = k.compute_gradient(f, mu)
        assert np.all(g > 0)

    def test_gradient_zero_when_f_equals_mu(self):
        k = L2Kernel()
        f = np.array([0.5, 0.3, 0.7])
        g = k.compute_gradient(f, f.copy())
        np.testing.assert_allclose(g, np.zeros(3), atol=1e-12)


# ── KernelSelector Phase 2 rule boundaries ────────────────────────────────────

class TestKernelSelectorPreliminaryRecommendation:
    def test_uniform_sigma_recommends_l2(self):
        """Uniform noise (ratio=1.0) and zero correlation → L2."""
        ks = KernelSelector(d=6, sigma_per_factor=np.full(6, 0.5), correlation_max=0.0)
        rec = ks.preliminary_recommendation()
        assert rec.recommended_kernel == "l2"
        assert rec.method == "rule"

    def test_high_noise_ratio_recommends_diagonal(self):
        """sigma_max/sigma_min = 20 > 1.5 → diagonal."""
        sigma = np.array([0.1, 2.0, 0.1, 0.1, 0.1, 0.1])
        ks = KernelSelector(d=6, sigma_per_factor=sigma, correlation_max=0.0)
        rec = ks.preliminary_recommendation()
        assert rec.recommended_kernel == "diagonal"

    def test_high_correlation_recommends_shrinkage(self):
        """rho >= 0.30 → shrinkage."""
        ks = KernelSelector(d=4, sigma_per_factor=np.full(4, 0.5), correlation_max=0.35)
        rec = ks.preliminary_recommendation()
        assert rec.recommended_kernel == "shrinkage"

    def test_noise_ratio_below_threshold_stays_l2(self):
        """noise_ratio = 1.4 < 1.5 → L2 (just below threshold)."""
        # max=0.14, min=0.1 → ratio = 1.4
        ks = KernelSelector(d=2, sigma_per_factor=np.array([0.1, 0.14]),
                            correlation_max=0.0)
        rec = ks.preliminary_recommendation()
        assert rec.recommended_kernel == "l2"

    def test_noise_ratio_above_threshold_picks_diagonal(self):
        """noise_ratio = 2.0 > 1.5 → diagonal."""
        ks = KernelSelector(d=2, sigma_per_factor=np.array([0.1, 0.2]),
                            correlation_max=0.0)
        rec = ks.preliminary_recommendation()
        assert rec.recommended_kernel == "diagonal"

    def test_correlation_below_02_with_high_noise_is_diagonal(self):
        """rho=0.1 < 0.2 but noise_ratio > 1.5 → diagonal."""
        ks = KernelSelector(d=4, sigma_per_factor=np.array([0.1, 0.5, 0.1, 0.1]),
                            correlation_max=0.1)
        rec = ks.preliminary_recommendation()
        assert rec.recommended_kernel == "diagonal"

    def test_recommendation_has_reason_string(self):
        ks = KernelSelector(d=4, sigma_per_factor=np.full(4, 0.5), correlation_max=0.0)
        rec = ks.preliminary_recommendation()
        assert isinstance(rec.reason, str)
        assert len(rec.reason) > 0

    def test_rule_recommendation_method_is_rule(self):
        ks = KernelSelector(d=4, sigma_per_factor=np.full(4, 0.5), correlation_max=0.0)
        rec = ks.preliminary_recommendation()
        assert rec.method == "rule"

    def test_three_kernels_always_built(self):
        ks = KernelSelector(d=4, sigma_per_factor=np.full(4, 0.5))
        assert "l2" in ks.kernels
        assert "diagonal" in ks.kernels
        assert "shrinkage" in ks.kernels

    def test_shrinkage_threshold_at_rho_030(self):
        """rho = 0.30 exactly → shrinkage."""
        ks = KernelSelector(d=4, sigma_per_factor=np.full(4, 0.5), correlation_max=0.30)
        rec = ks.preliminary_recommendation()
        assert rec.recommended_kernel == "shrinkage"

    def test_rho_just_below_030_not_shrinkage(self):
        """rho = 0.29 < 0.30 → not shrinkage (l2 or diagonal depending on noise)."""
        ks = KernelSelector(d=4, sigma_per_factor=np.full(4, 0.5), correlation_max=0.29)
        rec = ks.preliminary_recommendation()
        assert rec.recommended_kernel != "shrinkage"


# ── KernelSelector record_comparison ─────────────────────────────────────────

class TestKernelSelectorRecordComparison:
    def _make_ks_and_mu(self, d=6, n_cat=2, n_act=3, seed=42):
        ks = KernelSelector(d=d, sigma_per_factor=np.full(d, 0.5))
        rng = np.random.default_rng(seed)
        mu = rng.uniform(0.1, 0.9, (n_cat, n_act, d))
        actions = [f"a{i}" for i in range(n_act)]
        return ks, mu, actions

    def test_record_comparison_returns_dict(self):
        ks, mu, actions = self._make_ks_and_mu()
        preds = ks.record_comparison(np.full(6, 0.5), 0, mu, 0, actions)
        assert isinstance(preds, dict)

    def test_record_comparison_has_all_kernels(self):
        ks, mu, actions = self._make_ks_and_mu()
        preds = ks.record_comparison(np.full(6, 0.5), 0, mu, 0, actions)
        assert "l2" in preds
        assert "diagonal" in preds
        assert "shrinkage" in preds

    def test_record_comparison_increments_total(self):
        ks, mu, actions = self._make_ks_and_mu()
        assert ks.scores["l2"].total_decisions == 0
        ks.record_comparison(np.full(6, 0.5), 0, mu, 0, actions)
        assert ks.scores["l2"].total_decisions == 1
        assert ks.scores["diagonal"].total_decisions == 1

    def test_record_comparison_predictions_are_valid_indices(self):
        ks, mu, actions = self._make_ks_and_mu(n_act=3)
        preds = ks.record_comparison(np.full(6, 0.5), 0, mu, 0, actions)
        for name, idx in preds.items():
            assert 0 <= idx < 3, f"Kernel {name} predicted invalid index {idx}"

    def test_record_comparison_multiple_calls_accumulate(self):
        ks, mu, actions = self._make_ks_and_mu()
        rng = np.random.default_rng(7)
        for _ in range(10):
            f = rng.uniform(0.1, 0.9, 6)
            ks.record_comparison(f, 0, mu, 0, actions)
        assert ks.scores["l2"].total_decisions == 10
