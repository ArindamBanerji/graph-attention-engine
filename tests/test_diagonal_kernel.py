"""
tests/test_diagonal_kernel.py — DiagonalKernel sigma-based workflow.

Validates the sigma → W = 1/σ² → normalized weights pipeline used by
CLAIM-64 and SVM-003b/004b. DiagonalKernel accepts sigma directly; the
constructor computes W = 1/σ² and normalises to weights = W / W.max().

Normalization rule (V-MV-KERNEL, SVM-004):
  self.weights = W / W.max()  where W = 1/σ²
  compute_gradient() uses these [0,1]-normalized weights directly.
  _W_baseline_max = W.max() is frozen at construction and captures
  absolute signal scale — critical for enrichment validation.
"""

import numpy as np
import pytest
import gae


class TestDiagonalKernelSigmaWorkflow:

    def test_low_sigma_gets_higher_weight(self):
        """
        Lower σ → higher W = 1/σ² → higher importance weight.

        Factor 0 (σ=0.05) is most reliable → highest weight (= 1.0 after normalization).
        Factor 5 (σ=0.30) is noisiest     → lowest weight.

        CLAIM-64 / SVM-004: this ordering drives the +13.2pp accuracy gain.
        """
        sigma = np.array([0.05, 0.25, 0.15, 0.20, 0.10, 0.30])
        kernel = gae.DiagonalKernel(sigma)

        assert kernel.weights[0] > kernel.weights[5], (
            f"σ=0.05 weight {kernel.weights[0]:.4f} must exceed σ=0.30 weight "
            f"{kernel.weights[5]:.4f}: lower noise → higher importance"
        )
        assert kernel.weights[0] == pytest.approx(kernel.weights.max()), (
            "σ=0.05 (lowest noise) must have the maximum weight after normalization"
        )

    def test_normalization_anchored_to_baseline(self):
        """
        _W_baseline_max captures absolute signal scale, not just relative ordering.

        When σ is halved (enrichment), W = 1/σ² increases 4×.
        The enriched kernel's _W_baseline_max must be 4× the original.
        This is the anchor the enrichment advisor uses to measure gain magnitude.

        CRITICAL: both kernels normalize to [0,1] weights (directional learning
        unchanged), but _W_baseline_max preserves the absolute scale difference.
        """
        sigma = np.array([0.10, 0.20, 0.15, 0.25, 0.12, 0.18])
        kernel = gae.DiagonalKernel(sigma)

        sigma_enriched = sigma * 0.5                      # σ halved → W quadrupled
        kernel_enriched = gae.DiagonalKernel(sigma_enriched)

        assert kernel_enriched._W_baseline_max > kernel._W_baseline_max, (
            f"Enriched _W_baseline_max {kernel_enriched._W_baseline_max:.1f} must exceed "
            f"original {kernel._W_baseline_max:.1f}: σ halved → W quadrupled"
        )
        expected_ratio = kernel_enriched._W_baseline_max / kernel._W_baseline_max
        assert abs(expected_ratio - 4.0) < 1e-9, (
            f"W ratio must be 4.0 (σ halved → W = 4×W_orig). Got {expected_ratio:.6f}"
        )

    def test_diagonal_kernel_in_gae_api(self):
        """
        API surface test — catches import drift that broke SVM-003/004.

        Verifies DiagonalKernel is importable from top-level gae namespace,
        accepts sigma values, and has the expected attributes for CLAIM-64.
        """
        assert hasattr(gae, "DiagonalKernel"), (
            "gae.DiagonalKernel not found — import drift that broke SVM-003/004"
        )

        sigma = np.array([0.1, 0.2, 0.15, 0.25, 0.12, 0.18])
        k = gae.DiagonalKernel(sigma)

        assert k.weights is not None
        assert len(k.weights) == 6
        assert k._W_baseline_max is not None
        # _W_baseline_max = max(1/σ²) = 1/0.1² = 100.0
        assert k._W_baseline_max == pytest.approx(100.0)
        # weights are normalized: max = 1.0
        assert k.weights.max() == pytest.approx(1.0)

    def test_low_sigma_factor_dominates_distance(self):
        """
        Low-σ factor must contribute MORE to distance than high-σ factor.
        This is the core behavioral guarantee — not just weight ordering.
        """
        sigma = np.array([0.10, 0.50, 0.30, 0.40, 0.20, 0.45])
        kernel = gae.DiagonalKernel(sigma)
        mu = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        # f1: only factor 0 (low σ=0.10) differs from mu
        f1 = np.array([0.8, 0.5, 0.5, 0.5, 0.5, 0.5])
        # f2: only factor 1 (high σ=0.50) differs from mu — same magnitude
        f2 = np.array([0.5, 0.8, 0.5, 0.5, 0.5, 0.5])

        d1 = kernel.compute_distance(f1, mu.reshape(1, -1))
        d2 = kernel.compute_distance(f2, mu.reshape(1, -1))

        # Low-σ factor (factor 0) must produce larger distance
        assert d1[0] > d2[0], (
            f"Low-σ factor should dominate distance. "
            f"d1={d1[0]:.4f} d2={d2[0]:.4f}"
        )

    def test_gradient_larger_for_low_sigma_factor(self):
        """
        Gradient magnitude must be larger for low-σ factors.
        Confirms the learning update is correctly weighted.
        """
        sigma = np.array([0.10, 0.50, 0.30, 0.40, 0.20, 0.45])
        kernel = gae.DiagonalKernel(sigma)
        f  = np.array([0.8, 0.8, 0.5, 0.5, 0.5, 0.5])
        mu = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        # Both factors 0 and 1 differ by same magnitude (0.3)
        # Factor 0 (σ=0.10) must have larger gradient than factor 1 (σ=0.50)
        grad = kernel.compute_gradient(f, mu)
        assert abs(grad[0]) > abs(grad[1]), (
            f"Low-σ factor gradient should be larger. "
            f"grad[0]={grad[0]:.4f} grad[1]={grad[1]:.4f}"
        )

    def test_raw_weights_are_unnormalized(self):
        """
        raw_weights returns W = 1/σ² before normalization.
        Unlike .weights (max=1.0), raw_weights preserves absolute scale.
        """
        sigma = np.array([0.1, 0.2, 0.15, 0.25, 0.12, 0.18])
        kernel = gae.DiagonalKernel(sigma)
        raw = kernel.raw_weights

        expected_raw_max = 1.0 / (0.1 ** 2)   # = 100.0
        assert abs(raw.max() - expected_raw_max) < 0.01
        assert abs(raw[0] - 100.0) < 0.01    # factor 0: 1/0.1² = 100
        assert abs(raw[3] - 16.0) < 0.01     # factor 3: 1/0.25² = 16
        # raw_weights must NOT be normalized to [0,1]
        assert raw.max() > 1.0
