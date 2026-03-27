"""
tests/test_diagonal_kernel.py — DiagonalKernel sigma-to-weights workflow.

Validates the sigma → W = 1/σ² → DiagonalKernel pipeline used by
CLAIM-64 and SVM-003b/004b. All three tests use pre-computed W so the
existing weights-based constructor is preserved without breaking any
prior tests.

Normalization rule (V-MV-KERNEL, SVM-004):
  W = 1/σ²
  weights passed to DiagonalKernel may be raw W or W/W.max().
  compute_gradient() always normalises by weights.max() internally.
  _W_baseline_max = weights.max() is frozen at construction and captures
  absolute signal scale — critical for enrichment validation.
"""

import numpy as np
import pytest
import gae


class TestDiagonalKernelSigmaWorkflow:

    def test_low_sigma_gets_higher_weight(self):
        """
        Lower σ → higher W = 1/σ² → higher importance weight.

        Factor 0 (σ=0.05) is most reliable → highest weight (= W.max(), normalized to 1.0).
        Factor 5 (σ=0.30) is noisiest     → lowest weight.

        CLAIM-64 / SVM-004: this ordering drives the +13.2pp accuracy gain.
        """
        sigma = np.array([0.05, 0.25, 0.15, 0.20, 0.10, 0.30])
        W = 1.0 / sigma ** 2                      # [400, 16, 44.4, 25, 100, 11.1]
        kernel = gae.DiagonalKernel(W / W.max())  # normalize: max weight = 1.0

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
        W = 1.0 / sigma ** 2
        kernel = gae.DiagonalKernel(W)                    # raw W; gradient normalises internally

        sigma_enriched = sigma * 0.5                      # σ halved → W quadrupled
        W_enriched = 1.0 / sigma_enriched ** 2            # = 4 × W
        kernel_enriched = gae.DiagonalKernel(W_enriched)

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
        constructs correctly from pre-computed weights, and has the expected
        attributes for CLAIM-64 integration.
        """
        assert hasattr(gae, "DiagonalKernel"), (
            "gae.DiagonalKernel not found — import drift that broke SVM-003/004"
        )

        sigma = np.array([0.1, 0.2, 0.15, 0.25, 0.12, 0.18])
        W = 1.0 / sigma ** 2
        k = gae.DiagonalKernel(W / W.max())

        assert k.weights is not None
        assert len(k.weights) == 6
        assert k._W_baseline_max is not None
        assert k._W_baseline_max == pytest.approx(1.0)   # max of normalized weights
