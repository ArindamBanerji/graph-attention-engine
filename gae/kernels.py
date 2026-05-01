"""
Pluggable scoring kernels for ProfileScorer.

v6.0: L2Kernel (default, proven) + DiagonalKernel (weighted, continuous mask).
v7.0 research (not committed): ShrinkageKernel + MahalanobisKernel (gated by V-MV-KERNEL).

The kernel controls HOW distance between factor vector f and centroid μ is
measured. Different kernels weight dimensions differently.

All kernels implement ScoringKernel protocol:
  compute_distance(f, mu_matrix) -> distances array, shape (A,)
  compute_gradient(f, mu_single) -> gradient vector, shape (d,)

Reference: docs/gae_design_v10_6.md §9; v6.0 kernel roadmap.
"""

from __future__ import annotations

import numpy as np
from typing import Protocol, runtime_checkable


@runtime_checkable
class ScoringKernel(Protocol):
    """Protocol for all scoring kernels."""

    def compute_distance(self, f: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """
        Compute distance from factor vector to each action's centroid.

        Parameters
        ----------
        f : np.ndarray, shape (d,)
            Factor vector.
        mu : np.ndarray, shape (A, d)
            Centroid matrix for one category — one row per action.

        Returns
        -------
        np.ndarray, shape (A,)
            One non-negative distance per action.
        """
        ...

    def compute_gradient(self, f: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """
        Compute gradient direction for centroid update.

        Parameters
        ----------
        f : np.ndarray, shape (d,)
            Observed factor vector.
        mu : np.ndarray, shape (d,)
            Single centroid vector.

        Returns
        -------
        np.ndarray, shape (d,)
            Direction to move centroid toward f.
        """
        ...


class L2Kernel:
    """
    Standard L2 (Euclidean) distance kernel. Exact v5.5 behavior.

    distance(f, μ_a) = Σ_j (f_j − μ_{a,j})²
    gradient(f, μ)   = f − μ

    All dimensions weighted equally. Default kernel for ProfileScorer.

    Reference: docs/gae_design_v10_6.md §9.1; EXP-C1 (97.89% accuracy).
    """

    def compute_distance(self, f: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """
        Squared L2 distance from f to each row of mu.

        Parameters
        ----------
        f : shape (d,)
        mu : shape (A, d)

        Returns
        -------
        shape (A,) — squared L2 per action.
        """
        diff = f - mu   # (A, d) via broadcasting
        return np.sum(diff ** 2, axis=-1)

    def compute_gradient(self, f: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """
        Gradient direction: f − μ.

        Parameters
        ----------
        f : shape (d,)
        mu : shape (d,)

        Returns
        -------
        shape (d,)
        """
        return f - mu


class DiagonalKernel:
    """
    Diagonal kernel: K(f, μ) = (f−μ)ᵀ W (f−μ), W = diag(1/σ²).

    v6.0 default for deployments with noise_ratio > 1.5.

    GRADIENT: max-normalized kernel-aware gradient.
        G(f, μ) = (W / W.max()) · (f − μ)

    The max-normalization bounds gradient magnitude across
    heterogeneous factor weights (GAE-GRADIENT-001 fix, v0.7.7).
    Preserves gradient DIRECTION while preventing the highest-
    weighted factor from dominating learning by magnitude.

    Per-factor effective learning rate: η · w_j / w_max.
    Per-factor convergence half-life: N_half_j = (w_max/w_j) · ln(2)/η.

    The maximum-weighted factor retains the canonical L2 half-life
    of ln(2)/η ≈ 14 verified decisions at η=0.05.

    All 390 factorial validation cells (V-MV-KERNEL) used this
    gradient form. When W = I (uniform σ): reduces to L2.
    """

    def __init__(
        self,
        sigma: np.ndarray | None = None,
        *,
        weights: np.ndarray | None = None,
    ) -> None:
        """
        Parameters
        ----------
        sigma : np.ndarray, shape (d,)
            Per-factor noise standard deviation. All values must be > 0.
            Weights are computed as W = 1/σ² then normalised by W.max():
            self.weights = W / W.max() ∈ [0, 1].
        weights : np.ndarray, shape (d,)
            Optional direct weight vector. All values must be finite and > 0.
            Used when the caller already has effective diagonal weights.

        Attributes set at construction
        --------------------------------
        _W_baseline_max : float
            max(1/σ²) frozen at construction. Captures absolute signal
            scale — higher value = richer signal. Use to compare kernels
            built from different σ measurements (enrichment validation).
        """
        if sigma is not None and weights is not None:
            raise ValueError("Provide either sigma or weights, not both")
        if sigma is None and weights is None:
            raise ValueError("Either sigma or weights must be provided")

        if weights is not None:
            weight_array = np.asarray(weights, dtype=np.float64)
            assert weight_array.ndim == 1, (
                f"weights must be 1-D, got shape {weight_array.shape}"
            )
            if not np.all(np.isfinite(weight_array)) or np.any(weight_array <= 0):
                raise ValueError(
                    f"DiagonalKernel: all weights must be finite and > 0. Got {weight_array}"
                )
            self._W_baseline_max = float(weight_array.max())
            self.weights = weight_array.copy()
            self.sigma = np.sqrt(1.0 / weight_array)
            return

        sigma = np.asarray(sigma, dtype=np.float64)
        assert sigma.ndim == 1, (
            f"sigma must be 1-D, got shape {sigma.shape}"
        )
        if np.any(sigma <= 0):
            raise ValueError(
                f"DiagonalKernel: all sigma values must be > 0. Got {sigma}"
            )
        W = 1.0 / sigma ** 2
        self._W_baseline_max = float(W.max())
        self.weights = W / self._W_baseline_max
        self.sigma = sigma.copy()

    def compute_distance(self, f: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """
        Weighted squared L2 distance from f to each row of mu.

        Parameters
        ----------
        f : shape (d,)
        mu : shape (A, d)

        Returns
        -------
        shape (A,) — weighted squared L2 per action.
        """
        diff = f - mu   # (A, d)
        return np.sum(self.weights * diff ** 2, axis=-1)

    def compute_gradient(self, f: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """
        Normalised weighted gradient: (W / w_max) ⊙ (f − μ).

        Weights are normalised by their maximum before multiplying the residual.
        This preserves directional preference (reliable factors learn proportionally
        faster) without gradient amplification.

        Max step = η × 1.0 × max|f−μ| = η — identical bound to L2Kernel.
        Noisier factors: step = η × (w_j / w_max) × |f−μ| < η — correctly slower.

        compute_distance() is NOT affected — scoring uses full self.weights.

        Parameters
        ----------
        f : shape (d,)
        mu : shape (d,)

        Returns
        -------
        shape (d,)
        """
        w_max = max(self.weights.max(), 1e-9)
        return (self.weights / w_max) * (f - mu)

    @property
    def noise_ratio(self) -> float:
        """
        max(σ)/min(σ) derived from weights = 1/σ².

        noise_ratio = σ_max/σ_min = sqrt(W_max/W_min) = sqrt(1/weights_min)
        (since weights are proportional to W = 1/σ², W_max = _W_baseline_max × 1).

        KernelSelector trigger criterion: recommend DiagonalKernel when > 1.5.

        Reference: V-MV-KERNEL; docs/gae_design_v10_6.md §9.
        """
        w_min = max(float(self.weights.min()), 1e-12)
        return float(np.sqrt(self._W_baseline_max / w_min))

    @property
    def raw_weights(self) -> np.ndarray:
        """
        Raw inverse-variance weights W_i = 1/σ_i² before normalization.
        Use for cross-instance comparisons (e.g. enrichment-level
        Fisher path: W/W_baseline.max()).

        For scoring and gradient computation, use .weights (normalized).
        """
        return 1.0 / (self.sigma ** 2)

    def refresh_weights(self, sigma_per_factor: np.ndarray) -> "DiagonalKernel":
        """
        Return a NEW DiagonalKernel with weights updated from fresh sigma estimates.
        Does NOT mutate self — returns a new instance (immutable kernel design).

        Weights are computed as 1 / σ², where σ is clipped to ≥ 1e-6 to prevent
        division by zero. Higher σ (noisier factor) → lower weight.

        Args:
            sigma_per_factor: array of shape (d,) — per-factor noise estimates
                              from CovarianceEstimator.get_per_factor_sigma().

        Returns:
            New DiagonalKernel with weights = 1 / sigma_per_factor**2

        Reference: V-CGA-FROZEN gap closure; docs/gae_design_v10_6.md §9.
        """
        sigma_per_factor = np.asarray(sigma_per_factor, dtype=np.float64)
        assert sigma_per_factor.shape == self.weights.shape, (
            f"sigma_per_factor.shape={sigma_per_factor.shape} must match "
            f"weights.shape={self.weights.shape}"
        )
        clipped = np.maximum(sigma_per_factor, 1e-6)
        return DiagonalKernel(clipped)
