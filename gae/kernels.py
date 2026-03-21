"""
Pluggable scoring kernels for ProfileScorer.

v6.0: L2Kernel (default, proven) + DiagonalKernel (weighted, continuous mask).
v6.5: ShrinkageKernel + MahalanobisKernel (gated by V-MV-KERNEL).

The kernel controls HOW distance between factor vector f and centroid μ is
measured. Different kernels weight dimensions differently.

All kernels implement ScoringKernel protocol:
  compute_distance(f, mu_matrix) -> distances array, shape (A,)
  compute_gradient(f, mu_single) -> gradient vector, shape (d,)

Reference: docs/gae_design_v5.md §9; v6.0 kernel roadmap.
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

    Reference: docs/gae_design_v5.md §9.1; EXP-C1 (97.89% accuracy).
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
    Weighted L2 distance. Continuous replacement for binary factor mask.

    distance(f, μ_a) = Σ_j w_j (f_j − μ_{a,j})²
    gradient(f, μ)   = W ⊙ (f − μ)   where W = diag(weights)

    High weight = trust this dimension. Low weight = attenuate.
    When all weights = 1.0: identical to L2Kernel.
    When weights are 0/1:   equivalent to factor_mask.

    v6.0: ships alongside L2Kernel. Customer choice via DomainConfig.
    v6.5: replaced by ShrinkageKernel (W derived from Σ̂ automatically).

    Reference: docs/gae_design_v5.md §9; v6.0 kernel roadmap.
    """

    def __init__(self, weights: np.ndarray) -> None:
        """
        Parameters
        ----------
        weights : np.ndarray, shape (d,)
            Per-factor importance weights. Typically 1/σ² (inverse noise
            variance) or binary 0/1. All values should be ≥ 0.
        """
        self.weights = np.asarray(weights, dtype=np.float64)
        assert self.weights.ndim == 1, (
            f"weights must be 1-D, got shape {self.weights.shape}"
        )

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
        Weighted gradient: W ⊙ (f − μ).

        Parameters
        ----------
        f : shape (d,)
        mu : shape (d,)

        Returns
        -------
        shape (d,)
        """
        return self.weights * (f - mu)
