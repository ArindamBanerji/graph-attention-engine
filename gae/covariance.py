"""
Online covariance estimation with Ledoit-Wolf shrinkage.

v6.0: COLLECTS data, computes Σ̂, but does NOT affect scoring.
v7.0 research (not committed): Feeds ShrinkageKernel and MahalanobisKernel.

One estimator per (category, action) pair. SOC: 6×4 = 24 estimators.
Uses exponentially-weighted Welford-style accumulation so recent decisions
weight more — tracks regime shifts without discarding history entirely.

Reference: docs/gae_design_v5.md §9; v6.5 kernel roadmap.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CovarianceSnapshot:
    """
    Frozen view of the covariance estimator state at a point in time.

    Attributes
    ----------
    sigma : np.ndarray, shape (d, d)
        Shrinkage-regularised covariance estimate Σ̂.
    sigma_inv : np.ndarray or None, shape (d, d)
        Inverse of Σ̂. None if Σ̂ is singular.
    correlation : np.ndarray, shape (d, d)
        Normalised correlation matrix (diagonal = 1).
    shrinkage_lambda : float
        Current shrinkage intensity λ ∈ [0, 1].
        1.0 = pure diagonal (no data), 0.0 = full sample covariance.
    condition_number : float
        κ(Σ̂) — high value indicates near-singular (unstable) estimate.
    n_samples : int
        Cumulative decisions seen (not decayed).
    per_factor_sigma : np.ndarray, shape (d,)
        Diagonal of Σ̂ — per-factor variance.
    """

    sigma: np.ndarray
    sigma_inv: Optional[np.ndarray]
    correlation: np.ndarray
    shrinkage_lambda: float
    condition_number: float
    n_samples: int
    per_factor_sigma: np.ndarray


class CovarianceEstimator:
    """
    Online covariance estimator with Ledoit-Wolf optimal shrinkage.

    Class attributes
    ----------------
    MIN_SAMPLES_FOR_SIGMA : int
        Minimum number of observations required before get_per_factor_sigma()
        returns reliable estimates. Below this threshold it returns None.

    Accumulates exponentially-weighted statistics. The decay parameter
    downweights old observations so the estimator tracks regime shifts.

    v6.0: collect only — output is logged but does NOT feed scoring.
    v7.0 research (not committed): snapshot feeds ShrinkageKernel / MahalanobisKernel.

    Usage
    -----
    est = CovarianceEstimator(d=6)
    for each decision:
        est.update(factor_vector)
    snapshot = est.get_snapshot()

    Reference: docs/gae_design_v5.md §9; Ledoit & Wolf (2004).
    """

    MIN_SAMPLES_FOR_SIGMA: int = 50

    def __init__(self, d: int, half_life_decisions: int = 300) -> None:
        """
        Parameters
        ----------
        d : int
            Number of factors.
        half_life_decisions : int
            Exponential decay half-life in decisions.
            300 ≈ 90 days at V=100 with 30 % verification.
            Recent decisions weight more — tracks regime shifts.
        """
        assert d > 0, f"d must be positive, got {d}"
        assert half_life_decisions >= 0, (
            f"half_life_decisions must be ≥ 0, got {half_life_decisions}"
        )
        self.d = d
        self.half_life = half_life_decisions
        self.decay = (
            0.5 ** (1.0 / half_life_decisions)
            if half_life_decisions > 0
            else 1.0
        )
        self.n_samples: int = 0

        # Running statistics (exponentially weighted)
        self.weighted_sum: np.ndarray = np.zeros(d, dtype=np.float64)
        self.weighted_outer: np.ndarray = np.zeros((d, d), dtype=np.float64)
        self.total_weight: float = 0.0

    def update(self, f: np.ndarray) -> None:
        """
        Incorporate a new factor vector into the running statistics.

        Uses exponential weighting: existing statistics are decayed by
        `self.decay` before each new observation (weight 1.0) is added.
        This makes recent observations count more.

        Parameters
        ----------
        f : np.ndarray, shape (d,)
            Observed factor vector.
        """
        f = np.asarray(f, dtype=np.float64)
        assert f.shape == (self.d,), (
            f"f.shape={f.shape} must be ({self.d},)"
        )

        # Decay existing statistics toward zero
        self.weighted_sum    *= self.decay
        self.weighted_outer  *= self.decay
        self.total_weight    *= self.decay

        # Accumulate new observation with weight 1.0
        self.weighted_sum   += f
        self.weighted_outer += np.outer(f, f)
        self.total_weight   += 1.0
        self.n_samples      += 1

    def get_snapshot(self) -> CovarianceSnapshot:
        """
        Compute the current covariance estimate with Ledoit-Wolf shrinkage.

        Falls back to identity when fewer than 2 samples have been seen.

        Returns
        -------
        CovarianceSnapshot
        """
        if self.n_samples < 2:
            return CovarianceSnapshot(
                sigma=np.eye(self.d),
                sigma_inv=np.eye(self.d),
                correlation=np.eye(self.d),
                shrinkage_lambda=1.0,
                condition_number=1.0,
                n_samples=self.n_samples,
                per_factor_sigma=np.ones(self.d),
            )

        # Weighted mean and raw covariance
        mean = self.weighted_sum / self.total_weight         # (d,)
        cov  = self.weighted_outer / self.total_weight - np.outer(mean, mean)
        assert cov.shape == (self.d, self.d), (
            f"cov.shape={cov.shape} != ({self.d}, {self.d})"
        )

        # Ledoit-Wolf shrinkage toward diagonal target
        target = np.diag(np.diag(cov))
        lam = self._ledoit_wolf_lambda(cov, target)
        sigma = (1.0 - lam) * cov + lam * target

        # Symmetrize and ensure positive semi-definite
        sigma = (sigma + sigma.T) / 2.0
        eigvals = np.linalg.eigvalsh(sigma)
        min_eig = eigvals.min()
        if min_eig < 1e-10:
            sigma += np.eye(self.d) * (1e-10 - min_eig)

        # Inverse (None if singular)
        try:
            sigma_inv: Optional[np.ndarray] = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            sigma_inv = None

        # Correlation matrix
        diag_sqrt = np.sqrt(np.diag(sigma))
        diag_sqrt = np.where(diag_sqrt < 1e-10, 1e-10, diag_sqrt)
        correlation = sigma / np.outer(diag_sqrt, diag_sqrt)

        cond = float(np.linalg.cond(sigma))

        return CovarianceSnapshot(
            sigma=sigma,
            sigma_inv=sigma_inv,
            correlation=correlation,
            shrinkage_lambda=lam,
            condition_number=cond,
            n_samples=self.n_samples,
            per_factor_sigma=np.diag(sigma).copy(),
        )

    def _ledoit_wolf_lambda(self, S: np.ndarray, T: np.ndarray) -> float:
        """
        Ledoit-Wolf optimal shrinkage intensity λ ∈ [0, 1].

        Analytic approximation — no iterative optimisation required.
        Biased toward full shrinkage (diagonal) when n is small relative to d.

        Parameters
        ----------
        S : np.ndarray, shape (d, d) — sample covariance.
        T : np.ndarray, shape (d, d) — shrinkage target (diagonal of S).

        Returns
        -------
        float — λ (1 = pure target, 0 = pure sample).
        """
        n = max(self.n_samples, 2)
        d = self.d

        diff = S - T
        num = float(np.sum(diff ** 2))   # ‖S − T‖²_F

        if num < 1e-10:
            return 0.0   # S is already diagonal — no shrinkage needed

        # Oracle Approximating Shrinkage (simplified)
        lam = min(1.0, max(0.0, (1.0 / n) * num / max(float(np.sum(S ** 2)), 1e-10)))

        # Bias toward diagonal for small samples (d/n correction)
        sample_correction = min(1.0, float(d) / max(n, 1))
        lam = max(lam, sample_correction)

        return float(np.clip(lam, 0.0, 1.0))

    def get_per_factor_sigma(self) -> Optional[np.ndarray]:
        """
        Return current per-factor sigma (std dev) estimates, shape (d,).

        Computes sqrt of per-factor variance from the accumulated covariance
        snapshot diagonal. Returns None when fewer than MIN_SAMPLES_FOR_SIGMA
        observations have been collected — estimates are unreliable below this
        threshold and callers must not use them for weight updates.

        Returns
        -------
        np.ndarray of shape (d,), or None if insufficient data.

        Reference: V-CGA-FROZEN gap closure; DiagonalKernel.refresh_weights().
        """
        if self.n_samples < self.MIN_SAMPLES_FOR_SIGMA:
            return None
        snapshot = self.get_snapshot()
        variance = snapshot.per_factor_sigma   # shape (d,) — diagonal of Σ̂
        assert variance.shape == (self.d,), (
            f"variance.shape={variance.shape} must be ({self.d},)"
        )
        return np.sqrt(variance)

    def get_change_rate(self, previous_snapshot: CovarianceSnapshot) -> float:
        """
        Frobenius norm of the change in Σ̂ since a previous snapshot.

        A large value signals a regime shift in the factor correlation structure.

        Parameters
        ----------
        previous_snapshot : CovarianceSnapshot
            Snapshot captured at an earlier point in time.

        Returns
        -------
        float — ‖Σ̂_now − Σ̂_prev‖_F ≥ 0.
        """
        current = self.get_snapshot()
        return float(np.linalg.norm(current.sigma - previous_snapshot.sigma, 'fro'))
