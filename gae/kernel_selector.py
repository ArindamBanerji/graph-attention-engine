"""
KernelSelector — empirical kernel selection during shadow mode.

During shadow mode (P28 Phase 3), scores every alert with all available
kernels simultaneously. Tracks per-kernel analyst agreement rate.
At Phase 4 (QUALIFY), recommends the kernel with highest agreement.

During live operation, monitors for conditions that warrant kernel switch:
  - σ re-measurement (new source connected)
  - CovarianceEstimator λ dropping (correlations stabilizing)
  - Σ̂ change-rate spike (regime shift)

The mechanism:
  Phase 2 (COMPUTE): Preliminary kernel from rule (σ ratio + ρ_max)
  Phase 3 (SHADOW):  Empirical comparison on THIS customer's data
  Phase 4 (QUALIFY): Lock the winner. Ongoing monitoring.

Source: Three-judge consensus + roadmap session design.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from gae.kernels import L2Kernel, DiagonalKernel


@dataclass
class KernelScore:
    """Per-kernel tracking during shadow comparison."""

    kernel_name: str
    total_decisions: int = 0
    agreements: int = 0
    disagreements: int = 0
    cumulative_confidence: float = 0.0

    @property
    def agreement_rate(self) -> float:
        """Fraction of decisions where kernel matched analyst. 0.0 if no data."""
        if self.total_decisions == 0:
            return 0.0
        return self.agreements / self.total_decisions

    @property
    def mean_confidence(self) -> float:
        """Mean softmax confidence across all decisions. 0.0 if no data."""
        if self.total_decisions == 0:
            return 0.0
        return self.cumulative_confidence / self.total_decisions


@dataclass
class KernelRecommendation:
    """Output of the kernel selection process."""

    recommended_kernel: str   # 'l2', 'diagonal', or 'shrinkage'
    confidence: float         # margin over runner-up (0.0 for rule-based)
    scores: Dict[str, Dict]   # per-kernel snapshot: agreement_rate, total
    method: str               # 'empirical' (Phase 4) or 'rule' (Phase 2)
    reason: str               # human-readable explanation
    sufficient_data: bool     # True when shadow data meets MIN_DECISIONS


class KernelSelector:
    """
    Selects the optimal scoring kernel for a deployment via empirical comparison.

    Phase 2 (COMPUTE):
        preliminary_recommendation() uses σ noise ratio and ρ_max rules to
        pick a starting kernel before any labelled decisions are available.
        Blog Eq. 9.1 kernel selection heuristic.

    Phase 3 (SHADOW):
        record_comparison() scores every alert with ALL kernels simultaneously
        and accumulates analyst-agreement statistics without affecting live scoring.

    Phase 4 (QUALIFY):
        recommend() returns the kernel with highest cumulative agreement rate
        once MIN_DECISIONS labelled decisions have been collected.

    Monitoring:
        should_reconsider() fires if σ, ρ_max, or covariance λ change enough
        to invalidate the earlier selection. Returns a reason string or None.

    Reference: docs/gae_design_v5.md §9; v6.0 kernel roadmap.
    """

    MIN_DECISIONS_FOR_RECOMMENDATION: int = 100

    def __init__(
        self,
        d: int,
        sigma_per_factor: np.ndarray,
        correlation_max: float = 0.0,
        window_size: int = 100,
    ) -> None:
        """
        Parameters
        ----------
        d : int
            Number of factors.
        sigma_per_factor : np.ndarray, shape (d,)
            Per-factor noise standard deviations from P28 Phase 2.
        correlation_max : float
            Maximum absolute off-diagonal correlation |ρ| from Phase 2.
            0.0 when no covariance estimate is available yet.
        window_size : int
            Rolling window length for agreement tracking (default 100).
            recommend() uses the last window_size decisions to pick the
            winner, avoiding cold-start bias from early cumulative counts.
        """
        assert d > 0, f"d must be positive, got {d}"
        sigma_per_factor = np.asarray(sigma_per_factor, dtype=np.float64)
        assert sigma_per_factor.shape == (d,), (
            f"sigma_per_factor.shape={sigma_per_factor.shape} must be ({d},)"
        )
        assert window_size > 0, f"window_size must be positive, got {window_size}"

        self.d = d
        self.sigma = sigma_per_factor
        self.correlation_max = float(correlation_max)
        self.window_size: int = window_size

        self.kernels: Dict = self._build_kernels()
        self.scores: Dict[str, KernelScore] = {
            name: KernelScore(kernel_name=name) for name in self.kernels
        }
        # Rolling agreement buffers — list of bool, capped at window_size
        self._buffers: Dict[str, List[bool]] = {
            name: [] for name in self.kernels
        }

    # ------------------------------------------------------------------ #
    # Kernel construction                                                  #
    # ------------------------------------------------------------------ #

    def _build_kernels(self) -> Dict:
        """
        Instantiate all candidate kernels from measured σ.

        L2:        plain squared Euclidean — all dimensions equal weight.
        diagonal:  weights = 1/σ², normalised to max=1 — high-noise dims attenuated.
        shrinkage: same as diagonal for v6.0; v7.0 research (not committed) will use full Σ̂ off-diagonal.

        Shape: sigma (d,) → weights (d,).
        """
        kernels: Dict = {"l2": L2Kernel()}

        # 1/σ² inverse-variance weights, normalised so max weight = 1
        inv_var = 1.0 / np.maximum(self.sigma ** 2, 0.001)
        assert inv_var.shape == (self.d,), (
            f"inv_var.shape={inv_var.shape} must be ({self.d},)"
        )
        weights = inv_var / inv_var.max()
        assert weights.shape == (self.d,), (
            f"weights.shape={weights.shape} must be ({self.d},)"
        )

        kernels["diagonal"] = DiagonalKernel(weights)
        # v6.0 shrinkage proxy — identical to diagonal until Σ̂ is available
        # proxy: DiagonalKernel until true ShrinkageKernel ships (v7.0 research)
        kernels["shrinkage"] = DiagonalKernel(weights.copy())

        return kernels

    # ------------------------------------------------------------------ #
    # Phase 2: rule-based preliminary                                     #
    # ------------------------------------------------------------------ #

    def preliminary_recommendation(self) -> KernelRecommendation:
        """
        Phase 2: rule-based kernel selection from measured σ and ρ_max.

        Rules (blog Eq. 9.1 heuristic):
          noise_ratio < 1.5 AND ρ_max < 0.20  → l2
          ρ_max ≥ 0.30                         → shrinkage
          otherwise                            → diagonal

        Returns a KernelRecommendation with method='rule'.
        Overridden by Phase 4 empirical recommendation once data accumulates.
        """
        if self.sigma.max() < 0.001:
            return self._make_rec("l2", "rule", "No noise data — default to L2")

        noise_ratio = float(self.sigma.max() / max(float(self.sigma.min()), 0.001))
        rho = self.correlation_max

        if noise_ratio < 1.5 and rho < 0.2:
            kernel = "l2"
            reason = (
                f"Near-uniform noise (ratio={noise_ratio:.1f}×) and "
                f"low correlation (ρ_max={rho:.2f}). L2 sufficient."
            )
        elif rho >= 0.3:
            kernel = "shrinkage"
            reason = (
                f"Significant factor correlation (ρ_max={rho:.2f}). "
                f"Shrinkage kernel recommended to handle off-diagonal structure."
            )
        else:
            kernel = "diagonal"
            reason = (
                f"Heterogeneous noise (ratio={noise_ratio:.1f}×) with "
                f"low correlation (ρ_max={rho:.2f}). Diagonal weighting recommended."
            )

        return self._make_rec(kernel, "rule", reason)

    # ------------------------------------------------------------------ #
    # Phase 3: shadow comparison                                          #
    # ------------------------------------------------------------------ #

    def record_comparison(
        self,
        factors: np.ndarray,
        category_index: int,
        mu: np.ndarray,
        analyst_action_index: int,
        actions: list,
    ) -> Dict[str, int]:
        """
        Score one alert with ALL kernels and record analyst agreement.

        Called during shadow mode (Phase 3) for every verified decision.
        Does not affect live scoring — tracking only.

        Parameters
        ----------
        factors : np.ndarray, shape (d,)
            Factor vector for this alert.
        category_index : int
            Category index c ∈ [0, n_categories).
        mu : np.ndarray, shape (n_categories, n_actions, d)
            Full centroid tensor.
        analyst_action_index : int
            Action chosen by the analyst (ground truth).
        actions : list of str
            Ordered action names.

        Returns
        -------
        Dict[str, int]
            Mapping kernel_name → predicted action index for this alert.
        """
        factors = np.asarray(factors, dtype=np.float64)
        assert factors.shape == (self.d,), (
            f"factors.shape={factors.shape} must be ({self.d},)"
        )
        mu_c = mu[category_index]  # shape (A, d)
        n_actions = len(actions)
        assert mu_c.shape == (n_actions, self.d), (
            f"mu_c.shape={mu_c.shape} must be ({n_actions}, {self.d})"
        )

        predictions: Dict[str, int] = {}

        for name, kernel in self.kernels.items():
            distances = kernel.compute_distance(factors, mu_c)
            assert distances.shape == (n_actions,), (
                f"distances.shape={distances.shape} must be ({n_actions},)"
            )

            # Softmax with τ=0.1 (same for all kernels — fair comparison)
            logits = -distances / 0.1
            logits = logits - logits.max()   # numerical stability
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum()

            action_idx = int(np.argmax(probs))
            confidence = float(probs[action_idx])
            predictions[name] = action_idx

            agreed = action_idx == analyst_action_index

            score = self.scores[name]
            score.total_decisions += 1
            score.cumulative_confidence += confidence
            if agreed:
                score.agreements += 1
            else:
                score.disagreements += 1

            # Rolling window: keep only the last window_size decisions
            buf = self._buffers[name]
            buf.append(agreed)
            if len(buf) > self.window_size:
                buf.pop(0)

        return predictions

    # ------------------------------------------------------------------ #
    # Phase 4: empirical recommendation                                   #
    # ------------------------------------------------------------------ #

    def recommend(self) -> KernelRecommendation:
        """
        Phase 4: empirical kernel recommendation from accumulated shadow data.

        Returns the kernel with the highest analyst agreement rate once at
        least MIN_DECISIONS labelled decisions have been collected.
        Falls back to rule-based preliminary recommendation if data is scarce.

        Returns
        -------
        KernelRecommendation
            method='empirical' when sufficient data exists; 'rule' otherwise.
        """
        has_enough = any(
            s.total_decisions >= self.MIN_DECISIONS_FOR_RECOMMENDATION
            for s in self.scores.values()
        )
        if not has_enough:
            prelim = self.preliminary_recommendation()
            prelim.sufficient_data = False
            prelim.reason = (
                f"Insufficient shadow data "
                f"(need {self.MIN_DECISIONS_FOR_RECOMMENDATION} decisions). "
                f"Using rule-based recommendation: {prelim.reason}"
            )
            return prelim

        def _rolling_rate(name: str) -> float:
            buf = self._buffers[name]
            return sum(buf) / max(len(buf), 1)

        best_name = max(self.scores, key=_rolling_rate)
        best_score = self.scores[best_name]

        others = {k: v for k, v in self.scores.items() if k != best_name}
        runner_up = max(others, key=_rolling_rate)
        margin = _rolling_rate(best_name) - _rolling_rate(runner_up)

        reason = (
            f"{best_name} had highest rolling agreement: "
            f"{_rolling_rate(best_name):.1%} "
            f"(last {len(self._buffers[best_name])} decisions). "
            f"Runner-up {runner_up}: "
            f"{_rolling_rate(runner_up):.1%}. "
            f"Margin: {margin:+.1%}."
        )

        return self._make_rec(best_name, "empirical", reason,
                              margin=margin, sufficient=True)

    # ------------------------------------------------------------------ #
    # Monitoring                                                          #
    # ------------------------------------------------------------------ #

    def get_comparison_summary(self) -> Dict:
        """
        Current state of the shadow comparison — per-kernel metrics.

        Returns
        -------
        Dict mapping kernel_name → {agreement_rate, mean_confidence,
                                     total_decisions, agreements}.
        """
        return {
            name: {
                "agreement_rate": s.agreement_rate,
                "rolling_agreement_rate": (
                    sum(self._buffers[name]) / max(len(self._buffers[name]), 1)
                ),
                "mean_confidence": s.mean_confidence,
                "total_decisions": s.total_decisions,
                "agreements": s.agreements,
            }
            for name, s in self.scores.items()
        }

    def should_reconsider(
        self,
        new_sigma: Optional[np.ndarray] = None,
        new_rho_max: Optional[float] = None,
        covariance_lambda: Optional[float] = None,
    ) -> Optional[str]:
        """
        Ongoing monitoring: check if conditions warrant kernel reconsideration.

        Triggers when:
          - σ noise ratio changes by > 0.5×
          - ρ_max changes by > 0.15
          - covariance λ drops below 0.3 with low current ρ_max
            (correlations stabilising → shrinkage worth considering)

        Parameters
        ----------
        new_sigma : np.ndarray, shape (d,), optional
            Updated per-factor noise from a new σ measurement pass.
        new_rho_max : float, optional
            Updated maximum absolute correlation from new covariance estimate.
        covariance_lambda : float, optional
            Current Ledoit-Wolf shrinkage intensity from CovarianceEstimator.

        Returns
        -------
        str if reconsideration is warranted; None otherwise.
        """
        reasons = []

        if new_sigma is not None:
            new_sigma = np.asarray(new_sigma, dtype=np.float64)
            old_ratio = float(self.sigma.max() / max(float(self.sigma.min()), 0.001))
            new_ratio = float(new_sigma.max() / max(float(new_sigma.min()), 0.001))
            if abs(new_ratio - old_ratio) > 0.5:
                reasons.append(
                    f"Noise ratio changed: {old_ratio:.1f}× → {new_ratio:.1f}×"
                )

        if new_rho_max is not None:
            if abs(float(new_rho_max) - self.correlation_max) > 0.15:
                reasons.append(
                    f"Max correlation changed: "
                    f"{self.correlation_max:.2f} → {float(new_rho_max):.2f}"
                )

        if covariance_lambda is not None:
            if float(covariance_lambda) < 0.3 and self.correlation_max < 0.3:
                reasons.append(
                    f"Covariance stabilizing (λ={float(covariance_lambda):.2f}). "
                    f"Consider upgrade to shrinkage."
                )

        return "; ".join(reasons) if reasons else None

    def reset_comparison(self) -> None:
        """Reset all tracking for a new shadow comparison period."""
        self.scores = {
            name: KernelScore(kernel_name=name) for name in self.kernels
        }
        self._buffers = {name: [] for name in self.kernels}

    # ------------------------------------------------------------------ #
    # Internal                                                            #
    # ------------------------------------------------------------------ #

    def _make_rec(
        self,
        kernel: str,
        method: str,
        reason: str,
        margin: float = 0.0,
        sufficient: bool = True,
    ) -> KernelRecommendation:
        return KernelRecommendation(
            recommended_kernel=kernel,
            confidence=margin,
            scores={
                name: {
                    "agreement_rate": s.agreement_rate,
                    "total": s.total_decisions,
                }
                for name, s in self.scores.items()
            },
            method=method,
            reason=reason,
            sufficient_data=sufficient,
        )
