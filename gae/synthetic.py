"""
gae/synthetic.py — Oracle separation framework for γ theorem validation.

Architecture note: This module is for experiment design and testing,
NOT production logging. The three EXP-G1 log fields
(centroid_distance_to_canonical, pattern_history_value,
alert_category_distribution) are logged in the SOC triage path
(ci-platform / reconvergence_logger.py), not here.

Validated: oracle separation experiments April 2026.
  Exp A: factor quality clean, regime differentiation confirmed.
  v8 (ε_sim=0.05 < 0.125): γ=0.714 < 1 ✓ (theorem prediction correct)
  v3 (ε_sim=0.20 > 0.125): γ=1.033 > 1 ✓ (theorem prediction correct)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Literal

from gae.convergence import (
    centroid_distance_to_canonical,
    gamma_threshold,
    ConvergenceTrace,
)
from gae.profile_scorer import ProfileScorer


@dataclass
class FactorVectorSample:
    """A single sampled factor vector with metadata."""
    f: np.ndarray                 # shape (d,) — factor vector in [0,1]^d
    regime: str                   # "cold_start" | "post_disruption" | "enriched"
    sigma_per_factor: np.ndarray  # σ for each factor at sampling time
    generation_seed: int


class FactorVectorSampler:
    """
    Samples realistic factor vectors for a GAE domain using parametric
    (Gaussian mixture) generation.

    Oracle separation principle: generates factor vectors only.
    Oracle labels correctness from centroid distance.
    LLM competence prior is fully removed from γ estimation.

    Validated: Exp A (April 2026) confirmed oracle separation produces
    well-differentiated regime vectors without analyst persona corruption.
    """

    def __init__(
        self,
        d: int,
        sigma_profile: np.ndarray,
        seed: int = 42,
    ) -> None:
        """
        Args:
            d:             Factor dimension
            sigma_profile: Per-factor noise profile, shape (d,)
            seed:          Random seed for reproducibility
        """
        self.d = d
        self.sigma_profile = sigma_profile
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        regime: str,
        n: int,
        mean_offset: Optional[np.ndarray] = None,
    ) -> List[FactorVectorSample]:
        """
        Sample n factor vectors for the given regime.

        Args:
            regime:      "cold_start" | "post_disruption" | "enriched"
            n:           Number of samples
            mean_offset: Optional per-factor mean shift (for regime simulation)
        Returns:
            List of FactorVectorSample
        """
        base_mean = np.full(self.d, 0.5)
        if mean_offset is not None:
            base_mean = np.clip(base_mean + mean_offset, 0.0, 1.0)

        samples = []
        for i in range(n):
            f = self.rng.normal(base_mean, self.sigma_profile)
            f = np.clip(f, 0.0, 1.0)
            samples.append(FactorVectorSample(
                f=f,
                regime=regime,
                sigma_per_factor=self.sigma_profile.copy(),
                generation_seed=i,
            ))
        return samples


@dataclass
class CanonicalCentroid:
    """
    Manages canonical centroids (ground truth) for oracle separation.

    NOT the same as BACKLOG-015 centroid_distance_to_canonical (which logs
    distance from the live deployed centroid to a pre-deployment snapshot).
    This class is for experiment design — the mathematical GT that the oracle
    uses to label correctness.
    """
    gt: np.ndarray  # Ground truth centroid, shape (C, A, d)

    @classmethod
    def from_ground_truth(cls, gt: np.ndarray) -> "CanonicalCentroid":
        return cls(gt=gt.copy())

    def apply_disruption(
        self,
        delta: np.ndarray,
        categories: List[int],
    ) -> "CanonicalCentroid":
        """
        Returns new CanonicalCentroid representing GT_2 = GT_1 + Δ
        applied only to the specified disrupted categories.

        Args:
            delta:      Disruption vector, shape (A, d)
            categories: Category indices to disrupt
        Returns:
            New CanonicalCentroid with disruption applied
        """
        new_gt = self.gt.copy()
        for c in categories:
            new_gt[c] = np.clip(new_gt[c] + delta, 0.0, 1.0)
        return CanonicalCentroid(gt=new_gt)

    def distance_from(self, mu: np.ndarray) -> float:
        """Frobenius distance from mu to this canonical centroid."""
        return centroid_distance_to_canonical(mu, self.gt)


@dataclass
class Phase1Result:
    """Result of oracle separation Phase 1."""
    n_half: Optional[int]   # None if Phase 1 DNF
    mu_final: np.ndarray    # centroid tensor at end of Phase 1
    trace: ConvergenceTrace
    dnf: bool               # Did Not Finish (N_half not reached)


@dataclass
class Phase2Result:
    """Result of oracle separation Phase 2."""
    n_half: Optional[int]   # None if Phase 2 DNF
    trace: ConvergenceTrace
    dnf: bool


@dataclass
class GammaResult:
    """
    γ measurement from oracle separation experiment.
    γ = N_half,1 / N_half,2 — re-convergence speed ratio.
    """
    n_half_1: Optional[int]           # None if Phase 1 DNF
    n_half_2: Optional[int]           # None if Phase 2 DNF
    gamma: Optional[float]            # None if either phase DNF
    centroid_dist_phase1: List[float] # dist(t) per decision
    centroid_dist_phase2: List[float]
    n_half_gap_detected: bool         # N_half fired before centroid converged
    epsilon_firm: float
    threshold: float                  # gamma_threshold() for these params
    theorem_prediction: str           # "gamma_gt_1" | "gamma_lt_1"
    simulation_confirms: Optional[bool]  # None if DNF
    note: str

    @property
    def is_above_threshold(self) -> bool:
        return self.epsilon_firm > self.threshold


class OracleSeparationExperiment:
    """
    Oracle separation protocol for γ theorem validation.

    Phase 1: cold-start calibration from mu_0 toward GT_1.
    Phase 2: post-disruption re-convergence from mu_T1 toward GT_2.
    Correctness labeled by oracle distance, not LLM judgment.

    Binary validation (April 2026):
      v8 (ε_sim=0.05 < 0.125): γ=0.714 < 1 ✓
      v3 (ε_sim=0.20 > 0.125): γ=1.033 > 1 ✓
    """

    def __init__(
        self,
        scorer: ProfileScorer,
        canonical_gt1: CanonicalCentroid,
        epsilon_firm: float,
        disruption_magnitude: float,
        disrupted_categories: List[int],
        alpha_cat: Optional[float] = None,
        window: int = 10,
        theta: float = 0.85,
        max_decisions: int = 600,
    ) -> None:
        self.scorer = scorer
        self.canonical_gt1 = canonical_gt1
        self.epsilon_firm = epsilon_firm
        self.disruption_magnitude = disruption_magnitude
        self.disrupted_categories = disrupted_categories
        self.window = window
        self.theta = theta
        self.max_decisions = max_decisions

        C, A, d = scorer.mu.shape
        self.alpha_cat = alpha_cat or len(disrupted_categories) / C
        self.threshold = gamma_threshold(
            self.alpha_cat, disruption_magnitude, theta
        )

    def _oracle_correct(
        self,
        f: np.ndarray,
        category_index: int,
        action_index: int,
        canonical: CanonicalCentroid,
    ) -> bool:
        """
        Oracle labels correctness by centroid distance.
        Correct = action_index matches argmin distance to canonical GT.
        """
        mu_c = canonical.gt[category_index]  # shape (A, d)
        distances = np.sum((f - mu_c) ** 2, axis=-1)
        gt_action = int(np.argmin(distances))
        return action_index == gt_action

    def _run_phase(
        self,
        samples: List[FactorVectorSample],
        canonical: CanonicalCentroid,
        phase: str,
    ) -> tuple:
        """Run one phase, return (n_half, distances, trace)."""
        distances = []
        rolling_acc = []
        correct_window: List[int] = []
        n_half = None

        for i, sample in enumerate(samples[:self.max_decisions]):
            f = sample.f
            result = self.scorer.score(f, category_index=0)
            correct = self._oracle_correct(
                f, 0, result.action_index, canonical
            )
            self.scorer.update(f, 0, result.action_index, correct)

            correct_window.append(1 if correct else 0)
            if len(correct_window) > self.window:
                correct_window.pop(0)

            dist = canonical.distance_from(self.scorer.mu)
            distances.append(dist)

            acc = float(np.mean(correct_window))
            rolling_acc.append(acc)

            if n_half is None and len(correct_window) == self.window \
                    and acc >= self.theta:
                n_half = i + 1

        # Detect N_half gap: did N_half fire before centroid converged?
        n_half_gap = False
        if n_half is not None and len(distances) > n_half:
            initial_dist = distances[0]
            dist_at_n_half = distances[n_half - 1]
            # Gap if centroid has not dropped 20% by N_half
            n_half_gap = dist_at_n_half > 0.80 * initial_dist

        trace = ConvergenceTrace(
            centroid_distances=distances,
            rolling_accuracy=rolling_acc,
            n_half=n_half,
            centroid_converged_at=None,  # simplification
            n_half_gap=n_half_gap,
            phase=phase,
            epsilon_firm=self.epsilon_firm,
        )
        return n_half, distances, trace

    def run_phase1(
        self,
        factor_samples: List[FactorVectorSample],
    ) -> Phase1Result:
        n_half, distances, trace = self._run_phase(
            factor_samples, self.canonical_gt1, "phase1"
        )
        return Phase1Result(
            n_half=n_half,
            mu_final=self.scorer.mu.copy(),
            trace=trace,
            dnf=n_half is None,
        )

    def run_phase2(
        self,
        factor_samples: List[FactorVectorSample],
        phase1_result: Phase1Result,
    ) -> Phase2Result:
        # Apply disruption to GT
        delta = np.full(
            self.scorer.mu.shape[1:],
            self.disruption_magnitude / np.sqrt(
                self.scorer.mu.shape[1] * self.scorer.mu.shape[2]
            )
        )
        canonical_gt2 = self.canonical_gt1.apply_disruption(
            delta, self.disrupted_categories
        )

        # Reset scorer to phase1 final state
        self.scorer.mu = phase1_result.mu_final.copy()

        n_half, distances, trace = self._run_phase(
            factor_samples, canonical_gt2, "phase2"
        )
        return Phase2Result(
            n_half=n_half,
            trace=trace,
            dnf=n_half is None,
        )

    def compute_gamma(
        self,
        r1: Phase1Result,
        r2: Phase2Result,
    ) -> GammaResult:
        gamma = None
        confirms = None
        note = ""

        if not r1.dnf and not r2.dnf:
            gamma = r1.n_half / r2.n_half
            confirms = (gamma > 1) == self.is_above_threshold
            note = (
                f"gamma={gamma:.3f}. "
                f"epsilon_firm={self.epsilon_firm:.3f} "
                f"{'>' if self.is_above_threshold else '<'} "
                f"threshold={self.threshold:.3f}. "
                f"Theorem {'confirmed' if confirms else 'NOT confirmed'}."
            )
        elif r1.dnf:
            note = "Phase 1 DNF — eta_neg trap or insufficient decisions."
        else:
            note = "Phase 2 DNF."

        gap = r1.trace.n_half_gap or r2.trace.n_half_gap

        return GammaResult(
            n_half_1=r1.n_half,
            n_half_2=r2.n_half,
            gamma=gamma,
            centroid_dist_phase1=r1.trace.centroid_distances,
            centroid_dist_phase2=r2.trace.centroid_distances,
            n_half_gap_detected=gap,
            epsilon_firm=self.epsilon_firm,
            threshold=self.threshold,
            theorem_prediction=self.theorem_prediction,
            simulation_confirms=confirms,
            note=note,
        )

    @property
    def is_above_threshold(self) -> bool:
        return self.epsilon_firm > self.threshold

    @property
    def theorem_prediction(self) -> str:
        """'gamma_gt_1' if epsilon_firm > threshold, else 'gamma_lt_1'."""
        return "gamma_gt_1" if self.is_above_threshold else "gamma_lt_1"
