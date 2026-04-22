"""
GAE ProfileScorer — centroid-proximity scoring for v5.0.

Replaces ScoringMatrix (deprecated, TD-029). Architecture settled by 14
experiments. Key validated numbers:
  - L2 kernel: 97.89% oracle accuracy (EXP-C1), 98.2% with learning (EXP-B1)
  - τ=0.1: ECE=0.036 (V3B). τ=0.25 was wrong (ECE=0.19).
  - Centroid clipping [0,1]: required (V2: escape at dec 6-12 without it)
  - Global Mahalanobis: harmful on multi-category data (FX-1-CORRECTED)
  - DOT product: 61% on [0,1] factors (EXP-C1)

Reference: docs/gae_design_v10_7.md §9; blog Eq. 4-final, 4b-final.
"""

from __future__ import annotations

import warnings

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from gae.calibration import CalibrationProfile

from gae.primitives import compute_entropy

# V-STABILITY F=8.14 — η change-rate cap (UNCONDITIONAL).
# No single η × gradient step may move any centroid coordinate by more than
# ±MAX_ETA_DELTA, regardless of η value or gradient magnitude.
MAX_ETA_DELTA: float = 0.005


@dataclass
class CentroidUpdate:
    """
    Return value from ProfileScorer.update().

    Captures the magnitude of centroid movement for a single learning step.
    Used for visualization, Neo4j persistence, and drift monitoring.

    Reference: docs/gae_design_v10_6.md §9.5; IKS pipeline.
    """

    centroid_delta_norm: float   # ‖η*(f - μ[c,a,:])‖₂ before in-place update (predicted push)
    category_index: int          # c — the alert category updated
    action_index: int            # a — the verified action updated
    category_name: str           # human-readable, for logging
    action_name: str             # human-readable, for logging
    decision_count: int          # total decisions in this ProfileScorer instance
    gt_delta_norm: float = 0.0   # ‖η*(f - μ[c,gt,:])‖₂ for gt pull (0.0 if correct=True or no gt)
    outcome: str = 'applied'     # 'applied', 'gated_low_confidence', or 'frozen'


class KernelType(Enum):
    """
    Distance kernel for profile scoring.

    Validated by EXP-E1 (kernel generalization experiment):
      L2:          97.9% on [0,1] factors — DEFAULT
      MAHALANOBIS: 97.7% on [0,1], 92.9% on mixed-scale
                   WARNING: per-category covariance REQUIRED
                   Global covariance: ECE degrades 10.8x (FX-1-CORRECTED)
      COSINE:      96.4% on [0,1], 61.2% on mixed-scale
      DOT:         61.0% on [0,1] — WARNING emitted at init

    Reference: docs/gae_design_v10_6.md §9.1; EXP-E1.
    """

    L2 = "l2"
    DIAGONAL = "diagonal"
    MAHALANOBIS = "mahalanobis"
    COSINE = "cosine"
    DOT = "dot"


@dataclass
class ScoringResult:
    """
    Result of ProfileScorer.score().

    Reference: docs/gae_design_v10_6.md §9.2; blog Eq. 4-final.

    Attributes
    ----------
    action_index : int
        Index of recommended action (argmax of probabilities).
    action_name : str
        Name of recommended action.
    probabilities : np.ndarray, shape (n_actions,)
        Softmax distribution over all actions. Sums to 1.0.
    distances : np.ndarray, shape (n_actions,)
        Raw L2/kernel distances. Lower = more similar for L2.
    confidence : float
        Probability of recommended action (max of probabilities).
    entropy : float
        Shannon entropy of probabilities in nats. 0.0 default.
    confidence_gap : float
        top_p - second_p. 0.0 if n_actions < 2. 0.0 default.
    """

    action_index: int
    action_name: str
    probabilities: np.ndarray   # shape (n_actions,)
    distances: np.ndarray       # shape (n_actions,)
    confidence: float
    entropy: float = 0.0
    confidence_gap: float = 0.0


class ProfileScorer:
    """
    Profile-based scoring using centroid proximity.

    Implements Eq. 4-final (validated):
      P(a|f,c) = softmax( -‖f - μ[c,a,:]‖² / τ )

    And Eq. 4b-final (centroid pull/push learning):
      On correct:   μ[c,a,:] += η     * (f - μ[c,a,:])   [pull toward f]
      On incorrect: μ[c,b,:] -= η_neg * (f - μ[c,b,:])   [push away, all b]
      All updates clip to [0.0, 1.0] — V2 requirement.

    Profile centroids μ[c, a, :] represent:
      "What action a looks like for category c"
      Shape: (n_categories, n_actions, n_factors)

    Experimental validation:
      EXP-C1: 97.89% zero-learning accuracy (centroidal synthetic)
      EXP-B1: 98.2% with online learning
      V3B:    ECE=0.036 at τ=0.1
      V2:     Clipping prevents centroid escape under adversarial updates

    Reference: docs/gae_design_v10_6.md §9; blog Eq. 4-final, 4b-final.
    """

    def __init__(
        self,
        mu: np.ndarray,
        actions: List[str],
        kernel: KernelType = KernelType.L2,
        profile: Optional["CalibrationProfile"] = None,
        categories: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        eta_override: Optional[float] = None,
        factor_mask: Optional[np.ndarray] = None,
        scoring_kernel=None,
        auto_pause_on_amber: bool = False,
    ) -> None:
        """
        Args:
          mu:             Initial profile centroids, shape (n_categories, n_actions, n_factors).
                          Values should be in [0.0, 1.0]. Copied — caller's array not mutated.
          actions:        Ordered action names. len(actions) == mu.shape[1].
          kernel:         Distance kernel. L2 is validated default.
          profile:        CalibrationProfile for hyperparameters. If None, uses validated
                          defaults: τ=0.1, η=0.05, η_neg=0.05, decay=0.001.
          min_confidence: Minimum system confidence for centroid update. If the system's
                          confidence in its original decision was below this threshold,
                          the update is skipped (gated). Prevents noisy corrections from
                          low-confidence decisions degrading centroids.

                          Default: 0.0 (all updates fire — backward compatible).
                          Recommended production value: 0.30–0.50.

                          Source: Block 5B Proxy persona testing. 9 LLM-judge personas
                          showed 13–27pp accuracy degradation over 60 days without gate.
          scoring_kernel: ScoringKernel instance (L2Kernel, DiagonalKernel, …).
                          If None, defaults to L2Kernel (identical v5.5 behavior).
                          Only active when the KernelType is L2 (the validated path).
                          COSINE/DOT/MAHALANOBIS still use _compute_distances().

                          v6.0: L2Kernel (default) + DiagonalKernel.
                          v7.0 research (not committed): ShrinkageKernel (auto from CovarianceEstimator).
                          Source: v6.0 kernel roadmap.
          factor_mask:    np.ndarray of shape (n_factors,) with 1.0 (include) and 0.0
                          (exclude). If None, all factors are active (backward compatible).
                          When set, score() uses only unmasked dimensions for distance,
                          and update() only modifies unmasked centroid dimensions.

                          v6.0: binary mask. v6.5: continuous W weighting (Adjustment A).
                          Source: Three-judge consensus.
          eta_override:   Learning rate for analyst overrides (correct=False path).
                          If None, uses eta_neg for push and eta for GT pull (backward
                          compatible — same as before). When set, both the push-away and
                          GT-pull operations use eta_override, applying an attenuated
                          rate that reflects the lower signal quality of overrides.

                          Recommended production value: 0.01 (Q5 validated).
                          Source: Q5 sweep (9 personas × 6 η values). +2–6pp improvement
                          for high-quality teams, zero regression for low-quality teams.
          auto_pause_on_amber: If True, centroid updates are blocked whenever the
                          conservation monitor reports AMBER or RED status. Learning
                          resumes automatically when status returns to GREEN.
                          Default: False (backward compatible).
                          Call set_conservation_status() to update.
                          Source: G6a three-judge consensus (v6.0 blunt freeze).

        Reference: docs/gae_design_v10_6.md §9.3; V3B (τ), V2 (clipping).
        """
        assert mu.ndim == 3, (
            f"mu must be shape (n_cat, n_act, n_fac), got {mu.shape}"
        )
        assert mu.shape[1] == len(actions), (
            f"mu.shape[1]={mu.shape[1]} must equal len(actions)={len(actions)}"
        )

        self.mu = mu.copy().astype(np.float64)
        self.actions = list(actions)
        self.categories: Optional[List[str]] = list(categories) if categories is not None else None
        self.kernel = kernel
        self.n_categories: int = mu.shape[0]
        self.n_actions: int = mu.shape[1]
        self.n_factors: int = mu.shape[2]
        self.decision_count: int = 0

        # Hyperparameters — from CalibrationProfile or validated defaults
        if profile is not None:
            self.tau     = float(profile.temperature)
            self.eta     = float(profile.extensions.get("eta", 0.05))
            self.eta_neg = float(profile.extensions.get("eta_neg", 0.05))
            self.decay   = float(profile.extensions.get("count_decay", 0.001))
        else:
            self.tau     = 0.1    # V3B validated: ECE=0.036
            self.eta     = 0.05
            self.eta_neg = 0.05
            self.decay   = 0.001

        # INV-4: eta_neg >= 1.0 destroys calibration (ECE=0.49). Hard guard.
        if self.eta_neg >= 1.0:
            raise ValueError(
                f"eta_neg={self.eta_neg} is forbidden. "
                f"eta_neg >= 1.0 destroys calibration (ECE=0.49). "
                f"Canonical value is 0.05."
            )

        self.min_confidence: float = min_confidence
        self.eta_override: Optional[float] = eta_override  # None = use eta/eta_neg

        # v6.0 pluggable scoring kernel (L2 path only; COSINE/DOT/MAH use _compute_distances)
        from gae.kernels import L2Kernel as _L2Kernel
        self.scoring_kernel = scoring_kernel if scoring_kernel is not None else _L2Kernel()

        # Factor quarantine mask (v6.0 binary; None = all active)
        if factor_mask is not None:
            factor_mask = np.asarray(factor_mask, dtype=np.float64)
            assert factor_mask.shape == (self.n_factors,), (
                f"factor_mask.shape={factor_mask.shape} must be ({self.n_factors},)"
            )
        self.factor_mask: Optional[np.ndarray] = factor_mask

        # AMBER auto-pause (G6a three-judge consensus — v6.0 blunt freeze)
        self.auto_pause_on_amber: bool = auto_pause_on_amber
        self._conservation_status: str = 'GREEN'
        self._paused_by_conservation: bool = False

        # Gate statistics (Block 5B Proxy — min_confidence gate)
        self._gated_count: int = 0
        self._applied_count: int = 0

        # Per-(category, action) observation counts for learning rate decay
        self.counts = np.zeros((self.n_categories, self.n_actions), dtype=np.int64)
        self._frozen: bool = False
        self._eta_override_warned: bool = False
        assert self.counts.shape == (self.n_categories, self.n_actions), (
            f"counts shape {self.counts.shape} != ({self.n_categories}, {self.n_actions})"
        )

        # Kernel-specific setup and warnings
        if kernel == KernelType.DOT:
            warnings.warn(
                "KernelType.DOT is magnitude-confounded on non-normalized [0,1] data. "
                "EXP-C1: 61.0% (DOT) vs 97.89% (L2) on identical data. "
                "Use KernelType.L2 or normalize factor vectors first.",
                UserWarning,
                stacklevel=2,
            )

        if kernel == KernelType.MAHALANOBIS:
            warnings.warn(
                "KernelType.MAHALANOBIS requires per-category covariance matrices. "
                "Global covariance degrades ECE 10.8x on multi-category data "
                "(FX-1-CORRECTED: ECE 0.0255 L2 vs 0.2750 global Mahalanobis). "
                "Provide per_category_cov via set_covariance() or use KernelType.L2.",
                UserWarning,
                stacklevel=2,
            )
            # Per-category covariance inverses — None until set_covariance() called
            self._cov_inv: Optional[np.ndarray] = None

    @classmethod
    def for_soc(cls, mu, actions=None, **kwargs):
        """
        Factory: ProfileScorer with SOC-validated defaults.

        Sets eta_override=0.01 (P0 fix), auto_pause_on_amber=True,
        and SOC canonical actions if not provided.

        Usage:
            scorer = ProfileScorer.for_soc(mu=centroids)
        """
        if actions is None:
            actions = ["escalate", "investigate", "suppress", "monitor"]
        return cls(
            mu=mu,
            actions=actions,
            eta_override=kwargs.pop('eta_override', 0.01),
            auto_pause_on_amber=kwargs.pop('auto_pause_on_amber', True),
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Scoring                                                             #
    # ------------------------------------------------------------------ #

    def score(self, f: np.ndarray, category_index: int) -> ScoringResult:
        """
        Score factor vector f for category c. Returns action recommendation.

        Implements Eq. 4-final:
          P(a|f,c) = softmax( -dist(f, μ[c,a,:]) / τ )

        Args:
          f:              Factor vector, shape (n_factors,). Values in [0,1].
          category_index: Category index c ∈ [0, n_categories).

        Returns:
          ScoringResult with recommended action and full probability distribution.

        Reference: docs/gae_design_v10_6.md §9.4; blog Eq. 4-final.
        """
        assert 0 <= category_index < self.n_categories, (
            f"category_index {category_index} out of range [0, {self.n_categories})"
        )
        f = np.asarray(f, dtype=np.float64)
        assert f.shape == (self.n_factors,), (
            f"f.shape={f.shape} must be ({self.n_factors},)"
        )
        if not np.all(np.isfinite(f)):
            raise ValueError("Factor vector contains NaN or Inf values")
        if self.tau <= 0:
            raise ValueError(
                f"Temperature tau must be positive, got {self.tau}"
            )

        mu_c = self.mu[category_index]  # shape (n_actions, n_factors)
        assert mu_c.shape == (self.n_actions, self.n_factors), (
            f"mu_c.shape={mu_c.shape} != ({self.n_actions}, {self.n_factors})"
        )

        # Factor quarantine mask: zero out excluded dimensions in both f and mu_c
        # so masked dimensions contribute 0 to distance regardless of their values.
        if self.factor_mask is not None:
            f = f * self.factor_mask                   # (n_factors,)
            mu_c = mu_c * self.factor_mask             # (n_actions, n_factors) broadcast

        # v6.0 kernel dispatch: scoring_kernel active on L2 and DIAGONAL paths.
        # COSINE/DOT/MAHALANOBIS still go through _compute_distances.
        if self.kernel in (KernelType.L2, KernelType.DIAGONAL):
            distances = self.scoring_kernel.compute_distance(f, mu_c)
        else:
            distances = self._compute_distances(f, mu_c, category_index)
        assert distances.shape == (self.n_actions,), (
            f"distances.shape={distances.shape} != ({self.n_actions},)"
        )

        # Eq. 4-final: softmax over negative distances scaled by τ
        logits = -distances / self.tau
        logits -= logits.max()   # numerical stability
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()
        assert probs.shape == (self.n_actions,), (
            f"probs.shape={probs.shape} != ({self.n_actions},)"
        )

        action_idx = int(np.argmax(probs))
        entropy_val = compute_entropy(probs)
        if len(probs) >= 2:
            sorted_p = np.sort(probs)[::-1]
            gap_val = float(sorted_p[0] - sorted_p[1])
        else:
            gap_val = 0.0
        return ScoringResult(
            action_index=action_idx,
            action_name=self.actions[action_idx],
            probabilities=probs,
            distances=distances,
            confidence=float(probs[action_idx]),
            entropy=entropy_val,
            confidence_gap=gap_val,
        )

    def _compute_distances(
        self,
        f: np.ndarray,
        mu_c: np.ndarray,
        category_index: int,
    ) -> np.ndarray:
        """
        Compute distances between f and each action centroid in category c.

        Returns shape (n_actions,). Lower = more similar (for L2/Mahalanobis).

        Reference: docs/gae_design_v10_6.md §9.1; EXP-E1 (kernel comparison).
        """
        if self.kernel == KernelType.L2:
            diff = f - mu_c                          # (n_actions, n_factors)
            assert diff.shape == (self.n_actions, self.n_factors), (
                f"diff.shape={diff.shape} != ({self.n_actions}, {self.n_factors})"
            )
            return np.sum(diff ** 2, axis=1)         # squared L2, (n_actions,)

        elif self.kernel == KernelType.COSINE:
            # Cosine distance: 1 - cosine_similarity
            f_norm   = np.linalg.norm(f)
            mu_norms = np.linalg.norm(mu_c, axis=1)  # (n_actions,)
            denom    = (f_norm * mu_norms) + 1e-10
            sims     = (mu_c @ f) / denom             # (n_actions,)
            return 1.0 - sims

        elif self.kernel == KernelType.DOT:
            # Dot product: negate so that higher dot = lower "distance"
            return -(mu_c @ f)                        # (n_actions,)

        elif self.kernel == KernelType.MAHALANOBIS:
            if self._cov_inv is None:
                # Fall back to L2 with warning if covariance not set
                warnings.warn(
                    "Mahalanobis kernel: no covariance set for category "
                    f"{category_index}. Falling back to L2.",
                    UserWarning,
                    stacklevel=3,
                )
                diff = f - mu_c
                return np.sum(diff ** 2, axis=1)
            # Per-category Mahalanobis: (f-μ)^T Σ^-1 (f-μ) per action
            cov_inv_c = self._cov_inv[category_index]  # (n_actions, n_fac, n_fac)
            assert cov_inv_c.shape == (self.n_actions, self.n_factors, self.n_factors), (
                f"cov_inv_c.shape={cov_inv_c.shape} != "
                f"({self.n_actions}, {self.n_factors}, {self.n_factors})"
            )
            diff = f - mu_c                            # (n_actions, n_factors)
            distances = np.array([
                diff[a] @ cov_inv_c[a] @ diff[a]
                for a in range(self.n_actions)
            ])
            assert distances.shape == (self.n_actions,), (
                f"distances.shape={distances.shape} != ({self.n_actions},)"
            )
            return distances

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    # ------------------------------------------------------------------ #
    # Name lookup helpers                                                 #
    # ------------------------------------------------------------------ #

    def _category_name(self, c: int) -> str:
        """Return human-readable category name for index c."""
        if self.categories is not None and c < len(self.categories):
            return self.categories[c]
        return f"cat_{c}"

    def _action_name(self, a: int) -> str:
        """Return human-readable action name for index a."""
        if a < len(self.actions):
            return self.actions[a]
        return f"action_{a}"

    def freeze(self) -> None:
        """
        Freeze all centroid learning. Subsequent update() calls return a
        CentroidUpdate with centroid_delta_norm=0.0 and do not mutate mu.

        Reference: docs/gae_design_v10_6.md §9.5.
        """
        self._frozen = True

    def unfreeze(self) -> None:
        """Re-enable centroid learning after a freeze() call."""
        self._frozen = False

    def set_kernel(self, kernel) -> None:
        """
        Replace the active scoring kernel.

        Safe to call at any time, including during a frozen window. Only
        affects the scoring_kernel used on the L2 path (KernelType.L2).
        The centroid tensor μ is never modified.

        Parameters
        ----------
        kernel : ScoringKernel
            New kernel instance (L2Kernel, DiagonalKernel, …).

        Reference: docs/gae_design_v10_6.md §9; V-CGA-FROZEN gap closure.
        """
        self.scoring_kernel = kernel

    def kernel_weight_refresh(self, covariance_estimator) -> bool:
        """
        Refresh DiagonalKernel weights from accumulated CovarianceEstimator data.

        Safe to call during a frozen window (learning_enabled=False / _frozen=True).
        The centroid tensor μ is never modified — only the scoring kernel's weights
        are updated. This closes the gap identified in V-CGA-FROZEN: sigma reduction
        from graph enrichment now flows into the kernel geometry even during frozen
        periods.

        Parameters
        ----------
        covariance_estimator : CovarianceEstimator
            Estimator with accumulated factor observations.

        Returns
        -------
        bool — True if weights were updated, False if:
            - current scoring_kernel is not a DiagonalKernel (L2 has no weights), or
            - covariance_estimator has fewer than MIN_SAMPLES_FOR_SIGMA observations.

        Reference: V-CGA-FROZEN gap closure; docs/gae_design_v10_6.md §9.
        """
        from gae.kernels import DiagonalKernel as _DiagonalKernel
        if not isinstance(self.scoring_kernel, _DiagonalKernel):
            return False

        sigma = covariance_estimator.get_per_factor_sigma()
        if sigma is None:
            return False

        updated_kernel = self.scoring_kernel.refresh_weights(sigma)
        self.set_kernel(updated_kernel)
        return True

    @property
    def centroids(self) -> np.ndarray:
        """
        Profile centroid tensor, shape (n_categories, n_actions, n_factors).

        Alias for self.mu — use for read access and diagnostics.
        """
        return self.mu

    @centroids.setter
    def centroids(self, value: np.ndarray) -> None:
        """
        Set centroid tensor. Validates shape matches current mu.

        This is the public write API for centroids. Internal code
        uses self.mu directly. Consumer code should use
        scorer.centroids = new_value.
        """
        value = np.asarray(value, dtype=np.float64)
        if value.shape != self.mu.shape:
            raise ValueError(
                f"centroids shape {value.shape} != expected {self.mu.shape}"
            )
        self.mu = value

    @property
    def update_gate_stats(self) -> Dict:
        """
        Gate statistics for the min_confidence filter.

        Returns counts of applied vs gated updates and the overall gate rate.
        Source: Block 5B Proxy persona testing (min_confidence gate).

        Returns
        -------
        dict with keys: gated, applied, total, gate_rate
        """
        total = self._gated_count + self._applied_count
        return {
            'gated': self._gated_count,
            'applied': self._applied_count,
            'total': total,
            'gate_rate': self._gated_count / max(total, 1),
        }

    def set_conservation_status(self, status: str) -> None:
        """
        Update conservation monitor status and adjust learning pause state.

        Called by the ConservationMonitor when the signal crosses a threshold.
        Only has effect when auto_pause_on_amber=True.

        Parameters
        ----------
        status : str
            One of 'GREEN', 'AMBER', 'RED', 'CALIBRATING'.
            AMBER/RED → freeze learning. GREEN → resume learning.

        Reference: G6a three-judge consensus; docs/gae_design_v10_6.md §9.6.
        """
        self._conservation_status = status
        if self.auto_pause_on_amber and status in ('AMBER', 'RED'):
            self._paused_by_conservation = True
        elif status == 'GREEN':
            self._paused_by_conservation = False

    @property
    def conservation_status(self) -> str:
        """Current conservation monitor status string ('GREEN', 'AMBER', 'RED', …)."""
        return self._conservation_status

    @property
    def is_paused(self) -> bool:
        """True when learning is paused due to conservation AMBER/RED signal."""
        return self._paused_by_conservation

    # ------------------------------------------------------------------ #
    # Learning                                                            #
    # ------------------------------------------------------------------ #

    def _compute_gradient(self, f: np.ndarray, mu_single: np.ndarray) -> np.ndarray:
        """
        Gradient direction for centroid update.

        Uses scoring_kernel on the L2 path; falls back to (f − μ) for
        COSINE/DOT/MAHALANOBIS (those kernels are not differentiable here).

        Parameters
        ----------
        f : shape (n_factors,)
        mu_single : shape (n_factors,)

        Returns
        -------
        shape (n_factors,)
        """
        if self.kernel in (KernelType.L2, KernelType.DIAGONAL):
            return self.scoring_kernel.compute_gradient(f, mu_single)
        return f - mu_single

    def update(
        self,
        f: np.ndarray,
        category_index: int,
        action_index: int,
        correct: bool,
        gt_action_index: Optional[int] = None,
        confidence: Optional[float] = None,
    ) -> CentroidUpdate:
        """
        Update profile centroids from observed outcome.

        Implements Eq. 4b-final (pull/push centroid update):
          Correct:   μ[c,a,:]  += η_eff     * (f - μ[c,a,:])   [pull toward f]
          Incorrect: μ[c,a,:]  -= η_neg_eff * (f - μ[c,a,:])   [push predicted away]
                     μ[c,gt,:] += η_eff     * (f - μ[c,gt,:])  [pull GT toward f]
          All values clipped to [0.0, 1.0] after update (V2 requirement).

        Learning rate decays with experience:
          η_eff = η / (1 + decay * count[c, a])

        Args:
          f:               Factor vector, shape (n_factors,).
          category_index:  Category index c.
          action_index:    The predicted (recommended) action index a.
          correct:         Whether the action was correct (matched GT).
          gt_action_index: Ground-truth action index. Required when correct=False
                           to enable GT-pull learning. If omitted, falls back to
                           push-predicted-only with a DeprecationWarning.
          confidence:      System's confidence in its original decision (0–1).
                           If provided and below min_confidence, update is skipped
                           (gated) and centroids are not modified. If None, the
                           gate is bypassed entirely (backward compatible).

        Note:
          When ProfileScorer is constructed with eta_override, the override path
          (correct=False) uses eta_override for both push-away and GT-pull operations.
          The confirm path (correct=True) always uses self.eta unchanged (Q5 validated).

        Returns:
          CentroidUpdate capturing the magnitude of centroid movement.
          outcome='gated_low_confidence' when the gate fires (centroid_delta_norm=0.0).

        Reference: docs/gae_design_v10_6.md §9.5; blog Eq. 4b-final; V2 (clipping).
        """
        f = np.asarray(f, dtype=np.float64)
        assert f.shape == (self.n_factors,), (
            f"f.shape={f.shape} must be ({self.n_factors},)"
        )
        if not np.all(np.isfinite(f)):
            raise ValueError("Factor vector contains NaN or Inf values")
        if self.tau <= 0:
            raise ValueError(
                f"Temperature tau must be positive, got {self.tau}"
            )
        c = category_index
        a = action_index

        # Conservation auto-pause (G6a — AMBER/RED blocks all centroid updates)
        if self._paused_by_conservation:
            self._gated_count += 1
            self.decision_count += 1
            return CentroidUpdate(
                centroid_delta_norm=0.0,
                category_index=c,
                action_index=a,
                category_name=self._category_name(c),
                action_name=self._action_name(a),
                decision_count=self.decision_count,
                gt_delta_norm=0.0,
                outcome='paused_conservation',
            )

        # Min-confidence gate (Block 5B Proxy — bypassed when confidence=None)
        if confidence is not None and confidence < self.min_confidence:
            self._gated_count += 1
            self.decision_count += 1
            return CentroidUpdate(
                centroid_delta_norm=0.0,
                category_index=c,
                action_index=a,
                category_name=self._category_name(c),
                action_name=self._action_name(a),
                decision_count=self.decision_count,
                gt_delta_norm=0.0,
                outcome='gated_low_confidence',
            )

        if self._frozen:
            self.decision_count += 1
            return CentroidUpdate(
                centroid_delta_norm=0.0,
                category_index=c,
                action_index=a,
                category_name=self._category_name(c),
                action_name=self._action_name(a),
                decision_count=self.decision_count,
                gt_delta_norm=0.0,
            )

        count = self.counts[c, a]
        eta_eff     = self.eta     / (1.0 + self.decay * count)
        eta_neg_eff = self.eta_neg / (1.0 + self.decay * count)

        if (self.eta_override is None
                and self.eta != self.eta_neg
                and not getattr(self, '_eta_override_warned', False)):
            warnings.warn(
                "ProfileScorer constructed with eta_override=None and "
                f"asymmetric rates (eta={self.eta}, eta_neg={self.eta_neg}). "
                "On future override calls, the override path will use "
                "eta_neg, not the validated P0 attenuation. "
                "Pass eta_override=0.01 (SOC default) or "
                "call compute_eta_override() for your domain.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._eta_override_warned = True

        # Asymmetric η (Q5 validated): override path uses attenuated rate
        # for both push-away and GT-pull. None = backward compatible.
        if not correct and self.eta_override is not None:
            eta_override_eff = self.eta_override / (1.0 + self.decay * count)
        else:
            eta_override_eff = None

        gt_delta_norm = 0.0

        if correct:
            # Confirmation path — clean signal, full learning rate (unchanged)
            delta_vector = eta_eff * self._compute_gradient(f, self.mu[c, a, :])
            if self.factor_mask is not None:
                delta_vector = delta_vector * self.factor_mask
            delta_vector = np.clip(delta_vector, -MAX_ETA_DELTA, MAX_ETA_DELTA)  # V-STABILITY F=8.14
            centroid_delta_norm = float(np.linalg.norm(delta_vector))
            self.mu[c, a, :] += delta_vector
        else:
            # Override path — select attenuated rate when eta_override is set
            push_rate = eta_override_eff if eta_override_eff is not None else eta_neg_eff
            pull_rate = eta_override_eff if eta_override_eff is not None else eta_eff

            if gt_action_index is None:
                # Backward-compat: push predicted centroid only, no GT pull.
                # Callers should provide gt_action_index for correct learning.
                warnings.warn(
                    "ProfileScorer.update() called with correct=False but no "
                    "gt_action_index. Falling back to push-predicted-only. "
                    "Pass gt_action_index for full push-predicted + pull-GT "
                    "learning (Eq. 4b-final).",
                    DeprecationWarning,
                    stacklevel=2,
                )
                delta_vector = -push_rate * self._compute_gradient(f, self.mu[c, a, :])
                if self.factor_mask is not None:
                    delta_vector = delta_vector * self.factor_mask
                delta_vector = np.clip(delta_vector, -MAX_ETA_DELTA, MAX_ETA_DELTA)  # V-STABILITY F=8.14
                centroid_delta_norm = float(np.linalg.norm(delta_vector))
                self.mu[c, a, :] += delta_vector
            else:
                gt = gt_action_index
                # Push predicted (wrong) centroid away from f
                delta_vector = -push_rate * self._compute_gradient(f, self.mu[c, a, :])
                if self.factor_mask is not None:
                    delta_vector = delta_vector * self.factor_mask
                delta_vector = np.clip(delta_vector, -MAX_ETA_DELTA, MAX_ETA_DELTA)  # V-STABILITY F=8.14
                centroid_delta_norm = float(np.linalg.norm(delta_vector))
                self.mu[c, a, :] += delta_vector
                # Pull ground-truth centroid toward f
                gt_delta_vector = pull_rate * self._compute_gradient(f, self.mu[c, gt, :])
                if self.factor_mask is not None:
                    gt_delta_vector = gt_delta_vector * self.factor_mask
                gt_delta_vector = np.clip(gt_delta_vector, -MAX_ETA_DELTA, MAX_ETA_DELTA)  # V-STABILITY F=8.14
                gt_delta_norm = float(np.linalg.norm(gt_delta_vector))
                self.mu[c, gt, :] += gt_delta_vector

        # V2 requirement: clip ALL centroids after update to prevent escape
        self.mu[c, :, :] = np.clip(self.mu[c, :, :], 0.0, 1.0)

        # Increment observation count for this (category, action) pair
        self.counts[c, a] += 1
        self.decision_count += 1
        self._applied_count += 1

        return CentroidUpdate(
            centroid_delta_norm=centroid_delta_norm,
            category_index=c,
            action_index=a,
            category_name=self._category_name(c),
            action_name=self._action_name(a),
            decision_count=self.decision_count,
            gt_delta_norm=gt_delta_norm,
        )

    # ------------------------------------------------------------------ #
    # Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def diagnostics(self) -> Dict:
        """
        Return per-category centroid separation diagnostics.

        Centroid separation = mean pairwise L2 distance between action
        centroids within a category. Higher = more discriminative profiles.

        Rule of thumb:
          separation > 0.5:   well-separated, high accuracy expected
          separation 0.2–0.5: moderate, may need more data
          separation < 0.2:   poorly discriminative — review profile design

        Returns dict with:
          per_category: {category_idx: {"separation": float, "min_dist": float}}
          overall_mean_separation: float
          decisions_per_category: list of per-action counts summed per category

        Reference: docs/gae_design_v10_6.md §9.6.
        """
        per_category: Dict = {}
        separations: List[float] = []

        for c in range(self.n_categories):
            mu_c = self.mu[c]  # (n_actions, n_factors)
            assert mu_c.shape == (self.n_actions, self.n_factors), (
                f"mu_c.shape={mu_c.shape} != ({self.n_actions}, {self.n_factors})"
            )
            dists = []
            for a1 in range(self.n_actions):
                for a2 in range(a1 + 1, self.n_actions):
                    d = float(np.sqrt(np.sum((mu_c[a1] - mu_c[a2]) ** 2)))
                    dists.append(d)
            mean_sep = float(np.mean(dists)) if dists else 0.0
            min_dist = float(np.min(dists)) if dists else 0.0
            per_category[c] = {
                "separation": mean_sep,
                "min_dist": min_dist,
            }
            separations.append(mean_sep)

        return {
            "per_category": per_category,
            "overall_mean_separation": float(np.mean(separations)),
            "decisions_per_category": [
                int(self.counts[c].sum()) for c in range(self.n_categories)
            ],
        }

    def set_covariance(self, cov_inv: np.ndarray) -> None:
        """
        Set per-category covariance inverse matrices for Mahalanobis kernel.

        Args:
          cov_inv: shape (n_categories, n_actions, n_factors, n_factors).
                   Pre-inverted per-category-action covariance matrices.
                   Compute from category-filtered training data only.
                   Global covariance is harmful — FX-1-CORRECTED.

        Reference: docs/gae_design_v10_6.md §9.1; FX-1-CORRECTED.
        """
        assert self.kernel == KernelType.MAHALANOBIS, (
            "set_covariance() is only for KernelType.MAHALANOBIS"
        )
        expected = (self.n_categories, self.n_actions, self.n_factors, self.n_factors)
        assert cov_inv.shape == expected, (
            f"cov_inv.shape={cov_inv.shape} must be {expected}"
        )
        self._cov_inv = cov_inv.astype(np.float64)

    # ------------------------------------------------------------------ #
    # Factory                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def init_from_config(
        cls,
        config_dict: Dict,
        actions: List[str],
        profile: Optional["CalibrationProfile"] = None,
    ) -> "ProfileScorer":
        """
        Build ProfileScorer from a nested config dictionary.

        config_dict format:
          {
            "categories": ["cat_0", "cat_1", ...],
            "centroids": {
              "cat_0": {
                "action_0": [f0, f1, f2, ...],
                "action_1": [f0, f1, f2, ...],
              },
              ...
            },
            "kernel": "l2",   # optional, default "l2"
          }

        Missing category/action combinations default to 0.5 (uniform prior).

        Reference: docs/gae_design_v10_6.md §9.7.
        """
        categories = config_dict["categories"]
        centroids  = config_dict["centroids"]

        # Infer n_factors from first defined centroid
        n_fac: Optional[int] = None
        for cat_name in categories:
            if cat_name in centroids:
                for act_name in actions:
                    if act_name in centroids[cat_name]:
                        n_fac = len(centroids[cat_name][act_name])
                        break
            if n_fac is not None:
                break
        if n_fac is None:
            raise ValueError(
                "config_dict contains no centroid values — "
                "cannot infer n_factors."
            )

        n_cat = len(categories)
        n_act = len(actions)

        # Build mu tensor, defaulting to 0.5 for missing entries
        mu = np.full((n_cat, n_act, n_fac), 0.5, dtype=np.float64)
        assert mu.shape == (n_cat, n_act, n_fac), (
            f"mu.shape={mu.shape} != ({n_cat}, {n_act}, {n_fac})"
        )

        for c, cat_name in enumerate(categories):
            if cat_name in centroids:
                for a, act_name in enumerate(actions):
                    if act_name in centroids[cat_name]:
                        values = centroids[cat_name][act_name]
                        if len(values) != n_fac:
                            raise ValueError(
                                f"Centroid [{cat_name}][{act_name}] has "
                                f"{len(values)} factors, expected {n_fac}."
                            )
                        mu[c, a, :] = np.array(values, dtype=np.float64)

        # Validate all values in [0, 1]
        if np.any(mu < 0.0) or np.any(mu > 1.0):
            raise ValueError(
                "All centroid values must be in [0.0, 1.0]. "
                "Check your config_dict."
            )

        kernel_str = config_dict.get("kernel", "l2").lower()
        kernel = KernelType(kernel_str)

        return cls(mu=mu, actions=actions, kernel=kernel, profile=profile)


def build_profile_scorer(
    categories: List[str],
    actions: List[str],
    centroids: Dict,
    n_factors: int,
    kernel: KernelType = KernelType.L2,
    profile: Optional["CalibrationProfile"] = None,
) -> ProfileScorer:
    """
    Convenience factory for building a ProfileScorer from explicit arguments.

    Args:
      categories: List of category names.
      actions:    List of action names.
      centroids:  Nested dict {cat_name: {act_name: [factor_values]}}.
                  Missing entries default to 0.5.
      n_factors:  Number of factors per centroid (used for validation only;
                  actual n_factors inferred from centroids dict).
      kernel:     Distance kernel (default L2).
      profile:    CalibrationProfile for hyperparameters.

    Reference: docs/gae_design_v10_6.md §9.7.
    """
    config_dict = {
        "categories": categories,
        "centroids": centroids,
        "kernel": kernel.value,
    }
    return ProfileScorer.init_from_config(config_dict, actions, profile)
