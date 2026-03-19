"""
GAE ProfileScorer — centroid-proximity scoring for v5.0.

Replaces ScoringMatrix (deprecated, TD-029). Architecture settled by 14
experiments. Key validated numbers:
  - L2 kernel: 97.89% oracle accuracy (EXP-C1), 98.2% with learning (EXP-B1)
  - τ=0.1: ECE=0.036 (V3B). τ=0.25 was wrong (ECE=0.19).
  - Centroid clipping [0,1]: required (V2: escape at dec 6-12 without it)
  - Global Mahalanobis: harmful on multi-category data (FX-1-CORRECTED)
  - DOT product: 61% on [0,1] factors (EXP-C1)

Reference: docs/gae_design_v5.md §9; blog Eq. 4-final, 4b-final.
"""

from __future__ import annotations

import warnings

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from gae.calibration import CalibrationProfile


@dataclass
class CentroidUpdate:
    """
    Return value from ProfileScorer.update().

    Captures the magnitude of centroid movement for a single learning step.
    Used for visualization, Neo4j persistence, and drift monitoring.

    Reference: docs/gae_design_v5.md §9.5; IKS pipeline.
    """

    centroid_delta_norm: float   # ‖η*(f - μ[c,a,:])‖₂ before in-place update (predicted push)
    category_index: int          # c — the alert category updated
    action_index: int            # a — the verified action updated
    category_name: str           # human-readable, for logging
    action_name: str             # human-readable, for logging
    decision_count: int          # total decisions in this ProfileScorer instance
    gt_delta_norm: float = 0.0   # ‖η*(f - μ[c,gt,:])‖₂ for gt pull (0.0 if correct=True or no gt)


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

    Reference: docs/gae_design_v5.md §9.1; EXP-E1.
    """

    L2 = "l2"
    MAHALANOBIS = "mahalanobis"
    COSINE = "cosine"
    DOT = "dot"


@dataclass
class ScoringResult:
    """
    Result of ProfileScorer.score().

    Reference: docs/gae_design_v5.md §9.2; blog Eq. 4-final.

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
    """

    action_index: int
    action_name: str
    probabilities: np.ndarray   # shape (n_actions,)
    distances: np.ndarray       # shape (n_actions,)
    confidence: float


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

    Reference: docs/gae_design_v5.md §9; blog Eq. 4-final, 4b-final.
    """

    def __init__(
        self,
        mu: np.ndarray,
        actions: List[str],
        kernel: KernelType = KernelType.L2,
        profile: Optional["CalibrationProfile"] = None,
        categories: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
          mu:      Initial profile centroids, shape (n_categories, n_actions, n_factors).
                   Values should be in [0.0, 1.0]. Copied — caller's array not mutated.
          actions: Ordered action names. len(actions) == mu.shape[1].
          kernel:  Distance kernel. L2 is validated default.
          profile: CalibrationProfile for hyperparameters. If None, uses validated
                   defaults: τ=0.1, η=0.05, η_neg=0.05, decay=0.001.

        Reference: docs/gae_design_v5.md §9.3; V3B (τ), V2 (clipping).
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

        # Per-(category, action) observation counts for learning rate decay
        self.counts = np.zeros((self.n_categories, self.n_actions), dtype=np.int64)
        self._frozen: bool = False
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

        Reference: docs/gae_design_v5.md §9.4; blog Eq. 4-final.
        """
        assert 0 <= category_index < self.n_categories, (
            f"category_index {category_index} out of range [0, {self.n_categories})"
        )
        f = np.asarray(f, dtype=np.float64)
        assert f.shape == (self.n_factors,), (
            f"f.shape={f.shape} must be ({self.n_factors},)"
        )

        mu_c = self.mu[category_index]  # shape (n_actions, n_factors)
        assert mu_c.shape == (self.n_actions, self.n_factors), (
            f"mu_c.shape={mu_c.shape} != ({self.n_actions}, {self.n_factors})"
        )

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
        return ScoringResult(
            action_index=action_idx,
            action_name=self.actions[action_idx],
            probabilities=probs,
            distances=distances,
            confidence=float(probs[action_idx]),
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

        Reference: docs/gae_design_v5.md §9.1; EXP-E1 (kernel comparison).
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

        Reference: docs/gae_design_v5.md §9.5.
        """
        self._frozen = True

    def unfreeze(self) -> None:
        """Re-enable centroid learning after a freeze() call."""
        self._frozen = False

    # ------------------------------------------------------------------ #
    # Learning                                                            #
    # ------------------------------------------------------------------ #

    def update(
        self,
        f: np.ndarray,
        category_index: int,
        action_index: int,
        correct: bool,
        gt_action_index: Optional[int] = None,
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

        Returns:
          CentroidUpdate capturing the magnitude of centroid movement.

        Reference: docs/gae_design_v5.md §9.5; blog Eq. 4b-final; V2 (clipping).
        """
        f = np.asarray(f, dtype=np.float64)
        assert f.shape == (self.n_factors,), (
            f"f.shape={f.shape} must be ({self.n_factors},)"
        )
        c = category_index
        a = action_index

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

        gt_delta_norm = 0.0

        if correct:
            # Pull predicted centroid toward f (predicted == ground truth here)
            delta_vector = eta_eff * (f - self.mu[c, a, :])
            centroid_delta_norm = float(np.linalg.norm(delta_vector))
            self.mu[c, a, :] += delta_vector
        else:
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
                delta_vector = -eta_neg_eff * (f - self.mu[c, a, :])
                centroid_delta_norm = float(np.linalg.norm(delta_vector))
                self.mu[c, a, :] += delta_vector
            else:
                gt = gt_action_index
                # Push predicted (wrong) centroid away from f
                delta_vector = -eta_neg_eff * (f - self.mu[c, a, :])
                centroid_delta_norm = float(np.linalg.norm(delta_vector))
                self.mu[c, a, :] += delta_vector
                # Pull ground-truth centroid toward f
                gt_delta_vector = eta_eff * (f - self.mu[c, gt, :])
                gt_delta_norm = float(np.linalg.norm(gt_delta_vector))
                self.mu[c, gt, :] += gt_delta_vector

        # V2 requirement: clip ALL centroids after update to prevent escape
        self.mu[c, :, :] = np.clip(self.mu[c, :, :], 0.0, 1.0)

        # Increment observation count for this (category, action) pair
        self.counts[c, a] += 1
        self.decision_count += 1

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

        Reference: docs/gae_design_v5.md §9.6.
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

        Reference: docs/gae_design_v5.md §9.1; FX-1-CORRECTED.
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

        Reference: docs/gae_design_v5.md §9.7.
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

    Reference: docs/gae_design_v5.md §9.7.
    """
    config_dict = {
        "categories": categories,
        "centroids": centroids,
        "kernel": kernel.value,
    }
    return ProfileScorer.init_from_config(config_dict, actions, profile)
