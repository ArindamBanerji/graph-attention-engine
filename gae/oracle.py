"""
GAE Oracle — ground-truth outcome providers for evaluation and simulation.

OracleProvider is a Protocol for generating GT labels. Two implementations:
  GTAlignedOracle: GT from centroid proximity — the "perfect labeler"
  BernoulliOracle: random correctness at configurable rate

The oracle does NOT score — it only answers "was this action correct?"
Used by evaluation suite (GAE-EVAL-1) and ablation baselines (GAE-ABL-1).

Retained from retired bridge_layer design (TD-032).

Reference: docs/gae_design_v5.md §10; EXP-C1 (GTAlignedOracle accuracy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class OracleResult:
    """
    Result from an OracleProvider query.

    Reference: docs/gae_design_v5.md §10.1.

    Attributes
    ----------
    correct : bool
        Whether the taken action was correct per GT.
    gt_action_idx : int
        Ground-truth action index.
    gt_action_name : str
        Ground-truth action name.
    confidence : float
        Oracle confidence in GT label ∈ [0, 1].
        1.0 for GTAlignedOracle (deterministic centroid nearest-neighbor).
        correct_rate for BernoulliOracle.
    """

    correct: bool
    gt_action_idx: int
    gt_action_name: str
    confidence: float


@runtime_checkable
class OracleProvider(Protocol):
    """
    Protocol for ground-truth outcome generation.

    Implementations provide GT labels for evaluation and simulation.
    The oracle does not score — it only labels.

    Used by: GAE-EVAL-1, GAE-ABL-1, SOC simulation mode.

    Reference: docs/gae_design_v5.md §10.2.
    """

    def query(
        self,
        f: np.ndarray,
        category_index: int,
        taken_action_index: int,
    ) -> OracleResult:
        """
        Return ground-truth outcome for a decision.

        Args:
          f:                  Factor vector, shape (n_factors,).
          category_index:     Category of the alert/entity.
          taken_action_index: The action that was taken.

        Returns:
          OracleResult with correct flag and GT action.
        """
        ...


class GTAlignedOracle:
    """
    Oracle that derives GT from profile centroid proximity.

    GT action = argmin squared-L2 distance from f to centroids in
    category c. This is the "perfect labeler" — knows the true profile
    structure.

    Correctness: taken_action correct iff it matches the centroid-nearest
    action. On well-separated bimodal centroids produces ~97.89% accuracy
    (EXP-C1 reference).

    Reference: docs/gae_design_v5.md §10.3; EXP-C1.
    """

    def __init__(self, mu: np.ndarray, actions: list[str]) -> None:
        """
        Args:
          mu:      Profile centroids, shape (n_cat, n_act, n_fac).
                   Copied — caller's array is not mutated.
          actions: Ordered action names. len(actions) == mu.shape[1].

        Reference: docs/gae_design_v5.md §10.3.
        """
        assert mu.ndim == 3, (
            f"mu must be 3-D (n_cat, n_act, n_fac), got shape {mu.shape}"
        )
        assert mu.shape[1] == len(actions), (
            f"mu.shape[1]={mu.shape[1]} must equal len(actions)={len(actions)}"
        )
        self.mu = mu.copy().astype(np.float64)
        self.actions = list(actions)

    def query(
        self,
        f: np.ndarray,
        category_index: int,
        taken_action_index: int,
    ) -> OracleResult:
        """
        GT = centroid-nearest action for category c.

        Args:
          f:                  Factor vector, shape (n_factors,).
          category_index:     Category index c.
          taken_action_index: Action that was taken.

        Returns:
          OracleResult. confidence=1.0 (deterministic).

        Reference: docs/gae_design_v5.md §10.3; blog Eq. 4-final (L2 kernel).
        """
        f = np.asarray(f, dtype=np.float64)
        mu_c = self.mu[category_index]              # (n_actions, n_factors)
        assert mu_c.ndim == 2, (
            f"mu_c.ndim={mu_c.ndim}, expected 2"
        )
        diff = f - mu_c                              # (n_actions, n_factors)
        dists = np.sum(diff ** 2, axis=1)            # (n_actions,)
        assert dists.shape == (mu_c.shape[0],), (
            f"dists.shape={dists.shape} != ({mu_c.shape[0]},)"
        )
        gt_idx = int(np.argmin(dists))
        correct = (taken_action_index == gt_idx)
        return OracleResult(
            correct=correct,
            gt_action_idx=gt_idx,
            gt_action_name=self.actions[gt_idx],
            confidence=1.0,
        )

    @classmethod
    def from_profile_scorer(cls, scorer) -> "GTAlignedOracle":
        """
        Build GTAlignedOracle from an existing ProfileScorer.

        Shares the same centroid geometry — GT labels align perfectly
        with the scorer's decision boundaries.

        Args:
          scorer: A ProfileScorer instance (duck-typed; reads .mu, .actions).

        Reference: docs/gae_design_v5.md §10.3.
        """
        return cls(mu=scorer.mu, actions=scorer.actions)


class BernoulliOracle:
    """
    Oracle that returns correct with fixed probability p.

    Used in ablation RANDOM baseline (p=0.25 for 4 actions) and noise
    experiments. When incorrect, GT is drawn uniformly from non-taken
    actions.

    Reference: docs/gae_design_v5.md §10.4; GAE-ABL-1.
    """

    def __init__(
        self,
        n_actions: int,
        actions: list[str],
        correct_rate: float = 0.25,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
          n_actions:    Number of possible actions.
          actions:      Ordered action names. len(actions) == n_actions.
          correct_rate: P(correct) ∈ [0, 1]. Default 0.25 = random for
                        4-action problems.
          seed:         Random seed for reproducibility.

        Reference: docs/gae_design_v5.md §10.4.
        """
        assert 0.0 <= correct_rate <= 1.0, (
            f"correct_rate {correct_rate} must be in [0.0, 1.0]"
        )
        assert n_actions == len(actions), (
            f"n_actions={n_actions} must equal len(actions)={len(actions)}"
        )
        self.n_actions = n_actions
        self.actions = list(actions)
        self.correct_rate = correct_rate
        self.rng = np.random.default_rng(seed)

    def query(
        self,
        f: np.ndarray,
        category_index: int,
        taken_action_index: int,
    ) -> OracleResult:
        """
        Return correct=True with probability correct_rate.

        When incorrect, GT drawn uniformly from non-taken actions.
        confidence = correct_rate (reflects oracle uncertainty).

        Args:
          f:                  Factor vector (not used — Bernoulli is f-agnostic).
          category_index:     Category index (not used).
          taken_action_index: Action that was taken.

        Returns:
          OracleResult. confidence = self.correct_rate.

        Reference: docs/gae_design_v5.md §10.4; GAE-ABL-1.
        """
        if self.rng.random() < self.correct_rate:
            gt_idx = taken_action_index
            correct = True
        else:
            other = [i for i in range(self.n_actions)
                     if i != taken_action_index]
            gt_idx = int(self.rng.choice(other)) if other else taken_action_index
            correct = False
        return OracleResult(
            correct=correct,
            gt_action_idx=gt_idx,
            gt_action_name=self.actions[gt_idx],
            confidence=self.correct_rate,
        )
