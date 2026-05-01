"""
Two-phase learning state and freeze policies for category-level scoring.

`ProfileScorer.score()` computes one softmax across all actions inside a
category. For that reason, two-phase state is tracked per category rather than
per `(category, action)` pair.

FW-01 intentionally introduces only standalone state and policy helpers.
Callers may continue invoking update paths as they do today; later work gates
mean updates by phase without changing the fact that verified decisions are
still recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


MEAN_CONVERGENCE = "MEAN_CONVERGENCE"
VARIANCE_LEARNING = "VARIANCE_LEARNING"


@dataclass
class CategoryState:
    """
    Per-category two-phase learning state.

    This state is category-level because `score()` compares all actions in a
    category in one softmax. Pair-level `(category, action)` state is
    intentionally not used.
    """

    phase: str = MEAN_CONVERGENCE
    n_decisions: int = 0
    freeze_point: Optional[int] = None

    def record_decision(self) -> None:
        """Record one verified decision for this category."""
        self.n_decisions += 1

    def freeze(self) -> None:
        """
        Transition to variance learning and capture the first freeze point.

        The freeze point is the category decision count at the moment of first
        freeze. Repeated calls are idempotent and preserve the original count.
        """
        self.phase = VARIANCE_LEARNING
        if self.freeze_point is None:
            self.freeze_point = self.n_decisions


class PhasePolicy(Protocol):
    """Protocol for category-level freeze policies."""

    def should_freeze(self, state: CategoryState) -> bool:
        """Return True when the category should transition to phase 2."""
        ...


class DecisionCountPolicy:
    """Freeze after a category has accumulated at least `n` verified decisions."""

    def __init__(self, n: int = 200) -> None:
        self.n = n

    def should_freeze(self, state: CategoryState) -> bool:
        """Freeze when the category decision count reaches the configured threshold."""
        return state.n_decisions >= self.n


class ManualPolicy:
    """Never auto-freeze; callers must invoke `CategoryState.freeze()` explicitly."""

    def should_freeze(self, state: CategoryState) -> bool:
        """Manual policy is inert unless the caller freezes the state directly."""
        return False


class RollingAccuracyDeltaPolicy:
    """
    Placeholder for rolling-accuracy freeze logic.

    This policy is FW-05+ dependent and is intentionally inert in FW-01 so
    that introducing the module does not change runtime behavior.
    """

    def __init__(self, threshold_pp: float = 0.5) -> None:
        self.threshold_pp = threshold_pp

    def should_freeze(self, state: CategoryState) -> bool:
        """Inert placeholder until rolling-accuracy inputs exist."""
        return False
