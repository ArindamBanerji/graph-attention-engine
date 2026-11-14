"""Scorer-level comparison primitive for SDK AgentEvolver integrations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from gae.profile_scorer import ProfileScorer


class ComparisonMode(Enum):
    """How a candidate scorer is compared with its baseline."""

    ALL_CATEGORIES = "all"
    WEIGHTED_AVERAGE = "weighted"
    ANY_IMPROVEMENT = "any"


@dataclass(frozen=True)
class EvaluationResult:
    """Immutable scorer-comparison observation with no deployment side effect."""

    verdict: str
    base_accuracy: Mapping[str, float]
    candidate_accuracy: Mapping[str, float]
    delta: Mapping[str, float]
    candidate_verified: int
    candidate_conservation: str
    elapsed_seconds: float
    reason: str

    def __post_init__(self) -> None:
        """Copy and freeze comparison maps for the verified-outcome audit."""
        for name in ("base_accuracy", "candidate_accuracy", "delta"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


class AgentEvolver:
    """Scorer-level comparison for :mod:`copilot_sdk.evolution.evolver`.

    This is not the SDK orchestrator. It compares two ProfileScorer instances
    that already accumulated verified outcomes, and returns a conservation-
    gated verdict for an SDK ShadowRunner or PromotionGate to consume.
    """

    _ANY_REGRESSION_TOLERANCE = 0.01

    def __init__(
        self,
        base_scorer: ProfileScorer,
        candidate_scorer: ProfileScorer,
        min_decisions: int = 200,
        timeout_seconds: float = 172800,
        comparison_mode: ComparisonMode = ComparisonMode.ALL_CATEGORIES,
        variant_id: str = "unnamed",
    ) -> None:
        if base_scorer.n_categories != candidate_scorer.n_categories:
            raise ValueError("base and candidate must have the same category count")
        if min_decisions < 1:
            raise ValueError("min_decisions must be positive")
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        self.base_scorer = base_scorer
        self.candidate_scorer = candidate_scorer
        self.min_decisions = int(min_decisions)
        self.timeout_seconds = float(timeout_seconds)
        self.comparison_mode = ComparisonMode(comparison_mode)
        self.variant_id = str(variant_id)
        self._started_at = time.monotonic()
        self._history: list[EvaluationResult] = []
        self._category_map = self._category_names()

    def evaluate(self) -> EvaluationResult:
        """Return a verdict from current scorer observations and append it to history."""
        elapsed = time.monotonic() - self._started_at
        base, candidate, delta = self._comparison_values()
        verified = sum(
            self.candidate_scorer.verified_count(ci)
            for _, ci in self._category_map.values()
        )
        conservation = str(self.candidate_scorer.conservation_status).upper()
        if elapsed >= self.timeout_seconds:
            verdict, reason = "timeout", "comparison window expired"
        elif verified < self.min_decisions:
            verdict, reason = "waiting", f"{verified}/{self.min_decisions} verified decisions"
        elif conservation != "GREEN":
            verdict, reason = "waiting", f"candidate conservation is {conservation}, not GREEN"
        else:
            verdict, reason = self._compare(base, candidate, delta, verified)
        result = EvaluationResult(
            verdict=verdict,
            base_accuracy=MappingProxyType(base),
            candidate_accuracy=MappingProxyType(candidate),
            delta=MappingProxyType(delta),
            candidate_verified=verified,
            candidate_conservation=conservation,
            elapsed_seconds=elapsed,
            reason=reason,
        )
        self._history.append(result)
        return result

    @property
    def history(self) -> list[EvaluationResult]:
        """Return a shallow copy of the immutable evaluation audit trail."""
        return list(self._history)

    def _category_names(self) -> dict[str, tuple[int, int]]:
        """Align verified accuracy by category identity, not tensor position."""
        base = self.base_scorer.categories
        candidate = self.candidate_scorer.categories
        if base is None:
            base = [f"cat_{i}" for i in range(self.base_scorer.n_categories)]
        if candidate is None:
            candidate = [f"cat_{i}" for i in range(self.candidate_scorer.n_categories)]
        if (len(base) != self.base_scorer.n_categories
                or len(candidate) != self.candidate_scorer.n_categories
                or len(set(base)) != len(base)
                or len(set(candidate)) != len(candidate)
                or set(base) != set(candidate)):
            raise ValueError("Base and candidate must have identical category sets with unique names")
        candidate_indices = {name: i for i, name in enumerate(candidate)}
        return {name: (i, candidate_indices[name]) for i, name in enumerate(base)}

    def _comparison_values(self) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        base = {name: self.base_scorer.accuracy(bi) for name, (bi, ci) in self._category_map.items()}
        candidate = {name: self.candidate_scorer.accuracy(ci) for name, (bi, ci) in self._category_map.items()}
        return base, candidate, {name: candidate[name] - base[name] for name in base}

    def _compare(
        self,
        base: dict[str, float],
        candidate: dict[str, float],
        delta: dict[str, float],
        verified: int,
    ) -> tuple[str, str]:
        if self.comparison_mode is ComparisonMode.ALL_CATEGORIES:
            passed = all(value >= 0.0 for value in delta.values())
        elif self.comparison_mode is ComparisonMode.WEIGHTED_AVERAGE:
            weights = {name: self.candidate_scorer.verified_count(ci) for name, (_, ci) in self._category_map.items()}
            total = sum(weights.values())
            passed = total > 0 and sum(delta[name] * weights[name] for name in delta) / total >= 0.0
        else:
            passed = any(value > 0.0 for value in delta.values()) and all(
                value >= -self._ANY_REGRESSION_TOLERANCE for value in delta.values()
            )
        if passed:
            return "promoted", f"{self.variant_id} passed {self.comparison_mode.value} comparison at {verified} decisions"
        return "rolled_back", f"{self.variant_id} failed {self.comparison_mode.value} comparison"
