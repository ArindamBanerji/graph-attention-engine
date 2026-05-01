"""
Batch-oriented helpers for composing candidate weight updates.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Protocol

import numpy as np


class BatchCompositionPolicy(Protocol):
    """Protocol for deciding when a batch should be evaluated."""

    def should_trigger(
        self,
        accumulator: float,
        n_verified_decisions: int,
        category_index: int,
    ) -> bool:
        """Return True when batch evaluation should run."""
        ...

    def record_trigger(self, category_index: int, n_verified_decisions: int) -> None:
        """Record that a batch evaluation was attempted."""
        ...


class NoveltyThresholdPolicy:
    """Trigger when accumulated novelty exceeds a threshold."""

    def __init__(
        self,
        threshold: float,
        min_decisions: int = 1,
        cooldown: int = 0,
    ) -> None:
        if threshold < 0.0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        if min_decisions < 1:
            raise ValueError(f"min_decisions must be >= 1, got {min_decisions}")
        if cooldown < 0:
            raise ValueError(f"cooldown must be >= 0, got {cooldown}")
        self.threshold = float(threshold)
        self.min_decisions = int(min_decisions)
        self.cooldown = int(cooldown)
        self._last_trigger: Dict[int, int] = {}

    def should_trigger(
        self,
        accumulator: float,
        n_verified_decisions: int,
        category_index: int,
    ) -> bool:
        """Return True when the threshold and cooldown checks pass."""
        if n_verified_decisions < self.min_decisions:
            return False
        if not self._cooldown_elapsed(category_index, n_verified_decisions):
            return False
        return accumulator >= self.threshold

    def record_trigger(self, category_index: int, n_verified_decisions: int) -> None:
        """Record the decision count at which evaluation was attempted."""
        self._last_trigger[category_index] = int(n_verified_decisions)

    def _cooldown_elapsed(self, category_index: int, n_verified_decisions: int) -> bool:
        if category_index not in self._last_trigger:
            return True
        return (n_verified_decisions - self._last_trigger[category_index]) >= self.cooldown


class FixedIntervalPolicy:
    """Trigger every fixed number of verified decisions."""

    def __init__(self, interval: int, cooldown: int = 0) -> None:
        if interval < 1:
            raise ValueError(f"interval must be >= 1, got {interval}")
        if cooldown < 0:
            raise ValueError(f"cooldown must be >= 0, got {cooldown}")
        self.interval = int(interval)
        self.cooldown = int(cooldown)
        self._last_trigger: Dict[int, int] = {}

    def should_trigger(
        self,
        accumulator: float,
        n_verified_decisions: int,
        category_index: int,
    ) -> bool:
        """Return True on interval boundaries when cooldown has elapsed."""
        del accumulator
        if n_verified_decisions < self.interval:
            return False
        if (n_verified_decisions % self.interval) != 0:
            return False
        if category_index not in self._last_trigger:
            return True
        return (n_verified_decisions - self._last_trigger[category_index]) >= self.cooldown

    def record_trigger(self, category_index: int, n_verified_decisions: int) -> None:
        """Record the decision count at which evaluation was attempted."""
        self._last_trigger[category_index] = int(n_verified_decisions)


@dataclass
class GateVerdict:
    """Outcome of evaluating whether candidate weights should be promoted."""

    promoted: bool
    superiority_delta: float
    old_accuracy: float
    new_accuracy: float
    superiority_pass: bool
    floor_pass: bool
    conservation_pass: bool
    variance_pass: bool
    var_ratio: float
    reason: str


class PromotionGate(Protocol):
    """Protocol for deciding whether candidate weights should be promoted."""

    def evaluate(
        self,
        old_accuracy: float,
        new_accuracy: float,
        old_weights: Optional[np.ndarray],
        new_weights: np.ndarray,
    ) -> GateVerdict:
        """Return a verdict for the candidate weights."""
        ...


class DefaultPromotionGate:
    """Default gate that checks accuracy and variance stability."""

    def __init__(
        self,
        superiority_margin: float = 0.05,
        floor: float = 0.75,
        max_variance_ratio: float = 2.0,
    ) -> None:
        if superiority_margin < 0.0:
            raise ValueError(
                f"superiority_margin must be >= 0, got {superiority_margin}"
            )
        if floor < 0.0 or floor > 1.0:
            raise ValueError(f"floor must be in [0, 1], got {floor}")
        if max_variance_ratio <= 0.0:
            raise ValueError(
                f"max_variance_ratio must be > 0, got {max_variance_ratio}"
            )
        self.superiority_margin = float(superiority_margin)
        self.floor = float(floor)
        self.max_variance_ratio = float(max_variance_ratio)

    def evaluate(
        self,
        old_accuracy: float,
        new_accuracy: float,
        old_weights: Optional[np.ndarray],
        new_weights: np.ndarray,
    ) -> GateVerdict:
        """Evaluate the candidate weights with deterministic checks."""
        superiority_delta = float(new_accuracy - old_accuracy)
        superiority_pass = superiority_delta >= self.superiority_margin
        floor_pass = float(new_accuracy) >= self.floor
        conservation_pass = True
        var_ratio = self._compute_var_ratio(old_weights, new_weights)
        variance_pass = var_ratio <= self.max_variance_ratio

        failures: List[str] = []
        if not superiority_pass:
            failures.append("superiority_fail")
        if not floor_pass:
            failures.append("floor_fail")
        if not conservation_pass:
            failures.append("conservation_fail")
        if not variance_pass:
            failures.append("variance_fail")

        promoted = not failures
        placeholder_reason = "placeholder_always_pass"
        if promoted:
            reason = f"pass; {placeholder_reason}"
        else:
            reason = "; ".join([*failures, placeholder_reason])
        return GateVerdict(
            promoted=promoted,
            superiority_delta=superiority_delta,
            old_accuracy=float(old_accuracy),
            new_accuracy=float(new_accuracy),
            superiority_pass=superiority_pass,
            floor_pass=floor_pass,
            conservation_pass=conservation_pass,
            variance_pass=variance_pass,
            var_ratio=float(var_ratio),
            reason=reason,
        )

    def _compute_var_ratio(
        self,
        old_weights: Optional[np.ndarray],
        new_weights: np.ndarray,
    ) -> float:
        if old_weights is None:
            return 1.0
        old_var = float(np.var(old_weights))
        new_var = float(np.var(new_weights))
        if old_var <= 0.0:
            return 1.0 if new_var <= 0.0 else float("inf")
        return new_var / old_var


@dataclass
class BatchRecord:
    """Stored record for one batch evaluation attempt."""

    category_index: int
    attempted_at: datetime
    old_accuracy: float
    new_accuracy: float
    promoted: bool
    reason: str
    old_weights_hash: Optional[str]
    new_weights_hash: str
    verdict: GateVerdict


@dataclass
class BatchHistory:
    """Bounded history of batch evaluation attempts."""

    max_records: int = 100
    _records: List[BatchRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_records < 1:
            raise ValueError(f"max_records must be >= 1, got {self.max_records}")

    def record(
        self,
        category_index: int,
        old_accuracy: float,
        new_accuracy: float,
        old_weights: Optional[np.ndarray],
        new_weights: np.ndarray,
        verdict: GateVerdict,
        attempted_at: Optional[datetime] = None,
    ) -> BatchRecord:
        """Store a new batch attempt and trim to the configured bound."""
        record = BatchRecord(
            category_index=int(category_index),
            attempted_at=datetime.utcnow() if attempted_at is None else attempted_at,
            old_accuracy=float(old_accuracy),
            new_accuracy=float(new_accuracy),
            promoted=bool(verdict.promoted),
            reason=verdict.reason,
            old_weights_hash=None if old_weights is None else self._hash(old_weights),
            new_weights_hash=self._hash(new_weights),
            verdict=verdict,
        )
        self._records.append(record)
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records :]
        return record

    def get_records(
        self,
        category_index: Optional[int] = None,
        promoted_only: bool = False,
    ) -> List[BatchRecord]:
        """Return stored records filtered by category and promotion status."""
        records = self._records
        if category_index is not None:
            records = [record for record in records if record.category_index == category_index]
        if promoted_only:
            records = [record for record in records if record.promoted]
        return list(records)

    def total_promotions(self) -> int:
        """Return the number of promoted records."""
        return sum(1 for record in self._records if record.promoted)

    def total_attempts(self) -> int:
        """Return the total number of stored records."""
        return len(self._records)

    def _hash(self, arr: np.ndarray) -> str:
        import hashlib

        digest = hashlib.sha256(arr.tobytes()).hexdigest()
        return digest[:16]
