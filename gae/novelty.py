"""
Novelty tracking for category-conditioned factor vectors.

Purpose
-------
This module measures how different a new factor vector is from recent history
within one category using nearest-neighbor distance, while also tracking
novelty rate and an accumulator that can drive higher-level policies.

Failure modes
-------------
1. No prior history in a category.
   Handled: `compute_novelty()` returns 1.0 and `record()` treats the first
   observation as novel.
2. Exact or near-duplicate observations.
   Handled: nearest-neighbor distance goes to 0.0 for identical vectors once
   a prior copy has been recorded.
3. Slow concept drift where only recent history matters.
   Partially handled: history is trimmed to a bounded recent window
   (`max_look`), but no explicit time-decay weighting is applied.
4. Novelty caused by scale mismatch or representation shift.
   Not handled: distances are raw Euclidean distances on the provided factor
   vectors, so callers must ensure a compatible representation.
"""

import numpy as np
from typing import Protocol, Optional, List, Tuple, Dict


class NoveltyTracker(Protocol):
    def compute_novelty(self, f: np.ndarray, category_index: int) -> float:
        """Return a novelty score for one factor vector in one category."""

    def record(self, f: np.ndarray, category_index: int) -> None:
        """Record one factor vector and update novelty diagnostics."""

    def get_novelty_rate(self, category_index: int, window: int = 50) -> float:
        """Return the recent fraction of recorded vectors marked novel."""

    def get_accumulator(self, category_index: int) -> float:
        """Return the accumulated novelty score since the last reset."""

    def reset_accumulator(self, category_index: int) -> None:
        """Reset the novelty accumulator for one category."""


class NearestNeighborNovelty:
    """Bounded nearest-neighbor novelty tracker."""

    def __init__(
        self,
        max_look: int = 300,
        threshold: float = 0.1,
        n_categories: Optional[int] = None,
    ) -> None:
        if max_look < 1:
            raise ValueError(f"max_look must be >= 1, got {max_look}")
        if threshold < 0.0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        if n_categories is not None and n_categories < 1:
            raise ValueError(f"n_categories must be >= 1, got {n_categories}")

        self.max_look = max_look
        self.threshold = float(threshold)
        self.n_categories = n_categories
        self._history: Dict[int, List[np.ndarray]] = {}
        self._novelty_scores: Dict[int, List[Tuple[float, bool]]] = {}
        self._accumulator: Dict[int, float] = {}

        if n_categories is not None:
            for category_index in range(n_categories):
                self._history[category_index] = []
                self._novelty_scores[category_index] = []
                self._accumulator[category_index] = 0.0

    def compute_novelty(self, f: np.ndarray, category_index: int) -> float:
        """Compute nearest-neighbor novelty without mutating tracker state."""
        self._ensure_category(category_index)
        vector = np.asarray(f, dtype=np.float64)
        history = self._history[category_index]
        if not history:
            return 1.0

        distances = [
            float(np.linalg.norm(vector - prior))
            for prior in history
        ]
        return float(min(distances))

    def record(self, f: np.ndarray, category_index: int) -> None:
        """Record one factor vector using a defensive copy."""
        self._ensure_category(category_index)
        novelty_score = self.compute_novelty(f, category_index)
        is_novel = novelty_score > self.threshold

        self._novelty_scores[category_index].append((novelty_score, is_novel))
        if len(self._novelty_scores[category_index]) > self.max_look:
            self._novelty_scores[category_index] = self._novelty_scores[category_index][-self.max_look :]

        self._accumulator[category_index] += novelty_score

        vector_copy = np.asarray(f, dtype=np.float64).copy()
        self._history[category_index].append(vector_copy)
        if len(self._history[category_index]) > self.max_look:
            self._history[category_index] = self._history[category_index][-self.max_look :]

    def get_novelty_rate(self, category_index: int, window: int = 50) -> float:
        """Return the fraction of recent recorded vectors marked as novel."""
        self._ensure_category(category_index)
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")

        recent = self._novelty_scores[category_index][-window:]
        if not recent:
            return 0.0
        return float(sum(1 for _, is_novel in recent if is_novel) / len(recent))

    def get_accumulator(self, category_index: int) -> float:
        """Return accumulated novelty since the last reset."""
        self._ensure_category(category_index)
        return float(self._accumulator[category_index])

    def reset_accumulator(self, category_index: int) -> None:
        """Reset the accumulator for one category."""
        self._ensure_category(category_index)
        self._accumulator[category_index] = 0.0

    def get_history_size(self, category_index: int) -> int:
        """Return the number of retained historical vectors for a category."""
        self._ensure_category(category_index)
        return len(self._history[category_index])

    def _ensure_category(self, category_index: int) -> None:
        if category_index < 0:
            raise ValueError(f"category_index must be >= 0, got {category_index}")
        if self.n_categories is not None and category_index >= self.n_categories:
            raise ValueError(
                f"category_index {category_index} out of range [0, {self.n_categories})"
            )
        if category_index not in self._history:
            self._history[category_index] = []
            self._novelty_scores[category_index] = []
            self._accumulator[category_index] = 0.0
