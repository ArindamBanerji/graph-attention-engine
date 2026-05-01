"""
James-Stein-style shrinkage helpers for interpolating learned weights.

This module defines simple shrinkage schedules that interpolate between
per-dimension learned weights and a neutral all-ones baseline. The effective
weights follow the alpha interpolation:

    w_eff = alpha * w_dk + (1 - alpha)

where alpha in [0, 1] controls how strongly the learned weights influence the
final result.
"""

from __future__ import annotations

import numpy as np
from typing import Protocol

from gae.two_phase import CategoryState


class ShrinkageSchedule(Protocol):
    """Protocol for schedules that produce shrinkage interpolation weights."""

    def compute_alpha(self, state: CategoryState) -> float:
        """Return the interpolation weight alpha for the given category state."""
        ...


class FixedAlpha:
    """Return the same alpha for every category state."""

    def __init__(self, alpha: float = 0.5) -> None:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = float(alpha)

    def compute_alpha(self, state: CategoryState) -> float:
        """Return the configured constant alpha."""
        return self.alpha


class LinearRampAlpha:
    """Linearly ramp alpha upward after the category freeze point."""

    def __init__(
        self,
        start: float = 0.1,
        end: float = 0.5,
        ramp_decisions: int = 1000,
    ) -> None:
        if start < 0.0 or start > 1.0:
            raise ValueError(f"start must be in [0, 1], got {start}")
        if end < 0.0 or end > 1.0:
            raise ValueError(f"end must be in [0, 1], got {end}")
        if ramp_decisions <= 0:
            raise ValueError(
                f"ramp_decisions must be > 0, got {ramp_decisions}"
            )
        self.start = float(start)
        self.end = float(end)
        self.ramp_decisions = int(ramp_decisions)

    def compute_alpha(self, state: CategoryState) -> float:
        """Return a linearly ramped alpha based on decisions since freeze."""
        if state.freeze_point is None:
            return self.start

        decisions_since_freeze = state.n_decisions - state.freeze_point
        if decisions_since_freeze <= 0:
            return self.start

        progress = min(decisions_since_freeze / self.ramp_decisions, 1.0)
        return self.start + progress * (self.end - self.start)


def compute_effective_weights(w_dk: np.ndarray, alpha: float) -> np.ndarray:
    """Blend learned weights with a neutral all-ones baseline."""
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return alpha * w_dk + (1.0 - alpha)
