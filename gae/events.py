"""
GAE event dataclasses — plain value objects emitted after key computations.

These are *data containers only*.  No event bus.  No async dispatch.
Callers receive them as return values or can pass them to logging/monitoring
hooks they own.

Reference: docs/gae_design_v10_6.md §3 (Event model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass(frozen=True)
class FactorComputedEvent:
    """
    Emitted when a single factor vector has been assembled for one node.

    Reference: docs/gae_design_v10_6.md §3.1; blog factor assembly section.

    Attributes
    ----------
    node_id : str
        Opaque identifier for the node whose factor was computed.
    factor_vector : np.ndarray, shape (d_f,)
        The assembled factor vector.  Caller must not mutate.
    factor_names : tuple[str, ...]
        Ordered names of the scalar factors packed into *factor_vector*.
    """

    node_id: str
    factor_vector: np.ndarray
    factor_names: tuple[str, ...]

    def __post_init__(self) -> None:
        assert isinstance(self.factor_vector, np.ndarray), (
            "FactorComputedEvent.factor_vector must be np.ndarray"
        )
        assert self.factor_vector.ndim == 1, (
            f"factor_vector must be 1-D, got shape {self.factor_vector.shape}"
        )
        assert len(self.factor_names) == self.factor_vector.shape[0], (
            f"factor_names length {len(self.factor_names)} != "
            f"factor_vector length {self.factor_vector.shape[0]}"
        )


@dataclass(frozen=True)
class WeightsUpdatedEvent:
    """
    Emitted after the learning rule updates factor weights.

    Reference: docs/gae_design_v10_6.md §3.2; blog Eq. 4b/4c weight update.

    Attributes
    ----------
    weights_before : np.ndarray, shape (d_f,)
        Weight vector before the update step.
    weights_after : np.ndarray, shape (d_f,)
        Weight vector after the update step.
    delta_norm : float
        L2 norm of (weights_after - weights_before).
    step : int
        Training step index at which the update occurred.
    """

    weights_before: np.ndarray
    weights_after: np.ndarray
    delta_norm: float
    step: int

    def __post_init__(self) -> None:
        assert isinstance(self.weights_before, np.ndarray), (
            "WeightsUpdatedEvent.weights_before must be np.ndarray"
        )
        assert isinstance(self.weights_after, np.ndarray), (
            "WeightsUpdatedEvent.weights_after must be np.ndarray"
        )
        assert self.weights_before.ndim == 1, (
            f"weights_before must be 1-D, got shape {self.weights_before.shape}"
        )
        assert self.weights_after.shape == self.weights_before.shape, (
            f"weights shape mismatch: before={self.weights_before.shape}, "
            f"after={self.weights_after.shape}"
        )
        assert self.step >= 0, f"step must be non-negative, got {self.step}"


@dataclass(frozen=True)
class ConvergenceEvent:
    """
    Emitted when the convergence monitor detects a state change.

    Reference: docs/gae_design_v10_6.md §3.3; blog convergence criterion.

    Attributes
    ----------
    step : int
        Training step at which convergence state changed.
    converged : bool
        True if the learning process has converged.
    delta_norm : float
        Weight delta norm that triggered the event.
    threshold : float
        Convergence threshold that was compared against *delta_norm*.
    """

    step: int
    converged: bool
    delta_norm: float
    threshold: float

    def __post_init__(self) -> None:
        assert self.step >= 0, f"step must be non-negative, got {self.step}"
        assert self.threshold > 0.0, (
            f"threshold must be positive, got {self.threshold}"
        )
