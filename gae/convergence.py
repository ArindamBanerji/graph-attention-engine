"""
GAE Convergence — monitoring weight learning stability and accuracy.

Provides a standalone function that inspects a LearningState and returns
diagnostic metrics used to determine whether training has converged.

Convergence criterion (docs/gae_design_v5.md §8.3):
    converged := stability < STABILITY_THRESHOLD AND accuracy > ACCURACY_THRESHOLD

Where:
    stability := std(||W_after||₂ over the last 10 history entries)
    accuracy  := fraction of recent outcomes == +1

Three failure modes detected via the returned metrics:
    FM1 — Action confusion  : weight_norm is low → actions score similarly
    FM2 — Asymmetric oscillation : alternating correct/incorrect → accuracy ≈ 0.5
    FM3 — Decay competition  : stability is high → W norm never stabilises

Reference: docs/gae_design_v5.md §8.3; blog convergence criterion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gae.learning import LearningState


# ---------------------------------------------------------------------------
# Convergence thresholds
# ---------------------------------------------------------------------------

STABILITY_THRESHOLD: float = 0.05   # std of recent W norms below this → stable
ACCURACY_THRESHOLD: float = 0.80    # recent accuracy above this → good
RECENCY_WINDOW: int = 20            # decisions used for accuracy estimate
STABILITY_WINDOW: int = 10          # decisions used for stability estimate


# ---------------------------------------------------------------------------
# get_convergence_metrics
# ---------------------------------------------------------------------------

def get_convergence_metrics(state: "LearningState") -> dict:
    """
    Compute convergence and health diagnostics for a LearningState.

    Stability is measured as the standard deviation of the Frobenius norm
    of W_after over the last STABILITY_WINDOW (10) history entries.
    A small std means the weight matrix has stopped changing significantly.

    Accuracy is the fraction of outcome == +1 among the last RECENCY_WINDOW
    (20) history entries.

    Reference: docs/gae_design_v5.md §8.3; blog convergence criterion.

    Parameters
    ----------
    state : LearningState
        The active Tier 3 learning state.

    Returns
    -------
    dict with keys:
        decisions           int   — total completed updates
        weight_norm         float — Frobenius norm of current W
        stability           float — std of W norms over last 10 updates
                                    (0.0 when fewer than 2 history entries)
        accuracy            float — fraction of correct in last 20 updates
                                    (0.0 when history is empty)
        converged           bool  — stability < threshold AND accuracy > threshold
        provisional_dimensions int — count of A4 provisional W columns
        pending_autonomous  int   — count of C3 deferred validations
    """
    weight_norm = float(np.linalg.norm(state.W))

    if not state.history:
        return {
            "decisions": state.decision_count,
            "weight_norm": weight_norm,
            "stability": 0.0,
            "accuracy": 0.0,
            "converged": False,
            "provisional_dimensions": sum(
                1 for dm in state.dimension_metadata if dm.state == "provisional"
            ),
            "pending_autonomous": len(state.pending_validations),
        }

    # Stability: std of ||W_after||_F over last STABILITY_WINDOW entries
    recent_history = state.history[-STABILITY_WINDOW:]
    norms = [float(np.linalg.norm(h.W_after)) for h in recent_history]
    stability = float(np.std(norms)) if len(norms) >= 2 else 0.0

    # Accuracy: fraction correct over last RECENCY_WINDOW entries
    outcome_window = state.history[-RECENCY_WINDOW:]
    accuracy = (
        sum(1 for h in outcome_window if h.outcome == +1) / len(outcome_window)
    )

    converged = stability < STABILITY_THRESHOLD and accuracy > ACCURACY_THRESHOLD

    return {
        "decisions": state.decision_count,
        "weight_norm": weight_norm,
        "stability": stability,
        "accuracy": accuracy,
        "converged": converged,
        "provisional_dimensions": sum(
            1 for dm in state.dimension_metadata if dm.state == "provisional"
        ),
        "pending_autonomous": len(state.pending_validations),
    }
