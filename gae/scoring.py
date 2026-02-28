"""
GAE Scoring — Tier 2: Scoring Matrix.

Implements Eq. 4 from the math blog:

    P(action | alert) = softmax( f · Wᵀ / τ )              [Eq. 4]

Shape annotation (from math blog §3):
    f      : (1, n_f)      factor vector (Tier 1 output)
    W      : (n_a, n_f)    weight matrix (Tier 3 state)
    Wᵀ     : (n_f, n_a)
    f · Wᵀ : (1, n_a)      raw scores
    probs  : (1, n_a)      softmax probabilities over actions

τ (temperature) controls sharpness:
    small τ → hard argmax;  τ = 1.0 → uniform mixing

Reference: docs/gae_design_v5.md §7; blog Eq. 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gae.primitives import softmax


# ---------------------------------------------------------------------------
# ScoringResult
# ---------------------------------------------------------------------------

@dataclass
class ScoringResult:
    """
    Full output of one Eq. 4 scoring call.

    Reference: docs/gae_design_v5.md §7.2.

    Attributes
    ----------
    action_probabilities : np.ndarray, shape (1, n_a)
        Softmax distribution over actions. Rows sum to 1.0.
    selected_action : str
        Name of the highest-probability action (argmax).
    confidence : float
        Probability of *selected_action* ∈ (0, 1].
    raw_scores : np.ndarray, shape (1, n_a)
        f · Wᵀ / τ — logits before softmax.
    factor_vector : np.ndarray, shape (1, n_f)
        Original factor vector f, preserved per Requirement R4.
        Must not be modified by the caller after scoring.
    temperature : float
        τ value used in this call.
    """

    action_probabilities: np.ndarray  # (1, n_a)
    selected_action: str
    confidence: float
    raw_scores: np.ndarray            # (1, n_a)
    factor_vector: np.ndarray         # (1, n_f)  — R4 preservation
    temperature: float


# ---------------------------------------------------------------------------
# score_alert
# ---------------------------------------------------------------------------

def score_alert(
    f: np.ndarray,
    W: np.ndarray,
    actions: list[str],
    tau: float = 0.25,
) -> ScoringResult:
    """
    Tier 2 scoring matrix — Eq. 4 from the math blog.

    Computes:
        raw   = f · Wᵀ / τ                                 [Eq. 4, numerator]
        probs = softmax(raw)                                [Eq. 4, softmax]

    Reference: docs/gae_design_v5.md §7.1; blog Eq. 4.

    Parameters
    ----------
    f : np.ndarray, shape (1, n_f)
        Factor vector from Tier 1 (assemble_factor_vector).
    W : np.ndarray, shape (n_a, n_f)
        Weight matrix; rows = actions, columns = factors.
    actions : list[str]
        Action names in the same row order as W.
    tau : float, default 0.25
        Temperature scalar τ > 0.
        Smaller τ sharpens the distribution toward a hard argmax.

    Returns
    -------
    ScoringResult
        Full scoring result.

    Raises
    ------
    ValueError
        If tau <= 0.
    AssertionError
        On shape mismatches or empty action list.
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")

    assert isinstance(f, np.ndarray), f"f must be np.ndarray, got {type(f)}"
    assert isinstance(W, np.ndarray), f"W must be np.ndarray, got {type(W)}"
    assert f.ndim == 2 and f.shape[0] == 1, (
        f"f must be shape (1, n_f), got {f.shape}"
    )
    assert W.ndim == 2, f"W must be 2-D, got shape {W.shape}"

    n_f = f.shape[1]
    n_a, n_f_w = W.shape

    assert n_f == n_f_w, (
        f"f n_factors ({n_f}) must equal W n_factors ({n_f_w}); "
        f"f.shape={f.shape}, W.shape={W.shape}"
    )
    assert len(actions) == n_a, (
        f"len(actions) ({len(actions)}) must equal W n_actions ({n_a})"
    )
    assert n_a > 0, "actions list must be non-empty"

    # Eq. 4 — raw scores: (1, n_f) @ (n_f, n_a) → (1, n_a)
    raw_scores = (f @ W.T) / tau
    assert raw_scores.shape == (1, n_a), (
        f"raw_scores shape {raw_scores.shape} != expected (1, {n_a})"
    )

    # Eq. 4 — softmax over the single row → (1, n_a)
    probs = softmax(raw_scores, axis=-1)
    assert probs.shape == (1, n_a), (
        f"probs shape {probs.shape} != expected (1, {n_a})"
    )
    assert abs(float(probs.sum()) - 1.0) < 1e-6, (
        f"Probabilities must sum to 1.0, got {probs.sum()}"
    )

    selected_idx = int(np.argmax(probs.flatten()))
    return ScoringResult(
        action_probabilities=probs,
        selected_action=actions[selected_idx],
        confidence=float(probs.flatten()[selected_idx]),
        raw_scores=raw_scores,
        factor_vector=f,
        temperature=tau,
    )
