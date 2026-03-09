"""
GAE Judgment — institutional judgment layer for ProfileScorer decisions.

Translates a raw ScoringResult into a human-readable decision rationale:
explains WHY an action was recommended, which factors dominated, and
how confident the system is.

Zero SOC knowledge. NumPy only.

Reference: docs/gae_design_v8_3.md §18 (judgment framework); GAE-JUDG-1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# Confidence tier thresholds (validated: V3B ECE=0.036 at τ=0.1)
CONFIDENCE_HIGH   = 0.80
CONFIDENCE_MEDIUM = 0.50
# Below CONFIDENCE_MEDIUM -> "discovery" tier


@dataclass
class JudgmentResult:
    """
    Human-readable rationale for a ProfileScorer decision.

    Reference: docs/gae_design_v8_3.md §18.1; GAE-JUDG-1.

    Attributes
    ----------
    action : str
        Recommended action name.
    confidence : float
        Confidence score in [0, 1]. Rounded to 4 decimal places.
    confidence_tier : str
        "high" (≥ 0.80) / "medium" (≥ 0.50) / "discovery" (< 0.50).
    dominant_factors : list[str]
        Factor names ranked by contribution, highest first (top 3 max).
    factor_contributions : dict[str, float]
        factor_name -> contribution score in [0, 1].
        Higher = factor value was closer to the winning centroid.
    rationale : str
        One-sentence plain-English explanation of the decision.
    action_scores : dict[str, float]
        action_name -> raw probability for all actions (transparency).
    auto_approvable : bool
        True if confidence >= CONFIDENCE_HIGH and action != "escalate".
    """

    action: str
    confidence: float
    confidence_tier: str
    dominant_factors: list[str]
    factor_contributions: dict[str, float]
    rationale: str
    action_scores: dict[str, float]
    auto_approvable: bool


def _confidence_tier(confidence: float) -> str:
    """
    Map a confidence float to a tier string.

    Tiers:
      "high"      confidence >= CONFIDENCE_HIGH (0.80)
      "medium"    confidence >= CONFIDENCE_MEDIUM (0.50)
      "discovery" confidence <  CONFIDENCE_MEDIUM (0.50)

    Reference: docs/gae_design_v8_3.md §18.2; GAE-JUDG-1.
    """
    if confidence >= CONFIDENCE_HIGH:
        return "high"
    elif confidence >= CONFIDENCE_MEDIUM:
        return "medium"
    else:
        return "discovery"


def _dominant_factors(
    f: np.ndarray,
    mu_row: np.ndarray,
    factor_names: list[str],
    top_n: int = 3,
) -> tuple[list[str], dict[str, float]]:
    """
    Identify which factors contributed most to the scoring decision.

    Contribution for factor i:
      contribution[i] = clip(1.0 - |f[i] - mu_row[i]|, 0.0, 1.0)

    Factors closest to the winning centroid score highest.
    Returns at most top_n dominant factor names, ranked descending.

    Args:
      f:            Factor vector, shape (n_factors,).
      mu_row:       Winning centroid row mu[category, action, :],
                    shape (n_factors,).
      factor_names: Names for each factor dimension.
      top_n:        Maximum number of dominant factors to return.

    Returns:
      (dominant_factor_names, factor_contributions_dict)

    Reference: docs/gae_design_v8_3.md §18.3; GAE-JUDG-1.
    """
    assert f.ndim == 1, f"f must be 1-D, got {f.shape}"
    assert mu_row.ndim == 1, f"mu_row must be 1-D, got {mu_row.shape}"
    assert len(f) == len(mu_row), (
        f"f length {len(f)} != mu_row length {len(mu_row)}"
    )

    n = min(len(f), len(factor_names))
    diff = np.abs(f[:n] - mu_row[:n])           # shape (n,)
    assert diff.shape == (n,), f"diff.shape={diff.shape} != ({n},)"

    raw_contribs = np.clip(1.0 - diff, 0.0, 1.0)  # shape (n,)
    assert raw_contribs.shape == (n,), (
        f"raw_contribs.shape={raw_contribs.shape} != ({n},)"
    )

    contributions: dict[str, float] = {
        factor_names[i]: round(float(raw_contribs[i]), 4)
        for i in range(n)
    }

    dominant = sorted(
        contributions, key=contributions.__getitem__, reverse=True
    )
    return dominant[:top_n], contributions


def compute_judgment(
    scoring_result,
    f: np.ndarray,
    mu: np.ndarray,
    category_index: int,
    factor_names: list[str],
    actions: Optional[list[str]] = None,
) -> JudgmentResult:
    """
    Produce a JudgmentResult from a ScoringResult.

    Reads scoring_result.probabilities (shape n_actions) for action scores.
    Uses scoring_result.action_index to look up the winning centroid in mu.

    Args:
      scoring_result:  ScoringResult from ProfileScorer.score().
      f:               Factor vector used for scoring, shape (n_factors,).
      mu:              Full centroid array,
                       shape (n_categories, n_actions, n_factors).
      category_index:  Category index used for scoring.
      factor_names:    Names for each factor dimension.
      actions:         Action names list. If None, uses "action_0", "action_1",
                       ... as fallback keys in action_scores.

    Returns:
      JudgmentResult with rationale, dominant factors, and transparency scores.

    Reference: docs/gae_design_v8_3.md §18.4; GAE-JUDG-1.
    """
    assert f.ndim == 1, f"f must be 1-D, got {f.shape}"
    assert mu.ndim == 3, f"mu must be 3-D (n_cat, n_act, n_fac), got {mu.shape}"
    assert 0 <= category_index < mu.shape[0], (
        f"category_index {category_index} out of range [0, {mu.shape[0]})"
    )

    action_name = scoring_result.action_name
    action_idx  = scoring_result.action_index
    confidence  = scoring_result.confidence
    # ScoringResult uses .probabilities (not .scores)
    probs = scoring_result.probabilities   # shape (n_actions,)

    assert probs.ndim == 1, f"probabilities must be 1-D, got {probs.shape}"
    assert 0.0 <= confidence <= 1.0, (
        f"confidence {confidence} out of [0, 1]"
    )

    tier = _confidence_tier(confidence)

    # Winning centroid row for dominant-factor calculation
    mu_row = mu[category_index, action_idx, :]   # shape (n_factors,)
    assert mu_row.ndim == 1, f"mu_row must be 1-D, got {mu_row.shape}"

    dominant, contributions = _dominant_factors(f, mu_row, factor_names)

    # Build action_scores dict
    n_actions = len(probs)
    if actions is not None:
        assert len(actions) == n_actions, (
            f"len(actions)={len(actions)} != len(probabilities)={n_actions}"
        )
        action_scores: dict[str, float] = {
            actions[i]: round(float(probs[i]), 4)
            for i in range(n_actions)
        }
    else:
        action_scores = {
            f"action_{i}": round(float(probs[i]), 4)
            for i in range(n_actions)
        }

    # auto_approvable: high confidence, non-escalation action
    auto_approvable = (tier == "high" and action_name != "escalate")

    # Plain-English rationale
    top_factor = dominant[0] if dominant else "unknown"
    rationale = (
        f"Recommended '{action_name}' with {tier} confidence "
        f"({confidence:.0%}). "
        f"Primary driver: {top_factor}."
    )

    return JudgmentResult(
        action=action_name,
        confidence=round(confidence, 4),
        confidence_tier=tier,
        dominant_factors=dominant,
        factor_contributions=contributions,
        rationale=rationale,
        action_scores=action_scores,
        auto_approvable=auto_approvable,
    )
