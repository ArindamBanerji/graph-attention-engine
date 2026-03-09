"""
GAE Evaluation — ground-truth evaluation framework for ProfileScorer.

Evaluates ProfileScorer against EvaluationScenarios: structured test cases
with known factor vectors and expected actions. Computes overall accuracy,
per-category accuracy, per-action precision/recall, and ECE.

Zero SOC knowledge. NumPy only.

Reference: docs/gae_design_v8_3.md §17 (evaluation framework); GAE-EVAL-1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EvaluationScenario:
    """
    A structured test case with known factor vector and expected action.

    Reference: docs/gae_design_v8_3.md §17.1.

    Attributes
    ----------
    scenario_id : str
        Unique identifier for this scenario.
    domain : str
        Domain label (e.g. "fraud", "network"). No SOC semantics enforced.
    category : str
        Human-readable category name (matches category_index in ProfileScorer).
    category_index : int
        Integer index into ProfileScorer's category dimension.
    factors : np.ndarray
        Factor vector, shape (n_factors,). Values in [0, 1].
    expected_action : str
        Expected action name (ground truth when no oracle is provided).
    expected_action_index : int
        Expected action index (ground truth when no oracle is provided).
    expected_dominant_factors : list[str]
        Optional list of factor names expected to dominate scoring.
    confidence_tier : str
        Expected confidence tier: "high", "medium", or "low".
    description : str
        Human-readable scenario description.
    learning_prerequisite : Optional[str]
        Scenario ID that must run before this one (for sequential learning tests).
    """

    scenario_id: str
    domain: str
    category: str
    category_index: int
    factors: np.ndarray
    expected_action: str
    expected_action_index: int
    expected_dominant_factors: list[str] = field(default_factory=list)
    confidence_tier: str = "high"
    description: str = ""
    learning_prerequisite: Optional[str] = None


@dataclass
class EvaluationReport:
    """
    Aggregated results from run_evaluation().

    Reference: docs/gae_design_v8_3.md §17.2.

    Attributes
    ----------
    accuracy : float
        Overall fraction of correct predictions. Rounded to 4 decimal places.
    by_category : dict[str, float]
        Per-category accuracy keyed by category name.
    precision_per_action : dict[str, float]
        Per-action precision: TP / (TP + FP). 0.0 if no predictions made.
    recall_per_action : dict[str, float]
        Per-action recall: TP / (TP + FN). 0.0 if never the GT action.
    ece : float
        Expected Calibration Error. See compute_ece().
    scenario_results : list[dict]
        Per-scenario detail dicts with keys: scenario_id, category,
        predicted_action, expected_action, correct, confidence.
    n_scenarios : int
        Total number of scenarios evaluated.
    n_correct : int
        Total number of correct predictions.
    """

    accuracy: float
    by_category: dict[str, float]
    precision_per_action: dict[str, float]
    recall_per_action: dict[str, float]
    ece: float
    scenario_results: list[dict]
    n_scenarios: int
    n_correct: int


def compute_ece(
    confidences: list[float],
    correct_flags: list[bool],
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Bins predictions by confidence into n_bins equal-width buckets over [0, 1].
    ECE = Σ_{b} (|b| / n) * |mean_confidence(b) - mean_accuracy(b)|
    over all non-empty bins.

    Returns 0.0 if fewer than 2 scenarios.

    Args:
      confidences:   Predicted confidence values, each in [0, 1].
      correct_flags: Boolean correctness flags aligned with confidences.
      n_bins:        Number of calibration bins. Default 10.

    Returns:
      ECE as a float rounded to 6 decimal places.

    Reference: docs/gae_design_v8_3.md §17.3; blog Eq. 4-final calibration (V3B).
    """
    assert len(confidences) == len(correct_flags), (
        f"confidences length {len(confidences)} != correct_flags length "
        f"{len(correct_flags)}"
    )

    if len(confidences) < 2:
        return 0.0

    confidences_arr = np.array(confidences, dtype=np.float64)
    correct_arr = np.array(correct_flags, dtype=np.float64)
    assert confidences_arr.shape == correct_arr.shape, (
        f"confidences shape {confidences_arr.shape} != correct shape "
        f"{correct_arr.shape}"
    )

    n = len(confidences_arr)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences_arr >= lo) & (
            confidences_arr <= hi if i == n_bins - 1 else confidences_arr < hi
        )
        if mask.sum() == 0:
            continue
        bin_weight = mask.sum() / n
        ece += bin_weight * abs(confidences_arr[mask].mean() - correct_arr[mask].mean())

    return float(round(ece, 6))


def run_evaluation(
    profile_scorer,
    scenarios: list[EvaluationScenario],
    oracle=None,
    learn: bool = False,
) -> EvaluationReport:
    """
    Evaluate ProfileScorer against a list of EvaluationScenarios.

    For each scenario:
      1. Assert f is 1-D.
      2. Call profile_scorer.score(f, category_index).
      3. Assert confidence in [0, 1].
      4. Compare predicted vs expected (or oracle-provided) action.
      5. If oracle provided and learn=True: call profile_scorer.update().

    Precision and recall are tracked per-action across all scenarios:
      TP[a] += 1 when predicted == a and correct.
      FP[a] += 1 when predicted == a and incorrect.
      FN[a] += 1 when gt == a and incorrect.

    Args:
      profile_scorer: ProfileScorer instance with .score(), .update(), .actions.
      scenarios:      List of EvaluationScenario instances.
      oracle:         Optional OracleProvider. If None, uses
                      scenario.expected_action_index as ground truth.
      learn:          If True and oracle is provided, call profile_scorer.update()
                      after each scenario using the oracle's GT action index.

    Returns:
      EvaluationReport with accuracy, per-category, per-action, and ECE metrics.

    Reference: docs/gae_design_v8_3.md §17.4; GAE-EVAL-1.
    """
    if not scenarios:
        return EvaluationReport(
            accuracy=0.0,
            by_category={},
            precision_per_action={},
            recall_per_action={},
            ece=0.0,
            scenario_results=[],
            n_scenarios=0,
            n_correct=0,
        )

    actions = profile_scorer.actions
    tp: dict[str, int] = {a: 0 for a in actions}
    fp: dict[str, int] = {a: 0 for a in actions}
    fn: dict[str, int] = {a: 0 for a in actions}
    cat_total: dict[str, int] = {}
    cat_correct: dict[str, int] = {}
    confidences: list[float] = []
    correct_flags: list[bool] = []
    scenario_results: list[dict] = []

    for scenario in scenarios:
        f = np.asarray(scenario.factors, dtype=np.float64)
        assert f.ndim == 1, (
            f"scenario {scenario.scenario_id}: factors must be 1-D, got shape {f.shape}"
        )
        assert isinstance(scenario.category_index, int), (
            f"scenario {scenario.scenario_id}: category_index must be int"
        )

        result = profile_scorer.score(f, scenario.category_index)
        confidence = result.confidence
        assert 0.0 <= confidence <= 1.0, (
            f"scenario {scenario.scenario_id}: confidence {confidence} out of [0, 1]"
        )

        predicted_action = result.action_name
        predicted_idx = result.action_index

        if oracle is not None:
            oracle_result = oracle.query(f, scenario.category_index, predicted_idx)
            correct = oracle_result.correct
            gt_action = oracle_result.gt_action_name
            gt_idx = oracle_result.gt_action_idx
        else:
            gt_action = scenario.expected_action
            gt_idx = scenario.expected_action_index
            correct = (predicted_idx == gt_idx)

        if learn and oracle is not None:
            profile_scorer.update(
                f=f,
                category_index=scenario.category_index,
                action_index=gt_idx,
                correct=correct,
            )

        confidences.append(confidence)
        correct_flags.append(correct)

        cat = scenario.category
        cat_total[cat] = cat_total.get(cat, 0) + 1
        if correct:
            cat_correct[cat] = cat_correct.get(cat, 0) + 1

        # TP/FP/FN tallying
        if predicted_action in tp:
            if correct:
                tp[predicted_action] += 1
            else:
                fp[predicted_action] += 1
        if gt_action in fn and not correct:
            fn[gt_action] += 1

        scenario_results.append({
            "scenario_id": scenario.scenario_id,
            "category": scenario.category,
            "predicted_action": predicted_action,
            "expected_action": gt_action,
            "correct": correct,
            "confidence": round(confidence, 4),
        })

    n = len(scenarios)
    n_correct = sum(correct_flags)

    by_category = {
        cat: round(cat_correct.get(cat, 0) / cat_total[cat], 4)
        for cat in cat_total
    }

    precision_per_action: dict[str, float] = {}
    recall_per_action: dict[str, float] = {}
    for a in actions:
        denom_p = tp[a] + fp[a]
        denom_r = tp[a] + fn[a]
        precision_per_action[a] = round(tp[a] / denom_p, 4) if denom_p > 0 else 0.0
        recall_per_action[a] = round(tp[a] / denom_r, 4) if denom_r > 0 else 0.0

    return EvaluationReport(
        accuracy=round(n_correct / n, 4),
        by_category=by_category,
        precision_per_action=precision_per_action,
        recall_per_action=recall_per_action,
        ece=compute_ece(confidences, correct_flags),
        scenario_results=scenario_results,
        n_scenarios=n,
        n_correct=n_correct,
    )
