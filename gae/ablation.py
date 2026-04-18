"""
GAE Ablation — factor importance measurement via leave-one-out ablation.

Measures how much each factor contributes to accuracy by zeroing out one
factor at a time and re-evaluating. A large accuracy drop when factor X
is zeroed means X is important.

Zero SOC knowledge. NumPy only.
Builds on gae/evaluation.py (run_evaluation, EvaluationScenario).

Reference: docs/gae_design_v10_6.md §20 (ablation framework); GAE-ABL-1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gae.evaluation import EvaluationReport, EvaluationScenario, run_evaluation


@dataclass
class AblationResult:
    """
    Result of ablating one factor from the evaluation.

    Reference: docs/gae_design_v10_6.md §20.1; GAE-ABL-1.

    Attributes
    ----------
    factor_index : int
        Index of the ablated factor.
    factor_name : str
        Name of the ablated factor.
    baseline_accuracy : float
        Accuracy with all factors present.
    ablated_accuracy : float
        Accuracy with this factor zeroed out.
    accuracy_drop : float
        baseline_accuracy - ablated_accuracy.
        Positive = factor helps. Negative = factor hurts (noise).
    importance_rank : int
        Rank by accuracy_drop (1 = most important).
        Set after run_ablation() completes.
    """

    factor_index: int
    factor_name: str
    baseline_accuracy: float
    ablated_accuracy: float
    accuracy_drop: float
    importance_rank: int = 0


@dataclass
class AblationReport:
    """
    Full ablation study results across all factors.

    Reference: docs/gae_design_v10_6.md §20.2; GAE-ABL-1.

    Attributes
    ----------
    baseline_accuracy : float
        Accuracy with all factors present.
    results : list[AblationResult]
        AblationResult per factor, sorted by accuracy_drop descending.
    most_important : str
        Factor name with largest accuracy drop.
    least_important : str
        Factor name with smallest accuracy drop (or most negative — noise).
    n_factors : int
        Number of factors ablated.
    n_scenarios : int
        Number of evaluation scenarios used.
    """

    baseline_accuracy: float
    results: list[AblationResult]
    most_important: str
    least_important: str
    n_factors: int
    n_scenarios: int


def _zero_factor(
    scenarios: list[EvaluationScenario],
    factor_index: int,
) -> list[EvaluationScenario]:
    """
    Return a copy of scenarios with factor_index zeroed out.

    Does NOT modify original scenarios. Creates new EvaluationScenario
    objects with copied factor arrays.

    Args:
      scenarios:     Original scenario list. Unchanged.
      factor_index:  Index into each scenario's factors vector to zero.

    Returns:
      New list of EvaluationScenario with factor_index set to 0.0.

    Reference: docs/gae_design_v10_6.md §20.3; GAE-ABL-1.
    """
    ablated = []
    for s in scenarios:
        f_copy = s.factors.copy()
        f_copy[factor_index] = 0.0
        ablated.append(EvaluationScenario(
            scenario_id=s.scenario_id,
            domain=s.domain,
            category=s.category,
            category_index=s.category_index,
            factors=f_copy,
            expected_action=s.expected_action,
            expected_action_index=s.expected_action_index,
            expected_dominant_factors=s.expected_dominant_factors,
            confidence_tier=s.confidence_tier,
            description=s.description,
            learning_prerequisite=s.learning_prerequisite,
        ))
    return ablated


def run_ablation(
    profile_scorer,
    scenarios: list[EvaluationScenario],
    factor_names: list[str],
) -> AblationReport:
    """
    Run ablation study: evaluate accuracy with each factor zeroed.

    Steps:
      1. Compute baseline accuracy (all factors present).
      2. For each factor i in range(n_factors):
         a. Zero out factor i across all scenarios (_zero_factor).
         b. Run evaluation on ablated scenarios.
         c. Record accuracy_drop = baseline - ablated_accuracy.
      3. Sort results by accuracy_drop descending.
      4. Assign importance_rank (1 = most important).

    Args:
      profile_scorer: ProfileScorer instance. NOT modified.
      scenarios:      List of EvaluationScenario. NOT modified.
      factor_names:   Names for each factor dimension.
                      len(factor_names) must match factors vector length.

    Returns:
      AblationReport sorted by importance (most important first).

    Raises:
      ValueError: if scenarios is empty or factor_names is empty.
      ValueError: if len(factor_names) != factors vector length.

    Reference: docs/gae_design_v10_6.md §20.4; GAE-ABL-1.
    """
    if not scenarios:
        raise ValueError("scenarios must not be empty")
    if not factor_names:
        raise ValueError("factor_names must not be empty")

    n_factors = len(scenarios[0].factors)
    if len(factor_names) != n_factors:
        raise ValueError(
            f"factor_names length {len(factor_names)} != "
            f"factors length {n_factors}"
        )

    # Step 1: baseline with all factors
    baseline_report = run_evaluation(profile_scorer, scenarios)
    baseline_accuracy = baseline_report.accuracy

    # Step 2: ablate each factor
    results: list[AblationResult] = []
    for i, name in enumerate(factor_names):
        ablated_scenarios = _zero_factor(scenarios, i)
        ablated_report = run_evaluation(profile_scorer, ablated_scenarios)
        drop = round(baseline_accuracy - ablated_report.accuracy, 4)
        results.append(AblationResult(
            factor_index=i,
            factor_name=name,
            baseline_accuracy=baseline_accuracy,
            ablated_accuracy=ablated_report.accuracy,
            accuracy_drop=drop,
        ))

    # Step 3: sort by accuracy_drop descending, assign ranks
    results.sort(key=lambda r: r.accuracy_drop, reverse=True)
    for rank, result in enumerate(results, start=1):
        result.importance_rank = rank

    most_important = results[0].factor_name
    least_important = results[-1].factor_name

    return AblationReport(
        baseline_accuracy=baseline_accuracy,
        results=results,
        most_important=most_important,
        least_important=least_important,
        n_factors=n_factors,
        n_scenarios=len(scenarios),
    )
