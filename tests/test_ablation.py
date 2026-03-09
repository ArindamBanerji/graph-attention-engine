"""Tests for gae.ablation — run_ablation, AblationReport, AblationResult, _zero_factor."""

import numpy as np
import pytest

from gae.ablation import AblationReport, AblationResult, run_ablation
from gae.evaluation import EvaluationScenario


def make_ablation_inputs():
    """
    Build a ProfileScorer and scenarios where factor 0 is the dominant signal.

    mu[0, 0, :] = [0.9, 0.1, 0.1, 0.1] -> escalate centroid
    mu[0, 1, :] = [0.1, 0.5, 0.5, 0.5] -> suppress centroid
    All scenarios have f=[0.9, 0.1, 0.1, 0.1] -> should pick escalate.
    Ablating factor 0 (0.9->0.0) should cause the largest accuracy drop.
    """
    from gae.profile_scorer import ProfileScorer

    mu = np.zeros((1, 2, 4))
    mu[0, 0, :] = [0.9, 0.1, 0.1, 0.1]  # action0 centroid
    mu[0, 1, :] = [0.1, 0.5, 0.5, 0.5]  # action1 centroid

    scorer = ProfileScorer(mu=mu, actions=["escalate", "suppress"])

    factor_names = [
        "travel_match",
        "asset_criticality",
        "threat_intel",
        "pattern_history",
    ]

    scenarios = [
        EvaluationScenario(
            scenario_id=f"S{i}",
            domain="test",
            category="cat0",
            category_index=0,
            factors=np.array([0.9, 0.1, 0.1, 0.1]),
            expected_action="escalate",
            expected_action_index=0,
        )
        for i in range(5)
    ]
    return scorer, scenarios, factor_names


class TestAblationReportFields:
    def test_ablation_report_fields(self):
        scorer, scenarios, factor_names = make_ablation_inputs()
        report = run_ablation(scorer, scenarios, factor_names)
        assert report.n_factors == 4
        assert report.n_scenarios == 5
        assert isinstance(report.baseline_accuracy, float)
        assert isinstance(report.most_important, str)
        assert isinstance(report.least_important, str)


class TestAblationResultsCount:
    def test_ablation_results_count(self):
        scorer, scenarios, factor_names = make_ablation_inputs()
        report = run_ablation(scorer, scenarios, factor_names)
        assert len(report.results) == 4


class TestAblationRanksAssigned:
    def test_ablation_ranks_assigned(self):
        scorer, scenarios, factor_names = make_ablation_inputs()
        report = run_ablation(scorer, scenarios, factor_names)
        ranks = [r.importance_rank for r in report.results]
        assert sorted(ranks) == list(range(1, 5))


class TestDominantFactorIdentified:
    def test_dominant_factor_identified(self):
        scorer, scenarios, factor_names = make_ablation_inputs()
        report = run_ablation(scorer, scenarios, factor_names)
        assert report.most_important == "travel_match"


class TestAccuracyDropSortedDescending:
    def test_accuracy_drop_sorted_descending(self):
        scorer, scenarios, factor_names = make_ablation_inputs()
        report = run_ablation(scorer, scenarios, factor_names)
        drops = [r.accuracy_drop for r in report.results]
        assert drops == sorted(drops, reverse=True)


class TestOriginalScenariosNotModified:
    def test_original_scenarios_not_modified(self):
        scorer, scenarios, factor_names = make_ablation_inputs()
        original_factors = [s.factors.copy() for s in scenarios]
        run_ablation(scorer, scenarios, factor_names)
        for i, s in enumerate(scenarios):
            np.testing.assert_array_equal(s.factors, original_factors[i])


class TestEmptyScenariosRaises:
    def test_empty_scenarios_raises(self):
        from gae.profile_scorer import ProfileScorer

        scorer = ProfileScorer(
            mu=np.zeros((1, 2, 4)), actions=["escalate", "suppress"]
        )
        with pytest.raises(ValueError):
            run_ablation(scorer, [], ["f0", "f1", "f2", "f3"])


class TestFactorNamesLengthMismatchRaises:
    def test_factor_names_length_mismatch_raises(self):
        scorer, scenarios, factor_names = make_ablation_inputs()
        with pytest.raises(ValueError):
            run_ablation(scorer, scenarios, ["only_one"])
