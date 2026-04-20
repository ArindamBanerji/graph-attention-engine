"""Tests for gae.evaluation — EvaluationScenario, EvaluationReport, run_evaluation, compute_ece."""

import numpy as np
import pytest

from gae.evaluation import EvaluationReport, EvaluationScenario, compute_ece, run_evaluation


def make_scorer_and_scenarios():
    """
    Build a well-separated ProfileScorer and 3 EvaluationScenarios.

    mu[c, a, :] set so each scenario's factor vector is closest to its
    expected action centroid — guarantees 100% accuracy on these scenarios.
    """
    from gae.profile_scorer import ProfileScorer

    mu = np.zeros((2, 3, 4))
    mu[0, 0, :] = [0.9, 0.1, 0.1, 0.1]
    mu[0, 1, :] = [0.1, 0.9, 0.1, 0.1]
    mu[0, 2, :] = [0.1, 0.1, 0.9, 0.1]
    mu[1, 0, :] = [0.8, 0.2, 0.1, 0.1]
    mu[1, 1, :] = [0.1, 0.1, 0.8, 0.2]
    mu[1, 2, :] = [0.1, 0.1, 0.1, 0.9]

    scorer = ProfileScorer(mu=mu, actions=["escalate", "investigate", "suppress"])

    scenarios = [
        EvaluationScenario(
            scenario_id="S1",
            domain="test",
            category="cat0",
            category_index=0,
            factors=np.array([0.9, 0.1, 0.1, 0.1]),
            expected_action="escalate",
            expected_action_index=0,
        ),
        EvaluationScenario(
            scenario_id="S2",
            domain="test",
            category="cat0",
            category_index=0,
            factors=np.array([0.1, 0.9, 0.1, 0.1]),
            expected_action="investigate",
            expected_action_index=1,
        ),
        EvaluationScenario(
            scenario_id="S3",
            domain="test",
            category="cat1",
            category_index=1,
            factors=np.array([0.8, 0.2, 0.1, 0.1]),
            expected_action="escalate",
            expected_action_index=0,
        ),
    ]
    return scorer, scenarios


class TestPerfectScorerAccuracy:
    def test_perfect_scorer_accuracy(self):
        scorer, scenarios = make_scorer_and_scenarios()
        report = run_evaluation(scorer, scenarios)
        assert report.accuracy == 1.0
        assert report.n_correct == 3
        assert report.n_scenarios == 3


class TestReportFieldsPopulated:
    def test_report_fields_populated(self):
        scorer, scenarios = make_scorer_and_scenarios()
        report = run_evaluation(scorer, scenarios)
        assert "cat0" in report.by_category
        assert "cat1" in report.by_category
        assert "escalate" in report.precision_per_action
        assert isinstance(report.ece, float)
        assert len(report.scenario_results) == 3


class TestEmptyScenarios:
    def test_empty_scenarios_returns_zero_report(self):
        from gae.profile_scorer import ProfileScorer

        scorer = ProfileScorer(
            mu=np.full((2, 3, 4), 0.5),
            actions=["a", "b", "c"],
        )
        report = run_evaluation(scorer, [])
        assert report.accuracy == 0.0
        assert report.n_scenarios == 0


class TestPerCategoryAccuracy:
    def test_per_category_accuracy(self):
        scorer, scenarios = make_scorer_and_scenarios()
        report = run_evaluation(scorer, scenarios)
        assert report.by_category["cat0"] == 1.0
        assert report.by_category["cat1"] == 1.0


class TestScenarioResultsDetail:
    def test_scenario_results_detail(self):
        scorer, scenarios = make_scorer_and_scenarios()
        report = run_evaluation(scorer, scenarios)
        for r in report.scenario_results:
            assert "scenario_id" in r
            assert "predicted_action" in r
            assert "correct" in r
            assert "confidence" in r


class TestECEValidRange:
    def test_ece_in_valid_range(self):
        scorer, scenarios = make_scorer_and_scenarios()
        report = run_evaluation(scorer, scenarios)
        assert 0.0 <= report.ece <= 1.0


class TestComputeECEOverconfident:
    def test_compute_ece_overconfident(self):
        ece = compute_ece([0.95] * 100, [False] * 100)
        assert ece > 0.4


class TestLearningChangesScoreOutput:
    def test_learning_changes_score_output(self):
        from gae.oracle import GTAlignedOracle

        scorer, scenarios = make_scorer_and_scenarios()
        oracle = GTAlignedOracle(mu=scorer.centroids, actions=scorer.actions)

        f0 = scenarios[0].factors
        result_before = scorer.score(f0, scenarios[0].category_index)
        conf_before = result_before.confidence

        run_evaluation(scorer, scenarios, oracle=oracle, learn=True)

        result_after = scorer.score(f0, scenarios[0].category_index)
        conf_after = result_after.confidence

        assert result_after.action_index == scenarios[0].expected_action_index
        assert conf_after >= conf_before - 0.01
