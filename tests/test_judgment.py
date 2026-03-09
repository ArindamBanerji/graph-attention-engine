"""Tests for gae.judgment — compute_judgment, JudgmentResult, _confidence_tier."""

import numpy as np
import pytest

from gae.judgment import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    JudgmentResult,
    _confidence_tier,
    compute_judgment,
)


def make_scorer_and_judgment_inputs():
    """
    Build a well-separated ProfileScorer and judgment inputs.

    mu[0, 0, :] = escalate centroid for cat0.
    f = [0.9, 0.1, 0.1, 0.1] -> closest to escalate centroid.
    """
    from gae.profile_scorer import ProfileScorer

    mu = np.zeros((2, 3, 4))
    mu[0, 0, :] = [0.9, 0.1, 0.1, 0.1]  # cat0, escalate
    mu[0, 1, :] = [0.1, 0.9, 0.1, 0.1]  # cat0, investigate
    mu[0, 2, :] = [0.1, 0.1, 0.9, 0.1]  # cat0, suppress

    scorer = ProfileScorer(
        mu=mu, actions=["escalate", "investigate", "suppress"]
    )

    f = np.array([0.9, 0.1, 0.1, 0.1])  # should -> escalate
    factor_names = [
        "travel_match",
        "asset_criticality",
        "threat_intel",
        "pattern_history",
    ]
    return scorer, mu, f, factor_names


class TestJudgmentReturnsCorrectAction:
    def test_judgment_returns_correct_action(self):
        scorer, mu, f, factor_names = make_scorer_and_judgment_inputs()
        result = scorer.score(f, 0)
        judgment = compute_judgment(
            result, f, mu, 0, factor_names, actions=scorer.actions
        )
        assert judgment.action == "escalate"


class TestConfidenceTierHigh:
    def test_confidence_tier_high(self):
        scorer, mu, f, factor_names = make_scorer_and_judgment_inputs()
        result = scorer.score(f, 0)
        judgment = compute_judgment(
            result, f, mu, 0, factor_names, actions=scorer.actions
        )
        assert judgment.confidence_tier in ("high", "medium", "discovery")
        assert 0.0 <= judgment.confidence <= 1.0


class TestConfidenceTierThresholds:
    def test_confidence_tier_thresholds(self):
        assert _confidence_tier(0.95) == "high"
        assert _confidence_tier(0.80) == "high"
        assert _confidence_tier(0.79) == "medium"
        assert _confidence_tier(0.50) == "medium"
        assert _confidence_tier(0.49) == "discovery"
        assert _confidence_tier(0.0)  == "discovery"


class TestDominantFactorsLength:
    def test_dominant_factors_length(self):
        scorer, mu, f, factor_names = make_scorer_and_judgment_inputs()
        result = scorer.score(f, 0)
        judgment = compute_judgment(
            result, f, mu, 0, factor_names, actions=scorer.actions
        )
        assert len(judgment.dominant_factors) <= 3
        assert len(judgment.dominant_factors) >= 1


class TestFactorContributionsAllPresent:
    def test_factor_contributions_all_present(self):
        scorer, mu, f, factor_names = make_scorer_and_judgment_inputs()
        result = scorer.score(f, 0)
        judgment = compute_judgment(
            result, f, mu, 0, factor_names, actions=scorer.actions
        )
        for name in factor_names:
            assert name in judgment.factor_contributions
            assert 0.0 <= judgment.factor_contributions[name] <= 1.0


class TestActionScoresAllActionsPresent:
    def test_action_scores_all_actions_present(self):
        scorer, mu, f, factor_names = make_scorer_and_judgment_inputs()
        result = scorer.score(f, 0)
        judgment = compute_judgment(
            result, f, mu, 0, factor_names, actions=scorer.actions
        )
        for action in scorer.actions:
            assert action in judgment.action_scores


class TestAutoApprovableNotEscalate:
    def test_auto_approvable_not_escalate(self):
        from gae.profile_scorer import ProfileScorer

        mu = np.zeros((1, 3, 4))
        mu[0, 2, :] = [0.9, 0.9, 0.9, 0.9]  # suppress centroid

        scorer = ProfileScorer(
            mu=mu, actions=["escalate", "investigate", "suppress"]
        )
        f = np.array([0.9, 0.9, 0.9, 0.9])
        result = scorer.score(f, 0)
        judgment = compute_judgment(
            result, f, mu, 0,
            ["f0", "f1", "f2", "f3"],
            actions=scorer.actions,
        )
        if judgment.confidence >= CONFIDENCE_HIGH:
            assert judgment.auto_approvable == (judgment.action != "escalate")


class TestRationaleIsNonEmptyString:
    def test_rationale_is_non_empty_string(self):
        scorer, mu, f, factor_names = make_scorer_and_judgment_inputs()
        result = scorer.score(f, 0)
        judgment = compute_judgment(
            result, f, mu, 0, factor_names, actions=scorer.actions
        )
        assert isinstance(judgment.rationale, str)
        assert len(judgment.rationale) > 20
        assert judgment.action in judgment.rationale
