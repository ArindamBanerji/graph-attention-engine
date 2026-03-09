"""Tests for gae.bootstrap — bootstrap_calibration, BootstrapResult."""

import numpy as np
import pytest

from gae.bootstrap import BootstrapResult, bootstrap_calibration
from gae.profile_scorer import ProfileScorer


def make_scorer():
    """Build a simple ProfileScorer with well-separated centroids."""
    mu = np.zeros((2, 3, 4))
    mu[0, 0, :] = [0.9, 0.1, 0.1, 0.1]
    mu[0, 1, :] = [0.1, 0.9, 0.1, 0.1]
    mu[0, 2, :] = [0.1, 0.1, 0.9, 0.1]
    mu[1, 0, :] = [0.8, 0.2, 0.1, 0.1]
    mu[1, 1, :] = [0.1, 0.1, 0.8, 0.2]
    mu[1, 2, :] = [0.1, 0.1, 0.1, 0.9]
    return ProfileScorer(mu=mu, actions=["escalate", "investigate", "suppress"])


CATEGORIES = ["hardware", "software"]


class TestBootstrapReturnsResult:
    def test_bootstrap_returns_result(self):
        scorer = make_scorer()
        result = bootstrap_calibration(scorer, CATEGORIES, n_rounds=2, seed=42)
        assert isinstance(result, BootstrapResult)


class TestBootstrapMutatesScorer:
    def test_bootstrap_mutates_scorer(self):
        scorer = make_scorer()
        mu_before = scorer.mu.copy()
        bootstrap_calibration(scorer, CATEGORIES, n_rounds=3, seed=42)
        assert not np.allclose(scorer.mu, mu_before), (
            "scorer.mu should have changed after bootstrap"
        )


class TestBootstrapDeterministic:
    def test_bootstrap_deterministic(self):
        scorer_a = make_scorer()
        scorer_b = make_scorer()
        result_a = bootstrap_calibration(
            scorer_a, CATEGORIES, n_rounds=3, seed=99
        )
        result_b = bootstrap_calibration(
            scorer_b, CATEGORIES, n_rounds=3, seed=99
        )
        assert result_a.final_drift == result_b.final_drift
        assert result_a.n_decisions == result_b.n_decisions
        np.testing.assert_array_equal(scorer_a.mu, scorer_b.mu)


class TestBootstrapDriftNonnegative:
    def test_bootstrap_drift_nonnegative(self):
        scorer = make_scorer()
        result = bootstrap_calibration(scorer, CATEGORIES, n_rounds=2, seed=42)
        assert result.final_drift >= 0.0


class TestBootstrapDecisionsCount:
    def test_bootstrap_decisions_count(self):
        scorer = make_scorer()
        n_rounds = 4
        samples_per_action = 3
        result = bootstrap_calibration(
            scorer, CATEGORIES,
            n_rounds=n_rounds,
            samples_per_action=samples_per_action,
            seed=42,
        )
        expected = (
            n_rounds
            * len(CATEGORIES)
            * scorer.n_actions
            * samples_per_action
        )
        assert result.n_decisions == expected


class TestBootstrapCentroidsClipped:
    def test_bootstrap_centroids_clipped(self):
        scorer = make_scorer()
        bootstrap_calibration(
            scorer, CATEGORIES, n_rounds=5, sigma=0.5, seed=42
        )
        assert np.all(scorer.mu >= 0.0), "centroid values must be >= 0.0"
        assert np.all(scorer.mu <= 1.0), "centroid values must be <= 1.0"


class TestBootstrapZeroRounds:
    def test_bootstrap_zero_rounds(self):
        scorer = make_scorer()
        result = bootstrap_calibration(
            scorer, CATEGORIES, n_rounds=0, seed=42
        )
        assert result.n_decisions == 0
        assert result.converged is False
        assert result.final_drift == 0.0


class TestBootstrapConvergenceFlag:
    def test_bootstrap_convergence_flag(self):
        # Small sigma means samples stay very close to centroids;
        # many rounds of correct pulls converge quickly.
        scorer = make_scorer()
        result = bootstrap_calibration(
            scorer, CATEGORIES,
            n_rounds=50,
            samples_per_action=10,
            sigma=0.01,
            convergence_tol=0.05,
            seed=42,
        )
        assert result.converged is True, (
            f"Expected converged=True, got final_drift={result.final_drift:.6f}"
        )
