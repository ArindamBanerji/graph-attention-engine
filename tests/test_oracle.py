"""
Tests for gae.oracle — OracleProvider protocol, GTAlignedOracle, BernoulliOracle.

8 tests covering: protocol compliance, correctness on centroid, accuracy
on noisy centroidal data, from_profile_scorer factory, Bernoulli rates.
"""

import numpy as np
import pytest

from gae.oracle import (
    BernoulliOracle,
    GTAlignedOracle,
    OracleProvider,
    OracleResult,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_separated_mu(n_cat: int = 2, n_act: int = 3, n_fac: int = 4) -> np.ndarray:
    """Centroids with one dominant dimension per action."""
    mu = np.zeros((n_cat, n_act, n_fac))
    for c in range(n_cat):
        for a in range(n_act):
            vec = np.zeros(n_fac)
            vec[a % n_fac] = 0.9
            mu[c, a, :] = vec
    return mu


# ---------------------------------------------------------------------------
# TEST 1 — both classes satisfy OracleProvider protocol
# ---------------------------------------------------------------------------

def test_oracle_provider_protocol():
    mu = make_separated_mu()
    gt_oracle = GTAlignedOracle(mu=mu, actions=["a0", "a1", "a2"])
    b_oracle   = BernoulliOracle(n_actions=3, actions=["a0", "a1", "a2"])
    assert isinstance(gt_oracle, OracleProvider)
    assert isinstance(b_oracle, OracleProvider)


# ---------------------------------------------------------------------------
# TEST 2 — GTAlignedOracle: f exactly at centroid → correct, confidence=1.0
# ---------------------------------------------------------------------------

def test_gt_aligned_oracle_correct_on_centroid():
    mu = make_separated_mu()
    oracle = GTAlignedOracle(mu=mu, actions=["a0", "a1", "a2"])

    # f exactly equal to centroid [0, 0, :]
    f = mu[0, 0, :].copy()
    result = oracle.query(f, category_index=0, taken_action_index=0)

    assert isinstance(result, OracleResult)
    assert result.correct is True
    assert result.gt_action_idx == 0
    assert result.gt_action_name == "a0"
    assert result.confidence == 1.0


# ---------------------------------------------------------------------------
# TEST 3 — GTAlignedOracle: f closest to action 0, taken=1 → incorrect
# ---------------------------------------------------------------------------

def test_gt_aligned_oracle_incorrect_on_wrong_action():
    mu = make_separated_mu()
    oracle = GTAlignedOracle(mu=mu, actions=["a0", "a1", "a2"])

    # f is at centroid 0 — GT must be action 0
    f = mu[0, 0, :].copy()
    result = oracle.query(f, category_index=0, taken_action_index=1)

    assert result.correct is False
    assert result.gt_action_idx == 0
    assert result.gt_action_name == "a0"


# ---------------------------------------------------------------------------
# TEST 4 — GTAlignedOracle: accuracy > 95% on centroidal noisy data
# ---------------------------------------------------------------------------

def test_gt_aligned_oracle_accuracy_on_centroidal_data():
    """
    f = centroid + small Gaussian noise (sigma=0.05).
    3 categories x 4 actions x 20 samples. Expect accuracy > 95%.
    Reference: EXP-C1 (97.89%).
    """
    n_cat, n_act, n_fac = 3, 4, 6
    np.random.seed(0)

    # Bimodal centroids (0.1/0.9 values) — well-separated
    mu = np.zeros((n_cat, n_act, n_fac))
    for c in range(n_cat):
        for a in range(n_act):
            vec = np.random.choice([0.1, 0.9], size=n_fac)
            mu[c, a, :] = vec

    actions = [f"a{i}" for i in range(n_act)]
    oracle = GTAlignedOracle(mu=mu, actions=actions)

    correct = 0
    total = 0
    for c in range(n_cat):
        for a in range(n_act):
            for _ in range(20):
                f = np.clip(
                    mu[c, a, :] + np.random.normal(0, 0.05, n_fac),
                    0.0, 1.0,
                )
                result = oracle.query(f, category_index=c,
                                      taken_action_index=a)
                if result.correct:
                    correct += 1
                total += 1

    accuracy = correct / total
    assert accuracy > 0.95, (
        f"GTAlignedOracle accuracy {accuracy:.3f} < 0.95 "
        "(expected >95% on centroidal data, ref EXP-C1: 97.89%)"
    )


# ---------------------------------------------------------------------------
# TEST 5 — GTAlignedOracle.from_profile_scorer round-trip
# ---------------------------------------------------------------------------

def test_gt_aligned_from_profile_scorer():
    from gae.profile_scorer import ProfileScorer

    mu = make_separated_mu()
    scorer = ProfileScorer(mu=mu, actions=["a0", "a1", "a2"])
    oracle = GTAlignedOracle.from_profile_scorer(scorer)

    assert oracle.mu.shape == scorer.mu.shape
    assert oracle.actions == scorer.actions
    np.testing.assert_array_equal(oracle.mu, scorer.mu)


# ---------------------------------------------------------------------------
# TEST 6 — BernoulliOracle: correct_rate=1.0 → always correct
# ---------------------------------------------------------------------------

def test_bernoulli_oracle_correct_rate_1():
    oracle = BernoulliOracle(
        n_actions=4,
        actions=["a0", "a1", "a2", "a3"],
        correct_rate=1.0,
        seed=0,
    )
    f = np.zeros(4)
    for _ in range(100):
        result = oracle.query(f, category_index=0, taken_action_index=0)
        assert result.correct is True
        assert result.gt_action_idx == 0


# ---------------------------------------------------------------------------
# TEST 7 — BernoulliOracle: correct_rate=0.0 → always incorrect
# ---------------------------------------------------------------------------

def test_bernoulli_oracle_correct_rate_0():
    oracle = BernoulliOracle(
        n_actions=4,
        actions=["a0", "a1", "a2", "a3"],
        correct_rate=0.0,
        seed=0,
    )
    f = np.zeros(4)
    for _ in range(100):
        result = oracle.query(f, category_index=0, taken_action_index=0)
        assert result.correct is False
        assert result.gt_action_idx != 0


# ---------------------------------------------------------------------------
# TEST 8 — BernoulliOracle: empirical rate ≈ correct_rate
# ---------------------------------------------------------------------------

def test_bernoulli_oracle_rate_approximately_correct():
    oracle = BernoulliOracle(
        n_actions=4,
        actions=["a0", "a1", "a2", "a3"],
        correct_rate=0.75,
        seed=42,
    )
    f = np.zeros(4)
    correct = sum(
        oracle.query(f, 0, 0).correct
        for _ in range(1000)
    )
    rate = correct / 1000
    assert 0.70 < rate < 0.80, (
        f"Empirical correct rate {rate:.3f} not in (0.70, 0.80) "
        "for correct_rate=0.75"
    )
