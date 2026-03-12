"""
Tests for gae.profile_scorer — ProfileScorer, KernelType, ScoringResult.

12 tests covering: scoring, learning, calibration, factory methods,
kernel warnings, and diagnostics.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from gae.profile_scorer import (
    CentroidUpdate,
    KernelType,
    ProfileScorer,
    ScoringResult,
    build_profile_scorer,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_simple_scorer(
    n_cat: int = 3,
    n_act: int = 4,
    n_fac: int = 6,
    kernel: KernelType = KernelType.L2,
    tau: float = 0.1,
) -> ProfileScorer:
    """Build a scorer with random [0,1] centroids and a simple profile."""
    np.random.seed(42)
    mu = np.random.uniform(0.0, 1.0, (n_cat, n_act, n_fac))
    actions = [f"action_{i}" for i in range(n_act)]
    profile = SimpleNamespace(
        temperature=tau,
        extensions={"eta": 0.05, "eta_neg": 0.05, "count_decay": 0.001},
    )
    return ProfileScorer(mu=mu, actions=actions, kernel=kernel, profile=profile)


# ---------------------------------------------------------------------------
# TEST 1 — score returns a well-formed ScoringResult
# ---------------------------------------------------------------------------

def test_score_returns_scoring_result():
    scorer = make_simple_scorer()
    np.random.seed(7)
    f = np.random.uniform(0, 1, 6)
    result = scorer.score(f, category_index=0)

    assert isinstance(result, ScoringResult)
    assert result.action_index in range(4)
    assert result.action_name in [f"action_{i}" for i in range(4)]
    assert result.probabilities.shape == (4,)
    assert abs(result.probabilities.sum() - 1.0) < 1e-6
    assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# TEST 2 — probabilities always sum to 1.0
# ---------------------------------------------------------------------------

def test_probabilities_sum_to_one():
    scorer = make_simple_scorer()
    np.random.seed(99)
    for _ in range(20):
        f = np.random.uniform(0, 1, 6)
        result = scorer.score(f, category_index=0)
        assert abs(result.probabilities.sum() - 1.0) < 1e-6, (
            f"probabilities sum={result.probabilities.sum()}, expected 1.0"
        )


# ---------------------------------------------------------------------------
# TEST 3 — L2 self-test accuracy > 95% (EXP-C1 condition)
# ---------------------------------------------------------------------------

def test_l2_self_test_accuracy():
    """
    Centroidal synthetic benchmark: well-separated centroids, Gaussian noise.
    Validates the EXP-C1 condition: 97.89% on [0,1] factors with L2 kernel.
    """
    n_cat, n_act, n_fac = 2, 3, 6
    np.random.seed(0)

    # Create well-separated centroids (bimodal: 0.1 or 0.9)
    mu = np.zeros((n_cat, n_act, n_fac))
    for c in range(n_cat):
        for a in range(n_act):
            vec = np.random.choice([0.1, 0.9], size=n_fac)
            mu[c, a, :] = vec

    actions = ["a0", "a1", "a2"]
    scorer = ProfileScorer(mu=mu, actions=actions)  # default τ=0.1

    correct = 0
    total = 0
    for c in range(n_cat):
        for a in range(n_act):
            for _ in range(34):  # ~204 total
                f = np.clip(
                    mu[c, a, :] + np.random.normal(0, 0.05, n_fac),
                    0.0, 1.0,
                )
                result = scorer.score(f, c)
                if result.action_index == a:
                    correct += 1
                total += 1

    accuracy = correct / total
    assert accuracy > 0.95, (
        f"L2 self-test accuracy {accuracy:.3f} < 0.95. "
        "Expected >95% on centroidal synthetic data (ref EXP-C1: 97.89%)."
    )


# ---------------------------------------------------------------------------
# TEST 4 — correct update pulls centroid toward f
# ---------------------------------------------------------------------------

def test_update_pull_moves_centroid_toward_f():
    scorer = make_simple_scorer()
    np.random.seed(3)
    f = np.random.uniform(0, 1, 6)

    result = scorer.score(f, category_index=0)
    a = result.action_index

    mu_before = scorer.mu[0, a, :].copy()
    scorer.update(f, 0, a, correct=True)
    mu_after = scorer.mu[0, a, :]

    dist_before = np.linalg.norm(mu_before - f)
    dist_after  = np.linalg.norm(mu_after  - f)
    assert dist_after < dist_before, (
        f"Centroid did not move closer to f: {dist_before:.6f} → {dist_after:.6f}"
    )


# ---------------------------------------------------------------------------
# TEST 5 — update clips centroids to [0.0, 1.0]
# ---------------------------------------------------------------------------

def test_update_clips_to_unit_interval():
    # Upper bound: centroid near 1.0, f = 1.0, 100 correct updates
    mu_hi = np.full((1, 1, 6), 0.9)
    scorer_hi = ProfileScorer(mu=mu_hi, actions=["a0"])
    f_hi = np.ones(6)
    for _ in range(100):
        scorer_hi.update(f_hi, 0, 0, correct=True)
    assert np.all(scorer_hi.mu[0, 0, :] <= 1.0), (
        f"Centroid escaped above 1.0: {scorer_hi.mu[0, 0, :]}"
    )

    # Lower bound: centroid near 0.0, f = 0.0, 100 correct updates
    mu_lo = np.full((1, 1, 6), 0.1)
    scorer_lo = ProfileScorer(mu=mu_lo, actions=["a0"])
    f_lo = np.zeros(6)
    for _ in range(100):
        scorer_lo.update(f_lo, 0, 0, correct=True)
    assert np.all(scorer_lo.mu[0, 0, :] >= 0.0), (
        f"Centroid escaped below 0.0: {scorer_lo.mu[0, 0, :]}"
    )


# ---------------------------------------------------------------------------
# TEST 6 — update increments observation count
# ---------------------------------------------------------------------------

def test_update_increments_count():
    scorer = make_simple_scorer()
    assert scorer.counts[0, 0] == 0

    f = np.ones(6) * 0.5
    scorer.update(f, 0, 0, correct=True)
    assert scorer.counts[0, 0] == 1

    scorer.update(f, 0, 0, correct=True)
    assert scorer.counts[0, 0] == 2


# ---------------------------------------------------------------------------
# TEST 7 — lower τ produces higher confidence (sharper distribution)
# ---------------------------------------------------------------------------

def test_tau_affects_confidence():
    """Lower τ → sharper softmax → higher max probability."""
    mu = np.zeros((1, 2, 4))
    mu[0, 0, :] = [0.9, 0.9, 0.1, 0.1]
    mu[0, 1, :] = [0.1, 0.1, 0.9, 0.9]
    f = np.array([0.9, 0.9, 0.1, 0.1])

    scorer_sharp = ProfileScorer(mu=mu, actions=["a0", "a1"])  # τ=0.1 default

    p_flat = SimpleNamespace(
        temperature=0.5,
        extensions={"eta": 0.05, "eta_neg": 0.05, "count_decay": 0.001},
    )
    scorer_flat = ProfileScorer(mu=mu, actions=["a0", "a1"], profile=p_flat)

    res_sharp = scorer_sharp.score(f, 0)
    res_flat  = scorer_flat.score(f, 0)

    assert res_sharp.confidence > res_flat.confidence, (
        f"Lower τ should produce higher confidence. "
        f"τ=0.1: {res_sharp.confidence:.4f}, τ=0.5: {res_flat.confidence:.4f}"
    )


# ---------------------------------------------------------------------------
# TEST 8 — DOT kernel emits UserWarning at construction
# ---------------------------------------------------------------------------

def test_dot_kernel_emits_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = make_simple_scorer(kernel=KernelType.DOT)
    assert any(
        "DOT" in str(warning.message) or "dot" in str(warning.message).lower()
        for warning in w
    ), "DOT kernel should emit a UserWarning"


# ---------------------------------------------------------------------------
# TEST 9 — MAHALANOBIS kernel emits UserWarning at construction
# ---------------------------------------------------------------------------

def test_mahalanobis_kernel_emits_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = make_simple_scorer(kernel=KernelType.MAHALANOBIS)
    assert any(
        "Mahalanobis" in str(warning.message)
        or "covariance" in str(warning.message).lower()
        for warning in w
    ), "MAHALANOBIS kernel should emit a UserWarning about covariance"


# ---------------------------------------------------------------------------
# TEST 10 — init_from_config round-trip and default fill
# ---------------------------------------------------------------------------

def test_init_from_config_round_trip():
    config_dict = {
        "categories": ["cat_a", "cat_b"],
        "centroids": {
            "cat_a": {
                "escalate": [0.9, 0.8, 0.1, 0.2],
                "suppress": [0.1, 0.2, 0.9, 0.8],
            },
            "cat_b": {
                "escalate": [0.7, 0.8, 0.2, 0.1],
                "suppress": [0.2, 0.1, 0.8, 0.7],
            },
        },
        "kernel": "l2",
    }
    actions = ["escalate", "suppress"]
    scorer = ProfileScorer.init_from_config(config_dict, actions)

    assert scorer.mu.shape == (2, 2, 4)
    assert scorer.tau == 0.1
    np.testing.assert_array_almost_equal(
        scorer.mu[0, 0, :], [0.9, 0.8, 0.1, 0.2]
    )

    # Third unspecified category should default to 0.5
    config2 = dict(config_dict)
    config2["categories"] = ["cat_a", "cat_b", "cat_c"]
    scorer2 = ProfileScorer.init_from_config(config2, actions)
    assert scorer2.mu.shape == (3, 2, 4)
    np.testing.assert_array_almost_equal(
        scorer2.mu[2, :, :], np.full((2, 4), 0.5)
    )


# ---------------------------------------------------------------------------
# TEST 11 — init_from_config raises ValueError on out-of-range values
# ---------------------------------------------------------------------------

def test_init_from_config_invalid_values_raises():
    config_dict = {
        "categories": ["cat_a"],
        "centroids": {
            "cat_a": {"escalate": [1.5, 0.5, 0.5]}  # 1.5 > 1.0
        },
        "kernel": "l2",
    }
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        ProfileScorer.init_from_config(config_dict, ["escalate"])


# ---------------------------------------------------------------------------
# TEST 12 — diagnostics returns correct structure and non-negative separations
# ---------------------------------------------------------------------------

def test_diagnostics_returns_expected_shape():
    scorer = make_simple_scorer(n_cat=3, n_act=4, n_fac=6)
    diag = scorer.diagnostics()

    assert "per_category" in diag
    assert "overall_mean_separation" in diag
    assert "decisions_per_category" in diag
    assert len(diag["per_category"]) == 3
    assert len(diag["decisions_per_category"]) == 3

    for c in range(3):
        assert "separation" in diag["per_category"][c]
        assert "min_dist"   in diag["per_category"][c]
        assert diag["per_category"][c]["separation"] >= 0.0


# ---------------------------------------------------------------------------
# Helper for WIRING-1 tests — accepts dimensional args, not centroid dicts
# ---------------------------------------------------------------------------

def _make_scorer(n_categories: int, n_actions: int, n_factors: int) -> ProfileScorer:
    """Build a ProfileScorer from dimensional args with random [0,1] centroids."""
    mu = np.random.rand(n_categories, n_actions, n_factors)
    actions = [f"action_{i}" for i in range(n_actions)]
    categories = [f"cat_{i}" for i in range(n_categories)]
    return ProfileScorer(mu=mu, actions=actions, categories=categories)


# ---------------------------------------------------------------------------
# TEST 13 — update() returns CentroidUpdate (GAE-WIRING-1)
# ---------------------------------------------------------------------------

def test_update_returns_centroid_update():
    scorer = _make_scorer(n_categories=5, n_actions=5, n_factors=6)
    f = np.random.rand(6)
    result = scorer.update(f, category_index=0, action_index=1, correct=True)
    assert isinstance(result, CentroidUpdate)
    assert result.centroid_delta_norm >= 0.0


# ---------------------------------------------------------------------------
# TEST 14 — frozen scorer returns delta_norm == 0.0 (GAE-WIRING-1)
# ---------------------------------------------------------------------------

def test_update_delta_norm_zero_on_frozen():
    scorer = _make_scorer(n_categories=5, n_actions=5, n_factors=6)
    scorer.freeze()
    f = np.random.rand(6)
    result = scorer.update(f, category_index=0, action_index=1, correct=True)
    assert result.centroid_delta_norm == 0.0


# ---------------------------------------------------------------------------
# TEST 15 — centroid stays in [0.0, 1.0] after update with all-ones f
# ---------------------------------------------------------------------------

def test_centroid_clipped_after_update():
    scorer = _make_scorer(n_categories=5, n_actions=5, n_factors=6)
    f = np.ones(6)
    scorer.update(f, category_index=0, action_index=0, correct=True)
    assert np.all(scorer.mu[0, 0, :] <= 1.0)
    assert np.all(scorer.mu[0, 0, :] >= 0.0)
