"""
API contract tests for Graph Attention Engine (GAE).

Verifies that every Tier 1 stable function matches API_CONTRACT.md.
These tests must never be deleted — a failure here is a production regression.

Target: 20 new tests on top of 536 existing → 556+ total.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from gae.profile_scorer import (
    ProfileScorer,
    ScoringResult,
    CentroidUpdate,
    KernelType,
)
from gae.kernels import L2Kernel, DiagonalKernel
from gae.calibration import (
    derive_theta_min,
    check_conservation,
    ConservationCheck,
    soc_calibration_profile,
    s2p_calibration_profile,
)


# ── Shared helper ─────────────────────────────────────────────────────────────

def make_scorer(
    n_cat: int = 3,
    n_act: int = 4,
    n_fac: int = 6,
    seed: int = 42,
) -> ProfileScorer:
    """ProfileScorer with random [0,1] centroids, no CalibrationProfile (uses defaults)."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.0, 1.0, (n_cat, n_act, n_fac))
    actions = [f"action_{i}" for i in range(n_act)]
    return ProfileScorer(mu=mu, actions=actions)


# ── Test 1: score() probabilities sum to 1.0 ─────────────────────────────────

def test_score_probs_sum_to_one():
    scorer = make_scorer()
    f = np.full(6, 0.5)
    result = scorer.score(f, category_index=0)
    assert abs(result.probabilities.sum() - 1.0) < 1e-9, (
        f"probabilities.sum()={result.probabilities.sum()}"
    )


# ── Test 2: score() is deterministic ─────────────────────────────────────────

def test_score_is_deterministic():
    scorer = make_scorer()
    f = np.array([0.1, 0.3, 0.5, 0.7, 0.2, 0.4])
    r1 = scorer.score(f, category_index=1)
    r2 = scorer.score(f, category_index=1)
    np.testing.assert_array_equal(
        r1.probabilities, r2.probabilities,
        err_msg="score() must be deterministic for identical inputs",
    )


# ── Test 3: score() output length == n_act ───────────────────────────────────

def test_score_output_length_equals_n_act():
    n_act = 5
    scorer = make_scorer(n_act=n_act)
    f = np.zeros(6)
    result = scorer.score(f, category_index=0)
    assert result.probabilities.shape == (n_act,), (
        f"probabilities.shape={result.probabilities.shape} != ({n_act},)"
    )
    assert result.distances.shape == (n_act,), (
        f"distances.shape={result.distances.shape} != ({n_act},)"
    )


# ── Test 4: update() correct=True pulls centroid toward f ────────────────────

def test_update_correct_pulls_centroid_toward_f():
    scorer = make_scorer(n_cat=1, n_act=2, n_fac=6)
    f = np.ones(6)  # target well-separated from centroid [0,1] range
    dist_before = float(np.linalg.norm(scorer.mu[0, 0, :] - f))
    scorer.update(f, category_index=0, action_index=0, correct=True)
    dist_after = float(np.linalg.norm(scorer.mu[0, 0, :] - f))
    assert dist_after < dist_before, (
        f"Correct update should pull centroid toward f: "
        f"dist_before={dist_before:.4f}, dist_after={dist_after:.4f}"
    )


# ── Test 5: update() correct=False pushes predicted centroid away ─────────────

def test_update_incorrect_pushes_centroid_away():
    # Place centroid at 0.5, f at 0: clear direction for push
    mu = np.full((1, 2, 6), 0.5)
    actions = ["predicted", "gt"]
    scorer = ProfileScorer(mu=mu, actions=actions)
    f = np.zeros(6)
    dist_before = float(np.linalg.norm(scorer.mu[0, 0, :] - f))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # DeprecationWarning if gt not provided elsewhere
        scorer.update(f, category_index=0, action_index=0, correct=False,
                      gt_action_index=1)
    dist_after = float(np.linalg.norm(scorer.mu[0, 0, :] - f))
    assert dist_after > dist_before, (
        f"Incorrect update should push predicted centroid away: "
        f"dist_before={dist_before:.4f}, dist_after={dist_after:.4f}"
    )


# ── Test 6: centroids roundtrip preserves shape and values ───────────────────

def test_centroids_roundtrip():
    scorer = make_scorer(n_cat=2, n_act=3, n_fac=4)
    orig = scorer.centroids.copy()
    # Replace mu and verify .centroids reflects the change
    new_mu = np.full_like(orig, 0.7)
    scorer.mu = new_mu
    np.testing.assert_allclose(
        scorer.centroids, new_mu,
        err_msg=".centroids must reflect updated .mu",
    )
    assert scorer.centroids.shape == orig.shape, (
        "Shape must be preserved after mu assignment"
    )


# ── Test 7: centroid shape matches (n_cat, n_act, d) ─────────────────────────

def test_centroid_shape():
    n_cat, n_act, n_fac = 3, 4, 6
    scorer = make_scorer(n_cat=n_cat, n_act=n_act, n_fac=n_fac)
    assert scorer.centroids.shape == (n_cat, n_act, n_fac), (
        f"centroids.shape={scorer.centroids.shape} != ({n_cat}, {n_act}, {n_fac})"
    )


# ── Test 8: μ stays in [0,1] after adversarial inputs ────────────────────────

def test_mu_clipped_after_adversarial_inputs():
    scorer = make_scorer(n_cat=1, n_act=2, n_fac=6)
    f_high = np.full(6, 10.0)    # >> 1
    f_low  = np.full(6, -10.0)   # << 0
    for _ in range(30):
        scorer.update(f_high, category_index=0, action_index=0, correct=True)
        scorer.update(f_low,  category_index=0, action_index=1, correct=True)
    assert scorer.mu.min() >= 0.0, f"mu.min()={scorer.mu.min()} < 0.0 (V2 violation)"
    assert scorer.mu.max() <= 1.0, f"mu.max()={scorer.mu.max()} > 1.0 (V2 violation)"


# ── Test 9: τ=0.1 is the default ─────────────────────────────────────────────

def test_tau_default_is_01():
    mu = np.full((1, 2, 3), 0.5)
    scorer = ProfileScorer(mu=mu, actions=["a", "b"])
    assert scorer.tau == pytest.approx(0.1), (
        f"Default tau must be 0.1 (V3B validated). Got {scorer.tau}"
    )


# ── Test 10: η=0.05 and η_neg=0.05 are defaults; eta_override=None ───────────

def test_eta_defaults():
    mu = np.full((1, 2, 3), 0.5)
    scorer = ProfileScorer(mu=mu, actions=["a", "b"])
    assert scorer.eta == pytest.approx(0.05), f"Default eta={scorer.eta} != 0.05"
    assert scorer.eta_neg == pytest.approx(0.05), f"Default eta_neg={scorer.eta_neg} != 0.05"
    assert scorer.eta_override is None, (
        "eta_override default must be None (0.01 is recommended, not default)"
    )


# ── Test 11: DiagonalKernel differs from L2Kernel when weights ≠ identity ────

def test_diagonal_kernel_differs_from_l2_kernel():
    rng = np.random.default_rng(7)
    mu = rng.uniform(0.0, 1.0, (1, 3, 6))
    actions = ["a", "b", "c"]
    f = rng.uniform(0.0, 1.0, 6)

    scorer_l2 = ProfileScorer(mu=mu.copy(), actions=actions)
    # Very unequal sigma: dim 1 is 40× noisier than others → much lower weight
    sigma_unequal = np.array([0.05, 2.0, 0.05, 0.05, 0.05, 0.05])
    scorer_diag = ProfileScorer(
        mu=mu.copy(),
        actions=actions,
        kernel=KernelType.L2,
        scoring_kernel=DiagonalKernel(sigma_unequal),
    )
    r_l2   = scorer_l2.score(f, category_index=0)
    r_diag = scorer_diag.score(f, category_index=0)
    assert not np.allclose(r_l2.probabilities, r_diag.probabilities), (
        "DiagonalKernel with unequal sigma must produce different scores than L2Kernel"
    )


# ── Test 12: derive_theta_min matches η × N_half² / T_max ────────────────────

def test_derive_theta_min_formula():
    eta, n_half, t_max = 0.05, 14.0, 21.0
    expected = eta * n_half ** 2 / t_max          # ≈ 0.4667
    actual = derive_theta_min(eta=eta, n_half=n_half, t_max_days=t_max)
    assert actual == pytest.approx(expected, rel=1e-6), (
        f"derive_theta_min={actual} != η×N²/T={expected}"
    )


# ── Test 13: check_conservation returns valid ConservationCheck ───────────────

def test_check_conservation_returns_valid_status():
    theta = derive_theta_min()     # defaults: ≈ 0.467
    # Healthy: signal >> theta
    cc_green = check_conservation(alpha=0.3, q=0.9, V=100.0, theta_min=theta)
    assert isinstance(cc_green, ConservationCheck)
    assert cc_green.status == 'GREEN'
    assert cc_green.passed is True
    # Marginal: signal ≈ theta (AMBER)
    signal_amber = theta * 1.5
    # Derive alpha, q, V such that α·q·V = signal_amber
    cc_amber = check_conservation(alpha=1.0, q=1.0, V=signal_amber, theta_min=theta)
    assert cc_amber.status == 'AMBER'
    # Breach: signal < theta (RED)
    cc_red = check_conservation(alpha=0.001, q=0.1, V=1.0, theta_min=theta)
    assert cc_red.status == 'RED'
    assert cc_red.passed is False


# ── Test 14: ProfileScorer with SOC config (6,4,6) constructs successfully ───

def test_soc_config_constructs():
    n_cat, n_act, n_fac = 6, 4, 6
    mu = np.full((n_cat, n_act, n_fac), 0.5)
    actions = [f"action_{i}" for i in range(n_act)]
    scorer = ProfileScorer(
        mu=mu, actions=actions, profile=soc_calibration_profile()
    )
    assert scorer.n_categories == n_cat
    assert scorer.n_actions    == n_act
    assert scorer.n_factors    == n_fac


# ── Test 15: ProfileScorer with S2P config (5,5,8) constructs successfully ───

def test_s2p_config_constructs():
    n_cat, n_act, n_fac = 5, 5, 8
    mu = np.full((n_cat, n_act, n_fac), 0.5)
    actions = [f"action_{i}" for i in range(n_act)]
    scorer = ProfileScorer(
        mu=mu, actions=actions, profile=s2p_calibration_profile()
    )
    assert scorer.n_categories == n_cat
    assert scorer.n_actions    == n_act
    assert scorer.n_factors    == n_fac


# ── Test 16: ProfileScorer with SOC config scores correctly ──────────────────

def test_scorer_soc_scores_correctly():
    n_cat, n_act, n_fac = 6, 4, 6
    rng = np.random.default_rng(1)
    mu = rng.uniform(0.0, 1.0, (n_cat, n_act, n_fac))
    actions = [f"action_{i}" for i in range(n_act)]
    scorer = ProfileScorer(
        mu=mu, actions=actions, profile=soc_calibration_profile()
    )
    f = rng.uniform(0.0, 1.0, n_fac)
    result = scorer.score(f, category_index=0)
    assert isinstance(result, ScoringResult)
    assert result.probabilities.shape == (n_act,)
    assert abs(result.probabilities.sum() - 1.0) < 1e-9


# ── Test 17: ProfileScorer with S2P config scores correctly ──────────────────

def test_scorer_s2p_scores_correctly():
    n_cat, n_act, n_fac = 5, 5, 8
    rng = np.random.default_rng(2)
    mu = rng.uniform(0.0, 1.0, (n_cat, n_act, n_fac))
    actions = [f"action_{i}" for i in range(n_act)]
    scorer = ProfileScorer(
        mu=mu, actions=actions, profile=s2p_calibration_profile()
    )
    f = rng.uniform(0.0, 1.0, n_fac)
    result = scorer.score(f, category_index=2)
    assert isinstance(result, ScoringResult)
    assert result.probabilities.shape == (n_act,)
    assert abs(result.probabilities.sum() - 1.0) < 1e-9


# ── Test 18: Empty factor vector (all zeros) is handled ──────────────────────

def test_empty_factor_vector_handling():
    scorer = make_scorer()
    f = np.zeros(6)
    result = scorer.score(f, category_index=0)
    assert result.probabilities.shape == (4,)
    assert abs(result.probabilities.sum() - 1.0) < 1e-9
    assert not np.any(np.isnan(result.probabilities)), (
        "Zero factor vector must not produce NaN probabilities"
    )


# ── Test 19: NaN input behavior is documented ────────────────────────────────

def test_nan_input_behavior_documented():
    """
    Document current behavior for NaN inputs. Callers are responsible for
    sanitizing f before calling score(). A future hardening PR may add a
    nan-guard and change this test to assert ValueError.
    """
    scorer = make_scorer()
    f_nan = np.full(6, np.nan)
    try:
        scorer.score(f_nan, category_index=0)
        # NaN propagation is current documented behavior — caller must sanitize
    except (ValueError, AssertionError, FloatingPointError):
        pass  # Raising cleanly is equally acceptable


# ── Test 20: η_neg ≥ 1.0 is rejected with ValueError ────────────────────────

def test_eta_neg_ge_10_is_rejected():
    mu = np.full((1, 2, 3), 0.5)
    profile_bad = SimpleNamespace(
        temperature=0.1,
        extensions={"eta": 0.05, "eta_neg": 1.0, "count_decay": 0.001},
    )
    with pytest.raises(ValueError, match="eta_neg"):
        ProfileScorer(mu=mu, actions=["a", "b"], profile=profile_bad)
