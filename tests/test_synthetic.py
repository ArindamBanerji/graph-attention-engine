"""
tests/test_synthetic.py — Oracle separation framework tests.

Validates gae/synthetic.py: FactorVectorSampler, CanonicalCentroid,
OracleSeparationExperiment, GammaResult, Phase1Result, Phase2Result.

Reference: gae_design_v10.4 §10.14; oracle separation experiments April 2026.
"""

import numpy as np
import pytest

from gae.synthetic import (
    OracleSeparationExperiment,
    FactorVectorSampler,
    CanonicalCentroid,
    GammaResult,
)
from gae.profile_scorer import ProfileScorer


# ── helpers ──────────────────────────────────────────────────────────

def _make_scorer(C=3, A=2, d=4):
    mu = np.full((C, A, d), 0.5)
    return ProfileScorer(mu=mu, actions=["a0", "a1"])


def _make_exp(epsilon_firm, scorer=None, C=3, A=2, d=4):
    if scorer is None:
        scorer = _make_scorer(C=C, A=A, d=d)
    gt = np.full((C, A, d), 0.7)
    canonical = CanonicalCentroid.from_ground_truth(gt)
    return OracleSeparationExperiment(
        scorer=scorer,
        canonical_gt1=canonical,
        epsilon_firm=epsilon_firm,
        disruption_magnitude=0.25,
        disrupted_categories=[0],
        alpha_cat=1 / 3,
        window=10,
        theta=0.85,
        max_decisions=200,
    )


# ── tests ─────────────────────────────────────────────────────────────

def test_oracle_separation_gamma_lt_1_below_threshold():
    """ε_firm < threshold → γ < 1 (theorem prediction)."""
    exp = _make_exp(epsilon_firm=0.05)   # below threshold ≈ 0.125
    assert not exp.is_above_threshold
    assert exp.theorem_prediction == "gamma_lt_1"


def test_oracle_separation_gamma_gt_1_above_threshold():
    """ε_firm > threshold → γ > 1 (theorem prediction)."""
    exp = _make_exp(epsilon_firm=0.20)   # above threshold ≈ 0.125
    assert exp.is_above_threshold
    assert exp.theorem_prediction == "gamma_gt_1"


def test_factor_vector_sampler_variance_within_spec():
    """Generated factor vectors have per-factor variance matching sigma."""
    sigma = np.array([0.05, 0.15, 0.10, 0.20])
    sampler = FactorVectorSampler(d=4, sigma_profile=sigma, seed=42)
    samples = sampler.sample("cold_start", n=500)
    vectors = np.array([s.f for s in samples])

    empirical_std = vectors.std(axis=0)
    for i in range(4):
        assert abs(empirical_std[i] - sigma[i]) < sigma[i] * 0.8, (
            f"Factor {i}: empirical_std={empirical_std[i]:.3f} "
            f"vs sigma={sigma[i]:.3f}"
        )


def test_canonical_centroid_apply_disruption_magnitude():
    """Disruption magnitude matches on disrupted categories only."""
    gt = np.zeros((3, 2, 4))
    canonical = CanonicalCentroid.from_ground_truth(gt)

    delta = np.full((2, 4), 0.1)
    disrupted = canonical.apply_disruption(delta, categories=[0])

    # Category 0 should have moved
    assert not np.allclose(disrupted.gt[0], gt[0])
    # Categories 1, 2 should be unchanged
    assert np.allclose(disrupted.gt[1], gt[1])
    assert np.allclose(disrupted.gt[2], gt[2])


def test_gamma_result_n_half_gap_detection():
    """n_half_gap_detected is True when N_half fires before centroid converges."""
    result = GammaResult(
        n_half_1=10, n_half_2=12, gamma=0.833,
        centroid_dist_phase1=[3.0, 2.9, 2.8],
        centroid_dist_phase2=[2.8, 2.7, 2.6],
        n_half_gap_detected=True,
        epsilon_firm=0.05,
        threshold=0.125,
        theorem_prediction="gamma_lt_1",
        simulation_confirms=True,
        note="test",
    )
    assert result.n_half_gap_detected is True
    assert not result.is_above_threshold
