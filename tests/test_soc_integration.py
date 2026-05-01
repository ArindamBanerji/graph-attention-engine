"""
Integration tests for GAE — SOC Copilot consumer patterns.

Verifies the EXACT usage patterns from the SOC backend
(gen-ai-roi-demo-v4-v50). These tests catch regressions that would
break the SOC backend's 290+ call sites.

SOC tensor: (6 categories, 4 actions, d factors)
Categories : credential_access, lateral_movement, data_exfiltration,
             malware_execution, insider_threat, cloud_infrastructure
Actions    : escalate, investigate, suppress, monitor
             NOTE: refer_to_analyst is NOT a scorer action.
Factors    : travel_match, asset_criticality, threat_intel_enrichment,
             pattern_history, time_anomaly, device_trust
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gae.profile_scorer import ProfileScorer, ScoringResult, CentroidUpdate
from gae.kernels import L2Kernel, DiagonalKernel
from gae.kernel_selector import KernelSelector
from gae.calibration import (
    soc_calibration_profile,
    derive_theta_min,
    check_conservation,
)


# ── SOC domain constants ──────────────────────────────────────────────────────

SOC_CATEGORIES = [
    "credential_access",
    "lateral_movement",
    "data_exfiltration",
    "malware_execution",
    "insider_threat",
    "cloud_infrastructure",
]

SOC_ACTIONS = ["escalate", "investigate", "suppress", "monitor"]

N_CAT, N_ACT, N_FAC = 6, 4, 6


# ── Shared helpers ────────────────────────────────────────────────────────────

def make_soc_scorer(seed: int = 42) -> ProfileScorer:
    """ProfileScorer with SOC fixture tensor and default hyperparameters."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.1, 0.9, (N_CAT, N_ACT, N_FAC))
    return ProfileScorer(mu=mu, actions=SOC_ACTIONS, categories=SOC_CATEGORIES)


# ── SOC Tensor Configuration ──────────────────────────────────────────────────

class TestSOCTensorConfiguration:
    def test_soc_scorer_constructs_with_correct_dimensions(self):
        """ProfileScorer(6,4,6) constructs without error."""
        scorer = make_soc_scorer()
        assert scorer.n_categories == N_CAT
        assert scorer.n_actions == N_ACT
        assert scorer.n_factors == N_FAC

    def test_soc_centroids_shape_is_6_4_6(self):
        """Centroid tensor matches the SOC fixture dimensions."""
        scorer = make_soc_scorer()
        assert scorer.centroids.shape == (N_CAT, N_ACT, N_FAC)
        assert scorer.centroids.shape == (N_CAT, N_ACT, N_FAC)

    def test_travel_login_score_valid(self):
        """[0.25,0.5,0.0,0.4,0.7,1.0] — travel login alert scores without error."""
        scorer = make_soc_scorer()
        f = np.array([0.25, 0.5, 0.0, 0.4, 0.7, 1.0])
        result = scorer.score(f, category_index=0)  # credential_access
        assert isinstance(result, ScoringResult)
        assert result.probabilities.shape == (N_ACT,)
        assert abs(result.probabilities.sum() - 1.0) < 1e-9
        assert not np.any(np.isnan(result.probabilities))

    def test_insider_threat_score_valid(self):
        """[0.5,0.5,0.0,0.4,0.0,0.333] — insider threat alert scores correctly."""
        scorer = make_soc_scorer()
        f = np.array([0.5, 0.5, 0.0, 0.4, 0.0, 0.333])
        result = scorer.score(f, category_index=4)  # insider_threat
        assert isinstance(result, ScoringResult)
        assert abs(result.probabilities.sum() - 1.0) < 1e-9
        assert not np.any(np.isnan(result.probabilities))

    def test_data_exfil_score_valid(self):
        """[0.5,0.5,0.0,0.4,1.0,1.0] — data exfiltration alert scores correctly."""
        scorer = make_soc_scorer()
        f = np.array([0.5, 0.5, 0.0, 0.4, 1.0, 1.0])
        result = scorer.score(f, category_index=2)  # data_exfiltration
        assert isinstance(result, ScoringResult)
        assert abs(result.probabilities.sum() - 1.0) < 1e-9

    def test_score_output_is_4_actions_not_5(self):
        """Score always returns exactly 4 actions — refer_to_analyst is NOT a scorer action."""
        scorer = make_soc_scorer()
        result = scorer.score(np.full(N_FAC, 0.5), category_index=0)
        assert result.probabilities.shape == (4,)
        assert result.distances.shape == (4,)

    def test_all_6_category_indices_score_valid(self):
        """All 6 SOC category indices (0-5) produce valid ScoringResults."""
        scorer = make_soc_scorer()
        f = np.array([0.3, 0.6, 0.1, 0.5, 0.4, 0.8])
        for c in range(N_CAT):
            result = scorer.score(f, category_index=c)
            assert abs(result.probabilities.sum() - 1.0) < 1e-9, f"category {c} failed"
            assert 0 <= result.action_index < N_ACT

    def test_all_4_action_indices_are_valid_update_targets(self):
        """All 4 action indices (0-3) can be used in update() without error."""
        scorer = make_soc_scorer()
        f = np.full(N_FAC, 0.5)
        for a in range(N_ACT):
            result = scorer.update(f, category_index=0, action_index=a, correct=True)
            assert isinstance(result, CentroidUpdate)

    def test_action_index_4_raises(self):
        """Action index 4 (refer_to_analyst) does not exist — scorer must raise."""
        scorer = make_soc_scorer()
        with pytest.raises((IndexError, AssertionError, ValueError)):
            scorer.update(np.full(N_FAC, 0.5), category_index=0,
                          action_index=4, correct=True)


# ── Learning Loop Pattern ─────────────────────────────────────────────────────

class TestSOCLearningLoop:
    def test_100_iteration_score_update_cycle_no_crash(self):
        """score → update × 100 with SOC (6,4,6): completes without error."""
        scorer = make_soc_scorer()
        rng = np.random.default_rng(0)
        for _ in range(100):
            f = rng.uniform(0.0, 1.0, N_FAC)
            c = int(rng.integers(N_CAT))
            result = scorer.score(f, c)
            scorer.update(f, c, result.action_index, correct=True)
        assert scorer.decision_count == 100

    def test_100_correct_updates_centroid_converges(self):
        """100 correct updates toward fixed f: centroid moves closer to f."""
        scorer = make_soc_scorer(seed=1)
        f_target = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        dist_before = np.linalg.norm(scorer.centroids[0, 0, :] - f_target)
        for _ in range(100):
            scorer.update(f_target, category_index=0, action_index=0, correct=True)
        dist_after = np.linalg.norm(scorer.centroids[0, 0, :] - f_target)
        assert dist_after < dist_before

    def test_alternating_correct_incorrect_maintains_stability(self):
        """Alternating correct/incorrect for 200 steps: centroids stay finite."""
        scorer = make_soc_scorer()
        rng = np.random.default_rng(2)
        for i in range(200):
            f = rng.uniform(0.0, 1.0, N_FAC)
            correct = (i % 2 == 0)
            with np.errstate(all="ignore"):
                scorer.update(f, 0, 0, correct=correct,
                              gt_action_index=1 if not correct else None)
        assert not np.any(np.isnan(scorer.centroids))
        assert not np.any(np.isinf(scorer.centroids))

    def test_centroids_stay_in_01_after_500_iterations(self):
        """500 mixed updates with SOC (6,4,6): mu stays in [0,1] — V2 invariant."""
        scorer = make_soc_scorer()
        rng = np.random.default_rng(3)
        for _ in range(500):
            f = rng.uniform(0.0, 1.0, N_FAC)
            c = int(rng.integers(N_CAT))
            a = int(rng.integers(N_ACT))
            scorer.update(f, c, a, correct=True)
        assert scorer.centroids.min() >= 0.0
        assert scorer.centroids.max() <= 1.0

    def test_all_6_categories_evolve_independently(self):
        """Updates to category 0 must not modify category 5 centroids."""
        scorer = make_soc_scorer()
        mu_cat5_before = scorer.centroids[5, :, :].copy()
        for _ in range(100):
            scorer.update(np.full(N_FAC, 0.9), category_index=0,
                          action_index=0, correct=True)
        np.testing.assert_array_equal(scorer.centroids[5, :, :], mu_cat5_before)

    def test_mu_stays_finite_after_500_updates(self):
        """After 500 updates mu contains no NaN or Inf."""
        scorer = make_soc_scorer()
        rng = np.random.default_rng(4)
        for _ in range(500):
            f = rng.uniform(0.0, 1.0, N_FAC)
            c = int(rng.integers(N_CAT))
            a = int(rng.integers(N_ACT))
            scorer.update(f, c, a, correct=True)
        assert not np.any(np.isnan(scorer.centroids))
        assert not np.any(np.isinf(scorer.centroids))

    def test_score_still_deterministic_after_learning_loop(self):
        """After 200 updates, score() remains deterministic for same input."""
        scorer = make_soc_scorer()
        rng = np.random.default_rng(5)
        for _ in range(200):
            f = rng.uniform(0.0, 1.0, N_FAC)
            scorer.update(f, 0, 0, correct=True)
        f_test = np.array([0.3, 0.5, 0.7, 0.2, 0.4, 0.6])
        r1 = scorer.score(f_test, 0)
        r2 = scorer.score(f_test, 0)
        np.testing.assert_array_equal(r1.probabilities, r2.probabilities)

    def test_eta_confirm_pulls_centroid_toward_f(self):
        """Default η_confirm=0.05: one correct update pulls centroid toward f."""
        scorer = make_soc_scorer()
        f = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        dist_before = np.linalg.norm(scorer.centroids[0, 0, :] - f)
        scorer.update(f, category_index=0, action_index=0, correct=True)
        dist_after = np.linalg.norm(scorer.centroids[0, 0, :] - f)
        assert dist_after < dist_before

    def test_eta_override_gt_pull_smaller_than_confirm(self):
        """η_override=0.01: GT-pull magnitude is smaller than η_confirm=0.05 pull."""
        rng = np.random.default_rng(6)
        mu = rng.uniform(0.3, 0.7, (N_CAT, N_ACT, N_FAC))

        scorer_confirm  = ProfileScorer(mu=mu.copy(), actions=SOC_ACTIONS)
        scorer_override = ProfileScorer(mu=mu.copy(), actions=SOC_ACTIONS,
                                        eta_override=0.01)

        # Small f-mu gap (0.05/dim) so neither scorer hits the MAX_ETA_DELTA cap
        f = np.clip(mu[0, 1, :] + 0.05, 0.0, 1.0)

        mu1_gt_before = scorer_confirm.centroids[0, 1, :].copy()
        mu2_gt_before = scorer_override.centroids[0, 1, :].copy()

        # Confirm: pulls action-1 centroid toward f at rate η=0.05
        scorer_confirm.update(f, category_index=0, action_index=1, correct=True)
        # Override: pulls GT centroid (action-1) toward f at rate η_override=0.01
        scorer_override.update(f, category_index=0, action_index=0,
                               correct=False, gt_action_index=1)

        delta_confirm = np.linalg.norm(
            scorer_confirm.centroids[0, 1, :] - mu1_gt_before
        )
        delta_override = np.linalg.norm(
            scorer_override.centroids[0, 1, :] - mu2_gt_before
        )
        assert delta_override < delta_confirm, (
            f"η_override GT pull ({delta_override:.6f}) must be smaller "
            f"than η_confirm pull ({delta_confirm:.6f})"
        )


# ── Centroid Management Pattern ───────────────────────────────────────────────

class TestSOCCentroidManagement:
    def test_modify_centroids_changes_score(self):
        """Zeroing all centroids produces uniform scores (all-equal distances)."""
        scorer = make_soc_scorer()
        # With all centroids equal (all zeros), probabilities must be uniform
        scorer.centroids[:] = 0.0
        result = scorer.score(np.full(N_FAC, 0.5), category_index=0)
        assert abs(result.probabilities.sum() - 1.0) < 1e-9
        np.testing.assert_allclose(result.probabilities,
                                   np.full(N_ACT, 1.0 / N_ACT), atol=1e-9)

    def test_centroids_json_roundtrip_preserves_scores(self):
        """Serialize centroids to JSON list → restore → scores identical."""
        scorer = make_soc_scorer()
        f = np.array([0.3, 0.6, 0.1, 0.5, 0.4, 0.8])
        r_before = scorer.score(f, category_index=2)

        restored_mu = np.array(json.loads(json.dumps(scorer.centroids.tolist())))
        scorer2 = ProfileScorer(mu=restored_mu, actions=SOC_ACTIONS,
                                categories=SOC_CATEGORIES)
        r_after = scorer2.score(f, category_index=2)
        np.testing.assert_allclose(r_before.probabilities, r_after.probabilities,
                                   atol=1e-12)

    def test_different_centroids_produce_different_scores(self):
        """Scorers with different random centroids produce different scores."""
        rng1 = np.random.default_rng(10)
        rng2 = np.random.default_rng(99)
        mu1 = rng1.uniform(0.1, 0.9, (N_CAT, N_ACT, N_FAC))
        mu2 = rng2.uniform(0.1, 0.9, (N_CAT, N_ACT, N_FAC))
        s1 = ProfileScorer(mu=mu1, actions=SOC_ACTIONS)
        s2 = ProfileScorer(mu=mu2, actions=SOC_ACTIONS)
        f = np.full(N_FAC, 0.5)
        r1 = s1.score(f, 0)
        r2 = s2.score(f, 0)
        assert not np.allclose(r1.probabilities, r2.probabilities)

    def test_centroids_after_50_updates_restore_identical_behavior(self):
        """Extract centroids after 50 updates → restore into new scorer → identical scores."""
        scorer = make_soc_scorer()
        rng = np.random.default_rng(7)
        for _ in range(50):
            f = rng.uniform(0.0, 1.0, N_FAC)
            scorer.update(f, category_index=0, action_index=0, correct=True)

        mu_saved = scorer.centroids.copy()
        scorer2 = ProfileScorer(mu=mu_saved, actions=SOC_ACTIONS, categories=SOC_CATEGORIES)

        f_test = np.array([0.4, 0.7, 0.2, 0.5, 0.3, 0.9])
        for c in range(N_CAT):
            r1 = scorer.score(f_test, c)
            r2 = scorer2.score(f_test, c)
            np.testing.assert_array_equal(r1.probabilities, r2.probabilities,
                                          err_msg=f"category {c} mismatch after restore")


# ── Kernel Selection Pattern ──────────────────────────────────────────────────

class TestSOCKernelSelection:
    def test_kernel_selector_soc_6_factors_starts_l2(self):
        """KernelSelector with SOC fixture d and uniform sigma recommends L2 initially."""
        sigma = np.full(N_FAC, 0.3)  # uniform noise → noise_ratio=1.0 < 1.5
        ks = KernelSelector(d=N_FAC, sigma_per_factor=sigma, correlation_max=0.0)
        rec = ks.preliminary_recommendation()
        assert rec.recommended_kernel == "l2"
        assert rec.method == "rule"

    def test_diagonal_kernel_6d_produces_valid_soc_scores(self):
        """DiagonalKernel with 6 SOC factors: valid scores on SOC alerts."""
        sigma = np.array([0.2, 0.3, 0.5, 0.1, 0.4, 0.2])
        kernel = DiagonalKernel(sigma)
        scorer = make_soc_scorer()
        scorer.set_kernel(kernel)
        result = scorer.score(np.array([0.3, 0.6, 0.0, 0.5, 0.4, 0.8]),
                              category_index=0)
        assert result.probabilities.shape == (N_ACT,)
        assert abs(result.probabilities.sum() - 1.0) < 1e-9

    def test_l2_kernel_6d_produces_valid_soc_scores(self):
        """L2Kernel with 6 SOC factors: valid scores on SOC alerts."""
        scorer = make_soc_scorer()
        scorer.set_kernel(L2Kernel())
        result = scorer.score(np.full(N_FAC, 0.5), category_index=0)
        assert result.probabilities.shape == (N_ACT,)
        assert abs(result.probabilities.sum() - 1.0) < 1e-9

    def test_kernel_switch_valid_before_and_after(self):
        """Switching from L2 to DiagonalKernel mid-session: both produce valid scores."""
        scorer = make_soc_scorer()
        f = np.array([0.3, 0.6, 0.1, 0.5, 0.4, 0.8])

        r_before = scorer.score(f, category_index=0)
        assert abs(r_before.probabilities.sum() - 1.0) < 1e-9

        sigma = np.array([0.2, 0.3, 0.5, 0.1, 0.4, 0.2])
        scorer.set_kernel(DiagonalKernel(sigma))
        r_after = scorer.score(f, category_index=0)
        assert abs(r_after.probabilities.sum() - 1.0) < 1e-9

    def test_diagonal_kernel_produces_different_ranking_than_l2(self):
        """DiagonalKernel with very unequal sigma differs from L2 in action ranking."""
        rng = np.random.default_rng(20)
        mu = rng.uniform(0.1, 0.9, (N_CAT, N_ACT, N_FAC))

        scorer_l2 = ProfileScorer(mu=mu.copy(), actions=SOC_ACTIONS)
        # dim 1 (asset_criticality) 40× noisier → strongly downweighted in diagonal
        sigma_unequal = np.array([0.05, 2.0, 0.05, 0.05, 0.05, 0.05])
        scorer_diag = ProfileScorer(mu=mu.copy(), actions=SOC_ACTIONS,
                                    scoring_kernel=DiagonalKernel(sigma_unequal))
        f = rng.uniform(0.0, 1.0, N_FAC)
        r_l2   = scorer_l2.score(f, category_index=0)
        r_diag = scorer_diag.score(f, category_index=0)
        assert not np.allclose(r_l2.probabilities, r_diag.probabilities), (
            "DiagonalKernel with unequal sigma must produce different ranking than L2"
        )


# ── Bootstrap / Checkpoint Pattern ───────────────────────────────────────────

class TestSOCCheckpointPattern:
    def test_save_restore_centroids_same_behavior(self):
        """Save centroids (numpy copy) → restore into new scorer → scores identical."""
        scorer = make_soc_scorer()
        f_test = np.array([0.4, 0.7, 0.2, 0.5, 0.3, 0.9])
        r_before = scorer.score(f_test, category_index=0)

        mu_checkpoint = scorer.centroids.copy()
        scorer2 = ProfileScorer(mu=mu_checkpoint, actions=SOC_ACTIONS,
                                categories=SOC_CATEGORIES)
        r_after = scorer2.score(f_test, category_index=0)
        np.testing.assert_array_equal(r_before.probabilities, r_after.probabilities)

    def test_checkpoint_exact_float_equality(self):
        """Score before checkpoint == score after restore — exact float equality, not approx."""
        scorer = make_soc_scorer(seed=55)
        f_test = np.array([0.1, 0.9, 0.3, 0.7, 0.5, 0.2])
        r_before = scorer.score(f_test, category_index=3)

        scorer2 = ProfileScorer(mu=scorer.centroids.copy(), actions=SOC_ACTIONS)
        r_after = scorer2.score(f_test, category_index=3)
        np.testing.assert_array_equal(r_before.probabilities, r_after.probabilities)

    def test_update_after_restore_continues_learning(self):
        """Restored scorer continues learning correctly — not a reset."""
        scorer = make_soc_scorer()
        f = np.full(N_FAC, 0.8)
        for _ in range(20):
            scorer.update(f, category_index=0, action_index=0, correct=True)

        mu_checkpoint = scorer.centroids.copy()
        dist_at_checkpoint = np.linalg.norm(scorer.centroids[0, 0, :] - f)

        scorer2 = ProfileScorer(mu=mu_checkpoint, actions=SOC_ACTIONS)
        scorer2.update(f, category_index=0, action_index=0, correct=True)
        dist_after = np.linalg.norm(scorer2.centroids[0, 0, :] - f)
        assert dist_after < dist_at_checkpoint

    def test_checkpoint_with_10000_accumulated_updates(self):
        """Centroid checkpoint survives 10000 accumulated updates."""
        scorer = make_soc_scorer()
        rng = np.random.default_rng(8)
        for _ in range(10_000):
            f = rng.uniform(0.0, 1.0, N_FAC)
            c = int(rng.integers(N_CAT))
            a = int(rng.integers(N_ACT))
            scorer.update(f, c, a, correct=True)

        mu_saved = scorer.centroids.copy()
        scorer2 = ProfileScorer(mu=mu_saved, actions=SOC_ACTIONS)
        f_test = np.full(N_FAC, 0.5)
        r1 = scorer.score(f_test, category_index=0)
        r2 = scorer2.score(f_test, category_index=0)
        np.testing.assert_array_equal(r1.probabilities, r2.probabilities)


# ── Edge Cases from Production ────────────────────────────────────────────────

class TestSOCProductionEdgeCases:
    def test_zero_threat_intel_enrichment_score_valid(self):
        """threat_intel_enrichment=0.0 (no TI data available): score still valid."""
        scorer = make_soc_scorer()
        # factor order: [travel_match, asset_criticality, threat_intel_enrichment=0, ...]
        f = np.array([0.5, 0.8, 0.0, 0.6, 0.3, 0.9])
        result = scorer.score(f, category_index=0)
        assert result.probabilities.shape == (N_ACT,)
        assert abs(result.probabilities.sum() - 1.0) < 1e-9
        assert not np.any(np.isnan(result.probabilities))

    def test_all_zeros_cold_start_alert_no_crash(self):
        """Cold start alert (all factor values zero): no crash, valid distribution."""
        scorer = make_soc_scorer()
        f = np.zeros(N_FAC)
        result = scorer.score(f, category_index=0)
        assert result.probabilities.shape == (N_ACT,)
        assert abs(result.probabilities.sum() - 1.0) < 1e-9
        assert not np.any(np.isnan(result.probabilities))

    def test_fresh_scorer_returns_valid_distribution(self):
        """score() before any updates on fresh scorer: valid probability distribution."""
        scorer = make_soc_scorer()
        result = scorer.score(np.full(N_FAC, 0.5), category_index=0)
        assert abs(result.probabilities.sum() - 1.0) < 1e-9
        assert np.all(result.probabilities >= 0.0)
        assert np.all(result.probabilities <= 1.0)

    def test_multiple_score_calls_same_alert_idempotent(self):
        """score() is idempotent — repeated calls on same alert produce identical results."""
        scorer = make_soc_scorer()
        f = np.array([0.3, 0.6, 0.1, 0.5, 0.4, 0.8])
        count_before = scorer.decision_count
        r1 = scorer.score(f, category_index=0)
        r2 = scorer.score(f, category_index=0)
        r3 = scorer.score(f, category_index=0)
        assert scorer.decision_count == count_before  # score() must not increment count
        np.testing.assert_array_equal(r1.probabilities, r2.probabilities)
        np.testing.assert_array_equal(r2.probabilities, r3.probabilities)

    def test_repeated_same_f_update_centroid_converges_no_oscillation(self):
        """Same factor vector 100 times: centroid moves toward f monotonically."""
        scorer = make_soc_scorer()
        f = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        dist_before = np.linalg.norm(scorer.centroids[0, 0, :] - f)
        for _ in range(100):
            scorer.update(f, category_index=0, action_index=0, correct=True)
        dist_after = np.linalg.norm(scorer.centroids[0, 0, :] - f)
        assert dist_after < dist_before
        assert scorer.centroids.min() >= 0.0
        assert scorer.centroids.max() <= 1.0

    def test_two_soc_instances_completely_independent(self):
        """Two SOC scorer instances with identical config share no state."""
        rng = np.random.default_rng(15)
        mu = rng.uniform(0.1, 0.9, (N_CAT, N_ACT, N_FAC))
        s1 = ProfileScorer(mu=mu.copy(), actions=SOC_ACTIONS)
        s2 = ProfileScorer(mu=mu.copy(), actions=SOC_ACTIONS)
        mu2_before = s2.centroids.copy()

        for _ in range(100):
            s1.update(np.full(N_FAC, 0.9), category_index=0,
                      action_index=0, correct=True)

        np.testing.assert_array_equal(
            s2.centroids,
            mu2_before,
            err_msg="s2 must not be affected by s1 updates",
        )

    def test_category_0_score_unchanged_after_scoring_category_5(self):
        """Scoring category 5 must not affect category 0 (categories are independent)."""
        scorer = make_soc_scorer()
        f = np.array([0.4, 0.6, 0.3, 0.5, 0.7, 0.2])
        r_cat0_first = scorer.score(f, category_index=0)
        _ = scorer.score(f, category_index=5)  # cloud_infrastructure
        r_cat0_again = scorer.score(f, category_index=0)
        np.testing.assert_array_equal(
            r_cat0_first.probabilities, r_cat0_again.probabilities,
            err_msg="category 0 result must be unchanged after scoring category 5",
        )


# ── Conservation Monitor with SOC Parameters ─────────────────────────────────

class TestSOCConservationMonitor:
    def test_conservation_soc_realistic_parameters(self):
        """check_conservation with SOC-realistic alpha=0.33, V=10000 is GREEN."""
        theta = derive_theta_min()
        # Realistic SOC: 33% override rate, 84% quality, 10000 verified decisions/day
        cc = check_conservation(alpha=0.33, q=0.84, V=10000.0, theta_min=theta)
        # signal = 0.33 × 0.84 × 10000 = 2772 >> theta ≈ 0.467
        assert cc.status == "GREEN"
        assert cc.passed is True
        assert cc.signal == pytest.approx(0.33 * 0.84 * 10000.0, rel=1e-3)

    def test_theta_min_soc_deployment_parameters(self):
        """derive_theta_min with SOC defaults: η=0.05, N_half=14, T_max=21 days."""
        theta = derive_theta_min(eta=0.05, n_half=14.0, t_max_days=21.0)
        # θ_min = 0.05 × 196 / 21 ≈ 0.467
        assert theta == pytest.approx(0.05 * 14.0**2 / 21.0, rel=1e-6)
        assert theta > 0.4   # canonical SOC floor is ~0.467

    def test_conservation_signal_degrades_with_accuracy(self):
        """Conservation signal decreases monotonically as SOC accuracy drops 84→70→60%."""
        theta = derive_theta_min()
        alpha, V = 0.33, 10000.0

        cc_84 = check_conservation(alpha=alpha, q=0.84, V=V, theta_min=theta)
        cc_70 = check_conservation(alpha=alpha, q=0.70, V=V, theta_min=theta)
        cc_60 = check_conservation(alpha=alpha, q=0.60, V=V, theta_min=theta)

        # All still GREEN at V=10000 (far above threshold)
        assert cc_84.status == "GREEN"
        assert cc_70.status == "GREEN"
        assert cc_60.status == "GREEN"

        # Signal decreases monotonically with quality degradation
        assert cc_70.signal < cc_84.signal, "Signal must drop as quality drops 84→70"
        assert cc_60.signal < cc_70.signal, "Signal must drop as quality drops 70→60"
