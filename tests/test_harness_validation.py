"""
tests/test_harness_validation.py

Harness validation suite — converts a validated Colab notebook
(7 suites, zero failures) into pytest form.

Reference results (notebook run, GAE 0.7.23):
  Suite A: variance_argmax=5, variance_argmin=2
  Suite B: accuracy_100=0.64, conservation=GREEN
  Suite C: dist_self=0.0, disrupted=[0,4]
  Suite D: phase1_n_half=25, phase1_dnf=false
  Suite E: calibration_period=50, yellow_fires
  Suite F: n_half_eta005=13.51
  Suite G: acc_pre_mean=0.8, gamma_proxy=1.0
"""

import pytest
import numpy as np

from gae.profile_scorer import ProfileScorer
from gae.kernels import DiagonalKernel, L2Kernel
from gae.synthetic import (
    FactorVectorSampler,
    CanonicalCentroid,
    OracleSeparationExperiment,
    centroid_distance_to_canonical,
)
from gae.oracle import GTAlignedOracle
from gae.convergence import (
    ConservationMonitor,
    compute_n_half,
    gamma_threshold,
    CALIBRATION_PERIOD,
)

# ── Domain constants (SOC, 6×4×6) ────────────────────────────────────────────

C, A, d = 6, 4, 6

CATEGORIES = [
    "credential_access",
    "lateral_movement",
    "malware_execution",
    "data_exfiltration",
    "insider_threat",
    "cloud_infrastructure",
]
ACTIONS = ["investigate", "escalate", "dismiss", "monitor"]
FACTOR_NAMES = [
    "travel_match",
    "asset_criticality",
    "threat_intel_enrichment",
    "pattern_history",
    "time_anomaly",
    "device_trust",
]
SIGMA = [0.18, 0.12, 0.07, 0.15, 0.20, 0.28]


# ── Suite A: FactorVectorSampler ──────────────────────────────────────────────

class TestSuiteA_SamplerValidation:

    def test_all_regime_strings_work(self):
        """All regime strings produce valid samples."""
        sampler = FactorVectorSampler(d=6, sigma_profile=SIGMA, seed=42)
        for regime in ["normal", "cold_start", "post_disruption",
                       "enriched", "high_confidence", "low_confidence",
                       "adversarial"]:
            samples = sampler.sample(regime=regime, n=5)
            assert len(samples) == 5
            assert all(hasattr(s, "f") for s in samples)

    def test_mean_offset_shifts_distribution(self):
        """mean_offset=+0.25 produces higher mean than no offset."""
        sampler = FactorVectorSampler(d=6, sigma_profile=SIGMA, seed=42)
        no_offset = sampler.sample(regime="normal", n=50)
        with_offset = sampler.sample(regime="normal", n=50,
                                     mean_offset=np.full(6, 0.25))
        mean_no = np.array([s.f for s in no_offset]).mean()
        mean_off = np.array([s.f for s in with_offset]).mean()
        assert mean_off > mean_no + 0.05

    def test_all_values_in_0_1_range(self):
        """All factor values clipped to [0, 1]."""
        sampler = FactorVectorSampler(d=6, sigma_profile=SIGMA, seed=42)
        samples = sampler.sample(regime="normal", n=100,
                                 mean_offset=np.full(6, 0.25))
        for s in samples:
            assert all(0.0 <= v <= 1.0 for v in s.f)

    def test_variance_argmax_is_device_trust(self):
        """Highest variance factor is device_trust (idx 5, σ=0.28)."""
        sampler = FactorVectorSampler(d=6, sigma_profile=SIGMA, seed=42)
        samples = sampler.sample(regime="normal", n=50)
        fv = np.array([s.f for s in samples])
        variances = np.var(fv, axis=0)
        assert np.argmax(variances) == 5

    def test_variance_argmin_is_threat_intel(self):
        """Lowest variance factor is threat_intel (idx 2, σ=0.07)."""
        sampler = FactorVectorSampler(d=6, sigma_profile=SIGMA, seed=42)
        samples = sampler.sample(regime="normal", n=50)
        fv = np.array([s.f for s in samples])
        variances = np.var(fv, axis=0)
        assert np.argmin(variances) == 2

    def test_sample_has_required_fields(self):
        """FactorVectorSample has f, regime, sigma_per_factor, generation_seed."""
        sampler = FactorVectorSampler(d=6, sigma_profile=SIGMA, seed=42)
        s = sampler.sample(regime="normal", n=1)[0]
        for field in ["f", "regime", "sigma_per_factor", "generation_seed"]:
            assert hasattr(s, field)


# ── Suite B: GTAlignedOracle + Scoring Loop ───────────────────────────────────

class TestSuiteB_EndToEndScoring:

    def _make_scorer_and_oracle(self):
        rng = np.random.default_rng(42)
        gt = np.clip(rng.normal(0.5, 0.1, (C, A, d)), 0.1, 0.9)
        offset = rng.uniform(-0.05, 0.05, gt.shape)
        mu_init = np.clip(gt + offset, 0.05, 0.95)
        scorer = ProfileScorer(
            mu=mu_init.copy(), actions=ACTIONS, categories=CATEGORIES,
            eta_override=0.01, min_confidence=0.0,
            scoring_kernel=DiagonalKernel(sigma=np.array(SIGMA)),
            auto_pause_on_amber=False,
        )
        scorer.eta = 0.05
        scorer.decay = 0.0
        oracle = GTAlignedOracle(mu=gt, actions=ACTIONS)
        sampler = FactorVectorSampler(d=d, sigma_profile=SIGMA, seed=42)
        return scorer, oracle, sampler, gt

    @pytest.mark.slow
    def test_accuracy_in_expected_range(self):
        """100 decisions produce 50-90% accuracy."""
        scorer, oracle, sampler, gt = self._make_scorer_and_oracle()
        correct = 0
        for i in range(100):
            cat = i % C
            f = sampler.sample(regime="normal", n=1,
                               mean_offset=gt[cat, 0] - 0.5)[0].f
            r = scorer.score(f, cat)
            o = oracle.query(f=f, category_index=cat,
                             taken_action_index=r.action_index)
            correct += int(o.correct)
            scorer.update(f, cat, r.action_index, correct=o.correct,
                          gt_action_index=o.gt_action_idx if not o.correct else None)
        accuracy = correct / 100
        assert 0.50 <= accuracy <= 0.90

    def test_confidence_below_cold_start_ceiling(self):
        """conf_max < 0.55 at N=50 (cold-start ceiling is ~1/A)."""
        scorer, oracle, sampler, gt = self._make_scorer_and_oracle()
        confidences = []
        for i in range(50):
            cat = i % C
            f = sampler.sample(regime="normal", n=1,
                               mean_offset=gt[cat, 0] - 0.5)[0].f
            r = scorer.score(f, cat)
            confidences.append(r.confidence)
            o = oracle.query(f=f, category_index=cat,
                             taken_action_index=r.action_index)
            scorer.update(f, cat, r.action_index, correct=o.correct,
                          gt_action_index=o.gt_action_idx if not o.correct else None)
        assert max(confidences) < 0.55

    def test_centroid_distance_decreases(self):
        """Centroids move toward GT (dist_after < dist_before)."""
        scorer, oracle, sampler, gt = self._make_scorer_and_oracle()
        dist_before = centroid_distance_to_canonical(scorer.mu, gt)
        for i in range(50):
            cat = i % C
            f = sampler.sample(regime="normal", n=1,
                               mean_offset=gt[cat, 0] - 0.5)[0].f
            r = scorer.score(f, cat)
            o = oracle.query(f=f, category_index=cat,
                             taken_action_index=r.action_index)
            scorer.update(f, cat, r.action_index, correct=o.correct,
                          gt_action_index=o.gt_action_idx if not o.correct else None)
        dist_after = centroid_distance_to_canonical(scorer.mu, gt)
        assert dist_after < dist_before

    def test_conservation_green_after_scoring(self):
        """Conservation status stays GREEN during healthy scoring."""
        scorer, oracle, sampler, gt = self._make_scorer_and_oracle()
        for i in range(50):
            cat = i % C
            f = sampler.sample(regime="normal", n=1,
                               mean_offset=gt[cat, 0] - 0.5)[0].f
            r = scorer.score(f, cat)
            o = oracle.query(f=f, category_index=cat,
                             taken_action_index=r.action_index)
            scorer.update(f, cat, r.action_index, correct=o.correct,
                          gt_action_index=o.gt_action_idx if not o.correct else None)
        assert scorer.conservation_status == "GREEN"


# ── Suite C: CanonicalCentroid + Disruption ───────────────────────────────────

class TestSuiteC_DisruptionMechanics:

    def test_distance_from_self_is_zero(self):
        rng = np.random.default_rng(99)
        gt = np.clip(rng.normal(0.5, 0.1, (C, A, d)), 0.1, 0.9)
        cc = CanonicalCentroid(gt=gt)
        assert abs(cc.distance_from(gt)) < 1e-10

    def test_disruption_changes_only_specified_categories(self):
        rng = np.random.default_rng(99)
        gt = np.clip(rng.normal(0.5, 0.1, (C, A, d)), 0.1, 0.9)
        cc = CanonicalCentroid(gt=gt)
        delta = np.full((A, d), 0.20)
        cc2 = cc.apply_disruption(delta=delta, categories=[0, 4])
        for cat in [1, 2, 3, 5]:
            assert np.allclose(cc.gt[cat], cc2.gt[cat])
        for cat in [0, 4]:
            assert not np.allclose(cc.gt[cat], cc2.gt[cat])

    def test_gt1_gt2_distance_meaningful(self):
        rng = np.random.default_rng(99)
        gt = np.clip(rng.normal(0.5, 0.1, (C, A, d)), 0.1, 0.9)
        cc = CanonicalCentroid(gt=gt)
        delta = np.full((A, d), 0.20)
        cc2 = cc.apply_disruption(delta=delta, categories=[0, 4])
        assert cc.distance_from(cc2.gt) > 0.10

    def test_gt2_values_in_valid_range(self):
        rng = np.random.default_rng(99)
        gt = np.clip(rng.normal(0.5, 0.1, (C, A, d)), 0.1, 0.9)
        cc = CanonicalCentroid(gt=gt)
        delta = np.full((A, d), 0.20)
        cc2 = cc.apply_disruption(delta=delta, categories=[0, 4])
        assert cc2.gt.min() >= 0.0
        assert cc2.gt.max() <= 1.0


# ── Suite D: OracleSeparationExperiment ──────────────────────────────────────

class TestSuiteD_OracleSeparation:

    @pytest.mark.slow
    def test_phase1_converges(self):
        """Phase 1 finds n_half (not DNF) with 200 samples."""
        rng = np.random.default_rng(7)
        gt = np.clip(rng.normal(0.5, 0.1, (C, A, d)), 0.1, 0.9)
        offset = rng.uniform(-0.05, 0.05, gt.shape)
        mu_init = np.clip(gt + offset, 0.05, 0.95)
        scorer = ProfileScorer(
            mu=mu_init.copy(), actions=ACTIONS, categories=CATEGORIES,
            eta_override=0.01, min_confidence=0.0,
            scoring_kernel=DiagonalKernel(sigma=np.array(SIGMA)),
            auto_pause_on_amber=False,
        )
        scorer.eta = 0.05
        scorer.decay = 0.0
        cc = CanonicalCentroid(gt=gt)
        sampler = FactorVectorSampler(d=d, sigma_profile=SIGMA, seed=7)
        ose = OracleSeparationExperiment(
            scorer=scorer, canonical_gt1=cc,
            epsilon_firm=0.20, disruption_magnitude=0.15,
            disrupted_categories=[0, 4],
        )
        samples = sampler.sample(regime="normal", n=200,
                                 mean_offset=gt[0, 0] - 0.5)
        ph1 = ose.run_phase1(factor_samples=samples)
        assert hasattr(ph1, "n_half")
        assert hasattr(ph1, "dnf")
        assert not ph1.dnf

    def test_ose_threshold_positive(self):
        rng = np.random.default_rng(7)
        gt = np.clip(rng.normal(0.5, 0.1, (C, A, d)), 0.1, 0.9)
        scorer = ProfileScorer(
            mu=gt.copy(), actions=ACTIONS, categories=CATEGORIES,
            eta_override=0.01, min_confidence=0.0,
            scoring_kernel=DiagonalKernel(sigma=np.array(SIGMA)),
            auto_pause_on_amber=False,
        )
        cc = CanonicalCentroid(gt=gt)
        ose = OracleSeparationExperiment(
            scorer=scorer, canonical_gt1=cc,
            epsilon_firm=0.20, disruption_magnitude=0.15,
            disrupted_categories=[0, 4],
        )
        assert ose.threshold > 0


# ── Suite E: ConservationMonitor ──────────────────────────────────────────────

class TestSuiteE_ConservationLaw:

    def test_baseline_not_set_initially(self):
        scorer = ProfileScorer(
            mu=np.ones((C, A, d)) * 0.5, actions=ACTIONS,
            categories=CATEGORIES, eta_override=0.01,
            min_confidence=0.0,
            scoring_kernel=DiagonalKernel(sigma=np.array(SIGMA)),
            auto_pause_on_amber=True,
        )
        cm = ConservationMonitor(scorer=scorer)
        assert not cm.baseline_set

    def test_baseline_set_after_calibration_period(self):
        scorer = ProfileScorer(
            mu=np.ones((C, A, d)) * 0.5, actions=ACTIONS,
            categories=CATEGORIES, eta_override=0.01,
            min_confidence=0.0,
            scoring_kernel=DiagonalKernel(sigma=np.array(SIGMA)),
            auto_pause_on_amber=True,
        )
        scorer.eta = 0.05
        scorer.decay = 0.0
        cm = ConservationMonitor(scorer=scorer)
        for _ in range(CALIBRATION_PERIOD):
            cm.record_quality(0.85)
        assert cm.baseline_set
        assert abs(cm.q_baseline - 0.85) < 0.01

    def test_yellow_warning_on_sustained_decline(self):
        scorer = ProfileScorer(
            mu=np.ones((C, A, d)) * 0.5, actions=ACTIONS,
            categories=CATEGORIES, eta_override=0.01,
            min_confidence=0.0,
            scoring_kernel=DiagonalKernel(sigma=np.array(SIGMA)),
            auto_pause_on_amber=True,
        )
        scorer.eta = 0.05
        scorer.decay = 0.0
        cm = ConservationMonitor(scorer=scorer)
        for _ in range(CALIBRATION_PERIOD):
            cm.record_quality(0.85)
        fired = False
        for i in range(400):
            cm.record_quality(0.30)
            if cm.yellow_warning:
                fired = True
                break
        assert fired, "CUSUM yellow_warning never fired"

    def test_amber_pauses_scorer(self):
        scorer = ProfileScorer(
            mu=np.ones((C, A, d)) * 0.5, actions=ACTIONS,
            categories=CATEGORIES, eta_override=0.01,
            min_confidence=0.0,
            scoring_kernel=DiagonalKernel(sigma=np.array(SIGMA)),
            auto_pause_on_amber=True,
        )
        cm = ConservationMonitor(scorer=scorer)
        cm.update_conservation_signal("AMBER")
        assert scorer.conservation_status == "AMBER"
        assert scorer.is_paused

    def test_green_unpauses_scorer(self):
        scorer = ProfileScorer(
            mu=np.ones((C, A, d)) * 0.5, actions=ACTIONS,
            categories=CATEGORIES, eta_override=0.01,
            min_confidence=0.0,
            scoring_kernel=DiagonalKernel(sigma=np.array(SIGMA)),
            auto_pause_on_amber=True,
        )
        cm = ConservationMonitor(scorer=scorer)
        cm.update_conservation_signal("AMBER")
        assert scorer.is_paused
        cm.update_conservation_signal("GREEN")
        assert not scorer.is_paused

    def test_centroids_frozen_while_paused(self):
        scorer = ProfileScorer(
            mu=np.ones((C, A, d)) * 0.5, actions=ACTIONS,
            categories=CATEGORIES, eta_override=0.01,
            min_confidence=0.0,
            scoring_kernel=DiagonalKernel(sigma=np.array(SIGMA)),
            auto_pause_on_amber=True,
        )
        scorer.eta = 0.05
        cm = ConservationMonitor(scorer=scorer)
        cm.update_conservation_signal("AMBER")
        mu_before = scorer.mu.copy()
        sampler = FactorVectorSampler(d=d, sigma_profile=SIGMA, seed=0)
        f = sampler.sample(regime="normal", n=1)[0].f
        scorer.update(f, 0, 0, correct=False, gt_action_index=1)
        assert np.allclose(mu_before, scorer.mu)


# ── Suite F: Convergence Math ─────────────────────────────────────────────────

class TestSuiteF_ConvergenceMath:

    def test_n_half_positive(self):
        n_half = compute_n_half(eta=0.05)
        assert isinstance(n_half, float)
        assert n_half > 0

    def test_n_half_in_expected_range(self):
        """N_half at η=0.05 should be ~13-15."""
        n_half = compute_n_half(eta=0.05)
        assert 10.0 < n_half < 20.0

    def test_n_half_default_matches_explicit(self):
        assert abs(compute_n_half() - compute_n_half(eta=0.05)) < 1e-10

    def test_gamma_threshold_positive(self):
        thresh = gamma_threshold(alpha_cat=2 / 6, delta_norm=0.20, theta=0.85)
        assert thresh > 0

    def test_centroid_distance_zero_for_identical(self):
        mu = np.zeros((C, A, d))
        assert abs(centroid_distance_to_canonical(mu, mu)) < 1e-10

    def test_centroid_distance_positive_for_offset(self):
        mu = np.full((C, A, d), 0.1)
        canonical = np.zeros((C, A, d))
        assert centroid_distance_to_canonical(mu, canonical) > 0


# ── Suite G: End-to-End Disruption Pattern ────────────────────────────────────

class TestSuiteG_DisruptionReconvergence:

    @pytest.mark.slow
    def test_pre_disruption_accuracy_above_50_percent(self):
        rng = np.random.default_rng(42)
        gt = np.clip(rng.normal(0.5, 0.1, (C, A, d)), 0.1, 0.9)
        offset = rng.uniform(-0.05, 0.05, gt.shape)
        mu_init = np.clip(gt + offset, 0.05, 0.95)
        scorer = ProfileScorer(
            mu=mu_init.copy(), actions=ACTIONS, categories=CATEGORIES,
            eta_override=0.01, min_confidence=0.0,
            scoring_kernel=DiagonalKernel(sigma=np.array(SIGMA)),
            auto_pause_on_amber=False,
        )
        scorer.eta = 0.05
        scorer.decay = 0.0
        oracle = GTAlignedOracle(mu=gt, actions=ACTIONS)
        sampler = FactorVectorSampler(d=d, sigma_profile=SIGMA, seed=42)
        correct = 0
        for i in range(50):
            cat = i % C
            f = sampler.sample(regime="normal", n=1,
                               mean_offset=gt[cat, 0] - 0.5)[0].f
            r = scorer.score(f, cat)
            o = oracle.query(f=f, category_index=cat,
                             taken_action_index=r.action_index)
            correct += int(o.correct)
            scorer.update(f, cat, r.action_index, correct=o.correct,
                          gt_action_index=o.gt_action_idx if not o.correct else None)
        assert correct / 50 > 0.50

    @pytest.mark.slow
    def test_post_disruption_distance_increases(self):
        """After disruption, centroid distance to GT2 starts high."""
        rng = np.random.default_rng(42)
        gt1 = np.clip(rng.normal(0.5, 0.1, (C, A, d)), 0.1, 0.9)
        offset = rng.uniform(-0.05, 0.05, gt1.shape)
        mu_init = np.clip(gt1 + offset, 0.05, 0.95)
        scorer = ProfileScorer(
            mu=mu_init.copy(), actions=ACTIONS, categories=CATEGORIES,
            eta_override=0.01, min_confidence=0.0,
            scoring_kernel=DiagonalKernel(sigma=np.array(SIGMA)),
            auto_pause_on_amber=False,
        )
        scorer.eta = 0.05
        scorer.decay = 0.0
        oracle1 = GTAlignedOracle(mu=gt1, actions=ACTIONS)
        sampler = FactorVectorSampler(d=d, sigma_profile=SIGMA, seed=42)
        for i in range(50):
            cat = i % C
            f = sampler.sample(regime="normal", n=1,
                               mean_offset=gt1[cat, 0] - 0.5)[0].f
            r = scorer.score(f, cat)
            o = oracle1.query(f=f, category_index=cat,
                              taken_action_index=r.action_index)
            scorer.update(f, cat, r.action_index, correct=o.correct,
                          gt_action_index=o.gt_action_idx if not o.correct else None)
        dist_pre = centroid_distance_to_canonical(scorer.mu, gt1)
        cc = CanonicalCentroid(gt=gt1)
        delta = np.full((A, d), 0.20)
        cc2 = cc.apply_disruption(delta=delta, categories=[0, 4])
        gt2 = cc2.gt
        dist_post = centroid_distance_to_canonical(scorer.mu, gt2)
        assert dist_post > dist_pre


# ── Bonus: Conservation Parameter Validation (F2 finding) ────────────────────

class TestConservationParameters:
    """Validates F2: conservation law at production scale."""

    def _compute_status(self, q, alpha, V):
        theta_min = 23.53 / (alpha * V)
        signal = alpha * q * V
        if signal >= theta_min * 1.5:
            return "GREEN"
        elif signal >= theta_min:
            return "AMBER"
        else:
            return "RED"

    def test_v200_alpha005_healthy_is_green(self):
        """V=200, α=0.05, q=0.85 → GREEN."""
        assert self._compute_status(0.85, 0.05, 200) == "GREEN"

    def test_v200_alpha005_degraded_is_red(self):
        """V=200, α=0.05, q=0.20 → RED."""
        assert self._compute_status(0.20, 0.05, 200) == "RED"

    def test_v20_alpha025_healthy_is_not_green(self):
        """V=20, α=0.25, q=0.85 → NOT GREEN (confirms F2)."""
        status = self._compute_status(0.85, 0.25, 20)
        assert status != "GREEN"
