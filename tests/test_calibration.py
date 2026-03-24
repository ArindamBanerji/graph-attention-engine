"""Tests for gae.calibration — CalibrationProfile, factory functions, conservation law."""

import numpy as np
import pytest

from gae.calibration import (
    CalibrationProfile,
    ConservationCheck,
    check_conservation,
    check_meta_conservation,
    compute_breach_window,
    compute_eta_override,
    compute_factor_mask,
    compute_optimal_tau,
    compute_transfer_prior,
    derive_theta_min,
    mask_to_array,
    s2p_calibration_profile,
    soc_calibration_profile,
)
from gae.learning import LearningState
from gae.scoring import score_entity


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

W2x3 = np.array([[0.3, -0.1, 0.2], [-0.1, 0.2, -0.3]], dtype=np.float64)
F1x3 = np.array([[0.7, 0.4, 0.9]])
FACTOR_NAMES = ["f0", "f1", "f2"]
ACTIONS = ["act_a", "act_b"]


# ---------------------------------------------------------------------------
# Test 1: Default CalibrationProfile creates valid parameters
# ---------------------------------------------------------------------------

def test_default_profile_valid_parameters():
    """Default CalibrationProfile must pass its own validate() with no warnings."""
    profile = CalibrationProfile()
    warnings = profile.validate()
    assert warnings == [], f"Default profile should be valid, got: {warnings}"
    assert profile.learning_rate == pytest.approx(0.02)
    assert profile.penalty_ratio == pytest.approx(20.0)
    assert profile.temperature == pytest.approx(0.25)
    assert profile.epsilon_default == pytest.approx(0.001)
    assert profile.discount_strength == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 2: soc_calibration_profile() has penalty_ratio=20.0
# ---------------------------------------------------------------------------

def test_soc_profile_penalty_ratio():
    """soc_calibration_profile() must have penalty_ratio=20.0."""
    profile = soc_calibration_profile()
    assert isinstance(profile, CalibrationProfile)
    assert profile.penalty_ratio == pytest.approx(20.0)
    assert profile.validate() == [], "SOC profile should be valid"


# ---------------------------------------------------------------------------
# Test 3: s2p_calibration_profile() has penalty_ratio=5.0
# ---------------------------------------------------------------------------

def test_s2p_profile_penalty_ratio():
    """s2p_calibration_profile() must have penalty_ratio=5.0."""
    profile = s2p_calibration_profile()
    assert isinstance(profile, CalibrationProfile)
    assert profile.penalty_ratio == pytest.approx(5.0)
    assert profile.validate() == [], "S2P profile should be valid"


# ---------------------------------------------------------------------------
# Test 4: validate() warns on out-of-range parameters
# ---------------------------------------------------------------------------

def test_validate_warns_on_out_of_range():
    """validate() returns one warning per out-of-range parameter."""
    # learning_rate too low, penalty_ratio too high, temperature too low
    bad = CalibrationProfile(
        learning_rate=0.0001,   # below [0.001, 0.5]
        penalty_ratio=200.0,    # above [1.0, 100.0]
        temperature=0.01,       # below [0.05, 2.0]
        discount_strength=1.5,  # above [0.0, 1.0]
    )
    warnings = bad.validate()
    assert len(warnings) == 4
    # All warnings are non-empty strings
    for w in warnings:
        assert isinstance(w, str) and len(w) > 0

    # A profile with only one bad param gives exactly one warning
    one_bad = CalibrationProfile(learning_rate=0.999)
    assert len(one_bad.validate()) == 1


# ---------------------------------------------------------------------------
# Test 5: LearningState with soc profile uses profile.learning_rate
# ---------------------------------------------------------------------------

def test_learning_state_uses_profile_learning_rate():
    """LearningState with soc profile applies profile.learning_rate in Eq. 4b."""
    soc = soc_calibration_profile()   # learning_rate=0.02
    s2p = s2p_calibration_profile()   # learning_rate=0.01

    W = np.zeros((2, 3), dtype=np.float64)

    state_soc = LearningState(W=W.copy(), n_actions=2, n_factors=3,
                               factor_names=FACTOR_NAMES[:], profile=soc)
    state_s2p = LearningState(W=W.copy(), n_actions=2, n_factors=3,
                               factor_names=FACTOR_NAMES[:], profile=s2p)

    u_soc = state_soc.update(0, "act_a", +1, F1x3)
    u_s2p = state_s2p.update(0, "act_a", +1, F1x3)

    # SOC alpha_effective should be 2× s2p alpha_effective (0.02 vs 0.01)
    assert u_soc.alpha_effective == pytest.approx(soc.learning_rate)
    assert u_s2p.alpha_effective == pytest.approx(s2p.learning_rate)
    ratio = u_soc.alpha_effective / u_s2p.alpha_effective
    assert ratio == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Test 6: score_entity uses profile temperature
# ---------------------------------------------------------------------------

def test_score_entity_uses_profile_temperature():
    """score_entity with profile= uses profile.temperature, stored in result."""
    profile_sharp = CalibrationProfile(temperature=0.1)
    profile_soft = CalibrationProfile(temperature=1.5)

    result_sharp = score_entity(F1x3, W2x3, ACTIONS, profile=profile_sharp)
    result_soft = score_entity(F1x3, W2x3, ACTIONS, profile=profile_soft)

    # temperature attribute on ScoringResult must match profile
    assert result_sharp.temperature == pytest.approx(0.1)
    assert result_soft.temperature == pytest.approx(1.5)

    # sharper temperature → higher confidence
    assert result_sharp.confidence > result_soft.confidence


# ---------------------------------------------------------------------------
# Test 7: build_epsilon_vector returns correct per-factor rates
# ---------------------------------------------------------------------------

def test_build_epsilon_vector_per_factor_rates():
    """build_epsilon_vector() assigns profile-mapped rates to each factor."""
    profile = CalibrationProfile(
        factor_decay_classes={
            "stable_factor": "permanent",
            "fleeting_factor": "campaign",
            # "standard_factor" is absent → defaults to "standard"
        }
    )
    names = ["stable_factor", "fleeting_factor", "standard_factor"]
    W = np.zeros((2, 3), dtype=np.float64)
    state = LearningState(W=W, n_actions=2, n_factors=3,
                          factor_names=names, profile=profile)

    eps = state.build_epsilon_vector()

    assert eps.shape == (3,), f"Expected shape (3,), got {eps.shape}"
    assert eps[0] == pytest.approx(profile.decay_class_rates["permanent"])  # 0.0001
    assert eps[1] == pytest.approx(profile.decay_class_rates["campaign"])   # 0.005
    assert eps[2] == pytest.approx(profile.decay_class_rates["standard"])   # 0.001


# ---------------------------------------------------------------------------
# Test 8: Unmapped factor defaults to "standard" rate
# ---------------------------------------------------------------------------

def test_unmapped_factor_defaults_to_standard_rate():
    """A factor not in factor_decay_classes gets the "standard" class rate."""
    profile = CalibrationProfile()  # factor_decay_classes is empty
    names = ["unknown_factor"]
    W = np.zeros((2, 1), dtype=np.float64)
    state = LearningState(W=W, n_actions=2, n_factors=1,
                          factor_names=names, profile=profile)

    eps = state.build_epsilon_vector()

    assert eps.shape == (1,)
    assert eps[0] == pytest.approx(profile.decay_class_rates["standard"])


# ---------------------------------------------------------------------------
# Test 9: campaign-class factor decays faster than permanent-class
# ---------------------------------------------------------------------------

def test_campaign_decays_faster_than_permanent():
    """Permanent-class column retains more weight than campaign-class after updates."""
    profile = CalibrationProfile(
        factor_decay_classes={
            "device_trust": "permanent",       # ε = 0.0001
            "threat_intel": "campaign",        # ε = 0.005
        }
    )
    names = ["device_trust", "threat_intel"]
    W_init = np.ones((2, 2), dtype=np.float64)
    state = LearningState(W=W_init.copy(), n_actions=2, n_factors=2,
                          factor_names=names, profile=profile)

    # Apply 100 updates with zero f so no learning — only decay runs
    f_zero = np.zeros((1, 2))
    for _ in range(100):
        state.update(0, "act_a", +1, f_zero)

    # Permanent column (ε=0.0001): value ≈ (1-0.0001)^100 ≈ 0.990
    # Campaign column  (ε=0.005):  value ≈ (1-0.005)^100  ≈ 0.606
    permanent_mean = float(np.mean(np.abs(state.W[:, 0])))
    campaign_mean = float(np.mean(np.abs(state.W[:, 1])))
    assert permanent_mean > campaign_mean, (
        f"Permanent ({permanent_mean:.4f}) should retain more weight than "
        f"campaign ({campaign_mean:.4f}) after 100 decay steps"
    )


# ---------------------------------------------------------------------------
# Test 10: Default CalibrationProfile (no factor_decay_classes) → all "standard"
# ---------------------------------------------------------------------------

def test_default_profile_all_factors_get_standard_rate():
    """Default profile has empty factor_decay_classes — all factors get standard rate."""
    profile = CalibrationProfile()
    assert profile.factor_decay_classes == {}, "Default should have empty factor_decay_classes"

    names = ["alpha", "beta", "gamma"]
    W = np.zeros((2, 3), dtype=np.float64)
    state = LearningState(W=W, n_actions=2, n_factors=3,
                          factor_names=names, profile=profile)

    eps = state.build_epsilon_vector()

    standard_rate = profile.decay_class_rates["standard"]
    np.testing.assert_allclose(eps, np.full(3, standard_rate),
                               err_msg="All unmapped factors should get standard rate")


# ---------------------------------------------------------------------------
# Conservation law tests (research_note_v3)
# ---------------------------------------------------------------------------

class TestDeriveTheta:
    def test_soc_default(self):
        """SOC: η=0.05, N_half=14, T_max=21 → θ_min ≈ 0.467."""
        theta = derive_theta_min(0.05, 14.0, 21.0)
        assert abs(theta - 0.467) < 0.01

    def test_s2p_lower_than_soc(self):
        """S2P longer cycle → lower θ_min."""
        theta_soc = derive_theta_min(0.05, 13.51, 21.0)
        theta_s2p = derive_theta_min(0.05, 13.51, 26.0)
        assert theta_s2p < theta_soc


class TestCheckConservation:
    _THETA = derive_theta_min(0.05, 13.51, 21.0)

    def test_green(self):
        """Healthy signal → GREEN."""
        result = check_conservation(0.30, 0.80, 5.0, self._THETA)
        assert result.status == 'GREEN'
        assert result.passed is True
        assert result.headroom > 2.0

    def test_amber(self):
        """Thinning signal → AMBER."""
        result = check_conservation(0.15, 0.80, 5.0, self._THETA)
        assert result.status == 'AMBER'
        assert result.passed is True

    def test_red(self):
        """Breached signal → RED."""
        result = check_conservation(0.05, 0.80, 5.0, self._THETA)
        assert result.status == 'RED'
        assert result.passed is False

    def test_automation_complacency(self):
        """α=0.09 (70% auto-approve) → signal 0.36 < θ_min → RED."""
        result = check_conservation(0.09, 0.80, 5.0, self._THETA)
        assert result.status == 'RED'
        assert result.passed is False

    def test_is_namedtuple(self):
        """ConservationCheck is accessible by attribute name."""
        result = check_conservation(0.30, 0.80, 5.0, self._THETA)
        assert hasattr(result, 'signal')
        assert hasattr(result, 'status')
        assert hasattr(result, 'headroom')


class TestComputeBreachWindow:
    _THETA = derive_theta_min(0.05, 13.51, 21.0)

    def test_healthy_signal_fast_detection(self):
        """Healthy signal (well above θ_min): W < 5 days."""
        W = compute_breach_window(0.05, 1.20, self._THETA)
        assert W < 5.0

    def test_marginal_signal_slow_detection(self):
        """Marginal signal (barely above θ_min): W > 10 days."""
        W = compute_breach_window(0.05, 0.50, self._THETA)
        assert W > 10.0

    def test_already_breaching_returns_inf(self):
        """Signal below θ_min → inf (already breaching)."""
        W = compute_breach_window(0.05, 0.30, self._THETA)
        assert W == float('inf')


class TestComputeOptimalTau:
    def test_confident_yields_high_tau(self):
        """Low covariance (confident) → high τ (sharp)."""
        cov = np.eye(6) * 0.01
        tau = compute_optimal_tau(cov)
        assert tau > 0.15

    def test_uncertain_yields_low_tau(self):
        """High covariance (uncertain) → low τ (soft)."""
        cov = np.eye(6) * 0.5
        tau = compute_optimal_tau(cov)
        assert tau < 0.10

    def test_always_in_range(self):
        """τ always within bounds."""
        for tr_val in [0.01, 0.1, 0.5, 1.0, 2.0]:
            cov = np.eye(6) * tr_val / 6
            tau = compute_optimal_tau(cov, tau_range=(0.05, 0.20))
            assert 0.05 <= tau <= 0.20


class TestTransferPrior:
    def test_shape(self):
        """Prior mean and std match centroid shape."""
        centroids = {
            'cat1': np.random.rand(4, 6),
            'cat2': np.random.rand(4, 6),
            'cat3': np.random.rand(4, 6),
        }
        mean, std = compute_transfer_prior(centroids)
        assert mean.shape == (4, 6)
        assert std.shape == (4, 6)

    def test_empty_returns_defaults(self):
        """Empty centroids returns scalar defaults."""
        mean, std = compute_transfer_prior({})
        assert mean.shape == (1,)


class TestMetaConservation:
    def test_small_divergence_passes(self):
        """Small divergence → passed."""
        old = np.ones((4, 6)) * 0.5
        new = old + 0.01
        passed, details = check_meta_conservation(new, {}, old, epsilon=0.05)
        assert passed is True
        assert details['max_divergence'] < 0.05

    def test_large_divergence_fails(self):
        """Large divergence → failed with review_required."""
        old = np.ones((4, 6)) * 0.5
        new = old + 0.10
        passed, details = check_meta_conservation(new, {}, old, epsilon=0.05)
        assert passed is False
        assert details['recommendation'] == 'review_required'


class TestComputeEtaOverride:
    def test_mixed_team_in_expected_range(self):
        """Mixed team (q̄=0.75, σ²=0.03) gives η_override in [0.005, 0.03]."""
        eta = compute_eta_override(0.05, 0.75, 0.03)
        assert 0.005 <= eta <= 0.03

    def test_lower_quality_gives_lower_eta(self):
        """Junior-heavy team → lower η_override than mixed team."""
        eta_mixed = compute_eta_override(0.05, 0.75, 0.03)
        eta_junior = compute_eta_override(0.05, 0.65, 0.05)
        assert eta_junior < eta_mixed

    def test_higher_quality_gives_higher_eta(self):
        """High-quality team → higher η_override than mixed team."""
        eta_mixed = compute_eta_override(0.05, 0.75, 0.03)
        eta_perfect = compute_eta_override(0.05, 0.95, 0.01)
        assert eta_perfect > eta_mixed

    def test_zero_signal_returns_floor(self):
        """mean_quality ≤ 0.5 → returns floor 0.005."""
        assert compute_eta_override(0.05, 0.45, 0.05) == 0.005
        assert compute_eta_override(0.05, 0.50, 0.05) == 0.005

    def test_result_is_rounded(self):
        """Result is rounded to 4 decimal places."""
        eta = compute_eta_override(0.05, 0.75, 0.03)
        assert eta == round(eta, 4)

    def test_safety_margin_scales_result(self):
        """Higher safety_margin → higher η_override."""
        eta_half = compute_eta_override(0.05, 0.75, 0.03, safety_margin=0.5)
        eta_full = compute_eta_override(0.05, 0.75, 0.03, safety_margin=1.0)
        assert eta_full > eta_half


class TestComputeFactorMask:
    def test_basic_threshold(self):
        """σ < threshold → True (include); σ ≥ threshold → False (exclude)."""
        mask = compute_factor_mask({'a': 0.15, 'b': 0.25, 'c': 0.08}, 0.20)
        assert mask == {'a': True, 'b': False, 'c': True}

    def test_all_clean(self):
        """All σ below threshold → all True."""
        mask = compute_factor_mask({'a': 0.05, 'b': 0.10}, 0.20)
        assert all(mask.values())

    def test_all_noisy(self):
        """All σ above threshold → all False."""
        mask = compute_factor_mask({'a': 0.25, 'b': 0.30}, 0.20)
        assert not any(mask.values())

    def test_at_threshold_excluded(self):
        """σ == threshold is excluded (strict less-than)."""
        mask = compute_factor_mask({'a': 0.20}, 0.20)
        assert mask['a'] is False

    def test_empty_input(self):
        """Empty dict returns empty dict."""
        assert compute_factor_mask({}) == {}


class TestMaskToArray:
    def test_shape_and_values(self):
        """Array shape (6,) with correct 1.0/0.0 for SOC default order."""
        mask = {
            'travel_match': True, 'asset_criticality': True,
            'threat_intel_enrichment': True, 'time_anomaly': False,
            'pattern_history': True, 'device_trust': False,
        }
        arr = mask_to_array(mask)
        assert arr.shape == (6,)
        assert arr[0] == 1.0   # travel_match
        assert arr[3] == 0.0   # time_anomaly
        assert arr[5] == 0.0   # device_trust

    def test_custom_order(self):
        """Custom factor_names respected."""
        mask = {'x': True, 'y': False, 'z': True}
        arr = mask_to_array(mask, factor_names=['x', 'y', 'z'])
        np.testing.assert_array_equal(arr, [1.0, 0.0, 1.0])

    def test_missing_key_defaults_to_include(self):
        """Factor absent from mask dict defaults to 1.0 (include)."""
        mask = {'travel_match': False}  # device_trust missing
        arr = mask_to_array(mask)
        assert arr[5] == 1.0   # device_trust absent → include


# ── Enriched Bootstrap Prior Tests ───────────────────────────────────────────

from gae.calibration import compute_enriched_bootstrap_prior
from gae.bootstrap import bootstrap_enriched_prior
from dataclasses import dataclass


@dataclass
class _DomainConfig:
    """Minimal domain_config stub for tests."""
    factor_names: list
    n_cat: int
    n_act: int


class TestEnrichedBootstrapPrior:
    def _make_config(self):
        return _DomainConfig(
            factor_names=["threat_intel", "pattern_history"],
            n_cat=2,
            n_act=2,
        )

    def _make_decisions(self, n=20):
        """Historical decisions: all push threat_intel high, pattern_history low."""
        rng = np.random.default_rng(0)
        decisions = []
        for _ in range(n):
            # threat_intel = 0.9, pattern_history = 0.1
            f = np.array([0.9, 0.1])
            decisions.append((0, 0, f))
        return decisions

    def test_enriched_bootstrap_upweights_low_sigma_factors(self):
        """
        threat_intel σ=0.05 → W=400; pattern_history σ=0.20 → W=25.
        After enrichment, threat_intel dimension of μ₀ should be closer
        to the historical signal (0.9) than pattern_history dimension (0.1).
        """
        cfg = self._make_config()
        sigma = {"threat_intel": 0.05, "pattern_history": 0.20}
        decisions = self._make_decisions(n=50)

        mu = compute_enriched_bootstrap_prior(
            historical_decisions=decisions,
            measured_sigma=sigma,
            domain_config=cfg,
            n_cat=cfg.n_cat,
            n_act=cfg.n_act,
            n_factors=len(cfg.factor_names),
        )

        assert mu.shape == (2, 2, 2)
        # threat_intel dimension (idx 0) of updated centroid should be farther
        # from 0.5 (initial) than pattern_history dimension (idx 1),
        # because high W pulls it more strongly toward the enriched signal.
        ti_delta = abs(mu[0, 0, 0] - 0.5)
        ph_delta = abs(mu[0, 0, 1] - 0.5)
        assert ti_delta > ph_delta, (
            f"Expected threat_intel pull ({ti_delta:.4f}) > "
            f"pattern_history pull ({ph_delta:.4f}) due to lower σ weight"
        )

    def test_enriched_bootstrap_clips_to_unit_interval(self):
        """All μ₀_enriched values must lie in [0.0, 1.0]."""
        cfg = self._make_config()
        sigma = {"threat_intel": 0.05, "pattern_history": 0.05}
        # Extreme f vectors that would push enriched values out of [0,1] if unclipped
        decisions = [(c, a, np.array([1.0, 1.0])) for c in range(2) for a in range(2)] * 200

        mu = compute_enriched_bootstrap_prior(
            historical_decisions=decisions,
            measured_sigma=sigma,
            domain_config=cfg,
            n_cat=cfg.n_cat,
            n_act=cfg.n_act,
            n_factors=len(cfg.factor_names),
        )

        assert np.all(mu >= 0.0), f"μ₀ contains values < 0.0: min={mu.min()}"
        assert np.all(mu <= 1.0), f"μ₀ contains values > 1.0: max={mu.max()}"

    def test_enriched_bootstrap_gradient_not_vector_scaling(self):
        """
        Proves the fix: W_normalized multiplies (f - mu), NOT f.

        Setup: mu_start = [0.5, 0.5], f = [0.8, 0.2], W_normalized = [2.0, 0.5]
        eta = 0.05 (_ETA_CONFIRM)

        WRONG result (old code): update = eta * (W*f - mu)
          = 0.05 * ([1.6, 0.1] - [0.5, 0.5])
          = 0.05 * [1.1, -0.4]
          mu_wrong = [0.555, 0.480]   <- dim 0 shoots above 0.53 (f distorted)

        CORRECT result: update = eta * W * (f - mu)
          = 0.05 * [2.0, 0.5] * [0.3, -0.3]
          = 0.05 * [0.6, -0.15]
          mu_correct = [0.530, 0.4925]  <- both move toward f; reliable dim faster
        """
        # Reverse-engineer sigma so W_raw mean gives W_norm = [2.0, 0.5]
        # W_norm = W_raw / W_raw.mean() = [2.0, 0.5]  → W_raw = k*[2.0, 0.5]
        # Choose k=1.25 → W_raw=[2.5, 0.625], mean=1.5625... recalculate:
        # mean([2.5,0.625]) = 1.5625 → W_norm=[1.6, 0.4].  Not [2.0,0.5].
        # Correct: W_norm=[2,0.5] requires W_raw proportional to [2,0.5].
        # W_raw = [2,0.5] (any k): mean=1.25, W_norm=[1.6, 0.4]. Still no.
        # The only way W_norm=[2.0,0.5] is if W_raw=[2.0,0.5] AND mean=1.0,
        # i.e., W_raw must already sum to 2 with mean=1.25... let's use [4.0,1.0]:
        # mean([4,1])=2.5, W_norm=[1.6,0.4]. Still not [2.0,0.5].
        # W_norm=[2.0,0.5] means W_raw=[2a, 0.5a], mean=1.25a, norm=[1.6,0.4].
        # Impossible with 2 elements. Use W_raw that normalises to exactly [2.0,0.5]:
        # Need mean=1 so W_raw=W_norm → W_raw=[2.0,0.5], mean=1.25 → W_norm=[1.6,0.4].
        # Use a 4-element case instead: W_raw=[4,1,1,1] mean=1.75→norm=[2.286,...]
        # Simplest: directly assert on a known sigma pair where ratio is exact.
        # sigma_r / sigma_n = sqrt(4) = 2 → W_r/W_n = 4.
        # Let sigma_r=0.1, sigma_n=0.2 → W_r=100, W_n=25, mean=62.5 → W_norm=[1.6,0.4]
        # Exact result: mu[0] = 0.5 + 0.05*1.6*(0.8-0.5) = 0.5+0.024 = 0.524
        #               mu[1] = 0.5 + 0.05*0.4*(0.2-0.5) = 0.5-0.006 = 0.494
        # Wrong result: 0.5 + 0.05*(1.6*0.8 - 0.5) = 0.5+0.05*0.78 = 0.539
        #               0.5 + 0.05*(0.4*0.2 - 0.5) = 0.5+0.05*(-0.42) = 0.479

        @dataclass
        class _Cfg2:
            factor_names: list
            n_cat: int = 1
            n_act: int = 1

        cfg = _Cfg2(factor_names=["reliable", "noisy"])
        sigma = {"reliable": 0.1, "noisy": 0.2}

        # Verify W_normalized used internally
        W_raw  = np.array([1/0.1**2, 1/0.2**2])    # [100, 25]
        W_norm = W_raw / W_raw.mean()               # [1.6, 0.4]
        np.testing.assert_allclose(W_norm, [1.6, 0.4], atol=1e-10)

        decisions = [(0, 0, np.array([0.8, 0.2]))]
        mu_out = compute_enriched_bootstrap_prior(
            historical_decisions=decisions,
            measured_sigma=sigma,
            domain_config=cfg,
            n_cat=1,
            n_act=1,
            n_factors=2,
        )

        # Correct: eta * W_norm * (f - mu_start)
        #   dim 0: 0.5 + 0.05 * 1.6 * (0.8 - 0.5) = 0.5 + 0.024 = 0.524
        #   dim 1: 0.5 + 0.05 * 0.4 * (0.2 - 0.5) = 0.5 - 0.006 = 0.494
        np.testing.assert_allclose(mu_out[0, 0, 0], 0.524, atol=1e-10,
            err_msg="dim 0: correct gradient eq gives 0.524; wrong f-scaling gives 0.539")
        np.testing.assert_allclose(mu_out[0, 0, 1], 0.494, atol=1e-10,
            err_msg="dim 1: correct gradient eq gives 0.494; wrong f-scaling gives 0.479")

    def test_enriched_bootstrap_reliable_factors_converge_faster(self):
        """
        With σ_reliable=0.05 and σ_noisy=0.20 (W ratio 16:1), after 100 decisions
        where both factors signal 0.9, mu[reliable_dim] is closer to 0.9.
        Proves reliable factors converge faster; f is never distorted.
        """
        @dataclass
        class _Cfg2:
            factor_names: list
            n_cat: int = 1
            n_act: int = 1

        cfg = _Cfg2(factor_names=["reliable", "noisy"])
        sigma = {"reliable": 0.05, "noisy": 0.20}

        decisions = [(0, 0, np.array([0.9, 0.9])) for _ in range(100)]

        mu_out = compute_enriched_bootstrap_prior(
            historical_decisions=decisions,
            measured_sigma=sigma,
            domain_config=cfg,
            n_cat=1,
            n_act=1,
            n_factors=2,
        )

        dist_reliable = abs(mu_out[0, 0, 0] - 0.9)
        dist_noisy    = abs(mu_out[0, 0, 1] - 0.9)

        assert dist_reliable < dist_noisy, (
            f"Reliable dim (σ=0.05) should be closer to 0.9 than noisy dim (σ=0.20). "
            f"dist_reliable={dist_reliable:.4f}, dist_noisy={dist_noisy:.4f}"
        )

    def test_bootstrap_enriched_prior_writes_anchor_once(self, tmp_path):
        """
        First call to bootstrap_enriched_prior() succeeds and writes anchor.
        Second call raises RuntimeError (write-once guard fires).
        """
        cfg = self._make_config()
        sigma = {"threat_intel": 0.10, "pattern_history": 0.15}
        decisions = self._make_decisions(n=10)
        anchor = str(tmp_path / "iks_bootstrap_soc.json")

        # First call must succeed and return correct shape
        mu = bootstrap_enriched_prior(
            historical_decisions=decisions,
            measured_sigma=sigma,
            domain_config=cfg,
            anchor_filepath=anchor,
        )
        assert mu.shape == (cfg.n_cat, cfg.n_act, len(cfg.factor_names))
        import os
        assert os.path.exists(anchor)

        # Second call must raise RuntimeError
        with pytest.raises(RuntimeError, match="IKS anchor"):
            bootstrap_enriched_prior(
                historical_decisions=decisions,
                measured_sigma=sigma,
                domain_config=cfg,
                anchor_filepath=anchor,
            )

    def test_delta_sigma_upweights_enriched_factors(self):
        """
        Δσ scheme: W_j = sigma_before_j² / sigma_after_j⁴.

        f0 heavily enriched: sigma_before=0.24, sigma_after=0.09
          W0 = 0.24² / 0.09⁴ = 0.0576 / 6.561e-5 ≈ 878
        f1 not enriched:     sigma_before=0.09, sigma_after=0.09
          W1 = 0.09² / 0.09⁴ = 1/0.09² ≈ 123

        W_normalized ratio W0/W1 ≈ 7.1 (enriched factor gets ~7× higher weight).
        """
        @dataclass
        class _Cfg2:
            factor_names: list
            n_cat: int = 1
            n_act: int = 1

        cfg = _Cfg2(factor_names=["enriched", "fixed"])

        sigma_after  = {"enriched": 0.09, "fixed": 0.09}
        sigma_before = {"enriched": 0.24, "fixed": 0.09}

        # Compute expected W values
        W0 = 0.24**2 / 0.09**4
        W1 = 0.09**2 / 0.09**4
        W_arr = np.array([W0, W1])
        W_norm = W_arr / W_arr.mean()

        # Verify ratio ≈ 7.1
        ratio = W_norm[0] / W_norm[1]
        assert abs(ratio - 7.111) < 0.01, (
            f"Expected W_norm ratio ≈ 7.1, got {ratio:.3f}"
        )

        # Verify the function produces the expected convergence differential.
        # One decision: f = [0.9, 0.9]. Enriched factor should move faster toward 0.9.
        decisions = [(0, 0, np.array([0.9, 0.9]))]
        mu_out = compute_enriched_bootstrap_prior(
            historical_decisions=decisions,
            measured_sigma=sigma_after,
            domain_config=cfg,
            n_cat=1,
            n_act=1,
            n_factors=2,
            sigma_before=sigma_before,
        )

        dist_enriched = abs(mu_out[0, 0, 0] - 0.9)
        dist_fixed    = abs(mu_out[0, 0, 1] - 0.9)
        assert dist_enriched < dist_fixed, (
            f"Enriched factor (W_norm={W_norm[0]:.1f}) should converge faster "
            f"than fixed factor (W_norm={W_norm[1]:.1f}). "
            f"dist_enriched={dist_enriched:.4f}, dist_fixed={dist_fixed:.4f}"
        )

    def test_fixed_factors_same_in_both_schemes(self):
        """
        For factor where sigma_before == sigma_after:
          W_delta = sigma² / sigma⁴ = 1/sigma²  (same as original scheme).
        Both schemes must produce identical mu for un-enriched factors.
        """
        @dataclass
        class _Cfg2:
            factor_names: list
            n_cat: int = 1
            n_act: int = 1

        cfg = _Cfg2(factor_names=["f0", "f1"])
        sigma = {"f0": 0.15, "f1": 0.20}

        # sigma_before == sigma_after → Δσ scheme reduces to 1/σ²
        decisions = [(0, 0, np.array([0.7, 0.3])) for _ in range(10)]

        mu_original = compute_enriched_bootstrap_prior(
            historical_decisions=decisions,
            measured_sigma=sigma,
            domain_config=cfg,
            n_cat=1,
            n_act=1,
            n_factors=2,
            sigma_before=None,      # original scheme
        )

        mu_delta = compute_enriched_bootstrap_prior(
            historical_decisions=decisions,
            measured_sigma=sigma,
            domain_config=cfg,
            n_cat=1,
            n_act=1,
            n_factors=2,
            sigma_before=sigma,     # sigma_before == sigma_after → same as original
        )

        np.testing.assert_allclose(
            mu_original, mu_delta, atol=1e-12,
            err_msg="When sigma_before == sigma_after, Δσ scheme must equal original scheme."
        )


# ── V-BOOTSTRAP-GEOM tests ────────────────────────────────────────────────────

from gae.calibration import (
    compute_dominant_axis,
    compute_enriched_bootstrap_prior_geom,
)
from dataclasses import dataclass as _dc2


@_dc2
class _Cfg6:
    """6-factor SOC-like domain config stub."""
    factor_names: list
    n_cat: int = 2
    n_act: int = 2


def _make_cfg6():
    return _Cfg6(
        factor_names=['travel_match', 'asset_criticality', 'threat_intel',
                      'time_anomaly', 'pattern_history', 'device_trust'],
    )


class TestVBootstrapGeom:
    def test_geom_attenuates_discriminating_factors(self):
        """
        threat_intel has high centroid variance (discriminating) →
        dominant_axis high → W_geom attenuated.
        device_trust has low centroid variance (non-discriminating) →
        dominant_axis low → W_geom full.
        Both enriched equally. Assert W_geom[device_trust] > 2× W_geom[threat_intel].
        """
        cfg = _make_cfg6()
        n_factors = 6
        # threat_intel (idx 2): large spread across centroid positions
        # device_trust (idx 5): small spread (near 0.50 everywhere)
        mu = np.full((2, 2, n_factors), 0.5)
        mu[:, 0, 2] = 0.9;  mu[:, 1, 2] = 0.1   # threat_intel: high variance
        mu[:, :, 5] = 0.5 + np.array([[0.02, -0.02], [-0.01, 0.01]])  # device_trust: low

        # Equal enrichment for both factors (sigma_before/after ratio = 2.0)
        sigma_after  = {f: 0.10 for f in cfg.factor_names}
        sigma_before = {f: 0.20 for f in cfg.factor_names}

        sa = np.array([sigma_after[f]  for f in cfg.factor_names])
        sb = np.array([sigma_before[f] for f in cfg.factor_names])
        enrichment_ratio = (sb / sa) ** 2          # all equal = 4.0
        dom = compute_dominant_axis(mu)
        W_geom = np.clip(enrichment_ratio * (1.0 - dom), 1e-6, None)

        ti_idx = 2; dt_idx = 5
        assert W_geom[dt_idx] > 2.0 * W_geom[ti_idx], (
            f"device_trust W_geom={W_geom[dt_idx]:.6f} should be >2x "
            f"threat_intel W_geom={W_geom[ti_idx]:.6f} (B1 attenuation check)"
        )

    def test_geom_uniform_mu_uses_enrichment_ratio_only(self):
        """
        When mu is uniform (all positions identical), dominant_axis = zeros.
        W_geom = enrichment_ratio * (1 - 0) = enrichment_ratio = (sb/sa)².
        The function uses pure enrichment-benefit weighting — geometry plays no role.

        Verified analytically: one decision, compute expected mu from
        W_norm = enrichment_ratio / mean(enrichment_ratio).
        """
        from dataclasses import dataclass as _dc3

        @_dc3
        class _Cfg2u:
            factor_names: list
            n_cat: int = 1
            n_act: int = 1

        cfg = _Cfg2u(factor_names=["f0", "f1"])
        n_factors = 2
        mu_uniform = np.full((1, 1, n_factors), 0.5)

        # dominant_axis must be zeros for uniform mu
        dom = compute_dominant_axis(mu_uniform)
        np.testing.assert_array_equal(dom, np.zeros(n_factors),
            err_msg="Uniform mu must produce zero dominant_axis scores")

        # sigma values — f0 enriched 3×, f1 enriched 1× (unchanged)
        sigma_after  = {"f0": 0.10, "f1": 0.15}
        sigma_before = {"f0": 0.30, "f1": 0.15}

        # Expected W = enrichment_ratio = (sb/sa)²
        sa = np.array([0.10, 0.15])
        sb = np.array([0.30, 0.15])
        er = (sb / sa) ** 2           # [9.0, 1.0]
        W_norm = er / er.mean()       # [1.8, 0.2]

        # One decision: f = [0.8, 0.4]
        f = np.array([0.8, 0.4])
        # Expected: mu = 0.5 + 0.05 * W_norm * (f - 0.5)
        expected = np.clip(0.5 + 0.05 * W_norm * (f - 0.5), 0.0, 1.0)

        mu_geom = compute_enriched_bootstrap_prior_geom(
            historical_decisions=[(0, 0, f)],
            measured_sigma=sigma_after,
            sigma_before=sigma_before,
            mu_current=mu_uniform,
            domain_config=cfg,
            n_cat=1, n_act=1, n_factors=n_factors,
        )

        np.testing.assert_allclose(
            mu_geom[0, 0, :], expected, atol=1e-12,
            err_msg=(
                "With uniform mu (dom_axis=0), W_geom = enrichment_ratio. "
                f"Expected {expected}, got {mu_geom[0, 0, :]}"
            ),
        )

    def test_geom_clips_weights_above_zero(self):
        """
        A perfectly discriminating factor (dominant_axis = 1.0) produces
        W_geom = enrichment_ratio * 0.0 = 0, clipped to 1e-6.
        All W_geom values must be >= 1e-6.
        """
        cfg = _make_cfg6()
        n_factors = 6
        # Make one factor perfectly discriminating: constant 0 or 1 across all positions
        mu = np.full((2, 2, n_factors), 0.5)
        mu[0, 0, 0] = 1.0; mu[0, 1, 0] = 0.0   # factor 0: max variance
        mu[1, 0, 0] = 1.0; mu[1, 1, 0] = 0.0

        sigma_after  = {f: 0.10 for f in cfg.factor_names}
        sigma_before = {f: 0.20 for f in cfg.factor_names}

        sa = np.array([sigma_after[f]  for f in cfg.factor_names])
        sb = np.array([sigma_before[f] for f in cfg.factor_names])
        enrichment_ratio = (sb / sa) ** 2
        dom = compute_dominant_axis(mu)

        # Factor 0 should be the dominant axis (normalized to 1.0)
        assert dom[0] == pytest.approx(1.0), (
            f"Factor 0 has max variance; dominant_axis[0] should be 1.0, got {dom[0]}"
        )

        W_geom = np.clip(enrichment_ratio * (1.0 - dom), 1e-6, None)
        assert np.all(W_geom >= 1e-6), (
            f"All W_geom must be >= 1e-6. Min={W_geom.min():.2e}"
        )
        # The perfectly discriminating factor must be exactly at the clip floor
        assert W_geom[0] == pytest.approx(1e-6), (
            f"Dominant factor (axis=1.0) must be clipped to 1e-6, got {W_geom[0]:.2e}"
        )
