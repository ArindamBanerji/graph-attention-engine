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
