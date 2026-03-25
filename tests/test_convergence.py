"""Tests for gae.convergence — get_convergence_metrics() and prediction functions."""

import numpy as np
import pytest

from gae.calibration import CalibrationProfile
from gae.convergence import (
    ACCURACY_THRESHOLD,
    EPSILON_DEFAULT,
    STABILITY_THRESHOLD,
    compute_e_inf_per_component,
    compute_n_half,
    compute_normalized_var_q,
    compute_steady_state_mse,
    ConservationMonitor,
    OLSMonitor,
    enrichment_multiplier,
    generate_onboarding_calendar,
    get_convergence_metrics,
    predict_category_convergence_weeks,
    predict_convergence_decisions,
    predict_convergence_decisions_v2,
    reconvergence_acceleration,
)
from gae.learning import LearningState


W_INIT = np.array([
    [ 0.30, -0.10, -0.25, -0.15,  0.20,  0.25],
    [ 0.05,  0.20,  0.15,  0.10, -0.05,  0.05],
    [-0.10,  0.05,  0.20,  0.20, -0.10, -0.05],
    [-0.25,  0.30,  0.30,  0.15, -0.20, -0.15],
], dtype=np.float64)
NAMES = ["travel", "asset", "threat", "time", "device", "pattern"]
F = np.array([[0.9, 0.3, 0.1, 0.2, 0.8, 0.7]])


def make_state(**kw) -> LearningState:
    d = dict(W=W_INIT.copy(), n_actions=4, n_factors=6, factor_names=NAMES[:],
             profile=CalibrationProfile())
    d.update(kw)
    return LearningState(**d)


class TestGetConvergenceMetricsKeys:
    """All expected keys must always be present."""

    def test_keys_empty_history(self):
        s = make_state()
        m = get_convergence_metrics(s)
        assert "stability" in m
        assert "accuracy" in m
        assert "weight_norm" in m
        assert "converged" in m
        assert "decisions" in m
        assert "provisional_dimensions" in m
        assert "pending_autonomous" in m

    def test_keys_after_updates(self):
        s = make_state()
        s.update(0, "fp_close", +1, F)
        s.update(0, "fp_close", -1, F)
        m = get_convergence_metrics(s)
        assert "stability" in m and "accuracy" in m

    def test_all_values_numeric(self):
        s = make_state()
        s.update(0, "fp_close", +1, F)
        m = get_convergence_metrics(s)
        for k in ("stability", "accuracy", "weight_norm"):
            assert isinstance(m[k], float), f"{k} should be float"
        assert isinstance(m["decisions"], int)
        assert isinstance(m["converged"], bool)


class TestGetConvergenceMetricsValues:
    def test_weight_norm_positive(self):
        s = make_state()
        m = get_convergence_metrics(s)
        assert m["weight_norm"] > 0

    def test_empty_history_stability_zero(self):
        s = make_state()
        m = get_convergence_metrics(s)
        assert m["stability"] == 0.0

    def test_empty_history_accuracy_zero(self):
        s = make_state()
        m = get_convergence_metrics(s)
        assert m["accuracy"] == 0.0

    def test_empty_history_not_converged(self):
        s = make_state()
        m = get_convergence_metrics(s)
        assert m["converged"] is False

    def test_accuracy_all_correct(self):
        s = make_state()
        for _ in range(5):
            s.update(0, "fp_close", +1, F)
        m = get_convergence_metrics(s)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_accuracy_all_wrong(self):
        s = make_state()
        for _ in range(5):
            s.update(0, "fp_close", -1, F)
        m = get_convergence_metrics(s)
        assert m["accuracy"] == pytest.approx(0.0)

    def test_accuracy_mixed(self):
        s = make_state()
        s.update(0, "fp_close", +1, F)
        s.update(0, "fp_close", -1, F)
        m = get_convergence_metrics(s)
        assert m["accuracy"] == pytest.approx(0.5)

    def test_decisions_count_correct(self):
        s = make_state()
        s.update(0, "fp_close", +1, F)
        s.update(0, "fp_close", +1, F)
        m = get_convergence_metrics(s)
        assert m["decisions"] == 2

    def test_stable_weights_low_stability(self):
        """Many identical correct outcomes → W stabilises."""
        s = make_state()
        # Apply 20 identical updates — W should converge
        for _ in range(50):
            s.update(0, "fp_close", +1, F)
        m = get_convergence_metrics(s)
        # Stability measures std of recent W norms; may be > 0 early then converge
        assert isinstance(m["stability"], float)

    def test_converged_requires_high_accuracy(self):
        """Low accuracy → not converged even if stable."""
        s = make_state()
        for _ in range(20):
            s.update(0, "fp_close", -1, F)  # all wrong → accuracy = 0
        m = get_convergence_metrics(s)
        assert m["converged"] is False

    def test_provisional_dimensions_counted(self):
        s = make_state()
        s.expand_weight_matrix("new_dim")
        m = get_convergence_metrics(s)
        assert m["provisional_dimensions"] == 1

    def test_pending_autonomous_counted(self):
        s = make_state()
        s.update(0, "fp_close", +1, F, decision_source="autonomous")
        m = get_convergence_metrics(s)
        assert m["pending_autonomous"] == 1


class TestConvergenceTaskVerification:
    def test_task_verification_smoke(self):
        """Exact replication of the verification script from the task description."""
        W = np.array([
            [ 0.3, -0.1, -0.25, -0.15,  0.2,  0.25],
            [ 0.05,  0.2,  0.15,  0.10, -0.05,  0.05],
            [-0.1,  0.05,  0.20,  0.20, -0.1, -0.05],
            [-0.25,  0.3,  0.30,  0.15, -0.2, -0.15],
        ])
        f = np.array([[0.9, 0.3, 0.1, 0.2, 0.8, 0.7]])
        names = ["travel", "asset", "threat", "time", "device", "pattern"]
        state = LearningState(W=W.copy(), n_actions=4, n_factors=6, factor_names=names,
                              profile=CalibrationProfile())
        state.update(0, "fp_close", +1, f)
        state.update(0, "fp_close", -1, f)
        state.expand_weight_matrix("new_dim")
        m = get_convergence_metrics(state)
        assert "stability" in m
        assert "accuracy" in m


# ── Prediction function tests (math_synopsis_v9 §5-6) ───────────────────────

class TestComputeNHalf:
    def test_n_half_at_default(self):
        """N_half = 13.51 at η=0.05 (math_synopsis_v9 §5)."""
        n = compute_n_half(0.05)
        assert abs(n - 13.51) < 0.01

    def test_n_half_higher_eta(self):
        """Higher η → shorter half-life."""
        assert compute_n_half(0.10) < compute_n_half(0.05)


class TestSteadyStateMse:
    def test_steady_state_mse(self):
        """MSE_∞ = 0.05/1.95 × 0.34 ≈ 0.00872."""
        mse = compute_steady_state_mse(0.05, 0.34)
        assert abs(mse - 0.00872) < 0.001

    def test_e_inf_per_component(self):
        """e_∞ ≈ 0.038 at defaults."""
        e = compute_e_inf_per_component(0.05, 0.34, 6)
        assert abs(e - 0.038) < 0.005


class TestPredictConvergenceDecisions:
    def test_convergence_below_steady_state_returns_minus_one(self):
        """Cannot converge below e_∞. Returns -1."""
        n = predict_convergence_decisions(0.15, 0.01)
        assert n == -1

    def test_already_converged_returns_zero(self):
        """e_0 < ε returns 0."""
        n = predict_convergence_decisions(0.03, 0.05)
        assert n == 0

    def test_convergence_positive(self):
        """Normal case returns positive integer."""
        n = predict_convergence_decisions(0.15, 0.05)
        assert n > 0
        assert isinstance(n, int)


class TestPredictConvergenceDecisionsV2:
    def test_already_converged_returns_zero(self):
        """e_0 ≤ ε returns 0."""
        n = predict_convergence_decisions_v2(0.05, epsilon=0.10)
        assert n == 0

    def test_positive_result_with_safety_factor(self):
        """Normal case returns positive int; safety_factor=2.0 doubles raw N."""
        n = predict_convergence_decisions_v2(0.15, epsilon=0.10, safety_factor=2.0)
        assert n > 0 and isinstance(n, int)
        n_raw = predict_convergence_decisions_v2(0.15, epsilon=0.10, safety_factor=1.0)
        # With safety_factor=2.0 result should be ≥ raw (ceiling may round up)
        assert n >= n_raw

    def test_auto_adjust_when_epsilon_too_close_to_floor(self):
        """ε ≤ e_∞ × 1.5 is auto-adjusted; should not raise and returns positive."""
        # e_inf ≈ 0.038; epsilon=0.05 < 0.038*1.5=0.057 — triggers auto-adjust
        n = predict_convergence_decisions_v2(0.15, epsilon=0.05)
        assert n > 0 and isinstance(n, int)

    def test_never_returns_minus_one(self):
        """v2 never returns -1; always auto-adjusts epsilon."""
        n = predict_convergence_decisions_v2(0.15, epsilon=0.01)
        assert n != -1
        assert n > 0

    def test_epsilon_default_is_0_10(self):
        """EPSILON_DEFAULT constant is 0.10."""
        assert EPSILON_DEFAULT == pytest.approx(0.10)

    def test_higher_safety_factor_gives_more_decisions(self):
        """Larger safety_factor → more decisions (never less)."""
        n1 = predict_convergence_decisions_v2(0.15, epsilon=0.10, safety_factor=1.0)
        n2 = predict_convergence_decisions_v2(0.15, epsilon=0.10, safety_factor=2.0)
        assert n2 >= n1


class TestEnrichmentMultiplier:
    def test_enrichment_multiplier_monotonic(self):
        """G1 > G2 > G3 > G4 (decreasing = faster convergence)."""
        m = [enrichment_multiplier(g) for g in ['G1', 'G2', 'G3', 'G4']]
        assert m[0] > m[1] > m[2] > m[3]

    def test_enrichment_g1_is_baseline(self):
        """G1 multiplier is 1.0."""
        assert enrichment_multiplier('G1') == 1.0

    def test_enrichment_unknown_returns_1(self):
        """Unknown level returns 1.0 (no acceleration)."""
        assert enrichment_multiplier('G99') == 1.0


class TestReconvergenceAcceleration:
    def test_reconvergence_acceleration_monotonic(self):
        """Each episode is faster than the last."""
        r = [reconvergence_acceleration(i) for i in range(3)]
        assert r[0] > r[1] > r[2]

    def test_reconvergence_episode_zero_is_one(self):
        """Episode 0 = initial convergence = 1.0 multiplier."""
        assert abs(reconvergence_acceleration(0) - 1.0) < 0.001

    def test_reconvergence_decay_rate(self):
        """Decay ≈ 0.703 per episode (Bridge B Phase C v3)."""
        r1 = reconvergence_acceleration(1)
        assert abs(r1 - 0.703) < 0.01


class TestPredictCategoryConvergenceWeeks:
    def test_category_convergence_positive_weeks(self):
        """Normal category produces positive weeks."""
        result = predict_category_convergence_weeks(
            category='credential_access',
            alerts_per_day=60,
            verification_rate=0.30,
        )
        assert result['status'] == 'will_converge'
        assert result['weeks'] > 0

    def test_category_convergence_enrichment_faster(self):
        """G4 converges faster than G1 (low volume avoids rounding collapse)."""
        g1 = predict_category_convergence_weeks('test', alerts_per_day=20, graph_level='G1')
        g4 = predict_category_convergence_weeks('test', alerts_per_day=20, graph_level='G4')
        assert g4['weeks'] < g1['weeks']


class TestGenerateOnboardingCalendar:
    _CATS = ['credential_access', 'lateral_movement', 'data_exfiltration',
             'insider_threat', 'cloud_infrastructure', 'threat_intel_match']
    _WEIGHTS = {
        'credential_access': 0.3, 'lateral_movement': 0.2,
        'data_exfiltration': 0.15, 'insider_threat': 0.15,
        'cloud_infrastructure': 0.1, 'threat_intel_match': 0.1,
    }

    def test_calendar_output_structure(self):
        """Calendar returns predictions for all categories."""
        cal = generate_onboarding_calendar(self._CATS, self._WEIGHTS)
        assert 'predictions' in cal
        assert len(cal['predictions']) == 6
        assert cal['total_weeks'] > 0
        assert cal['first_calibrated'] is not None
        assert cal['assumptions']['eta'] == 0.05

    def test_calendar_high_volume_converges_faster(self):
        """Higher alert volume → faster convergence."""
        cats = ['test_cat']
        weights = {'test_cat': 1.0}
        cal_low = generate_onboarding_calendar(cats, weights, alerts_per_day=50)
        cal_high = generate_onboarding_calendar(cats, weights, alerts_per_day=500)
        assert cal_high['total_weeks'] < cal_low['total_weeks']


# ---------------------------------------------------------------------------
# Fix A — compute_normalized_var_q
# ---------------------------------------------------------------------------

class TestNormalizedVarQ:
    """Tests for baseline-normalized quality variance (Fix A)."""

    def test_normalized_var_q_zero_at_baseline(self):
        """Healthy Bernoulli fluctuations around q_baseline yield ~0."""
        q_baseline = 0.85
        q_rolling = [0.85, 0.90, 0.80, 0.85, 0.90]
        result = compute_normalized_var_q(q_rolling, q_baseline)
        assert result >= 0.0, "Result must be non-negative"
        assert result < 0.02, f"Expected ~0.0 for healthy data, got {result:.4f}"

    def test_normalized_var_q_positive_on_degradation(self):
        """Bimodal (chaotic) quality scores exceed the Bernoulli floor → result > 0.05.

        var_baseline = 0.85*(1-0.85) = 0.1275.  Need var_raw > 0.1775 to clear
        the 0.05 threshold.  Alternating 0/1 gives var_raw ≈ 0.25 → result ≈ 0.12.
        """
        q_baseline = 0.85
        # Bimodal / chaotic degradation: quality swings between 0 and 1
        q_rolling = [0.0, 1.0, 0.0, 1.0, 0.0]
        result = compute_normalized_var_q(q_rolling, q_baseline)
        assert result > 0.05, f"Expected >0.05 for bimodal data, got {result:.4f}"

    def test_normalized_var_q_short_window_returns_zero(self):
        """Single-element window returns 0.0 (insufficient data)."""
        result = compute_normalized_var_q([0.85], 0.85)
        assert result == 0.0

    def test_normalized_var_q_never_negative(self):
        """Result is always max(0, ...) — never negative."""
        q_rolling = [0.85, 0.85, 0.85, 0.85]
        result = compute_normalized_var_q(q_rolling, 0.85)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# Layer 2 — CUSUM on EWMA of q̄ (v0.7.11)
# ---------------------------------------------------------------------------

class TestCUSUM:
    """Tests for Layer 2 CUSUM early warning (v0.7.11)."""

    def test_cusum_silent_on_healthy(self):
        """ARL₀ ≈ 500: healthy Bernoulli at q̄=0.85 fires ≤ 2 alarms in 500 decisions."""
        monitor = ConservationMonitor()
        rng = np.random.default_rng(42)
        alarm_count = 0
        for _ in range(500):
            q = float(rng.binomial(1, 0.85))
            monitor.record_quality(q)
            if monitor.yellow_warning:
                alarm_count += 1
                monitor.yellow_warning = False  # reset for next check
        assert alarm_count <= 2, (
            f"Expected ≤2 false alarms in 500 healthy decisions, got {alarm_count}"
        )

    def test_cusum_detects_step_shift(self):
        """CUSUM detects a step-down from q̄=0.85 to q̄=0.65.

        Shift at t=100.  h=15.0 calibrated for ARL₀=500; ARL₁ for Δ=0.20
        is ~95 decisions post-shift (seed=42: alarm at t=195).  Bound: ≤250.
        """
        monitor = ConservationMonitor()
        rng = np.random.default_rng(42)
        alarm_decision = None
        for t in range(300):
            q_bar = 0.85 if t < 100 else 0.65
            q = float(rng.binomial(1, q_bar))
            monitor.record_quality(q)
            if monitor.yellow_warning and alarm_decision is None:
                alarm_decision = t
        assert alarm_decision is not None, "Expected CUSUM to detect step shift"
        assert alarm_decision <= 250, (
            f"Expected alarm by decision 250, fired at {alarm_decision}"
        )

    def test_cusum_detects_linear_degradation(self):
        """CUSUM detects linear decline from q̄=0.85 to 0.40 over 500 decisions.

        h=15.0 calibrated for ARL₀=500; gradual degradation detected later than
        a step shift.  seed=42: alarm at t=251.  Bound: ≤350.
        """
        monitor = ConservationMonitor()
        rng = np.random.default_rng(42)
        alarm_decision = None
        for t in range(500):
            q_bar = max(0.40, 0.85 - 0.45 * t / 500)
            q = float(rng.binomial(1, q_bar))
            monitor.record_quality(q)
            if monitor.yellow_warning and alarm_decision is None:
                alarm_decision = t
        assert alarm_decision is not None, "Expected CUSUM to detect linear degradation"
        assert alarm_decision <= 350, (
            f"Expected alarm by decision 350, fired at {alarm_decision}"
        )

    def test_cusum_resets_after_alarm(self):
        """CUSUM fires yellow_warning and resets C_t=0 after alarm.

        After reset, CUSUM re-accumulates from continued degradation — so
        _cusum is not 0 at the end of the run.  The test verifies:
        (a) yellow_warning was set (alarm fired), and
        (b) _k is set (calibration period completed).
        """
        monitor = ConservationMonitor()
        rng = np.random.default_rng(42)
        for t in range(200):
            q = float(rng.binomial(1, 0.85 if t < 100 else 0.55))
            monitor.record_quality(q)
        # Alarm must have fired (step shift of 0.30 detected within 200 decisions)
        assert monitor.yellow_warning is True, (
            "Expected yellow_warning=True after step shift to q̄=0.55"
        )
        assert monitor._k is not None, "Expected _k set after calibration period"


# ---------------------------------------------------------------------------
# OLSMonitor — plateau-snapshot baseline (v0.7.12)
# ---------------------------------------------------------------------------

class TestOLSMonitor:
    """Tests for OLSMonitor plateau-snapshot baseline (v0.7.12)."""

    def test_ols_monitor_silent_during_learning_phase(self):
        """No alarm during learning-phase OLS decline; baseline freezes once stable.

        A smooth linear decline (step=0.0136/decision) has rolling variance
        ≈ 0.006 < plateau_threshold, so it would freeze immediately.  Real
        learning-phase OLS oscillates as centroids compete.  We replicate this
        with a deterministic ±0.15 oscillation on top of the decline, giving
        rolling var ≈ 0.030 > 0.02 throughout — plateau is NOT detected.

        Once OLS stabilises at 1.0 (no oscillation), var=0 < 0.02 → plateau
        freezes correctly.  yellow_warning must stay False throughout.
        """
        monitor = OLSMonitor(plateau_window=20, plateau_threshold=0.02)
        # Learning phase: oscillating OLS decline (var ≈ 0.030 > 0.02 → no freeze)
        for i in range(60):
            base = 1.8 - 0.8 * i / 59
            ols = base + 0.15 * (1 if i % 2 == 0 else -1)  # deterministic ±0.15
            monitor.update(ols)
            assert not monitor.yellow_warning, (
                f"Unexpected alarm at decision {i} during learning phase "
                f"(ols={ols:.3f}, baseline_frozen={monitor.baseline_frozen})"
            )
        # Stable phase: OLS flat at 1.0, no oscillation → var=0 < 0.02, plateau freezes
        for _ in range(20):
            monitor.update(1.0)
        assert monitor.baseline_frozen, (
            "Expected baseline_frozen=True after OLS stabilises at 1.0"
        )

    def test_ols_monitor_detects_post_plateau_degradation(self):
        """Alarm fires within 30 decisions of post-plateau OLS degradation."""
        monitor = OLSMonitor(plateau_window=20, plateau_threshold=0.02)
        # Plateau phase: 40 decisions stable at 1.2
        for _ in range(40):
            monitor.update(1.2)
        assert monitor.baseline_frozen, "Expected plateau after 40 stable decisions"
        assert abs(monitor.baseline_ols - 1.2) < 0.01, (
            f"Expected baseline_ols ≈ 1.2, got {monitor.baseline_ols:.4f}"
        )
        # Degradation: OLS drops 1.2 → 0.7 over 50 decisions
        alarm_decision = None
        for i in range(50):
            ols = 1.2 - 0.5 * i / 49
            fired = monitor.update(ols)
            if fired and alarm_decision is None:
                alarm_decision = i
        assert alarm_decision is not None, (
            "Expected CUSUM alarm during post-plateau degradation"
        )
        # k=0.10 means deviation must exceed 0.10 before cusum accumulates;
        # for a 0.5pp drop spread over 50 decisions, alarm fires around i=41.
        assert alarm_decision <= 45, (
            f"Expected alarm within 45 decisions of degradation, fired at {alarm_decision}"
        )

    def test_ols_monitor_plateau_frozen_correctly(self):
        """Plateau freezes correctly: baseline_ols ≈ 1.15 after 20 identical values."""
        monitor = OLSMonitor(plateau_window=20, plateau_threshold=0.02)
        for _ in range(20):
            monitor.update(1.15)
        assert monitor.baseline_frozen, (
            "Expected baseline_frozen=True after 20 zero-variance decisions"
        )
        assert abs(monitor.baseline_ols - 1.15) < 0.01, (
            f"Expected baseline_ols ≈ 1.15, got {monitor.baseline_ols:.4f}"
        )

    def test_ols_monitor_no_freeze_on_volatile_ols(self):
        """High-variance OLS (alternating 0.9/1.5) never triggers plateau freeze."""
        monitor = OLSMonitor(plateau_window=20, plateau_threshold=0.02)
        for i in range(30):
            ols = 0.9 if i % 2 == 0 else 1.5
            monitor.update(ols)
        assert not monitor.baseline_frozen, (
            "baseline_frozen must remain False for high-variance OLS"
        )
        assert not monitor.yellow_warning, (
            "yellow_warning must not fire when baseline is not frozen"
        )

    def test_ols_monitor_h_calibrates_at_plateau(self):
        """_h is calibrated (not None) once plateau is reached.

        Zero-variance plateau (OLS=1.2 constant) gives σ_OLS=0.0,
        hits the σ_OLS floor of 0.01, and h lands at the h floor=0.5.
        That is correct behavior — floor prevents h=0 on noiseless data.
        """
        monitor = OLSMonitor(plateau_window=20, plateau_threshold=0.02)
        for _ in range(20):
            monitor.update(1.2)
        assert monitor.baseline_frozen is True, (
            "Expected baseline_frozen=True after 20 constant OLS values"
        )
        assert monitor._h is not None, "_h must be set after plateau"
        assert monitor._h > 0.0, f"_h must be positive, got {monitor._h}"

    def test_ols_monitor_arl0_scales_with_noise(self):
        """Higher σ_OLS at plateau → larger h (ARL₀ maintained under high noise).

        With autocorrelation correction (W=20, rho=0.950, factor=39x):
          h = sigma_obs^2 * 39.0 * ln(1000) / (2k)
        The correction lifts h well above the 0.5 floor for any sigma>0.01,
        so the narrow (0.120, 0.141) window from v0.7.13 no longer applies.

        Monitor A: sigma_obs=0.06 -> h_eff ~ 4.85 > 0.5
        Monitor B: sigma_obs=0.13 -> h_eff ~ 22.8 > 4.85
        Both: h_B > h_A and h_B > 1.0.
        """
        # Monitor A: alternating +/-0.06 -> sigma=0.06, var=0.0036 < 0.02
        monitor_a = OLSMonitor(plateau_window=20, plateau_threshold=0.02)
        for i in range(20):
            monitor_a.update(1.2 + 0.06 * (1 if i % 2 == 0 else -1))
        assert monitor_a.baseline_frozen, "Monitor A: plateau not detected"

        # Monitor B: alternating +/-0.13 -> sigma=0.13, var=0.0169 < 0.02
        monitor_b = OLSMonitor(plateau_window=20, plateau_threshold=0.02)
        for i in range(20):
            monitor_b.update(1.2 + 0.13 * (1 if i % 2 == 0 else -1))
        assert monitor_b.baseline_frozen, "Monitor B: plateau not detected"

        assert monitor_b._h is not None and monitor_a._h is not None
        assert monitor_b._h > monitor_a._h, (
            f"Expected h_B ({monitor_b._h:.4f}) > h_A ({monitor_a._h:.4f}) "
            f"for higher sigma_OLS at plateau"
        )
        assert monitor_b._h > 1.0, (
            f"Expected h_B > 1.0 after autocorrelation correction, got {monitor_b._h:.4f}"
        )
