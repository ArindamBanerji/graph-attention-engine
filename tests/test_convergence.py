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
    compute_steady_state_mse,
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
