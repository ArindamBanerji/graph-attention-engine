"""
Conservation monitor tests for GAE.

Tests ConservationMonitor Layer 1 (AMBER/RED/GREEN status propagation)
and Layer 2 CUSUM (YELLOW early warning), plus derive_theta_min and
check_conservation edge cases from calibration.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from gae.convergence import ConservationMonitor, CALIBRATION_PERIOD, CUSUM_H
from gae.calibration import derive_theta_min, check_conservation


# ── Layer 1: conservation signal (AMBER/RED/GREEN) ────────────────────────────

class TestConservationMonitorLayer1:
    def test_initial_status_is_green(self):
        monitor = ConservationMonitor()
        assert monitor.conservation_status == "GREEN"

    def test_update_to_amber(self):
        monitor = ConservationMonitor()
        monitor.update_conservation_signal("AMBER")
        assert monitor.conservation_status == "AMBER"

    def test_update_to_red(self):
        monitor = ConservationMonitor()
        monitor.update_conservation_signal("RED")
        assert monitor.conservation_status == "RED"

    def test_update_back_to_green(self):
        monitor = ConservationMonitor()
        monitor.update_conservation_signal("RED")
        monitor.update_conservation_signal("GREEN")
        assert monitor.conservation_status == "GREEN"

    def test_multiple_updates_last_wins(self):
        monitor = ConservationMonitor()
        for status in ["AMBER", "RED", "AMBER", "GREEN", "RED"]:
            monitor.update_conservation_signal(status)
        assert monitor.conservation_status == "RED"

    def test_two_monitors_are_independent(self):
        m1 = ConservationMonitor()
        m2 = ConservationMonitor()
        m1.update_conservation_signal("RED")
        assert m2.conservation_status == "GREEN"

    def test_layer1_status_does_not_affect_yellow_warning(self):
        monitor = ConservationMonitor()
        monitor.update_conservation_signal("RED")
        assert monitor.yellow_warning is False

    def test_update_amber_red_amber_sequence(self):
        monitor = ConservationMonitor()
        monitor.update_conservation_signal("AMBER")
        assert monitor.conservation_status == "AMBER"
        monitor.update_conservation_signal("RED")
        assert monitor.conservation_status == "RED"
        monitor.update_conservation_signal("AMBER")
        assert monitor.conservation_status == "AMBER"


# ── Layer 2: CUSUM quality monitor (YELLOW) ───────────────────────────────────

class TestConservationMonitorLayer2:
    def test_baseline_not_set_initially(self):
        monitor = ConservationMonitor()
        assert monitor.baseline_set is False

    def test_q_baseline_zero_before_calibration(self):
        monitor = ConservationMonitor()
        assert monitor.q_baseline == pytest.approx(0.0)

    def test_baseline_set_after_calibration_period(self):
        monitor = ConservationMonitor()
        for _ in range(CALIBRATION_PERIOD):
            monitor.record_quality(0.85)
        assert monitor.baseline_set is True

    def test_baseline_not_set_one_below_calibration(self):
        monitor = ConservationMonitor()
        for _ in range(CALIBRATION_PERIOD - 1):
            monitor.record_quality(0.85)
        assert monitor.baseline_set is False

    def test_baseline_is_mean_of_calibration_window(self):
        monitor = ConservationMonitor()
        rng = np.random.default_rng(0)
        q_vals = rng.uniform(0.7, 0.95, CALIBRATION_PERIOD).tolist()
        for q in q_vals:
            monitor.record_quality(q)
        expected = float(np.mean(q_vals[:CALIBRATION_PERIOD]))
        assert monitor.q_baseline == pytest.approx(expected, rel=1e-9)

    def test_baseline_frozen_after_calibration(self):
        """Additional calls beyond CALIBRATION_PERIOD must not update q_baseline."""
        monitor = ConservationMonitor()
        for _ in range(CALIBRATION_PERIOD):
            monitor.record_quality(0.85)
        baseline_frozen = monitor.q_baseline
        for _ in range(100):
            monitor.record_quality(0.0)  # very different quality
        assert monitor.q_baseline == pytest.approx(baseline_frozen)

    def test_no_yellow_warning_under_stable_quality(self):
        monitor = ConservationMonitor()
        for _ in range(CALIBRATION_PERIOD + 200):
            monitor.record_quality(0.9)
        assert monitor.yellow_warning is False

    def test_yellow_warning_triggers_on_severe_quality_drop(self):
        monitor = ConservationMonitor()
        for _ in range(CALIBRATION_PERIOD):
            monitor.record_quality(0.9)
        for _ in range(300):
            monitor.record_quality(0.0)
        assert monitor.yellow_warning is True

    def test_yellow_warning_false_initially(self):
        monitor = ConservationMonitor()
        assert monitor.yellow_warning is False

    def test_yellow_reason_populated_on_alarm(self):
        monitor = ConservationMonitor()
        for _ in range(CALIBRATION_PERIOD):
            monitor.record_quality(0.9)
        for _ in range(300):
            monitor.record_quality(0.0)
        if monitor.yellow_warning:
            assert isinstance(monitor.yellow_reason, str)
            assert len(monitor.yellow_reason) > 0

    def test_layer2_does_not_affect_layer1_status(self):
        """CUSUM alarm (yellow) must not change the GREEN Layer 1 status."""
        monitor = ConservationMonitor()
        for _ in range(CALIBRATION_PERIOD):
            monitor.record_quality(0.9)
        for _ in range(300):
            monitor.record_quality(0.0)
        assert monitor.conservation_status == "GREEN"

    def test_layer1_does_not_affect_layer2(self):
        """RED Layer 1 status must not set baseline or trigger yellow."""
        monitor = ConservationMonitor()
        monitor.update_conservation_signal("RED")
        assert monitor.yellow_warning is False
        assert monitor.baseline_set is False

    def test_two_monitors_have_independent_layer2(self):
        m1 = ConservationMonitor()
        m2 = ConservationMonitor()
        for _ in range(CALIBRATION_PERIOD):
            m1.record_quality(0.9)
        for _ in range(300):
            m1.record_quality(0.0)
        assert m2.yellow_warning is False
        assert m2.baseline_set is False

    def test_record_quality_before_baseline_does_not_crash(self):
        """record_quality() on a fresh monitor must not raise."""
        monitor = ConservationMonitor()
        for _ in range(CALIBRATION_PERIOD // 2):
            monitor.record_quality(0.8)
        assert monitor.baseline_set is False
        assert monitor.yellow_warning is False

    def test_record_quality_with_zero_values(self):
        """All-zero quality values must not crash; only baseline changes."""
        monitor = ConservationMonitor()
        for _ in range(CALIBRATION_PERIOD):
            monitor.record_quality(0.0)
        assert monitor.baseline_set is True
        assert monitor.q_baseline == pytest.approx(0.0, abs=1e-12)

    def test_record_quality_with_one_values(self):
        """All-one quality followed by gradual decline stays non-crash."""
        monitor = ConservationMonitor()
        for _ in range(CALIBRATION_PERIOD):
            monitor.record_quality(1.0)
        for _ in range(50):
            monitor.record_quality(0.5)
        # No crash expected; yellow may or may not have triggered
        assert isinstance(monitor.yellow_warning, bool)


# ── derive_theta_min ──────────────────────────────────────────────────────────

class TestDeriveTheta:
    def test_default_theta_min_formula(self):
        """eta=0.05, n_half=14, t_max=21 → 0.05*196/21."""
        theta = derive_theta_min()
        assert theta == pytest.approx(0.05 * 14.0**2 / 21.0, rel=1e-6)

    def test_custom_theta_min(self):
        actual = derive_theta_min(eta=0.10, n_half=10.0, t_max_days=10.0)
        assert actual == pytest.approx(0.10 * 100.0 / 10.0, rel=1e-6)

    def test_theta_min_zero_n_half(self):
        actual = derive_theta_min(eta=0.05, n_half=0.0, t_max_days=21.0)
        assert actual == pytest.approx(0.0, abs=1e-15)

    def test_theta_min_positive_for_defaults(self):
        assert derive_theta_min() > 0.0

    def test_theta_scales_linearly_with_eta(self):
        t1 = derive_theta_min(eta=0.05, n_half=14.0, t_max_days=21.0)
        t2 = derive_theta_min(eta=0.10, n_half=14.0, t_max_days=21.0)
        assert t2 == pytest.approx(2.0 * t1, rel=1e-9)

    def test_theta_scales_quadratically_with_n_half(self):
        t1 = derive_theta_min(eta=0.05, n_half=7.0, t_max_days=21.0)
        t2 = derive_theta_min(eta=0.05, n_half=14.0, t_max_days=21.0)
        assert t2 == pytest.approx(4.0 * t1, rel=1e-9)


# ── check_conservation ────────────────────────────────────────────────────────

class TestCheckConservation:
    def test_green_when_signal_ge_2x_theta(self):
        theta = derive_theta_min()
        cc = check_conservation(alpha=1.0, q=1.0, V=2.0 * theta + 1.0, theta_min=theta)
        assert cc.status == "GREEN"
        assert cc.passed is True

    def test_amber_when_signal_between_theta_and_2x_theta(self):
        theta = derive_theta_min()
        # signal = 1.5 * theta → AMBER
        cc = check_conservation(alpha=1.0, q=1.0, V=theta * 1.5, theta_min=theta)
        assert cc.status == "AMBER"

    def test_red_when_signal_lt_theta(self):
        theta = derive_theta_min()
        cc = check_conservation(alpha=1.0, q=1.0, V=theta * 0.5, theta_min=theta)
        assert cc.status == "RED"
        assert cc.passed is False

    def test_signal_formula_is_alpha_q_v(self):
        alpha, q, V, theta = 0.4, 0.75, 10.0, 0.1
        cc = check_conservation(alpha=alpha, q=q, V=V, theta_min=theta)
        assert cc.signal == pytest.approx(alpha * q * V, rel=1e-9)

    def test_headroom_formula(self):
        """headroom = signal / theta_min."""
        theta = 1.0
        cc = check_conservation(alpha=1.0, q=1.0, V=5.0, theta_min=theta)
        assert cc.headroom == pytest.approx(5.0 / theta, rel=1e-6)

    def test_theta_min_stored_in_result(self):
        theta = derive_theta_min()
        cc = check_conservation(alpha=0.3, q=0.8, V=50.0, theta_min=theta)
        assert cc.theta_min == pytest.approx(theta, rel=1e-3)

    def test_result_fields_not_nan(self):
        theta = derive_theta_min()
        cc = check_conservation(alpha=0.5, q=0.7, V=20.0, theta_min=theta)
        assert not np.isnan(cc.signal)
        assert not np.isnan(cc.headroom)
        assert cc.status in ("GREEN", "AMBER", "RED")

    def test_green_requires_passed_true(self):
        theta = derive_theta_min()
        cc = check_conservation(alpha=1.0, q=1.0, V=100.0, theta_min=theta)
        if cc.status == "GREEN":
            assert cc.passed is True

    def test_red_requires_passed_false(self):
        theta = derive_theta_min()
        cc = check_conservation(alpha=0.0001, q=0.1, V=1.0, theta_min=theta)
        if cc.status == "RED":
            assert cc.passed is False

    def test_headroom_monotone_increasing_with_signal(self):
        """Higher signal → strictly higher headroom."""
        theta = derive_theta_min()
        cc1 = check_conservation(alpha=0.1, q=0.5, V=20.0, theta_min=theta)
        cc2 = check_conservation(alpha=0.3, q=0.5, V=20.0, theta_min=theta)
        assert cc2.headroom > cc1.headroom

    def test_alpha_zero_signal_is_zero(self):
        theta = derive_theta_min()
        cc = check_conservation(alpha=0.0, q=0.9, V=100.0, theta_min=theta)
        assert cc.signal == pytest.approx(0.0, abs=1e-15)
        assert cc.status == "RED"

    def test_v_zero_signal_is_zero(self):
        theta = derive_theta_min()
        cc = check_conservation(alpha=0.5, q=0.9, V=0.0, theta_min=theta)
        assert cc.signal == pytest.approx(0.0, abs=1e-15)
        assert cc.status == "RED"

    def test_theta_zero_no_crash(self):
        """theta_min=0 must not divide by zero; headroom must be finite or inf."""
        cc = check_conservation(alpha=0.5, q=0.8, V=10.0, theta_min=0.0)
        assert not np.isnan(cc.signal)
        assert not np.isnan(cc.headroom)

    def test_extreme_values_no_overflow(self):
        theta = derive_theta_min()
        cc = check_conservation(alpha=0.99, q=0.99, V=1e6, theta_min=theta)
        assert not np.isinf(cc.signal)
        assert not np.isnan(cc.signal)
        assert cc.status in ("GREEN", "AMBER", "RED")
