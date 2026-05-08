"""Tests for conservation transition state machine and OLS alarm reset."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from gae.convergence import ConservationStateMachine, OLSMonitor
from gae.profile_scorer import ProfileScorer


def _make_scorer(auto_pause_on_amber: bool = True) -> ProfileScorer:
    mu = np.full((2, 2, 3), 0.5, dtype=np.float64)
    return ProfileScorer(
        mu=mu,
        actions=["a", "b"],
        categories=["cat_a", "cat_b"],
        auto_pause_on_amber=auto_pause_on_amber,
    )


def _alarm_monitor() -> OLSMonitor:
    monitor = OLSMonitor()
    monitor.cusum = 3.5
    monitor.yellow_warning = True
    monitor.yellow_reason = "alarm"
    monitor.baseline_ols = 1.2
    monitor.baseline_frozen = True
    monitor.ols_history.extend([1.2, 1.1, 1.0])
    return monitor


def test_state_machine_default_initial_state_is_calibrating():
    sm = ConservationStateMachine()
    assert sm.state == "CALIBRATING"


def test_state_machine_custom_initial_state_works():
    sm = ConservationStateMachine(initial_state="GREEN")
    assert sm.state == "GREEN"


def test_state_machine_invalid_initial_state_raises():
    with pytest.raises(ValueError):
        ConservationStateMachine(initial_state="UNKNOWN")


def test_state_machine_transition_changes_state():
    sm = ConservationStateMachine()
    assert sm.transition("GREEN") is True
    assert sm.state == "GREEN"


def test_state_machine_same_state_noop_returns_true():
    sm = ConservationStateMachine(initial_state="GREEN")
    assert sm.transition("GREEN") is True
    assert sm.state == "GREEN"


def test_state_machine_invalid_target_returns_false_and_keeps_state():
    sm = ConservationStateMachine(initial_state="GREEN")
    assert sm.transition("UNKNOWN") is False
    assert sm.state == "GREEN"


def test_state_machine_specific_handler_fires():
    sm = ConservationStateMachine(initial_state="AMBER")
    calls = []
    sm.register_handler("AMBER", "GREEN", lambda old, new: calls.append((old, new)))
    assert sm.transition("GREEN") is True
    assert calls == [("AMBER", "GREEN")]


def test_state_machine_handler_does_not_fire_for_different_transition():
    sm = ConservationStateMachine(initial_state="RED")
    handler = Mock()
    sm.register_handler("AMBER", "GREEN", handler)
    assert sm.transition("GREEN") is True
    handler.assert_not_called()


def test_state_machine_wildcard_handler_fires():
    sm = ConservationStateMachine(initial_state="RED")
    calls = []
    sm.register_handler("*", "GREEN", lambda old, new: calls.append((old, new)))
    assert sm.transition("GREEN") is True
    assert calls == [("RED", "GREEN")]


def test_state_machine_guard_blocks_before_state_change():
    sm = ConservationStateMachine(initial_state="AMBER")
    handler = Mock()
    sm.register_guard("AMBER", "GREEN", lambda: False)
    sm.register_handler("AMBER", "GREEN", handler)
    assert sm.transition("GREEN") is False
    assert sm.state == "AMBER"
    handler.assert_not_called()


def test_state_machine_handler_exception_does_not_block_transition():
    sm = ConservationStateMachine(initial_state="AMBER")

    def fail(old_state, new_state):
        raise RuntimeError("boom")

    sm.register_handler("AMBER", "GREEN", fail)
    assert sm.transition("GREEN") is True
    assert sm.state == "GREEN"


def test_state_machine_specific_handler_fires_before_wildcard():
    sm = ConservationStateMachine(initial_state="AMBER")
    calls = []
    sm.register_handler("AMBER", "GREEN", lambda old, new: calls.append("specific"))
    sm.register_handler("*", "GREEN", lambda old, new: calls.append("wildcard"))
    assert sm.transition("GREEN") is True
    assert calls == ["specific", "wildcard"]


def test_ols_reset_alarm_clears_transient_alarm_state():
    monitor = _alarm_monitor()
    monitor.reset_alarm("AMBER", "GREEN")
    assert monitor.cusum == 0.0
    assert monitor.yellow_warning is False
    assert monitor.yellow_reason == ""


def test_ols_reset_alarm_preserves_baseline_and_history():
    monitor = _alarm_monitor()
    history_before = list(monitor.ols_history)
    monitor.reset_alarm("RED", "GREEN")
    assert monitor.baseline_ols == pytest.approx(1.2)
    assert monitor.baseline_frozen is True
    assert monitor.ols_history == history_before


def test_ols_reset_alarm_accepts_optional_transition_args():
    monitor = _alarm_monitor()
    monitor.reset_alarm()
    assert monitor.cusum == 0.0
    assert monitor.yellow_warning is False


def test_amber_to_green_triggers_ols_reset():
    scorer = _make_scorer()
    monitor = _alarm_monitor()
    scorer.register_ols_monitor(monitor)
    scorer.set_conservation_status("AMBER")
    scorer.set_conservation_status("GREEN")
    assert monitor.cusum == 0.0
    assert monitor.yellow_warning is False
    assert monitor.yellow_reason == ""


def test_red_to_green_triggers_ols_reset():
    scorer = _make_scorer()
    monitor = _alarm_monitor()
    scorer.register_ols_monitor(monitor)
    scorer.set_conservation_status("RED")
    scorer.set_conservation_status("GREEN")
    assert monitor.cusum == 0.0
    assert monitor.yellow_warning is False


def test_green_to_amber_does_not_trigger_ols_reset():
    scorer = _make_scorer()
    monitor = _alarm_monitor()
    scorer.register_ols_monitor(monitor)
    scorer.set_conservation_status("GREEN")
    scorer.set_conservation_status("AMBER")
    assert monitor.cusum == pytest.approx(3.5)
    assert monitor.yellow_warning is True


def test_blocked_transition_does_not_reset_ols():
    scorer = _make_scorer()
    monitor = _alarm_monitor()
    scorer.register_ols_monitor(monitor)
    scorer.set_conservation_status("AMBER")
    scorer.conservation_state_machine.register_guard("AMBER", "GREEN", lambda: False)
    scorer.set_conservation_status("GREEN")
    assert monitor.cusum == pytest.approx(3.5)
    assert monitor.yellow_warning is True


def test_same_state_green_does_not_call_ols_handler():
    scorer = _make_scorer()
    monitor = _alarm_monitor()
    scorer.register_ols_monitor(monitor)
    scorer.set_conservation_status("GREEN")
    monitor.cusum = 4.0
    monitor.yellow_warning = True
    scorer.set_conservation_status("GREEN")
    assert monitor.cusum == pytest.approx(4.0)
    assert monitor.yellow_warning is True


def test_profile_scorer_set_amber_pauses():
    scorer = _make_scorer()
    scorer.set_conservation_status("AMBER")
    assert scorer.conservation_status == "AMBER"
    assert scorer.is_paused is True


def test_profile_scorer_set_red_pauses():
    scorer = _make_scorer()
    scorer.set_conservation_status("RED")
    assert scorer.conservation_status == "RED"
    assert scorer.is_paused is True


def test_profile_scorer_set_green_resumes():
    scorer = _make_scorer()
    scorer.set_conservation_status("AMBER")
    assert scorer.is_paused is True
    scorer.set_conservation_status("GREEN")
    assert scorer.conservation_status == "GREEN"
    assert scorer.is_paused is False


def test_set_conservation_status_invalid_status_preserves_previous_status():
    scorer = _make_scorer()
    scorer.set_conservation_status("GREEN")
    previous_status = scorer._conservation_status
    previous_state = scorer.conservation_state_machine.state

    scorer.set_conservation_status("UNKNOWN")

    assert scorer._conservation_status == previous_status
    assert scorer.conservation_state_machine.state == previous_state


def test_profile_scorer_conservation_state_machine_property():
    scorer = _make_scorer()
    assert isinstance(scorer.conservation_state_machine, ConservationStateMachine)


def test_profile_scorer_register_ols_monitor_wires_amber_green_reset():
    scorer = _make_scorer()
    monitor = _alarm_monitor()
    scorer.register_ols_monitor(monitor)
    scorer.set_conservation_status("AMBER")
    scorer.set_conservation_status("GREEN")
    assert monitor.yellow_warning is False


def test_profile_scorer_register_ols_monitor_wires_red_green_reset():
    scorer = _make_scorer()
    monitor = _alarm_monitor()
    scorer.register_ols_monitor(monitor)
    scorer.set_conservation_status("RED")
    scorer.set_conservation_status("GREEN")
    assert monitor.yellow_warning is False
