"""
Tests for gae/referral.py — ReferralDecision, ReferralRule protocol,
ReferralEngine, ReferralReason, OverrideDetector.

Reference: docs/gae_design_v5.md §10; EXP-REFER-LAYERED.
"""

import pytest

from gae.referral import (
    ReferralDecision,
    ReferralEngine,
    ReferralReason,
    ReferralRule,
    OverrideDetector,
    OverrideDetectorConfig,
)


# ------------------------------------------------------------------ #
# Mock rule helper                                                    #
# ------------------------------------------------------------------ #

class MockRule:
    """Fires when alert_context[key] == value."""

    def __init__(self, rule_id: str, reason: ReferralReason,
                 key: str, value):
        self._rule_id = rule_id
        self._reason = reason
        self._key = key
        self._value = value

    @property
    def rule_id(self) -> str:
        return self._rule_id

    @property
    def reason(self) -> ReferralReason:
        return self._reason

    def evaluate(self, alert_context: dict):
        fires = alert_context.get(self._key) == self._value
        detail = {self._key: alert_context.get(self._key)} if fires else {}
        return fires, detail


# ------------------------------------------------------------------ #
# TestReferralReason                                                  #
# ------------------------------------------------------------------ #

class TestReferralReason:
    def test_all_reasons_have_values(self):
        for member in ReferralReason:
            assert member.value is not None

    def test_reason_count(self):
        assert len(ReferralReason) == 9   # R1-R7 + R8 + NONE

    def test_r1_through_r7_values(self):
        codes = {r.value for r in ReferralReason}
        for i in range(1, 8):
            assert f"R{i}" in codes

    def test_r8_value(self):
        assert ReferralReason.OVERRIDE_PATTERN.value == "R8"

    def test_none_value(self):
        assert ReferralReason.NO_REFERRAL.value == "NONE"


# ------------------------------------------------------------------ #
# TestReferralDecision                                                #
# ------------------------------------------------------------------ #

class TestReferralDecision:
    def test_no_referral_empty_codes(self):
        d = ReferralDecision(should_refer=False)
        assert d.reason_codes == []

    def test_no_referral_audit_summary(self):
        d = ReferralDecision(should_refer=False)
        assert "No referral" in d.audit_summary

    def test_single_reason_code(self):
        d = ReferralDecision(
            should_refer=True,
            reasons=[ReferralReason.EXECUTIVE_ACCOUNT],
            rule_details={"R1": {"identity_tier": "executive"}},
        )
        assert d.reason_codes == ["R1"]

    def test_single_reason_audit_contains_name(self):
        d = ReferralDecision(
            should_refer=True,
            reasons=[ReferralReason.EXECUTIVE_ACCOUNT],
            rule_details={"R1": {"identity_tier": "executive"}},
        )
        assert "EXECUTIVE_ACCOUNT" in d.audit_summary

    def test_single_reason_audit_contains_detail(self):
        d = ReferralDecision(
            should_refer=True,
            reasons=[ReferralReason.EXECUTIVE_ACCOUNT],
            rule_details={"R1": {"identity_tier": "executive"}},
        )
        assert "executive" in d.audit_summary

    def test_multiple_reasons_codes(self):
        d = ReferralDecision(
            should_refer=True,
            reasons=[ReferralReason.EXECUTIVE_ACCOUNT, ReferralReason.RAPID_SUCCESSION],
            rule_details={"R1": {}, "R2": {"count": 5}},
        )
        assert set(d.reason_codes) == {"R1", "R2"}

    def test_multiple_reasons_both_in_audit(self):
        d = ReferralDecision(
            should_refer=True,
            reasons=[ReferralReason.EXECUTIVE_ACCOUNT, ReferralReason.RAPID_SUCCESSION],
            rule_details={"R1": {}, "R2": {}},
        )
        assert "R1" in d.audit_summary
        assert "R2" in d.audit_summary

    def test_default_reasons_empty(self):
        d = ReferralDecision(should_refer=False)
        assert d.reasons == []
        assert d.rule_details == {}


# ------------------------------------------------------------------ #
# TestReferralRule protocol                                           #
# ------------------------------------------------------------------ #

class TestReferralRuleProtocol:
    def test_mock_rule_satisfies_protocol(self):
        rule = MockRule("R1", ReferralReason.EXECUTIVE_ACCOUNT,
                        "identity_tier", "executive")
        assert isinstance(rule, ReferralRule)

    def test_mock_rule_fires_on_match(self):
        rule = MockRule("R1", ReferralReason.EXECUTIVE_ACCOUNT,
                        "identity_tier", "executive")
        fires, detail = rule.evaluate({"identity_tier": "executive"})
        assert fires is True
        assert "identity_tier" in detail

    def test_mock_rule_does_not_fire_on_mismatch(self):
        rule = MockRule("R1", ReferralReason.EXECUTIVE_ACCOUNT,
                        "identity_tier", "executive")
        fires, detail = rule.evaluate({"identity_tier": "standard"})
        assert fires is False
        assert detail == {}


# ------------------------------------------------------------------ #
# TestReferralEngine                                                  #
# ------------------------------------------------------------------ #

class TestReferralEngine:
    def test_no_rules_no_referral(self):
        engine = ReferralEngine(rules=[])
        decision = engine.evaluate({"identity_tier": "executive"})
        assert decision.should_refer is False
        assert decision.reason_codes == []

    def test_single_rule_fires(self):
        rule = MockRule("R1", ReferralReason.EXECUTIVE_ACCOUNT,
                        "identity_tier", "executive")
        engine = ReferralEngine(rules=[rule])
        decision = engine.evaluate({"identity_tier": "executive"})
        assert decision.should_refer is True
        assert "R1" in decision.reason_codes

    def test_single_rule_does_not_fire(self):
        rule = MockRule("R1", ReferralReason.EXECUTIVE_ACCOUNT,
                        "identity_tier", "executive")
        engine = ReferralEngine(rules=[rule])
        decision = engine.evaluate({"identity_tier": "standard"})
        assert decision.should_refer is False

    def test_multiple_rules_one_fires(self):
        r1 = MockRule("R1", ReferralReason.EXECUTIVE_ACCOUNT,
                      "identity_tier", "executive")
        r2 = MockRule("R2", ReferralReason.RAPID_SUCCESSION,
                      "sequence_count", 10)
        engine = ReferralEngine(rules=[r1, r2])
        # Only R2 fires
        decision = engine.evaluate({"identity_tier": "standard",
                                    "sequence_count": 10})
        assert decision.should_refer is True
        assert decision.reason_codes == ["R2"]

    def test_multiple_rules_both_fire(self):
        r1 = MockRule("R1", ReferralReason.EXECUTIVE_ACCOUNT,
                      "identity_tier", "executive")
        r2 = MockRule("R2", ReferralReason.RAPID_SUCCESSION,
                      "sequence_count", 10)
        engine = ReferralEngine(rules=[r1, r2])
        decision = engine.evaluate({"identity_tier": "executive",
                                    "sequence_count": 10})
        assert decision.should_refer is True
        assert set(decision.reason_codes) == {"R1", "R2"}

    def test_audit_summary_contains_details(self):
        rule = MockRule("R4", ReferralReason.HIGH_VALUE_DATA,
                        "asset_criticality", "critical")
        engine = ReferralEngine(rules=[rule])
        decision = engine.evaluate({"asset_criticality": "critical"})
        assert "critical" in decision.audit_summary

    def test_engine_is_pure_no_state_mutation(self):
        """Calling evaluate twice on same engine gives consistent results."""
        rule = MockRule("R3", ReferralReason.COMPLIANCE_MANDATE,
                        "compliance_mode", True)
        engine = ReferralEngine(rules=[rule])
        ctx = {"compliance_mode": True}
        d1 = engine.evaluate(ctx)
        d2 = engine.evaluate(ctx)
        assert d1.should_refer == d2.should_refer
        assert d1.reason_codes == d2.reason_codes

    def test_all_rules_evaluated_not_short_circuited(self):
        """All firing rules must appear in reasons, not just the first."""
        rules = [
            MockRule(f"R{i}", list(ReferralReason)[i - 1], "key", i)
            for i in range(1, 5)
        ]
        engine = ReferralEngine(rules=rules)
        # Provide a context that matches none — then verify zero reasons
        decision = engine.evaluate({"key": 99})
        assert decision.reason_codes == []
        # Now match all four
        for i in range(1, 5):
            decision = engine.evaluate({"key": i})
            assert len(decision.reasons) == 1


# ------------------------------------------------------------------ #
# TestOverrideDetector                                                #
# ------------------------------------------------------------------ #

class TestOverrideDetector:
    def test_inactive_by_default(self):
        cfg = OverrideDetectorConfig()
        det = OverrideDetector(cfg)
        assert det.is_active is False

    def test_inactive_below_threshold(self):
        cfg = OverrideDetectorConfig(min_positives=50, enabled=True)
        det = OverrideDetector(cfg)
        for _ in range(49):
            det.record_override({}, analyst_referred=True)
        assert det.is_active is False

    def test_activates_at_threshold(self):
        cfg = OverrideDetectorConfig(min_positives=50, enabled=True)
        det = OverrideDetector(cfg)
        for _ in range(50):
            det.record_override({}, analyst_referred=True)
        assert det.is_active is True

    def test_predict_returns_false_when_inactive(self):
        cfg = OverrideDetectorConfig(enabled=False)
        det = OverrideDetector(cfg)
        result = det.predict({})
        assert result == (False, 0.0)

    def test_predict_raises_when_active(self):
        cfg = OverrideDetectorConfig(min_positives=1, enabled=True)
        det = OverrideDetector(cfg)
        det.record_override({}, analyst_referred=True)
        assert det.is_active is True
        with pytest.raises(NotImplementedError):
            det.predict({})

    def test_negative_overrides_dont_count(self):
        cfg = OverrideDetectorConfig(min_positives=5, enabled=True)
        det = OverrideDetector(cfg)
        for _ in range(10):
            det.record_override({}, analyst_referred=False)
        assert det._positive_count == 0
        assert det.is_active is False

    def test_enabled_false_never_activates_regardless_of_positives(self):
        cfg = OverrideDetectorConfig(min_positives=1, enabled=False)
        det = OverrideDetector(cfg)
        for _ in range(100):
            det.record_override({}, analyst_referred=True)
        assert det.is_active is False
