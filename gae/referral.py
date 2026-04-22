"""
Referral routing — domain-agnostic protocol and engine.

A referral is a VETO mechanism: any firing rule overrides auto-approve
at any confidence level. Action routing (ProfileScorer) and referral
routing (ReferralEngine) are independent pipelines that both run on
every alert. Referral wins if it fires.

Architecture (EXP-REFER-LAYERED, March 21):
  - Rules R1-R7: 72.7% DR, 12% FPR, 978 net min/100 alerts
  - Confidence gate: REJECTED (14% precision, active harm)
  - Override learning (R8): deferred to v6.5 (data-gated, 50+ positives)
  - gae/referral.py: protocol + engine only (domain-agnostic)
  - SOC-specific rules R1-R7: live in SOC repo, implement ReferralRule

Phase 2 (v6.5):
  OverrideDetector (R8) activates once 50+ production positives collected.
  Stub defined here — interface contract only, NotImplementedError at v6.0.

Reference: docs/gae_design_v10_6.md §10; EXP-REFER-LAYERED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Protocol, Tuple, runtime_checkable


# ------------------------------------------------------------------ #
# Reason enum                                                         #
# ------------------------------------------------------------------ #

class ReferralReason(Enum):
    """
    Canonical referral reason codes.

    R1-R7: deterministic rule-based reasons (v6.0).
    R8:    learned override pattern (v6.5, data-gated).
    NONE:  sentinel for no-referral decisions.

    SOC implements R1-R7 as concrete ReferralRule objects in the SOC repo.
    S2P implements its own set. gae/ defines only the protocol.

    Reference: EXP-REFER-LAYERED §3; docs/gae_design_v10_6.md §10.
    """

    EXECUTIVE_ACCOUNT  = "R1"
    RAPID_SUCCESSION   = "R2"
    COMPLIANCE_MANDATE = "R3"
    HIGH_VALUE_DATA    = "R4"
    ACTIVE_INCIDENT    = "R5"
    NEW_ASSET          = "R6"
    CROSS_CATEGORY     = "R7"
    OVERRIDE_PATTERN   = "R8"   # Phase 2 — learned, v6.5
    NO_REFERRAL        = "NONE"


# ------------------------------------------------------------------ #
# ReferralDecision                                                    #
# ------------------------------------------------------------------ #

@dataclass
class ReferralDecision:
    """
    Output of ReferralEngine.evaluate().

    Immutable view of which rules fired and why. Designed for Evidence Ledger
    persistence and human-readable audit trails.

    Attributes
    ----------
    should_refer : bool
        True if ANY rule fired. Overrides auto-approve regardless of confidence.
    reasons : list[ReferralReason]
        All reasons whose rules fired. Empty if should_refer=False.
    rule_details : dict
        Per-rule detail dicts keyed by reason code string (e.g. "R1").
        Empty if should_refer=False.

    Reference: EXP-REFER-LAYERED §4; docs/gae_design_v10_6.md §10.
    """

    should_refer: bool
    reasons: List[ReferralReason] = field(default_factory=list)
    rule_details: Dict[str, dict] = field(default_factory=dict)

    @property
    def reason_codes(self) -> List[str]:
        """
        Ordered list of reason code strings for each firing rule.

        Returns e.g. ["R1", "R2"] — empty list if not referred.

        Reference: EXP-REFER-LAYERED §4.
        """
        return [r.value for r in self.reasons]

    @property
    def audit_summary(self) -> str:
        """
        Human-readable summary for the Evidence Ledger.

        Format when referred:
          "R1 (EXECUTIVE_ACCOUNT): {'identity_tier': 'executive'}; R2 ..."
        Format when not referred:
          "No referral rules triggered."

        Reference: EXP-REFER-LAYERED §4; Evidence Ledger schema.
        """
        if not self.should_refer:
            return "No referral rules triggered."

        parts = []
        for reason in self.reasons:
            detail = self.rule_details.get(reason.value, {})
            parts.append(f"{reason.value} ({reason.name}): {detail}")
        return "; ".join(parts)


# ------------------------------------------------------------------ #
# ReferralRule protocol                                               #
# ------------------------------------------------------------------ #

@runtime_checkable
class ReferralRule(Protocol):
    """
    Protocol for domain-agnostic referral rules.

    Each rule is a pure function: entity_context → (fires, detail).
    No state. No ML. No side effects. Fully auditable.

    SOC implements R1-R7 in the SOC repo; S2P implements its own set.
    gae/ defines only this protocol — zero domain knowledge here.

    Properties
    ----------
    rule_id : str
        Short identifier, e.g. "R1", "R2".
    reason : ReferralReason
        The enum value this rule maps to on firing.

    Method
    ------
    evaluate(entity_context: dict) → tuple[bool, dict]
        entity_context is a plain dict with domain-specific keys.
        Returns (fires: bool, detail: dict) where detail is included
        in the audit trail when fires=True.

    Reference: EXP-REFER-LAYERED §2; docs/gae_design_v10_6.md §10.
    """

    @property
    def rule_id(self) -> str:
        """Short rule identifier, e.g. 'R1'."""
        ...

    @property
    def reason(self) -> ReferralReason:
        """Enum reason this rule maps to."""
        ...

    def evaluate(self, entity_context: dict) -> Tuple[bool, dict]:
        """
        Evaluate this rule against the alert context.

        Parameters
        ----------
        entity_context : dict
            Domain-specific keys such as identity_tier, sequence_count,
            category, compliance_mode, asset_criticality, stage1_action.

        Returns
        -------
        tuple[bool, dict]
            (fires, detail) — detail is included in audit trail if fires.
        """
        ...


# ------------------------------------------------------------------ #
# ReferralEngine                                                      #
# ------------------------------------------------------------------ #

@dataclass
class ReferralEngine:
    """
    Evaluates all registered rules against an alert context.

    ANY rule firing → should_refer=True. Collects all firing reasons.
    Pure function: no state mutation, no side effects, fully deterministic.

    Usage
    -----
    engine = ReferralEngine(rules=[R1Rule(), R2Rule(), ...])
    decision = engine.evaluate(entity_context)
    if decision.should_refer:
        route_to_analyst(alert, decision.audit_summary)

    Reference: EXP-REFER-LAYERED §2; docs/gae_design_v10_6.md §10.
    """

    rules: List  # list of objects implementing ReferralRule protocol

    def evaluate(self, entity_context: dict) -> ReferralDecision:
        """
        Evaluate all rules against entity_context.

        Evaluates every rule (no short-circuit) so the audit trail captures
        all firing rules even when the first would have been sufficient.

        Parameters
        ----------
        entity_context : dict
            Domain-specific context for rule evaluation.

        Returns
        -------
        ReferralDecision
            should_refer=True if any rule fired; all firing reasons collected.
        """
        fired_reasons: List[ReferralReason] = []
        fired_details: Dict[str, dict] = {}

        for rule in self.rules:
            fires, detail = rule.evaluate(entity_context)
            if fires:
                fired_reasons.append(rule.reason)
                fired_details[rule.rule_id] = detail

        return ReferralDecision(
            should_refer=bool(fired_reasons),
            reasons=fired_reasons,
            rule_details=fired_details,
        )


# ------------------------------------------------------------------ #
# OverrideDetector — Phase 2 stub (v6.5)                             #
# ------------------------------------------------------------------ #

@dataclass
class OverrideDetectorConfig:
    """
    Configuration for the learned override detector (R8).

    Activation is data-gated: enabled=False AND min_positives not reached
    until 50+ production referral positives have been collected.

    Reference: EXP-REFER-LAYERED §5; v6.5 roadmap.
    """

    min_positives: int = 50       # data gate — must reach this before activating
    retrain_interval: int = 500   # decisions between model refreshes
    fpr_threshold: float = 0.05   # target false positive rate
    enabled: bool = False         # explicitly disabled at v6.0


class OverrideDetector:
    """
    Learned override pattern detector — Phase 2, v6.5.

    NOT IMPLEMENTED at v6.0. Defines interface contract only.
    predict() raises NotImplementedError when is_active=True.
    predict() returns (False, 0.0) when inactive (safe default).

    Activation requires BOTH:
      - config.enabled = True
      - _positive_count >= config.min_positives

    This prevents accidental activation before sufficient data exists.
    50 positives ≈ 6-12 months of referral feedback at typical volumes.

    Reference: EXP-REFER-LAYERED §5; docs/gae_design_v10_6.md §10.
    """

    def __init__(self, config: OverrideDetectorConfig) -> None:
        """
        Parameters
        ----------
        config : OverrideDetectorConfig
            Activation and training configuration.
        """
        self.config = config
        self._positive_count: int = 0

    @property
    def is_active(self) -> bool:
        """
        True only when enabled AND enough positives have been collected.

        Both conditions must hold simultaneously — enabled alone does not
        activate (protects against premature activation on new deployments).
        """
        return (
            self.config.enabled
            and self._positive_count >= self.config.min_positives
        )

    def record_override(self, entity_context: dict, analyst_referred: bool) -> None:
        """
        Record one labelled decision from the analyst.

        Only positive labels (analyst_referred=True) count toward the data gate.
        Negative labels are ignored — base rate is high enough that negatives
        do not constrain activation.

        Parameters
        ----------
        entity_context : dict
            Feature context for the alert (stored for future training).
        analyst_referred : bool
            True if analyst chose to refer; False if auto-approve was accepted.
        """
        if analyst_referred:
            self._positive_count += 1

    def predict(self, entity_context: dict) -> Tuple[bool, float]:
        """
        Predict whether this alert matches an override pattern.

        Returns (False, 0.0) when inactive — safe pass-through default.
        Raises NotImplementedError when active (implementation pending v6.5).

        Parameters
        ----------
        entity_context : dict
            Feature context for prediction.

        Returns
        -------
        tuple[bool, float]
            (should_refer, confidence) — (False, 0.0) when inactive.

        Raises
        ------
        NotImplementedError
            When is_active=True. Model training not yet implemented.
        """
        if not self.is_active:
            return (False, 0.0)
        raise NotImplementedError(
            "OverrideDetector.predict() is not implemented at v6.0. "
            "Activate only after v6.5 model training is complete."
        )
