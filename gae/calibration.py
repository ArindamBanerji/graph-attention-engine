"""
GAE Calibration — domain-configurable learning hyperparameters.

CalibrationProfile replaces hardcoded constants (ALPHA, LAMBDA_NEG,
EPSILON_DEFAULT) with a structured object that each domain provides.

Reference: docs/gae_design_v5.md §8; blog Eq. 4b, 4c.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CalibrationProfile:
    """Domain-configurable learning hyperparameters.

    Replaces hardcoded constants (ALPHA, LAMBDA_NEG, EPSILON_DEFAULT).
    Each domain provides its own profile via DomainConfig.

    Reference: docs/gae_design_v5.md §8; blog Eq. 4b, 4c.

    Attributes
    ----------
    learning_rate : float
        Base learning rate α (was ALPHA). Default 0.02.
    penalty_ratio : float
        Asymmetric penalty multiplier λ_neg (was LAMBDA_NEG). Default 20.0.
    temperature : float
        Softmax temperature τ (was tau parameter). Default 0.25.
    epsilon_default : float
        Default per-factor decay rate ε (was EPSILON_DEFAULT). Default 0.001.
    discount_strength : float
        A1 confirmation-bias discount ∈ [0, 1]. 0.0 = disabled. Default 0.0.
    decay_class_rates : dict
        Per-class ε rates for A2 hardening.
    factor_decay_classes : dict
        Maps factor_name → decay_class_name for A2 per-factor decay.
        Factors absent from this mapping fall back to the "standard" class.
        Example: {"device_trust": "permanent", "threat_intel_enrichment": "campaign"}.
    extensions : dict
        Reserved for domain-specific extra parameters.
    """

    learning_rate: float = 0.02
    penalty_ratio: float = 20.0
    temperature: float = 0.25
    epsilon_default: float = 0.001
    discount_strength: float = 0.0
    decay_class_rates: dict = field(default_factory=lambda: {
        "permanent": 0.0001,
        "standard": 0.001,
        "campaign": 0.005,
        "transient": 0.02,
    })
    factor_decay_classes: dict = field(default_factory=dict)
    extensions: dict = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Return list of warnings if parameters are out of expected range.

        Reference: docs/gae_design_v5.md §8 (CalibrationProfile validation).

        Returns
        -------
        list[str]
            Empty list when all parameters are in expected ranges.
            One entry per out-of-range parameter.
        """
        warnings = []
        if not (0.001 <= self.learning_rate <= 0.5):
            warnings.append(
                f"learning_rate {self.learning_rate} outside [0.001, 0.5]"
            )
        if not (1.0 <= self.penalty_ratio <= 100.0):
            warnings.append(
                f"penalty_ratio {self.penalty_ratio} outside [1.0, 100.0]"
            )
        if not (0.05 <= self.temperature <= 2.0):
            warnings.append(
                f"temperature {self.temperature} outside [0.05, 2.0]"
            )
        if not (0.0 <= self.discount_strength <= 1.0):
            warnings.append(
                f"discount_strength {self.discount_strength} outside [0.0, 1.0]"
            )
        return warnings


def soc_calibration_profile() -> CalibrationProfile:
    """SOC domain defaults. 20:1 penalty, sharp temperature.

    Reference: docs/gae_design_v5.md §8 (SOC calibration).

    Returns
    -------
    CalibrationProfile
        Profile tuned for security-operations workloads.
    """
    return CalibrationProfile(
        learning_rate=0.02,
        penalty_ratio=20.0,
        temperature=0.25,
        factor_decay_classes={
            "pattern_history": "standard",
            "travel_match": "standard",
            "time_anomaly": "standard",
            "device_trust": "permanent",
            "threat_intel_enrichment": "campaign",
            "asset_criticality": "permanent",
        },
    )


def s2p_calibration_profile() -> CalibrationProfile:
    """S2P domain defaults. 5:1 penalty, softer temperature.

    Reference: docs/gae_design_v5.md §8 (S2P calibration).

    Returns
    -------
    CalibrationProfile
        Profile tuned for source-to-pay / supply-chain workloads.
    """
    return CalibrationProfile(
        learning_rate=0.01,
        penalty_ratio=5.0,
        temperature=0.4,
    )
