"""
GAE Calibration — domain-configurable learning hyperparameters.

CalibrationProfile replaces hardcoded constants (ALPHA, LAMBDA_NEG,
EPSILON_DEFAULT) with a structured object that each domain provides.

Reference: docs/gae_design_v5.md §8; blog Eq. 4b, 4c.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, NamedTuple

import numpy as np


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


# ── Conservation law primitives ──────────────────────────────────────────────
# Source: research_note_v3, three-judge validated (Borkar/Wu/Kazerouni),
# Bridge B Phase C v3 (re-convergence compounding).


class ConservationCheck(NamedTuple):
    """Result of a conservation law check α·q·V ≥ θ_min."""

    signal: float     # α·q·V current value
    theta_min: float  # floor threshold
    headroom: float   # signal / theta_min (>1 = healthy)
    status: str       # 'GREEN' (≥2×θ), 'AMBER' (≥θ), 'RED' (<θ)
    passed: bool      # signal ≥ theta_min


def derive_theta_min(
    eta: float = 0.05,
    n_half: float = 14.0,
    t_max_days: float = 21.0,
) -> float:
    """
    Minimum conservation law signal for centroid convergence.

    θ_min = η × N_half² / T_max

    Ensures enough verified decisions flow through the system for
    centroids to track within the convergence half-life.

    SOC default: 0.05 × 14² / 21 ≈ 0.467 (T_max=21 days canonical).
    S2P default: 0.05 × 14² / 26 ≈ 0.377 (longer cycle).

    Reference: research_note_v3; N_half from math_synopsis_v9 §5.

    Parameters
    ----------
    eta : float
        Learning rate.
    n_half : float
        Convergence half-life in decisions.
    t_max_days : float
        Maximum acceptable convergence time in days.

    Returns
    -------
    float
        θ_min threshold.
    """
    assert t_max_days > 0, f"t_max_days must be positive, got {t_max_days}"
    return float(eta * n_half ** 2 / t_max_days)


def check_conservation(
    alpha: float,
    q: float,
    V: float,
    theta_min: float,
) -> ConservationCheck:
    """
    Check conservation law: α·q·V ≥ θ_min.

    Status levels (research_note_v3):
        GREEN : signal ≥ 2 × θ_min — healthy, well above floor.
        AMBER : θ_min ≤ signal < 2 × θ_min — thinning, monitor closely.
        RED   : signal < θ_min — breach, learning signal insufficient.

    Reference: research_note_v3; Bridge B Phase C v3.

    Parameters
    ----------
    alpha : float
        Override rate — fraction of decisions where analyst changed action.
    q : float
        Override quality — fraction of overrides that improved outcome.
    V : float
        Verified decisions per day.
    theta_min : float
        Floor threshold from derive_theta_min().

    Returns
    -------
    ConservationCheck
        Named tuple with signal, theta_min, headroom, status, passed.
    """
    signal = alpha * q * V
    headroom = signal / theta_min if theta_min > 0 else float('inf')

    if signal >= 2 * theta_min:
        status = 'GREEN'
    elif signal >= theta_min:
        status = 'AMBER'
    else:
        status = 'RED'

    return ConservationCheck(
        signal=round(signal, 4),
        theta_min=round(theta_min, 4),
        headroom=round(headroom, 3),
        status=status,
        passed=bool(signal >= theta_min),
    )


def compute_breach_window(
    signal_variance: float,
    signal_mean: float,
    theta_min: float,
    delta: float = 0.05,
) -> float:
    """
    Hoeffding-derived breach detection window in days.

    How many days of observations are needed to detect that the true
    signal has dropped below θ_min with confidence (1−δ)?

    W = R² × ln(1/δ) / (2 × (μ − θ_min)²),  R = 4σ (sub-Gaussian range).

    At healthy signal (μ >> θ_min): W is very small (fast detection).
    At marginal signal (μ ≈ θ_min): W grows — the dangerous case.
    Engineering choice W=14 is for the marginal case.

    Reference: research_note_v3 §Hoeffding derivation.

    Parameters
    ----------
    signal_variance : float
        Variance of daily α·q·V signal.
    signal_mean : float
        Mean daily α·q·V signal.
    theta_min : float
        Conservation floor.
    delta : float
        Confidence parameter (default 0.05).

    Returns
    -------
    float
        Window in days. Returns float('inf') if signal_mean ≤ theta_min.
    """
    if signal_mean <= theta_min:
        return float('inf')

    R = 4.0 * float(np.sqrt(signal_variance))
    gap = signal_mean - theta_min
    W = R ** 2 * float(np.log(1.0 / delta)) / (2.0 * gap ** 2)
    return max(1.0, W)


def compute_optimal_tau(
    centroid_covariance: np.ndarray,
    tau_range: Tuple[float, float] = (0.05, 0.20),
) -> float:
    """
    Gain-scheduled τ from centroid covariance matrix Σ_c.

    τ_opt = τ_min + (τ_max − τ_min) × (1 − tr(Σ_c) / tr_max)

    Higher Σ_c (uncertain centroids) → lower τ (softer decisions).
    Lower Σ_c (confident centroids) → higher τ (sharper decisions).

    Used by GainScheduler at v6.5. Math ships at v6.0.
    Reference: research_note_v3 §gain-scheduling.

    Parameters
    ----------
    centroid_covariance : np.ndarray
        Σ_c matrix (d × d) from recent centroid updates.
    tau_range : tuple
        (τ_min, τ_max) bounds.

    Returns
    -------
    float
        Optimal τ value within tau_range.
    """
    assert centroid_covariance.ndim == 2, (
        f"centroid_covariance must be 2-D, got shape {centroid_covariance.shape}"
    )
    tau_min, tau_max = tau_range
    tr_sigma = float(np.trace(centroid_covariance))
    tr_max = 1.0  # maximum expected trace
    confidence = max(0.0, min(1.0, 1.0 - tr_sigma / tr_max))
    return float(tau_min + (tau_max - tau_min) * confidence)


def compute_transfer_prior(
    calibrated_centroids: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical Bayes prior from calibrated category centroids.

    When a new category is added, initialize its centroids from the
    population mean and variance of existing calibrated categories.
    Better than uniform initialization.

    Used by TransferPriorManager at v7.0. Math ships at v6.0.
    Reference: research_note_v3 §transfer-prior.

    Parameters
    ----------
    calibrated_centroids : dict
        {category_name: centroid_array (A × d)} — only converged categories.

    Returns
    -------
    (prior_mean, prior_std) : both shape (A, d).
        prior_mean = mean across categories.
        prior_std  = std across categories (uncertainty estimate).
        Returns (zeros(1), ones(1)) when dict is empty.
    """
    if not calibrated_centroids:
        return np.zeros(1), np.ones(1)

    arrays = list(calibrated_centroids.values())
    stacked = np.stack(arrays)  # shape: (n_categories, A, d)
    assert stacked.ndim == 3, f"Expected 3-D stack, got shape {stacked.shape}"
    prior_mean = np.mean(stacked, axis=0)
    prior_std = np.std(stacked, axis=0)
    return prior_mean, prior_std


def compute_eta_override(
    eta_confirm: float = 0.05,
    mean_quality: float = 0.75,
    quality_variance: float = 0.03,
    safety_margin: float = 0.5,
) -> float:
    """
    Compute optimal override learning rate from analyst quality distribution.

    Derived from signal-to-noise optimization:
        η_override* ∝ (2q̄−1) / (2σ²_q + (2q̄−1))

    Theoretical prediction validated against Q5 sweep on 9 personas.
    Formula matches the empirical optimum within one sweep step.
    Empirical default: 0.01 (captures 80%+ of gain across all personas).

    Parameters
    ----------
    eta_confirm : float
        Confirmation learning rate (default 0.05).
    mean_quality : float
        Mean analyst override quality ∈ (0, 1). Values ≤ 0.5 return the floor.
    quality_variance : float
        Variance of analyst quality distribution.
    safety_margin : float
        Conservative multiplier (0.5 recommended by Q5 sweep).

    Returns
    -------
    float
        Recommended η_override value. Floored at 0.005 when signal ≤ 0.

    Source: Q5 sweep (9 personas × 6 η values), roadmap session formula.
    """
    signal = 2.0 * mean_quality - 1.0
    noise = 2.0 * quality_variance
    if signal <= 0:
        return 0.005  # floor: near-zero learning from overrides
    ratio = signal / max(noise + signal, 0.01)
    return round(eta_confirm * ratio * safety_margin, 4)


def check_meta_conservation(
    new_prior: np.ndarray,
    calibrated_centroids: Dict[str, np.ndarray],
    old_prior: np.ndarray,
    epsilon: float = 0.05,
) -> Tuple[bool, Dict]:
    """
    Meta-conservation gate for transfer priors.

    Ensures updating the prior from new calibrated data doesn't
    destabilize already-calibrated categories. The new prior must not
    diverge from the old prior by more than ε per component.

    Used by TransferPriorManager at v7.0. Math ships at v6.0.
    Reference: research_note_v3 §meta-conservation.

    Parameters
    ----------
    new_prior : np.ndarray
        Proposed new prior (A × d).
    calibrated_centroids : dict
        Currently calibrated category centroids (unused in gate math,
        reserved for future directional checks).
    old_prior : np.ndarray
        Previous prior (A × d), same shape as new_prior.
    epsilon : float
        Maximum per-component divergence.

    Returns
    -------
    (passed, details) : (bool, dict)
        details keys: max_divergence, mean_divergence, affected_components,
        total_components, epsilon, recommendation.
    """
    assert new_prior.shape == old_prior.shape, (
        f"Prior shape mismatch: {new_prior.shape} vs {old_prior.shape}"
    )
    divergence = np.abs(new_prior - old_prior)
    max_div = float(np.max(divergence))
    mean_div = float(np.mean(divergence))
    affected = int(np.sum(divergence > epsilon))
    total = divergence.size
    passed = bool(max_div <= epsilon)

    return passed, {
        'max_divergence': round(max_div, 4),
        'mean_divergence': round(mean_div, 4),
        'affected_components': affected,
        'total_components': total,
        'epsilon': epsilon,
        'recommendation': 'safe_to_update' if passed else 'review_required',
    }
