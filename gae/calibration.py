"""
GAE Calibration — domain-configurable learning hyperparameters.

CalibrationProfile replaces hardcoded constants (ALPHA, LAMBDA_NEG,
EPSILON_DEFAULT) with a structured object that each domain provides.

Reference: docs/gae_design_v10_6.md §8; blog Eq. 4b, 4c.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, NamedTuple

import numpy as np


@dataclass
class CalibrationProfile:
    """Domain-configurable learning hyperparameters.

    Replaces hardcoded constants (ALPHA, LAMBDA_NEG, EPSILON_DEFAULT).
    Each domain provides its own profile via DomainConfig.

    Reference: docs/gae_design_v10_6.md §8; blog Eq. 4b, 4c.

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

        Reference: docs/gae_design_v10_6.md §8 (CalibrationProfile validation).

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

    Reference: docs/gae_design_v10_6.md §8 (SOC calibration).

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

    Reference: docs/gae_design_v10_6.md §8 (S2P calibration).

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
    Conservation law floor: minimum daily correct-correction
    signal required to deliver one effective learning half-life
    within T_max calendar days.

    θ_min = η × N_half² / T_max_days

    This is a convergence budget threshold, not a rate or flux.

    At η=0.05, N_half=14, T_max=21: θ_min ≈ 0.467 (SOC default).
    At η=0.05, N_half=14, T_max=28: θ_min ≈ 0.350 (S2P default —
    longer acceptable convergence due to 5:1 vs 20:1 penalty ratio).

    NOTE (April 16, 2026 — three-judge consensus): The previously
    documented formula θ_min = 23.53/(α×V) was structurally
    incorrect — it double-counted α and V which already appear on
    the LHS of the conservation inequality α·q·V ≥ θ_min. The
    correct form is this function's existing implementation: a
    constant derived from convergence requirements, not a
    deployment-scaled formula.

    V=50 deployments are feasible: at α=0.25, q=0.85,
    α·q·V = 10.6 >> 0.467. The "V=50 impossible" language in
    earlier documentation was incorrect.
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
    worst_case_quality: Optional[float] = None,
) -> float:
    """
    Principled η_override from override-signal SNR scaling.

    PRIMARY FORMULA (worst-case calibration — RECOMMENDED):
        η_override = η_confirm × (2 · q̄_worst − 1)

    where q̄_worst is the worst-case analyst quality (5th percentile
    or stress-test minimum). This bounds centroid degradation across
    the full quality range.

    For SOC (q̄_worst=0.60 from 24-persona V-Q5 validation):
        η_override = 0.05 × (2·0.60 − 1) = 0.05 × 0.20 = 0.01

    The ratio η_override/η_confirm = 2q̄−1 is the signal-to-noise
    scaling for the override learning channel. Derivation: Gemini
    (April 16, 2026 math poll), confirmed by Grok.

    DIAGNOSTIC FORMULA (mean + variance — for comparison only):
        η_override* ∝ (2q̄ − 1) / (2σ²_q + signal)

    Returns worst-case calibration if worst_case_quality provided,
    otherwise falls back to the diagnostic formula.

    Three-judge consensus April 16, 2026: worst-case calibration
    is the safety choice.

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
    worst_case_quality : float, optional
        Worst-case analyst quality (5th percentile or stress-test minimum).
        When provided, uses the primary formula and ignores other params.

    Returns
    -------
    float
        η_override value.
        - worst_case_quality path: η_confirm × max(0, 2q̄-1).
          Returns 0.0 at q̄ ≤ 0.5 (learning disabled).
        - diagnostic path (worst_case_quality=None):
          Floored at 0.005 when signal ≤ 0.

    Source: Q5 sweep (9 personas × 6 η values), roadmap session formula.
    Three-judge consensus April 16, 2026.
    """
    if worst_case_quality is not None:
        return float(eta_confirm * max(0.0, 2.0 * worst_case_quality - 1.0))
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


# ── Factor quarantine mask ────────────────────────────────────────────────────
# v6.0: binary mask (simple, auditable, CISO-explainable).
# v6.5: replaced by continuous W weighting (Adjustment A).
# Source: Three-judge consensus (Judge 1 proposed, Gemini + Opus approved).

_SOC_FACTOR_ORDER: List[str] = [
    'travel_match', 'asset_criticality', 'threat_intel_enrichment',
    'time_anomaly', 'pattern_history', 'device_trust',
]


def compute_factor_mask(
    sigma_per_factor: Dict[str, float],
    threshold: float = 0.20,
) -> Dict[str, bool]:
    """
    Binary mask: include clean factors, exclude noisy ones.

    v6.0: binary (simple, auditable, CISO-explainable).
    v6.5: replaced by continuous W weighting (Adjustment A).

    Factors with σ ≥ threshold are masked out (False); factors below threshold
    are included (True).

    Parameters
    ----------
    sigma_per_factor : dict
        Per-factor noise estimate from DeploymentQualifier.
        E.g., {'travel_match': 0.12, 'device_trust': 0.28, ...}
    threshold : float
        Factors with σ ≥ threshold are excluded. Default 0.20.

    Returns
    -------
    Dict[str, bool]
        True = include, False = exclude.

    Example
    -------
    >>> compute_factor_mask({'device_trust': 0.28, 'travel_match': 0.12}, 0.20)
    {'device_trust': False, 'travel_match': True}

    Source: Three-judge consensus (Judge 1 proposed, Gemini + Opus approved).
    """
    return {factor: sigma < threshold for factor, sigma in sigma_per_factor.items()}


_ETA_CONFIRM: float = 0.05  # standard confirm rate — P28 Phase 2 bootstrap


def compute_enriched_bootstrap_prior(
    historical_decisions: list,
    measured_sigma: Dict[str, float],
    domain_config,
    n_cat: int,
    n_act: int,
    n_factors: int,
    sigma_before: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Empirical Bayes bootstrap: re-run centroid calibration using
    σ-weighted factor vectors from historical SIEM decisions.

    Called ONCE at P28 Phase 2 completion, after per-factor σ is measured.
    Never called again. The result is stored via write_iks_bootstrap_anchor()
    which enforces write-once semantics.

    Blog Eq. 4b (enriched prior variant): μ₀ ← μ₀ + η·W·(f − μ₀).

    Args:
        historical_decisions: list of (category_idx, action_idx, factor_vector)
                              tuples from SIEM historical import.
        measured_sigma: dict mapping factor_name → σ_after value from
                       CovarianceEstimator.get_per_factor_sigma()
        domain_config: object with .factor_names list attribute.
        n_cat, n_act, n_factors: tensor dimensions.
        sigma_before: optional dict mapping factor_name → σ_before (pre-enrichment).
                      If provided, uses Δσ scheme: W_j = σ_before_j² / σ_after_j⁴.
                      Factors that improved most (large σ_before/σ_after ratio) get
                      highest weight. Fixed factors (σ_before == σ_after) reduce to
                      1/σ² — identical to the original scheme.
                      If None (default), uses original scheme: W = 1/σ_after².

    Returns:
        μ₀_enriched: np.ndarray shape (n_cat, n_act, n_factors)
                     Enriched bootstrap prior. 22% closer to operational
                     optimum than standard bootstrap (simulation).

    Mechanism:
        1. Build weight vector W per factor (shape (n_factors,)):
             sigma_before=None:  W_j = 1 / σ_after_j²          (absolute reliability)
             sigma_before given: W_j = σ_before_j² / σ_after_j⁴ (enrichment benefit)
        2. Normalize: W_normalized = W / W.mean() — preserves scale.
        3. For each historical decision (c, a, f):
           Compute kernel-weighted gradient: gradient = W_normalized * (f − μ[c,a,:]).
           Update prior: μ[c,a,:] += η * gradient.
           f is NEVER modified — W enters through the gradient, not by scaling f.
           Clip μ[c,a,:] to [0.0, 1.0] after each update (invariant).
        4. Return final μ₀_enriched.

    Note: This is NOT a call to ProfileScorer.update() — it is a
    standalone prior computation that does not touch any live scorer.
    The Loop 2 firewall is maintained: σ enters the prior computation
    at initialization time only, not during live learning.

    Reference: docs/gae_design_v10_6.md §9; T1 architecture — μ₀ enrichment.
    """
    assert n_factors > 0, f"n_factors must be positive, got {n_factors}"
    assert n_cat > 0, f"n_cat must be positive, got {n_cat}"
    assert n_act > 0, f"n_act must be positive, got {n_act}"

    factor_names: List[str] = domain_config.factor_names
    assert len(factor_names) == n_factors, (
        f"len(domain_config.factor_names)={len(factor_names)} != n_factors={n_factors}"
    )

    # Step 1: build W per factor, shape (n_factors,)
    sigma_after_arr = np.array([measured_sigma[name] for name in factor_names], dtype=float)
    assert sigma_after_arr.shape == (n_factors,), (
        f"sigma_after_arr.shape={sigma_after_arr.shape} != ({n_factors},)"
    )

    if sigma_before is not None:
        # Δσ scheme: weight by enrichment benefit
        # W_j = σ_before_j² / σ_after_j⁴
        # Fixed factors (σ_before == σ_after): W_j = σ² / σ⁴ = 1/σ² — same as original.
        sigma_before_arr = np.array([sigma_before[name] for name in factor_names], dtype=float)
        assert sigma_before_arr.shape == (n_factors,), (
            f"sigma_before_arr.shape={sigma_before_arr.shape} != ({n_factors},)"
        )
        W = sigma_before_arr ** 2 / sigma_after_arr ** 4
    else:
        # Original scheme: weight by absolute reliability
        W = 1.0 / sigma_after_arr ** 2

    assert W.shape == (n_factors,), f"W.shape={W.shape} != ({n_factors},)"

    # Step 2: normalize so mean weight = 1.0 (preserves scale)
    W_normalized = W / W.mean()
    assert W_normalized.shape == (n_factors,), (
        f"W_normalized.shape={W_normalized.shape} != ({n_factors},)"
    )

    # Initialize μ₀ to uniform 0.5
    mu = np.full((n_cat, n_act, n_factors), 0.5, dtype=float)
    assert mu.shape == (n_cat, n_act, n_factors), (
        f"mu.shape={mu.shape} != ({n_cat}, {n_act}, {n_factors})"
    )

    # Step 3: process each historical decision
    for decision in historical_decisions:
        c, a, f = decision
        f = np.asarray(f, dtype=float)
        assert f.shape == (n_factors,), (
            f"factor vector shape {f.shape} != ({n_factors},)"
        )
        assert 0 <= c < n_cat, f"category_idx={c} out of range [0, {n_cat})"
        assert 0 <= a < n_act, f"action_idx={a} out of range [0, {n_act})"

        gradient = f - mu[c, a, :]                         # standard residual, f unchanged
        assert gradient.shape == (n_factors,), (
            f"gradient.shape={gradient.shape} != ({n_factors},)"
        )

        mu[c, a, :] += _ETA_CONFIRM * W_normalized * gradient  # kernel-weighted correction
        mu[c, a, :] = np.clip(mu[c, a, :], 0.0, 1.0)

    return mu


def compute_dominant_axis(mu: np.ndarray) -> np.ndarray:
    """
    Per-factor centroid-separation score from variance across all (cat, act) pairs.

    For each factor dimension j, computes variance of μ[:,:,j] across all
    (category, action) positions. High variance means this factor drives centroid
    separation (discriminating). Low variance means this factor is not discriminating.

    Result is normalized to [0, 1] so scores are comparable across deployments.

    Parameters
    ----------
    mu : np.ndarray, shape (n_cat, n_act, n_factors)
        Current centroid tensor. In enriched bootstrap context: μ₀ from standard
        bootstrap (before enrichment pass).

    Returns
    -------
    np.ndarray, shape (n_factors,)
        Score in [0, 1] per factor. 1.0 = dominant separator, 0.0 = non-discriminating.
        All zeros when all factors have identical variance (uniform μ).

    Reference: V-BOOTSTRAP-GEOM design §4.5; B1 ceiling fix.
    """
    assert mu.ndim == 3, f"mu must be 3-D (n_cat, n_act, n_factors), got shape {mu.shape}"
    n_factors = mu.shape[-1]
    # Flatten (n_cat, n_act) → (n_positions,) for each factor
    flat = mu.reshape(-1, n_factors)          # shape (n_cat*n_act, n_factors)
    assert flat.shape[1] == n_factors, (
        f"flat.shape={flat.shape}, expected (:, {n_factors})"
    )
    per_factor_variance = np.var(flat, axis=0)  # shape (n_factors,)
    assert per_factor_variance.shape == (n_factors,), (
        f"per_factor_variance.shape={per_factor_variance.shape} != ({n_factors},)"
    )
    max_var = per_factor_variance.max()
    if max_var > 0:
        return per_factor_variance / max_var
    return np.zeros(n_factors, dtype=float)


def compute_enriched_bootstrap_prior_geom(
    historical_decisions: list,
    measured_sigma: Dict[str, float],
    sigma_before: Dict[str, float],
    mu_current: np.ndarray,
    domain_config,
    n_cat: int,
    n_act: int,
    n_factors: int,
) -> np.ndarray:
    """
    Geometry-aware Empirical Bayes bootstrap (V-BOOTSTRAP-GEOM).

    Combines enrichment benefit (Δσ) with centroid geometry awareness:
      W_boot_j = enrichment_ratio_j × (1 − dominant_axis_j)

    Where:
      enrichment_ratio_j = (σ_before_j / σ_after_j)²
        High when factor j benefited most from graph enrichment.
      dominant_axis_j = per-factor centroid-separation score from compute_dominant_axis()
        High when factor j drives the primary centroid separations.

    B1 deployment fix: in B1 geometries, enriched factors ARE the primary
    discriminators. Without geometry awareness the bootstrap gradient competes
    with the dominant structure → d≈0.21–0.23 ceiling. This scheme attenuates
    those factors' bootstrap weight, allowing other factors to guide the prior.

    B0 deployment: dominant_axis ≈ 0 → W_geom ≈ enrichment_ratio → reduces to
    Δσ scheme. When μ is uniform, compute_dominant_axis() returns zeros and
    this function is exactly equivalent to compute_enriched_bootstrap_prior()
    with sigma_before provided.

    W_geom clipped to ≥ 1e-6 to prevent zero or negative weights.

    Args:
        historical_decisions: list of (category_idx, action_idx, factor_vector).
        measured_sigma: dict factor_name → σ_after (post-enrichment).
        sigma_before: dict factor_name → σ_before (pre-enrichment). Required.
        mu_current: centroid tensor shape (n_cat, n_act, n_factors). Used to
                    compute dominant axis. Typically μ₀ from standard bootstrap.
        domain_config: object with .factor_names list attribute.
        n_cat, n_act, n_factors: tensor dimensions.

    Returns:
        μ₀_geom: np.ndarray shape (n_cat, n_act, n_factors)

    Reference: docs/gae_design_v10_6.md §9; V-BOOTSTRAP-GEOM §4.5.
    """
    assert n_factors > 0, f"n_factors must be positive, got {n_factors}"
    assert n_cat > 0, f"n_cat must be positive, got {n_cat}"
    assert n_act > 0, f"n_act must be positive, got {n_act}"
    assert mu_current.shape == (n_cat, n_act, n_factors), (
        f"mu_current.shape={mu_current.shape} != ({n_cat}, {n_act}, {n_factors})"
    )

    factor_names: List[str] = domain_config.factor_names
    assert len(factor_names) == n_factors, (
        f"len(domain_config.factor_names)={len(factor_names)} != n_factors={n_factors}"
    )

    # Step 1: enrichment benefit ratio per factor, shape (n_factors,)
    sa = np.array([measured_sigma[name] for name in factor_names], dtype=float)
    sb = np.array([sigma_before[name]   for name in factor_names], dtype=float)
    assert sa.shape == (n_factors,), f"sa.shape={sa.shape} != ({n_factors},)"
    assert sb.shape == (n_factors,), f"sb.shape={sb.shape} != ({n_factors},)"
    enrichment_ratio = (sb / sa) ** 2          # shape (n_factors,)

    # Step 2: dominant axis scores — how much each factor drives centroid separation
    dominant_axis = compute_dominant_axis(mu_current)   # shape (n_factors,)
    assert dominant_axis.shape == (n_factors,), (
        f"dominant_axis.shape={dominant_axis.shape} != ({n_factors},)"
    )

    # Step 3: geometry-aware weights — attenuate factors aligned with dominant structure
    W_geom = enrichment_ratio * (1.0 - dominant_axis)
    W_geom = np.clip(W_geom, 1e-6, None)               # never zero or negative
    assert W_geom.shape == (n_factors,), f"W_geom.shape={W_geom.shape} != ({n_factors},)"

    W_normalized = W_geom / W_geom.mean()
    assert W_normalized.shape == (n_factors,), (
        f"W_normalized.shape={W_normalized.shape} != ({n_factors},)"
    )

    # Step 4: bootstrap update — identical loop to existing scheme
    mu = np.full((n_cat, n_act, n_factors), 0.5, dtype=float)

    for decision in historical_decisions:
        c, a, f = decision
        f = np.asarray(f, dtype=float)
        assert f.shape == (n_factors,), (
            f"factor vector shape {f.shape} != ({n_factors},)"
        )
        assert 0 <= c < n_cat, f"category_idx={c} out of range [0, {n_cat})"
        assert 0 <= a < n_act, f"action_idx={a} out of range [0, {n_act})"

        gradient = f - mu[c, a, :]                               # f unchanged
        assert gradient.shape == (n_factors,), (
            f"gradient.shape={gradient.shape} != ({n_factors},)"
        )
        mu[c, a, :] += _ETA_CONFIRM * W_normalized * gradient   # kernel-weighted
        mu[c, a, :] = np.clip(mu[c, a, :], 0.0, 1.0)

    return mu


def mask_to_array(
    mask: Dict[str, bool],
    factor_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Convert factor mask dict to numpy array (1.0 = include, 0.0 = exclude).

    Parameters
    ----------
    mask : dict
        From compute_factor_mask().
    factor_names : list of str, optional
        Ordered factor names. If None, uses SOC default order:
        [travel_match, asset_criticality, threat_intel_enrichment,
         time_anomaly, pattern_history, device_trust]

    Returns
    -------
    np.ndarray, shape (d,)
        1.0 for included factors, 0.0 for excluded factors.
    """
    if factor_names is None:
        factor_names = _SOC_FACTOR_ORDER
    arr = np.array([1.0 if mask.get(f, True) else 0.0 for f in factor_names])
    assert arr.shape == (len(factor_names),), (
        f"mask array shape {arr.shape} != ({len(factor_names)},)"
    )
    return arr
