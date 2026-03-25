"""
GAE Convergence — monitoring weight learning stability and accuracy.

Provides a standalone function that inspects a LearningState and returns
diagnostic metrics used to determine whether training has converged.

Convergence criterion (docs/gae_design_v5.md §8.3):
    converged := stability < STABILITY_THRESHOLD AND accuracy > ACCURACY_THRESHOLD

Where:
    stability := std(||W_after||₂ over the last 10 history entries)
    accuracy  := fraction of recent outcomes == +1

Three failure modes detected via the returned metrics:
    FM1 — Action confusion  : weight_norm is low → actions score similarly
    FM2 — Asymmetric oscillation : alternating correct/incorrect → accuracy ≈ 0.5
    FM3 — Decay competition  : stability is high → W norm never stabilises

Reference: docs/gae_design_v5.md §8.3; blog convergence criterion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple, Optional

import numpy as np

if TYPE_CHECKING:
    from gae.learning import LearningState


# ── Empirical constants ──────────────────────────────────────────────────────
# Source: math_synopsis_v9 §5-6, Bridge B Phase B/C, P1 (Σ_f), P6 (σ_max)

ETA_DEFAULT: float = 0.05      # Canonical learning rate (P1 confirmed)
ETA_NEG_DEFAULT: float = 0.05  # Canonical negative learning rate (P1 confirmed)
TR_SIGMA_F: float = 0.34       # tr(Σ_f) measured FX-1-PROXY-REAL (P1); design=0.24 (+44%)
SIGMA_MAX: float = 0.034       # 10th-percentile L2 margin (P6)
N_HALF_DEFAULT: float = 13.51  # ln(2)/ln(1/(1−0.05)) decisions (math_synopsis_v9 §5)
D: int = 6                     # Factor dimensionality

# ε=0.10 gives 2.6× headroom over e_∞≈0.038; ε=0.05 gives only 1.3× (too tight).
# PROD-5: all 6 categories failed the ε=0.05 test despite healthy accuracy gains.
EPSILON_DEFAULT: float = 0.10  # Noise-aware convergence threshold (PROD-5 validated)


# ── Prediction helpers ───────────────────────────────────────────────────────

def compute_n_half(eta: float = ETA_DEFAULT) -> float:
    """
    Half-life of centroid error in decisions.

    N_half = ln(2) / ln(1/(1−η))

    At η=0.05: N_half = 13.51 decisions per (category, action) pair.
    Source: math_synopsis_v9 §5.

    Parameters
    ----------
    eta : float
        Learning rate.

    Returns
    -------
    float
        N_half in decisions.
    """
    return float(np.log(2) / np.log(1.0 / (1.0 - eta)))


def compute_steady_state_mse(
    eta: float = ETA_DEFAULT,
    tr_sigma_f: float = TR_SIGMA_F,
) -> float:
    """
    Steady-state MSE of centroid tracking.

    MSE_∞ = η/(2−η) × tr(Σ_f)

    At η=0.05, tr(Σ_f)=0.34: MSE_∞ ≈ 0.0087.
    Source: math_synopsis_v9 §5; tr(Σ_f) from P1.

    Parameters
    ----------
    eta : float
        Learning rate.
    tr_sigma_f : float
        Trace of factor covariance matrix (measured: 0.34, P1).

    Returns
    -------
    float
        Steady-state MSE.
    """
    assert eta > 0 and eta < 2, f"eta must be in (0,2), got {eta}"
    return float(eta / (2.0 - eta) * tr_sigma_f)


def compute_e_inf_per_component(
    eta: float = ETA_DEFAULT,
    tr_sigma_f: float = TR_SIGMA_F,
    d: int = D,
) -> float:
    """
    Per-component steady-state error.

    e_∞ = sqrt(MSE_∞ / d)

    At defaults (η=0.05, tr(Σ_f)=0.34, d=6): e_∞ ≈ 0.038 per factor component.
    Source: math_synopsis_v9 §5; P1 for tr(Σ_f).

    Parameters
    ----------
    eta : float
        Learning rate.
    tr_sigma_f : float
        Trace of factor covariance matrix.
    d : int
        Factor dimensionality.

    Returns
    -------
    float
        Per-component steady-state error.
    """
    assert d > 0, f"d must be positive, got {d}"
    mse = compute_steady_state_mse(eta, tr_sigma_f)
    return float(np.sqrt(mse / d))


def predict_convergence_decisions(
    e_0: float,
    epsilon: float = 0.10,
    eta: float = ETA_DEFAULT,
    tr_sigma_f: float = TR_SIGMA_F,
) -> int:
    """
    Predicted decisions to converge from initial error e_0 to threshold ε.

    N = ceil(ln(e_0/ε) / ln(1/(1−η)))

    Only valid when ε > e_∞ (steady-state error).
    Returns -1 if ε ≤ e_∞ (cannot converge below steady-state).
    Returns 0 if e_0 ≤ ε (already converged).
    Source: math_synopsis_v9 §5.

    Parameters
    ----------
    e_0 : float
        Initial per-component error (e.g., 0.15).
    epsilon : float
        Convergence threshold (e.g., 0.05).
    eta : float
        Learning rate.
    tr_sigma_f : float
        Trace of factor covariance matrix.

    Returns
    -------
    int
        Decisions required, or -1 if convergence is impossible.
    """
    e_inf = compute_e_inf_per_component(eta, tr_sigma_f)
    if epsilon <= e_inf:
        return -1
    if e_0 <= epsilon:
        return 0
    n = np.log(e_0 / epsilon) / np.log(1.0 / (1.0 - eta))
    return int(np.ceil(n))


def predict_convergence_decisions_v2(
    e_0: float,
    epsilon: float = EPSILON_DEFAULT,
    eta: float = ETA_DEFAULT,
    tr_sigma_f: float = TR_SIGMA_F,
    safety_factor: float = 2.0,
) -> int:
    """
    Noise-aware convergence prediction.

    N = ceil(ln(e_0/ε) / ln(1/(1−η)) × safety_factor)

    Accounts for the steady-state noise floor: centroids can never converge
    below e_∞ = sqrt(η/(2−η) × tr(Σ_f) / d). When ε ≤ e_∞ × 1.5 (too close
    to the noise floor), ε is auto-adjusted to e_∞ × 2.5 to avoid
    non-convergence in practice.

    The safety_factor multiplies the raw geometric prediction to account for
    noise-induced oscillations near the threshold.

    PROD-5 validated: safety_factor=2.0 brings predictions within 1.5× of
    simulated convergence at ε=0.10.
    Source: math_synopsis_v9 §5; PROD-5 validation.

    Parameters
    ----------
    e_0 : float
        Initial per-component error (e.g., 0.15).
    epsilon : float
        Convergence threshold. Defaults to EPSILON_DEFAULT=0.10.
    eta : float
        Learning rate.
    tr_sigma_f : float
        Trace of factor covariance matrix.
    safety_factor : float
        Multiplier on raw N to account for noise-floor oscillations.
        PROD-5 validated at 2.0.

    Returns
    -------
    int
        Decisions required (0 if already converged). Never returns -1;
        epsilon is auto-adjusted if too close to noise floor.
    """
    e_inf = compute_e_inf_per_component(eta, tr_sigma_f)
    if epsilon <= e_inf * 1.5:
        # Too close to noise floor — auto-adjust to safe threshold
        epsilon = e_inf * 2.5
    if e_0 <= epsilon:
        return 0
    n = np.log(e_0 / epsilon) / np.log(1.0 / (1.0 - eta))
    return int(np.ceil(n * safety_factor))


def enrichment_multiplier(
    graph_level: str,
    rho: float = 0.8,
) -> float:
    """
    Convergence acceleration factor by graph maturity level.

    Empirical values from Bridge B Phase B (validated at A=4, ρ=0.8).
    Multiplier < 1.0 means faster convergence than G1 baseline.
    E.g., 0.737 means G₄ converges in 73.7% of G₁ decisions.
    Source: Bridge B Phase B results (G₂: −15.1%, G₄: −26.3% at ε=0.05).

    Parameters
    ----------
    graph_level : str
        One of 'G1', 'G2', 'G3', 'G4'.
    rho : float
        Cross-source correlation (default 0.8, validated).

    Returns
    -------
    float
        Multiplier on N_converge. Unknown levels return 1.0.
    """
    multipliers = {
        'G1': 1.000,
        'G2': 0.849,   # −15.1% (Bridge B Phase B, ρ=0.8 validated)
        'G3': 0.822,   # −17.8%
        'G4': 0.737,   # −26.3%
    }
    return multipliers.get(graph_level, 1.0)


def reconvergence_acceleration(episode: int) -> float:
    """
    Re-convergence acceleration from Bridge B Phase C v3.

    Each distribution shift re-converges faster as graph enriches.
    Empirical decay: N_converge ratio ≈ 0.703 per episode (p<0.0001).
    Episode 0 = initial convergence (multiplier 1.0).
    Episode 1 = first shift (≈546 decisions vs 1404 initial).
    Source: Bridge B Phase C v3.

    Parameters
    ----------
    episode : int
        0 = initial, 1 = first shift, 2 = second shift, ...

    Returns
    -------
    float
        Multiplier on N_converge relative to initial.
    """
    assert episode >= 0, f"episode must be non-negative, got {episode}"
    decay_per_episode = 0.703
    return float(decay_per_episode ** episode)


def predict_category_convergence_weeks(
    category: str,
    alerts_per_day: float = 200,
    verification_rate: float = 0.30,
    n_actions: int = 4,
    e_0: float = 0.15,
    graph_level: str = 'G1',
    eta: float = ETA_DEFAULT,
    tr_sigma_f: float = TR_SIGMA_F,
) -> Dict:
    """
    Predict weeks to convergence for a specific category.

    Converts raw N_converge decisions into calendar weeks given
    alert volume and analyst verification rate.
    Source: math_synopsis_v9 §6; Bridge B Phase B enrichment multipliers.

    Parameters
    ----------
    category : str
        Category name (for display only).
    alerts_per_day : float
        Total alert volume for this category.
    verification_rate : float
        Fraction verified by analysts.
    n_actions : int
        Number of actions (4 for A=4 design).
    e_0 : float
        Initial per-component error.
    graph_level : str
        Graph maturity level ('G1'–'G4').
    eta : float
        Learning rate.
    tr_sigma_f : float
        Trace of factor covariance matrix.

    Returns
    -------
    Dict
        weeks, decisions_needed, days, status, and supporting fields.
    """
    verified_per_day = alerts_per_day * verification_rate
    verified_per_action_per_day = verified_per_day / max(n_actions, 1)

    n_decisions = predict_convergence_decisions_v2(e_0, EPSILON_DEFAULT, eta, tr_sigma_f)
    if n_decisions == 0:
        return {
            'category': category,
            'weeks': 0,
            'decisions_needed': 0,
            'status': 'already_converged',
        }

    multiplier = enrichment_multiplier(graph_level)
    n_adjusted = int(np.ceil(n_decisions * multiplier))

    days = n_adjusted / max(verified_per_action_per_day, 0.01)
    weeks = days / 7.0

    return {
        'category': category,
        'weeks': round(weeks, 1),
        'decisions_needed': n_adjusted,
        'days': round(days, 1),
        'verified_per_action_day': round(verified_per_action_per_day, 2),
        'graph_level': graph_level,
        'enrichment_multiplier': multiplier,
        'status': 'will_converge',
    }


def generate_onboarding_calendar(
    categories: list,
    category_weights: dict,
    alerts_per_day: float = 200,
    verification_rate: float = 0.30,
    graph_level: str = 'G1',
    eta: float = ETA_DEFAULT,
    tr_sigma_f: float = TR_SIGMA_F,
) -> Dict:
    """
    Generate full onboarding calendar for all categories.

    Returns per-category convergence predictions sorted by weeks,
    plus an overall summary. Weights control the fraction of total
    alert volume assigned to each category.
    Source: math_synopsis_v9 §6; Bridge B Phase B enrichment multipliers.

    Parameters
    ----------
    categories : list
        Ordered list of category names.
    category_weights : dict
        Fraction of total alerts per category. Missing keys default
        to 1/n_categories.
    alerts_per_day : float
        Total alert volume across all categories.
    verification_rate : float
        Fraction verified by analysts.
    graph_level : str
        Graph maturity level ('G1'–'G4').
    eta : float
        Learning rate.
    tr_sigma_f : float
        Trace of factor covariance matrix.

    Returns
    -------
    Dict
        predictions (list), first_calibrated, last_calibrated,
        total_weeks, graph_level, assumptions.
    """
    n_categories = len(categories)
    predictions = []
    for cat in categories:
        weight = category_weights.get(cat, 1.0 / max(n_categories, 1))
        cat_alerts_per_day = alerts_per_day * weight
        pred = predict_category_convergence_weeks(
            category=cat,
            alerts_per_day=cat_alerts_per_day,
            verification_rate=verification_rate,
            graph_level=graph_level,
            eta=eta,
            tr_sigma_f=tr_sigma_f,
        )
        predictions.append(pred)

    valid = [p for p in predictions if p['weeks'] > 0]
    valid.sort(key=lambda p: p['weeks'])

    return {
        'predictions': predictions,
        'first_calibrated': valid[0] if valid else None,
        'last_calibrated': valid[-1] if valid else None,
        'total_weeks': max(p['weeks'] for p in valid) if valid else -1,
        'graph_level': graph_level,
        'assumptions': {
            'alerts_per_day': alerts_per_day,
            'verification_rate': verification_rate,
            'eta': eta,
            'tr_sigma_f': tr_sigma_f,
            'enrichment': graph_level,
        },
    }


# ---------------------------------------------------------------------------
# Convergence thresholds
# ---------------------------------------------------------------------------

STABILITY_THRESHOLD: float = 0.05   # std of recent W norms below this → stable
ACCURACY_THRESHOLD: float = 0.80    # recent accuracy above this → good
RECENCY_WINDOW: int = 20            # decisions used for accuracy estimate
STABILITY_WINDOW: int = 10          # decisions used for stability estimate


# ---------------------------------------------------------------------------
# get_convergence_metrics
# ---------------------------------------------------------------------------

def get_convergence_metrics(state: "LearningState") -> dict:
    """
    Compute convergence and health diagnostics for a LearningState.

    Stability is measured as the standard deviation of the Frobenius norm
    of W_after over the last STABILITY_WINDOW (10) history entries.
    A small std means the weight matrix has stopped changing significantly.

    Accuracy is the fraction of outcome == +1 among the last RECENCY_WINDOW
    (20) history entries.

    Reference: docs/gae_design_v5.md §8.3; blog convergence criterion.

    Parameters
    ----------
    state : LearningState
        The active Tier 3 learning state.

    Returns
    -------
    dict with keys:
        decisions           int   — total completed updates
        weight_norm         float — Frobenius norm of current W
        stability           float — std of W norms over last 10 updates
                                    (0.0 when fewer than 2 history entries)
        accuracy            float — fraction of correct in last 20 updates
                                    (0.0 when history is empty)
        converged           bool  — stability < threshold AND accuracy > threshold
        provisional_dimensions int — count of A4 provisional W columns
        pending_autonomous  int   — count of C3 deferred validations
    """
    weight_norm = float(np.linalg.norm(state.W))

    if not state.history:
        return {
            "decisions": state.decision_count,
            "weight_norm": weight_norm,
            "stability": 0.0,
            "accuracy": 0.0,
            "converged": False,
            "provisional_dimensions": sum(
                1 for dm in state.dimension_metadata if dm.state == "provisional"
            ),
            "pending_autonomous": len(state.pending_validations),
        }

    # Stability: std of ||W_after||_F over last STABILITY_WINDOW entries
    recent_history = state.history[-STABILITY_WINDOW:]
    norms = [float(np.linalg.norm(h.W_after)) for h in recent_history]
    stability = float(np.std(norms)) if len(norms) >= 2 else 0.0

    # Accuracy: fraction correct over last RECENCY_WINDOW entries
    outcome_window = state.history[-RECENCY_WINDOW:]
    accuracy = (
        sum(1 for h in outcome_window if h.outcome == +1) / len(outcome_window)
    )

    converged = stability < STABILITY_THRESHOLD and accuracy > ACCURACY_THRESHOLD

    return {
        "decisions": state.decision_count,
        "weight_norm": weight_norm,
        "stability": stability,
        "accuracy": accuracy,
        "converged": converged,
        "provisional_dimensions": sum(
            1 for dm in state.dimension_metadata if dm.state == "provisional"
        ),
        "pending_autonomous": len(state.pending_validations),
    }


# ---------------------------------------------------------------------------
# Fix A — Baseline-normalized Var(q)
# ---------------------------------------------------------------------------

def compute_normalized_var_q(
    q_rolling: list,
    q_baseline: float,
) -> float:
    """
    Baseline-normalized quality variance (Fix A).

    Subtracts the expected Bernoulli variance for a healthy signal at
    q_baseline, yielding ~0 under healthy conditions and positive when
    degradation begins.

    var_norm = max(0, Var(q_rolling) − q_baseline·(1−q_baseline))

    P1 reference: supplementary quality-monitoring signal.
    Shape: scalar (float).

    Parameters
    ----------
    q_rolling : list[float]
        Recent per-decision quality scores (rolling window).
    q_baseline : float
        Mean quality over the first 50 decisions (calibration period).
        Fixed at calibration; never updated afterward.

    Returns
    -------
    float
        Baseline-normalized variance. 0.0 when len(q_rolling) < 2.
    """
    if len(q_rolling) < 2:
        return 0.0
    var_raw = float(np.var(q_rolling))
    var_baseline = q_baseline * (1.0 - q_baseline)
    return max(0.0, var_raw - var_baseline)


# ---------------------------------------------------------------------------
# Fix B — Layer 2 trend detection (YELLOW early warning)
# ---------------------------------------------------------------------------

def check_gradual_degradation(
    q_history: list,
    q_baseline: float,
    slope_threshold: float = -0.003,
    var_threshold: float = 0.05,
    window: int = 25,
) -> tuple:
    """
    Layer 2 early warning for gradual quality degradation (Fix B).

    Operates independently of Layer 1 (α·q·V < θ_min → AMBER/RED).
    Does NOT pause learning — YELLOW warning only.

    Fires only when BOTH conditions hold simultaneously (AND logic):
    Trigger 2a: slope of q̄_rolling_25 < slope_threshold per decision.
    Trigger 2b: compute_normalized_var_q(recent, q_baseline) > var_threshold.

    P1 reference: supplementary Layer 2 quality trend signal.
    Shape: window-length slice of q_history fed to polyfit(degree=1).

    Parameters
    ----------
    q_history : list[float]
        Full per-decision quality history.
    q_baseline : float
        Mean quality over the first 50 decisions (fixed at calibration).
    slope_threshold : float
        Fires if OLS slope of last `window` q values < this (default -0.003).
    var_threshold : float
        Fires if normalized variance > this (default 0.05).
    window : int
        Number of recent decisions to examine (default 25).

    Returns
    -------
    tuple[bool, str]
        (fires, reason) — fires=True when BOTH triggers activate simultaneously;
        reason is empty string when fires=False.
    """
    if len(q_history) < window:
        return False, ""
    recent = q_history[-window:]
    assert len(recent) == window, f"recent length {len(recent)} != window {window}"
    x = np.arange(window, dtype=np.float64)
    slope = float(np.polyfit(x, recent, 1)[0])
    var_norm = compute_normalized_var_q(recent, q_baseline)
    if slope < slope_threshold and var_norm > var_threshold:
        return True, f"slope={slope:.4f} and var_norm={var_norm:.4f}"
    return False, ""


# ---------------------------------------------------------------------------
# ConservationMonitor — Layer 1 (AMBER/RED) + Layer 2 (YELLOW)
# ---------------------------------------------------------------------------

#: Default threshold for Layer 2 normalized variance trigger (Fix A).
VAR_Q_THRESHOLD: float = 0.05

#: Number of decisions in the calibration period before q_baseline is set.
CALIBRATION_PERIOD: int = 50


class ConservationMonitor:
    """
    Two-layer quality conservation monitor.

    Layer 1 (AMBER/RED): caller-driven via update_conservation_signal().
    Fires when α·q·V < θ_min and pauses learning via set_conservation_status()
    on the attached ProfileScorer. Layer 1 behavior is UNCHANGED by Fix B.

    Layer 2 (YELLOW): fires check_gradual_degradation() every update after
    the 50-decision calibration period. YELLOW warning is stored in
    self.yellow_warning but does NOT pause learning.

    q_baseline is computed as the mean of the first CALIBRATION_PERIOD (50)
    quality scores and is never updated afterward.

    P1 reference: conservation monitoring supplementary design.

    Parameters
    ----------
    scorer : optional
        ProfileScorer instance. When provided, Layer 1 calls
        scorer.set_conservation_status(). May be None for standalone use.
    slope_threshold : float
        Layer 2 slope trigger (default -0.003/decision).
    var_threshold : float
        Layer 2 normalized-variance trigger (default 0.05).
    window : int
        Layer 2 rolling window size (default 25).
    """

    def __init__(
        self,
        scorer=None,
        slope_threshold: float = -0.003,
        var_threshold: float = VAR_Q_THRESHOLD,
        window: int = 25,
    ) -> None:
        self._scorer = scorer
        self._slope_threshold = slope_threshold
        self._var_threshold = var_threshold
        self._window = window

        # Layer 2 state
        self._q_history: list = []
        self._q_baseline: float = 0.0
        self._baseline_set: bool = False
        self.yellow_warning: bool = False
        self.yellow_reason: str = ""

        # Layer 1 state (caller-driven)
        self._conservation_status: str = "GREEN"

    # ------------------------------------------------------------------
    # Layer 1: existing conservation signal (AMBER/RED)
    # ------------------------------------------------------------------

    def update_conservation_signal(self, status: str) -> None:
        """
        Update Layer 1 conservation status (AMBER/RED/GREEN).

        Called by external logic when α·q·V crosses θ_min.
        Propagates to the attached scorer if present.
        Layer 1 behavior is UNCHANGED by Fix B.

        Parameters
        ----------
        status : str
            One of 'GREEN', 'AMBER', 'RED'.
        """
        self._conservation_status = status
        if self._scorer is not None:
            self._scorer.set_conservation_status(status)

    @property
    def conservation_status(self) -> str:
        """Current Layer 1 status string ('GREEN', 'AMBER', 'RED')."""
        return self._conservation_status

    # ------------------------------------------------------------------
    # Layer 2: gradual degradation (YELLOW early warning)
    # ------------------------------------------------------------------

    def record_quality(self, q: float) -> None:
        """
        Record a per-decision quality score and run Layer 2 checks.

        Sets q_baseline after the first CALIBRATION_PERIOD decisions
        (never updated afterward). Runs check_gradual_degradation()
        on every call once the baseline is established.
        Updates self.yellow_warning and self.yellow_reason.
        Does NOT affect learning or Layer 1 status.

        Parameters
        ----------
        q : float
            Quality score for this decision (e.g. 0.0–1.0).
        """
        self._q_history.append(float(q))

        if not self._baseline_set and len(self._q_history) >= CALIBRATION_PERIOD:
            self._q_baseline = float(
                np.mean(self._q_history[:CALIBRATION_PERIOD])
            )
            self._baseline_set = True

        if self._baseline_set:
            fires, reason = check_gradual_degradation(
                self._q_history,
                self._q_baseline,
                slope_threshold=self._slope_threshold,
                var_threshold=self._var_threshold,
                window=self._window,
            )
            self.yellow_warning = fires
            self.yellow_reason = reason

    @property
    def q_baseline(self) -> float:
        """Calibration-period mean quality (0.0 until baseline is set)."""
        return self._q_baseline

    @property
    def baseline_set(self) -> bool:
        """True after CALIBRATION_PERIOD decisions have been recorded."""
        return self._baseline_set
