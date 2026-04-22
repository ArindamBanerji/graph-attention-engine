"""
GAE Learning — Hebbian weight learning.

Implements Eq. 4b and Eq. 4c from the math blog:

    Eq. 4b: W[a, :] += α_eff · r(t) · f(t) · δ(t)
    Eq. 4c: W        ← (1 − ε_vector) · W

where:
    a       : action index taken at decision time
    α_eff   : effective learning rate (base ALPHA, discounted via A1)
    r(t)    : outcome ∈ {+1, −1}
    f(t)    : factor vector from the ORIGINAL decision (Requirement R4)
    δ(t)    : asymmetry multiplier — 1.0 if correct, LAMBDA_NEG if incorrect
    ε_vector: per-factor decay rates, shape (n_f,)

After each update W is clamped to [−W_CLAMP, +W_CLAMP].

Constants (math blog §4):
    ALPHA      = 0.02    base learning rate
    LAMBDA_NEG = 20.0    asymmetric penalty (20 : 1 risk preference)
    EPSILON    = 0.001   default per-factor decay (half-life ≈ 693 decisions)
    W_CLAMP    = 5.0     weight saturation limit

Hardening:
    A1: Confidence-discounted α — reduces learning rate when the system was
        already confident at decision time (confirmation bias mitigation).
    A2: Per-factor ε_vector — permanent factors decay slowly, campaign factors
        decay fast (see docs/gae_design_v10_6.md §8.4).
    A4: Provisional dimension tracking — newly discovered dimensions enter with
        10× accelerated decay; auto-pruned if they fail to earn reinforcement.
    C3: Deferred validation for autonomous decisions — they enter
        pending_validations instead of applying learning immediately.

Default values are set for NO-OP behaviour:
    discount_strength = 0.0  → A1 inactive
    epsilon_vector    = None → built from profile via build_epsilon_vector()
    dimension_metadata = []  → no provisional dimensions tracked
    pending_validations = [] → no deferred validations

Reference: docs/gae_design_v10_6.md §8, §8.3; blog Eq. 4b, 4c.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from gae.calibration import CalibrationProfile

from gae.profile_scorer import CentroidUpdate

if TYPE_CHECKING:
    from gae.profile_scorer import ProfileScorer


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

ALPHA: float = 0.02       # base learning rate (blog §4)
LAMBDA_NEG: float = 20.0  # asymmetric penalty multiplier (blog §4)
EPSILON: float = 0.001    # default per-factor decay rate (blog §4)
W_CLAMP: float = 5.0      # weight saturation limit


# ---------------------------------------------------------------------------
# DimensionMetadata  (A4 hardening)
# ---------------------------------------------------------------------------

@dataclass
class DimensionMetadata:
    """
    Tracks the lifecycle of one scoring dimension in the weight matrix W.

    Used to implement A4 hardening: provisional dimensions (discovered
    cross-graph patterns) enter with accelerated decay and are pruned if
    they fail to earn reinforcement within the establishment window.

    Reference: docs/gae_design_v10_6.md §8.3 (A4 hardening).

    Attributes
    ----------
    factor_name : str
        Name of the scoring dimension this entry tracks.
    col_index : int
        Column index in W that this dimension occupies.
        Stored explicitly to avoid off-by-one errors during pruning.
    created_at : int
        decision_count at the time this dimension was added.
    state : str
        "original" | "provisional" | "established"
    decay_rate : float
        Active decay rate for this dimension's ε.
    reinforcement_count : int
        Number of updates where |delta_applied[col_index]| > 0.
    establishment_threshold : int
        Reinforcements required to transition provisional → established.
    """

    factor_name: str
    col_index: int
    created_at: int
    state: str = "provisional"
    decay_rate: float = 0.01       # provisional: 10× faster than standard
    reinforcement_count: int = 0
    establishment_threshold: int = 50


# ---------------------------------------------------------------------------
# PendingValidation  (C3 hardening)
# ---------------------------------------------------------------------------

@dataclass
class PendingValidation:
    """
    An autonomous decision awaiting outcome validation.

    Autonomous decisions do not apply learning immediately (C3 hardening).
    Instead they are held here until the validation window expires, then
    processed by process_pending_validations().

    Reference: docs/gae_design_v10_6.md §8.2 (C3 hardening).

    Attributes
    ----------
    entity_id : str
        Identifier for the entity that triggered this decision.
    action : str
        Action name taken.
    action_index : int
        Row index in W for *action*.
    factor_vector : np.ndarray, shape (1, n_f)
        f(t) from the ORIGINAL decision — preserved per R4.
    auto_decided_at : float
        Unix timestamp when the autonomous decision was made.
    validation_window_days : int
        Days to wait before applying learning (default 14).
    """

    entity_id: str
    action: str
    action_index: int
    factor_vector: np.ndarray
    auto_decided_at: float
    validation_window_days: int = 14


# ---------------------------------------------------------------------------
# WeightUpdate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WeightUpdate:
    """
    Immutable record of one Eq. 4b weight update.

    Carries full provenance: who decided, what outcome, how W changed.

    Reference: docs/gae_design_v10_6.md §8.2; blog Eq. 4b.

    Attributes
    ----------
    decision_number : int
        Monotonically increasing decision counter at the time of this update.
    timestamp : float
        Unix timestamp of the update.
    action_index : int
        Row of W that was updated.
    action_name : str
        Human-readable action label.
    outcome : int
        r(t) ∈ {+1, −1}.
    factor_vector : np.ndarray, shape (1, n_f)
        f(t) from the original decision — preserved per R4.
    delta_applied : np.ndarray, shape (n_f,)
        The update vector added to W[action_index, :] by Eq. 4b.
    W_before : np.ndarray, shape (n_a, n_f)
        Snapshot of W immediately before this update.
    W_after : np.ndarray, shape (n_a, n_f)
        Snapshot of W after Eq. 4b + Eq. 4c + clamp.
    alpha_effective : float
        α_eff used — may differ from ALPHA if A1 discounting was active.
    confidence_at_decision : float
        System confidence when the decision was made (used by A1).
    """

    decision_number: int
    timestamp: float
    action_index: int
    action_name: str
    outcome: int
    factor_vector: np.ndarray
    delta_applied: np.ndarray
    W_before: np.ndarray
    W_after: np.ndarray
    alpha_effective: float
    confidence_at_decision: float
    centroid_update: Optional[CentroidUpdate] = None


# ---------------------------------------------------------------------------
# LearningState
# ---------------------------------------------------------------------------

@dataclass
class LearningState:
    """
    Persistent state for the weight matrix and its learning history.

    Reference: docs/gae_design_v10_6.md §8.2; blog Eq. 4b, 4c.

    Attributes
    ----------
    W : np.ndarray, shape (n_a, n_f)
        Current weight matrix. Rows = actions, columns = factors.
    n_actions : int
        Number of actions (rows of W). Fixed after construction.
    n_factors : int
        Number of factors (columns of W). Grows via expand_weight_matrix().
    factor_names : list[str]
        Factor names in column order. len == n_factors at all times.
    profile : CalibrationProfile
        Domain-configurable hyperparameters (learning rate, penalty ratio,
        temperature, per-factor decay classes). Required.
    decision_count : int
        Total completed (non-deferred) update steps.
    history : list[WeightUpdate]
        Ordered list of all completed weight updates.
    expansion_history : list[dict]
        Ordered log of weight matrix expansions.
    discount_strength : float
        A1 hardening: confidence discount ∈ [0, 1].
        0.0 = inactive (NO-OP default); 0.5 = design spec value.
    epsilon_vector : np.ndarray or None, shape (n_f,)
        A2 hardening: per-factor decay rates.
        None → built from profile via build_epsilon_vector() in __post_init__.
    dimension_metadata : list[DimensionMetadata]
        A4 hardening: lifecycle records for all non-original dimensions.
        [] = no provisional dimensions tracked (NO-OP default).
    pending_validations : list[PendingValidation]
        C3 hardening: autonomous decisions awaiting validation.
        [] = no deferred validations (NO-OP default).
    """

    W: np.ndarray
    n_actions: int
    n_factors: int
    factor_names: list[str]
    profile: CalibrationProfile
    decision_count: int = 0
    history: list[WeightUpdate] = field(default_factory=list)
    expansion_history: list[dict[str, Any]] = field(default_factory=list)
    discount_strength: float = 0.0
    epsilon_vector: np.ndarray | None = None
    dimension_metadata: list[DimensionMetadata] = field(default_factory=list)
    pending_validations: list[PendingValidation] = field(default_factory=list)
    profile_scorer: Optional["ProfileScorer"] = None

    def __post_init__(self) -> None:
        assert isinstance(self.W, np.ndarray), (
            "LearningState.W must be np.ndarray"
        )
        assert self.W.ndim == 2, (
            f"LearningState.W must be 2-D, got shape {self.W.shape}"
        )
        assert self.W.shape == (self.n_actions, self.n_factors), (
            f"W.shape {self.W.shape} != (n_actions={self.n_actions}, "
            f"n_factors={self.n_factors})"
        )
        assert len(self.factor_names) == self.n_factors, (
            f"len(factor_names) ({len(self.factor_names)}) "
            f"!= n_factors ({self.n_factors})"
        )
        if self.epsilon_vector is None:
            self.epsilon_vector = self.build_epsilon_vector()
        assert self.epsilon_vector.shape == (self.n_factors,), (
            f"epsilon_vector.shape {self.epsilon_vector.shape} "
            f"!= (n_factors={self.n_factors},)"
        )

    # ------------------------------------------------------------------
    # build_epsilon_vector  — A2 per-factor decay (Eq. 4c)
    # ------------------------------------------------------------------

    def build_epsilon_vector(self) -> np.ndarray:
        """Build per-factor decay vector from profile.

        Implements A2 hardening: maps each scoring dimension to its decay class,
        then resolves the per-class rate from profile.decay_class_rates.

        For each factor (by index in factor_names):
          1. Look up decay class from profile.factor_decay_classes (default: "standard")
          2. Look up rate from profile.decay_class_rates
          3. Set epsilon[i] = rate

        Factors absent from profile.factor_decay_classes receive the "standard" class.
        Decay classes absent from profile.decay_class_rates fall back to
        profile.epsilon_default.

        Reference: docs/gae_design_v10_6.md §8.4 (A2 hardening); blog Eq. 4c.

        Returns
        -------
        np.ndarray, shape (n_factors,)
            Per-factor ε values for use in Eq. 4c decay.
        """
        eps = np.empty(self.n_factors, dtype=np.float64)
        for i, name in enumerate(self.factor_names):
            decay_class = self.profile.factor_decay_classes.get(name, "standard")
            eps[i] = self.profile.decay_class_rates.get(
                decay_class, self.profile.epsilon_default
            )
        assert eps.shape == (self.n_factors,), (
            f"build_epsilon_vector: result shape {eps.shape} != ({self.n_factors},)"
        )
        return eps

    # ------------------------------------------------------------------
    # update  — Eq. 4b + Eq. 4c
    # ------------------------------------------------------------------

    def update(
        self,
        action_index: int,
        action_name: str,
        outcome: int,
        f: np.ndarray,
        confidence_at_decision: float | None = None,
        decision_source: str = "analyst",
        category_index: int = 0,
    ) -> WeightUpdate | None:
        """
        Apply one Hebbian update to the weight matrix.

        Implements:
            Eq. 4b: W[a, :] += α_eff · r(t) · f(t) · δ(t)
            Eq. 4c: W        ← (1 − ε_vector) · W
            Clamp:  W        ← clip(W, −W_CLAMP, +W_CLAMP)

        where δ(t) = 1.0 if outcome == +1, else profile.penalty_ratio.

        Reference: docs/gae_design_v10_6.md §8.2; blog Eq. 4b, 4c.

        Parameters
        ----------
        action_index : int
            Row of W corresponding to the action taken.
        action_name : str
            Human-readable action label.
        outcome : int
            r(t) ∈ {+1, −1}.
        f : np.ndarray, shape (1, n_f)
            Factor vector from the ORIGINAL decision — Requirement R4.
            Must not be recomputed at outcome time.
        confidence_at_decision : float or None
            System confidence when the decision was made.
            Required for A1 hardening; ignored when discount_strength == 0.
        decision_source : str
            "analyst" (default) → immediate learning.
            "autonomous" → deferred to pending_validations (C3 hardening).
            "validated_autonomous" → same as analyst (C3 resolved).

        Returns
        -------
        WeightUpdate
            Full provenance record of this update.
        None
            When decision_source is "autonomous" (learning is deferred).

        Raises
        ------
        AssertionError
            If outcome ∉ {+1, −1} or f has wrong shape.
        """
        assert outcome in (+1, -1), (
            f"outcome must be +1 or -1, got {outcome}"
        )
        assert isinstance(f, np.ndarray), (
            f"f must be np.ndarray, got {type(f)}"
        )
        assert f.shape == (1, self.n_factors), (
            f"f must be shape (1, {self.n_factors}), got {f.shape}"
        )

        # C3: Autonomous decisions enter pending validation — no immediate learning
        if decision_source == "autonomous":
            self.pending_validations.append(PendingValidation(
                entity_id=f"auto-{self.decision_count}",
                action=action_name,
                action_index=action_index,
                factor_vector=f.copy(),
                auto_decided_at=time.time(),
            ))
            return None

        # v5.0 delegation path: if ProfileScorer is wired in, use it
        if self.profile_scorer is not None:
            cu: CentroidUpdate = self.profile_scorer.update(
                f=f.flatten(),
                category_index=category_index,
                action_index=action_index,
                correct=(outcome == +1),
            )
            # Build a WeightUpdate record so history is populated (Charts A-D)
            self.decision_count += 1
            _zeros = np.zeros(self.n_factors)
            record = WeightUpdate(
                decision_number=self.decision_count,
                timestamp=time.time(),
                action_index=action_index,
                action_name=action_name,
                outcome=outcome,
                factor_vector=f.copy(),
                delta_applied=_zeros,
                W_before=self.W.copy(),
                W_after=self.W.copy(),
                alpha_effective=self.profile.learning_rate,
                confidence_at_decision=confidence_at_decision if confidence_at_decision is not None else 0.0,
                centroid_update=cu,
            )
            self.history.append(record)
            return record  # ProfileScorer owns the centroid update — W unchanged

        # Legacy path: existing W-matrix Hebbian update continues below
        # (unchanged)
        W_before = self.W.copy()

        # Eq. 4b — asymmetric δ(t): correct=1.0, incorrect=profile.penalty_ratio
        delta_t = 1.0 if outcome == +1 else self.profile.penalty_ratio

        # A1: Confidence-discounted learning rate (confirmation bias mitigation)
        alpha = self.profile.learning_rate
        if outcome == +1 and self.discount_strength > 0.0 and confidence_at_decision is not None:
            discount = 1.0 - self.discount_strength * confidence_at_decision
            alpha = self.profile.learning_rate * max(discount, 0.05)  # floor at 5%

        # Eq. 4b — compute update vector for action row, shape (n_f,)
        update_vector = alpha * outcome * f.flatten() * delta_t
        assert update_vector.shape == (self.n_factors,), (
            f"update_vector shape {update_vector.shape} != ({self.n_factors},)"
        )

        # Eq. 4b — update the action row
        self.W[action_index, :] += update_vector

        # Eq. 4c — per-factor multiplicative decay (A2 hardening)
        self.W *= (1.0 - self.epsilon_vector)

        # Clamp to [-W_CLAMP, +W_CLAMP]
        self.W = np.clip(self.W, -W_CLAMP, W_CLAMP)

        assert self.W.shape == (self.n_actions, self.n_factors), (
            f"W shape corrupted: {self.W.shape} after update"
        )

        # A4: Update reinforcement counts for provisional dimensions
        for dm in self.dimension_metadata:
            if dm.state == "provisional":
                col = dm.col_index
                if col < len(update_vector) and abs(update_vector[col]) > 1e-6:
                    dm.reinforcement_count += 1
                    if dm.reinforcement_count >= dm.establishment_threshold:
                        dm.state = "established"
                        self.epsilon_vector[col] = dm.decay_rate

        # A4: Prune provisional dimensions that have decayed to near-zero
        self._prune_provisional_dimensions()

        self.decision_count += 1

        record = WeightUpdate(
            decision_number=self.decision_count,
            timestamp=time.time(),
            action_index=action_index,
            action_name=action_name,
            outcome=outcome,
            factor_vector=f.copy(),
            delta_applied=update_vector.copy(),
            W_before=W_before,
            W_after=self.W.copy(),
            alpha_effective=alpha,
            confidence_at_decision=confidence_at_decision if confidence_at_decision is not None else 0.0,
        )
        self.history.append(record)
        return record

    # ------------------------------------------------------------------
    # ProfileScorer wiring — v5.0
    # ------------------------------------------------------------------

    def attach_profile_scorer(self, scorer: "ProfileScorer") -> None:
        """
        Wire a ProfileScorer into this LearningState.
        After this call, update() delegates to scorer.update().
        The legacy W-matrix update is bypassed.

        Used by SOC-PROF-2 to switch from W-matrix to profile learning.

        Reference: docs/gae_design_v10_6.md §9.5; GAE-PROF-3.
        """
        self.profile_scorer = scorer

    @property
    def is_profile_mode(self) -> bool:
        """True if update() delegates to ProfileScorer.

        Reference: docs/gae_design_v10_6.md §9.5; GAE-PROF-3.
        """
        return self.profile_scorer is not None

    # ------------------------------------------------------------------
    # expand_weight_matrix  — R5 + A4 hardening
    # ------------------------------------------------------------------

    def expand_weight_matrix(
        self,
        new_factor_name: str,
        init_scale: float = 0.05,
    ) -> None:
        """
        Add one new scoring dimension to W — Meta Loop (Requirement R5).

        W grows from (n_a, n_f) to (n_a, n_f + 1). The new column enters
        as "provisional" with accelerated decay (10× standard) and is
        auto-pruned if it fails to earn reinforcement (A4 hardening).

        Reference: docs/gae_design_v10_6.md §8.3; blog R5.

        Parameters
        ----------
        new_factor_name : str
            Name for the new scoring dimension.
        init_scale : float, default 0.05
            Standard deviation for the random initial column values.

        Raises
        ------
        AssertionError
            If *new_factor_name* is already registered.
        """
        assert new_factor_name not in self.factor_names, (
            f"Factor '{new_factor_name}' already exists in weight matrix"
        )

        rng = np.random.default_rng()
        # Floor init_scale so new column cannot be immediately pruned by
        # _prune_provisional_dimensions (theta_prune=0.01).
        init_scale = max(init_scale, 0.02)
        new_column = rng.standard_normal((self.n_actions, 1)) * init_scale
        self.W = np.hstack([self.W, new_column])

        provisional_decay = 0.01  # 10× faster than EPSILON=0.001
        new_col_idx = self.n_factors          # index BEFORE incrementing

        self.epsilon_vector = np.append(self.epsilon_vector, provisional_decay)
        self.n_factors += 1
        self.factor_names.append(new_factor_name)

        self.dimension_metadata.append(DimensionMetadata(
            factor_name=new_factor_name,
            col_index=new_col_idx,
            created_at=self.decision_count,
            state="provisional",
            decay_rate=provisional_decay,
            reinforcement_count=0,
        ))

        self.expansion_history.append({
            "decision_number": self.decision_count,
            "new_factor": new_factor_name,
            "col_index": new_col_idx,
            "new_shape": list(self.W.shape),
            "trigger": "discovery",
            "state": "provisional",
        })

        assert self.W.shape == (self.n_actions, self.n_factors), (
            f"W shape {self.W.shape} != ({self.n_actions}, {self.n_factors}) after expand"
        )
        assert self.epsilon_vector.shape == (self.n_factors,), (
            f"epsilon_vector shape {self.epsilon_vector.shape} != ({self.n_factors},)"
        )

    # ------------------------------------------------------------------
    # _prune_provisional_dimensions  — A4 internal
    # ------------------------------------------------------------------

    def _prune_provisional_dimensions(self, theta_prune: float = 0.01) -> None:
        """
        Remove provisional dimensions whose W column has decayed to near-zero.

        A4 hardening: false discoveries self-correct by decaying away.
        Uses DimensionMetadata.col_index (not enumerate index) to correctly
        identify the target W column regardless of how many original dimensions
        were present at construction time.

        Reference: docs/gae_design_v10_6.md §8.3 (A4 pruning).

        Parameters
        ----------
        theta_prune : float, default 0.01
            Maximum |W[:, col]| below which a provisional dimension is pruned.
        """
        to_remove_cols: list[int] = []
        for dm in self.dimension_metadata:
            if dm.state == "provisional":
                col = dm.col_index
                if np.max(np.abs(self.W[:, col])) < theta_prune:
                    to_remove_cols.append(col)

        if not to_remove_cols:
            return

        keep_cols = [i for i in range(self.n_factors) if i not in to_remove_cols]
        self.W = self.W[:, keep_cols]
        self.epsilon_vector = self.epsilon_vector[keep_cols]
        removed_set = set(to_remove_cols)
        self.factor_names = [n for i, n in enumerate(self.factor_names) if i not in removed_set]
        self.dimension_metadata = [
            dm for dm in self.dimension_metadata if dm.col_index not in removed_set
        ]
        # Remap col_index values for surviving metadata entries
        old_to_new = {old: new for new, old in enumerate(keep_cols)}
        for dm in self.dimension_metadata:
            dm.col_index = old_to_new[dm.col_index]

        self.n_factors = len(keep_cols)

        assert self.W.shape == (self.n_actions, self.n_factors), (
            f"W shape {self.W.shape} != ({self.n_actions}, {self.n_factors}) after prune"
        )

    # ------------------------------------------------------------------
    # process_pending_validations  — C3 internal
    # ------------------------------------------------------------------

    def process_pending_validations(self, incident_checker: Any) -> int:
        """
        Apply deferred learning for expired autonomous validations.

        C3 hardening: autonomous decisions are held in pending_validations.
        When the validation window expires, outcome is inferred from
        *incident_checker* and the learning update is applied.

        Reference: docs/gae_design_v10_6.md §8.2 (C3 hardening).

        Parameters
        ----------
        incident_checker : callable
            incident_checker(alert_id) → bool
            Returns True if the alert escalated to an incident (outcome = −1).

        Returns
        -------
        int
            Number of validations processed.
        """
        now = time.time()
        expired = [
            pv for pv in self.pending_validations
            if (now - pv.auto_decided_at) > pv.validation_window_days * 86400.0
        ]
        for pv in expired:
            r_t = -1 if incident_checker(pv.entity_id) else +1
            self.update(
                action_index=pv.action_index,
                action_name=pv.action,
                outcome=r_t,
                f=pv.factor_vector,
                decision_source="validated_autonomous",
            )
            self.pending_validations.remove(pv)
        return len(expired)
