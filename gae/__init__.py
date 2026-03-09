"""
Graph Attention Engine — public API surface.

Equations implemented here are documented in docs/gae_design_v5.md and the
companion blog post at dakshineshwari.net.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Calibration — domain-configurable hyperparameters
from gae.calibration import (
    CalibrationProfile,
    soc_calibration_profile,
    s2p_calibration_profile,
)

# Tier 1 — primitives (Eq. 1)
from gae.primitives import scaled_dot_product_attention, softmax

# Tier 2 — scoring matrix (Eq. 4)
from gae.scoring import ScoringResult, score_entity, score_alert  # score_alert is alias

# Tier 3 — weight learning (Eq. 4b, 4c)
from gae.learning import (
    ALPHA,
    EPSILON,
    LAMBDA_NEG,
    W_CLAMP,
    DimensionMetadata,
    LearningState,
    PendingValidation,
    WeightUpdate,
)

# Convergence monitoring
from gae.convergence import get_convergence_metrics

# Infrastructure — events, contracts, factors
from gae.events import (
    FactorComputedEvent,
    WeightsUpdatedEvent,
    ConvergenceEvent,
)
from gae.contracts import (
    PropertySpec,
    EmbeddingContract,
    SchemaContract,
)
from gae.factors import FactorComputer, assemble_factor_vector

# Persistence (note: gae.store.LearningState is the simpler JSON-only state)
from gae.store import save_state, load_state

# ── v5.0 ProfileScorer API ──────────────────────────────────────────
# Primary scoring path. Replaces score_alert() / ScoringMatrix (TD-029)
from gae.profile_scorer import (
    ProfileScorer,
    KernelType,
    ScoringResult as ProfileScoringResult,
    build_profile_scorer,
)
from gae.scoring import score_with_profile
# ────────────────────────────────────────────────────────────────────

# ── v5.0 Oracle API ─────────────────────────────────────────────────
from gae.oracle import (
    OracleProvider,
    OracleResult,
    GTAlignedOracle,
    BernoulliOracle,
)
# ────────────────────────────────────────────────────────────────────

__all__ = [
    "__version__",
    # calibration
    "CalibrationProfile",
    "soc_calibration_profile",
    "s2p_calibration_profile",
    # primitives
    "scaled_dot_product_attention",
    "softmax",
    # scoring
    "ScoringResult",
    "score_entity",
    "score_alert",      # backward-compatible alias for score_entity
    # learning
    "ALPHA",
    "EPSILON",
    "LAMBDA_NEG",
    "W_CLAMP",
    "DimensionMetadata",
    "LearningState",
    "PendingValidation",
    "WeightUpdate",
    # convergence
    "get_convergence_metrics",
    # events
    "FactorComputedEvent",
    "WeightsUpdatedEvent",
    "ConvergenceEvent",
    # contracts
    "PropertySpec",
    "EmbeddingContract",
    "SchemaContract",
    # factors
    "FactorComputer",
    "assemble_factor_vector",
    # persistence
    "save_state",
    "load_state",
    # v5.0 ProfileScorer
    "ProfileScorer",
    "KernelType",
    "ProfileScoringResult",
    "build_profile_scorer",
    "score_with_profile",
    # v5.0 Oracle
    "OracleProvider",
    "OracleResult",
    "GTAlignedOracle",
    "BernoulliOracle",
]
