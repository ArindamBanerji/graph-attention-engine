"""
Graph Attention Engine (GAE) v0.7.9
Public API surface.

Core scoring:
  ProfileScorer, ScoringResult, KernelType, build_profile_scorer

Oracle:
  OracleProvider, GTAlignedOracle, BernoulliOracle, OracleResult

Evaluation:
  EvaluationScenario, EvaluationReport, compute_ece, run_evaluation

Judgment:
  JudgmentResult, compute_judgment, CONFIDENCE_HIGH, CONFIDENCE_MEDIUM

Ablation:
  AblationResult, AblationReport, run_ablation

Calibration:
  CalibrationProfile, soc_calibration_profile, s2p_calibration_profile

Primitives (Tier 1):
  scaled_dot_product_attention, softmax

Learning (Tier 3):
  LearningState, DimensionMetadata, PendingValidation, WeightUpdate,
  ALPHA, EPSILON, LAMBDA_NEG, W_CLAMP

Convergence:
  get_convergence_metrics

Infrastructure:
  FactorComputedEvent, WeightsUpdatedEvent, ConvergenceEvent,
  PropertySpec, EmbeddingContract, SchemaContract,
  FactorComputer, assemble_factor_vector,
  save_state, load_state

Kernels:
  L2Kernel, DiagonalKernel, CovarianceEstimator, KernelSelector

Referral:
  ReferralEngine, ReferralRule, ReferralDecision, ReferralReason, OverrideDetector

Deprecated (TD-029 — remove in v6.0):
  score_entity, score_alert, score_with_profile, ProfileScoringResult
"""

from __future__ import annotations

__version__ = "0.7.12"

# ── Core Scoring ─────────────────────────────────────────────────────
from gae.kernels import L2Kernel, DiagonalKernel
from gae.covariance import CovarianceEstimator
from gae.kernel_selector import KernelSelector, KernelRecommendation
from gae.referral import (
    ReferralEngine,
    ReferralRule,
    ReferralDecision,
    ReferralReason,
    OverrideDetector,
)
from gae.profile_scorer import (
    CentroidUpdate,
    ProfileScorer,
    ScoringResult,
    KernelType,
    build_profile_scorer,
)
# Backward-compatible alias — ProfileScoringResult was the v4.x name
ProfileScoringResult = ScoringResult

# ── Oracle ───────────────────────────────────────────────────────────
from gae.oracle import (
    OracleProvider,
    OracleResult,
    GTAlignedOracle,
    BernoulliOracle,
)

# ── Evaluation ───────────────────────────────────────────────────────
from gae.evaluation import (
    EvaluationScenario,
    EvaluationReport,
    compute_ece,
    run_evaluation,
)

# ── Judgment ─────────────────────────────────────────────────────────
from gae.judgment import (
    JudgmentResult,
    compute_judgment,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
)

# ── Ablation ─────────────────────────────────────────────────────────
from gae.ablation import (
    AblationResult,
    AblationReport,
    run_ablation,
)

# ── Bootstrap ────────────────────────────────────────────────────────
from gae.bootstrap import (
    BootstrapResult,
    bootstrap_calibration,
)

# ── Calibration ──────────────────────────────────────────────────────
from gae.calibration import (
    CalibrationProfile,
    soc_calibration_profile,
    s2p_calibration_profile,
)

# ── Primitives (Tier 1, Eq. 1) ───────────────────────────────────────
from gae.primitives import scaled_dot_product_attention, softmax

# ── Learning (Tier 3, Eq. 4b / 4c) ──────────────────────────────────
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

# ── Convergence ──────────────────────────────────────────────────────
from gae.convergence import get_convergence_metrics

# ── Infrastructure — events, contracts, factors ──────────────────────
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

# ── Persistence ──────────────────────────────────────────────────────
from gae.store import save_state, load_state

# ── Deprecated (TD-029 — remove in v6.0) ────────────────────────────
from gae.scoring import score_entity, score_alert, score_with_profile

__all__ = [
    "__version__",
    # Core scoring
    "CentroidUpdate",
    "ProfileScorer",
    "ScoringResult",
    "KernelType",
    "build_profile_scorer",
    "ProfileScoringResult",       # backward-compat alias
    # Oracle
    "OracleProvider",
    "OracleResult",
    "GTAlignedOracle",
    "BernoulliOracle",
    # Evaluation
    "EvaluationScenario",
    "EvaluationReport",
    "compute_ece",
    "run_evaluation",
    # Judgment
    "JudgmentResult",
    "compute_judgment",
    "CONFIDENCE_HIGH",
    "CONFIDENCE_MEDIUM",
    # Ablation
    "AblationResult",
    "AblationReport",
    "run_ablation",
    # Calibration
    "CalibrationProfile",
    "soc_calibration_profile",
    "s2p_calibration_profile",
    # Primitives
    "scaled_dot_product_attention",
    "softmax",
    # Learning
    "ALPHA",
    "EPSILON",
    "LAMBDA_NEG",
    "W_CLAMP",
    "DimensionMetadata",
    "LearningState",
    "PendingValidation",
    "WeightUpdate",
    # Convergence
    "get_convergence_metrics",
    # Events
    "FactorComputedEvent",
    "WeightsUpdatedEvent",
    "ConvergenceEvent",
    # Contracts
    "PropertySpec",
    "EmbeddingContract",
    "SchemaContract",
    # Factors
    "FactorComputer",
    "assemble_factor_vector",
    # Persistence
    "save_state",
    "load_state",
    # Bootstrap
    "BootstrapResult",
    "bootstrap_calibration",
    # Kernels
    "L2Kernel",
    "DiagonalKernel",
    "CovarianceEstimator",
    "KernelSelector",
    "KernelRecommendation",
    # Referral
    "ReferralEngine",
    "ReferralRule",
    "ReferralDecision",
    "ReferralReason",
    "OverrideDetector",
    # Deprecated (TD-029 — remove in v6.0)
    "score_entity",
    "score_alert",
    "score_with_profile",
]
