# Graph Report - graph-attention-engine-v50  (2026-05-03)

## Corpus Check
- 84 files · ~200,122 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2798 nodes · 5333 edges · 76 communities detected
- Extraction: 64% EXTRACTED · 36% INFERRED · 0% AMBIGUOUS · INFERRED: 1919 edges (avg confidence: 0.66)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]

## God Nodes (most connected - your core abstractions)
1. `ProfileScorer` - 312 edges
2. `DiagonalKernel` - 183 edges
3. `KernelSelector` - 117 edges
4. `L2Kernel` - 112 edges
5. `CalibrationProfile` - 77 edges
6. `ConservationMonitor` - 69 edges
7. `ScoringResult` - 57 edges
8. `LearningState` - 56 edges
9. `CovarianceEstimator` - 49 edges
10. `derive_theta_min()` - 39 edges

## Surprising Connections (you probably didn't know these)
- `test_derive_theta_min_formula()` --calls--> `derive_theta_min()`  [INFERRED]
  tests\test_api_contract.py → gae\calibration.py
- `test_update_clips_to_unit_interval()` --calls--> `ProfileScorer`  [INFERRED]
  tests\test_profile_scorer.py → gae\profile_scorer.py
- `test_update_delta_norm_zero_on_frozen()` --calls--> `ProfileScorer`  [INFERRED]
  tests\test_profile_scorer.py → gae\profile_scorer.py
- `test_profile_scorer_dk_weights_raw_and_normalized_accessors()` --calls--> `ProfileScorer`  [INFERRED]
  tests\test_weight_provenance.py → gae\profile_scorer.py
- `test_profile_scorer_normalized_accessor_preserves_none_behavior()` --calls--> `ProfileScorer`  [INFERRED]
  tests\test_weight_provenance.py → gae\profile_scorer.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.01
Nodes (237): CalibrationProfile, compute_breach_window(), compute_eta_override(), compute_factor_mask(), compute_optimal_tau(), ConservationCheck, mask_to_array(), Result of a conservation law check α·q·V ≥ θ_min. (+229 more)

### Community 1 - "Community 1"
Cohesion: 0.01
Nodes (154): DKEstimator, Protocol for estimators that produce per-category `(C, D)` weights., Return estimated weights with shape `(n_categories, n_dims)`., build_profile_scorer(), CentroidUpdate, for_soc_twophase(), init_from_config(), KernelType (+146 more)

### Community 2 - "Community 2"
Cohesion: 0.02
Nodes (148): S2P domain defaults. 5:1 penalty, softer temperature.      Reference: docs/gae, SOC domain defaults. 20:1 penalty, sharp temperature.      Reference: docs/gae, s2p_calibration_profile(), soc_calibration_profile(), compute_judgment(), _confidence_tier(), _dominant_factors(), JudgmentResult (+140 more)

### Community 3 - "Community 3"
Cohesion: 0.02
Nodes (82): DiagonalKernel, L2Kernel, Gradient direction: f − μ.          Parameters         ----------         f, Diagonal kernel: K(f, μ) = (f−μ)ᵀ W (f−μ), W = diag(1/σ²).      v6.0 default f, Parameters         ----------         sigma : np.ndarray, shape (d,), Weighted squared L2 distance from f to each row of mu.          Parameters, Normalised weighted gradient: (W / w_max) ⊙ (f − μ).          Weights are norm, Max-normalized [0,1] view. Safe for display regardless of provenance.          T (+74 more)

### Community 4 - "Community 4"
Cohesion: 0.02
Nodes (63): KernelRecommendation, KernelScore, KernelSelector, KernelSelector — empirical kernel selection during shadow mode.  During shadow, Parameters         ----------         d : int             Number of factors., Instantiate all candidate kernels from measured σ.          L2:        plain s, Phase 2: rule-based kernel selection from measured σ and ρ_max.          Rules, Score one entity with ALL kernels and record analyst agreement.          Calle (+55 more)

### Community 5 - "Community 5"
Cohesion: 0.02
Nodes (98): centroid_distance_to_canonical(), compute_n_half(), gamma_threshold(), Frobenius distance between current centroid tensor and canonical snapshot., Computes ε_firm threshold below which γ ≤ 1 (re-convergence theorem).      The, Scalar convergence half-life.      N_half = ln(2) / ln(1/(1-η))      This is, BernoulliOracle, GTAlignedOracle (+90 more)

### Community 6 - "Community 6"
Cohesion: 0.03
Nodes (104): BatchCompositionPolicy, BatchHistory, BatchRecord, DefaultPromotionGate, FixedIntervalPolicy, GateVerdict, NoveltyThresholdPolicy, PromotionGate (+96 more)

### Community 7 - "Community 7"
Cohesion: 0.02
Nodes (124): _make_asym_scorers(), _make_asymmetric_scorer(), _make_gated_scorer(), _make_mask_scorer(), _make_pause_scorer(), _make_scorer(), make_simple_scorer(), Tests for gae.profile_scorer — ProfileScorer, KernelType, ScoringResult.  12 t (+116 more)

### Community 8 - "Community 8"
Cohesion: 0.03
Nodes (45): check_conservation(), check_meta_conservation(), compute_theta_min(), compute_transfer_prior(), derive_theta_min(), GAE Calibration — domain-configurable learning hyperparameters.  CalibrationPr, Conservation law floor: minimum daily correct-correction     signal required to, Deployment-aware conservation threshold: θ_min = 23.53 / (α × V).      Use thi (+37 more)

### Community 9 - "Community 9"
Cohesion: 0.05
Nodes (32): Enum, Pluggable scoring kernels for ProfileScorer.  v6.0: L2Kernel (default, proven), Declares the semantic type of DK weights.      Consumers use this to determine c, WeightProvenance, OverrideDetector, OverrideDetectorConfig, Referral routing — domain-agnostic protocol and engine.  A referral is a VETO, Protocol for domain-agnostic referral rules.      Each rule is a pure function (+24 more)

### Community 10 - "Community 10"
Cohesion: 0.05
Nodes (30): EmbeddingContract, PropertySpec, GAE schema contracts — declarative descriptions of node property schemas and em, Full schema declaration for one node type: its scalar properties and     option, Ordered property names, matching the factor_vector layout.          Reference:, Look up *name* in *raw*, falling back to the spec default when the         prop, Declares one scalar property that a node type exposes.      Reference: docs/ga, Return True if *value* satisfies the bounds declared in this spec.          Re (+22 more)

### Community 11 - "Community 11"
Cohesion: 0.05
Nodes (28): from_dict(), LearningState, load_state(), GAE persistence — JSON-backed storage for LearningState.  LearningState holds, Atomically persist *state* to *path* as JSON.      Uses a temp-file-and-rename, Load and return a LearningState from a JSON file written by *save_state*., Mutable container for the current learning state.      Reference: docs/gae_des, Convert to a JSON-serialisable dict.          Reference: docs/gae_design_v10_6 (+20 more)

### Community 12 - "Community 12"
Cohesion: 0.04
Nodes (35): DimensionMetadata, PendingValidation, GAE Learning — Hebbian weight learning.  Implements Eq. 4b and Eq. 4c from the, An autonomous decision awaiting outcome validation.      Autonomous decisions, Immutable record of one Eq. 4b weight update.      Carries full provenance: wh, Apply one Hebbian update to the weight matrix.          Implements:, Add one new scoring dimension to W — Meta Loop (Requirement R5).          W gr, Remove provisional dimensions whose W column has decayed to near-zero. (+27 more)

### Community 13 - "Community 13"
Cohesion: 0.07
Nodes (41): AblationReport, AblationResult, GAE Ablation — factor importance measurement via leave-one-out ablation.  Meas, Run ablation study: evaluate accuracy with each factor zeroed.      Steps:, Result of ablating one factor from the evaluation.      Reference: docs/gae_de, Full ablation study results across all factors.      Reference: docs/gae_desig, Return a copy of scenarios with factor_index zeroed out.      Does NOT modify, run_ablation() (+33 more)

### Community 14 - "Community 14"
Cohesion: 0.05
Nodes (19): ConservationMonitor, Two-layer quality conservation monitor.      Layer 1 (AMBER/RED): caller-drive, Update Layer 1 conservation status (AMBER/RED/GREEN).          Called by exter, Update CUSUM on EWMA of q̄ (Layer 2 — YELLOW warning).          Called interna, Record a per-decision quality score and run Layer 2 CUSUM.          Sets q_bas, Conservation monitor tests for GAE.  Tests ConservationMonitor Layer 1 (AMBER/, Additional calls beyond CALIBRATION_PERIOD must not update q_baseline., CUSUM alarm (yellow) must not change the GREEN Layer 1 status. (+11 more)

### Community 15 - "Community 15"
Cohesion: 0.06
Nodes (36): _assert_bootstrap_anchor_not_overwritten(), bootstrap_calibration(), bootstrap_enriched_prior(), BootstrapResult, GAE Bootstrap — domain-agnostic synthetic calibration for ProfileScorer.  Take, Run synthetic calibration rounds using the scorer's own prior as oracle., Orchestrate enriched bootstrap at P28 Phase 2.      Computes μ₀_enriched from, Raises RuntimeError if iks_bootstrap_soc.json already exists.     Called before (+28 more)

### Community 16 - "Community 16"
Cohesion: 0.07
Nodes (17): CovarianceEstimator, CovarianceSnapshot, Online covariance estimation with Ledoit-Wolf shrinkage.  v6.0: COLLECTS data,, Incorporate a new factor vector into the running statistics.          Uses exp, Compute the current covariance estimate with Ledoit-Wolf shrinkage.          F, Ledoit-Wolf optimal shrinkage intensity λ ∈ [0, 1].          Analytic approxim, Frozen view of the covariance estimator state at a point in time.      Attribu, Return current per-factor sigma (std dev) estimates, shape (d,).          Comp (+9 more)

### Community 17 - "Community 17"
Cohesion: 0.06
Nodes (25): make_scorer(), Adversarial input tests for GAE.  Verifies that the library raises ValueError, Factor values outside [0,1] are accepted; probabilities remain valid., Factor values >> 1 (1e6): softmax stability check., Factor values >> 1 (1e12): softmax must not produce NaN/Inf., Factor values near zero (1e-15): no underflow corruption., update() with f=zeros: centroids move toward zero, stay in [0,1]., update() with f=ones: centroids move toward 1, stay in [0,1]. (+17 more)

### Community 18 - "Community 18"
Cohesion: 0.11
Nodes (43): drive_to_phase2(), make_actions(), make_mu_binary(), make_mu_single_action(), make_mu_two_phase(), make_profile_learning_state(), make_two_phase_scorer(), FW-05 Phase A gate tests. (+35 more)

### Community 19 - "Community 19"
Cohesion: 0.08
Nodes (36): ConservationStateMachine, Owns conservation status transitions and registered transition handlers., Register a transition handler with signature handler(old, new)., Register a transition guard with signature guard() -> bool., Move to a new conservation state and fire registered handlers., Args:           mu:             Initial profile centroids, shape (n_categories,, _alarm_monitor(), _make_scorer() (+28 more)

### Community 20 - "Community 20"
Cohesion: 0.06
Nodes (20): compute_entropy(), GAE primitives — Tier 1 building blocks.  Implements the scaled dot-product at, Shannon entropy of probability distribution.      H(p) = -sum(p * log(p + eps), Numerically-stable row-wise softmax.      Implements:  softmax(x_i) = exp(x_i, Scaled dot-product attention (numpy implementation).      Implements Eq. 1 fro, scaled_dot_product_attention(), softmax(), Uniform probs → maximum entropy. (+12 more)

### Community 21 - "Community 21"
Cohesion: 0.1
Nodes (13): CoordinateDescentEstimator, Domain-agnostic per-category, per-dimension weight estimation.  This module esti, Apply deterministic per-category subsampling when needed., Select `(A, D)` centroids for a category or return shared centroids., Compute mean classification accuracy for one category.          Shapes         -, Coordinate-descent estimator for per-category dimension weights.      Parameters, Estimate per-category dimension weights.          Parameters         ----------, build_decisions_for_category() (+5 more)

### Community 22 - "Community 22"
Cohesion: 0.08
Nodes (19): make_scorer(), Determinism tests for GAE.  An open-source library must produce identical resu, update() must not mutate the caller's f array., Two separate ProfileScorer instances with identical mu produce identical scores., Updating s1 must not affect s2 (no shared mu reference)., ProfileScorer must copy mu at construction; external mutation must not propagate, scorer.centroids exposes the centroid tensor via the public API., Saving and restoring centroids via numpy gives identical score output. (+11 more)

### Community 23 - "Community 23"
Cohesion: 0.11
Nodes (7): make_scorer(), Shape contract tests for GAE.  Verifies that ProfileScorer, kernels, and all p, Shape contracts hold even after 100 update() calls., TestCentroidShapeContracts, TestInputShapeValidation, TestScoreOutputShapes, TestUpdateShapeContracts

### Community 24 - "Community 24"
Cohesion: 0.11
Nodes (25): compute_dominant_axis(), compute_enriched_bootstrap_prior_geom(), Per-factor centroid-separation score from variance across all (cat, act) pairs., Geometry-aware Empirical Bayes bootstrap (V-BOOTSTRAP-GEOM).      Combines enr, CategorySNR, compute_snr_report(), _default_names(), _pairwise_distances() (+17 more)

### Community 25 - "Community 25"
Cohesion: 0.12
Nodes (11): ConvergenceEvent, FactorComputedEvent, GAE event dataclasses — plain value objects emitted after key computations.  T, Emitted when a single factor vector has been assembled for one node.      Refe, Emitted after the learning rule updates factor weights.      Reference: docs/g, Emitted when the convergence monitor detects a state change.      Reference: d, WeightsUpdatedEvent, Tests for gae.events dataclasses. (+3 more)

### Community 26 - "Community 26"
Cohesion: 0.07
Nodes (8): Tests for gae.scoring — Eq. 4 scoring matrix., Exact replication of the verification script from the task description., Factor vector must be preserved unchanged (Requirement R4)., τ=1.0 is ordinary softmax(f·Wᵀ)., test_task_verification_smoke(), TestScoreAlert, TestShapeGuards, TestTemperature

### Community 27 - "Community 27"
Cohesion: 0.17
Nodes (19): evaluate_c1(), evaluate_c2(), evaluate_c3(), evaluate_c4(), evaluate_c5(), print_results(), tools/iks_bakeoff.py — V-IKS-BAKEOFF simulation.  Evaluates four IKS anchor op, C1 — IKS semantics: IKS(0)=0 AND IKS(convergence)≈100 for all profiles?     PAS (+11 more)

### Community 28 - "Community 28"
Cohesion: 0.15
Nodes (10): make_profile(), make_scorer_with_separated_centroids(), Mathematical property tests for GAE.  These tests verify the mathematical clai, Build a scorer where each action centroid is clearly different., Softmax output is strictly positive for finite distances., Lower L2 distance → higher softmax probability (monotonicity)., Ranked distances → inversely ranked probabilities., Softmax(logits + c) = Softmax(logits) — the -max shift must not change output. (+2 more)

### Community 29 - "Community 29"
Cohesion: 0.12
Nodes (18): Frozen scorer should return early — centroids must not change., update() with correct=True should not raise., update() with correct=False and gt_action_index should not raise., update() with correct=True should move centroids toward the factor vector., update() with invalid category_index should raise ValueError., update() with invalid action_index should raise ValueError., update() on category 0 should not change category 1 centroids., Larger eta_override on the correct=False path produces a larger centroid shift. (+10 more)

### Community 30 - "Community 30"
Cohesion: 0.11
Nodes (12): Zero factor vector should still produce a valid result., Large factor values should not crash., Negative factor values should produce a valid result., Out-of-bounds category index should raise., Wrong number of factors should raise., NaN in factor vector — scorer should either reject or produce finite confidence., test_score_all_zero_factors(), test_score_category_bounds() (+4 more)

### Community 31 - "Community 31"
Cohesion: 0.13
Nodes (8): tests/test_diagonal_kernel.py — DiagonalKernel sigma-based workflow.  Validate, Gradient magnitude must be larger for low-σ factors.         Confirms the learn, raw_weights returns W = 1/σ² before normalization.         Unlike .weights (max, Lower σ → higher W = 1/σ² → higher importance weight.          Factor 0 (σ=0.0, _W_baseline_max captures absolute signal scale, not just relative ordering., API surface test — catches import drift that broke SVM-003/004.          Verif, Low-σ factor must contribute MORE to distance than high-σ factor.         This, TestDiagonalKernelSigmaWorkflow

### Community 32 - "Community 32"
Cohesion: 0.35
Nodes (13): drive_phase2_with_weights(), drive_updates(), make_actions(), make_mu(), make_two_phase_scorer(), test_checkpoint_default_scorer(), test_checkpoint_partial_restore(), test_checkpoint_phase1_state() (+5 more)

### Community 33 - "Community 33"
Cohesion: 0.15
Nodes (7): NoveltyTracker, Novelty tracking for category-conditioned factor vectors.  Purpose ------- This, Return a novelty score for one factor vector in one category., Record one factor vector and update novelty diagnostics., Return the recent fraction of recorded vectors marked novel., Return the accumulated novelty score since the last reset., Reset the novelty accumulator for one category.

### Community 34 - "Community 34"
Cohesion: 0.25
Nodes (10): Consumer contract: score → update → score roundtrip., Full cycle: score, update with correct=True, score again., Updates to category 0 don't affect scoring in category 1., Save centroids then restore in a new scorer → identical scoring., 20 confirmed decisions should shift centroids measurably from 0.5., _soc_scorer(), test_checkpoint_restore_preserves_scoring(), test_learning_loop_20_decisions() (+2 more)

### Community 35 - "Community 35"
Cohesion: 0.48
Nodes (6): Setter copies data — mutation of source doesn't affect internal centroids., _soc_scorer(), test_centroids_setter_creates_copy(), test_centroids_setter_rejects_inf(), test_centroids_setter_rejects_neg_inf(), test_centroids_setter_shape_mismatch_raises()

### Community 36 - "Community 36"
Cohesion: 0.5
Nodes (3): rank_enrichment_opportunities(), GAE Enrichment Advisor — rank factors by expected Day-1 accuracy lift.  Refere, Rank factors by expected Day-1 accuracy lift from enrichment.      enrichment_

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Prompt 0 GAE: Structural map of Graph Attention Engine. Run from graph-attention

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): GAE minimal end-to-end example — IT helpdesk ticket classifier.  Demonstrates

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Graph Attention Engine (GAE) v0.7.9 Public API surface.  Core scoring:   Pro

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Number of scalar factors in the packed factor vector.          Reference: docs

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Current conservation state.

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Current Layer 1 status string ('GREEN', 'AMBER', 'RED').

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Calibration-period mean quality (0.0 until baseline is set).

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): True after CALIBRATION_PERIOD decisions have been recorded.

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Build a sigma-derived DK with max-normalized scoring weights.

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Build a discriminative DK from learned weights without normalization.

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Build an effective DK from shrinkage-blended weights.

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): max(σ)/min(σ) derived from weights = 1/σ².          noise_ratio = σ_max/σ_min

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Raw provenance-native weights before display normalization.         Use for cros

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): What kind of weights this kernel carries.

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Fraction of decisions where kernel matched analyst. 0.0 if no data.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Mean softmax confidence across all decisions. 0.0 if no data.

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Mean P(analyst_action | f) — proper scoring rule metric.          Unlike mean_

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): True if update() delegates to ProfileScorer.          Reference: docs/gae_desi

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Build GTAlignedOracle from an existing ProfileScorer.          Shares the same

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): [DEPRECATED] Factory: ProfileScorer with SOC defaults.          This method wi

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Factory: ProfileScorer with two-phase learning defaults.

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Profile centroid tensor, shape (n_categories, n_actions, n_factors).

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Set centroid tensor. Validates shape matches current mu.          This is the

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): Gate statistics for the min_confidence filter.          Returns counts of appl

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): Current conservation monitor status string ('GREEN', 'AMBER', 'RED', …).

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): True when learning is paused due to conservation AMBER/RED signal.

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): Access conservation state machine for handler registration.

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Build ProfileScorer from a nested config dictionary.          config_dict form

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Ordered list of reason code strings for each firing rule.          Returns e.g

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): Human-readable summary for the Evidence Ledger.          Format when referred:

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): Short rule identifier, e.g. 'R1'.

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (1): Enum reason this rule maps to.

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (1): True only when enabled AND enough positives have been collected.          Both

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (1): Reconstruct a LearningState from a dict produced by *to_dict*.          Refere

### Community 71 - "Community 71"
Cohesion: 1.0
Nodes (1): gamma_gt_1' if epsilon_firm > threshold, else 'gamma_lt_1'.

### Community 72 - "Community 72"
Cohesion: 1.0
Nodes (1): Feed baseline_window decisions at constant q to set _q_baseline.

### Community 73 - "Community 73"
Cohesion: 1.0
Nodes (1): 100 decisions produce 50-90% accuracy.

### Community 74 - "Community 74"
Cohesion: 1.0
Nodes (1): Phase 1 finds n_half (not DNF) with 200 samples.

### Community 75 - "Community 75"
Cohesion: 1.0
Nodes (1): After disruption, centroid distance to GT2 starts high.

## Knowledge Gaps
- **892 isolated node(s):** `Prompt 0 GAE: Structural map of Graph Attention Engine. Run from graph-attention`, `GAE minimal end-to-end example — IT helpdesk ticket classifier.  Demonstrates`, `GAE Ablation — factor importance measurement via leave-one-out ablation.  Meas`, `Result of ablating one factor from the evaluation.      Reference: docs/gae_de`, `Full ablation study results across all factors.      Reference: docs/gae_desig` (+887 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 37`** (2 nodes): `Prompt 0 GAE: Structural map of Graph Attention Engine. Run from graph-attention`, `prompt0_gae_structural_map.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (2 nodes): `run_example.py`, `GAE minimal end-to-end example — IT helpdesk ticket classifier.  Demonstrates`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (2 nodes): `__init__.py`, `Graph Attention Engine (GAE) v0.7.9 Public API surface.  Core scoring:   Pro`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Number of scalar factors in the packed factor vector.          Reference: docs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Current conservation state.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Current Layer 1 status string ('GREEN', 'AMBER', 'RED').`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Calibration-period mean quality (0.0 until baseline is set).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `True after CALIBRATION_PERIOD decisions have been recorded.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Build a sigma-derived DK with max-normalized scoring weights.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Build a discriminative DK from learned weights without normalization.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Build an effective DK from shrinkage-blended weights.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `max(σ)/min(σ) derived from weights = 1/σ².          noise_ratio = σ_max/σ_min`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Raw provenance-native weights before display normalization.         Use for cros`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `What kind of weights this kernel carries.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Fraction of decisions where kernel matched analyst. 0.0 if no data.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Mean softmax confidence across all decisions. 0.0 if no data.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Mean P(analyst_action | f) — proper scoring rule metric.          Unlike mean_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `True if update() delegates to ProfileScorer.          Reference: docs/gae_desi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Build GTAlignedOracle from an existing ProfileScorer.          Shares the same`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `[DEPRECATED] Factory: ProfileScorer with SOC defaults.          This method wi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Factory: ProfileScorer with two-phase learning defaults.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Profile centroid tensor, shape (n_categories, n_actions, n_factors).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Set centroid tensor. Validates shape matches current mu.          This is the`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `Gate statistics for the min_confidence filter.          Returns counts of appl`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `Current conservation monitor status string ('GREEN', 'AMBER', 'RED', …).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `True when learning is paused due to conservation AMBER/RED signal.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `Access conservation state machine for handler registration.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `Build ProfileScorer from a nested config dictionary.          config_dict form`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Ordered list of reason code strings for each firing rule.          Returns e.g`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `Human-readable summary for the Evidence Ledger.          Format when referred:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `Short rule identifier, e.g. 'R1'.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `Enum reason this rule maps to.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (1 nodes): `True only when enabled AND enough positives have been collected.          Both`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (1 nodes): `Reconstruct a LearningState from a dict produced by *to_dict*.          Refere`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 71`** (1 nodes): `gamma_gt_1' if epsilon_firm > threshold, else 'gamma_lt_1'.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 72`** (1 nodes): `Feed baseline_window decisions at constant q to set _q_baseline.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 73`** (1 nodes): `100 decisions produce 50-90% accuracy.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 74`** (1 nodes): `Phase 1 finds n_half (not DNF) with 200 samples.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 75`** (1 nodes): `After disruption, centroid distance to GT2 starts high.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ProfileScorer` connect `Community 2` to `Community 0`, `Community 1`, `Community 3`, `Community 5`, `Community 6`, `Community 7`, `Community 8`, `Community 11`, `Community 12`, `Community 13`, `Community 14`, `Community 15`, `Community 17`, `Community 18`, `Community 19`, `Community 21`, `Community 22`, `Community 23`, `Community 24`, `Community 28`, `Community 32`?**
  _High betweenness centrality (0.506) - this node is a cross-community bridge._
- **Why does `CalibrationProfile` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 7`, `Community 8`, `Community 10`, `Community 12`, `Community 15`, `Community 18`?**
  _High betweenness centrality (0.168) - this node is a cross-community bridge._
- **Why does `DiagonalKernel` connect `Community 3` to `Community 0`, `Community 1`, `Community 2`, `Community 4`, `Community 5`, `Community 8`, `Community 9`, `Community 14`, `Community 17`, `Community 22`, `Community 23`, `Community 28`?**
  _High betweenness centrality (0.116) - this node is a cross-community bridge._
- **Are the 284 inferred relationships involving `ProfileScorer` (e.g. with `BootstrapResult` and `DimensionMetadata`) actually correct?**
  _`ProfileScorer` has 284 INFERRED edges - model-reasoned connections that need verification._
- **Are the 176 inferred relationships involving `DiagonalKernel` (e.g. with `KernelScore` and `KernelRecommendation`) actually correct?**
  _`DiagonalKernel` has 176 INFERRED edges - model-reasoned connections that need verification._
- **Are the 105 inferred relationships involving `KernelSelector` (e.g. with `L2Kernel` and `DiagonalKernel`) actually correct?**
  _`KernelSelector` has 105 INFERRED edges - model-reasoned connections that need verification._
- **Are the 108 inferred relationships involving `L2Kernel` (e.g. with `KernelScore` and `KernelRecommendation`) actually correct?**
  _`L2Kernel` has 108 INFERRED edges - model-reasoned connections that need verification._