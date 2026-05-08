# Graph Attention Engine (GAE) Line-by-Line Code Review

## Scope
- Six files reviewed: `gae/profile_scorer.py`, `gae/kernels.py`, `gae/calibration.py`, `gae/convergence.py`, `gae/two_phase.py`, `gae/batch_pipeline.py`.
- Source code was read-only for this review.
- Generated from `REVIEW.part1.md`, `REVIEW.part2.md`, and `REVIEW.part3.md`.

## Executive Summary
- P1 findings summary: 2 total. Highest-impact findings are `DiagonalKernel.noise_ratio` overstates sigma-derived noise ratios and `compute_optimal_tau()` appears inverted relative to scorer softmax behavior.
- P2 findings summary: 40 total. Common themes are incomplete finite/range validation, live mutable state exposure, placeholder or hardcoded gate behavior, assert-based production guards, and inconsistent treatment of thresholds and history counters.
- P3 findings summary: 21 total. Common themes are stale design references, placeholder naming leaking into audit reasons, protocol/runtime-check limitations, mutable evidence records, and docstring/API clarity gaps.
- Highest-risk modules: `gae/kernels.py` for weighting math, `gae/calibration.py` for tau/conservation formulas, `gae/profile_scorer.py` for mutable state and scoring/update contracts, and `gae/batch_pipeline.py` for promotion-gate placeholder behavior.
# GAE Line-by-Line Review - Part 1

## gae/profile_scorer.py (1280 lines)

### Architecture
- This module owns the adaptive centroid scorer: it converts a factor vector into action probabilities, updates centroids from confirmation or override feedback, and exposes checkpoint/config helpers.
- Key design decisions visible in code: centroids are a mutable `(n_categories, n_actions, n_factors)` float64 tensor; public scoring is softmax over negative distances; updates are clipped by `MAX_ETA_DELTA` and then clipped back into `[0, 1]`; two-phase learning can buffer phase-2 decisions and replace L2 scoring with diagonal weights.
- Docstrings claim Eq. 4 scoring and Eq. 4b learning; actual code implements that for L2/diagonal paths, while cosine/dot/mahalanobis have separate behavior and some compatibility fallbacks.

### Class-by-Class / Function-by-Function
- **MAX_ETA_DELTA** (lines 38-41)
  - Purpose: Module-level cap for per-factor centroid delta.
  - Inputs: None.
  - Logic: Fixed float constant `0.05`.
  - Output: Used as symmetric clipping bound.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: Limits individual update coordinates in `ProfileScorer.update()`.

- **CentroidUpdate** (lines 45-62)
  - Purpose: Dataclass result describing an update attempt.
  - Inputs: Delta norms, category/action ids and names, decision count, optional GT delta, outcome.
  - Logic: Passive container.
  - Output: Instance returned by `update()`.
  - Side effects: None.
  - Edge cases handled: Default `gt_delta_norm=0.0`, default `outcome="applied"`.
  - Invariants/guards: None; code may return `"paused_conservation"` or `"phase2_buffered"` although the field comment only lists `"applied"`, `"gated_low_confidence"`, and `"frozen"`.

- **LearningStrategy** (lines 66-71)
  - Purpose: Enum for learning modes.
  - Inputs: None.
  - Logic: Defines `SINGLE_PHASE` and `TWO_PHASE`.
  - Output: Enum values.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: Used for constructor/config compatibility.

- **KernelType** (lines 74-93)
  - Purpose: Enum of scoring kernel names.
  - Inputs: None.
  - Logic: Defines `L2`, `COSINE`, `DOT`, `MAHALANOBIS`, `DIAGONAL`.
  - Output: Enum values.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: String values are accepted by `init_from_config()` through `KernelType(...)`.

- **ScoringResult** (lines 97-127)
  - Purpose: Dataclass return type for `score()`.
  - Inputs: Action id/name, probabilities, distances, confidence, entropy, confidence gap.
  - Logic: Passive container.
  - Output: Instance returned by `score()`.
  - Side effects: None.
  - Edge cases handled: Default entropy and confidence gap are `None`.
  - Invariants/guards: No runtime validation of probability sum, distance shape, or confidence range inside the dataclass.

- **ProfileScorer** (lines 130-1249)
  - Purpose: Main mutable scoring and learning object.
  - Inputs: Centroid tensor, categories, actions, optional profile, kernel configuration, learning rates, masks, conservation and two-phase options.
  - Logic: Stores state, dispatches to kernels, gates updates, mutates centroids, and supports diagnostics/checkpoints.
  - Output: Scoring and update APIs plus state views.
  - Side effects: Mutates `self.mu`, counts, gate counters, category phase state, DK weights, decision buffer, and warning flags.
  - Edge cases handled: Shape mismatches, non-finite factor vectors/centroids, invalid learning rates, low confidence, conservation pause, frozen scorer, missing GT on override, MAHALANOBIS fallback when covariance missing.
  - Invariants/guards: Core shape invariant is `mu.shape == (n_categories, n_actions, n_factors)`.

- **ProfileScorer.__init__** (lines 155-335)
  - Purpose: Construct scorer state and defaults.
  - Inputs: `mu`, `categories`, `actions`, optional `profile`, `kernel`, `scoring_kernel`, learning parameters, mask, conservation flag, and two-phase strategy objects.
  - Logic: Converts `mu` to float64, stores category/action metadata, derives defaults from `profile.extensions` or hardcoded values, builds a default `L2Kernel`, initializes counters, two-phase state, conservation state, and optional MAHALANOBIS covariance placeholder.
  - Output: Initialized object.
  - Side effects: Emits deprecation/runtime warnings for old positional `LearningStrategy`, `DOT`, and `MAHALANOBIS`.
  - Edge cases handled: Rejects old positional `LearningStrategy` in the `kernel` slot, invalid `mu` rank/action count, `eta_neg >= 1.0`, negative learning rates, invalid factor-mask shape, and incompatible two-phase options.
  - Invariants/guards: `mu` rank 3, action dimension matches actions, `factor_mask.shape == (n_factors,)`, rates nonnegative, `eta_neg < 1.0`.
  - Actual vs docstring: Constructor accepts a `CalibrationProfile`, but only uses `temperature` and selected `extensions`; it does not consume all fields from the profile dataclass.

- **ProfileScorer.for_soc** (lines 338-366)
  - Purpose: Deprecated SOC convenience factory.
  - Inputs: `mu`, optional categories/actions, `eta_override`, `auto_pause`.
  - Logic: Fills default action names and delegates to constructor with conservative SOC defaults.
  - Output: `ProfileScorer`.
  - Side effects: Emits `DeprecationWarning`.
  - Edge cases handled: Default categories/actions when omitted.
  - Invariants/guards: Constructor performs shape/rate validation.

- **ProfileScorer.for_soc_twophase** (lines 369-397)
  - Purpose: SOC factory with two-phase learning defaults.
  - Inputs: `mu`, optional categories/actions, phase policy, DK estimator, shrinkage schedule, `eta_override`, `auto_pause`.
  - Logic: Defaults to `DecisionCountPolicy(200)`, `CoordinateDescentEstimator()`, and `FixedAlpha(0.5)`, then delegates to constructor.
  - Output: `ProfileScorer`.
  - Side effects: None beyond constructor warnings/state.
  - Edge cases handled: Default categories/actions and strategy objects.
  - Invariants/guards: Constructor validates compatible strategy fields.

- **ProfileScorer.score** (lines 403-506)
  - Purpose: Score a factor vector against one category and recommend an action.
  - Inputs: `f` as `(n_factors,)`, `category_index`.
  - Logic: Validates category/factor/centroid shape and finiteness, applies `factor_mask`, optionally computes phase-2 DK weights, gets distances from the active kernel, computes stabilized softmax over `-distances / tau`, then calculates argmax, entropy, confidence, and confidence gap.
  - Output: `ScoringResult`.
  - Side effects: None expected.
  - Edge cases handled: Invalid category via assert, wrong factor shape via assert, non-finite inputs via `ValueError`, nonpositive `tau`, invalid distance/probability shapes via assert.
  - Invariants/guards: `distances.shape == (n_actions,)`, `probs.shape == (n_actions,)`; masked dimensions contribute zero to distance for L2/diagonal/phase-2 paths.
  - Actual vs docstring: Docstring says lower distance means better and inputs in `[0,1]`; code checks finiteness and shape, but does not enforce `f` values in `[0,1]`.

- **ProfileScorer._compute_distances** (lines 508-568)
  - Purpose: Internal distance dispatch for legacy non-pluggable kernels.
  - Inputs: Masked or raw `f`, category centroid matrix `mu_c`, and category index.
  - Logic: L2 returns squared distances; COSINE returns `1 - similarity`; DOT returns negative dot product; MAHALANOBIS uses per-action inverse covariance or falls back to L2 with warning.
  - Output: Distance vector `(n_actions,)`.
  - Side effects: MAHALANOBIS fallback emits `UserWarning`.
  - Edge cases handled: MAHALANOBIS missing covariance, unknown kernel.
  - Invariants/guards: L2 diff shape assert; covariance inverse shape assert; final MAHALANOBIS distance shape assert.

- **ProfileScorer._category_name** (lines 574-578)
  - Purpose: Safe category label lookup.
  - Inputs: Category index.
  - Logic: Returns indexed category name when in range, otherwise `category_<index>`.
  - Output: String.
  - Side effects: None.
  - Edge cases handled: Out-of-range indexes.
  - Invariants/guards: None.

- **ProfileScorer._action_name** (lines 580-584)
  - Purpose: Safe action label lookup.
  - Inputs: Action index.
  - Logic: Returns indexed action name when in range, otherwise `action_<index>`.
  - Output: String.
  - Side effects: None.
  - Edge cases handled: Out-of-range indexes.
  - Invariants/guards: None.

- **ProfileScorer.freeze** (lines 586-593)
  - Purpose: Disable centroid updates.
  - Inputs: None.
  - Logic: Sets `_frozen=True`.
  - Output: None.
  - Side effects: Mutates freeze state.
  - Edge cases handled: None.
  - Invariants/guards: Subsequent `update()` returns before validation of category/action bounds.

- **ProfileScorer.unfreeze** (lines 595-597)
  - Purpose: Re-enable centroid updates.
  - Inputs: None.
  - Logic: Sets `_frozen=False`.
  - Output: None.
  - Side effects: Mutates freeze state.
  - Edge cases handled: None.
  - Invariants/guards: None.

- **ProfileScorer.set_kernel** (lines 599-614)
  - Purpose: Change active kernel enum for experimentation.
  - Inputs: `kernel`.
  - Logic: Assigns `self.kernel` and returns `self`.
  - Output: `ProfileScorer`.
  - Side effects: Mutates active kernel.
  - Edge cases handled: None.
  - Invariants/guards: No type validation.

- **ProfileScorer.kernel_weight_refresh** (lines 616-649)
  - Purpose: Replace the scorer's pluggable kernel using fresh per-factor sigma estimates.
  - Inputs: `sigma_per_factor`.
  - Logic: Requires current `scoring_kernel` to support `refresh_weights()`, calls it, assigns returned kernel, and returns `self`.
  - Output: `ProfileScorer`.
  - Side effects: Mutates `self.scoring_kernel`.
  - Edge cases handled: Missing method raises `TypeError`.
  - Invariants/guards: Shape/value validation delegated to kernel implementation.

- **ProfileScorer.centroids** (getter, lines 652-658)
  - Purpose: Public read view of centroid tensor.
  - Inputs: None.
  - Logic: Returns `self.mu`.
  - Output: Live ndarray, not a copy.
  - Side effects: None on access; callers can mutate internal state through returned array.
  - Edge cases handled: None.
  - Invariants/guards: None.

- **ProfileScorer.centroids** (setter, lines 661-676)
  - Purpose: Validate and replace centroid tensor.
  - Inputs: New array.
  - Logic: Converts to float64, checks exact shape and finiteness, then assigns.
  - Output: None.
  - Side effects: Replaces `self.mu`.
  - Edge cases handled: Wrong shape, NaN, Inf.
  - Invariants/guards: New shape must equal existing centroid shape.

- **ProfileScorer.update_gate_stats** (property, lines 679-696)
  - Purpose: Report applied/gated update counters.
  - Inputs: None.
  - Logic: Computes total, applied, gated, and gate rate.
  - Output: Dict.
  - Side effects: None.
  - Edge cases handled: Zero total returns gate rate `0.0`.
  - Invariants/guards: Counter values are read from mutable internal fields.

- **ProfileScorer.set_conservation_status** (lines 698-717)
  - Purpose: Set conservation status and auto-pause learning for AMBER/RED when enabled.
  - Inputs: Status string.
  - Logic: Stores status uppercased, sets `_paused_by_conservation` if `auto_pause_on_conservation` and status is AMBER/RED.
  - Output: None.
  - Side effects: Mutates conservation status and pause flag.
  - Edge cases handled: Lowercase status normalized.
  - Invariants/guards: No validation that status is one of GREEN/AMBER/RED.

- **ProfileScorer.conservation_status** (property, lines 720-722)
  - Purpose: Expose current conservation status.
  - Inputs: None.
  - Logic: Returns stored string.
  - Output: String.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: None.

- **ProfileScorer.is_paused** (property, lines 725-727)
  - Purpose: Expose conservation pause flag.
  - Inputs: None.
  - Logic: Returns boolean `_paused_by_conservation`.
  - Output: Bool.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: Does not include `_frozen`.

- **ProfileScorer._compute_gradient** (lines 733-751)
  - Purpose: Internal centroid movement direction.
  - Inputs: `f` and one centroid vector.
  - Logic: Uses pluggable kernel gradient for L2/DIAGONAL, otherwise returns raw residual `f - mu`.
  - Output: Gradient vector `(n_factors,)`.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: Assumes caller already validated shapes and finiteness.

- **ProfileScorer.update** (lines 753-986)
  - Purpose: Mutate centroid state from a confirmed or overridden recommendation.
  - Inputs: `f`, `category_index`, `action_index`, `correct`, optional `gt_action_index`, optional `confidence`.
  - Logic: Validates factor shape/finiteness and temperature, gates conservation and low-confidence updates, skips if frozen, validates indexes, buffers phase-2 decisions, computes decayed learning rates, warns about asymmetric override rates, pulls correct centroid or pushes predicted and optionally pulls GT, clips deltas by `MAX_ETA_DELTA`, clips category centroids to `[0,1]`, records phase decisions, increments counters, and returns `CentroidUpdate`.
  - Output: `CentroidUpdate`.
  - Side effects: Mutates centroids, counts, decision count, applied/gated counters, two-phase category state, warning flag, and decision buffer.
  - Edge cases handled: Non-finite vectors, invalid tau, conservation pause, low confidence, frozen state, invalid category/action/GT indexes, variance-learning phase buffering, missing GT on incorrect update.
  - Invariants/guards: Factor shape exact; all update deltas coordinate-clipped; all centroids in updated category clipped into `[0,1]`.
  - Actual vs docstring: Docstring says `gt_action_index` is required for full incorrect learning; code supports missing GT with deprecation warning and push-only behavior. Frozen returns default outcome `"applied"`, not the field-commented `"frozen"`.

- **ProfileScorer.get_phase** (lines 988-992)
  - Purpose: Return category learning phase.
  - Inputs: Category index.
  - Logic: Returns `"single_phase"` if no category state exists, otherwise state phase.
  - Output: String.
  - Side effects: None.
  - Edge cases handled: No two-phase state.
  - Invariants/guards: No category index validation.

- **ProfileScorer.get_alpha** (lines 994-1000)
  - Purpose: Return current shrinkage alpha for a category.
  - Inputs: Category index.
  - Logic: Returns `None` without learning strategy/state; otherwise delegates to shrinkage schedule.
  - Output: Float or None.
  - Side effects: None.
  - Edge cases handled: No two-phase strategy/state.
  - Invariants/guards: No category index validation.

- **ProfileScorer.get_dk_weights** (lines 1002-1006)
  - Purpose: Return DK weights for a category.
  - Inputs: Category index.
  - Logic: Returns `None` if `_dk_weights` absent, otherwise the row for category.
  - Output: ndarray or None.
  - Side effects: Returns live slice.
  - Edge cases handled: No DK weights.
  - Invariants/guards: No category index validation.

- **ProfileScorer.reestimate_dk** (lines 1008-1023)
  - Purpose: Recompute DK weights from buffered decisions.
  - Inputs: Category index.
  - Logic: Filters `_decision_buffer` for correct decisions for the category, delegates to estimator, initializes `_dk_weights` to ones if needed, and stores the result.
  - Output: ndarray weights.
  - Side effects: Mutates `_dk_weights`.
  - Edge cases handled: Missing strategy, missing buffer, no samples.
  - Invariants/guards: Estimator validates/returns weights.

- **ProfileScorer.diagnostics** (lines 1029-1075)
  - Purpose: Report centroid separation and learning state.
  - Inputs: None.
  - Logic: Computes pairwise L2 separations per category and overall mean, then returns dict with counts, kernel, frozen flag, gate stats, and optional two-phase state.
  - Output: Dict.
  - Side effects: None.
  - Edge cases handled: Categories with fewer than two distances get `0.0`; no categories produce overall `0.0`.
  - Invariants/guards: Assumes `self.mu` shape is valid.

- **ProfileScorer.set_covariance** (lines 1077-1096)
  - Purpose: Set inverse covariance tensor for MAHALANOBIS scoring.
  - Inputs: `cov_inv` shaped `(n_categories, n_actions, n_factors, n_factors)`.
  - Logic: Requires current kernel to be MAHALANOBIS, validates shape, stores float64 copy/reference from `np.asarray`.
  - Output: None.
  - Side effects: Mutates `_cov_inv`.
  - Edge cases handled: Non-MAHALANOBIS kernel and wrong shape.
  - Invariants/guards: Shape exact; no finiteness or positive-definiteness check.

- **ProfileScorer.__getstate__** (lines 1098-1100)
  - Purpose: Pickle support.
  - Inputs: None.
  - Logic: Returns shallow dict copy of instance state.
  - Output: Dict.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: None.

- **ProfileScorer.get_checkpoint_state** (lines 1102-1127)
  - Purpose: Produce explicit checkpoint state.
  - Inputs: None.
  - Logic: Copies centroids, serializes two-phase category state summaries, copies DK weights, and records decision-buffer size.
  - Output: Dict.
  - Side effects: None.
  - Edge cases handled: Missing category states, DK weights, decision buffer.
  - Invariants/guards: Does not serialize buffered decision contents.

- **ProfileScorer.restore_checkpoint_state** (lines 1129-1160)
  - Purpose: Restore explicit checkpoint state.
  - Inputs: State dict.
  - Logic: Restores centroids through setter, DK weights by copy, and selected category state fields when present.
  - Output: None.
  - Side effects: Mutates centroids, DK weights, and category states.
  - Edge cases handled: Missing keys and category-state indexes beyond current count are ignored.
  - Invariants/guards: Centroid setter validates shape/finiteness; DK weight shape is not validated here.

- **ProfileScorer.__setstate__** (lines 1162-1166)
  - Purpose: Pickle compatibility backfill.
  - Inputs: State dict.
  - Logic: Updates instance dict and fills missing two-phase fields.
  - Output: None.
  - Side effects: Mutates whole object state.
  - Edge cases handled: Older pickles missing `_learning_strategy`, `_category_states`, `_dk_weights`, or `_decision_buffer`.
  - Invariants/guards: No validation of restored `mu` or hyperparameters.

- **ProfileScorer.init_from_config** (lines 1173-1249)
  - Purpose: Build a scorer from category/action/centroid config.
  - Inputs: Config dict with `categories`, optional `actions`, `centroids`, `temperature`, `eta`, `eta_neg`, `kernel`, `count_decay`.
  - Logic: Validates required categories/centroids, infers factor count from first centroid vector, initializes all centroids to 0.5, fills configured values, validates ranges, converts kernel string, and delegates to constructor.
  - Output: `ProfileScorer`.
  - Side effects: None beyond constructor warnings.
  - Edge cases handled: Missing categories/centroids, no valid centroids, unknown category/action names, centroid vector length mismatches, values outside `[0,1]`, unknown kernel values.
  - Invariants/guards: All initialized centroid values must be in `[0,1]`.

- **build_profile_scorer** (lines 1252-1280)
  - Purpose: Compatibility builder wrapper.
  - Inputs: `centroid_config`, `n_factors`, `actions`.
  - Logic: Adds actions to config copy when provided and delegates to `ProfileScorer.init_from_config()`.
  - Output: `ProfileScorer`.
  - Side effects: None.
  - Edge cases handled: None directly.
  - Invariants/guards: `n_factors` is documented but unused by the code.

### Invariants Enforced
- `MAX_ETA_DELTA = 0.05` clips every coordinate of update deltas before centroids mutate (lines 38-41, 920, 942, 951, 958).
- `mu` must be rank 3 and action dimension must match `len(actions)` at construction (lines 226-230).
- `eta_neg < 1.0`, `eta >= 0`, `eta_neg >= 0`, and `eta_override >= 0` when provided (lines 254-270).
- `factor_mask` must have shape `(n_factors,)` (lines 278-285).
- `score()` asserts category range, exact factor shape, exact centroid shape, exact distance shape, and exact probability shape (lines 419-424, 433-440, 478-489).
- `score()` rejects non-finite factor vectors, non-finite category centroids, and nonpositive `tau` (lines 426-440).
- `_compute_distances()` checks L2 diff shape, MAHALANOBIS covariance shape, MAHALANOBIS output shape, and raises on unknown kernel (lines 521-568).
- `centroids` setter validates shape and finiteness (lines 661-676).
- `update()` validates factor shape/finiteness/tau, then after non-mutating gates validates category, action, and GT bounds (lines 798-865).
- `update()` gates conservation pause and low confidence before index bounds checks (lines 811-839).
- `update()` clips all centroids in the updated category into `[0,1]` (lines 962-963).
- `set_covariance()` requires MAHALANOBIS kernel and exact covariance tensor shape (lines 1077-1096).
- `init_from_config()` enforces categories/centroids presence, action/category name validity, vector length consistency, and centroid value range `[0,1]` (lines 1173-1249).

### Potential Issues
#### P1
- None identified in `profile_scorer.py` during this pass.

#### P2
- `centroids` getter returns the live internal ndarray, so external callers can bypass setter validation and mutate scorer state directly (lines 652-658).
- `score()` does not check that kernel-produced distances are finite before softmax; custom kernels or malformed diagonal/MAHALANOBIS state can propagate NaN/Inf into probabilities (lines 473-486).
- If `set_kernel(KernelType.MAHALANOBIS)` is called on a scorer not originally constructed as MAHALANOBIS, `_compute_distances()` reads `self._cov_inv` before that attribute is guaranteed to exist (lines 599-614, 540-542).
- Frozen updates return a `CentroidUpdate` with default `outcome="applied"` instead of an explicit frozen outcome, which can mislead callers using audit/result logs (lines 841-851).
- Low-confidence and conservation gates run before category/action bounds validation, so invalid indexes can produce synthetic names instead of validation errors when a gate fires (lines 811-839, 853-865).
- `build_profile_scorer()` accepts `n_factors` but never validates it, despite the docstring saying it is used for validation (lines 1252-1280).
- `set_covariance()` does not validate finite values, symmetry, or positive definiteness of inverse covariance matrices (lines 1077-1096).
- `get_dk_weights()` returns a live slice of internal `_dk_weights`, allowing external mutation (lines 1002-1006).

#### P3
- The module and many docstrings reference `docs/gae_design_v10_6.md` while the module header claims v10.7 hardening (lines 1-12, 417, 519, 796).
- `CentroidUpdate.outcome` field comment omits outcomes actually returned by code, including `"paused_conservation"` and `"phase2_buffered"` (lines 58-62, 811-839, 867-885).
- The asymmetric-rate warning is evaluated before `correct` is considered, so a confirm update can warn about future override behavior (lines 891-905).
- `score()` docstring says factor values are in `[0,1]`, but code does not enforce that range for `f` (lines 410-427).

### Cross-Module Dependencies
- Assumes `gae.kernels.L2Kernel` and `gae.kernels.DiagonalKernel` expose `compute_distance()`, `compute_gradient()`, and optionally `refresh_weights()`.
- Assumes `gae.calibration.CalibrationProfile` has `temperature` and optional `extensions`.
- Assumes `gae.convergence.compute_entropy()` accepts a probability vector, and `compute_effective_weights()` returns a diagonal weight vector compatible with `DiagonalKernel(weights=...)`.
- Assumes `gae.two_phase` states have `phase`, `freeze_point`, `decision_count`, `record_decision()`, and `freeze()` fields/methods.
- Other modules and consumers likely assume `ProfileScorer.score()` is pure, `ProfileScorer.update()` is the mutating learning path, and `centroids`/`mu` expose the canonical centroid tensor.

## gae/kernels.py (278 lines)

### Architecture
- This module defines the pluggable scoring-kernel protocol plus L2 and diagonal weighted implementations.
- Key design decisions visible in code: kernels are lightweight and mostly stateless except `DiagonalKernel`; shape/finite validation is minimal and often delegated to `ProfileScorer`; diagonal gradients use max-normalized weights while diagonal distances use stored weights directly.
- Docstrings claim protocol contracts for `(d,)`, `(A,d)`, and `(A,)`; actual concrete methods rely on NumPy broadcasting and do not enforce those shapes.

### Class-by-Class / Function-by-Function
- **ScoringKernel** (lines 24-61)
  - Purpose: Runtime-checkable protocol for scoring kernels.
  - Inputs: Implementers must expose `compute_distance()` and `compute_gradient()`.
  - Logic: Structural protocol only.
  - Output: None directly.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: `@runtime_checkable` allows `isinstance` structural checks, but return shapes/values are not enforceable by the protocol.

- **ScoringKernel.compute_distance** (lines 27-43)
  - Purpose: Define required distance method.
  - Inputs: `f` shaped `(d,)`, `mu` shaped `(A,d)`.
  - Logic: Protocol ellipsis.
  - Output: Declared as ndarray `(A,)`.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: Documentation says non-negative distance per action.

- **ScoringKernel.compute_gradient** (lines 45-61)
  - Purpose: Define required gradient method.
  - Inputs: `f` and one centroid vector `mu`, both `(d,)`.
  - Logic: Protocol ellipsis.
  - Output: Declared as ndarray `(d,)`.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: Documentation says direction to move centroid toward `f`.

- **L2Kernel** (lines 64-105)
  - Purpose: Default squared L2 kernel.
  - Inputs: Factor vector and centroid arrays.
  - Logic: Uses raw residuals with equal weights.
  - Output: Squared distance vector and residual gradient.
  - Side effects: None.
  - Edge cases handled: None inside the class.
  - Invariants/guards: None; callers must validate shapes and finiteness.

- **L2Kernel.compute_distance** (lines 76-90)
  - Purpose: Compute squared L2 distance from `f` to each row of `mu`.
  - Inputs: `f` `(d,)`, `mu` expected `(A,d)`.
  - Logic: `diff = f - mu`; returns `np.sum(diff ** 2, axis=-1)`.
  - Output: Expected `(A,)`.
  - Side effects: None.
  - Edge cases handled: NumPy broadcasting handles compatible shapes.
  - Invariants/guards: No explicit shape, rank, non-negativity, or finite checks.

- **L2Kernel.compute_gradient** (lines 92-105)
  - Purpose: Compute centroid movement direction.
  - Inputs: `f` `(d,)`, one centroid `mu` `(d,)`.
  - Logic: Returns `f - mu`.
  - Output: Expected `(d,)`.
  - Side effects: None.
  - Edge cases handled: NumPy broadcasting.
  - Invariants/guards: No explicit shape or finite checks.

- **DiagonalKernel** (lines 108-278)
  - Purpose: Weighted diagonal kernel for per-factor reliability/noise weighting.
  - Inputs: Either `sigma` or direct `weights`.
  - Logic: Computes or stores positive factor weights; distances are weighted squared L2; gradients are normalized by max weight; `refresh_weights()` creates a new kernel.
  - Output: Kernel with `weights`, `sigma`, `_W_baseline_max`.
  - Side effects: None outside object construction.
  - Edge cases handled: Rejects both/neither `sigma` and `weights`, non-1D arrays by assert, nonpositive sigma, nonfinite/nonpositive weights, refresh shape mismatch.
  - Invariants/guards: `weights` path requires finite positive values; `sigma` path requires only positive values.

- **DiagonalKernel.__init__** (lines 132-186)
  - Purpose: Build diagonal weights from sigma or direct weights.
  - Inputs: `sigma` `(d,)` or keyword-only `weights` `(d,)`.
  - Logic: Rejects invalid argument combinations; validates rank; weights path stores direct finite positive weights and derives sigma; sigma path computes raw `W=1/sigma**2`, stores `_W_baseline_max=W.max()`, normalized `weights=W/W.max()`, and sigma copy.
  - Output: Initialized kernel.
  - Side effects: None.
  - Edge cases handled: Both/neither arguments, wrong rank, nonpositive sigma, nonfinite or nonpositive direct weights.
  - Invariants/guards: Sigma path does not reject NaN or Inf explicitly.

- **DiagonalKernel.compute_distance** (lines 188-202)
  - Purpose: Compute weighted squared L2 distances.
  - Inputs: `f` expected `(d,)`, `mu` expected `(A,d)`.
  - Logic: Uses broadcast residual and returns `sum(self.weights * diff**2, axis=-1)`.
  - Output: Expected `(A,)`.
  - Side effects: None.
  - Edge cases handled: NumPy broadcasting.
  - Invariants/guards: No explicit shape/finite check beyond constructor weight state.

- **DiagonalKernel.compute_gradient** (lines 204-227)
  - Purpose: Compute max-normalized weighted residual.
  - Inputs: `f` `(d,)`, `mu` `(d,)`.
  - Logic: Computes `w_max = max(self.weights.max(), 1e-9)` and returns `(self.weights / w_max) * (f - mu)`.
  - Output: Expected `(d,)`.
  - Side effects: None.
  - Edge cases handled: Very small max weight guarded by `1e-9`.
  - Invariants/guards: No shape/finite checks; assumes `self.weights` is valid.

- **DiagonalKernel.noise_ratio** (property, lines 230-242)
  - Purpose: Report inferred `sigma_max / sigma_min`.
  - Inputs: None.
  - Logic: Computes `w_min = max(self.weights.min(), 1e-12)` and returns `sqrt(self._W_baseline_max / w_min)`.
  - Output: Float.
  - Side effects: None.
  - Edge cases handled: Very small min weight clamped at `1e-12`.
  - Invariants/guards: Formula depends on internal normalization choice.

- **DiagonalKernel.raw_weights** (property, lines 245-253)
  - Purpose: Return inverse-variance weights from stored sigma.
  - Inputs: None.
  - Logic: Computes `1.0 / (self.sigma ** 2)`.
  - Output: ndarray.
  - Side effects: Returns a newly computed array.
  - Edge cases handled: None.
  - Invariants/guards: Relies on stored sigma being valid.

- **DiagonalKernel.refresh_weights** (lines 255-278)
  - Purpose: Create a new diagonal kernel from updated sigma estimates.
  - Inputs: `sigma_per_factor`.
  - Logic: Converts to float64, validates shape against current weights, clips values to at least `1e-6`, and returns `DiagonalKernel(clipped)`.
  - Output: New `DiagonalKernel`.
  - Side effects: Does not mutate current object.
  - Edge cases handled: Shape mismatch, zero/negative sigma via clipping.
  - Invariants/guards: Does not reject NaN; `np.maximum(np.nan, 1e-6)` remains NaN.

### Invariants Enforced
- `ScoringKernel` requires methods named `compute_distance()` and `compute_gradient()` structurally (lines 23-61).
- `DiagonalKernel.__init__()` requires exactly one of `sigma` or `weights` (lines 156-159).
- Direct weights must be 1-D, finite, and strictly positive (lines 161-173).
- Sigma must be 1-D and strictly positive by comparison, but finiteness is not checked (lines 175-186).
- `refresh_weights()` requires the refreshed sigma vector shape to match current weights and clips values below `1e-6` (lines 272-278).

### Potential Issues
#### P1
- `DiagonalKernel.noise_ratio` is incorrect for kernels constructed from sigma when `_W_baseline_max` is not `1.0`; because `self.weights` is already normalized, the documented formula `sqrt(1 / weights_min)` is not the implemented `sqrt(_W_baseline_max / weights_min)` (lines 183-185, 230-242). This can materially overstate noise ratio and affect kernel-selection thresholds.

#### P2
- `DiagonalKernel` accepts `sigma` arrays containing NaN or Inf because the sigma path only checks `sigma <= 0`; NaN can propagate into weights, distances, gradients, and softmax, while Inf produces zero weights despite the doc saying all values must be `> 0` and meaningful (lines 175-186).
- `DiagonalKernel.refresh_weights()` preserves NaN values through clipping and returns a kernel that can contain NaN weights (lines 272-278).
- `L2Kernel` and `DiagonalKernel` do not enforce protocol shapes; unexpected broadcast-compatible inputs can return scalars or wrongly shaped arrays that only fail later, if at all (lines 76-90, 188-202).
- Direct `weights=` are stored unnormalized, while sigma-derived weights are normalized; `compute_gradient()` renormalizes both at call time, but `compute_distance()` uses the stored scale directly, so scoring scale depends on construction path (lines 161-186, 201-202, 226-227).

#### P3
- The module docstring references v6/v7 and `docs/gae_design_v10_6.md`, which may lag current package/documentation naming (lines 1-15).
- Protocol docstrings state non-negative distances, but `ProfileScorer` also supports a DOT kernel outside this protocol that uses negative dot products as distances; this is a cross-module terminology mismatch (lines 27-43 in this file, profile scorer lines 536-538).
- Concrete kernel methods rely on caller validation; that is acceptable for performance but should be stated explicitly in docstrings if intentional (lines 76-105, 188-227).

### Cross-Module Dependencies
- `ProfileScorer.__init__()` imports `L2Kernel` as the default pluggable scoring kernel.
- `ProfileScorer.score()` constructs `DiagonalKernel(weights=w_tilde)` during phase-2 variance learning.
- `ProfileScorer.kernel_weight_refresh()` assumes kernels may expose `refresh_weights()` with the same contract as `DiagonalKernel.refresh_weights()`.
- `ProfileScorer._compute_gradient()` assumes L2 and DIAGONAL kernels expose a gradient compatible with centroid updates.
- `KernelSelector` and calibration logic outside this file likely rely on `DiagonalKernel.noise_ratio`, `raw_weights`, and the `weights` field for diagonal-kernel selection and diagnostics.


# GAE Line-by-Line Review - Part 2

## gae/calibration.py (861 lines)

### Architecture
- This module provides domain calibration defaults, conservation-law helper formulas, factor masking, transfer-prior helpers, and enriched bootstrap prior construction.
- Key design decisions visible in code: most checks use `assert`; calibration profiles are passive dataclasses; conservation checks return rounded reporting values; bootstrap helpers initialize centroids at `0.5` and update with fixed `_ETA_CONFIRM = 0.05`.
- Several docstrings reference research/design formulas, but some code paths now contradict older doc notes, especially `derive_theta_min()` versus `compute_theta_min()`.

### Class-by-Class / Function-by-Function
- **CalibrationProfile** (lines 21-93)
  - Purpose: Dataclass for domain-configurable learning hyperparameters.
  - Inputs: Learning rate, penalty ratio, temperature, epsilon, discount strength, decay-class maps, factor-class maps, extensions.
  - Logic: Stores defaults and exposes `validate()`.
  - Output: Passive profile object.
  - Side effects: None.
  - Edge cases handled: Mutable fields use `default_factory`.
  - Invariants/guards: None at construction; invalid values are reported only if `validate()` is called.

- **CalibrationProfile.validate** (lines 65-93)
  - Purpose: Return warning strings for out-of-range top-level profile fields.
  - Inputs: Current instance fields.
  - Logic: Checks learning rate, penalty ratio, temperature, and discount strength against fixed expected ranges.
  - Output: `list[str]`.
  - Side effects: None.
  - Edge cases handled: None for decay maps or extensions.
  - Invariants/guards: No raises; no finite/type checks.

- **soc_calibration_profile** (lines 96-118)
  - Purpose: Factory for SOC defaults.
  - Inputs: None.
  - Logic: Returns profile with `learning_rate=0.02`, `penalty_ratio=20.0`, `temperature=0.25`, and factor decay classes.
  - Output: `CalibrationProfile`.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: Values are not auto-validated.

- **s2p_calibration_profile** (lines 121-135)
  - Purpose: Factory for S2P defaults.
  - Inputs: None.
  - Logic: Returns profile with `learning_rate=0.01`, `penalty_ratio=5.0`, `temperature=0.4`.
  - Output: `CalibrationProfile`.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: Values are not auto-validated.

- **ConservationCheck** (lines 143-150)
  - Purpose: NamedTuple result for conservation-law checks.
  - Inputs: `signal`, `theta_min`, `headroom`, `status`, `passed`.
  - Logic: Passive tuple container.
  - Output: Immutable tuple-like result.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: No validation that status is GREEN/AMBER/RED.

- **derive_theta_min** (lines 153-191)
  - Purpose: Legacy convergence-budget theta floor.
  - Inputs: `eta`, `n_half`, `t_max_days`.
  - Logic: Emits deprecation warning, asserts positive `t_max_days`, returns `eta * n_half**2 / t_max_days`.
  - Output: Float theta.
  - Side effects: Emits `DeprecationWarning` on every call.
  - Edge cases handled: Nonpositive `t_max_days` via assert.
  - Invariants/guards: No validation for `eta` or `n_half`; uses asserts, which can be disabled.
  - Actual vs docstring: The body notes that `23.53/(alpha*V)` was structurally incorrect, but the warning says to use exactly that deployment-aware formula.

- **compute_theta_min** (lines 194-222)
  - Purpose: Deployment-aware theta floor.
  - Inputs: `alpha`, `V`.
  - Logic: Rejects nonpositive alpha or verified volume and returns `23.53 / (alpha * V)`.
  - Output: Float theta.
  - Side effects: None.
  - Edge cases handled: Nonpositive alpha/V.
  - Invariants/guards: Positive alpha and V required; no finite checks.

- **check_conservation** (lines 225-273)
  - Purpose: Evaluate conservation signal against theta.
  - Inputs: `alpha`, `q`, `V`, `theta_min`.
  - Logic: Computes `signal = alpha*q*V`, `headroom = signal/theta_min` or inf if theta is nonpositive, maps to GREEN/AMBER/RED, returns rounded result.
  - Output: `ConservationCheck`.
  - Side effects: None.
  - Edge cases handled: Nonpositive theta produces infinite headroom.
  - Invariants/guards: No range checks for `alpha`, `q`, `V`, or `theta_min`.

- **compute_breach_window** (lines 276-318)
  - Purpose: Estimate detection window from signal variance and gap to theta.
  - Inputs: `signal_variance`, `signal_mean`, `theta_min`, `delta`.
  - Logic: Returns infinity when mean is at/below theta; otherwise computes Hoeffding-style window with `R=4*sqrt(variance)` and floors result at `1.0`.
  - Output: Float days.
  - Side effects: None.
  - Edge cases handled: `signal_mean <= theta_min`.
  - Invariants/guards: No validation for variance >= 0 or `delta` in `(0,1)`.

- **compute_optimal_tau** (lines 321-355)
  - Purpose: Gain-scheduled temperature from centroid covariance trace.
  - Inputs: 2-D covariance matrix and `(tau_min, tau_max)`.
  - Logic: Asserts matrix is 2-D, computes trace, clamps confidence to `[0,1]`, interpolates tau.
  - Output: Float tau.
  - Side effects: None.
  - Edge cases handled: Large traces clamp to `tau_min`; negative traces can clamp to `tau_max`.
  - Invariants/guards: 2-D covariance shape only; no square/symmetry/finiteness/range validation.
  - Actual vs docstring: Code maps higher covariance to lower tau. In this repo's softmax, lower tau makes decisions sharper, so the docstring claim that higher covariance gives "softer decisions" conflicts with actual scoring behavior.

- **compute_transfer_prior** (lines 358-391)
  - Purpose: Build empirical Bayes prior mean/std from calibrated centroid tensors.
  - Inputs: Dict of category name to centroid array.
  - Logic: Empty dict returns `(zeros(1), ones(1))`; otherwise stacks values and returns mean/std across categories.
  - Output: `(prior_mean, prior_std)`.
  - Side effects: None.
  - Edge cases handled: Empty dict.
  - Invariants/guards: Stacked result must be rank 3; no shape consistency check before `np.stack`.

- **compute_eta_override** (lines 394-461)
  - Purpose: Compute override learning rate from quality assumptions.
  - Inputs: Confirmation eta, mean quality, quality variance, safety margin, optional worst-case quality.
  - Logic: Worst-case path clamps quality to `[0,1]` and returns `eta_confirm * max(0, 2q-1)`; diagnostic path computes signal/noise ratio, floors at `0.005` if signal <= 0, otherwise rounds scaled eta.
  - Output: Float eta override.
  - Side effects: None.
  - Edge cases handled: Worst-case qualities outside `[0,1]`; nonpositive diagnostic signal.
  - Invariants/guards: No validation for eta sign, quality variance sign, safety margin, or finite inputs.

- **check_meta_conservation** (lines 464-515)
  - Purpose: Gate transfer-prior changes by per-component divergence.
  - Inputs: New prior, calibrated centroids, old prior, epsilon.
  - Logic: Asserts same shapes, computes absolute divergence, returns boolean pass and rounded details.
  - Output: `(bool, dict)`.
  - Side effects: None.
  - Edge cases handled: Shape mismatch.
  - Invariants/guards: Maximum divergence must be <= epsilon to pass.
  - Actual vs docstring: `calibrated_centroids` is documented as reserved and is unused by code.

- **_SOC_FACTOR_ORDER** (lines 523-526)
  - Purpose: Default SOC factor order for mask conversion.
  - Inputs: None.
  - Logic: Fixed list of six factor names.
  - Output: Used by `mask_to_array()`.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: Order is implicit API.

- **compute_factor_mask** (lines 529-562)
  - Purpose: Convert per-factor sigma into binary include/exclude decisions.
  - Inputs: Dict factor -> sigma, threshold.
  - Logic: Returns `True` when `sigma < threshold`, else `False`.
  - Output: Dict factor -> bool.
  - Side effects: None.
  - Edge cases handled: Missing factors are irrelevant because it only iterates provided keys.
  - Invariants/guards: No validation for threshold, sigma sign, or finite values.

- **_ETA_CONFIRM** (line 565)
  - Purpose: Fixed learning rate for bootstrap prior updates.
  - Inputs: None.
  - Logic: Constant `0.05`.
  - Output: Used by enriched bootstrap functions.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: None.

- **compute_enriched_bootstrap_prior** (lines 568-685)
  - Purpose: Build enriched initial centroid tensor from historical decisions and measured sigma.
  - Inputs: Historical `(category, action, f)` tuples, measured sigma, domain config with factor names, tensor dimensions, optional pre-enrichment sigma.
  - Logic: Validates positive dimensions and factor-name count with asserts, builds inverse-variance or delta-sigma weights, normalizes by mean, initializes `mu` to `0.5`, applies weighted residual updates per historical decision, clips updated centroids to `[0,1]`.
  - Output: `mu` shaped `(n_cat, n_act, n_factors)`.
  - Side effects: None outside local tensor.
  - Edge cases handled: Bad dimensions, factor vector shape, category/action index bounds through asserts.
  - Invariants/guards: `mu` shape exact; factor vectors exact shape; per-update clipping to `[0,1]`.

- **compute_dominant_axis** (lines 688-726)
  - Purpose: Score per-factor centroid separation.
  - Inputs: `mu` shaped `(n_cat, n_act, n_factors)`.
  - Logic: Asserts rank 3, flattens category/action positions, computes variance by factor, normalizes by max variance, returns zeros if max variance is not positive.
  - Output: ndarray `(n_factors,)` in `[0,1]` for finite valid inputs.
  - Side effects: None.
  - Edge cases handled: Uniform variance returns zeros.
  - Invariants/guards: Rank and derived shapes via assert; no finite checks.

- **compute_enriched_bootstrap_prior_geom** (lines 729-831)
  - Purpose: Geometry-aware enriched bootstrap prior.
  - Inputs: Historical decisions, measured sigma, sigma-before, current mu, domain config, tensor dimensions.
  - Logic: Validates dimensions and `mu_current` shape, computes enrichment ratio `(sigma_before/sigma_after)^2`, computes dominant-axis attenuation, clips weights to at least `1e-6`, normalizes by mean, initializes `mu` to `0.5`, applies weighted residual updates, clips centroids to `[0,1]`.
  - Output: `mu` shaped `(n_cat, n_act, n_factors)`.
  - Side effects: None outside local tensor.
  - Edge cases handled: Bad dimensions, bad `mu_current` shape, bad factor vector shape, index bounds through asserts.
  - Invariants/guards: Weight vector shapes exact; clipped weights nonzero for finite inputs.

- **mask_to_array** (lines 834-861)
  - Purpose: Convert factor-mask dict into ordered float ndarray.
  - Inputs: Dict mask and optional factor-name order.
  - Logic: Defaults to `_SOC_FACTOR_ORDER`, uses `mask.get(f, True)` so missing factors are included, returns 1.0/0.0 array.
  - Output: ndarray `(len(factor_names),)`.
  - Side effects: None.
  - Edge cases handled: Missing mask entries default to include.
  - Invariants/guards: Output shape assert.

### Invariants Enforced
- Calibration profile `validate()` warns for learning rate outside `[0.001, 0.5]`, penalty outside `[1.0, 100.0]`, temperature outside `[0.05, 2.0]`, and discount outside `[0.0, 1.0]`.
- `derive_theta_min()` asserts `t_max_days > 0`.
- `compute_theta_min()` raises `ValueError` if `alpha <= 0` or `V <= 0`.
- `compute_breach_window()` falls back to `inf` when `signal_mean <= theta_min` and floors finite windows at `1.0`.
- `compute_optimal_tau()` asserts covariance is 2-D and clamps derived confidence to `[0,1]`.
- `compute_transfer_prior()` returns `(zeros(1), ones(1))` on empty input and asserts stacked centroids are rank 3.
- `check_meta_conservation()` asserts prior shape equality and passes only if max divergence <= epsilon.
- `compute_enriched_bootstrap_prior()` and `_geom()` assert positive tensor dimensions, factor-name count, vector shapes, and category/action bounds; both clip centroid values to `[0,1]`.
- `compute_dominant_axis()` asserts `mu.ndim == 3` and derived vector shapes.
- `mask_to_array()` defaults missing mask keys to include and asserts output shape.

### Potential Issues
#### P1
- `compute_optimal_tau()` appears directionally inconsistent with the scorer: it maps higher centroid covariance to lower tau, but lower tau makes softmax sharper, not softer, under `ProfileScorer.score()` (lines 321-355).

#### P2
- `derive_theta_min()` contains a doc/warning contradiction: the docstring says the deployment-scaled formula was structurally incorrect, while the runtime warning says to use that formula (lines 171-186, 183-189).
- Many guards use `assert`, so shape/range checks disappear under optimized Python (`derive_theta_min`, bootstrap helpers, `compute_dominant_axis`, `mask_to_array`).
- Conservation helpers do not validate `q`, `theta_min`, or finite inputs; negative or NaN values can produce rounded but invalid statuses (lines 225-273).
- `compute_breach_window()` does not validate `signal_variance >= 0` or `delta` in `(0,1)`, so invalid inputs can produce NaN or nonsensical windows (lines 276-318).
- Bootstrap weighting divides by sigma powers without checking positive finite sigma; zero, negative, NaN, or missing sigma can produce crashes, infinities, or silent invalid weights (lines 634-656, 789-807).
- `compute_dominant_axis()` returns zeros when `max_var` is NaN because `NaN > 0` is false, masking invalid centroid tensors (lines 719-726).
- `compute_eta_override()` can return a positive floor of `0.005` even if `eta_confirm` is zero or negative in diagnostic mode (lines 456-461).
- `compute_transfer_prior()` empty fallback shape `(1,)` does not match the documented `(A,d)` return contract (lines 376-384).

#### P3
- `calibrated_centroids` is an unused parameter in `check_meta_conservation()`, documented as reserved for future checks (lines 464-515).
- Module and function docs continue to reference `docs/gae_design_v10_6.md` even when comments mention later fixes.
- `compute_factor_mask()` silently treats NaN sigma as excluded because `nan < threshold` is false, but does not explain this policy (lines 529-562).

### Cross-Module Dependencies
- `ProfileScorer` consumes `CalibrationProfile.temperature` and selected `extensions`, but does not consume every dataclass field.
- `ConservationMonitor` in `convergence.py` can propagate conservation status into `ProfileScorer.set_conservation_status()`.
- `mask_to_array()` output is compatible with `ProfileScorer.factor_mask`.
- `DiagonalKernel` and `KernelSelector` style logic likely depend on sigma/weight semantics from the calibration and bootstrap helpers.

## gae/convergence.py (1178 lines)

### Architecture
- This module contains convergence prediction formulas, onboarding-calendar helpers, convergence metrics for a `LearningState`, quality/conservation monitors, and EXP-G1 trace helpers.
- Key design decisions visible in code: empirical constants live at module scope; many predictive formulas are pure functions; monitors store mutable histories and sticky yellow flags; convergence metrics assume a specific external `LearningState` shape.
- The module mixes mathematical helpers with stateful monitoring classes, and validation is inconsistent across helpers.

### Class-by-Class / Function-by-Function
- **Module constants** (lines 35-43, 476-479, 617-633)
  - Purpose: Default eta, covariance trace, sigma margin, half-life, epsilon, convergence thresholds, rolling windows, and CUSUM settings.
  - Inputs: None.
  - Logic: Fixed empirical values.
  - Output: Used by formulas and monitors.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: Implicit public tuning knobs.

- **compute_n_half** (lines 48-63)
  - Purpose: Exact discrete scalar convergence half-life.
  - Inputs: `eta`.
  - Logic: Returns `log(2) / log(1/(1-eta))`.
  - Output: Float decisions.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: No eta range validation.

- **compute_per_factor_n_half** (lines 66-98)
  - Purpose: Per-factor half-life under max-normalized diagonal weights.
  - Inputs: `weights`, `eta`.
  - Logic: Converts weights to float64, returns empty array for empty input, rejects nonpositive weights, returns `(weights.max()/weights) * (log(2)/eta)`.
  - Output: ndarray shaped like weights.
  - Side effects: None.
  - Edge cases handled: Empty weights and nonpositive weights.
  - Invariants/guards: Weight positivity; no eta or finite validation.

- **compute_steady_state_mse** (lines 101-126)
  - Purpose: Predict steady-state centroid tracking MSE.
  - Inputs: `eta`, `tr_sigma_f`.
  - Logic: Asserts `0 < eta < 2`, returns `eta/(2-eta) * tr_sigma_f`.
  - Output: Float MSE.
  - Side effects: None.
  - Edge cases handled: Invalid eta via assert.
  - Invariants/guards: Eta bounds only.

- **compute_e_inf_per_component** (lines 129-159)
  - Purpose: Per-component steady-state error.
  - Inputs: `eta`, `tr_sigma_f`, required `d`.
  - Logic: Asserts `d` is provided and positive, computes MSE, returns `sqrt(mse/d)`.
  - Output: Float error.
  - Side effects: None.
  - Edge cases handled: Missing or nonpositive `d`.
  - Invariants/guards: Dimension positive.

- **predict_convergence_decisions** (lines 162-201)
  - Purpose: Predict decisions to reach epsilon from initial error.
  - Inputs: `e_0`, `epsilon`, `eta`, `tr_sigma_f`, `d`.
  - Logic: Computes noise floor; returns `-1` if epsilon is below floor, `0` if already converged, otherwise ceiling of log decay formula.
  - Output: Int decisions or `-1`.
  - Side effects: None.
  - Edge cases handled: Below noise floor and already converged.
  - Invariants/guards: Delegates `d` and eta checks indirectly; no direct positivity checks for `e_0` or `epsilon`.

- **predict_convergence_decisions_v2** (lines 204-256)
  - Purpose: Noise-aware convergence prediction.
  - Inputs: `e_0`, `epsilon`, `eta`, `tr_sigma_f`, `safety_factor`, `d`.
  - Logic: Computes noise floor, raises epsilon to `2.5*e_inf` if too close to floor, returns zero if already converged, otherwise ceiling of log formula multiplied by safety factor.
  - Output: Int decisions.
  - Side effects: None.
  - Edge cases handled: Near-noise-floor epsilon and already converged.
  - Invariants/guards: Delegated `d`/eta asserts only.

- **enrichment_multiplier** (lines 259-289)
  - Purpose: Return graph-level convergence acceleration factor.
  - Inputs: `graph_level`, unused `rho`.
  - Logic: Looks up fixed multipliers for G1-G4, defaults unknown levels to `1.0`.
  - Output: Float multiplier.
  - Side effects: None.
  - Edge cases handled: Unknown graph levels.
  - Invariants/guards: None; `rho` is ignored.

- **reconvergence_acceleration** (lines 292-314)
  - Purpose: Episode-based reconvergence multiplier.
  - Inputs: Episode integer.
  - Logic: Asserts nonnegative episode and returns `0.703 ** episode`.
  - Output: Float multiplier.
  - Side effects: None.
  - Edge cases handled: Negative episode via assert.
  - Invariants/guards: Episode nonnegative.

- **predict_category_convergence_weeks** (lines 317-392)
  - Purpose: Convert decision convergence prediction into calendar time for one category.
  - Inputs: Category name, alert volume, verification rate, action count, initial error, graph level, eta, covariance trace, dimension.
  - Logic: Computes verified decisions per action per day, predicts decisions, applies graph multiplier, divides by daily verified/action rate, returns summary dict.
  - Output: Dict.
  - Side effects: None.
  - Edge cases handled: `n_actions <= 0` and low/negative verified rate are floored in denominators; already converged returns minimal dict.
  - Invariants/guards: No validation for negative alert volume or verification rate.

- **generate_onboarding_calendar** (lines 395-469)
  - Purpose: Build convergence predictions for multiple categories.
  - Inputs: Category list, category weights, shared volume/rate/graph/eta/covariance/dimension.
  - Logic: Allocates alert volume by weight, calls category predictor, sorts valid predictions by weeks, returns list and summary fields.
  - Output: Dict.
  - Side effects: None.
  - Edge cases handled: Missing category weights default to `1/n_categories`; no valid predictions returns `None` summaries and `total_weeks=-1`.
  - Invariants/guards: None for negative weights or empty categories beyond denominator fallback.

- **get_convergence_metrics** (lines 486-555)
  - Purpose: Inspect a `LearningState` and return convergence diagnostics.
  - Inputs: Object with `W`, `history`, `decision_count`, `dimension_metadata`, `pending_validations`.
  - Logic: Computes current weight norm, empty-history default metrics, stability as std of recent `W_after` norms, accuracy as fraction of `outcome == +1`, convergence as stability below threshold and accuracy above threshold.
  - Output: Dict metrics.
  - Side effects: None.
  - Edge cases handled: Empty history and one-entry stability.
  - Invariants/guards: Assumes `LearningState` fields and history entry fields exist.

- **compute_normalized_var_q** (lines 562-595)
  - Purpose: Baseline-normalized quality variance.
  - Inputs: Rolling quality list and baseline quality.
  - Logic: Returns zero for fewer than two samples; otherwise `max(0, var(q)-q_baseline*(1-q_baseline))`.
  - Output: Float.
  - Side effects: None.
  - Edge cases handled: Short window.
  - Invariants/guards: No range or finite checks.

- **ConservationMonitor** (lines 636-802)
  - Purpose: Stateful two-layer conservation monitor.
  - Inputs: Optional scorer in constructor; status strings and quality observations in methods.
  - Logic: Layer 1 stores status and propagates to scorer; Layer 2 establishes a baseline after calibration period and updates EWMA/CUSUM for yellow warning.
  - Output: Mutable monitor state and properties.
  - Side effects: Mutates histories, baseline, CUSUM, warning flags, and optionally scorer conservation state.
  - Edge cases handled: Baseline not set; `_k` required before CUSUM.
  - Invariants/guards: `_update_cusum()` asserts `_k` exists.

- **ConservationMonitor.__init__** (lines 672-692)
  - Purpose: Initialize monitor state.
  - Inputs: Optional scorer.
  - Logic: Stores scorer, initializes quality history, baseline flags, CUSUM internals, h/lambda constants, and GREEN status.
  - Output: Monitor instance.
  - Side effects: None outside instance.
  - Edge cases handled: `scorer=None`.
  - Invariants/guards: None.

- **ConservationMonitor.update_conservation_signal** (lines 698-713)
  - Purpose: Update Layer 1 status and propagate it.
  - Inputs: Status string.
  - Logic: Stores status exactly as provided; if scorer exists, calls `set_conservation_status(status)`.
  - Output: None.
  - Side effects: Mutates monitor and possibly scorer.
  - Edge cases handled: No scorer.
  - Invariants/guards: No validation or normalization of status.

- **ConservationMonitor.conservation_status** (property, lines 716-718)
  - Purpose: Expose Layer 1 status.
  - Inputs: None.
  - Logic: Returns `_conservation_status`.
  - Output: String.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: None.

- **ConservationMonitor._update_cusum** (lines 724-764)
  - Purpose: Update EWMA and CUSUM for yellow warning.
  - Inputs: Quality score.
  - Logic: Initializes or updates EWMA, asserts `_k` is set, accumulates `max(0, cusum + (k - q_ewma))`, sets yellow warning/reason and resets CUSUM when above h.
  - Output: None.
  - Side effects: Mutates `_q_ewma`, `_cusum`, `yellow_warning`, `yellow_reason`.
  - Edge cases handled: First EWMA sample.
  - Invariants/guards: `_k` must be set.

- **ConservationMonitor.record_quality** (lines 766-792)
  - Purpose: Record a quality observation and run Layer 2 monitoring.
  - Inputs: `q`.
  - Logic: Appends quality, sets baseline from first 50 values when available, sets `_k = baseline - 0.05`, then updates CUSUM.
  - Output: None.
  - Side effects: Mutates history, baseline state, `_k`, EWMA/CUSUM/warnings.
  - Edge cases handled: Pre-baseline calls only append.
  - Invariants/guards: No q range validation.

- **ConservationMonitor.q_baseline** (property, lines 795-797)
  - Purpose: Expose baseline quality.
  - Inputs: None.
  - Logic: Returns `_q_baseline`, initially 0.0.
  - Output: Float.
  - Side effects: None.
  - Edge cases handled: Pre-baseline returns 0.0.
  - Invariants/guards: None.

- **ConservationMonitor.baseline_set** (property, lines 800-802)
  - Purpose: Expose whether baseline is ready.
  - Inputs: None.
  - Logic: Returns `_baseline_set`.
  - Output: Bool.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: None.

- **OLSMonitor** (lines 809-960)
  - Purpose: Stateful CUSUM monitor for OLS score degradation after plateau.
  - Inputs: Plateau window/threshold/k and OLS observations.
  - Logic: Accumulates OLS history, freezes baseline when rolling variance is below threshold, calibrates CUSUM threshold h, then raises yellow warning when post-plateau drops accumulate.
  - Output: Boolean alarm from `update()` plus mutable warning state.
  - Side effects: Mutates OLS history, baseline, h, CUSUM, warning state.
  - Edge cases handled: No alarm before plateau or before h is set.
  - Invariants/guards: Recent plateau slice shape assert.

- **OLSMonitor.__init__** (lines 852-872)
  - Purpose: Initialize OLS monitor.
  - Inputs: `plateau_window`, `plateau_threshold`, `k`.
  - Logic: Stores parameters, target ARL0, history/baseline/CUSUM/warning state.
  - Output: Monitor instance.
  - Side effects: None outside instance.
  - Edge cases handled: None.
  - Invariants/guards: No validation for positive window or k.

- **OLSMonitor._calibrate_h** (lines 874-904)
  - Purpose: Set dynamic CUSUM threshold from plateau variance and autocorrelation correction.
  - Inputs: Current history and parameters.
  - Logic: Computes recent sigma, floors sigma at `0.01`, applies rolling-window autocorrelation correction, computes `h = sigma_eff_sq * log(arl0)/(2*k)`, floors h at `0.5`.
  - Output: None.
  - Side effects: Mutates `_h`.
  - Edge cases handled: Low sigma and low h floor.
  - Invariants/guards: Assumes `plateau_window > 0` and `_k > 0`.

- **OLSMonitor.update** (lines 906-960)
  - Purpose: Ingest one OLS observation and possibly alarm.
  - Inputs: `ols_t`.
  - Logic: Appends value, detects plateau when recent variance is below threshold, freezes baseline and calibrates h, then post-plateau accumulates positive deviations from baseline and returns true on alarm.
  - Output: Bool alarm.
  - Side effects: Mutates history, baseline, h, CUSUM, warning state.
  - Edge cases handled: No alarm before plateau; no alarm if h missing.
  - Invariants/guards: Recent slice shape assert.

- **VarQMonitor** (lines 967-1069)
  - Purpose: Stateful baseline-normalized Var(q) detector with persistence.
  - Inputs: Threshold, rolling window, persistence, baseline window, and quality observations.
  - Logic: Sets baseline after initial observations, waits for full rolling window, computes normalized variance, counts consecutive threshold crossings, raises yellow warning after persistence crossings.
  - Output: Bool alarm from `update()`.
  - Side effects: Mutates history, baseline, crossing count, warning flag.
  - Edge cases handled: Pre-baseline and pre-window calls return false; crossing count resets under threshold and after alarm.
  - Invariants/guards: Rolling recent length assert.

- **VarQMonitor.__init__** (lines 996-1011)
  - Purpose: Initialize Var(q) monitor.
  - Inputs: Threshold, window, persistence, baseline window.
  - Logic: Stores parameters and initializes history/baseline/counter/warning.
  - Output: Monitor instance.
  - Side effects: None outside instance.
  - Edge cases handled: None.
  - Invariants/guards: No positive-window/persistence validation.

- **VarQMonitor.update** (lines 1013-1069)
  - Purpose: Ingest one quality score and possibly alarm.
  - Inputs: `q_t`.
  - Logic: Appends value, sets baseline after baseline window, waits for full rolling window, computes `compute_normalized_var_q()`, tracks consecutive threshold crossings, returns true when persistence reached.
  - Output: Bool alarm.
  - Side effects: Mutates history, baseline, crossings, warning flag.
  - Edge cases handled: Pre-baseline and pre-window.
  - Invariants/guards: Recent length assert.

- **centroid_distance_to_canonical** (lines 1075-1095)
  - Purpose: Frobenius distance between current and canonical centroids.
  - Inputs: `mu`, `canonical`.
  - Logic: Flattens both arrays and computes L2 norm of difference.
  - Output: Float distance.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: No shape or finite checks.

- **gamma_threshold** (lines 1098-1120)
  - Purpose: Compute epsilon-firm threshold for gamma theorem.
  - Inputs: `alpha_cat`, `delta_norm`, `theta`.
  - Logic: Returns `alpha_cat * delta_norm / (1 - alpha_cat)`.
  - Output: Float threshold.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: No bounds; `theta` is unused.

- **phase2_effective_threshold** (lines 1123-1143)
  - Purpose: Compute disrupted-category accuracy threshold for Phase 2 completion.
  - Inputs: `alpha_cat`, `theta`.
  - Logic: Returns `(theta - (1 - alpha_cat)) / alpha_cat`.
  - Output: Float threshold.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: No bounds; division by zero possible.

- **ConvergenceTrace** (lines 1151-1178)
  - Purpose: Dataclass for phase convergence history and summary.
  - Inputs: Centroid distances, rolling accuracy, n-half crossing, centroid plateau decision, gap flag, phase, optional epsilon.
  - Logic: Stores lists/metadata and exposes summary.
  - Output: Trace object.
  - Side effects: None.
  - Edge cases handled: Empty centroid-distance list in summary.
  - Invariants/guards: No validation of list lengths, phase values, or metric ranges.

- **ConvergenceTrace.summary** (lines 1165-1178)
  - Purpose: Return key trace statistics.
  - Inputs: Current trace fields.
  - Logic: Returns phase, count, n-half, plateau, gap, initial/final distances, epsilon.
  - Output: Dict.
  - Side effects: None.
  - Edge cases handled: Empty centroid distances return `None` for initial/final.
  - Invariants/guards: None.

### Invariants Enforced
- `compute_per_factor_n_half()` returns empty for empty weights and raises for any weight <= 0.
- `compute_steady_state_mse()` asserts `0 < eta < 2`.
- `compute_e_inf_per_component()` asserts `d` is explicit and positive.
- `reconvergence_acceleration()` asserts nonnegative episode.
- `predict_category_convergence_weeks()` floors action count at 1 and verified/action/day denominator at 0.01.
- `get_convergence_metrics()` returns explicit non-converged defaults for empty history.
- `compute_normalized_var_q()` returns `0.0` for windows shorter than 2 and clamps negative normalized variance to `0.0`.
- `ConservationMonitor._update_cusum()` asserts `_k` is initialized.
- `OLSMonitor.update()` asserts recent plateau array shape.
- `OLSMonitor._calibrate_h()` floors sigma at `0.01` and h at `0.5`.
- `VarQMonitor.update()` asserts recent rolling-window length.

### Potential Issues
#### P1
- None identified in `convergence.py` during this pass.

#### P2
- `compute_n_half()` lacks eta validation; eta <= 0 or eta >= 1 can produce division-by-zero, NaN, or invalid logarithms (lines 48-63).
- `compute_per_factor_n_half()` does not reject NaN weights or nonpositive eta, so invalid values can propagate silently or divide by zero (lines 66-98).
- Prediction helpers rely on asserts or indirect checks and do not validate positive `e_0`, positive `epsilon`, nonnegative covariance trace, or finite inputs (lines 101-256).
- `enrichment_multiplier()` accepts `rho` but ignores it, although the docstring presents rho as a parameter affecting validation context (lines 259-289).
- Calendar helpers silently floor bad rates/volumes through denominator `max(...)`, which can turn negative inputs into plausible-looking long timelines instead of errors (lines 317-469).
- `get_convergence_metrics()` is tightly coupled to an external `LearningState` shape and can raise `AttributeError` if the state lacks expected fields; no protocol/runtime validation is provided (lines 486-555).
- `ConservationMonitor.record_quality()` documents q in `[0,1]` but uses `CUSUM_H=5.0`, while comments say h=5.0 is calibrated for OLS scale and h=15.0 for raw q scale (lines 629-633, 672-692, 766-792).
- Monitor constructors do not validate positive windows, thresholds, persistence, or CUSUM `k`; zero/negative values can make alarms impossible, immediate, or cause division errors (lines 852-904, 996-1069).
- `centroid_distance_to_canonical()` flattens arrays without shape comparison, so different shapes with equal element counts are treated as compatible (lines 1075-1095).
- `gamma_threshold()` can divide by zero at `alpha_cat=1`, accepts invalid category fractions, and ignores the `theta` parameter (lines 1098-1120).
- `phase2_effective_threshold()` divides by `alpha_cat` with no zero/bounds guard and may return values outside `[0,1]` (lines 1123-1143).

#### P3
- Many docstrings reference `docs/gae_design_v10_6.md` and historical experiment notes, but not the current repo version/state.
- `ConvergenceTrace` imports dataclass aliases near the bottom of the file instead of using the top-level import area (lines 1146-1151).
- `ConvergenceTrace.summary()` reports `n_decisions` as `len(centroid_distances)`, which may differ from rolling-accuracy length without any consistency check (lines 1157-1178).

### Cross-Module Dependencies
- `get_convergence_metrics()` depends on an external `gae.learning.LearningState` shape under `TYPE_CHECKING`, including `W`, `history`, `dimension_metadata`, and `pending_validations`.
- `ConservationMonitor` optionally calls `ProfileScorer.set_conservation_status()`.
- `compute_normalized_var_q()` is used by `VarQMonitor`.
- Prediction constants and formulas inform calibration/conservation thresholds used elsewhere in the package documentation and monitoring flow.


# GAE Line-by-Line Review - Part 3

## gae/two_phase.py (94 lines)

### Architecture
- This module defines the category-level state and policy hooks used for two-phase learning: first mean convergence, then variance/DK learning.
- Key design decisions visible in code: state is tracked per category, not per `(category, action)` pair; phase names are plain strings; policy objects are small and stateless except for configured thresholds.
- The docstring says this module is intended to be behavior-preserving scaffolding, and the current implementation matches that: only `DecisionCountPolicy` can auto-freeze, while rolling-accuracy logic is intentionally inert.

### Class-by-Class / Function-by-Function
- **MEAN_CONVERGENCE / VARIANCE_LEARNING** (lines 20-21)
  - Purpose: Module-level phase string constants.
  - Inputs: None.
  - Logic: Fixed string labels.
  - Output: Used by `CategoryState` and `ProfileScorer`.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: No enum/type protection; callers can still assign arbitrary strings to `CategoryState.phase`.

- **CategoryState** (lines 25-51)
  - Purpose: Mutable per-category two-phase state.
  - Inputs: `phase`, `n_decisions`, optional `freeze_point`.
  - Logic: Stores the phase, a category decision counter, and the first freeze count.
  - Output: Dataclass instance.
  - Side effects: Methods mutate decision count and phase fields.
  - Edge cases handled: `freeze()` preserves an existing `freeze_point`.
  - Invariants/guards: Defaults start in `MEAN_CONVERGENCE` with zero decisions; no validation on construction.

- **CategoryState.record_decision** (lines 38-40)
  - Purpose: Count one verified category decision.
  - Inputs: None.
  - Logic: Increments `n_decisions` by one.
  - Output: None.
  - Side effects: Mutates `n_decisions`.
  - Edge cases handled: None.
  - Invariants/guards: No upper/lower bounds; assumes counter starts valid.

- **CategoryState.freeze** (lines 42-51)
  - Purpose: Transition the category to variance-learning phase.
  - Inputs: None.
  - Logic: Sets `phase` to `VARIANCE_LEARNING`; if `freeze_point` is unset, captures current `n_decisions`.
  - Output: None.
  - Side effects: Mutates `phase` and possibly `freeze_point`.
  - Edge cases handled: Repeated calls are idempotent for `freeze_point`.
  - Invariants/guards: Does not check current phase or decision count.

- **PhasePolicy** (lines 54-59)
  - Purpose: Structural protocol for freeze policies.
  - Inputs: Implementers must define `should_freeze(state)`.
  - Logic: Protocol stub only.
  - Output: None directly.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: No runtime checking.

- **PhasePolicy.should_freeze** (lines 57-59)
  - Purpose: Required policy method.
  - Inputs: `CategoryState`.
  - Logic: Protocol ellipsis.
  - Output: Boolean by contract.
  - Side effects: None by contract.
  - Edge cases handled: None.
  - Invariants/guards: Not enforced at runtime.

- **DecisionCountPolicy** (lines 62-70)
  - Purpose: Freeze after a category reaches a configured verified-decision count.
  - Inputs: Threshold `n`.
  - Logic: Stores `n` and compares it with `state.n_decisions`.
  - Output: Policy object and boolean decisions.
  - Side effects: None after construction.
  - Edge cases handled: None.
  - Invariants/guards: No validation that `n` is positive.

- **DecisionCountPolicy.__init__** (lines 65-66)
  - Purpose: Configure freeze threshold.
  - Inputs: `n`, default `200`.
  - Logic: Assigns `self.n = n`.
  - Output: None.
  - Side effects: Mutates instance config.
  - Edge cases handled: None.
  - Invariants/guards: No type/range validation.

- **DecisionCountPolicy.should_freeze** (lines 68-70)
  - Purpose: Decide whether phase transition should occur.
  - Inputs: `CategoryState`.
  - Logic: Returns `state.n_decisions >= self.n`.
  - Output: Bool.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: Assumes `state.n_decisions` and `self.n` are comparable numbers.

- **ManualPolicy** (lines 73-78)
  - Purpose: Inert policy for caller-driven freeze.
  - Inputs: `CategoryState`.
  - Logic: Always returns `False`.
  - Output: Bool.
  - Side effects: None.
  - Edge cases handled: All states produce no auto-freeze.
  - Invariants/guards: None.

- **ManualPolicy.should_freeze** (lines 76-78)
  - Purpose: Implement inert manual policy.
  - Inputs: `CategoryState`.
  - Logic: Ignores state and returns `False`.
  - Output: Bool.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: None.

- **RollingAccuracyDeltaPolicy** (lines 81-94)
  - Purpose: Placeholder for future rolling-accuracy freeze logic.
  - Inputs: `threshold_pp`.
  - Logic: Stores threshold but currently always returns `False`.
  - Output: Policy object and inert decisions.
  - Side effects: None after construction.
  - Edge cases handled: All states produce no auto-freeze.
  - Invariants/guards: No validation of threshold.
  - Actual vs docstring: The docstring states placeholder/inert behavior, and the implementation matches it.

- **RollingAccuracyDeltaPolicy.__init__** (lines 89-90)
  - Purpose: Store future threshold setting.
  - Inputs: `threshold_pp`, default `0.5`.
  - Logic: Assigns `self.threshold_pp`.
  - Output: None.
  - Side effects: Mutates instance config.
  - Edge cases handled: None.
  - Invariants/guards: No range validation.

- **RollingAccuracyDeltaPolicy.should_freeze** (lines 92-94)
  - Purpose: Placeholder freeze decision.
  - Inputs: `CategoryState`.
  - Logic: Ignores state and returns `False`.
  - Output: Bool.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: None.

### Invariants Enforced
- `CategoryState.freeze()` preserves the first `freeze_point` once set.
- `CategoryState.record_decision()` increments category-level count by exactly one.
- `DecisionCountPolicy.should_freeze()` uses `state.n_decisions >= n` as the only promotion criterion.
- `ManualPolicy` and `RollingAccuracyDeltaPolicy` never auto-freeze.

### Potential Issues
#### P1
- None identified in `two_phase.py` during this pass.

#### P2
- `DecisionCountPolicy` accepts `n <= 0`, causing immediate freeze for default-initialized states and bypassing the intended decision-count gate (lines 65-70).
- `CategoryState` accepts arbitrary `phase`, negative `n_decisions`, or inconsistent `freeze_point` values at construction because the dataclass has no validation (lines 25-37).

#### P3
- Phase values are plain strings rather than an enum or constrained type, so typos can propagate until downstream checks silently fail (lines 20-21, 34).
- `RollingAccuracyDeltaPolicy.threshold_pp` is stored but unused by design, which is clear from comments but still an API footgun if callers expect active rolling-accuracy behavior (lines 81-94).
- `PhasePolicy` is not marked `runtime_checkable`, so runtime protocol checks are not available even though policy injection is used elsewhere (lines 54-59).

### Cross-Module Dependencies
- `ProfileScorer` imports `CategoryState`, `DecisionCountPolicy`, `MEAN_CONVERGENCE`, and `VARIANCE_LEARNING` to drive two-phase learning and diagnostics.
- `ProfileScorer.update()` records category decisions, calls `phase_policy.should_freeze()`, and invokes `CategoryState.freeze()`.
- `ProfileScorer.score()` checks for `VARIANCE_LEARNING` to switch into DK-weighted phase-2 scoring.

## gae/batch_pipeline.py (294 lines)

### Architecture
- This module provides batch-composition policies, promotion-gate verdict structures, default promotion criteria, and a bounded in-memory history of batch evaluation attempts.
- Key design decisions visible in code: trigger policies maintain per-category last-trigger counters; the default promotion gate checks accuracy superiority, an absolute accuracy floor, and variance-ratio stability; `BatchHistory` stores evidence hashes rather than full weight arrays.
- Conservation is represented in `GateVerdict`, but the default gate hardcodes conservation as passing and appends a placeholder reason.

### Class-by-Class / Function-by-Function
- **BatchCompositionPolicy** (lines 12-26)
  - Purpose: Protocol for deciding when batch evaluation should run.
  - Inputs: Implementers provide `should_trigger()` and `record_trigger()`.
  - Logic: Protocol stubs only.
  - Output: None directly.
  - Side effects: None by protocol.
  - Edge cases handled: None.
  - Invariants/guards: No runtime checking.

- **BatchCompositionPolicy.should_trigger** (lines 15-22)
  - Purpose: Required trigger method.
  - Inputs: Accumulator, verified-decision count, category index.
  - Logic: Protocol ellipsis.
  - Output: Bool by contract.
  - Side effects: None by contract.
  - Edge cases handled: None.
  - Invariants/guards: Not enforced.

- **BatchCompositionPolicy.record_trigger** (lines 24-26)
  - Purpose: Required notification that an evaluation was attempted.
  - Inputs: Category index and verified-decision count.
  - Logic: Protocol ellipsis.
  - Output: None.
  - Side effects: Implementers usually mutate trigger history.
  - Edge cases handled: None.
  - Invariants/guards: Not enforced.

- **NoveltyThresholdPolicy** (lines 29-69)
  - Purpose: Trigger evaluation when accumulated novelty crosses a threshold after minimum decisions and cooldown.
  - Inputs: `threshold`, `min_decisions`, `cooldown`.
  - Logic: Validates constructor parameters, stores per-category last-trigger counts, checks min decisions, cooldown, and accumulator threshold.
  - Output: Policy object and boolean trigger decisions.
  - Side effects: `record_trigger()` mutates `_last_trigger`.
  - Edge cases handled: Negative threshold, `min_decisions < 1`, negative cooldown.
  - Invariants/guards: Threshold is nonnegative, minimum decisions at least one, cooldown nonnegative.

- **NoveltyThresholdPolicy.__init__** (lines 32-47)
  - Purpose: Configure novelty trigger policy.
  - Inputs: `threshold`, `min_decisions`, `cooldown`.
  - Logic: Raises `ValueError` for invalid bounds, converts values to float/int, initializes `_last_trigger`.
  - Output: None.
  - Side effects: Mutates instance config.
  - Edge cases handled: Invalid negative/low constructor parameters.
  - Invariants/guards: Bound checks only; no finite checks.

- **NoveltyThresholdPolicy.should_trigger** (lines 49-60)
  - Purpose: Decide if a category should evaluate a batch.
  - Inputs: Accumulator, verified-decision count, category index.
  - Logic: Rejects when below minimum decisions or cooldown not elapsed; otherwise returns `accumulator >= threshold`.
  - Output: Bool.
  - Side effects: None.
  - Edge cases handled: Pre-minimum and cooldown period.
  - Invariants/guards: Assumes caller separately calls `record_trigger()` after an attempted evaluation.

- **NoveltyThresholdPolicy.record_trigger** (lines 62-64)
  - Purpose: Record attempted evaluation count.
  - Inputs: Category index and verified-decision count.
  - Logic: Stores count in `_last_trigger[category_index]`.
  - Output: None.
  - Side effects: Mutates `_last_trigger`.
  - Edge cases handled: Overwrites previous category entry.
  - Invariants/guards: Converts count to int.

- **NoveltyThresholdPolicy._cooldown_elapsed** (lines 66-69)
  - Purpose: Internal cooldown check.
  - Inputs: Category index and verified-decision count.
  - Logic: Returns true for unseen category; otherwise checks count delta against cooldown.
  - Output: Bool.
  - Side effects: None.
  - Edge cases handled: First trigger per category.
  - Invariants/guards: None for decreasing decision counts.

- **FixedIntervalPolicy** (lines 72-102)
  - Purpose: Trigger evaluation on fixed verified-decision intervals.
  - Inputs: `interval`, optional `cooldown`.
  - Logic: Validates constructor settings, ignores accumulator, triggers when count is at an interval boundary and cooldown passes.
  - Output: Policy object and boolean trigger decisions.
  - Side effects: `record_trigger()` mutates `_last_trigger`.
  - Edge cases handled: Invalid interval/cooldown; first trigger per category.
  - Invariants/guards: Interval at least one, cooldown nonnegative.

- **FixedIntervalPolicy.__init__** (lines 75-82)
  - Purpose: Configure fixed-interval trigger.
  - Inputs: `interval`, `cooldown`.
  - Logic: Raises on invalid values, stores ints, initializes `_last_trigger`.
  - Output: None.
  - Side effects: Mutates instance config.
  - Edge cases handled: `interval < 1`, `cooldown < 0`.
  - Invariants/guards: No finite/type checks beyond comparisons and int casts.

- **FixedIntervalPolicy.should_trigger** (lines 84-98)
  - Purpose: Decide if current count is an interval boundary.
  - Inputs: Accumulator, verified-decision count, category index.
  - Logic: Deletes unused accumulator, rejects counts below interval, rejects non-boundaries, passes first boundary or cooldown-elapsed boundary.
  - Output: Bool.
  - Side effects: None.
  - Edge cases handled: Pre-interval, non-boundary, first trigger.
  - Invariants/guards: Assumes verified-decision counts are monotonic.

- **FixedIntervalPolicy.record_trigger** (lines 100-102)
  - Purpose: Store last interval trigger count.
  - Inputs: Category index and verified-decision count.
  - Logic: Stores int count by category.
  - Output: None.
  - Side effects: Mutates `_last_trigger`.
  - Edge cases handled: Overwrites previous category entry.
  - Invariants/guards: None.

- **GateVerdict** (lines 106-118)
  - Purpose: Dataclass result for promotion evaluation.
  - Inputs: Promotion bool, accuracy metrics, pass/fail flags, variance ratio, reason.
  - Logic: Passive evidence container.
  - Output: Verdict instance.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: No validation that flags and `promoted` are internally consistent.

- **PromotionGate** (lines 121-132)
  - Purpose: Protocol for candidate-weight promotion gates.
  - Inputs: Old/new accuracy, optional old weights, new weights.
  - Logic: Protocol stub only.
  - Output: `GateVerdict` by contract.
  - Side effects: None by contract.
  - Edge cases handled: None.
  - Invariants/guards: No runtime checking.

- **PromotionGate.evaluate** (lines 124-132)
  - Purpose: Required gate evaluation method.
  - Inputs: Accuracy values and weights.
  - Logic: Protocol ellipsis.
  - Output: `GateVerdict`.
  - Side effects: None by contract.
  - Edge cases handled: None.
  - Invariants/guards: Not enforced.

- **DefaultPromotionGate** (lines 135-213)
  - Purpose: Default deterministic promotion gate.
  - Inputs: Superiority margin, accuracy floor, max variance ratio.
  - Logic: Validates gate thresholds; evaluates accuracy delta, absolute floor, hardcoded conservation pass, and variance ratio.
  - Output: Gate object and `GateVerdict` results.
  - Side effects: None.
  - Edge cases handled: Invalid constructor bounds; missing old weights.
  - Invariants/guards: Margin nonnegative, floor in `[0,1]`, max variance ratio positive.

- **DefaultPromotionGate.__init__** (lines 138-156)
  - Purpose: Configure default promotion thresholds.
  - Inputs: `superiority_margin=0.05`, `floor=0.75`, `max_variance_ratio=2.0`.
  - Logic: Raises on invalid threshold values and stores floats.
  - Output: None.
  - Side effects: Mutates instance config.
  - Edge cases handled: Negative margin, floor outside `[0,1]`, nonpositive variance ratio.
  - Invariants/guards: No finite checks for NaN values.

- **DefaultPromotionGate.evaluate** (lines 158-200)
  - Purpose: Evaluate whether candidate weights should be promoted.
  - Inputs: Old accuracy, new accuracy, old weights or None, new weights.
  - Logic: Computes `new-old` superiority delta, floor pass, hardcoded conservation pass, variance ratio, failure reason list, and returns a verdict.
  - Output: `GateVerdict`.
  - Side effects: None.
  - Edge cases handled: `old_weights=None` through `_compute_var_ratio()`.
  - Invariants/guards: No validation for accuracy ranges, shape compatibility, or finite weight values.
  - Actual vs docstring: The class doc says it checks accuracy and variance stability; `GateVerdict` includes conservation, but this implementation does not actually evaluate conservation.

- **DefaultPromotionGate._compute_var_ratio** (lines 202-213)
  - Purpose: Compute new/old weight variance ratio.
  - Inputs: Optional old weights and required new weights.
  - Logic: Returns `1.0` when old weights are missing; computes variances; if old variance is zero, returns `1.0` for zero new variance or `inf` otherwise; else returns `new_var / old_var`.
  - Output: Float ratio.
  - Side effects: None.
  - Edge cases handled: Missing old weights and zero old variance.
  - Invariants/guards: No shape/dtype/finiteness checks.

- **BatchRecord** (lines 217-228)
  - Purpose: Dataclass evidence record for one batch attempt.
  - Inputs: Category, timestamp, accuracies, promotion flag, reason, old/new weight hashes, verdict.
  - Logic: Passive container.
  - Output: Record instance.
  - Side effects: None.
  - Edge cases handled: Optional old hash.
  - Invariants/guards: No validation of hash/verdict consistency.

- **BatchHistory** (lines 232-294)
  - Purpose: Bounded in-memory history of batch evaluation attempts.
  - Inputs: `max_records` and optional existing records.
  - Logic: Validates bound, appends `BatchRecord`s, trims to latest records, filters records, counts attempts/promotions, hashes weight bytes.
  - Output: History object and record lists/counts.
  - Side effects: Mutates `_records`.
  - Edge cases handled: Invalid max record count; no old weights.
  - Invariants/guards: History length is trimmed to at most `max_records`.

- **BatchHistory.__post_init__** (lines 238-240)
  - Purpose: Validate bounded history size.
  - Inputs: Current dataclass fields.
  - Logic: Raises if `max_records < 1`.
  - Output: None.
  - Side effects: None.
  - Edge cases handled: Invalid bound.
  - Invariants/guards: `max_records >= 1`.

- **BatchHistory.record** (lines 242-267)
  - Purpose: Store a batch evaluation attempt.
  - Inputs: Category, old/new accuracy, optional old weights, new weights, verdict, optional timestamp.
  - Logic: Builds `BatchRecord`, hashes weights, appends it, trims to last `max_records`, returns the record.
  - Output: `BatchRecord`.
  - Side effects: Mutates `_records`.
  - Edge cases handled: Missing timestamp uses `datetime.utcnow()`; missing old weights produce `None` old hash.
  - Invariants/guards: Stored records are bounded by count.

- **BatchHistory.get_records** (lines 269-280)
  - Purpose: Return filtered stored records.
  - Inputs: Optional category index and promoted-only flag.
  - Logic: Filters current records by category and promotion status and returns a new list.
  - Output: `list[BatchRecord]`.
  - Side effects: None.
  - Edge cases handled: No filters.
  - Invariants/guards: Returns a new list but record objects are the original mutable dataclass instances.

- **BatchHistory.total_promotions** (lines 282-284)
  - Purpose: Count promoted stored records.
  - Inputs: None.
  - Logic: Sums records with `promoted=True`.
  - Output: Int count.
  - Side effects: None.
  - Edge cases handled: Empty history returns zero.
  - Invariants/guards: None.

- **BatchHistory.total_attempts** (lines 286-288)
  - Purpose: Count stored records.
  - Inputs: None.
  - Logic: Returns `len(_records)`.
  - Output: Int count.
  - Side effects: None.
  - Edge cases handled: Empty history returns zero.
  - Invariants/guards: Count is bounded by `max_records`, not lifetime attempts.

- **BatchHistory._hash** (lines 290-294)
  - Purpose: Produce compact hash of an ndarray's byte payload.
  - Inputs: `arr`.
  - Logic: Imports hashlib, hashes `arr.tobytes()` with SHA-256, returns first 16 hex chars.
  - Output: String hash prefix.
  - Side effects: None.
  - Edge cases handled: None.
  - Invariants/guards: Does not include dtype, shape, or byte order metadata.

### Invariants Enforced
- `NoveltyThresholdPolicy` requires nonnegative threshold, `min_decisions >= 1`, and nonnegative cooldown.
- `FixedIntervalPolicy` requires `interval >= 1` and nonnegative cooldown.
- Both trigger policies record last-trigger count per category and use it for cooldown checks.
- `DefaultPromotionGate` requires nonnegative superiority margin, floor in `[0,1]`, and positive max variance ratio.
- Promotion requires all pass flags to be true: superiority, floor, conservation, and variance.
- `_compute_var_ratio()` treats missing old weights as ratio `1.0` and zero old variance as either `1.0` or `inf`.
- `BatchHistory` requires `max_records >= 1` and trims stored records to the configured bound.

### Potential Issues
#### P1
- None identified in `batch_pipeline.py` during this pass.

#### P2
- `DefaultPromotionGate.evaluate()` hardcodes `conservation_pass=True`, so a verdict can claim conservation passed without evaluating any conservation signal (lines 166-181).
- Trigger policies with `cooldown=0` can return true repeatedly at the same decision count if the caller invokes `should_trigger()` again after `record_trigger()` without advancing decisions (lines 58-69, 96-102).
- Constructor checks do not reject NaN thresholds/margins/floors because comparisons with NaN are false; NaN settings can silently disable or distort trigger/gate behavior (lines 38-46, 144-156).
- Promotion evaluation does not validate accuracy values are finite or in `[0,1]`, so invalid metrics can produce misleading pass/fail decisions (lines 158-200).
- `_compute_var_ratio()` does not validate old/new weight shape compatibility or finiteness; shape changes and NaN/Inf weights are reduced to a variance ratio without explanation (lines 202-213).
- `BatchHistory._hash()` hashes only raw bytes; dtype/shape metadata is not included, so semantically different arrays with identical byte payloads produce the same evidence hash (lines 290-294).
- `BatchHistory.total_attempts()` reports only retained records, not lifetime attempts, which can be misleading after trimming (lines 264-288).

#### P3
- Protocols are not marked `runtime_checkable`, limiting runtime validation of injected policy/gate objects (lines 12-26, 121-132).
- `BatchHistory.record()` uses naive `datetime.utcnow()` timestamps rather than timezone-aware timestamps (lines 242-255).
- `get_records()` returns a new list but the contained `BatchRecord` objects are mutable dataclasses, so callers can mutate history evidence indirectly (lines 269-280).
- `reason` always includes `"placeholder_always_pass"`, even on legitimate failures, which weakens audit clarity (lines 183-188).
- No serialization helper is provided for `BatchHistory` or records, despite the module creating evidence structures (lines 216-294).

### Cross-Module Dependencies
- No imports from other `gae` modules are present in `batch_pipeline.py`.
- Callers are expected to supply old/new weights and old/new accuracy from scorer or evaluation code outside this module.
- `BatchHistory` evidence hashes assume NumPy arrays and preserve only byte-derived identifiers, so downstream audit code must retain shape/dtype elsewhere if needed.

