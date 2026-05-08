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
