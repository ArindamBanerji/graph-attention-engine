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
