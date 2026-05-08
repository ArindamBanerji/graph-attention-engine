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
