# GAE Test Categories

Tests are organized by severity. P0 failures block all merges.
P1 failures block scoring-path merges. P2/P3 are informational.

Gate: `python -m pytest tests/ -q` — all tests must pass before merge.

---

## P0 — Must Never Regress

Failures here = production incident. Run in every CI check.

### Numeric Stability
- NaN/Inf guards: output of `score()` never silently contains NaN/Inf when input is valid
- μ clip: `ProfileScorer.mu` always in `[0.0, 1.0]` after any sequence of updates
- τ=0.1 valid probabilities: softmax at default τ produces well-formed distribution
- DiagonalKernel zero-weights: `DiagonalKernel(sigma)` rejects σ ≤ 0 with `ValueError`
- Empty tensor: `ProfileScorer` rejects `mu.ndim != 3` with `AssertionError`
- MAX_ETA_DELTA cap: no centroid coordinate moves more than ±0.005 per update step

### Tensor Shape
- `score()` input `f.shape == (n_factors,)`, output `.probabilities.shape == (n_actions,)`
- `update()` returns `CentroidUpdate` with correct `category_index`, `action_index`
- `centroids` property returns shape `(n_categories, n_actions, n_factors)`
- SOC shape (6, 4, 6) and S2P shape (5, 5, 8) both construct and score without error
- `mu.shape[1] == len(actions)` enforced at construction

### API Contract
- Every Tier 1 function exists and is importable from `gae`
- `score()` signature: `(f: ndarray, category_index: int) → ScoringResult`
- `update()` signature: `(f, category_index, action_index, correct, ...) → CentroidUpdate`
- `L2Kernel`, `DiagonalKernel` importable from `gae.kernels`
- `derive_theta_min`, `check_conservation` importable from `gae.calibration`

---

## P1 — Behavioral (run on scoring-path PRs)

### Scoring
- Deterministic: `score(f, c)` called twice → identical probabilities
- Probabilities sum to 1.0 (within 1e-9)
- `DiagonalKernel` with unequal σ produces different scores than `L2Kernel`
- `confidence` equals max of `.probabilities`
- `action_index` equals argmax of `.probabilities`

### Learning
- `update(correct=True)` pulls `μ[c,a,:]` closer to `f` (L2 distance decreases)
- `update(correct=False)` pushes `μ[c,a,:]` farther from `f` (L2 distance increases)
- `eta_override < eta_neg` on the override path when `eta_override` is set
- μ stays in `[0,1]` after adversarial inputs (`f >> 1` or `f << 0`)
- Learning rate decays with `count`: `η_eff = η / (1 + decay × count)`

### Convergence
- `gamma_threshold(alpha, delta)` ≈ `alpha × delta / (1 - alpha)` (phase-2 formula)
- `derive_theta_min` matches formula `η × N_half² / T_max`
- `check_conservation` GREEN/AMBER/RED thresholds are monotone in `alpha × q × V`

### Conservation
- `check_conservation` returns GREEN when `signal ≥ 2 × theta_min`
- `check_conservation` returns AMBER when `theta_min ≤ signal < 2 × theta_min`
- `check_conservation` returns RED when `signal < theta_min`
- `ConservationMonitor.auto_pause_on_amber=True` blocks updates when AMBER/RED

---

## P2 — Integration (run on full-pipeline PRs)

### Full Pipeline
- `DomainConfig → CalibrationProfile → ProfileScorer → score → update` cycle: no errors
- 20-step learning loop with SOC (6,4,6): accuracy measurable, μ stays in [0,1]
- 20-step learning loop with S2P (5,5,8): accuracy measurable, μ stays in [0,1]

### Cross-Domain
- SOC (6 categories, 4 actions, 6 factors) and S2P (5 categories, 5 actions, 8 factors)
  use the same `ProfileScorer` class — no domain-specific subclass needed
- `soc_calibration_profile()` and `s2p_calibration_profile()` both construct
  `CalibrationProfile` with valid parameters (profile.validate() returns empty list)

### Persistence
- `save_state(scorer, path)` + `load_state(path)` roundtrip preserves μ, counts, settings
- Loaded scorer produces identical `score()` outputs to the original

---

## P3 — Research (informational, non-blocking)

### Experiment Reproduction
- `OracleSeparationExperiment` with EXP-C1 parameters reproduces ≥36.89pp separation
- `FactorVectorSampler` generates vectors in `[0,1]^d` with correct inter-class separation
- `CanonicalCentroid` computes expected centroid location within tolerance

### Synthetic Data
- `GammaResult.gamma` < 1 when `eps=0.05` (oracle separable)
- `GammaResult.gamma` > 1 when `eps=0.20` (oracle not separable)
- `ConvergenceTrace` records monotone distance reduction under correct learning
