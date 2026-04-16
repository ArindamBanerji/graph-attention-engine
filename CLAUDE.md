# ⚠️ GROUNDING CONTRACT (non-negotiable)

**These rules apply to every AI coding agent working in this repo.**

1. **Docs are aspirational until proven in code.** Check actual source files.
2. **Cite file + line for every behavioral claim.**
3. **Code and tests beat docs.** Discrepancy = DRIFT, report and stop.
4. **Check downstream consumers before changing interfaces.** This repo is
   consumed by gen-ai-roi-demo-v4-v50 and s2p-copilot.
5. **Verify after every change:** `python -m pytest tests/ -v` (856 tests)
6. **Authoritative design doc:** `docs/gae_design_v8_3.md` (not v5.md — deprecated).

---

## How to Think (read first, every session)

1. State assumptions before coding. Never silently pick a field name.
2. Minimum code that solves the problem.
3. Surgical changes only.
4. Verify after every step — "this should work" is not verification.
5. Before adding a constant: grep to check if it exists under a different name.

---

## What This Repo Is

Graph Attention Engine (GAE) — a **pure numpy scoring engine**. Apache 2.0.

- No database dependencies. No network dependencies. No event loop. Permanent.
- No asyncio imports in gae/ except FactorComputer protocol method
  (gae/factors.py:14-19,24-28,47).
- All inputs/outputs are numpy arrays and Python dicts.

### Consumed by
- gen-ai-roi-demo-v4-v50/backend/app/domains/soc/ (SOC scoring)
- s2p-copilot/backend/app/domains/s2p/ (S2P scoring)
- copilot-sdk (protocol definitions)

---

## Public API — Tier 1 Stable (never break without major version bump)

**Verified from code, not docs:**

- `ProfileScorer.score(f, category_index) → ScoringResult`
  Fields: action_index, action_name, probabilities, distances, confidence
  Source: gae/profile_scorer.py:77-101,286-347

- `ProfileScorer.update(...) → CentroidUpdate`
  Source: gae/profile_scorer.py:33-52,576-746

- `ProfileScorer.centroids` property → ndarray shape (n_cat, n_act, n_factors)
  Source: gae/profile_scorer.py:492-499
  Clipped to [0,1] after every update: gae/profile_scorer.py:730-731

- `DiagonalKernel.compute_distance()`, `DiagonalKernel.compute_gradient()`
  Gradient formula: W/W.max() * (f - μ) — no other form
  Source: gae/kernels.py:108-194

- `L2Kernel.compute_distance()`
  Source: gae/kernels.py:64-105

- `ConservationMonitor` — α·q·V ≥ θ_min enforcement

- `save_state`, `load_state` — atomic 1-D persistence (weights, step, converged)
  Source: gae/store.py:30-118,125-200

**ProfileScorer is the preferred scorer.** However, deprecated Tier-2 exports
remain public until removed in a versioned change (see below).

## Deprecated but Still Public (do not delete without major version)

Exported from gae/__init__.py:45-46,71-72,176-177,187,270-273:
- `score_entity`, `score_alert`, `score_with_profile`
- `ProfileScoringResult` (backward-compatible alias)

These are superseded by ProfileScorer but still in the public surface.
Downstream users may depend on them. Document removal in a versioned plan.

## Public API — Tier 2 Evolving

- KernelSelector (rolling 100-window, noise_ratio > 1.5 rule)
- CovarianceEstimator (per-factor σ, half_life=300)
- GainScheduler (Block 9.6, not yet validated)

---

## Architectural Invariants (from math_synopsis_v13)

These are mathematical constraints. Changing them changes the scoring math.

- μ ∈ [0,1]^d — centroids clipped after every update
- τ = 0.1, fixed (ECE=0.036 at τ=0.1)
- η_confirm = 0.05, η_override = 0.01 — asymmetric learning rates
- η_neg = 0.05 canonical — η_neg=1.0 FORBIDDEN (ECE=0.49)
- DiagonalKernel gradient = W/W.max() * (f - μ) — no other form
- h = 5.0 for OLS-scale CUSUM
- Learning flow: LearningState.update → ProfileScorer.update
  Source: gae/learning.py:400-426

### Tensor Shapes

SOC: (6 categories, 4 actions, 6 factors) = 144 centroid values
S2P: (5 categories, 5 actions, 8 factors) = 200 centroid values

Setup uses raw mu tensors directly — no DomainConfig object.
Source: tests/test_soc_integration.py:45-56; tests/test_api_contract.py:241-260

refer_to_analyst is NOT a scorable action (SOC has A=4, not A=5).

---

## Rules

- Do NOT use git directly. User handles all git operations.
- Do NOT add database or network dependencies. Pure numpy. Permanent.
- Do NOT add asyncio imports or async functions (except FactorComputer protocol).
- Do NOT change Tier 1 function signatures without grepping consumers.
- Every gae/ function needs a blog-equation docstring.
- Every matrix op needs shape assertions.
- If you change scoring math: verify against math_synopsis_v13.

## Forbidden Changes (from API_CONTRACT.md)

- Never change ProfileScorer.score() return type (ScoringResult)
- Never change ProfileScorer.update() return type (CentroidUpdate)
- Never change centroid tensor shape convention (n_cat, n_act, n_factors)
- Never change η_confirm/η_override defaults without major version bump
- Never remove a Tier 1 function
- Never add database or network dependencies

## After Any Change

1. `python -m pytest tests/ -v` (856 tests must pass)
2. If you changed a Tier 1 function signature: grep for it in
   gen-ai-roi-demo-v4-v50 AND s2p-copilot
3. If you changed scoring math: verify against math_synopsis_v13
