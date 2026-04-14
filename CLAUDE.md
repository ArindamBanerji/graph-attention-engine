## How to Think (read first, every session)

### 1. State Assumptions Before Coding
- Before implementing, state your assumptions explicitly
- If multiple interpretations exist, present them — don't pick silently
- NEVER silently pick a property name, field type, or API path — state it

Example of WRONG: "I'll use {id: $val} in the Cypher query"
Example of CORRECT: "Assuming property 'id'. Verifying: grep shows 'alert_id'. Using that."

### 2. Minimum Code That Solves the Problem
- No features beyond what was asked. No abstractions for single-use code.
- If 200 lines could be 50, rewrite it.

### 3. Surgical Changes
- Touch only what you must. Don't "improve" adjacent code.
- Every changed line traces directly to the request.

### 4. Goal-Driven Execution
- Before starting: Step → verify: [specific check] for each step.
- "This should work" is never verification. Show the output.

### 5. Dual Representation Rule
- Before adding any constant/tensor/property: check if it exists under a different name.
- Grep: get_actions(), SCORER_ACTIONS, SOC_PROFILE_CENTROIDS, alert_id, decision_id

# CLAUDE.md — Graph Attention Engine (GAE)
# License: Apache 2.0 — this is a PUBLIC library

## This repo is a pure numpy scoring engine
- No database dependencies
- No network dependencies
- No event loop dependencies
- All inputs/outputs are numpy arrays and Python dicts

## Consumed by
- gen-ai-roi-demo-v4-v50/backend/app/domains/soc/ (scoring logic)
- s2p-copilot/backend/app/domains/s2p/ (scoring logic)
- copilot-sdk (protocol definitions)

## Public API — Tier 1 Stable (never break without major version bump)
- ProfileScorer.score(f, category) → ActionDistribution
- ProfileScorer.update(f, category, action, correct, **kwargs) → None
- DiagonalKernel(weights) → Kernel
- L2Kernel() → Kernel
- DomainConfig(categories, actions, factors, ...) → Config
- get_profile_centroids() → ndarray shape (n_cat, n_act, d)
- ConservationMonitor — α·q·V ≥ θ_min enforcement

## Public API — Tier 2 Evolving
- KernelSelector (rolling 100-window, noise_ratio > 1.5 rule)
- CovarianceEstimator (collects per-factor σ, half_life=300)
- GainScheduler (Block 9.6, not yet validated)

## Public API — Tier 3 Experimental
- gae/experiments/* (research infrastructure)
- gae/synthetic.py (OracleSeparationExperiment, FactorVectorSampler)
- gae/convergence.py (centroid_distance_to_canonical, gamma_threshold)

## Architectural invariants (from math_synopsis_v13)
- μ ∈ [0,1]^d — clip after every update
- τ = 0.1, fixed — never change (ECE=0.036 at τ=0.1)
- η_confirm = 0.05, η_override = 0.01 — asymmetric, P0 fix
- η_neg = 0.05 canonical — η_neg=1.0 FORBIDDEN (ECE=0.49)
- DiagonalKernel gradient = W/W.max() * (f - μ) — no other form
- h = 5.0 for OLS-scale CUSUM — do not change
- SOC tensor: (6, 4, 6) = 144 values. S2P tensor: (5, 5, 8) = 200 values
- Loop 4 (synthesis σ) never updates μ — architecturally enforced

## After any change
1. python -m pytest tests/ -v (536 tests must pass)
2. If you changed a Tier 1 function signature: grep for it in
   gen-ai-roi-demo-v4-v50 AND s2p-copilot
3. If you changed scoring math: verify against math_synopsis_v13

## Forbidden changes (from API_CONTRACT.md)
- Never change ProfileScorer.score() return type
- Never change centroid tensor shape convention (n_cat, n_act, d)
- Never change η_confirm/η_override defaults without major version bump
- Never remove a Tier 1 function
- Never add database or network dependencies

### Simplicity Invariant
- Pure numpy. No database. No network. Permanent.
- ProfileScorer is THE scorer. No alternative scoring paths.
- DomainConfig defines tensor shape. refer_to_analyst NOT a scorable action (A=4 SOC).

### No Silent Failure on Displayed Metrics
- If a try/except computes a NUMBER shown in the UI: the except block
  must set a flag (estimated=True, source="fallback") — never bare pass
- If a try/except computes OPTIONAL enrichment: bare pass is acceptable
- NEVER hardcode a number that looks like a computed metric (0.89, 23, 127)
  without a comment explaining why it's a constant and not computed
- The test: if the graph is empty, does the UI show zeros or plausible-looking
  fake numbers? If fake numbers: it's a mockup, not a fallback.

### AGE Is Not Neo4j — Three Critical Differences

1. **SET n = {props} WIPES all other properties**
   - NEVER: `SET d = {category: 'x'}` — destroys every other property
   - ALWAYS: `SET d.category = 'x'` — preserves all other properties
   - SAFE for bulk: `SET d += {a: 1, b: 2}` — merges, preserves existing
   - AGEClient rejects the destructive form with ValueError

2. **Concurrent writes to the same node fail**
   - "Entity failed to be updated: 3" = PostgreSQL row lock conflict
   - AGEClient retries with jitter (3 attempts, 100-250ms backoff)
   - Avoid concurrent writes to the same node when possible

3. **Decision nodes must be created atomically with their edge**
   - ALWAYS: `MATCH (a:Alert) CREATE (d:Decision {...})-[:DECIDED_ON]->(a)`
   - NEVER: CREATE Decision as one query, then edge as a second
   - If MATCH finds no Alert, no Decision is created (proven atomic)
