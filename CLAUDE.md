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
