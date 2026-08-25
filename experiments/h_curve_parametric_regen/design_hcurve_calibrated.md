# H-CURVE calibrated re-convergence — design v1

## Recalibration

The prior H-CURVE geometry put all 24 action centroids within a small random displacement of
`0.5`, so fixed action labels were measured at chance. The recalibrated apparatus widens only the
GT action separation. It uses four balanced six-dimensional sign patterns:

```text
[-1,-1,-1,+1,+1,+1]
[-1,+1,+1,-1,-1,+1]
[+1,-1,+1,-1,+1,-1]
[+1,+1,-1,+1,-1,-1]
```

`GT1[c,a] = 0.5 + 0.28 * pattern[a] + firm_displacement[c,a]`, where the firm displacement is
the same `99 + run_seed` direction construction and ε_firm values as H-CURVE. The resulting
profiles remain in `[0,1]`; Phase 2 applies the original `||Δ||=0.25` shift to categories `[0,1]`.
Vectors remain `clip(GT_phase[c,a] + N(0,0.08),0,1)` and labels remain the fixed round-robin
action index `a`. No scorer state is used to construct GT or vectors.

A calibration-only Arm-A run is mandatory before Arm A/B re-convergence cells. The gate is passed
only when the final 100-decision Phase-1 accuracy is both greater than `0.50` and greater than the
unchanged `theta_min=0.467`, for every seed and ε cell.

## Two-arm test

After the gate, the same vectors, GT, seeds `[42,123,777]`, ε `{0.20,0.35}`, round-robin budgets
`720/1200`, target-per-phase distance, and scale-free fit are used in both arms:

- Arm A: real `ProfileScorer`, `eta_override=0.01`, `auto_pause_on_amber=True`; conservation status
  is set from `alpha*q*V >= theta_min` before every update.
- Arm B: clean-room `mu[c,a] += 0.05*(f-mu[c,a])`, without conservation gating.

The primary rate is `gamma_rate=k_phase2/k_phase1`, with Phase 1 fit on the full tensor and Phase 2
fit on the disrupted subspace. The fit uses the existing ten-decision block median pairwise log-slope
estimator and no absolute-distance gate.

## Required instrumentation and reading rules

Every decision persists overall rolling accuracy, pressure, θ_min, ENGAGED/PAUSED status, learning
permission, correctness, category accuracy, and centroid snapshots. Each cell persists GT1, GT2, μ0,
μfinal, both vector streams, rate records, gamma, and invariant assertions. The manifest hashes all
raw files and the verification PNG/PDF.

- **ARTIFACT CONFIRMED:** Arm A γ>1 and conservation remains ENGAGED (or recovers without blocking).
- **REAL GAP:** Arm A γ<1, Arm B recovers, and Arm A is PAUSED during recovery.
- **OTHER:** Arm A γ<1 while never PAUSED; investigate a non-conservation cause.

The fixed θ_min, labels, learning rules, GT construction boundary, and scale-free γ are not relaxed.
