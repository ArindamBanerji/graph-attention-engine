# R-HCURVE: conservation-compatible selective re-convergence

## Question and frozen apparatus

This experiment tests whether an experiment-local, disruption-aware policy can recover the H-CURVE
re-convergence result in the production `ProfileScorer` regime. It does not modify production code,
GT, oracle labels, Arm B, or the base update equation. The primary cells use the already-persisted
H-CURVE v4 vectors and GT for ε={0.20,0.35}, seeds 42/123/777; Arm B is the clean rule
`mu[c,a] += .05*(f-mu[c,a])`.

The conservation monitor uses `alpha=.80`, `V=1`, `theta_min=.467`, and `q` equal to the rolling
50-decision overall correctness. Its pressure is `alpha*q*V`. `GREEN` requires pressure ≥ theta_min;
otherwise the baseline ProfileScorer is placed in `AMBER` and its auto-pause blocks learning. The
reported safety floor is the minimum rolling q after the first full window and must remain ≥ theta_min.

## Detection and policy variants

The detector compares each category's phase-2 rolling-50 accuracy with its phase-1 category baseline.
The 50-decision window is required before a category is eligible for detection; this suppresses
single-cycle noise false positives in the uniform-degradation control. It additionally requires the
mean category drop to be no greater than 1pp, so a broad degradation is not reclassified as a sparse
event merely because finite-sample categories fluctuate.
After every category has at least two phase-2 observations, a sparse disruption is declared when at
least two categories drop by >10 percentage points and at least two other categories are within 5pp
of their phase-1 baseline. Selected categories are the categories with the largest drops. This is
an observable category-level rule; it does not inspect GT or scorer centroids.

`R1-selective-redistribution` uses the real ProfileScorer with `eta_override=.01`, but while the
conservation pressure is AMBER it permits updates only for selected categories and only while the
overall rolling q remains ≥ theta_min. Non-selected updates remain blocked. This redistributes the
existing update opportunity without relaxing the floor.

`R2-selective-boost` uses the same detector, safety guard, and selected-category routing, but raises
the real scorer's override rate to `.05` for a bounded 24-decision window (one coverage cycle), then
restores `.01`. This is a transient policy variant, not a permanent rate change.

The uniform-degradation selectivity control is a separate stress control: every Phase-2 observation
is centered on the farthest other action profile for its category, with the original action label
retained. Thus all six categories receive the same wrong-action stress while GT and the labeling
rule remain unchanged. The same detector
and policy run there; it must remain silent and must not elevate updates. Arm A baseline, both R1/R2,
and Arm B are run on the sparse-disruption primary cells. The uniform control runs Arm A and R2.

## Gate and falsifiers

For each arm/seed/epsilon, fit the pre-existing scale-free exponential rate: Phase 1 on all cells,
Phase 2 on disrupted categories [0,1], and `gamma=k_phase2/k_phase1`. The gate is fit validity only
(positive k and ≥5 retained ten-decision blocks). The pass gate requires the selected policy γ>1 at
both ε points, γ≥Arm B at each point, no safety-floor breach, and no mode activation under uniform
degradation. Any failure is reported as a result, not repaired by relaxing a gate.

## Persistence and chart

Each cell persists GT/μ0, vectors, per-category accuracy, per-decision correctness, conservation
pressure/status, selected categories, mode transitions, minimum rolling q, centroid trajectories,
gamma/rate records, and invariant assertions. A manifest hashes every artifact and the PNG/PDF
verification chart. The chart shows γ by ε for baseline A, the selected policy, and B with γ=1 and
ε*=.125 references.
