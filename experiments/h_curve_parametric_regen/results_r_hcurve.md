# R-HCURVE results v1

## Result summary

The conservation-compatible selective mode did **not** pass the arbiter. The frozen overall safety
floor is `theta_min=0.467`, but the observed rolling-50 correctness fell to 0.10–0.26 in the primary
cells after the first full window. Consequently, the mode could not legally authorize the blocked
updates; no policy recovered a valid gamma at both epsilon points. This is a FAIL/INCONCLUSIVE result
for R-HCURVE on this apparatus, not evidence that the theorem or the oracle labeling rule is false.

## Design fixes applied

1. The initial detector used a rolling-10 category window and produced finite-sample false positives
   in the uniform control. It was changed to a rolling-50 window, with detection unavailable until a
   full category window exists. This preserves the sparse-drop rule while reducing one-cycle noise.
2. A broad-degradation guard was added: the mean category drop must be no greater than 1 percentage
   point. A positive aggregate drop is not reclassified as a sparse redistribution target.
3. The original uniform control used a common target translation, which preserves action geometry and
   did not guarantee uniform observed degradation. The control now centers each Phase-2 observation on
   the farthest other action profile for its category, retains the original action label, and uses a
   small common noise term. This produces an all-category wrong-action stress while leaving the GT,
   oracle rule, and primary sparse-disruption cells unchanged.

These are apparatus fixes only. `ProfileScorer`, its base learning rule, GT construction, and the
oracle labels were not modified. The final run was performed after all three fixes.

## Variants tried

| Variant | ε=0.20 gamma (seeds 42/123/777) | ε=0.35 gamma (seeds 42/123/777) | Reading |
|---|---|---|---|
| Arm A baseline | DNF / DNF / DNF | DNF / DNF / DNF | Conservation auto-pause; no valid positive two-phase fit |
| R1 selective redistribution | DNF / DNF / DNF | 0.503 / DNF / 0.537 | Selected updates remain blocked by the same overall floor |
| R2 selective boost | DNF / DNF / DNF | 0.503 / DNF / 0.537 | Boost never authorized a safety-floor-breaching update |
| Arm B theorem control | 1.163 / 1.573 / 1.312 | 1.310 / 0.682 / 1.011 | Ideal rule control reproduces the prior persisted results |

DNF means the fit-validity gate did not yield a usable positive rate in both phases. The modified
policy is below Arm B on both valid ε=0.35 cells and is not valid at ε=0.20.

## Per-seed gamma and gate

| ε | Seed | Arm A | R1 | R2 | Arm B | R2 gate | R2 min q |
|---:|---:|---:|---:|---:|---:|---|---:|
| 0.20 | 42 | DNF | DNF | DNF | 1.163 | fail | 0.14 |
| 0.20 | 123 | DNF | DNF | DNF | 1.573 | fail | 0.16 |
| 0.20 | 777 | DNF | DNF | DNF | 1.312 | fail | 0.10 |
| 0.35 | 42 | DNF | 0.503 | 0.503 | 1.310 | fail | 0.26 |
| 0.35 | 123 | DNF | DNF | DNF | 0.682 | fail | 0.26 |
| 0.35 | 777 | DNF | 0.537 | 0.537 | 1.011 | fail | 0.22 |

R2 is not gamma-sign-consistent above the threshold: the two valid ε=0.35 values are both below 1,
and the third is DNF. It is also below Arm B wherever both are valid (`0.503 < 1.310`,
`0.537 < 1.011`).

## Safety-floor-held confirmation

| Arm/policy | ε=0.20 minimum q by seed | ε=0.35 minimum q by seed | Floor held in every run? |
|---|---|---|---|
| Arm A | 0.16 / 0.20 / 0.20 | 0.18 / 0.18 / 0.20 | No |
| R1 | 0.14 / 0.16 / 0.10 | 0.26 / 0.26 / 0.22 | No |
| R2 | 0.14 / 0.16 / 0.10 | 0.26 / 0.26 / 0.22 | No |
| Arm B | 0.10 / 0.14 / 0.16 | 0.24 / 0.24 / 0.24 | No (diagnostic control, not conservation-gated) |

No policy inflated or relaxed `theta_min`; the floor failure is measured, not hidden. The primary
apparatus therefore does not furnish a legal opportunity for selective redistribution to operate.

## Selectivity control

The final uniform-degradation R2 control activated the mode in **0/6** cells (seeds 42/123/777 ×
ε=0.20/0.35). Arm A also activated in 0/6. This satisfies the selectivity requirement for the
final control construction: the detector remained silent when every category was driven toward a
wrong action profile. The control's minimum q was 0.00–0.00 for R2, as expected for a deliberately
uniform wrong-action stress, but it was not used to claim a convergence result.

## Falsifiers and arbiter

- **F1 (gamma binary prediction): fired for the modified policy on the usable ε=0.35 cells.** The
  valid R2 values were 0.503 and 0.537, both `<1`; ε=0.20 had no valid R2 gamma.
- **F2 (direction agreement): not evaluable for R2 on most cells.** The fit-validity gate failed for
  the modified policy at all ε=0.20 cells and seed 123 at ε=0.35. No direction claim is made from
  censored cells.
- **F3 (fit-validity/safety result): fired as an apparatus/production-safety blocker.** The overall
  rolling correctness was below `theta_min` in every primary modified-policy run, so conservation
  correctly withheld the selective updates.
- **Pass gate:** failed. R2 did not achieve γ>1 at both ε values, did not meet γ≥Arm B, and the
  overall safety floor was not held by the observed decision stream.

The strongest conclusion supported by this run is that the tested conservation-compatible mechanism
cannot recover re-convergence in this H-CURVE apparatus because its required safety budget is already
below the frozen floor. This does not distinguish whether a calibrated production-grade q/θ_min
interface would change the result; that is a roadmap item, not a silently altered experiment.

## Invariants

Persisted invariant records report exact canonical `mu0`, GT distinctness, finite target tensors, a
scale-free fit gate, policy-local variation, and an unrelaxed overall floor for every primary and
control record. Primary vectors/GT/coverage were reused identically across policies; only the policy
path varied. Arm B remained the bare `mu += .05*(f-mu)` control. No production scorer source was
modified.

## Artifacts and persistence

- Harness: `experiments/h_curve_parametric_regen/run_r_hcurve.py`
- Design: `experiments/h_curve_parametric_regen/design_r_hcurve.md`
- Raw cells and uniform controls: `experiments/h_curve_parametric_regen/raw/r_hcurve_v1/`
- Summary: `experiments/h_curve_parametric_regen/raw/r_hcurve_v1/summary.json`
- Manifest: `experiments/h_curve_parametric_regen/raw/r_hcurve_v1/manifest.json`
- Chart: `experiments/h_curve_parametric_regen/figures/r_hcurve_gamma.png` and `.pdf`

Every persisted record contains per-decision correctness, category trajectories, q/pressure/status,
mode transitions, distances, rate records, and invariant records. The manifest records SHA-256 and
byte size for every raw artifact and chart; it is regenerated after the final run and verified before
this report is finalized. A scorer-free audit recomputes the fit inputs from the persisted records;
the report does not depend on a re-run.

## Verdict

**FAIL — no conservation-compatible disruption-aware policy tested here recovered γ>1 while keeping
the frozen overall safety floor; R-HCURVE remains an unresolved production/apparatus boundary, not a
validated roadmap mechanism.**
