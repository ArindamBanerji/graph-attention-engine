# H-CURVE calibrated re-convergence — results v1

## Calibration gate

The widened action-profile construction used `SEPARATION=0.28`. The calibration-only Arm-A
Phase-1 baseline passed before any Arm-A/B re-convergence records were accepted:

| seed | ε_firm | final-100 accuracy | θ_min | headroom | gate |
|---:|---:|---:|---:|---:|---|
| 42 | 0.20 | 1.000 | 0.467 | +0.533 | PASS |
| 42 | 0.35 | 1.000 | 0.467 | +0.533 | PASS |
| 123 | 0.20 | 1.000 | 0.467 | +0.533 | PASS |
| 123 | 0.35 | 1.000 | 0.467 | +0.533 | PASS |
| 777 | 0.20 | 1.000 | 0.467 | +0.533 | PASS |
| 777 | 0.35 | 1.000 | 0.467 | +0.533 | PASS |

The gate’s hard minimum was >0.50. The selected geometry exceeds the sibling positive-control band
and is therefore a deliberately strong calibration, not a claim that production accuracy is exactly
100%. A calibration-only sweep showed a sharp transition: lower separation left at least one cell
below the hard gate; `0.28` was retained because it clears every frozen cell before the arm test.

## Gamma: Arm A vs Arm B

| ε_firm | seed | Arm A γ_rate | Arm B γ_rate | Arm A status | Arm B status |
|---:|---:|---:|---:|---|---|
| 0.20 | 42 | 1.090 | 1.022 | valid | valid |
| 0.20 | 123 | 1.207 | 1.155 | valid | valid |
| 0.20 | 777 | 1.128 | 0.971 | valid | valid |
| 0.35 | 42 | 1.279 | 1.211 | valid | valid |
| 0.35 | 123 | 1.161 | 1.008 | valid | valid |
| 0.35 | 777 | 1.229 | 1.028 | valid | valid |

Arm A is γ>1 for all 6 cells and is sign-consistent across all three seeds at both ε values. Arm B
is γ>1 in 5/6 cells; its ε=0.20 seed-777 value is 0.971. Arm A is not below Arm B in this
calibrated regime; it exceeds Arm B in every paired cell.

## Conservation engagement evidence

| Arm | ε | seed | minimum Phase-2 q | engaged fraction | paused fraction |
|---|---:|---:|---:|---:|---:|
| A | 0.20 | 42/123/777 | 1.000 | 1.000 | 0.000 |
| A | 0.35 | 42/123/777 | 1.000 | 1.000 | 0.000 |

The per-decision traces contain `overall_accuracy`, `pressure=α·q·V`, `theta_min`, status, and
learning permission. Arm A remained `ENGAGED` for every Phase-2 decision in all six cells; no
conservation pause occurred. The verification chart shows this directly for seed 42, ε=0.35.

## Read outcome and verdict

**ARTIFACT CONFIRMED.** The recalibrated baseline clears θ_min with substantial headroom; Arm A
re-converges with γ>1; and conservation remains engaged throughout Phase 2. The earlier Arm-A gap
was a calibration artifact caused by chance-level action discrimination, not evidence that the
conservation auto-pause blocked re-convergence in this calibrated regime. R-HCURVE is retired under
this recalibrated operating point.

This result is scoped to the widened, bounded action-profile geometry. It does not claim that every
GT geometry will remain above θ_min; the prior chance-accuracy geometry remains a distinct invalid
calibration for a conservation test.

## Invariants and artifacts

The run kept the fixed action-index labeling, unchanged ProfileScorer learning rule, unchanged
`theta_min=0.467`, Arm B bare update, per-phase target/vector alignment, and scale-free γ. Each cell
persists GT1/GT2/μ0/μfinal, vectors, per-decision correctness, overall q vs θ_min, conservation
status, category trajectories, distances, rate/gamma records, and invariant records.

- Runner: `run_hcurve_calibrated.py`
- Raw artifacts: `raw/calibrated_v1/`
- Manifest: `raw/calibrated_v1/manifest.json`
- Verification chart: `figures/hcurve_calibrated_verification.png` and `.pdf`

The manifest includes byte sizes and SHA-256 hashes for every raw artifact, calibration record,
chart, and this design/results documentation. A scorer-free recomputation of the calibration tail
accuracy and γ is required by the manifest verification and is reported above.
