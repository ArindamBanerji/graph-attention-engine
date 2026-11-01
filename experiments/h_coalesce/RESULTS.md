# E-NEW-5 v4.3 — H-COALESCE adaptive accuracy-ruler results

Passing cell C,A,d,SEPARATION,near-mode,near-offset = (6, 4, 20, 0.12, 0.1, 0); competence threshold=0.90; q uses expanding/rolling-50 and competence is the first post-window crossing; AUT_ACC is area of 1-q. CHECK1 accepts 50–800 decisions: neither an immediate crossing nor a budget-edge result.

## Reused apparatus

- GT displacement convention and calibrated constructor basis: `experiments/h_curve_parametric_regen/run_hcurve_calibrated.py:65-91`.
- GT-centered noise and deterministic coverage: `experiments/h_curve_parametric_regen/run_hcurve_calibrated.py:94-109`.
- ProfileScorer asymmetric update: `experiments/h_curve_parametric_regen/run_hcurve_calibrated.py:15,165-167`; `gae/profile_scorer.py:815-850`.
- Conservation status/pressure and pause/resume: `experiments/h_curve_parametric_regen/run_hcurve_calibrated.py:132-150`; `gae/profile_scorer.py:712-744`.
- Distance/accuracy records and all v4.3 reported numbers: `experiments/h_coalesce/run_hcoalesce_v43.py:42-242`.

## Adaptive pre-flight search path

| C | A | d | sep | near mode | offset | CHECK1 | CHECK2 | CHECK3 | CHECK4 | cold N | warm N | warm q0 |
|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|
| 20 | 10 | 20 | 0.15 | 0.40 | 0.00 | False | True | False | True | None | None | 0.600 |
| 4 | 4 | 20 | 0.28 | 0.40 | 0.00 | True | True | False | True | 237 | None | 0.499 |
| 6 | 4 | 20 | 0.12 | 0.10 | 0.00 | True | True | True | True | 275 | 134 | 0.833 |

## Pre-flight validation at passing cell

| seed | CHECK1 | CHECK2 | CHECK3 | CHECK4 | COLD N | WARM N | OPTIMAL N | WARM q0 | OPTIMAL q0 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 42 | True | True | True | True | 275 | 134 | 0 | 0.833 | 1.000 |
| 123 | True | True | True | True | 502 | 200 | 0 | 0.833 | 1.000 |
| 777 | True | True | True | True | 468 | 68 | 0 | 0.833 | 1.000 |

All four pre-flight checks passed before the six-arm run.

## Design fixes applied

1. The first final invocation exposed a timepoint/budget indexing bug; preflight and final traces now use the same 3,000-decision budget, and the timepoint grid extends through 3,000.
2. The original FAR construction reused the A action-profile basis and produced centered cosine 0.9965; FAR now uses an independently seeded profile basis (`run_hcoalesce_v43.py:87-90`), yielding realized centered cosine −0.0648, 0.1230, and 0.0895.
3. The smooth near blend left WARM_RELATED at 100% static accuracy. NEAR now applies a deterministic one-category action permutation (`run_hcoalesce_v43.py:91-96`), producing static q0=0.833 and centered cosine ≈0.81; this is reported as a realized deviation from the ideal “near 1” target.
4. The adaptive ruler uses THR=0.90 and CHECK1=50–800 decisions: 90% is a high fraction of the perfect OPTIMAL plateau and the window excludes immediate and budget-edge crossings.

## Ruler limitation

The accuracy-matching random search did not find a RANDOM_SHARP_ACCMATCHED configuration near WARM_RELATED's q0. The realized ARM6 q0 values were 0.436, 0.450, and 0.427 versus ARM4's 0.833 for all seeds. ARM4 therefore has a large starting-accuracy advantage; ARM4–ARM6 is not an exact matched-start structural contrast. The observed gap is evidence that learned geometry can achieve target-aligned competence that the sampled random geometry did not, but it cannot by itself separate transferred content from that initial accuracy/geometry advantage.

## AUT_ACC / competence per seed

| seed | COLD | RANDOM_SHARP | WARM_UNRELATED | WARM_RELATED | OPTIMAL | RANDOM_SHARP_ACCMATCHED |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 154.562/275 | 2164.120/None | 2755.160/None | 216.337/134 | 0.000/0 | 776.973/1738 |
| 123 | 347.473/502 | 2711.232/None | 469.881/1181 | 215.017/200 | 0.000/0 | 669.756/1516 |
| 777 | 305.902/468 | 2412.203/None | 347.539/731 | 207.312/68 | 0.000/0 | 811.682/1679 |

## Aggregate AUT_ACC

| arm | mean AUT_ACC | mean competence |
|---|---:|---:|
| COLD | 269.312 | 415.0 |
| RANDOM_SHARP | 2429.185 | DNF |
| WARM_UNRELATED | 1190.860 | 956.0 |
| WARM_RELATED | 212.889 | 134.0 |
| OPTIMAL | 0.000 | 0.0 |
| RANDOM_SHARP_ACCMATCHED | 752.804 | 1644.3333333333333 |

| contrast left-right | mean delta | per-seed | left faster wins/3 |
|---|---:|---|---:|
| ARM1-ARM2 conditioning (COLD-RANDOM_SHARP) | -2159.873 | -2009.558, -2363.759, -2106.302 | 3/3 |
| ARM2-ARM3 wrong-vs-random (RANDOM_SHARP-WARM_UNRELATED) | 1238.325 | -591.040, 2241.351, 2064.665 | 1/3 |
| ARM3-ARM4 right-vs-wrong (WARM_UNRELATED-WARM_RELATED) | 977.971 | 2538.823, 254.864, 140.226 | 0/3 |
| ARM4-ARM6 decisive (WARM_RELATED-RANDOM_SHARP_ACCMATCHED) | -539.915 | -560.636, -454.739, -604.370 | 3/3 |
| ARM4-ARM5 ceiling (WARM_RELATED-OPTIMAL) | 212.889 | 216.337, 215.017, 207.312 | 0/3 |

## Accuracy q(t), pauses, centered/raw distance

| seed | arm | q0 | q10 | q25 | q50 | q100 | q200 | q400 | AUT_ACC | N | pauses | centered d0 | raw d0 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | COLD | 0.250 | 0.200 | 0.280 | 0.220 | 0.360 | 0.620 | 0.980 | 154.562 | 275 | 159 | 2.6087 | 2.6278 |
| 42 | RANDOM_SHARP | 0.258 | 0.100 | 0.240 | 0.180 | 0.280 | 0.300 | 0.320 | 2164.120 | None | 2885 | 4.6703 | 4.7129 |
| 42 | WARM_UNRELATED | 0.083 | 0.000 | 0.080 | 0.080 | 0.140 | 0.100 | 0.020 | 2755.160 | None | 3000 | 3.8499 | 3.8562 |
| 42 | WARM_RELATED | 0.833 | 0.900 | 0.800 | 0.840 | 0.840 | 0.860 | 0.840 | 216.337 | 134 | 7 | 1.6244 | 1.6245 |
| 42 | OPTIMAL | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0 | 0.0000 | 0.0000 |
| 42 | RANDOM_SHARP_ACCMATCHED | 0.436 | 0.500 | 0.600 | 0.520 | 0.480 | 0.360 | 0.480 | 776.973 | 1738 | 754 | 4.4350 | 4.4390 |
| 123 | COLD | 0.250 | 0.200 | 0.320 | 0.260 | 0.180 | 0.200 | 0.340 | 347.473 | 502 | 410 | 2.6133 | 2.6176 |
| 123 | RANDOM_SHARP | 0.088 | 0.000 | 0.040 | 0.100 | 0.100 | 0.100 | 0.060 | 2711.232 | None | 3000 | 4.9170 | 4.9178 |
| 123 | WARM_UNRELATED | 0.458 | 0.800 | 0.680 | 0.560 | 0.440 | 0.560 | 0.620 | 469.881 | 1181 | 396 | 3.5009 | 3.5074 |
| 123 | WARM_RELATED | 0.833 | 1.000 | 0.760 | 0.780 | 0.860 | 0.900 | 0.800 | 215.017 | 200 | 5 | 1.6416 | 1.6417 |
| 123 | OPTIMAL | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0 | 0.0000 | 0.0000 |
| 123 | RANDOM_SHARP_ACCMATCHED | 0.450 | 0.500 | 0.440 | 0.380 | 0.360 | 0.380 | 0.520 | 669.756 | 1516 | 594 | 4.4424 | 4.4426 |
| 777 | COLD | 0.250 | 0.100 | 0.200 | 0.200 | 0.300 | 0.240 | 0.440 | 305.902 | 468 | 378 | 2.6121 | 2.6330 |
| 777 | RANDOM_SHARP | 0.207 | 0.500 | 0.360 | 0.280 | 0.200 | 0.180 | 0.200 | 2412.203 | None | 2985 | 4.7212 | 4.7214 |
| 777 | WARM_UNRELATED | 0.382 | 0.500 | 0.240 | 0.340 | 0.460 | 0.440 | 0.660 | 347.539 | 731 | 322 | 3.5591 | 3.5766 |
| 777 | WARM_RELATED | 0.833 | 0.700 | 0.800 | 0.880 | 0.800 | 0.880 | 0.840 | 207.312 | 68 | 12 | 1.6390 | 1.6394 |
| 777 | OPTIMAL | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0 | 0.0000 | 0.0000 |
| 777 | RANDOM_SHARP_ACCMATCHED | 0.427 | 0.400 | 0.400 | 0.460 | 0.540 | 0.520 | 0.420 | 811.682 | 1679 | 828 | 4.2173 | 4.2192 |

## Geometry and accuracy matching

| seed | centered cos far | centered cos near | far sep | near sep | ARM4 q0 | ARM6 q0 | gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | -0.0648 | 0.8131 | 0.7266 | 0.7803 | 0.833 | 0.436 | 0.397 |
| 123 | 0.1230 | 0.8099 | 0.7995 | 0.7832 | 0.833 | 0.450 | 0.384 |
| 777 | 0.0895 | 0.8088 | 0.7895 | 0.7779 | 0.833 | 0.427 | 0.406 |

## Figures
- `figures\accuracy_seed_42.png`
- `figures\accuracy_seed_123.png`
- `figures\accuracy_seed_777.png`

## Interpretation

Measured outcome: **OUTCOME 1-like learned-geometry advantage, but H-COALESCE content is not cleanly identified**. ARM4-ARM6 AUT_ACC deltas are -560.636, -454.739, and -604.370, always favoring ARM4, while the supposed accuracy-matched control was not matched (residual q0 gaps +0.397, +0.383, +0.406). ARM3-ARM4 deltas are 2538.823, 254.864, and 140.226, so the right-target warm start is faster than the wrong-target warm start in all seeds. The load-bearing exact-match contrast failed its matching precondition; the valid conclusion is learned target-aligned geometry/conditioning evidence, not a provenance-clean content-only transfer claim.

## Persistence

Per-seed/per-arm JSON, search path, CSV, charts, and this report are hashed in `manifest.json`; prior raw runs are under `archive/`.
