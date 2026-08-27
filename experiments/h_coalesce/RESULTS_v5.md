# E-NEW-5 v5.1 — conservation-OFF decomposition

## Fixed apparatus and reuse

Cell is unchanged from v4.3: C=6, A=4, d=20, separation=0.12, competence=0.90, budget=3000, seeds=42/123/777.
GT, mu_A, draw streams, scorer, update rates, and metrics are loaded/reused from v4.3; only Deployment B uses `auto_pause_on_amber=False`.
- GT/profile construction and calibrated basis: `experiments/h_curve_parametric_regen/run_hcurve_calibrated.py:65-109`; v4.3 target construction: `experiments/h_coalesce/run_hcoalesce_v43.py:78-96`.
- ProfileScorer asymmetric update and conservation machinery: `experiments/h_curve_parametric_regen/run_hcurve_calibrated.py:15,132-167`; `gae/profile_scorer.py:712-744`.
- v4.3 draw replay and metrics: `experiments/h_coalesce/run_hcoalesce_v43.py:51-76`; OFF harness: `experiments/h_coalesce/run_hcoalesce_v51_off.py:51-157`.

## Verification gates

- Conservation OFF: **True**; all 15 pause counts are zero.
- Update every decision: **True**; every arm applied 3000/3000 updates.
- Input stream identity: **True** by deterministic v4.3 draw replay plus persisted category/oracle-action sequence checks. Raw `f` was not persisted by v4.3, so direct byte comparison is unavailable.
- Saved mu_A/GT provenance: **True**; loaded unchanged from v4.3 seed artifacts.

## Pre-flight sanity

COLD converged with conservation OFF for all seeds: N=275, 502, 468; pause count=0 in each case. The zero-pause OFF run did not DNF.

## Paired ON vs OFF

| seed | arm | ON AUT/N | OFF AUT/N | ON-OFF AUT | ON-OFF N | ON pauses | OFF pauses |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | COLD | 154.562/275 | 35.846/71 | 118.716 | 275 | 159 | 0 |
| 42 | RANDOM_SHARP | 2164.120/None | 601.835/632 | 1562.285 | NA | 2885 | 0 |
| 42 | WARM_UNRELATED | 2755.160/None | 478.959/572 | 2276.201 | NA | 3000 | 0 |
| 42 | WARM_RELATED | 216.337/134 | 209.712/50 | 6.625 | 134 | 7 | 0 |
| 42 | OPTIMAL | 0.000/0 | 0.000/0 | 0.000 | 0 | 0 | 0 |
| 123 | COLD | 347.473/502 | 30.550/52 | 316.923 | 502 | 410 | 0 |
| 123 | RANDOM_SHARP | 2711.232/None | 740.956/767 | 1970.276 | NA | 3000 | 0 |
| 123 | WARM_UNRELATED | 469.881/1181 | 247.453/295 | 222.429 | 1181 | 396 | 0 |
| 123 | WARM_RELATED | 215.017/200 | 207.083/58 | 7.934 | 200 | 5 | 0 |
| 123 | OPTIMAL | 0.000/0 | 0.000/0 | 0.000 | 0 | 0 | 0 |
| 777 | COLD | 305.902/468 | 29.131/50 | 276.771 | 468 | 378 | 0 |
| 777 | RANDOM_SHARP | 2412.203/None | 527.618/722 | 1884.585 | NA | 2985 | 0 |
| 777 | WARM_UNRELATED | 347.539/731 | 166.107/174 | 181.432 | 731 | 322 | 0 |
| 777 | WARM_RELATED | 207.312/68 | 200.804/50 | 6.509 | 68 | 12 | 0 |
| 777 | OPTIMAL | 0.000/0 | 0.000/0 | 0.000 | 0 | 0 | 0 |

## Content component

OFF WARM_RELATED minus OFF COLD (negative AUT means warm is faster):

| seed | AUT gap | competence gap |
|---:|---:|---:|
| 42 | 173.866 | -21 |
| 123 | 176.533 | 6 |
| 777 | 171.673 | 0 |

## Frozen-arm recovery OFF

| seed | RANDOM_SHARP OFF N | WARM_UNRELATED OFF N | ON first AMBER (random / unrelated) |
|---:|---:|---:|---|
| 42 | 632 | 572 | 1 / 1 |
| 123 | 767 | 295 | 1 / 20 |
| 777 | 722 | 174 | 2 / 6 |

## Interpretation

The OFF WARM_RELATED-vs-COLD AUT gap is positive in all three seeds (+173.866, +176.533, +171.673), so WARM_RELATED is slower on the primary trajectory metric after conservation is removed. Competence differences are mixed (-21, +6, 0 decisions for warm-minus-cold). This is OUTCOME B: the apparent ON competence advantage is conservation-mediated survival rather than an independently demonstrated content-transfer acceleration. RANDOM_SHARP and WARM_UNRELATED both recover to finite OFF competence values, showing that conservation caused much of their ON freezing, but neither establishes a surviving warm-content advantage under AUT_ACC.

## Artifacts

Per-seed arm outputs and verification are under `raw_v5_1/`; hashes are in `manifest_v5.json`.
