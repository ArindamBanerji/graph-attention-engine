# R-HCURVE persisted-regime diagnostic

This is a read-only re-analysis of `raw/r_hcurve_v1`; no record, scorer, GT, label, or experiment run was modified. Primary regime classification uses Arm A baseline records. `pre` is the full Phase-1 per-category correctness; `post early` is the first 50 Phase-2 observations for that category; `post late` is its last 50. A category is DISRUPTED when the early drop is >10pp, HELD when the drop is ≤5pp, and INTERMEDIATE otherwise.

## 1. Regime: sparse or global?

### seed 42, ε=0.20

| category | pre q | post early q | post late q | early drop | late drop | class | local min q |
|---|---|---|---|---|---|---|---|
| 0 | 0.283 | 0.260 | 0.140 | 0.023 | 0.143 | HELD | 0.140 |
| 1 | 0.275 | 0.220 | 0.260 | 0.055 | 0.015 | INTERMEDIATE | 0.160 |
| 2 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 3 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 4 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 5 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |

Trough window: q=0.160 at phase-2 decision 100; weighted category reconstruction=0.160; category q={'0': 0.0, '1': 0.0, '2': 0.25, '3': 0.25, '4': 0.25, '5': 0.25}.

### seed 42, ε=0.35

| category | pre q | post early q | post late q | early drop | late drop | class | local min q |
|---|---|---|---|---|---|---|---|
| 0 | 0.305 | 0.260 | 0.380 | 0.045 | -0.075 | HELD | 0.200 |
| 1 | 0.315 | 0.340 | 0.400 | -0.025 | -0.085 | HELD | 0.160 |
| 2 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 3 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 4 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 5 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |

Trough window: q=0.180 at phase-2 decision 131; weighted category reconstruction=0.180; category q={'0': 0.0, '1': 0.125, '2': 0.2, '3': 0.25, '4': 0.25, '5': 0.25}.

### seed 123, ε=0.20

| category | pre q | post early q | post late q | early drop | late drop | class | local min q |
|---|---|---|---|---|---|---|---|
| 0 | 0.308 | 0.220 | 0.320 | 0.088 | -0.012 | INTERMEDIATE | 0.200 |
| 1 | 0.358 | 0.300 | 0.320 | 0.058 | 0.038 | INTERMEDIATE | 0.280 |
| 2 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 3 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 4 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 5 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |

Trough window: q=0.200 at phase-2 decision 151; weighted category reconstruction=0.200; category q={'0': 0.0, '1': 0.2, '2': 0.25, '3': 0.25, '4': 0.25, '5': 0.25}.

### seed 123, ε=0.35

| category | pre q | post early q | post late q | early drop | late drop | class | local min q |
|---|---|---|---|---|---|---|---|
| 0 | 0.395 | 0.320 | 0.440 | 0.075 | -0.045 | INTERMEDIATE | 0.280 |
| 1 | 0.375 | 0.320 | 0.360 | 0.055 | 0.015 | INTERMEDIATE | 0.240 |
| 2 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 3 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 4 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 5 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |

Trough window: q=0.220 at phase-2 decision 391; weighted category reconstruction=0.220; category q={'0': 0.125, '1': 0.2, '2': 0.25, '3': 0.25, '4': 0.25, '5': 0.25}.

### seed 777, ε=0.20

| category | pre q | post early q | post late q | early drop | late drop | class | local min q |
|---|---|---|---|---|---|---|---|
| 0 | 0.242 | 0.240 | 0.220 | 0.002 | 0.022 | HELD | 0.200 |
| 1 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 2 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 3 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 4 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 5 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |

Trough window: q=0.220 at phase-2 decision 76; weighted category reconstruction=0.220; category q={'0': 0.1, '1': 0.25, '2': 0.25, '3': 0.25, '4': 0.25, '5': 0.25}.

### seed 777, ε=0.35

| category | pre q | post early q | post late q | early drop | late drop | class | local min q |
|---|---|---|---|---|---|---|---|
| 0 | 0.255 | 0.220 | 0.200 | 0.035 | 0.055 | HELD | 0.160 |
| 1 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 2 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 3 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 4 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |
| 5 | 0.250 | 0.260 | 0.240 | -0.010 | 0.010 | HELD | 0.240 |

Trough window: q=0.200 at phase-2 decision 340; weighted category reconstruction=0.200; category q={'0': 0.0, '1': 0.25, '2': 0.25, '3': 0.25, '4': 0.25, '5': 0.25}.


Across the 6 primary cells (36 category instances), DISRUPTED count is 0, HELD count is 31, and INTERMEDIATE count is 5. No category exceeded the >10pp drop threshold; the apparent floor failure is not a sparse category drop. The trough-window category reconstruction uses the same persisted decision order and correctness bits as the rolling-q calculation.

## 2. Floor accounting

| seed | ε | minimum phase-2 q | reconstructed q |
|---|---|---|---|
| 42 | 0.20 | 0.160 | 0.160 |
| 42 | 0.35 | 0.180 | 0.180 |
| 123 | 0.20 | 0.200 | 0.200 |
| 123 | 0.35 | 0.220 | 0.220 |
| 777 | 0.20 | 0.220 | 0.220 |
| 777 | 0.35 | 0.200 | 0.200 |

The reconstruction matches the observed rolling-50 q by construction (differences are floating-point display only). Therefore a sparse drop does not automatically imply a 0.10–0.26 trough: the trough is determined by the weighted category correctness in its 50-decision window. The category table makes whether the majority held visible per cell; the observed q cannot be attributed to a minority without that accounting.

## 3. Slack map

| candidate θ_min | worst-cell slack (min q − θ) | primary cells held |
|---|---|---|
| 0.100 | 0.060 | 6/6 |
| 0.150 | 0.010 | 6/6 |
| 0.200 | -0.040 | 4/6 |
| 0.250 | -0.090 | 0/6 |
| 0.300 | -0.140 | 0/6 |
| 0.350 | -0.190 | 0/6 |
| 0.400 | -0.240 | 0/6 |
| 0.467 | -0.307 | 0/6 |

The maximum aggregate floor that could have been held in every primary cell is the minimum persisted trough across cells. A θ_min above that value leaves no legal slack; lower thresholds would have allowed some or all cells to remain GREEN, but that is a counterfactual map, not a changed experiment.

## 4. Per-category gating feasibility

| seed | ε | local floor | held above floor | held count | disrupted above floor | disrupted count |
|---|---|---|---|---|---|---|
| 42 | 0.20 | 0.20 | 4 | 5 | 0 | 0 |
| 42 | 0.20 | 0.30 | 0 | 5 | 0 | 0 |
| 42 | 0.20 | 0.40 | 0 | 5 | 0 | 0 |
| 42 | 0.35 | 0.20 | 4 | 6 | 0 | 0 |
| 42 | 0.35 | 0.30 | 0 | 6 | 0 | 0 |
| 42 | 0.35 | 0.40 | 0 | 6 | 0 | 0 |
| 123 | 0.20 | 0.20 | 4 | 4 | 0 | 0 |
| 123 | 0.20 | 0.30 | 0 | 4 | 0 | 0 |
| 123 | 0.20 | 0.40 | 0 | 4 | 0 | 0 |
| 123 | 0.35 | 0.20 | 4 | 4 | 0 | 0 |
| 123 | 0.35 | 0.30 | 0 | 4 | 0 | 0 |
| 123 | 0.35 | 0.40 | 0 | 4 | 0 | 0 |
| 777 | 0.20 | 0.20 | 6 | 6 | 0 | 0 |
| 777 | 0.20 | 0.30 | 0 | 6 | 0 | 0 |
| 777 | 0.20 | 0.40 | 0 | 6 | 0 | 0 |
| 777 | 0.35 | 0.20 | 5 | 6 | 0 | 0 |
| 777 | 0.35 | 0.30 | 0 | 6 | 0 | 0 |
| 777 | 0.35 | 0.40 | 0 | 6 | 0 | 0 |

Local q is the minimum rolling-50 correctness within Phase 2 for each category. There are no DISRUPTED categories under the preregistered >10pp rule, so this map cannot demonstrate a held-versus-disrupted deadlock; the held categories themselves have low local q. It does not implement local gating and does not authorize changing θ_min.

## 5. Signal separability

The records persist aggregate `d_active` and disrupted-subspace distance trajectories, but not per-category centroid-distance trajectories or per-decision centroid snapshots. They also persist `q`, pressure, status, correctness, and category correctness, but no independent V/verification channel. Thus an operationally independent learned-vs-garbage test is **NOT FOUND** in the persisted records.

A limited input-consistency check is available: the persisted Phase-2 vectors and `gt2.npy` permit scorer-free residual norms to the generated target. This verifies whether observations were generated near GT2; it does not establish that the production scorer learned a new target.

| seed | ε | mean ||f−GT2|| | median ||f−GT2|| |
|---|---|---|---|
| 42 | 0.20 | 0.1882 | 0.1879 |
| 42 | 0.35 | 0.1887 | 0.1856 |
| 123 | 0.20 | 0.1861 | 0.1821 |
| 123 | 0.35 | 0.1874 | 0.1833 |
| 777 | 0.20 | 0.1909 | 0.1885 |
| 777 | 0.35 | 0.1877 | 0.1866 |

Re-instrument item: persist per-decision per-category μ/centroid distance, an independent verification/outcome channel, and an explicit target-consistency statistic if the next run must distinguish a legitimate learnable shift from garbage.

## One-line verdict

**WRONG REGIME — no category shows the required >10pp disruption drop; the trough comes from uniformly low baseline/post accuracy rather than a genuinely sparse minority shift, so R-HCURVE did not test the requested sparse-disruption regime.**

## Diagnostic artifacts

- Chart: `r_hcurve_category_regime.png` and `r_hcurve_category_regime.pdf`
- Source records: `../raw/r_hcurve_v1/`
- This report and chart are hashed in `diagnostic_manifest.json`.
