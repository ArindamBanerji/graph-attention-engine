# GAE API Contract
# License: Apache 2.0 — public library

This document is the authoritative reference for what callers may depend on.
Breaking a Tier 1 function is a production incident for the 290+ SOC call
sites and 58+ S2P call sites that consume GAE.

---

## Tier 1 — Stable (never break without major version bump)

### ProfileScorer

```python
from gae.profile_scorer import ProfileScorer, ScoringResult, CentroidUpdate, KernelType

ProfileScorer(
    mu: np.ndarray,          # shape (n_categories, n_actions, n_factors), values in [0,1]
    actions: List[str],      # len == mu.shape[1]
    kernel: KernelType = KernelType.L2,
    profile: CalibrationProfile | None = None,   # None → validated defaults
    categories: List[str] | None = None,
    min_confidence: float = 0.0,
    eta_override: float | None = None,           # recommended 0.01 (Q5 validated)
    factor_mask: np.ndarray | None = None,
    scoring_kernel: ScoringKernel | None = None, # None → L2Kernel()
    auto_pause_on_amber: bool = False,
)
```

#### Calibration constants (Tier 1, validated by experiments)

| Constant     | Default | Source       | Notes                                 |
|--------------|---------|--------------|---------------------------------------|
| `tau`        | `0.1`   | V3B ECE=0.036| NEVER change without major version    |
| `eta`        | `0.05`  | P1 confirmed | η_confirm learning rate               |
| `eta_neg`    | `0.05`  | P1 confirmed | η_neg learning rate; ≥1.0 is FORBIDDEN|
| `eta_override`| `None` | Q5 validated | Recommended value 0.01 when set       |
| `decay`      | `0.001` | —            | Per-(category,action) count decay     |

**FORBIDDEN**: `eta_neg >= 1.0` → `ValueError` (ECE=0.49, catastrophic).

#### ProfileScorer.score

```python
score(f: np.ndarray, category_index: int) -> ScoringResult
```

- `f` shape: `(n_factors,)`, values nominally in `[0,1]`
- Returns `ScoringResult`:
  - `.probabilities`: shape `(n_actions,)`, sums to 1.0, computed via softmax(-dist/τ)
  - `.distances`: shape `(n_actions,)`, raw kernel distances
  - `.action_index`: argmax of probabilities
  - `.action_name`: name string
  - `.confidence`: probability of recommended action
- **Deterministic**: identical inputs → identical output
- **No side effects**: does not mutate centroids

#### ProfileScorer.update

```python
update(
    f: np.ndarray,
    category_index: int,
    action_index: int,
    correct: bool,
    gt_action_index: int | None = None,
    confidence: float | None = None,
) -> CentroidUpdate
```

- `correct=True`: pull `μ[c,a,:]` toward f (confirm path, rate `η`)
- `correct=False`: push `μ[c,a,:]` away from f; pull `μ[c,gt,:]` toward f (override path)
- All centroid values clipped to `[0.0, 1.0]` after every update (V2 requirement)
- MAX_ETA_DELTA cap: ±0.005 per coordinate per step (V-STABILITY F=8.14)
- Returns `CentroidUpdate` with `centroid_delta_norm`, outcome, etc.

#### ProfileScorer.centroids

```python
@property
centroids -> np.ndarray  # shape (n_categories, n_actions, n_factors)
```

Read access to the centroid tensor `μ`. Alias for `.mu`.  
To replace centroids: assign directly to `scorer.mu = new_mu` (must be same shape).

---

### Kernels

#### L2Kernel

```python
from gae.kernels import L2Kernel

L2Kernel()
# distance(f, μ_a) = Σ_j (f_j − μ_{a,j})²      shape (A,)
# gradient(f, μ)   = f − μ                        shape (d,)
```

Default kernel. Validated: 97.89% oracle accuracy (EXP-C1).

#### DiagonalKernel

```python
from gae.kernels import DiagonalKernel

DiagonalKernel(sigma: np.ndarray)   # sigma shape (d,), all values > 0
# Internally: W = 1/σ², self.weights = W / W.max()  ∈ [0,1]
# distance(f, μ_a) = Σ_j weights_j × (f_j − μ_{a,j})²
# gradient(f, μ)   = (W / W.max()) ⊙ (f − μ)        [bounded, no amplification]
```

Higher σ_j (noisier factor) → lower weight. When all σ equal → identical to L2Kernel.
Use `KernelSelector` to choose between L2 and Diagonal based on `noise_ratio > 1.5`.

---

### Conservation law

```python
from gae.calibration import derive_theta_min, check_conservation, ConservationCheck

derive_theta_min(
    eta: float = 0.05,
    n_half: float = 14.0,
    t_max_days: float = 21.0,
) -> float
# θ_min = η × N_half² / T_max
# SOC default (21-day): ≈ 0.467
# S2P default (26-day): ≈ 0.377

check_conservation(
    alpha: float,   # override rate
    q: float,       # override quality
    V: float,       # verified decisions per day
    theta_min: float,
) -> ConservationCheck
# signal = α·q·V
# GREEN : signal ≥ 2·θ_min  (healthy)
# AMBER : θ_min ≤ signal < 2·θ_min  (thinning)
# RED   : signal < θ_min  (breach)
```

---

### CalibrationProfile

```python
from gae.calibration import CalibrationProfile, soc_calibration_profile, s2p_calibration_profile

CalibrationProfile(temperature, learning_rate, penalty_ratio, ...)
soc_calibration_profile()   # SOC defaults: τ=0.25, η=0.05, 20:1 penalty
s2p_calibration_profile()   # S2P defaults: τ=0.40, η=0.05, 5:1 penalty
```

Pass to `ProfileScorer(profile=...)` to override validated defaults.

---

### Domain tensor shapes (SOC and S2P)

| Domain | Categories | Actions | Factors | Tensor shape |
|--------|------------|---------|---------|--------------|
| SOC    | 6          | 4       | 6       | (6, 4, 6)   |
| S2P    | 5          | 5       | 8       | (5, 5, 8)   |

Both domains use the same `ProfileScorer` class with different `mu` shapes.

---

## Tier 2 — Evolving (may change between minor versions)

- `KernelSelector` — rolling 100-window, recommends Diagonal when `noise_ratio > 1.5`
- `CovarianceEstimator` — per-factor σ estimation, half_life=300
- `ConservationMonitor` — CUSUM-based YELLOW early warning (Layer 2)
- `OLSMonitor` — plateau-snapshot OLS baseline monitor
- `GainScheduler` — Block 9.6, not yet validated

---

## Tier 3 — Experimental (may disappear or change arbitrarily)

- `gae/experiments/*` — research infrastructure
- `gae/synthetic.py` — OracleSeparationExperiment, FactorVectorSampler
- `gae/convergence.py` — centroid_distance_to_canonical, gamma_threshold

---

## Forbidden Breaking Changes (10 rules)

1. **Never** change `ProfileScorer.score()` return type (`ScoringResult`)
2. **Never** change centroid tensor shape convention `(n_cat, n_act, d)`
3. **Never** change `tau=0.1` default without a major version bump
4. **Never** change `eta=0.05` / `eta_neg=0.05` defaults without a major version bump
5. **Never** remove a Tier 1 function or property
6. **Never** add database or network dependencies (GAE is pure numpy)
7. **Never** allow `eta_neg >= 1.0` (ECE=0.49, catastrophic — hard guard enforced)
8. **Never** change DiagonalKernel gradient formula (W/W.max() × (f − μ))
9. **Never** allow A=5 (`refer_to_analyst`) in the scoring path (removed at v6.0)
10. **Never** define action constants in multiple locations (single source of truth)
