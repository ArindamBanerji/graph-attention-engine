# Graph Attention Engine — Project Structure

**Version:** v0.1.0 · **Branch:** v4.5-dev · **License:** Apache 2.0
**Math reference:** https://www.dakshineshwari.net/post/cross-graph-attention-mathematical-foundation-with-experimental-validation

---

## 1. Overview

The Graph Attention Engine (GAE) is a standalone, pip-installable Python library
implementing cross-graph attention and Hebbian weight learning for decision systems.
It is domain-agnostic by design: no SOC, financial, or healthcare vocabulary appears
in any `gae/` file. The only runtime dependency is NumPy.

### Three-Tier Mathematical Stack

```
Tier 1  primitives.py    scaled dot-product attention (Eq. 1)
Tier 2  scoring.py       scoring matrix — action probabilities (Eq. 4)
Tier 3  learning.py      Hebbian weight learning (Eq. 4b / 4c)
Tier 4  (planned v5.5)   entity embeddings (Eq. 5)
Tier 5  (planned v5.5)   cross-graph attention + discovery (Eq. 6–9)
```

---

## 2. Repository Layout

```
graph-attention-engine-v45/
├── gae/
│   ├── __init__.py        Public API — all exports + __version__
│   ├── primitives.py      Tier 1: softmax, scaled_dot_product_attention
│   ├── scoring.py         Tier 2: ScoringResult, score_entity
│   ├── learning.py        Tier 3: LearningState + weight learning
│   ├── convergence.py     Convergence monitoring + diagnostics
│   ├── calibration.py     Domain-configurable hyperparameters
│   ├── contracts.py       PropertySpec, EmbeddingContract, SchemaContract
│   ├── events.py          Frozen event dataclasses (no bus, no async)
│   ├── factors.py         FactorComputer Protocol + assemble_factor_vector
│   └── store.py           JSON persistence for simple LearningState
├── tests/
│   ├── test_primitives.py       (16 tests)
│   ├── test_scoring.py          (18 tests)
│   ├── test_learning.py         (80+ tests)
│   ├── test_contracts.py        (15 tests)
│   ├── test_factors.py          (10 tests)
│   ├── test_convergence.py      (20 tests)
│   ├── test_store.py            (12 tests)
│   ├── test_events.py           (8 tests)
│   ├── test_calibration.py      (10 tests)
│   └── test_generic_domains.py  (40+ tests)
├── docs/
│   └── gae_design_v7.md   Full specification (v7 architecture, 200+ lines)
├── pyproject.toml
├── README.md
├── CLAUDE.md
└── LICENSE
```

---

## 3. Data Flow

```
Raw properties dict
    │
    ▼  assemble_factor_vector(raw, schema)
Factor vector f  — shape (1, n_f)
    │
    ▼  score_entity(f, W, actions, tau)
ScoringResult    — selected_action, confidence, probabilities
    │
System executes action
    │
    ▼  outcome ∈ {+1, −1}
LearningState.update(action_index, outcome, f_original)
    │
Updated W  — shape (n_a, n_f)
    │
    ▼  get_convergence_metrics(state)
{stable, accurate, converged, weight_norm, …}
```

**Requirement R4:** `f_original` must be the factor vector captured at decision time,
not recomputed at outcome time — preserving the graph state that drove the decision.

---

## 4. File Reference

### 4.1 `gae/__init__.py`

Public API surface. Exports every public symbol via an explicit `__all__` tuple.

| Export group | Symbols |
|---|---|
| Calibration | `CalibrationProfile`, `soc_calibration_profile`, `s2p_calibration_profile` |
| Tier 1 | `softmax`, `scaled_dot_product_attention` |
| Tier 2 | `ScoringResult`, `score_entity`, `score_alert` (backward-compat alias) |
| Tier 3 | `LearningState`, `WeightUpdate`, `DimensionMetadata`, `PendingValidation` |
| Constants | `ALPHA`, `EPSILON`, `LAMBDA_NEG`, `W_CLAMP` |
| Convergence | `get_convergence_metrics` |
| Events | `FactorComputedEvent`, `WeightsUpdatedEvent`, `ConvergenceEvent` |
| Contracts | `PropertySpec`, `EmbeddingContract`, `SchemaContract` |
| Factors | `FactorComputer`, `assemble_factor_vector` |
| Persistence | `save_state`, `load_state` |

---

### 4.2 `gae/primitives.py` — Tier 1

Implements **Equation 1** (scaled dot-product attention).

#### `softmax(x, axis=-1) → np.ndarray`
Numerically-stable row-wise softmax using the max-shift trick.
- Input: any ndarray
- Output: same shape, values ∈ (0, 1), summing to 1 along `axis`

#### `scaled_dot_product_attention(Q, K, V, mask=None) → (output, weights)`
| Argument | Shape | Description |
|---|---|---|
| `Q` | `(n, d_k)` | Query matrix |
| `K` | `(m, d_k)` | Key matrix |
| `V` | `(m, d_v)` | Value matrix |
| `mask` | `(n, m)` or None | Additive mask (applied before softmax) |

Returns `(output, weights)` where `output` is `(n, d_v)` and `weights` is `(n, m)`.

Computation: `softmax(Q @ K.T / sqrt(d_k) + mask) @ V`

Shape assertions guard every intermediate result.

---

### 4.3 `gae/scoring.py` — Tier 2

Implements **Equation 4** (scoring matrix).

#### `ScoringResult` (dataclass)
| Field | Type | Description |
|---|---|---|
| `action_probabilities` | `(1, n_a)` | Softmax distribution over actions |
| `selected_action` | `str` | Argmax action name |
| `confidence` | `float` | P(selected_action) ∈ (0, 1] |
| `raw_scores` | `(1, n_a)` | Logits before softmax |
| `factor_vector` | `(1, n_f)` | Original f (R4 preservation) |
| `temperature` | `float` | τ used for this call |

Invariant: `action_probabilities.sum() ≈ 1.0` (tolerance 1e-6).

#### `score_entity(f, W, actions, tau=0.25, profile=None) → ScoringResult`
| Argument | Description |
|---|---|
| `f` | `(1, n_f)` factor vector |
| `W` | `(n_a, n_f)` weight matrix |
| `actions` | List of action name strings (row order of W) |
| `tau` | Temperature — small (0.1) = sharp, large (1.0) = flat |
| `profile` | If provided, `profile.temperature` overrides `tau` |

Computation: `softmax(f @ W.T / tau)` → probabilities → argmax.

`score_alert` is a backward-compatible alias for `score_entity`.

---

### 4.4 `gae/learning.py` — Tier 3

Implements **Equations 4b and 4c** (Hebbian learning + per-factor decay).

#### Module constants
| Constant | Value | Role |
|---|---|---|
| `ALPHA` | 0.02 | Base learning rate |
| `LAMBDA_NEG` | 20.0 | Asymmetric penalty multiplier (20:1) |
| `EPSILON` | 0.001 | Default per-factor decay rate |
| `W_CLAMP` | 5.0 | Weight saturation limit |

#### `DimensionMetadata` (dataclass)
Tracks the lifecycle of a single W column (factor dimension).

| Field | Default | Description |
|---|---|---|
| `factor_name` | — | Column identifier |
| `col_index` | — | Position in W |
| `created_at` | — | `decision_count` at creation |
| `state` | `"provisional"` | `"provisional"` / `"established"` / `"original"` |
| `decay_rate` | 0.01 | Provisional: 10× faster than standard |
| `reinforcement_count` | 0 | Positive updates earned |
| `establishment_threshold` | 50 | Updates needed to become "established" |

#### `PendingValidation` (dataclass)
Holds an autonomous decision pending outcome confirmation (C3 hardening).

| Field | Description |
|---|---|
| `entity_id` | Entity that triggered the decision |
| `action` | Action taken |
| `action_index` | Row in W |
| `factor_vector` | `(1, n_f)` snapshot at decision time |
| `auto_decided_at` | Unix timestamp |
| `validation_window_days` | Days before applying deferred learning (default 14) |

#### `WeightUpdate` (frozen dataclass)
Immutable record of a single weight update — full provenance.

| Field | Description |
|---|---|
| `decision_number` | Monotonic counter |
| `timestamp` | Unix timestamp |
| `action_index` / `action_name` | Which action was reinforced |
| `outcome` | r(t) ∈ {+1, −1} |
| `factor_vector` | `(1, n_f)` at decision time |
| `delta_applied` | `(n_f,)` actual update vector |
| `W_before` / `W_after` | Full weight matrix snapshots |
| `alpha_effective` | α after A1 discounting |
| `confidence_at_decision` | System confidence at the time |

#### `LearningState` (dataclass)
Central mutable container for all Tier 3 state.

| Field | Description |
|---|---|
| `W` | `(n_a, n_f)` weight matrix |
| `n_actions` / `n_factors` | Dimensions of W |
| `factor_names` | Ordered list of factor identifiers |
| `profile` | `CalibrationProfile` supplying all hyperparameters |
| `decision_count` | Completed update count |
| `history` | `list[WeightUpdate]` — all updates |
| `expansion_history` | `list[dict]` — W expansion events |
| `discount_strength` | A1 hardening (0.0 = off) |
| `epsilon_vector` | `(n_f,)` per-factor decay rates (built from profile) |
| `dimension_metadata` | `list[DimensionMetadata]` — A4 tracking |
| `pending_validations` | `list[PendingValidation]` — C3 deferred updates |

**Key methods:**

`build_epsilon_vector() → np.ndarray`
Maps each factor name through `profile.factor_decay_classes` → `profile.decay_class_rates`
to construct the `(n_f,)` ε vector used in Eq. 4c.

`update(action_index, action_name, outcome, f, confidence_at_decision, decision_source) → WeightUpdate | None`

Steps:
1. If `decision_source == "autonomous"` → defer to `pending_validations`, return None (C3)
2. Compute asymmetric δ: `1.0` for positive, `profile.penalty_ratio` for negative
3. A1 discount: if positive and `discount_strength > 0`, reduce α by confidence (floor 5%)
4. **Eq. 4b:** `W[action_index, :] += α_eff * outcome * f.flatten() * δ`
5. **Eq. 4c:** `W *= (1 − epsilon_vector)`
6. Clamp: `W = clip(W, −W_CLAMP, +W_CLAMP)`
7. A4: increment reinforcement count; prune near-zero provisional columns
8. Record and return frozen `WeightUpdate`

`expand_weight_matrix(new_factor_name, init_scale=0.05) → None`
Appends a new column `N(0, init_scale)` to W, adds a provisional `DimensionMetadata`
entry, and extends `epsilon_vector` with provisional decay (0.01).

`_prune_provisional_dimensions(theta_prune=0.01) → None`
Removes any provisional column whose `max|W[:, col]| < theta_prune`.

`process_pending_validations(incident_checker) → int`
Applies deferred learning for expired C3 entries. `incident_checker(entity_id) → bool`
returns True if the decision escalated to an incident (outcome = −1).

---

### 4.5 `gae/convergence.py`

#### Module thresholds
| Constant | Value | Meaning |
|---|---|---|
| `STABILITY_THRESHOLD` | 0.05 | std(W norms over window) < this → stable |
| `ACCURACY_THRESHOLD` | 0.80 | fraction correct > this → accurate |
| `RECENCY_WINDOW` | 20 | decisions used for accuracy estimate |
| `STABILITY_WINDOW` | 10 | decisions used for stability estimate |

#### `get_convergence_metrics(state: LearningState) → dict`
Returns:
| Key | Description |
|---|---|
| `decisions` | Total completed updates |
| `weight_norm` | Frobenius norm of current W |
| `stability` | std of W norms over last `STABILITY_WINDOW` updates |
| `accuracy` | Fraction outcome==+1 over last `RECENCY_WINDOW` updates |
| `converged` | `stability < STABILITY_THRESHOLD and accuracy > ACCURACY_THRESHOLD` |
| `provisional_dimensions` | Count of A4 provisional columns |
| `pending_autonomous` | Count of C3 deferred validations |

Three failure modes detectable from the metrics:

| Mode | Signal | Diagnosis |
|---|---|---|
| FM1 Action Confusion | low `weight_norm` | Actions score nearly identically |
| FM2 Asymmetric Oscillation | `accuracy ≈ 0.5` | Alternating correct/incorrect |
| FM3 Decay Competition | high `stability` | W norm never stabilizes |

---

### 4.6 `gae/calibration.py`

#### `CalibrationProfile` (dataclass)
Replaces module-level hardcoded constants with per-domain configuration.

| Field | Default | Replaces |
|---|---|---|
| `learning_rate` | 0.02 | `ALPHA` |
| `penalty_ratio` | 20.0 | `LAMBDA_NEG` |
| `temperature` | 0.25 | τ in `score_entity` |
| `epsilon_default` | 0.001 | `EPSILON` |
| `discount_strength` | 0.0 | A1 hardening strength |
| `decay_class_rates` | see below | Per-class ε rates |
| `factor_decay_classes` | `{}` | Maps factor name → decay class |
| `extensions` | `{}` | Domain-specific extra params |

Default decay class rates:
```
"permanent"  → 0.0001   (trusted structural factors)
"standard"   → 0.001    (normal factors)
"campaign"   → 0.005    (transient campaign indicators)
"transient"  → 0.02     (very short-lived signals)
```

`validate() → list[str]` — returns warnings if any parameter is outside expected range.

#### Factory functions
`soc_calibration_profile()` — SOC defaults: 20:1 penalty, τ=0.25, 6 named factor decay classes.
`s2p_calibration_profile()` — Source-to-pay defaults: 5:1 penalty, τ=0.4.

---

### 4.7 `gae/contracts.py`

Declarative schema contracts for node properties and embeddings.

#### `PropertySpec` (frozen dataclass)
| Field | Default | Description |
|---|---|---|
| `name` | — | Property identifier |
| `dtype` | `"float"` | `"float"` / `"int"` / `"bool"` |
| `min_value` / `max_value` | None | Optional validation bounds |
| `required` | True | If False, uses `default_value` when absent |
| `default_value` | 0.0 | Substituted for absent optional properties |

`validate_value(value) → bool` — checks value ∈ [min_value, max_value].

#### `EmbeddingContract` (frozen dataclass)
| Field | Default | Description |
|---|---|---|
| `dim` | — | Embedding dimensionality (> 0) |
| `normalized` | False | L2-unit-norm expectation |
| `dtype_name` | `"float32"` | NumPy dtype name |

#### `SchemaContract` (frozen dataclass)
| Field | Description |
|---|---|
| `node_type` | Opaque node type label |
| `properties` | `tuple[PropertySpec, …]` — ordered property specs |
| `embedding` | Optional `EmbeddingContract` |

Properties:
- `factor_dim → int` — count of declared properties (W column count)
- `property_names() → tuple[str, …]` — ordered names

`resolve_value(name, raw) → float` — returns `raw[name]` for required properties,
`default_value` for absent optional ones, raises `KeyError` for missing required ones.

---

### 4.8 `gae/events.py`

Frozen dataclasses for library-emitted events. No event bus, no async transport.

#### `FactorComputedEvent`
| Field | Description |
|---|---|
| `node_id` | Opaque entity identifier |
| `factor_vector` | `(d_f,)` computed vector |
| `factor_names` | Ordered factor names (must match vector length) |

#### `WeightsUpdatedEvent`
| Field | Description |
|---|---|
| `weights_before` / `weights_after` | `(d_f,)` snapshots |
| `delta_norm` | L2 norm of change |
| `step` | Training step index |

#### `ConvergenceEvent`
| Field | Description |
|---|---|
| `step` | Training step at state change |
| `converged` | True if convergence criterion met |
| `delta_norm` | Weight delta norm |
| `threshold` | Convergence threshold used |

---

### 4.9 `gae/factors.py`

#### `FactorComputer` (runtime_checkable Protocol)
Duck-typed protocol for objects that compute raw property dicts. Implementors live
outside this library (e.g., domain adapters). The protocol declares `async compute()`
to accommodate I/O-bound implementors; the library itself never drives asyncio.

```python
async def compute(self, entity_id: str, context: Any = None) -> float:
    """Compute a normalized score ∈ [0, 1] for entity_id."""
```

#### `assemble_factor_vector(raw, schema) → np.ndarray`
Implements **Equation 2**: packs a raw properties dict into a dense `(d_f,)` factor
vector using `schema.resolve_value()` for each property in declaration order.
Raises `KeyError` for missing required properties.

---

### 4.10 `gae/store.py`

JSON persistence for a lightweight `LearningState` (weights + step + metadata).
Not to be confused with `learning.LearningState` — this is a simpler container for
persisting the weight vector between process restarts.

#### `LearningState` (dataclass, store module)
| Field | Description |
|---|---|
| `weights` | `(d_f,)` weight vector |
| `step` | Number of updates completed |
| `converged` | Convergence flag |
| `metadata` | Arbitrary caller dict |

Serialization: `to_dict()` / `from_dict(data)` (classmethod).

#### `save_state(state, path) → None`
Atomic write: creates a temp file in the same directory, writes JSON, then
renames over the target path. Prevents partial-write corruption on crash.

#### `load_state(path) → LearningState`
Reads and validates the JSON file. Raises `FileNotFoundError`, `json.JSONDecodeError`,
or `KeyError`/`ValueError` on malformed data.

---

## 5. Hardening Features

| ID | Name | Mechanism |
|---|---|---|
| **A1** | Confidence-discounted learning | Reduces α when system was already confident: `α_eff = α * max(1 − strength * confidence, 0.05)` |
| **A2** | Per-factor decay | Each factor column decays at its class rate (permanent/standard/campaign/transient) via `epsilon_vector` |
| **A4** | Provisional dimension lifecycle | New W columns enter as "provisional" with 10× decay; auto-pruned if no reinforcement; promoted at 50 updates |
| **C3** | Deferred autonomous validation | Autonomous decisions are queued in `pending_validations`; learning applied only after `process_pending_validations()` |

---

## 6. Design Principles

| Code | Rule |
|---|---|
| P1 | Every `gae/` function docstring references its blog equation |
| P2 | Every matrix operation has a shape assertion |
| P3 | Zero SOC / domain knowledge in `gae/` — no alerts, actions, or domain vocab |
| P4 | Event classes are frozen dataclasses — no mutation after creation |
| P5 | NumPy only — no PyTorch, sklearn, Neo4j, asyncio in `gae/` |
| R4 | Factor vectors are captured at decision time and preserved; never recomputed at outcome time |

---

## 7. Test Suite

| File | Tests | Focus |
|---|---|---|
| `test_primitives.py` | 16 | softmax + attention shape + numerical stability |
| `test_scoring.py` | 19 | score_entity, temperature effects, shape guards |
| `test_learning.py` | 33 | LearningState, Eq. 4b/4c, A1/A2/A4/C3 hardening |
| `test_contracts.py` | 19 | PropertySpec, EmbeddingContract, SchemaContract |
| `test_factors.py` | 10 | assemble_factor_vector, FactorComputer protocol |
| `test_convergence.py` | 16 | get_convergence_metrics, three failure modes |
| `test_store.py` | 14 | save/load round-trips, atomic write |
| `test_events.py` | 11 | Event dataclass validation |
| `test_calibration.py` | 10 | CalibrationProfile validation + factory functions |
| `test_generic_domains.py` | 39 | Supply-chain, financial services, healthcare — proves domain-agnosticism |
| **Total** | **187** | |

Run: `pip install -e ".[dev]"` then `pytest tests/ -v`

---

## 8. Dependencies

```
Runtime:  numpy>=1.24
Dev:      pytest>=7.4, pytest-asyncio>=0.23
```

---

## 9. Planned Extensions (v5.5+)

| File | Tier | Equations |
|---|---|---|
| `gae/embeddings.py` | 4 | Eq. 5 — entity embeddings |
| `gae/attention.py` | 5 | Eq. 6, 7, 9 — cross-graph attention |
| `gae/discovery.py` | 5 | Eq. 8a–8c — discovery extraction |
