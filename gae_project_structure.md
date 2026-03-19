# Graph Attention Engine — Project Structure

**Version:** v0.5.0 · **Branch:** v5.0-dev · **License:** Apache 2.0
**Math reference:** https://www.dakshineshwari.net/post/cross-graph-attention-mathematical-foundation-with-experimental-validation

---

## 1. Overview

The Graph Attention Engine (GAE) is a standalone, pip-installable Python library
implementing centroid-proximity scoring and Hebbian weight learning for decision
systems. It is domain-agnostic by design: no SOC, financial, or healthcare
vocabulary appears in any `gae/` file. The only runtime dependency is NumPy.

### Three-Tier Mathematical Stack

```
Tier 1  primitives.py      scaled dot-product attention (Eq. 1)
Tier 2  profile_scorer.py  centroid-proximity scoring (Eq. 4-final, 4b-final)
Tier 3  learning.py        Hebbian weight learning (Eq. 4b / 4c)
Tier 4  (planned v5.5)     entity embeddings (Eq. 5)
Tier 5  (planned v5.5)     cross-graph attention + discovery (Eq. 6–9)
```

`scoring.py` (matrix dot-product scorer) is **deprecated** (TD-029, remove in v6.0).
`profile_scorer.py` is the validated v5.0 primary scoring path.

---

## 2. Repository Layout

```
graph-attention-engine-v50/
├── gae/
│   ├── __init__.py          Public API — all exports + __version__
│   ├── primitives.py        Tier 1: softmax, scaled_dot_product_attention
│   ├── profile_scorer.py    Tier 2: ProfileScorer, CentroidUpdate, ScoringResult
│   ├── scoring.py           Tier 2 (deprecated TD-029): score_entity, score_alert
│   ├── learning.py          Tier 3: LearningState + weight learning
│   ├── oracle.py            GT outcome providers: GTAlignedOracle, BernoulliOracle
│   ├── evaluation.py        Evaluation framework: run_evaluation, EvaluationReport
│   ├── judgment.py          Decision rationale: compute_judgment, JudgmentResult
│   ├── ablation.py          Factor importance: run_ablation, AblationReport
│   ├── bootstrap.py         Synthetic calibration: bootstrap_calibration
│   ├── convergence.py       Convergence monitoring + diagnostics
│   ├── calibration.py       Domain-configurable hyperparameters
│   ├── contracts.py         PropertySpec, EmbeddingContract, SchemaContract
│   ├── events.py            Frozen event dataclasses (no bus, no async)
│   ├── factors.py           FactorComputer Protocol + assemble_factor_vector
│   └── store.py             JSON persistence for LearningState
├── tests/
│   ├── test_primitives.py        (16 tests)
│   ├── test_profile_scorer.py    (15 tests)
│   ├── test_scoring.py           (19 tests — deprecated scorer)
│   ├── test_learning.py          (37 tests)
│   ├── test_oracle.py            (8 tests)
│   ├── test_evaluation.py        (8 tests)
│   ├── test_judgment.py          (8 tests)
│   ├── test_ablation.py          (8 tests)
│   ├── test_bootstrap.py         (8 tests)
│   ├── test_contracts.py         (19 tests)
│   ├── test_factors.py           (10 tests)
│   ├── test_convergence.py       (16 tests)
│   ├── test_store.py             (14 tests)
│   ├── test_events.py            (11 tests)
│   ├── test_calibration.py       (10 tests)
│   └── test_generic_domains.py   (39 tests)
├── docs/
│   └── gae_design_v5.md    Full specification (v5.0 architecture)
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
Factor vector f  — shape (n_factors,)
    │
    ▼  ProfileScorer.score(f, category_index)
ScoringResult    — action_index, action_name, probabilities, confidence
    │
    ▼  compute_judgment(scoring_result, f, mu, category_index, factor_names)
JudgmentResult   — rationale, dominant_factors, auto_approvable
    │
System executes action
    │
    ▼  OracleProvider.query(f, category_index, taken_action_index)
OracleResult     — correct, gt_action_idx
    │
    ▼  ProfileScorer.update(f, category_index, action_index, correct)
CentroidUpdate   — centroid_delta_norm, category_index, action_index
```

---

## 4. File Reference

### 4.1 `gae/__init__.py`

Public API surface. Exports every public symbol via an explicit `__all__` list.

| Export group | Symbols |
|---|---|
| Core scoring | `CentroidUpdate`, `ProfileScorer`, `ScoringResult`, `KernelType`, `build_profile_scorer` |
| Oracle | `OracleProvider`, `OracleResult`, `GTAlignedOracle`, `BernoulliOracle` |
| Evaluation | `EvaluationScenario`, `EvaluationReport`, `compute_ece`, `run_evaluation` |
| Judgment | `JudgmentResult`, `compute_judgment`, `CONFIDENCE_HIGH`, `CONFIDENCE_MEDIUM` |
| Ablation | `AblationResult`, `AblationReport`, `run_ablation` |
| Bootstrap | `BootstrapResult`, `bootstrap_calibration` |
| Calibration | `CalibrationProfile`, `soc_calibration_profile`, `s2p_calibration_profile` |
| Tier 1 | `softmax`, `scaled_dot_product_attention` |
| Tier 3 | `LearningState`, `WeightUpdate`, `DimensionMetadata`, `PendingValidation` |
| Constants | `ALPHA`, `EPSILON`, `LAMBDA_NEG`, `W_CLAMP` |
| Convergence | `get_convergence_metrics` |
| Events | `FactorComputedEvent`, `WeightsUpdatedEvent`, `ConvergenceEvent` |
| Contracts | `PropertySpec`, `EmbeddingContract`, `SchemaContract` |
| Factors | `FactorComputer`, `assemble_factor_vector` |
| Persistence | `save_state`, `load_state` |
| Deprecated (TD-029) | `score_entity`, `score_alert`, `score_with_profile`, `ProfileScoringResult` |

---

### 4.2 `gae/primitives.py` — Tier 1

Implements **Equation 1** (scaled dot-product attention).

#### `softmax(x, axis=-1) → np.ndarray`
Numerically-stable softmax using the max-shift trick.
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

---

### 4.3 `gae/profile_scorer.py` — Tier 2 (primary)

Implements **Eq. 4-final** (centroid-proximity scoring) and **Eq. 4b-final**
(centroid pull/push learning). Experimentally validated: 97.89% oracle accuracy
(EXP-C1), 98.2% with learning (EXP-B1), ECE=0.036 at τ=0.1 (V3B).

#### `CentroidUpdate` (dataclass)
Return value from `ProfileScorer.update()`. Captures centroid movement magnitude.

| Field | Description |
|---|---|
| `centroid_delta_norm` | ‖η·(f − μ[c,a,:])‖₂ computed before in-place update |
| `category_index` | Category index c |
| `action_index` | Action index a |
| `category_name` | Human-readable category name |
| `action_name` | Human-readable action name |
| `decision_count` | Total decisions in this ProfileScorer instance |

`centroid_delta_norm=0.0` when the scorer is frozen.

#### `KernelType` (Enum)
| Value | Accuracy | Notes |
|---|---|---|
| `L2` | 97.89% | Default — validated |
| `MAHALANOBIS` | 97.7% | Requires per-category covariance (`set_covariance()`) |
| `COSINE` | 96.4% | Useful on unit-norm factors |
| `DOT` | 61.0% | Warning emitted at init; do not use on [0,1] factors |

#### `ScoringResult` (dataclass)
| Field | Description |
|---|---|
| `action_index` | Argmax action index |
| `action_name` | Argmax action name |
| `probabilities` | `(n_actions,)` softmax distribution |
| `distances` | `(n_actions,)` raw kernel distances |
| `confidence` | `probabilities[action_index]` |

#### `ProfileScorer`
Central scoring class. Holds centroid tensor `mu` of shape `(n_categories, n_actions, n_factors)`.

**Key methods:**

`score(f, category_index) → ScoringResult`
Eq. 4-final: `P(a|f,c) = softmax(−dist(f, μ[c,a,:]) / τ)`

`update(f, category_index, action_index, correct) → CentroidUpdate`
Eq. 4b-final pull/push update, clipped to [0.0, 1.0]:
- Correct: `μ[c,a,:] += η_eff * (f − μ[c,a,:])`
- Incorrect: `μ[c,b,:] −= η_neg_eff * (f − μ[c,b,:])` for all b

`freeze() / unfreeze()` — disable/re-enable centroid learning.

`set_covariance(cov_inv)` — supply per-category covariance for Mahalanobis kernel.

`diagnostics() → dict` — per-category centroid separation metrics.

`ProfileScorer.init_from_config(config_dict, actions) → ProfileScorer` — build from nested centroid dict.

#### `build_profile_scorer(categories, actions, centroids, n_factors, ...) → ProfileScorer`
Convenience factory for building from explicit argument lists.

---

### 4.4 `gae/oracle.py`

Ground-truth outcome providers used by the evaluation and bootstrap pipelines.
Oracles only answer "was this action correct?" — they do not score.

#### `OracleResult` (dataclass)
| Field | Description |
|---|---|
| `correct` | Whether taken action matched GT |
| `gt_action_idx` | Ground-truth action index |
| `gt_action_name` | Ground-truth action name |
| `confidence` | Oracle label confidence ∈ [0, 1] |

#### `OracleProvider` (runtime_checkable Protocol)
```python
def query(self, f, category_index, taken_action_index) -> OracleResult: ...
```

#### `GTAlignedOracle`
GT = argmin squared-L2 distance from f to centroids in category c.
"Perfect labeler" — knows true profile structure. `confidence=1.0` always.
`GTAlignedOracle.from_profile_scorer(scorer)` builds from an existing scorer.

#### `BernoulliOracle`
Returns `correct=True` with fixed probability `correct_rate` (default 0.25 = random
baseline for 4-action problems). Used in ablation RANDOM baseline (GAE-ABL-1).

---

### 4.5 `gae/evaluation.py`

Structured evaluation of `ProfileScorer` against known test cases.

#### `EvaluationScenario` (dataclass)
| Field | Description |
|---|---|
| `scenario_id` | Unique identifier |
| `domain` | Domain label (no SOC semantics) |
| `category` / `category_index` | Category name and index |
| `factors` | `(n_factors,)` factor vector |
| `expected_action` / `expected_action_index` | Ground truth (used when no oracle) |
| `expected_dominant_factors` | Optional expected top factors |
| `confidence_tier` | Expected tier: `"high"` / `"medium"` / `"low"` |
| `description` | Human-readable description |
| `learning_prerequisite` | Scenario ID that must run first |

#### `EvaluationReport` (dataclass)
| Field | Description |
|---|---|
| `accuracy` | Overall fraction correct (4 d.p.) |
| `by_category` | `{category: accuracy}` per category |
| `precision_per_action` | TP / (TP+FP) per action |
| `recall_per_action` | TP / (TP+FN) per action |
| `ece` | Expected Calibration Error |
| `scenario_results` | Per-scenario detail dicts |
| `n_scenarios` / `n_correct` | Totals |

#### `compute_ece(confidences, correct_flags, n_bins=10) → float`
Binned ECE: Σ `(|bin| / n) * |mean_confidence(bin) − mean_accuracy(bin)|`.
Returns 0.0 for fewer than 2 scenarios.

#### `run_evaluation(profile_scorer, scenarios, oracle=None, learn=False) → EvaluationReport`
Evaluates scorer against scenario list. If `oracle` provided, uses oracle GT
instead of `scenario.expected_action_index`. If `learn=True`, calls
`profile_scorer.update()` after each scenario.

---

### 4.6 `gae/judgment.py`

Translates a `ScoringResult` into a human-readable decision rationale.

#### Constants
| Constant | Value | Meaning |
|---|---|---|
| `CONFIDENCE_HIGH` | 0.80 | ≥ this → "high" tier |
| `CONFIDENCE_MEDIUM` | 0.50 | ≥ this → "medium" tier; below → "discovery" |

#### `JudgmentResult` (dataclass)
| Field | Description |
|---|---|
| `action` | Recommended action name |
| `confidence` | Confidence ∈ [0, 1] |
| `confidence_tier` | `"high"` / `"medium"` / `"discovery"` |
| `dominant_factors` | Top-3 factor names by proximity to winning centroid |
| `factor_contributions` | `{factor_name: contribution}` ∈ [0, 1] |
| `rationale` | One-sentence plain-English explanation |
| `action_scores` | `{action_name: probability}` for all actions |
| `auto_approvable` | `True` if tier == "high" and action != "escalate" |

Factor contribution: `clip(1.0 − |f[i] − μ_winner[i]|, 0, 1)`.

#### `compute_judgment(scoring_result, f, mu, category_index, factor_names, actions=None) → JudgmentResult`
Produces a full `JudgmentResult` from a `ScoringResult` + centroid array.

---

### 4.7 `gae/ablation.py`

Leave-one-out factor importance measurement.

#### `AblationResult` (dataclass)
| Field | Description |
|---|---|
| `factor_index` / `factor_name` | Which factor was ablated |
| `baseline_accuracy` | Accuracy with all factors |
| `ablated_accuracy` | Accuracy with this factor zeroed |
| `accuracy_drop` | `baseline − ablated`. Positive = factor helps. |
| `importance_rank` | 1 = most important |

#### `AblationReport` (dataclass)
`baseline_accuracy`, `results` (sorted by drop descending), `most_important`,
`least_important`, `n_factors`, `n_scenarios`.

#### `run_ablation(profile_scorer, scenarios, factor_names) → AblationReport`
For each factor i: zero that factor across all scenarios, re-evaluate, record
`accuracy_drop`. Results sorted descending; `importance_rank` assigned 1…n.
ProfileScorer and scenarios are NOT modified.

---

### 4.8 `gae/bootstrap.py`

Synthetic calibration using the scorer's own prior as oracle. Warm-starts
a ProfileScorer before live deployment.

#### `BootstrapResult` (dataclass)
| Field | Description |
|---|---|
| `n_decisions` | Total synthetic updates made |
| `n_rounds` | Rounds completed |
| `converged` | `final_drift < convergence_tol` |
| `final_drift` | Mean L2 drift of `mu` from prior after all rounds |
| `decisions_per_category` | `{category_name: count}` |
| `metadata` | `seed`, `sigma`, `n_rounds_requested`, `timestamp` |

#### `bootstrap_calibration(scorer, categories, n_rounds=10, samples_per_action=5, sigma=0.08, convergence_tol=0.01, seed=42) → BootstrapResult`
Algorithm: capture `prior_mu`; for each round × category × action, sample
`f ~ clip(N(prior_mu[c,a,:], sigma), 0, 1)`, find oracle action via nearest prior
centroid, call `scorer.update(f, c, oracle_a, correct=True)`.
Scorer mutated in place.

---

### 4.9 `gae/learning.py` — Tier 3

Implements **Equations 4b and 4c** (Hebbian learning + per-factor decay).

#### Module constants
| Constant | Value | Role |
|---|---|---|
| `ALPHA` | 0.02 | Base learning rate |
| `LAMBDA_NEG` | 20.0 | Asymmetric penalty multiplier (20:1) |
| `EPSILON` | 0.001 | Default per-factor decay rate |
| `W_CLAMP` | 5.0 | Weight saturation limit |

#### `LearningState` (dataclass)
Central mutable container for all Tier 3 state. Holds weight matrix `W`
of shape `(n_actions, n_factors)`, plus history, dimension metadata, and
pending autonomous validations.

**Key methods:**

`update(action_index, action_name, outcome, f, confidence_at_decision, decision_source) → WeightUpdate | None`
Applies Eq. 4b + 4c. If `decision_source == "autonomous"`, defers to C3.

`expand_weight_matrix(new_factor_name, init_scale=0.05)` — appends provisional column.

`process_pending_validations(incident_checker) → int` — applies deferred C3 learning.

#### `WeightUpdate` (frozen dataclass)
Full provenance record of one weight update: W_before, W_after, delta_applied,
alpha_effective, confidence_at_decision, etc.

---

### 4.10 `gae/convergence.py`

#### `get_convergence_metrics(state: LearningState) → dict`
Returns `decisions`, `weight_norm`, `stability`, `accuracy`, `converged`,
`provisional_dimensions`, `pending_autonomous`.

Three detectable failure modes: FM1 Action Confusion (low weight_norm),
FM2 Asymmetric Oscillation (accuracy ≈ 0.5), FM3 Decay Competition (high stability).

---

### 4.11 `gae/calibration.py`

#### `CalibrationProfile` (dataclass)
Per-domain hyperparameter configuration replacing module-level constants.
Fields: `learning_rate`, `penalty_ratio`, `temperature`, `epsilon_default`,
`discount_strength`, `decay_class_rates`, `factor_decay_classes`, `extensions`.

`validate()` — returns list of range warnings.

`soc_calibration_profile()` — 20:1 penalty, τ=0.25.
`s2p_calibration_profile()` — 5:1 penalty, τ=0.4.

---

### 4.12 `gae/contracts.py`

Declarative schema contracts for node properties and embeddings.
`PropertySpec` (field bounds, required/optional), `EmbeddingContract` (dim,
normalized, dtype), `SchemaContract` (node_type, properties tuple, embedding).

`assemble_factor_vector` (in `factors.py`) uses `SchemaContract.resolve_value()`.

---

### 4.13 `gae/events.py`

Frozen dataclasses for library-emitted events. No event bus, no async transport.

`FactorComputedEvent` — node_id, factor_vector, factor_names.
`WeightsUpdatedEvent` — weights_before, weights_after, delta_norm, step.
`ConvergenceEvent` — step, converged, delta_norm, threshold.

---

### 4.14 `gae/factors.py`

`FactorComputer` (runtime_checkable Protocol) — duck-typed protocol for objects
that compute normalized scores ∈ [0, 1]. Protocol declares `async compute()`
for I/O-bound implementors; the library never drives asyncio itself.

`assemble_factor_vector(raw, schema) → np.ndarray` — packs a raw properties dict
into a dense `(d_f,)` factor vector using `schema.resolve_value()`.

---

### 4.15 `gae/store.py`

Atomic JSON persistence for weight state between process restarts.

`save_state(state, path)` — temp-file + rename (crash-safe).
`load_state(path) → LearningState` — raises on malformed data.

---

## 5. Hardening Features

| ID | Name | Mechanism |
|---|---|---|
| **A1** | Confidence-discounted learning | Reduces α when system was already confident |
| **A2** | Per-factor decay | Each W column decays at its class rate via `epsilon_vector` |
| **A4** | Provisional dimension lifecycle | New W columns enter provisional with 10× decay; auto-pruned if unreinforced |
| **C3** | Deferred autonomous validation | Autonomous decisions queued; learning applied after `process_pending_validations()` |

---

## 6. Design Principles

| Code | Rule |
|---|---|
| P1 | Every `gae/` function docstring references its blog equation |
| P2 | Every matrix operation has a shape assertion |
| P3 | Zero SOC / domain knowledge in `gae/` — no alerts, actions, or domain vocab |
| P4 | Event classes are frozen dataclasses — no mutation after creation |
| P5 | NumPy only — no PyTorch, sklearn, Neo4j, asyncio in `gae/` |
| R4 | Factor vectors captured at decision time; never recomputed at outcome time |

---

## 7. Test Suite

| File | Tests | Focus |
|---|---|---|
| `test_primitives.py` | 16 | softmax + attention shape + numerical stability |
| `test_profile_scorer.py` | 15 | ProfileScorer scoring, learning, kernels, CentroidUpdate |
| `test_scoring.py` | 19 | score_entity (deprecated), temperature, shape guards |
| `test_learning.py` | 37 | LearningState, Eq. 4b/4c, A1/A2/A4/C3 hardening |
| `test_oracle.py` | 8 | GTAlignedOracle, BernoulliOracle, OracleProvider protocol |
| `test_evaluation.py` | 8 | run_evaluation, ECE, per-category/action metrics |
| `test_judgment.py` | 8 | JudgmentResult, confidence tiers, auto_approvable |
| `test_ablation.py` | 8 | run_ablation, importance ranking, leave-one-out |
| `test_bootstrap.py` | 8 | bootstrap_calibration, convergence, determinism |
| `test_contracts.py` | 19 | PropertySpec, EmbeddingContract, SchemaContract |
| `test_factors.py` | 10 | assemble_factor_vector, FactorComputer protocol |
| `test_convergence.py` | 16 | get_convergence_metrics, three failure modes |
| `test_store.py` | 14 | save/load round-trips, atomic write |
| `test_events.py` | 11 | Event dataclass validation |
| `test_calibration.py` | 10 | CalibrationProfile validation + factory functions |
| `test_generic_domains.py` | 39 | Supply-chain, financial services, healthcare — domain-agnosticism |
| **Total** | **246** | |

Run: `pip install -e ".[dev]"` then `pytest tests/ -v`

---

## 8. Dependencies

```
Runtime:  numpy>=1.24
Dev:      pytest>=7.4, pytest-asyncio>=0.23
```

---

## 9. Experimental Validation (v5.0)

| Experiment | Result | Notes |
|---|---|---|
| EXP-C1 | 97.89% accuracy | L2 kernel, zero-learning, centroidal synthetic data |
| EXP-B1 | 98.2% accuracy | L2 kernel with online learning |
| V3B | ECE=0.036 | τ=0.1 (τ=0.25 was wrong: ECE=0.19) |
| V2 | Clipping required | Centroid escape at dec 6–12 without [0,1] clip |
| FX-1-CORRECTED | L2 > global Mahalanobis | ECE 0.0255 vs 0.2750 — global cov harmful on multi-category |
| EXP-E1 | L2 97.9%, Cosine 96.4%, DOT 61.0% | Kernel generalization comparison |

---

## 10. Planned Extensions (v5.5+)

| File | Tier | Equations |
|---|---|---|
| `gae/embeddings.py` | 4 | Eq. 5 — entity embeddings |
| `gae/attention.py` | 5 | Eq. 6, 7, 9 — cross-graph attention |
| `gae/discovery.py` | 5 | Eq. 8a–8c — discovery extraction |
