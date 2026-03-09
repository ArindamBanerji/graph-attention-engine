# GAE v5.0 User Guide

## 1. Overview

GAE (Graph Attention Engine) is a NumPy-only Python library for profile-based
action scoring and online learning. It implements a scoring architecture where
domain expertise is compiled into centroid profiles, and decisions are made by
measuring which action centroid a factor vector is closest to.

**Core equation** (Eq. 4-final, validated):

```
P(a | f, c) = softmax( -||f - mu[c, a, :]||^2 / tau )
```

where `f` is a factor vector, `c` is a category index, `mu[c, a, :]` is the
centroid for action `a` in category `c`, and `tau` is the temperature.

**Validated numbers** (cross-graph experiments, 25 total):
- 97.89% zero-learning accuracy on centroidal synthetic data (EXP-C1)
- 98.2% accuracy with online learning (EXP-B1)
- tau=0.1 gives ECE=0.036 (V3B); tau=0.25 gives ECE=0.19 — do not use 0.25
- Scaling exponent b=2.11 validated by simulation (V1A)

GAE is **not** an LLM wrapper, a rules engine, or a domain-specific classifier.
It has zero SOC knowledge, zero Neo4j code, and zero async functions.

---

## 2. Installation

```bash
pip install -e .
```

```python
from gae import ProfileScorer, ScoringResult, run_evaluation, compute_judgment, run_ablation
```

---

## 3. Core Concepts

### Profile centroids (`mu`)

`mu` is a 3-D NumPy array of shape `(n_categories, n_actions, n_factors)`.
Each row `mu[c, a, :]` is the centroid representing "what action `a` looks like
in category `c`". Values must be in `[0.0, 1.0]`.

Initialize from domain expertise. Learning refines centroids over time.

### Factor vectors (`f`)

`f` is a 1-D NumPy array of shape `(n_factors,)` with values in `[0.0, 1.0]`.
Each element is a normalized signal (urgency, recurrence, asset value, etc.).
Factor vectors must be normalized before scoring — GAE does not normalize them.

### Scoring

`ProfileScorer.score(f, category_index)` computes squared L2 distances from `f`
to every centroid in category `c`, then applies softmax with temperature `tau`.
The recommended action is `argmax(probabilities)`.

### Learning (centroid pull/push)

`ProfileScorer.update(f, category_index, action_index, correct)` implements
Eq. 4b-final. Correct: pull winning centroid toward `f`. Incorrect: push all
centroids away from `f`. All values clipped to `[0.0, 1.0]` after each update
(V2 — prevents centroid escape under adversarial inputs).

### Temperature (`tau`)

Controls softmax sharpness. Validated default: `tau=0.1` (ECE=0.036, V3B).
Do not use 0.25 (ECE=0.19). Do not change without re-running calibration.

---

## 4. Quick Start

```python
import numpy as np
from gae import ProfileScorer, EvaluationScenario, run_evaluation

# Domain config
CATEGORIES   = ["hardware", "software", "network"]
ACTIONS      = ["auto_resolve", "assign_tier1", "assign_tier2", "escalate"]
FACTOR_NAMES = ["urgency", "recurrence", "affected_users", "asset_value"]

cat_idx = {name: i for i, name in enumerate(CATEGORIES)}
act_idx = {name: i for i, name in enumerate(ACTIONS)}

# Build centroid tensor — shape (3 categories, 4 actions, 4 factors)
mu = np.full((3, 4, 4), 0.3, dtype=np.float64)

# Set domain-expert centroids
mu[cat_idx["hardware"], act_idx["escalate"],    :] = [0.9, 0.8, 0.7, 0.9]
mu[cat_idx["hardware"], act_idx["auto_resolve"],:] = [0.1, 0.1, 0.1, 0.2]
mu[cat_idx["network"],  act_idx["escalate"],    :] = [0.8, 0.7, 0.9, 0.6]

scorer = ProfileScorer(mu=mu, actions=ACTIONS)

# Score a single factor vector
f = np.array([0.9, 0.8, 0.7, 0.9])           # high-severity hardware ticket
result = scorer.score(f, category_index=cat_idx["hardware"])

print(result.action_name)    # "escalate"
print(result.confidence)     # e.g. 0.993
print(result.probabilities)  # shape (4,), sums to 1.0

# Update from verified outcome
scorer.update(f, category_index=cat_idx["hardware"],
              action_index=result.action_index, correct=True)
```

---

## 5. API Reference

### 5.1 ProfileScorer

```python
ProfileScorer(
    mu: np.ndarray,           # shape (n_categories, n_actions, n_factors)
    actions: list[str],       # len == mu.shape[1]
    kernel: KernelType = KernelType.L2,
    profile: CalibrationProfile | None = None,
)
```

**Methods:** `score(f, category_index) -> ScoringResult`,
`update(f, category_index, action_index, correct) -> None`,
`diagnostics() -> dict`, `set_covariance(cov_inv)` (Mahalanobis only).

**Factory:** `build_profile_scorer(categories, actions, centroids, n_factors,
kernel=KernelType.L2)` — builds from nested dict `{cat: {act: [values]}}`.

**Kernel types** (`KernelType`): `L2` (validated default, 97.89%), `COSINE`
(96.4%), `MAHALANOBIS` (requires `set_covariance()`), `DOT` (61%, warns).

### 5.2 ScoringResult

```python
@dataclass
class ScoringResult:
    action_index:  int         # argmax(probabilities)
    action_name:   str         # actions[action_index]
    probabilities: np.ndarray  # shape (n_actions,), sums to 1.0
    distances:     np.ndarray  # shape (n_actions,), raw L2 distances
    confidence:    float       # max(probabilities)
```

Note: the field is `.probabilities`, not `.scores`.

### 5.3 Oracle

**`OracleProvider`** — `runtime_checkable` Protocol. Implement
`.query(f, category_index, taken_action_index) -> OracleResult`.

**`GTAlignedOracle(mu, actions)`** — deterministic GT via centroid
nearest-neighbor (`argmin ||f - mu[c,a,:]||^2`). `confidence=1.0`.

**`BernoulliOracle(n_actions, actions, correct_rate, seed)`** — stochastic GT.
Returns `correct=True` with probability `correct_rate`. RANDOM ablation baseline.

**`OracleResult`** fields: `correct`, `gt_action_idx`, `gt_action_name`,
`confidence`.

### 5.4 Evaluation

**`EvaluationScenario`** required fields: `scenario_id`, `domain`, `category`,
`category_index`, `factors` (shape `(n_factors,)`), `expected_action`,
`expected_action_index`. Optional: `description`, `confidence_tier`,
`expected_dominant_factors`, `learning_prerequisite`.

```python
run_evaluation(
    profile_scorer,
    scenarios: list[EvaluationScenario],
    oracle=None,      # if None, uses expected_action_index as GT
    learn: bool = False,  # call update() after each scenario if True
) -> EvaluationReport
```

**`EvaluationReport`** fields: `accuracy`, `by_category`, `precision_per_action`,
`recall_per_action`, `ece`, `scenario_results`, `n_scenarios`, `n_correct`.

`compute_ece(confidences, correct_flags, n_bins=10) -> float` — Expected
Calibration Error. Returns 0.0 for fewer than 2 scenarios.

### 5.5 Judgment

```python
compute_judgment(
    scoring_result,           # ScoringResult from ProfileScorer.score()
    f: np.ndarray,            # factor vector used for scoring
    mu: np.ndarray,           # full centroid array (n_cat, n_act, n_fac)
    category_index: int,
    factor_names: list[str],
    actions: list[str] | None = None,
) -> JudgmentResult
```

**`JudgmentResult`** fields:

| Field | Type | Description |
|-------|------|-------------|
| `action` | `str` | Recommended action name |
| `confidence` | `float` | Confidence in [0, 1] |
| `confidence_tier` | `str` | `"high"` / `"medium"` / `"discovery"` |
| `dominant_factors` | `list[str]` | Top-3 factor names by contribution |
| `factor_contributions` | `dict[str, float]` | Per-factor proximity score [0, 1] |
| `rationale` | `str` | One-sentence plain-English explanation |
| `action_scores` | `dict[str, float]` | All action probabilities |
| `auto_approvable` | `bool` | `True` if high confidence and not "escalate" |

**Confidence tiers**: `high` >= 0.80, `medium` >= 0.50, `discovery` < 0.50.

Constants: `CONFIDENCE_HIGH = 0.80`, `CONFIDENCE_MEDIUM = 0.50`.

### 5.6 Ablation

```python
run_ablation(
    profile_scorer,
    scenarios: list[EvaluationScenario],
    factor_names: list[str],       # len must equal n_factors
) -> AblationReport
```

Raises `ValueError` if `scenarios` is empty or `len(factor_names) != n_factors`.

**`AblationReport`** fields: `baseline_accuracy`, `results` (sorted by
`accuracy_drop` descending), `most_important`, `least_important`, `n_factors`,
`n_scenarios`. **`AblationResult`** fields: `factor_index`, `factor_name`,
`baseline_accuracy`, `ablated_accuracy`, `accuracy_drop`, `importance_rank`.

`accuracy_drop > 0`: factor helps. `accuracy_drop <= 0`: factor is noise or
redundant at the current centroid configuration.

---

## 6. Onboarding a New Domain

1. **Define config**: choose `CATEGORIES`, `ACTIONS`, `FACTOR_NAMES` for your
   domain. Keep factor values normalizable to `[0.0, 1.0]`.

2. **Initialize centroids**: set `mu[cat, act, :]` from domain expertise.
   Well-separated centroids (pairwise L2 > 0.5) give high accuracy out of the
   box. Neutral default for unknown pairs: 0.3 or 0.5.

3. **Build scorer**: `scorer = ProfileScorer(mu=mu, actions=ACTIONS)`.
   Call `scorer.diagnostics()` to verify centroid separation.

4. **Validate**: build `EvaluationScenario` objects from labelled examples,
   call `run_evaluation(scorer, scenarios)`. Target accuracy >= 90% before
   deploying. Run `run_ablation()` to confirm factor relevance.

5. **Deploy**: call `scorer.score(f, category_index)` in your inference path.
   Read `result.action_name` and `result.confidence`.

6. **Compound**: call `scorer.update(f, category_index, action_index, correct)`
   on every verified decision. Use `GTAlignedOracle` or human feedback.

See `examples/minimal_domain/run_example.py` for a complete working example.

---

## 7. Key Numbers and Constraints

| Parameter | Value | Source |
|-----------|-------|--------|
| Default tau | 0.1 | V3B — ECE=0.036. Do not use 0.25 (ECE=0.19). |
| Centroid clip range | [0.0, 1.0] | V2 — prevents escape under adversarial updates |
| Factor vector range | [0.0, 1.0] | Required. GAE does not normalize inputs. |
| Scaling exponent b | 2.11 | V1A simulation (EXP-G1 pending for t^gamma term) |
| Default eta | 0.05 | Correct pull learning rate |
| Default eta_neg | 0.05 | Incorrect push learning rate |
| Deprecated | score_entity, score_alert | TD-029 — remove in v6.0 |

---

## 8. What Is Not In This Library

- **No Neo4j queries.** Graph persistence lives in `ci-platform`.
- **No LLM calls.** GAE is deterministic NumPy; no model inference.
- **No SOC-specific logic.** Alert types, MITRE mappings, and action labels
  are defined by the consuming application, not by GAE.
- **No async functions.** GAE is synchronous. The `FactorComputer` Protocol
  declares `async compute()` for implementors only; GAE itself has no async code.
