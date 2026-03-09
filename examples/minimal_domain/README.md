# GAE Minimal Domain Example — IT Helpdesk Classifier

Shows how to onboard a new domain into the Graph Attention Engine in 5 steps.
No Neo4j, no external dependencies. Requires only `gae` + NumPy.

## Run

```bash
python examples/minimal_domain/run_example.py
```

## The 5 Steps

| Step | What it does |
|------|--------------|
| 1 | Define `CATEGORIES`, `ACTIONS`, `FACTOR_NAMES` for your domain |
| 2 | Build `ProfileScorer` by setting centroids from domain expertise |
| 3 | Run `run_evaluation()` on labelled scenarios; print accuracy + ECE |
| 4 | Call `compute_judgment()` to get rationale for a single decision |
| 5 | Call `run_ablation()` to rank factors by importance |

## Domain: IT Helpdesk

- **Categories**: `hardware`, `software`, `network`
- **Actions**: `auto_resolve`, `assign_tier1`, `assign_tier2`, `escalate`
- **Factors**: `urgency`, `recurrence`, `affected_users`, `asset_value`

## Key Imports

```python
from gae import (
    ProfileScorer,
    EvaluationScenario, run_evaluation,
    compute_judgment,
    run_ablation,
)
```

## Adapting for a New Domain

1. **Change the config** — set `CATEGORIES`, `ACTIONS`, `FACTOR_NAMES`.
2. **Set centroids** — `mu[cat_idx, act_idx, :] = [...]` encodes what each
   action looks like per category. Start from domain expertise; learning
   will refine them over time.
3. **Replace scenarios** — swap the hardcoded `EvaluationScenario` list
   with real labelled data from your system.
4. **Interpret ablation** — factors with large `accuracy_drop` are load-bearing;
   factors with near-zero drop may be redundant.

## Centroid Design Guidelines

- Values in `[0.0, 1.0]`. Normalise raw features before scoring.
- Well-separated centroids (pairwise L2 > 0.5) give high accuracy out of box.
- Use `ProfileScorer.diagnostics()` to check separation after building.
- Call `ProfileScorer.update()` online to refine centroids from feedback.
