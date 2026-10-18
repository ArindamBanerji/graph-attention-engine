# Graph Attention Engine (GAE)

Graph Attention Engine is a standalone, NumPy-only scoring engine that implements Cross-Graph Attention for domain-independent decision systems. It is pure Python and requires no database, network access, or event loop. GAE is licensed under Apache 2.0.

## Install

Install the published package:

```bash
pip install graph-attention-engine
```

For editable development installs:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Scaled dot-product attention

```python
import numpy as np
from gae import scaled_dot_product_attention

rng = np.random.default_rng(7)
Q = rng.normal(size=(4, 8))
K = rng.normal(size=(6, 8))
V = rng.normal(size=(6, 16))

out, weights = scaled_dot_product_attention(Q, K, V)

assert out.shape == (4, 16)
assert weights.shape == (4, 6)
```

### Profile scoring

`ProfileScorer` compares a factor vector with category-specific action centroids. The convenience factory accepts named categories and actions:

```python
import numpy as np
from gae import ProfileScorer, build_profile_scorer

categories = ["routine", "critical"]
actions = ["accept", "review", "escalate"]

# Shape: (2 categories, 3 actions, 4 factors)
mu = np.array(
    [
        [
            [0.2, 0.2, 0.1, 0.2],
            [0.5, 0.4, 0.5, 0.4],
            [0.8, 0.8, 0.9, 0.8],
        ],
        [
            [0.1, 0.2, 0.2, 0.1],
            [0.5, 0.6, 0.5, 0.6],
            [0.9, 0.9, 0.8, 0.9],
        ],
    ],
    dtype=np.float64,
)

centroids = {
    category: {
        action: mu[category_index, action_index].tolist()
        for action_index, action in enumerate(actions)
    }
    for category_index, category in enumerate(categories)
}

scorer: ProfileScorer = build_profile_scorer(
    categories=categories,
    actions=actions,
    centroids=centroids,
    n_factors=4,
)
result = scorer.score(np.array([0.85, 0.90, 0.80, 0.90]), category_index=1)

print(result.action_index)
print(result.confidence)
print(result.probabilities)
```

The returned `ScoringResult` contains the recommended `action_index`, `action_name`, calibrated `confidence`, full `probabilities`, distances, entropy, and confidence gap.

## Core API

### Core Scoring

| API | Description |
|---|---|
| `ProfileScorer` | Scores factor vectors against category and action centroids. |
| `ScoringResult` | Contains the recommendation, confidence, probabilities, and diagnostics. |
| `KernelType` | Selects the distance kernel used for scoring. |
| `build_profile_scorer` | Builds a scorer from named categories, actions, and centroid mappings. |

### Oracle

| API | Description |
|---|---|
| `OracleProvider` | Protocol for producing independent ground-truth outcomes. |
| `GTAlignedOracle` | Returns outcomes aligned with supplied ground truth. |
| `BernoulliOracle` | Samples deterministic-seeded Bernoulli outcomes. |

### Evaluation

| API | Description |
|---|---|
| `EvaluationScenario` | Defines one scored evaluation case and expected action. |
| `compute_ece` | Computes expected calibration error. |
| `run_evaluation` | Evaluates a scorer across a scenario collection. |

### Judgment

| API | Description |
|---|---|
| `compute_judgment` | Converts a scoring result into an explained judgment. |
| `CONFIDENCE_HIGH` | Threshold for high-confidence judgments. |
| `CONFIDENCE_MEDIUM` | Threshold for medium-confidence judgments. |

### Calibration

| API | Description |
|---|---|
| `CalibrationProfile` | Holds scoring and learning calibration parameters. |
| `compute_theta_min` | Computes the deployment-aware conservation threshold. |

### Primitives

| API | Description |
|---|---|
| `scaled_dot_product_attention` | Computes scaled dot-product attention and its weights. |
| `softmax` | Computes a numerically stable probability distribution. |

### Learning

| API | Description |
|---|---|
| `LearningState` | Tracks centroids and learning metadata. |
| `WeightUpdate` | Describes a validated learning update. |

### Convergence

| API | Description |
|---|---|
| `get_convergence_metrics` | Reports centroid movement and convergence diagnostics. |

### Kernels

| API | Description |
|---|---|
| `L2Kernel` | Computes standard squared Euclidean distance. |
| `DiagonalKernel` | Applies per-factor diagonal weights. |
| `CovarianceEstimator` | Estimates covariance for adaptive distance calculations. |
| `KernelSelector` | Recommends a kernel from observed data properties. |

### Persistence

| API | Description |
|---|---|
| `save_state` | Saves learning state to JSON. |
| `load_state` | Loads learning state from JSON. |

The complete public export list is available in [`gae/__init__.py`](gae/__init__.py).

## Conservation Law

GAE checks the constraint `alpha * q * V >= theta_min`, combining category coverage (`alpha`), observed quality (`q`), and verified volume (`V`). Meeting the threshold provides the operational guarantee that evidence is sufficient to maintain accuracy above the configured boundary; use [`compute_theta_min`](#calibration) to derive the deployment-aware threshold.

## Testing

```bash
pytest tests/ -v
```

The current suite contains 1,244 passing tests.

## Documentation

- [Authoritative design document](docs/design/gae_design_v10_8.md)
- [Cross-Graph Attention mathematical foundation](https://www.dakshineshwari.net/post/cross-graph-attention-mathematical-foundation-with-experimental-validation)
- [Dakshineshwari](https://www.dakshineshwari.net)

## License

Apache 2.0. See the [LICENSE](LICENSE) file.
