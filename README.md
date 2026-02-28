# Graph Attention Engine (GAE)

Standalone, pip-installable Python library implementing Cross-Graph Attention.
Zero domain knowledge. Zero external dependencies (numpy only). Apache 2.0.

## Install

```bash
pip install graph-attention-engine
# or editable dev install:
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from gae import scaled_dot_product_attention

Q = np.random.randn(4, 8)   # (seq, d_k)
K = np.random.randn(6, 8)
V = np.random.randn(6, 16)
out, weights = scaled_dot_product_attention(Q, K, V)
# out: (4, 16)  weights: (4, 6)
```

## Math reference

Full specification: `docs/gae_design_v5.md`
Blog: https://www.dakshineshwari.net/post/cross-graph-attention-mathematical-foundation-with-experimental-validation

## Testing

```bash
pytest tests/ -v
```
