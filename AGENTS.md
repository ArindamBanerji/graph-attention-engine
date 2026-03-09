# AGENTS.md — Graph Attention Engine

## Repository
graph-attention-engine — standalone pip-installable Python library.
Zero domain knowledge. Zero external dependencies (numpy only). Apache 2.0.

## Rules
- Do NOT use git directly. I handle all git operations.
- Do NOT start the debugger. Log-based debugging only.
- Read before write. One concern per prompt.
- Every gae/ function: docstring references blog equation (P1).
- Every matrix operation: shape assertion (P2).
- Zero SOC knowledge in gae/ files (P3) — no alerts, no actions, no domains.
- NumPy only (P5). No PyTorch, no sklearn, no Neo4j, no asyncio imports in gae/.
- No async functions in gae/ files. The library is synchronous + numpy.
  Exception: FactorComputer Protocol declares async compute() for implementors.

## Design Reference
- Authoritative spec (single source of truth): docs/gae_design_v8_3.md
- Deprecated: docs/gae_design_v5.md (do not use)
- Math blog: https://www.dakshineshwari.net/post/cross-graph-attention-mathematical-foundation-with-experimental-validation

## Package Structure
```
graph-attention-engine/
├── gae/
│   ├── __init__.py        # Exports + __version__
│   ├── scoring.py         # Tier 2: Eq. 4 scoring matrix
│   ├── learning.py        # Tier 3: Eq. 4b/4c weight learning
│   ├── factors.py         # FactorComputer Protocol + assemble_factor_vector()
│   ├── events.py          # Event type dataclasses (NO bus, NO async)
│   ├── contracts.py       # SchemaContract, EmbeddingContract, PropertySpec
│   ├── store.py           # JSON persistence for LearningState
│   ├── primitives.py      # Scaled dot-product attention (numpy)
│   ├── convergence.py     # Convergence monitoring
│   └── embeddings.py      # Tier 4 placeholder (v5.5)
├── tests/
├── pyproject.toml
├── LICENSE
└── README.md
```

## Testing
```bash
pip install -e ".[dev]"
pytest tests/ -v
```
