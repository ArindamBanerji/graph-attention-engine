# Graph Attention Engine — Design & Architecture v7

**Date:** March 1, 2026
**Version:** 7.0 (v5 + v6 addendum consolidated into single document)
**Status:** v4.1 tagged. Planning v4.5 GAE preamble + v5.0 GAE broadening.
**Repository:** graph-attention-engine (standalone, numpy-only)
**Scope:** v4.1 (Tiers 1-3 foundation), v4.5 (CalibrationProfile preamble), v5.0 (completion + API hardening + evaluation), v5.5 (Tiers 4-5 + open-source release)
**Companion repos:** ci-platform (UCL, agents, event bus, governance — v4.5+), soc-copilot (SOC domain — proprietary)
**Inputs:** Math blog (published), CI 4.0 blog (published), technology scorecard & gap analysis, open-source impact analysis, design_decisions_v1.md

> **Changes from v5/v6:**
> This document consolidates v5 (three-repo architecture, full tier specification) with the v6 addendum (CalibrationProfile, evaluation framework, domain schema protocol, institutional judgment metrics, ablation framework, EmbeddingProvider protocol). All v6 modifications are integrated inline — no separate addendum required.
>
> Key additions from v6:
> (1) **CalibrationProfile.** All learning hyperparameters (α, λ_neg, τ, ε) move from hardcoded constants to a domain-configurable dataclass.
> (2) **Three-layer decay design.** Domain schema declares decay classes → CalibrationProfile maps to rates → GAE learning loop consumes.
> (3) **Evaluation framework.** EvaluationScenario, EvaluationReport, and run_evaluation() provide a shared evaluation format across EVAL-1, SEED-2, and B3 gate consumers.
> (4) **Domain schema protocol.** DomainSchemaSpec creates a single source of truth for factor metadata.
> (5) **Institutional judgment metrics.** Quantifies accumulated institutional judgment for CISO/CPO stakeholders.
> (6) **Ablation framework.** Four baseline configs prove each architectural component's value.
> (7) **EmbeddingProvider protocol.** Replaces PropertyEmbedder with a protocol supporting multiple embedding strategies.
>
> Unchanged from v5: §1-3 (foundations, requirements, equations), §4.1/4.3-4.6 (repo strategy except structure), §5 (causal architecture), §6 (Tier 1 factors), §10 (Tier 5 attention/discovery), §11 (Tier 6 artifact evolution).

---

## 1. Requirements: Why This Document Exists

### 1.1 The Problem

The SOC Copilot's published math blog describes a precise computational system: cross-graph attention (Eq. 6), Hebbian weight learning (Eq. 4b), entity embeddings (Eq. 5), discovery extraction (Eq. 8a-8c), and scoring matrices (Eq. 4). The demo codebase (v3.2, refactored for multi-domain) implements none of this math. Factors are hardcoded floats. Scoring is if-else logic. Learning is a trust counter with ±delta. Discoveries are pre-seeded strings. Narratives are template substitution.

The gap is structural, not cosmetic. No amount of UI polish bridges the distance between "hardcoded factor returns 0.72" and "graph traversal computes 0.72 from entity relationships." The Graph Attention Engine (GAE) is the computational substrate that closes this gap.

### 1.2 Requirements

**R1 — Equation traceability.** Every equation must map to exactly one function in exactly one file. A reviewer can search for "Eq. 4" and find exactly one implementation.

**R2 — Causal closure specification.** The system's defining property is compounding intelligence — outputs of one cycle become inputs to the next. This requires precise specification of six causal connectors (§5) that wire the computation into closed loops. Without this specification, a developer builds a pipeline (alert → process → output) rather than a cybernetic system (alert → process → output → graph mutation → richer graph → richer processing). The difference is existential: a pipeline does not compound.

**R3 — Event-driven architecture.** The three causal loops operate at different timescales and trigger each other. Without an explicit event model, cross-loop triggers become ad-hoc if-checks scattered through router code.

**R4 — Factor vector preservation.** Eq. 4b requires the factor vector f(t) from the ORIGINAL decision, not f recomputed at outcome time. Between decision and outcome, the graph changes. Using a recomputed f would apply the weight update to the wrong context, corrupting the learning signal.

**R5 — Dynamic dimensionality.** When cross-graph attention discovers a new scoring dimension, the weight matrix W must expand from (4×6) to (4×7). All downstream computation must handle a matrix whose shape changed mid-operation.

**R6 — Technology selection justification.** The math blog's equations are structurally identical to transformer attention. §2 explains why importing HuggingFace transformers or PyTorch SDPA would fight us — the math is analogous but the dynamics are fundamentally different.

**R7 — Claude Code prompt decomposition.** The GAE implementation spans 10+ files and ~3,000 lines. Claude Code Sonnet operates best on single-file, tightly-scoped changes. This document decomposes the build into prompts sized for that constraint.

**R8 — Accumulation channel completeness.** Every graph mutation must produce a write-back read by at least one downstream computation. A missing write-back breaks compounding silently — the system computes correctly but doesn't get smarter. Verified by the 10-cycle compounding test (§13.3).

**R9 — Open-source API stability.** The GAE is released as an open-source library (`pip install graph-attention-engine`, Apache 2.0). Every public class and function must be versioned and backward-compatible across minor releases. Engine code never imports from product code (P12). The public API surface is defined in §15 and enforced by CI lint.

---

## 2. Technology Analysis: HuggingFace, PyTorch, and What to Build

### 2.1 What Transformer Attention Actually Is

The HuggingFace `BertSelfAttention` class reduces to five lines of actual math:

```python
Q = self.query(hidden_states)                    # learned linear projection
K = self.key(hidden_states)                       # learned linear projection
V = self.value(hidden_states)                     # learned linear projection
scores = Q @ K.transpose(-1, -2) / sqrt(d_k)     # scaled dot product
output = softmax(scores) @ V                      # weighted sum
```

The remaining ~245 lines provide: multi-head reshape/transpose, position embeddings, KV caching, dropout, LayerNorm + residual connections, attention mask handling, head pruning, gradient checkpointing, and SDPA kernel dispatch.

### 2.2 What's Relevant vs. What's Baggage

| HF Component | Our System | Verdict |
|---|---|---|
| `Q @ K.T / sqrt(d)` — core dot-product | Eq. 4 (scoring) and Eq. 6 (cross-graph attention) | **TAKE the math** |
| Learned linear projections (W_q, W_k, W_v) | Not needed for v5.0 (identity projections). v6+ option | **DEFER** |
| Multi-head attention | Eq. 9: multi-domain attention. Our "heads" are heterogeneous domain pairs | **TAKE concept, BUILD differently** |
| Position embeddings | Not applicable. Entities have graph structure, not sequence position | **LEAVE** |
| KV caching, Dropout, LayerNorm + residual | Not applicable (no autoregressive generation, no backprop) | **LEAVE** |
| Attention masks | Relevant for v5.5: mask impossible cross-domain entity pairs | **TAKE, adapted** |
| Flash Attention / SDPA | Only relevant at scale. 500×300 = 150K ops. Flash optimizes for millions | **DEFER to v6+** |

### 2.3 Scale Analysis and Backend Decision

| Version | Matrix Size | Operations | Backend | Latency |
|---|---|---|---|---|
| v5.0 | (1×6) @ (6×4) | 24 multiply-adds | NumPy | <0.01ms |
| v5.5 | (500×128) @ (128×300) | 2.25M | NumPy | <10ms |
| v6.0 | (10K×128) @ (128×10K) | 1.5B | PyTorch SDPA | ~100ms GPU |
| v6.5+ | (100K×128) @ (128×100K) | 10B × 21 pairs | FAISS + sparse | requires design |

**Decision:** v5.0 and v5.5 use NumPy only. Build with a backend-swappable attention primitive (`primitives.py`) that can swap NumPy → PyTorch SDPA → Flash Attention without changing any caller.

### 2.4 What IS Worth Borrowing

| Library | What We'd Use | When |
|---|---|---|
| **sentence-transformers** | Pre-trained embeddings for text-heavy entities | v5.5 (optional hybrid) |
| **torch_geometric** | GNN-based entity embeddings capturing multi-hop graph structure | v6.0+ |
| **scikit-learn** | Normalization, evaluation metrics, clustering | v5.0 |
| **FAISS** | Approximate nearest neighbor for large-scale embedding search | v6.5+ |

### 2.5 Why This Is Not a Transformer

The math blog draws a formal correspondence between our system and transformers. The correspondence is real — equations are structurally identical. But the dynamics are fundamentally different:

| Property | Transformer | Graph Attention Engine |
|---|---|---|
| **Data structure** | Sequential tokens, fixed-length window | Graph entities, growing heterogeneous knowledge graph |
| **Computation trigger** | Forward pass on input batch | Event-driven: alert arrival, outcome, scheduled sweep |
| **Learning mechanism** | Backpropagation on differentiable loss | Hebbian reinforcement from binary outcomes (Eq. 4b) |
| **State** | Weights frozen after training | Weights evolve continuously; graph grows permanently |
| **Attention scope** | All tokens attend to all tokens | Cross-domain: entities in domain i attend to entities in domain j |
| **Feedback** | None at inference (feed-forward) | Closed loop: decisions → outcomes → weight updates → decisions |
| **Dimensionality** | Fixed (d_model, n_heads) | Dynamic: W expands when new dimensions discovered |
| **Graph mutation** | N/A | Core mechanism: every decision, discovery, and ingest mutates the graph |

The system is closer to: **contextual bandits** (Li et al., 2010), **Graph Attention Networks** (Veličković et al., 2018), and **cybernetic control systems** (Wiener/Ashby). Importing transformer infrastructure would fight us.

---

## 3. Design Principles

**P1 — Equation traceability.** Every function docstring references the blog equation it implements.

**P2 — Shape checks in code.** Every matrix operation includes an assertion matching the blog's shape check.

**P3 — Domain-agnostic.** The GAE lives in `gae/`, not `domains/soc/`. SOC-specific factors live in the domain module.

**P4 — Incremental replacement.** Each tier activates independently. v5.0 activates Tiers 1-3. v5.5 activates Tiers 4-5. Existing demo behavior preserved as fallback.

**P5 — NumPy only for v5.0.** No PyTorch, no TensorFlow.

**P6 — Backend-swappable primitives.** `primitives.py` accepts a `backend` parameter. Callers never import backend-specific code.

**P7 — Factor vector preservation.** Every Decision node stores f(t) from the ORIGINAL decision (R4).

**P8 — Event-driven causal propagation.** Graph mutations emit events. Events trigger downstream computation (R3).

**P9 — No debugger, no git.** All Claude Code prompts respect existing development rules.

**P10 — Factor queries traverse relationships, not properties (R8).** Every FactorComputer's Cypher query MUST traverse at least one relationship. A query reading `u.travel_history` doesn't benefit from graph accumulation. A query doing `MATCH (u)-[:HAS_TRAVEL]->(t:TravelRecord)` DOES.

**P11 — Every graph mutation emits an event (R3, R8).** Decision written → emit `GraphMutated(type="decision")`. Outcome marked → emit `GraphMutated(type="outcome")`. Discovery written → emit `GraphMutated(type="discovery")`. Silent mutations break causal propagation.

**P12 — Engine/product boundary (R9).** Engine code (GAE repo: `gae/`, `examples/`) NEVER imports from product code (copilot repos: `domains/soc/`, `connectors/`, `routers/`, `frontend/`). Dependency arrow is strictly one-way: copilot → platform → GAE. Enforced structurally by separate repositories.

---

## 4. Package Structure — Three-Repository Architecture

### 4.1 Repository Overview

```
graph-attention-engine/          ← THIS REPO. pip-installable. Apache 2.0. numpy-only.
ci-platform/                     ← COMPANION REPO. Production infrastructure. v4.5+.
soc-copilot/                     ← DOMAIN REPO. SOC domain expertise. Proprietary.
```

**Dependency direction (strict one-way):**
```
graph-attention-engine           ← ZERO external deps (numpy only)
        ↑
ci-platform                      ← depends on gae + neo4j + asyncio
        ↑
soc-copilot                      ← depends on ci-platform + gae
```

GAE never imports from platform. Platform never imports from any copilot.

### 4.2 GAE Repository (This Document)

```
graph-attention-engine/
├── gae/
│   ├── __init__.py              # Public API exports (§15)
│   ├── scoring.py               # Tier 2: Eq. 4 scoring matrix (§7)
│   ├── learning.py              # Tier 3: Eq. 4b, 4c + hardening (§8)
│   ├── factors.py               # Tier 1: FactorComputer Protocol + assemble (§6)
│   ├── contracts.py             # Demand-side: SchemaContract, EmbeddingContract, PropertySpec (§12)
│   ├── primitives.py            # Backend-swappable attention (§2.3, P6)
│   ├── embeddings.py            # Tier 4: Eq. 5 entity embeddings (§9) [v5.5]
│   ├── attention.py             # Tier 5: Eq. 6, 7, 9 cross-graph attention (§10) [v5.5]
│   ├── discovery.py             # Tier 5: Eq. 8a-8c discovery extraction (§10) [v5.5]
│   ├── convergence.py           # Convergence monitoring + failure modes
│   ├── events.py                # Event TYPE definitions (dataclasses only, no bus)
│   ├── store.py                 # Weight + embedding persistence (JSON, no Neo4j)
│   └── types.py                 # Shared type aliases
├── tests/
│   ├── test_scoring.py          # All tests: numpy + pytest ONLY. No Neo4j.
│   ├── test_learning.py
│   ├── test_contracts.py
│   └── ...
├── examples/
│   └── minimal_domain/          # "Hello World" — 50-line DomainConfig (v5.0: ENG-2)
├── docs/
│   └── equations.md             # Equation-to-code traceability
├── pyproject.toml               # Dependencies: numpy>=1.24.0. That's it.
├── LICENSE                      # Apache 2.0
└── README.md
```

**Key constraint:** `pyproject.toml` declares `dependencies = ["numpy>=1.24.0"]`. No Neo4j, no asyncio, no HTTP clients. A researcher can `pip install graph-attention-engine` and experiment with the math in a Jupyter notebook.

### 4.3 ci-platform Repository (Companion — see ci_platform_design_v1)

```
ci-platform/
├── platform/
│   ├── events.py                # GraphEventBus (async dispatch, subscriber mgmt)
│   ├── state_manager.py         # Reset coordination
│   ├── domain_registry.py       # Domain lookup
│   ├── validation/
│   │   └── contract_checker.py  # Validates GAE SchemaContracts against live graph
│   ├── ucl/
│   │   ├── resolution.py        # INOVA entity resolution (v4.5)
│   │   ├── schema.py            # SchemaValidator — write-path validation (v6.0)
│   │   └── ontology.py          # DomainOntology — supply-side governance (v6.0)
│   └── agents/                  # Multi-agent artifact evolution (v6.0)
│       ├── meta_prompt_agent.py
│       ├── safeguard_agent.py
│       └── evaluation_framework.py
├── domains/
│   └── base.py                  # DomainConfig ABC + dataclasses
├── tests/
└── pyproject.toml               # depends on: graph-attention-engine, neo4j, asyncio
```

### 4.4 soc-copilot Repository (Domain — see soc_copilot_design_v1)

```
soc-copilot/
├── domains/soc/
│   ├── factors.py               # 6 FactorComputer implementations
│   ├── config.py                # SOCDomainConfig — actions, initial W, temperature
│   ├── situations.py            # Scoring-based situation classification
│   └── seed_data/               # SOC-specific seed parameters
├── connectors/                  # Pulsedive, GreyNoise, Health-ISAC, CISA KEV
├── frontend/                    # React UI (all tabs)
├── routers/                     # FastAPI endpoints
├── deployment/                  # Docker, VPS, cloud
└── pyproject.toml               # depends on: graph-attention-engine, ci-platform
```

### 4.5 Import Boundary (P12)

```python
# soc-copilot imports from GAE's public API:
from gae.scoring import score_alert, ScoringResult
from gae.learning import LearningState, WeightUpdate
from gae.factors import FactorComputer, assemble_factor_vector
from gae.contracts import SchemaContract, EmbeddingContract, PropertySpec
from gae.events import DecisionMade, OutcomeVerified, GraphMutated

# soc-copilot imports from ci-platform:
from platform.events import GraphEventBus
from platform.domains.base import DomainConfig, DomainAction

# GAE NEVER imports from platform or copilot.
# Platform NEVER imports from copilot.
# Enforced structurally — separate repos, separate packages.
```

### 4.6 Development Bridge

During v4.1 development (before PyPI release):

```bash
# In soc-copilot directory:
pip install -e ../graph-attention-engine
pip install -e ../ci-platform        # when it exists (v4.5+)
```

## 5. Causal Architecture: Three Loops, Six Connectors

### 5.1 Why Causal Architecture Matters (Requirement R2)

The GAE is not a pipeline of functions. It is a closed-loop cybernetic system where the graph is simultaneously input and output:

- **Input** to factor computation (Cypher queries traverse it)
- **Input** to embedding computation (entity properties extracted from it)
- **Output** of decisions (new Decision nodes written to it)
- **Output** of discoveries (new [:CALIBRATED_BY] relationships written to it)
- **Output** of learning (weight snapshots stored in it)

This circular causality — outputs of one cycle become inputs to the next — is what makes compounding intelligence possible. A pipeline processes; a cybernetic system compounds.

### 5.2 Three Causal Loops

```
╔══════════════════════════════════════════════════════════════╗
║  FAST LOOP (per-alert, seconds)                              ║
║                                                              ║
║  Alert → [Factor Computation] → f (1×n)                     ║
║            ↑ reads graph              ↓                      ║
║            │                    [Scoring Matrix] Eq. 4       ║
║            │                    f · Wᵀ / τ → softmax         ║
║            │                          ↓                      ║
║            │                    Action + Confidence           ║
║            │                          ↓                      ║
║            │                    [LLM Narrative]              ║
║            │                          ↓                      ║
║            │                    Human/Auto Decision           ║
║            │                          ↓                      ║
║            │                    [Outcome Verification]        ║
║            │                          ↓                      ║
║            │                    [Weight Update] Eq. 4b        ║
║            │                    W[a,:] += α·r(t)·f(t)·δ(t)  ║
║            │                          ↓                      ║
║            │                    [Graph Write-Back]            ║
║            │                    Decision node + relationships ║
║            │                          │                      ║
║            └──────────────────────────┘                      ║
║                                                              ║
║  CLOSURE: Graph is richer. W is adjusted.                    ║
║  NEXT ALERT: factors from richer graph, scored by            ║
║  adjusted W → different decision.                            ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║  SLOW LOOP (periodic, hours/daily) [v5.5]                    ║
║                                                              ║
║  [Scheduled Trigger]                                         ║
║         ↓                                                    ║
║  [Compute/Refresh Embeddings] Eq. 5                          ║
║  For each domain: extract properties → normalize → embed     ║
║         ↓                                                    ║
║  [Cross-Graph Attention Sweep] Eq. 6, 9                      ║
║  For each domain pair (i,j): Q=Eᵢ, K=Eⱼ                    ║
║  Compatibility = softmax(Eᵢ · Eⱼᵀ / √d)                    ║
║         ↓                                                    ║
║  [Discovery Extraction] Eq. 8a-8c                            ║
║  Threshold + top-K + margin → candidate discoveries          ║
║         ↓                                                    ║
║  [Discovery Validation + Graph Write-Back]                   ║
║  [:CALIBRATED_BY] edges + [:DISCOVERED_VIA] provenance       ║
║         ↓                                                    ║
║  [Re-Score Affected Alerts via Fast Loop]                    ║
║  New relationships change factor computation.                ║
║  The discovery ENRICHES the graph the Fast Loop reads.       ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║  META LOOP (rare, structural mutation) [v5.5+/v6.0]         ║
║                                                              ║
║  Two triggers, both structural:                              ║
║                                                              ║
║  (A) Discovery of type "new scoring dimension" [v5.5]        ║
║      → W Expands: (4×6) → (4×7)                             ║
║      → New FactorComputer registered                         ║
║      → System evaluates along axis that didn't exist         ║
║      → Eq. 4b calibrates new column from outcomes            ║
║                                                              ║
║  (B) Artifact evolution [v6.0]                               ║
║      → N outcomes accumulated → trigger evaluation           ║
║      → Meta-Prompt Agent generates K variant artifacts       ║
║      → R×T×Q framework evaluates variants                   ║
║      → Safeguard Agent checks constraint satisfaction        ║
║      → Shadow test → promote if variant > baseline           ║
║      → Operational pipeline structure changed                ║
║                                                              ║
║  CLOSURE: System changed its own evaluation criteria (A)     ║
║  or its own operational configuration (B).                   ║
║  Both are Axis 3 — Capability Extension.                    ║
╚══════════════════════════════════════════════════════════════╝
```

### 5.3 Six Causal Connectors

Each arrow in the causal loops represents a causal connector — a specific data transformation with defined trigger, input, output, mechanism, and causal dependency.

**Connector 1: Graph → Factors (read)**

| Attribute | Specification |
|---|---|
| Trigger | Alert arrives, or re-scoring triggered by discovery |
| Input | Alert data + Neo4j graph state |
| Output | Factor vector f ∈ ℝ^(1×n), per-factor provenance metadata |
| Mechanism | Each FactorComputer runs a Cypher query, computes normalized [0,1] score |
| Causal dependency | Factor values depend on EVERYTHING in the graph — past decisions, threat intel, org structure, device inventory, behavioral baselines, discovered relationships. The graph's richness becomes computational input. |
| Implementation | `gae/factors.py` → `assemble_factor_vector()` (§6) |

**Connector 2: Factors × Weights → Decision (compute)**

| Attribute | Specification |
|---|---|
| Trigger | Factor vector f ready |
| Input | f, W, τ, action list |
| Output | Action probabilities, selected action, confidence |
| Mechanism | Eq. 4: softmax(f · Wᵀ / τ) |
| Causal dependency | W encodes ALL prior learning. The decision is shaped by every previous verified outcome that updated W via Eq. 4b. The W matrix IS the accumulated judgment. |
| Implementation | `gae/scoring.py` → `score_alert()` (§7) |

**Connector 3: Decision + Outcome → Weight Update (learn)**

| Attribute | Specification |
|---|---|
| Trigger | Human verifies outcome, or auto-verification fires |
| Input | Action taken, outcome r(t) ∈ {+1, -1}, factor vector f(t) from ORIGINAL decision (R4) |
| Output | Updated W (weight matrix is different now) |
| Mechanism | Eq. 4b: W[a,:] += α_eff·r(t)·f(t)·δ(t) with 20:1 asymmetry. Eq. 4c: W *= (1-ε_vector). Hardened: confidence-discounted α (A1), per-factor decay (A2), delayed validation for autonomous decisions (C3). |
| Causal dependency | Reinforces/penalizes SPECIFIC factor-action associations. 20:1 asymmetry: one failure erases 20 successes — encodes SOC risk preference into learning dynamics. |
| Implementation | `gae/learning.py` → `LearningState.update()` (§8) |

**Connector 4: Graph → Embeddings → Attention → Discoveries (discover) [v5.5]**

| Attribute | Specification |
|---|---|
| Trigger | Scheduled sweep, or significant graph mutation (GraphMutated event) |
| Input | All entity properties across all domains |
| Output | Discovery hypotheses + validated discoveries as new graph relationships |
| Mechanism | Eq. 5 (embed) → Eq. 6 (cross-attend) → Eq. 8a-8c (extract) → graph write-back |
| Causal dependency | Discovery surface depends on CURRENT graph state. Adding a new TI entity changes domain 3 embedding, changes cross-attention, may surface a new discovery. This is where super-quadratic scaling comes from. |
| Implementation | `gae/embeddings.py` + `attention.py` + `discovery.py` (§9, §10) |

**Connector 5: Discovery → Graph Mutation → Factor Expansion (extend) [v5.5+]**

| Attribute | Specification |
|---|---|
| Trigger | Validated discovery of type "new scoring dimension" |
| Input | Discovery with validated cross-domain pattern |
| Output | New FactorComputer registered + W gains column + f gains dimension |
| Mechanism | `LearningState.expand_weight_matrix()` (§8.3). W: (n_a × n_f) → (n_a × (n_f+1)). New column enters as "provisional" with accelerated decay (A4 hardening). |
| Causal dependency | META-CAUSAL: changes STRUCTURE of computation. After this fires, Eq. 4 operates in higher-dimensional space. System evaluates alerts along criteria it invented from operational experience. |
| Implementation | `gae/learning.py` → `expand_weight_matrix()` + `gae/events.py` → `DiscoveryValidated` type (§8.3) |

**Connector 6: Outcomes → Artifact Evaluation → Artifact Mutation (evolve) [v6.0]**

| Attribute | Specification |
|---|---|
| Trigger | N verified outcomes accumulated since last evaluation cycle (configurable: e.g., every 100 decisions or weekly) |
| Input | Current baseline artifact (prompt/routing rule/tool config as YAML) + recent outcome history from experience pool + fitness metrics from convergence monitoring |
| Output | New baseline artifact (or: no change — current baseline retained) |
| Mechanism | (1) Meta-Prompt Agent generates K variant artifacts from baseline. (2) Safeguard Agent checks each variant for constraint satisfaction. (3) R×T×Q evaluation framework scores variants against ground truth. (4) Shadow orchestrator routes fraction of live traffic to top variant. (5) After N shadow decisions: compare variant vs. baseline outcomes. (6) If variant > baseline with significance threshold: promote. (7) Write [:TRIGGERED_EVOLUTION] with evaluation evidence. |
| Causal dependency | SECOND learning loop, orthogonal to Connector 3 (weight learning). Connector 3 changes PARAMETERS (which factor-action associations are strong). Connector 6 changes STRUCTURE (which prompts, routing rules, tool configurations are in effect). Both compound each other. |
| Implementation | ci-platform: `platform/agents/` + `platform/gae/evolution.py` (§11) |
| Dependencies | Requires Connector 3 for fitness function. Requires Connector 1 for evaluation scenarios. Requires ARC platform (YAML runtime, action ledger, typed-intent bus). Requires SC techniques (meta-prompt agent, R×T×Q framework, safeguard agent). |

### 5.4 The Residual Principle (Why It Compounds Instead of Replacing)

Eq. 3 from Vaswani (residual connection): `output = LayerNorm(x + Attention(x))`. In our system, the residual principle manifests at the graph level:

- **Decisions accumulate.** Each Decision node is ADDED. Pattern_history factor reads from ALL prior decisions, not just the most recent.
- **Discoveries layer.** [:CALIBRATED_BY] edges ADD TO existing entities. Enrichment, not replacement.
- **Weight updates refine.** Eq. 4b adds a delta. Eq. 4c decays ALL weights. Recent experience has more influence, but no single update wipes the slate. Effective half-life ~693 decisions (ε=0.001 default) means the system "remembers" roughly the last 2000 decisions with declining influence.
- **New dimensions extend.** W goes from (4×6) to (4×7). Existing 6 columns unchanged. New column starts small (provisional, per A4 hardening), calibrated by Eq. 4b.
- **Artifacts evolve.** Connector 6 promotes a new baseline but retains the old one as rollback. Improvement is additive to the operational history.

This is why "compounding" is a precise mathematical term, not marketing. Each cycle adds to what came before. The next cycle operates on a strictly richer substrate.

### 5.5 Event Model (Requirement R3)

GAE defines event **types** as pure Python dataclasses. The event **bus** (async dispatch, subscriber management) lives in ci-platform. This keeps GAE at zero external dependencies.

```python
# gae/events.py — PURE PYTHON DATACLASSES. No async. No bus.

@dataclass
class AlertArrived:
    """Triggers: Factor Computation → Scoring → Action Recommendation"""
    alert_id: str
    alert_data: dict
    timestamp: float

@dataclass
class DecisionMade:
    """Triggers: Graph Write-Back (Decision node + relationships)"""
    alert_id: str
    action: str
    confidence: float
    factor_vector: np.ndarray    # f(t) — MUST be preserved for Eq. 4b (R4)
    W_snapshot: np.ndarray       # W at decision time — for audit

@dataclass
class OutcomeVerified:
    """Triggers: Weight Update (Eq. 4b) → Convergence Check"""
    alert_id: str
    action: str
    outcome: int                 # +1 or -1
    factor_vector: np.ndarray    # f(t) from ORIGINAL decision (R4)
    verifier: str                # "analyst" or "auto"

@dataclass
class GraphMutated:
    """Triggers: Optional re-scoring of affected alerts.
    Triggers: Optional re-embedding of affected domains."""
    mutation_type: str           # "decision", "discovery", "ingest", "outcome"
    affected_entities: List[str]
    affected_domains: List[str]

@dataclass
class DiscoverySweepRequested:
    """Triggers: Embedding → Cross-attention → Discovery extraction"""
    trigger: str                 # "scheduled", "threshold", "manual"
    domains: List[str]           # default: all

@dataclass
class DiscoveryValidated:
    """Triggers: Graph write-back + optional W expansion (R5) + re-scoring"""
    discovery_id: str
    source_entity: str
    target_entity: str
    confidence: float
    creates_new_dimension: bool
    new_factor_name: str = None  # if creates_new_dimension is True

@dataclass
class ArtifactEvaluationRequested:
    """Triggers: Meta-Prompt Agent → Safeguard → R×T×Q eval [v6.0]"""
    artifact_type: str           # "prompt", "routing_rule", "tool_config"
    baseline_id: str
    trigger_reason: str          # "periodic", "regression_detected", "manual"
    outcome_count: int

@dataclass
class ArtifactPromoted:
    """Triggers: Graph write-back ([:TRIGGERED_EVOLUTION]) [v6.0]"""
    artifact_type: str
    old_baseline_id: str
    new_baseline_id: str
    evaluation_summary: dict
    significance: float
```

**The event bus** lives in ci-platform (see ci_platform_design_v1):

```python
# ci-platform: platform/events.py (NOT part of GAE — reference only)
class GraphEventBus:
    """Central event bus for causal propagation across loops."""
    def __init__(self):
        self._subscribers = defaultdict(list)

    def subscribe(self, event_type, handler):
        self._subscribers[event_type].append(handler)

    async def emit(self, event):
        for handler in self._subscribers[type(event)]:
            await handler(event)
```

### 5.6 Persistent and Ephemeral State

```
┌─────────────────────────────────────────────┐
│  PERSISTENT STATE (survives restarts)        │
│                                              │
│  Neo4j Graph:                                │
│    - Entity nodes (Users, Assets, Alerts...) │
│    - Relationship edges                      │
│    - Decision nodes (with f(t) per R4)       │
│    - Discovery nodes + [:CALIBRATED_BY]      │
│    - [:DISCOVERED_VIA] provenance            │
│    - [:TRIGGERED_EVOLUTION] audit            │
│                                              │
│  Weight Matrix Store:                        │
│    - Current W (n_actions × n_factors)       │
│    - W history (every snapshot)              │
│    - Factor computer registry + names        │
│    - Expansion history (R5)                  │
│    - DimensionMetadata per column (A4)       │
│                                              │
│  Embedding Store (v5.5):                     │
│    - Eᵢ for each domain                     │
│    - Last computation timestamp              │
│    - Entity ID ↔ row index mapping           │
│                                              │
│  Artifact Store (v6.0):                      │
│    - Current baseline artifacts (YAML)       │
│    - Variant history + evaluation results    │
│    - Promotion/retirement state              │
│                                              │
├─────────────────────────────────────────────┤
│  EPHEMERAL STATE (per-session)               │
│                                              │
│  Active alerts in queue                      │
│  Current triage context                      │
│  In-progress discovery sweep results         │
│  LLM narrative cache                         │
│  Pending autonomous validations (C3)         │
│                                              │
└─────────────────────────────────────────────┘
```

### 5.7 Five Accumulation Channels (Requirement R8)

The system compounds because each cycle leaves the graph richer. There are five distinct channels through which this enrichment occurs. Every FactorComputer must document which channels affect it.

| Channel | What Accumulates | Who Benefits | Wired By | Version |
|---|---|---|---|---|
| **A: Decision** | Decision nodes written to graph with f(t), action, confidence, `[:DECIDED_ON]→Alert` | PatternHistoryFactor (decision count, base rates) | Connector 2 write-back | v4.1 |
| **B: Outcome** | Decision nodes marked correct/incorrect with outcome timestamp | PatternHistoryFactor (accuracy), W via Eq. 4b | Connector 3 write-back | v4.1 |
| **C: Entity Ingestion** | New entities (TI campaigns, users, assets, devices) with relationships | ThreatIntelEnrichment, TravelMatch, DeviceTrust, AssetCriticality | Connector ecosystem + seed | v4.1+ |
| **D: Relationship Enrichment** | New edges between existing entities (`[:CALIBRATED_BY]`, analyst links, discovery write-back) | Any factor with relationship-traversing queries (P10) | Discovery write-back, analyst actions | v5.5 |
| **E: Structural Expansion** | New scoring dimensions → W expands (4×6) → (4×7) → new FactorComputer registered | ALL factors (new W column), ALL scoring (higher-dimensional space) | Meta loop: DiscoveryValidated → expand_weight_matrix() | v5.5 |

**Each factor's docstring must declare which channels affect it.**

### 5.8 Decision and Outcome Write-Back Specifications

These are the specific graph mutations that make Channels A and B work. Without them, the system computes without compounding.

**Connector 2 Output — Decision Write-Back (Channel A):**

```python
async def write_decision_to_graph(alert_id, result, f, neo4j):
    """
    Connector 2 output: Decision node carries f(t) for R4.
    This write-back is what makes the graph richer for the next cycle.
    PatternHistoryFactor reads these nodes.
    """
    query = """
    MATCH (a:Alert {id: $alert_id})
    CREATE (d:Decision {
        id: $decision_id,
        action: $action,
        confidence: $confidence,
        factor_vector: $factor_vector,
        timestamp: datetime(),
        outcome: null
    })
    CREATE (d)-[:DECIDED_ON]->(a)
    RETURN d.id
    """
    await neo4j.run(query, 
        alert_id=alert_id,
        decision_id=f"DEC-{alert_id}-{int(time.time())}",
        action=result.selected_action,
        confidence=result.confidence,
        factor_vector=f.flatten().tolist()
    )
```

**Connector 3 Input/Output — Outcome Write-Back (Channel B):**

```python
async def record_outcome_to_graph(decision_id, outcome, neo4j):
    """
    Connector 3: Retrieve f(t) from GRAPH per R4, mark outcome.
    PatternHistoryFactor reads d.correct to compute accuracy.
    """
    query = """
    MATCH (d:Decision {id: $decision_id})
    SET d.outcome = $outcome_label,
        d.outcome_timestamp = datetime(),
        d.correct = $correct
    RETURN d.factor_vector
    """
    result = await neo4j.run(query,
        decision_id=decision_id,
        outcome_label="correct" if outcome == +1 else "incorrect",
        correct=(outcome == +1)
    )
    record = await result.single()
    f_original = np.array(record["factor_vector"]).reshape(1, -1)  # R4
    return f_original
```

**Event Emission Pattern (P11):**

Four events per decision-outcome cycle. Decision and graph write-back are separate causal steps with different subscribers. A subscriber re-scoring pending alerts needs `GraphMutated`, not `DecisionMade`.

### 5.9 Semantic Accumulation Tests (R8 Verification)

**Test A — Decision Accumulation:** Reset graph → Analyze alert → PatternHistoryFactor returns 0.5 → Submit correct feedback → Repeat 4× → Analyze new same-type alert → PatternHistoryFactor returns ~1.0. VALUE CHANGED because graph accumulated Decision nodes.

**Test B — Entity Ingestion:** Analyze alert → ThreatIntelEnrichment returns X → Ingest new ThreatIntel entity → Re-analyze → Returns Y ≠ X. VALUE CHANGED because graph accumulated new entity.

**Test C — Weight Learning:** Analyze travel alert → Scores = S1 → Submit correct outcome → Re-analyze same type → Scores = S2 ≠ S1. SCORES CHANGED because W accumulated learning.


---

## 6. Tier 1 — Factor Protocol & Assembly (`gae/factors.py`)

### 6.1 Interface (GAE — numpy-only)

```python
# gae/factors.py
"""
Tier 1: Factor Computation Protocol and Vector Assembly.

GAE defines:
  - FactorComputer Protocol (abstract interface for domain implementations)
  - assemble_factor_vector() (synchronous, numpy-only — assembles pre-computed values)

Domain copilots implement:
  - Concrete FactorComputer classes with async Neo4j queries
  - compute_factor_vector() orchestrator (async, Neo4j-dependent — lives in copilot)

This separation keeps GAE at zero external dependencies.
The factor vector f is the input to Eq. 4: P(action|alert) = softmax(f · Wᵀ / τ).
"""

import numpy as np
from typing import Protocol, Optional, List, Tuple, runtime_checkable

@runtime_checkable
class FactorComputer(Protocol):
    """A single factor computation. Domain copilots implement these."""
    name: str
    contract: Optional['SchemaContract']  # declared schema requirements (§12), None until v5.0

    async def compute(self, alert: dict, neo4j) -> float:
        """Returns normalized score ∈ [0, 1] from graph traversal.
        
        NOTE: The `neo4j` parameter type is intentionally untyped here.
        GAE has no Neo4j dependency. Copilots pass their own session type.
        """
        ...

def assemble_factor_vector(
    values: List[float],
    names: List[str]
) -> Tuple[np.ndarray, dict]:
    """
    Assemble factor vector f from pre-computed factor values.
    
    This is the GAE-side function (synchronous, numpy-only).
    The copilot's compute_factor_vector() calls each FactorComputer,
    then passes the results here.
    
    Returns:
        f: np.ndarray, shape (1, n_factors) — Eq. 4 input
        metadata: dict — per-factor provenance
    """
    clamped = [np.clip(v, 0.0, 1.0) for v in values]
    f = np.array(clamped, dtype=np.float64).reshape(1, -1)
    assert f.shape == (1, len(names)), f"Shape check: f must be (1, {len(names)})"
    
    metadata = {name: {"raw": raw, "clamped": c, "computed_from": "graph_traversal"}
                for name, raw, c in zip(names, values, clamped)}
    return f, metadata
```

**In the copilot repo** (NOT in GAE — shown for reference only):

```python
# soc-copilot: domains/soc/orchestrator.py (reference — NOT part of GAE)
async def compute_factor_vector(alert, computers, neo4j):
    """Async orchestrator. Calls each FactorComputer, then delegates to GAE."""
    values, names = [], []
    for computer in computers:
        raw = await computer.compute(alert, neo4j)
        values.append(raw)
        names.append(computer.name)
    return assemble_factor_vector(values, names)
```

### 6.2 SOC Factor Implementations (soc-copilot repo — reference only)

Six FactorComputers, each with a Cypher query traversing at least one relationship (P10) and a declared SchemaContract (§12).

```python
class TravelMatchFactor(FactorComputer):
    """
    Cypher: MATCH (u:User {id: $user})-[:HAS_TRAVEL]->(t:TravelRecord)
            WHERE t.destination = $geo
    Score: count / (count + 3) — saturating. Recency boost +0.15 if < 7 days.
    Channels: C (new TravelRecords), D (new [:HAS_TRAVEL] edges)
    """
    name = "travel_match"
    contract = SchemaContract(
        factor_name="travel_match",
        required_labels=["User", "TravelRecord"],
        required_relationships=[("User", "HAS_TRAVEL", "TravelRecord")],
        required_properties={"User": ["id"], "TravelRecord": ["destination", "date"]},
        minimum_node_count={"User": 5, "TravelRecord": 1},
        decay_rate=0.003, decay_class="campaign",
    )


class AssetCriticalityFactor(FactorComputer):
    """
    Cypher: MATCH (a:Asset {id: $asset})-[:STORES]->(d:DataClass)
    Score: Map criticality (LOW=0.2, MED=0.5, HIGH=0.8, CRIT=1.0).
           Boost +0.1 per sensitive data class (PII, PHI, PCI).
    Channels: C (new Assets/DataClasses), D (new [:STORES] edges)
    """
    name = "asset_criticality"
    contract = SchemaContract(
        factor_name="asset_criticality",
        required_labels=["Asset", "DataClass"],
        required_relationships=[("Asset", "STORES", "DataClass")],
        required_properties={"Asset": ["id", "criticality"], "DataClass": ["sensitivity"]},
        decay_rate=0.0001, decay_class="permanent",
    )


class ThreatIntelEnrichmentFactor(FactorComputer):
    """
    Cypher: MATCH (ti:ThreatIntel)-[:ASSOCIATED_WITH]->(a:Alert {id: $alert})
    Score: Fuse across sources. Max severity normalized. Corroboration boost.
    Channels: C (new TI entities), D (new [:ASSOCIATED_WITH] edges)
    """
    name = "threat_intel_enrichment"
    contract = SchemaContract(
        factor_name="threat_intel_enrichment",
        required_labels=["ThreatIntel", "Alert"],
        required_relationships=[("ThreatIntel", "ASSOCIATED_WITH", "Alert")],
        required_properties={"ThreatIntel": ["severity", "source", "confidence"]},
        decay_rate=0.003, decay_class="campaign",
    )


class PatternHistoryFactor(FactorComputer):
    """
    THIS IS THE COMPOUNDING PROOF FACTOR.
    First alert: returns 0.5 (no history).
    After 5 correct decisions on same type: returns ~1.0.

    Cypher: MATCH (d:Decision)-[:DECIDED_ON]->(a:Alert)
            WHERE a.situation_type = $type AND d.outcome IS NOT NULL
    Score: correct / total (base rate). Minimum 5 decisions for non-default.
    Channels: A (Decision nodes), B (outcome markings)
    """
    name = "pattern_history"
    contract = SchemaContract(
        factor_name="pattern_history",
        required_labels=["Decision", "Alert"],
        required_relationships=[("Decision", "DECIDED_ON", "Alert")],
        required_properties={"Decision": ["action", "correct", "outcome"], "Alert": ["situation_type"]},
        decay_rate=0.001, decay_class="standard",
    )


class TimeAnomalyFactor(FactorComputer):
    """
    Cypher: MATCH (u:User {id: $user})-[:ACTIVE_AT]->(ts:TimeSlot)
    Score: 1.0 - P(current_hour ∈ normal_hours). Unusual time → high score.
    Channels: C (new TimeSlot associations)
    """
    name = "time_anomaly"


class DeviceTrustFactor(FactorComputer):
    """
    Cypher: MATCH (d:Device {id: $device})
    Score: 0.0 (fully trusted) to 1.0 (unknown device).
    Channels: C (new Device entities)
    """
    name = "device_trust"
```

### 6.3 Factor Registration

```python
# domains/soc/config.py — add to SOCDomainConfig
def get_factor_computers(self) -> List[FactorComputer]:
    return [
        TravelMatchFactor(), AssetCriticalityFactor(),
        ThreatIntelEnrichmentFactor(), TimeAnomalyFactor(),
        DeviceTrustFactor(), PatternHistoryFactor(),
    ]
```

---

## 7. Tier 2 — Scoring Matrix (`gae/scoring.py`)

### 7.1 What This Implements

**Eq. 4:** P(action | alert) = softmax(f · Wᵀ / τ)

### 7.2 Interface

```python
# gae/scoring.py
"""
Tier 2: The Scoring Matrix — Eq. 4 from the math blog.

P(action | alert) = softmax(f · Wᵀ / τ)

f: factor vector (1 × n_factors), computed by Tier 1
W: weight matrix (n_actions × n_factors), learned by Tier 3
τ: temperature scalar, controls softmax sharpness
    (v7: from CalibrationProfile if provided, else explicit τ param, else 0.25)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ScoringResult:
    """Output of Eq. 4."""
    action_probabilities: np.ndarray   # shape (1, n_actions)
    selected_action: str               # argmax action name
    confidence: float                  # max probability
    raw_scores: np.ndarray             # f · Wᵀ before softmax
    factor_vector: np.ndarray          # f — preserved per R4
    temperature: float                 # τ used

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def score_alert(
    f: np.ndarray, W: np.ndarray,
    actions: List[str], tau: float = None,
    profile: 'CalibrationProfile' = None
) -> ScoringResult:
    """
    Eq. 4: P(action | alert) = softmax(f · Wᵀ / τ)

    Temperature resolution order:
        1. Explicit tau parameter (if provided)
        2. profile.temperature (if profile provided)
        3. Default: 0.25

    Shape checks (from math blog):
        f (1 × n_factors) × Wᵀ (n_factors × n_actions) = (1 × n_actions)
    """
    if tau is None:
        tau = profile.temperature if profile else 0.25

    n_factors = f.shape[1]
    n_actions = W.shape[0]

    assert f.shape == (1, n_factors), f"f must be (1, {n_factors}), got {f.shape}"
    assert W.shape == (n_actions, n_factors), f"W must be ({n_actions}, {n_factors}), got {W.shape}"
    assert len(actions) == n_actions

    raw_scores = (f @ W.T) / tau
    probs = softmax(raw_scores.flatten())
    assert abs(probs.sum() - 1.0) < 1e-6, "Probabilities must sum to 1"

    selected_idx = int(np.argmax(probs))
    return ScoringResult(
        action_probabilities=probs.reshape(1, -1),
        selected_action=actions[selected_idx],
        confidence=float(probs[selected_idx]),
        raw_scores=raw_scores,
        factor_vector=f,
        temperature=tau
    )
```

### 7.3 Initial Weight Matrix (SOC Domain — PRODUCT)

```python
# domains/soc/config.py
def get_initial_weight_matrix(self) -> np.ndarray:
    """W₀: 4 actions × 6 factors. Initial heuristic weights.
    Overridden by Eq. 4b learning after verified outcomes."""
    return np.array([
        # travel  asset  threat  time  device  pattern
        [ 0.30,  -0.10, -0.25,  -0.15,  0.20,  0.25],  # false_positive_close
        [ 0.05,   0.20,  0.15,   0.10, -0.05,  0.05],  # escalate_tier2
        [-0.10,   0.05,  0.20,   0.20, -0.10, -0.05],  # enrich_and_wait
        [-0.25,   0.30,  0.30,   0.15, -0.20, -0.15],  # escalate_incident
    ], dtype=np.float64)
```

---

## 8. Tier 3 — Weight Learning (`gae/learning.py`)

### 8.1 What This Implements

**Eq. 4b:** W[a, :] ← W[a, :] + α_eff · r(t) · f(t) · δ(t)
**Eq. 4c:** W ← (1 − ε_vector) · W (per-factor decay)

Plus architectural hardening:
- **A1:** Confidence-discounted α (confirmation bias mitigation)
- **A2:** Per-factor temporal decay (permanent vs. campaign knowledge)
- **A4:** Soft discovery expansion with accelerated decay (false discovery protection)
- **C3:** Delayed outcome validation for autonomous decisions

> **v7 change from v5:** All hardcoded constants (ALPHA, LAMBDA_NEG, EPSILON_DEFAULT) replaced by CalibrationProfile (§16). Module-level constants retained as documentation/fallback. LearningState constructor accepts `profile: CalibrationProfile`.

### 8.2 Core Interface

```python
# gae/learning.py
"""
Tier 3: Hebbian Weight Updates — Eq. 4b, 4c from the math blog.

v7: All parameters from CalibrationProfile (§16). Module constants for reference:
    α = 0.02           base learning rate
    λ_neg = 20.0       asymmetric penalty multiplier
    ε = 0.001          default decay rate (half-life ~693 decisions)

Hardening parameters (calibrated post-v5.0):
    discount_strength = 0.5    A1: confirmation bias discount [0, 1]
    establishment_threshold = 50   A4: reinforcements to establish dimension
    θ_prune = 0.01             A4: max |W[:,i]| below which provisional dims pruned
    autonomous_validation_window = 14  C3: days before autonomous outcome trusted
"""

from .calibration import CalibrationProfile

# Module-level constants: documentation + fallback. LearningState reads from profile.
ALPHA = 0.02
LAMBDA_NEG = 20.0
EPSILON_DEFAULT = 0.001

@dataclass
class DimensionMetadata:
    """Tracks provisional vs. established dimensions (A4 hardening)."""
    factor_name: str
    created_at: int              # decision count when added
    state: str                   # "provisional" | "established" | "original"
    decay_rate: float            # provisional: 0.01 (fast), established: per-factor rate
    reinforcement_count: int     # how many times r(t) reinforced this dimension
    establishment_threshold: int = 50

@dataclass
class PendingValidation:
    """Autonomous decisions awaiting outcome validation (C3 hardening)."""
    alert_id: str
    action: str
    factor_vector: np.ndarray
    auto_decided_at: float
    validation_window_days: int = 14

@dataclass
class WeightUpdate:
    """Record of a single Eq. 4b update. Full provenance."""
    decision_number: int
    timestamp: float
    action_index: int
    action_name: str
    outcome: int
    factor_vector: np.ndarray
    delta_applied: np.ndarray
    W_before: np.ndarray
    W_after: np.ndarray
    alpha_effective: float       # after confidence discounting (A1)
    confidence_at_decision: float  # system's confidence when analyst decided

@dataclass
class LearningState:
    """Persistent state for the weight matrix and its history."""
    W: np.ndarray
    n_actions: int
    n_factors: int
    factor_names: List[str]
    profile: CalibrationProfile = field(default_factory=CalibrationProfile)
    decision_count: int = 0
    history: List[WeightUpdate] = field(default_factory=list)
    expansion_history: List[dict] = field(default_factory=list)
    dimension_metadata: List[DimensionMetadata] = field(default_factory=list)
    pending_validations: List[PendingValidation] = field(default_factory=list)

    # A2: Per-factor decay vector (one rate per factor)
    epsilon_vector: np.ndarray = None  # shape (n_factors,), built from CalibrationProfile
    
    # Factor decay class assignments (from domain schema or SchemaContracts)
    factor_decay_classes: List[str] = None

    def __post_init__(self):
        if self.epsilon_vector is None:
            self.epsilon_vector = self._build_epsilon_vector()

    def _build_epsilon_vector(self) -> np.ndarray:
        """Map each factor's decay_class → ε via CalibrationProfile."""
        if self.factor_decay_classes:
            return np.array([
                self.profile.get_decay_rate(dc) 
                for dc in self.factor_decay_classes
            ])
        return np.full(self.n_factors, self.profile.get_decay_rate("standard"))

    def update(
        self,
        action_index: int,
        action_name: str,
        outcome: int,
        f: np.ndarray,
        confidence_at_decision: float = None,
        decision_source: str = "analyst",
    ) -> WeightUpdate:
        """
        Eq. 4b: W[a,:] ← W[a,:] + α_eff · r(t) · f(t) · δ(t)
        Eq. 4c: W *= (1 − ε_vector)

        Reads from self.profile:
          - profile.learning_rate (was ALPHA)
          - profile.penalty_ratio (was LAMBDA_NEG)
          - profile.discount_strength (A1)
          
        Hardening:
          A1: α_eff = α × (1 - discount_strength × max_confidence) for r(t)=+1
          A2: ε_vector provides per-factor decay rates (from profile.decay_class_rates)
          C3: Autonomous decisions deferred to pending_validations
        """
        assert outcome in (+1, -1), "r(t) must be +1 or -1"
        assert f.shape == (1, self.n_factors), f"f must be (1, {self.n_factors})"

        # C3: Autonomous decisions enter pending validation, not immediate learning
        if decision_source == "autonomous":
            self.pending_validations.append(PendingValidation(
                alert_id=f"auto-{self.decision_count}",
                action=action_name,
                factor_vector=f.copy(),
                auto_decided_at=time.time(),
            ))
            return None  # no immediate learning signal

        W_before = self.W.copy()

        # Eq. 4b: asymmetric δ(t) — from profile
        delta_t = 1.0 if outcome == +1 else self.profile.penalty_ratio

        # A1: Confidence-discounted learning rate (only for confirmations) — from profile
        alpha = self.profile.learning_rate
        if outcome == +1 and confidence_at_decision is not None:
            confidence_discount = 1.0 - (self.profile.discount_strength * confidence_at_decision)
            alpha = self.profile.learning_rate * max(confidence_discount, 0.05)  # floor at 5% of base

        update_vector = alpha * outcome * f.flatten() * delta_t
        self.W[action_index, :] += update_vector

        # Eq. 4c: per-factor multiplicative decay (A2 hardening)
        self.W *= (1 - self.epsilon_vector)

        # A4: Update reinforcement counts for provisional dimensions
        for i, dm in enumerate(self.dimension_metadata):
            if dm.state == "provisional" and abs(update_vector[i]) > 1e-6:
                dm.reinforcement_count += 1
                if dm.reinforcement_count >= dm.establishment_threshold:
                    dm.state = "established"
                    self.epsilon_vector[i] = dm.decay_rate  # switch to normal decay

        # A4: Prune provisional dimensions that decayed to near-zero
        self._prune_provisional_dimensions()

        self.decision_count += 1

        record = WeightUpdate(
            decision_number=self.decision_count,
            timestamp=time.time(),
            action_index=action_index,
            action_name=action_name,
            outcome=outcome,
            factor_vector=f.copy(),
            delta_applied=update_vector,
            W_before=W_before,
            W_after=self.W.copy(),
            alpha_effective=alpha,
            confidence_at_decision=confidence_at_decision or 0.0,
        )
        self.history.append(record)
        return record

    def _prune_provisional_dimensions(self, theta_prune: float = 0.01):
        """A4: Remove provisional dimensions that decayed to near-zero."""
        to_remove = []
        for i, dm in enumerate(self.dimension_metadata):
            if dm.state == "provisional":
                if np.max(np.abs(self.W[:, i])) < theta_prune:
                    to_remove.append(i)
        if to_remove:
            keep = [i for i in range(self.n_factors) if i not in to_remove]
            self.W = self.W[:, keep]
            self.epsilon_vector = self.epsilon_vector[keep]
            self.factor_names = [self.factor_names[i] for i in keep]
            self.dimension_metadata = [self.dimension_metadata[i] for i in keep]
            self.n_factors = len(keep)

    def process_pending_validations(self, incident_checker):
        """C3: Check expired autonomous validations and apply learning."""
        now = time.time()
        expired = [pv for pv in self.pending_validations
                   if (now - pv.auto_decided_at) > pv.validation_window_days * 86400]
        for pv in expired:
            r_t = -1 if incident_checker(pv.alert_id) else +1
            self.update(
                action_index=self._action_index(pv.action),
                action_name=pv.action,
                outcome=r_t,
                f=pv.factor_vector,
                decision_source="validated_autonomous"
            )
            self.pending_validations.remove(pv)

    def get_convergence_metrics(self) -> dict:
        """Convergence metrics + three failure mode detection."""
        if len(self.history) < 2:
            return {"converged": False, "decisions": self.decision_count}

        norms = [np.linalg.norm(h.W_after) for h in self.history]
        recent = norms[-10:] if len(norms) >= 10 else norms
        stability = float(np.std(recent))

        window = 20
        recent_outcomes = [h.outcome for h in self.history[-window:]]
        accuracy = sum(1 for o in recent_outcomes if o == +1) / len(recent_outcomes)

        # Failure mode 1: action confusion (all actions → similar scores)
        # Failure mode 2: asymmetric oscillation (alternating correct/incorrect)
        # Failure mode 3: decay-rate competition

        return {
            "decisions": self.decision_count,
            "weight_norm": float(norms[-1]),
            "stability": stability,
            "recent_accuracy": accuracy,
            "converged": stability < 0.05 and accuracy > 0.80,
            "provisional_dimensions": sum(1 for dm in self.dimension_metadata if dm.state == "provisional"),
            "pending_autonomous": len(self.pending_validations),
        }
```

### 8.3 W Expansion Protocol (Requirement R5 + A4 Hardening)

```python
    def expand_weight_matrix(self, new_factor_name: str, init_scale: float = 0.05):
        """
        Meta Loop: W gains a column when a new scoring dimension is discovered.
        W: (n_actions × n_factors) → (n_actions × (n_factors + 1))

        A4 Hardening: New dimensions enter as "provisional" with accelerated
        decay (10× normal). After establishment_threshold reinforcements,
        transitions to "established". Provisional dimensions that decay to
        near-zero are pruned — false discoveries self-correct.
        """
        new_column = np.random.randn(self.n_actions, 1) * init_scale
        self.W = np.hstack([self.W, new_column])

        provisional_decay = 0.01  # 10× faster than standard (half-life ~69 decisions)
        self.epsilon_vector = np.append(self.epsilon_vector, provisional_decay)

        self.n_factors += 1
        self.factor_names.append(new_factor_name)
        self.dimension_metadata.append(DimensionMetadata(
            factor_name=new_factor_name,
            created_at=self.decision_count,
            state="provisional",
            decay_rate=provisional_decay,
            reinforcement_count=0,
        ))

        self.expansion_history.append({
            "decision_number": self.decision_count,
            "new_factor": new_factor_name,
            "new_shape": list(self.W.shape),
            "trigger": "discovery",
            "state": "provisional",
        })
```

### 8.4 Three-Layer Decay Design (Per-Factor Decay Configuration — A2)

> **v7 replaces the v5 static table.** v5 §8.4 showed a fixed per-factor decay table. v7 introduces the three-layer design where decay rates flow from domain schema through CalibrationProfile to the learning loop.

```
┌───────────────────────────────────────────────────────────┐
│ Layer 1: Domain Schema (domain expertise)                  │
│   "travel_match is campaign-class"                         │
│   "asset_criticality is permanent-class"                   │
│   Source: domain_schema.yaml in copilot repo               │
├───────────────────────────────────────────────────────────┤
│ Layer 2: CalibrationProfile (operational tuning)           │
│   "For SOC, campaign = 0.003, permanent = 0.0001"          │
│   "For S2P, campaign = 0.005, permanent = 0.0001"          │
│   Source: CalibrationProfile in DomainConfig                │
├───────────────────────────────────────────────────────────┤
│ Layer 3: GAE Learning Loop (math)                          │
│   factor → schema lookup → decay class → profile lookup    │
│   → ε value → Eq. 4c: W *= (1 - ε)                        │
│   Source: gae/learning.py                                   │
└───────────────────────────────────────────────────────────┘
```

**Reference decay rates (SOC domain):**

| Factor               | Decay Class | ε      | Half-life          |
|----------------------|-------------|--------|--------------------|
| asset_criticality    | permanent   | 0.0001 | ~6,930 decisions   |
| vip_status           | permanent   | 0.0001 | ~6,930 decisions   |
| travel_match         | campaign    | 0.003  | ~231 decisions     |
| device_trust         | standard    | 0.001  | ~693 decisions     |
| time_anomaly         | standard    | 0.001  | ~693 decisions     |
| pattern_history      | campaign    | 0.003  | ~231 decisions     |

Permanent: insider threat indicators. Campaign: active threat TTPs. Adding a new decay class is a schema + profile config change. No GAE code changes required.

---

## 9. Tier 4 — Entity Embeddings (`gae/embeddings.py`) [v5.5]

### 9.1 What This Implements

**Eq. 5:** Eᵢ — shape (mᵢ × d), entity embedding matrix per domain.

### 9.2 EmbeddingProvider Protocol

> **v7 change from v5:** PropertyEmbedder replaced by EmbeddingProvider protocol, supporting multiple embedding strategies. PropertyEmbeddingProvider (numpy-only, default) and TransformerEmbeddingProvider (optional [embeddings] extra).

```python
# gae/embeddings.py
"""
Tier 4: Entity Embeddings — Eq. 5.

v7: EmbeddingProvider protocol with two implementations:
  - PropertyEmbeddingProvider: numpy-only, ships with GAE core (default)
  - TransformerEmbeddingProvider: sentence-transformers, optional [embeddings] extra

v5.5: property-based embeddings (no training required).
    1. Extract numeric/categorical properties from each entity
    2. Normalize per dimension (z-score)
    3. L2 unit-norm per entity vector
    (Validated by math blog Experiment 2: F1=0.293, 110× above random)

Future (v6+): Learned embeddings via GNN or contrastive learning.
The EmbeddingProvider protocol and primitives.py backend-swappable
design mean a GNN-based embedder contributed by the community plugs in
without changing the engine's API.
"""

EMBEDDING_DIM = 128

from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for entity embedding computation.
    
    Two implementations:
    - PropertyEmbeddingProvider: numpy-only, ships with GAE core (default)
    - TransformerEmbeddingProvider: sentence-transformers, optional [embeddings] extra
    
    Domain copilots can implement custom providers for domain-specific embeddings.
    """
    def embed_entities(
        self, 
        entities: list[dict],          # [{id, properties...}, ...]
        schema: 'DomainSchemaSpec',    # For property extraction guidance
    ) -> tuple[np.ndarray, list[str]]: # (E: m×d, entity_ids)
        ...


class PropertyEmbeddingProvider:
    """Default provider. NumPy-only. Property-based embeddings.
    
    Validated by math blog Experiment 2: F1=0.293, 110× above random.
    Quality degrades with missing data and scale — B3 gate tests this.

    Calibration step from math blog:
    (a) z-score normalization per feature dimension
    (b) L2 unit-norm per entity vector

    "Without at least one of these alignment mechanisms,
     cross-domain dot products are uninterpretable."
     — Math blog, §4
    """
    def __init__(self, d: int = EMBEDDING_DIM):
        self.d = d
    
    def embed_entities(self, entities, schema):
        if not entities:
            return np.zeros((0, self.d)), []
        
        # Extract properties based on schema
        properties = [p.name for p in schema.factors[0].required_properties] if schema else []
        m = len(entities)
        raw = np.zeros((m, len(properties)))
        entity_ids = []
        
        for i, entity in enumerate(entities):
            entity_ids.append(entity.get("id", f"idx-{i}"))
            for j, prop in enumerate(properties):
                val = entity.get(prop, 0.0)
                raw[i, j] = float(val) if val is not None else 0.0

        # (a) Z-score normalization per dimension
        means = raw.mean(axis=0)
        stds = raw.std(axis=0)
        stds[stds == 0] = 1.0
        normalized = (raw - means) / stds

        # Pad/truncate to embedding dimension d
        if normalized.shape[1] < self.d:
            padded = np.zeros((m, self.d))
            padded[:, :normalized.shape[1]] = normalized
            normalized = padded
        elif normalized.shape[1] > self.d:
            normalized = normalized[:, :self.d]

        # (b) L2 unit-norm per entity vector
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        E = normalized / norms

        assert E.shape == (m, self.d), f"Eᵢ must be ({m}, {self.d}), got {E.shape}"
        return E, entity_ids


class TransformerEmbeddingProvider:
    """Optional provider. Requires [embeddings] extra.
    
    Better quality embeddings via sentence-transformers.
    Fallback if PropertyEmbeddingProvider fails B3 gate (F1 < 0.2).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", d: int = 384):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "TransformerEmbeddingProvider requires sentence-transformers. "
                "Install with: pip install graph-attention-engine[embeddings]"
            )
        self.model = SentenceTransformer(model_name)
        self.d = d
    
    def embed_entities(self, entities, schema):
        # Convert entity properties to text descriptions, encode via transformer
        pass
```

### 9.3 Legacy Interface (Copilot-Level — Reference)

```python
# For copilots using Neo4j directly (reference — NOT in GAE)
class PropertyEmbedder:
    """Legacy embedder using async Neo4j queries.
    Copilots can use this or implement EmbeddingProvider directly."""

    def __init__(self, feature_schema: Dict[str, List[str]], d: int = EMBEDDING_DIM):
        self.feature_schema = feature_schema
        self.d = d

    async def compute_domain_embeddings(
        self, domain_name: str, neo4j: AsyncSession,
        embedding_contract: 'EmbeddingContract' = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute Eᵢ for domain i from Neo4j graph."""
        # ... same implementation as v5 PropertyEmbedder
        pass
```

---

## 10. Tier 5 — Cross-Graph Attention & Discovery [v5.5]

### 10.1 What This Implements

**Eq. 6:** CrossAttention(Gᵢ, Gⱼ) = softmax(Eᵢ · Eⱼᵀ / √d) · Vⱼ
**Eq. 7a:** Sᵢ,ⱼ = Eᵢ · Eⱼᵀ / √d (logit matrix)
**Eq. 7b:** Aᵢ,ⱼ = softmax(Sᵢ,ⱼ) (attention matrix, row-normalized)
**Eq. 8a-8c:** Discovery extraction (threshold + top-K + margin)
**Eq. 9:** MultiDomainAttention — n(n-1)/2 heads

### 10.2 Attention Primitive (`gae/primitives.py`)

```python
# gae/primitives.py
"""
Backend-swappable attention primitive.
NumPy (v5.0/v5.5) → PyTorch SDPA (v6.0) → Flash Attention (v6.5+).
Callers never import backend-specific code (P6).
"""

_BACKEND = "numpy"

def get_backend() -> str:
    return _BACKEND

def set_backend(name: str):
    global _BACKEND
    _BACKEND = name

def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray = None,
    scale: float = None, mask: np.ndarray = None,
    backend: str = None
):
    """
    Eq. 6 from math blog / Eq. 1 from Vaswani et al.

    Returns: (output, logits, attention_weights)
        output: (n × dv)
        logits: (n × m) — Eq. 7a
        attention_weights: (n × m) — Eq. 7b
    """
    if V is None:
        V = K
    if scale is None:
        scale = Q.shape[-1]
    if backend is None:
        backend = _BACKEND

    if backend == "numpy":
        S = (Q @ K.T) / np.sqrt(scale)
        if mask is not None:
            S = np.where(mask, S, -1e9)
        S_shifted = S - S.max(axis=1, keepdims=True)
        exp_S = np.exp(S_shifted)
        A = exp_S / exp_S.sum(axis=1, keepdims=True)
        return A @ V, S, A

    elif backend == "torch":
        import torch
        import torch.nn.functional as F
        q = torch.from_numpy(Q).unsqueeze(0).float()
        k = torch.from_numpy(K).unsqueeze(0).float()
        v = torch.from_numpy(V).unsqueeze(0).float()
        out = F.scaled_dot_product_attention(q, k, v, scale=1.0/np.sqrt(scale))
        return out.squeeze(0).numpy(), None, None
```

### 10.3 Cross-Graph Attention (`gae/attention.py`)

```python
# gae/attention.py
"""Tier 5: Cross-Graph Attention — Eq. 6, 7a, 7b, 9."""

from .primitives import scaled_dot_product_attention
from itertools import combinations

@dataclass
class CrossAttentionResult:
    source_domain: str
    target_domain: str
    logit_matrix: np.ndarray        # Eq. 7a: (mᵢ, mⱼ)
    attention_matrix: np.ndarray    # Eq. 7b: (mᵢ, mⱼ)
    enriched: np.ndarray            # Eq. 6 output: (mᵢ, dᵥ)

def cross_attention(E_i, E_j, V_j=None, backend="numpy") -> CrossAttentionResult:
    """Eq. 6: CrossAttention(Gᵢ, Gⱼ) = softmax(Eᵢ · Eⱼᵀ / √d) · Vⱼ"""
    assert E_i.shape[1] == E_j.shape[1], "Embedding dims must match"
    enriched, S, A = scaled_dot_product_attention(Q=E_i, K=E_j, V=V_j, backend=backend)
    return CrossAttentionResult(
        source_domain="", target_domain="",
        logit_matrix=S, attention_matrix=A, enriched=enriched,
    )

def multi_domain_attention(embeddings: dict, backend="numpy") -> List[CrossAttentionResult]:
    """Eq. 9: n domains → n(n-1)/2 heads. n=6 → 15 heads."""
    domain_names = sorted(embeddings.keys())
    results = []
    for d_i, d_j in combinations(domain_names, 2):
        E_i, ids_i = embeddings[d_i]
        E_j, ids_j = embeddings[d_j]
        if E_i.shape[0] == 0 or E_j.shape[0] == 0:
            continue
        result = cross_attention(E_i, E_j, backend=backend)
        result.source_domain = d_i
        result.target_domain = d_j
        results.append(result)
    return results
```

### 10.4 Discovery Extraction (`gae/discovery.py`)

```python
# gae/discovery.py
"""
Discovery extraction — Eq. 8a, 8b, 8c.

Eq. 8a: S[k, l] > θ_logit           (absolute compatibility)
Eq. 8b: l ∈ top-K(A[k, :])          (relative salience)
Eq. 8c: s_best - s_runner_up ≥ Δ    (margin / discriminative confidence)

"In 500×300 logit matrix: 150K pairs → 500-1K after threshold
 → ~1,500 after top-K → 30-50 bidirectional discoveries per sweep."
"""

THETA_LOGIT = 0.5
TOP_K = 3
MARGIN_DELTA = 0.15

@dataclass
class Discovery:
    source_domain: str
    source_entity_id: str
    source_entity_index: int
    target_domain: str
    target_entity_id: str
    target_entity_index: int
    logit_score: float
    attention_weight: float
    margin: float
    confidence: float
    bidirectional: bool

def extract_discoveries(
    result: CrossAttentionResult,
    source_ids: List[str], target_ids: List[str],
    theta_logit=THETA_LOGIT, top_k=TOP_K, margin_delta=MARGIN_DELTA,
) -> List[Discovery]:
    """Two-stage extraction: Eq. 8a (threshold) → 8b (top-K) → 8c (margin)."""
    S = result.logit_matrix
    A = result.attention_matrix
    m_i, m_j = S.shape
    discoveries = []

    for k in range(m_i):
        row_logits = S[k, :]
        row_attention = A[k, :]

        if m_j <= top_k:
            top_indices = np.arange(m_j)
        else:
            top_indices = np.argpartition(row_attention, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(row_logits[top_indices])[::-1]]

        for rank, l in enumerate(top_indices):
            logit = float(row_logits[l])
            attention = float(row_attention[l])
            if logit < theta_logit:
                continue
            if rank == 0 and len(top_indices) > 1:
                runner_up = float(row_logits[top_indices[1]])
                margin = logit - runner_up
            else:
                margin = 0.0
            if rank == 0 and margin < margin_delta:
                continue

            confidence = float(np.sqrt(
                min(logit / 2.0, 1.0) * min(attention * m_j, 1.0)
            ))
            discoveries.append(Discovery(
                source_domain=result.source_domain,
                source_entity_id=source_ids[k] if k < len(source_ids) else f"idx-{k}",
                source_entity_index=k,
                target_domain=result.target_domain,
                target_entity_id=target_ids[l] if l < len(target_ids) else f"idx-{l}",
                target_entity_index=l,
                logit_score=logit, attention_weight=attention,
                margin=margin, confidence=confidence, bidirectional=False,
            ))
    return discoveries
```

---

## 11. Tier 6 — Artifact Evolution [v6.0] (ci-platform repo)

> **Note:** Tier 6 implementation lives in the ci-platform repository (`platform/agents/`, `platform/gae/evolution.py`). This section provides the conceptual overview. Full specification in ci_platform_design_v1.

### 11.1 What Tier 6 Does

Tiers 1-5 change what the system KNOWS (weights, discoveries). Tier 6 changes what the system DOES (prompts, routing rules, tool configurations). This is the second learning loop — orthogonal to weight learning (Connector 3). Weight learning makes the same pipeline produce better scores. Artifact evolution makes a better pipeline that weight learning then further calibrates.

### 11.2 GAE's Role in Tier 6

GAE provides scoring primitives that the evolution pipeline uses:
- `score_alert()` — evaluates artifact variants against scenarios
- `LearningState.get_convergence_metrics()` — detects if an artifact swap destabilized learning
- Event types (`ArtifactPromoted`, `ArtifactRejected`) — defined in `gae/events.py`

The evolution pipeline itself (meta-prompt agent, safeguard agent, R×T×Q evaluation, promotion state machine) is platform infrastructure, not core math. See ci_platform_design_v1 for full specification.

### 11.3 Connector 6 (Platform-Level)

| Connector | Input → Output | Lives In |
|---|---|---|
| C6: Outcomes → Artifact Evaluation → Artifact Mutation | Decision outcomes → variant generation → evaluation → promotion | ci-platform |

GAE emits events that C6 consumes. GAE does not implement C6.
## 12. Demand-Side Contracts (`gae/contracts.py`)

### 12.1 The Silent Dependency

Every FactorComputer encodes ontological claims in Cypher. If a connector writes `:Employee` instead of `:User`, or `:TRAVELED_TO` instead of `:HAS_TRAVEL`, the factor returns 0.0. The scoring matrix produces a confident decision. The decision is wrong — not because the math failed, but because the query found nothing in a graph that contains the answer under a different name.

**A real math engine running on ungoverned data is more dangerous than a fake math engine on seed data, because the real engine's outputs are trusted.**

### 12.2 What Lives in GAE vs ci-platform

| Component | Nature | Repo | Why |
|---|---|---|---|
| SchemaContract (dataclass) | Demand — "I need X" | **GAE** | Pure Python. Coupled to FactorComputer. |
| EmbeddingContract (dataclass) | Demand — "I need Y" | **GAE** | Pure Python. Coupled to PropertyEmbedder. |
| PropertySpec (dataclass) | Foundation for both | **GAE** | Used by both contract types. |
| Contract validation (check graph) | Infrastructure | **ci-platform** | Needs Neo4j to verify graph matches contracts. |
| DomainOntology (definition) | Supply — "graph contains Z" | **ci-platform** | Governance. Domain-structural. |
| SchemaValidator (enforcement) | Supply — "reject bad writes" | **ci-platform** | Write-path. Interacts with connectors. |
| Schema drift detection (C1) | Supply — "quality degraded" | **ci-platform** | Runtime monitoring. |

**The principle:** GAE contracts are **declarative**. Platform validation is **operational**. GAE says "I need X." Platform checks "X exists and is healthy." UCL ensures "only valid X gets written."

### 12.3 GAE Contract Definitions (Pure Python, numpy-only)

```python
# gae/contracts.py
"""
Demand-side contracts: what computation requires from the graph.

These are pure Python dataclasses with ZERO external dependencies.
A researcher using GAE standalone gets self-documenting factor requirements.
A production system using ci-platform gets runtime validation against Neo4j.
"""

@dataclass
class PropertySpec:
    """Specification for a single entity property."""
    name: str
    dtype: str                     # "float", "str", "datetime", "bool"
    required: bool = True
    missing_handling: str = "zero" # "zero", "mean", "skip", "error"
    decay_class: str = "standard"  # "permanent", "standard", "campaign", "transient"
    value_domain: Optional[list] = None

@dataclass
class SchemaContract:
    """What a FactorComputer requires from the graph."""
    factor_name: str
    required_labels: List[str]
    required_relationships: List[tuple]  # (source, rel_type, target)
    required_properties: Dict[str, List[str]]  # {label: [property_names]}
    optional_properties: Dict[str, List[str]] = field(default_factory=dict)
    minimum_node_count: Dict[str, int] = field(default_factory=dict)
    decay_rate: float = 0.001      # A2: per-factor temporal decay
    decay_class: str = "standard"  # "permanent" | "standard" | "campaign" | "transient"
    
    def describe(self) -> dict:
        """Machine-readable contract for validation by external systems."""
        return {
            "factor": self.factor_name,
            "labels": self.required_labels,
            "relationships": self.required_relationships,
            "properties": self.required_properties,
            "decay": {"rate": self.decay_rate, "class": self.decay_class},
        }

@dataclass
class EmbeddingContract:
    """What the embedder requires from entity properties."""
    domain_name: str
    properties: List[PropertySpec]
    minimum_coverage: float = 0.85  # 85% of entities must have required props
    alignment: str = "l2_unit_norm" # normalization method
```

### 12.4 Supply-Side Governance (ci-platform — reference only)

Write-path validation lives in ci-platform. See ci_platform_design_v1 for:
- `DomainOntology` — machine-readable ontology: labels, relationships, property schemas, aliases
- `SchemaValidator` — gates the write path: connector → validate_and_transform() → Neo4j
- `ContractChecker` — validates GAE SchemaContracts against live graph at startup
- Schema drift detection (C1) — if coverage drops, emit `SchemaRegression` event

The dependency chain:
- v5.0: SchemaContracts in GAE = declaration (detect problems at development time)
- v5.5: EmbeddingContracts in GAE = quality declaration (detect data quality issues)
- v6.0: DomainOntology in ci-platform = enforcement (prevent problems)

### 12.5 The Confidence Gradient Problem

**A real math engine running on ungoverned data is more dangerous than a fake math engine on seed data, because the real engine's outputs are trusted.** Before GAE, everyone knows factors are fake. After GAE, they come with mathematical provenance that makes them HARDER to question — even when they're wrong for structural reasons invisible to every downstream component.

This is why contracts are not a nice-to-have. They are a correctness requirement at every tier:

- **Tier 1 — Factors:** A zero from "no matching path" is indistinguishable from a zero from "evidence absent." SchemaContracts distinguish them.
- **Tier 4 — Embeddings:** Missing properties get 0.0, z-score treats this as "below average," L2 norm spreads the error. EmbeddingContracts handle missing data deliberately.
- **Tier 5 — Attention:** At F1=0.293 on clean data, ~70% of candidates are false positives. Inconsistent schemas make it worse. Coverage annotations on discoveries flag data quality.
- **Meta Loop:** A false discovery can trigger W expansion. Soft expansion with accelerated decay (§8.3 A4 hardening) limits the damage, but preventing false discoveries via schema governance (ci-platform) is better.

---

## 13. Integration — Public API & Pipeline Reference

### 13.1 Public API (`gae/__init__.py`)

```python
# gae/__init__.py
"""
Graph Attention Engine — public API.
Tiers 1-3 (v4.1): score_alert(), LearningState, assemble_factor_vector()
Calibration (v4.5): CalibrationProfile
Evaluation/Judgment/Ablation (v5.0): EvaluationScenario, InstitutionalJudgmentMetrics, AblationConfig
Tiers 4-5 (v5.5): EmbeddingProvider, cross_attention(), extract_discoveries()
"""

# Tier 1 — Factor Protocol + Assembly
from .factors import FactorComputer, assemble_factor_vector

# Tier 2 — Scoring Matrix
from .scoring import score_alert, ScoringResult

# Tier 3 — Weight Learning
from .learning import LearningState, WeightUpdate, PendingValidation

# Calibration (v4.5)
from .calibration import CalibrationProfile, soc_calibration_profile

# Contracts — Demand-side declarations
from .contracts import SchemaContract, EmbeddingContract, PropertySpec

# Domain Schema Protocol (v4.5/v5.0)
from .schema import DomainSchemaSpec, FactorSpec, load_domain_schema

# Convergence Monitoring
from .convergence import get_convergence_metrics

# Evaluation Framework (v5.0)
from .evaluation import EvaluationScenario, EvaluationResult, EvaluationReport

# Institutional Judgment (v5.0)
from .judgment import InstitutionalJudgmentMetrics

# Ablation (v5.0)
from .ablation import AblationConfig, AblationReport, static_config, full_config

# Event Types (dataclasses, no bus)
from .events import (
    AlertArrived, DecisionMade, OutcomeVerified,
    GraphMutated, DiscoverySweepRequested, DiscoveryValidated,
    ArtifactEvaluationRequested, ArtifactPromoted,
)

# Primitives
from .primitives import scaled_dot_product_attention, get_backend

# State Persistence (JSON)
from .store import save_learning_state, load_learning_state

# Tier 4 [v5.5]
# from .embeddings import EmbeddingProvider, PropertyEmbeddingProvider

# Tier 5 [v5.5]
# from .attention import cross_attention, multi_domain_attention
# from .discovery import extract_discoveries, Discovery
```

### 13.2 Full Causal Chain — Copilot-Level Reference

> The full triage pipeline lives in the copilot repo (soc-copilot), NOT in GAE.
> GAE provides the computation primitives. The copilot wires them together with
> Neo4j, the event bus (from ci-platform), and domain-specific logic.
> Full specification in soc_copilot_design_v1.

```python
# soc-copilot: routers/triage.py (REFERENCE ONLY — NOT part of GAE)
from gae.scoring import score_alert
from gae.learning import LearningState
from gae.factors import assemble_factor_vector
from gae.events import DecisionMade, GraphMutated, OutcomeVerified
# from platform.events import GraphEventBus  # ci-platform repo

async def handle_alert(alert_data, domain_config, neo4j, learning_state, event_bus):
    """One alert through the full causal chain."""
    # CONNECTOR 1: factor computation (copilot orchestrates, GAE assembles)
    computers = domain_config.get_factor_computers()
    values = [await c.compute(alert_data, neo4j) for c in computers]
    names = [c.name for c in computers]
    f, metadata = assemble_factor_vector(values, names)  # GAE function

    # CONNECTOR 2: scoring (pure GAE)
    result = score_alert(f, learning_state.W, domain_config.get_actions(), tau=0.25)

    # CHANNEL A: decision write-back (copilot writes to Neo4j)
    decision_id = await write_decision_to_graph(alert_data, result, f, neo4j)

    # Events (types from GAE, bus from platform)
    await event_bus.emit(DecisionMade(...))
    await event_bus.emit(GraphMutated(...))
    return result

async def handle_outcome(decision_id, outcome, neo4j, learning_state, event_bus):
    """Connector 3: verified outcome -> weight update."""
    f_original = await retrieve_factor_vector_from_graph(decision_id, neo4j)  # R4
    update = learning_state.update(...)  # GAE function
    await mark_decision_outcome(decision_id, outcome, neo4j)  # CHANNEL B
    await event_bus.emit(OutcomeVerified(...))
```

### 13.3 Decision and Outcome Write-Back Specifications

These are the specific graph mutations that make Channels A and B work.

**Connector 2 Output — Decision Write-Back (Channel A):**

```python
async def write_decision_to_graph(alert_id, result, f, neo4j):
    """
    Channel A: Decision node carries f(t) per R4.
    This write-back is what makes the graph richer for the next cycle.
    PatternHistoryFactor reads these nodes.
    """
    query = """
    MATCH (a:Alert {id: $alert_id})
    CREATE (d:Decision {
        id: $decision_id,
        action: $action,
        confidence: $confidence,
        factor_vector: $factor_vector,
        timestamp: datetime(),
        outcome: null
    })
    CREATE (d)-[:DECIDED_ON]->(a)
    RETURN d.id
    """
    await neo4j.run(query,
        alert_id=alert_id,
        decision_id=f"DEC-{alert_id}-{int(time.time())}",
        action=result.selected_action,
        confidence=result.confidence,
        factor_vector=f.flatten().tolist()
    )
```

**Connector 3 Input/Output — Outcome Write-Back (Channel B):**

```python
async def record_outcome_to_graph(decision_id, outcome, neo4j):
    """
    Channel B: Retrieve f(t) from GRAPH per R4, mark outcome.
    PatternHistoryFactor reads d.correct to compute accuracy.
    """
    query = """
    MATCH (d:Decision {id: $decision_id})
    SET d.outcome = $outcome_label,
        d.outcome_timestamp = datetime(),
        d.correct = $correct
    RETURN d.factor_vector
    """
    result = await neo4j.run(query,
        decision_id=decision_id,
        outcome_label="correct" if outcome == +1 else "incorrect",
        correct=(outcome == +1)
    )
    record = await result.single()
    f_original = np.array(record["factor_vector"]).reshape(1, -1)  # R4
    return f_original
```

### 13.4 End-to-End Compounding Verification (R8 Gate)

This is the definitive test that the system compounds, not just computes. Run after all Tier 1-3 prompts are complete. **If any step fails, a causal link is broken.**

```
SETUP: Fresh graph seed. Learning state reset to domain priors.

CYCLE 1 — BASELINE:
  1. Analyze ALERT-7823 (travel anomaly for jsmith)
     Record: factors_1, scores_1, confidence_1, action_1
  2. VERIFY: Decision node written to Neo4j
     MATCH (d:Decision)-[:DECIDED_ON]->(a:Alert {id:'ALERT-7823'})
     → d exists, d.factor_vector is list of 6 floats
  3. VERIFY: PatternHistoryFactor = 0.5 (no resolved history yet)

CYCLE 2 — FIRST FEEDBACK:
  4. Submit outcome: correct for ALERT-7823
  5. VERIFY: f(t) retrieved from Decision node in Neo4j (R4), NOT from memory
  6. VERIFY: W changed (compare W_before vs W_after from WeightUpdate)
  7. VERIFY: Decision node marked:
     MATCH (d:Decision)-[:DECIDED_ON]->(a:Alert {id:'ALERT-7823'})
     → d.correct = true, d.outcome = "correct"
  8. VERIFY: OutcomeVerified event emitted, GraphMutated event emitted

CYCLE 3 — ACCUMULATION TEST:
  9.  Analyze ALERT-7824 (travel anomaly, DIFFERENT user)
  10. VERIFY: PatternHistoryFactor query finds 1 resolved decision:
      MATCH (d:Decision)-[:DECIDED_ON]->(a:Alert)
      WHERE a.situation_type = $type AND d.outcome IS NOT NULL
      → count = 1
  11. VERIFY: scores_3 ≠ scores_1
      (because W updated in Cycle 2 AND PatternHistory reads accumulated decision)
      THIS IS THE COMPOUNDING PROOF.

CYCLES 4-8: Submit correct outcomes for 4 more travel-type alerts.

CYCLE 9 — COMPOUNDING VISIBLE:
  12. Analyze another travel alert
  13. VERIFY: PatternHistoryFactor returns ~1.0 (5+ correct accumulated)
  14. VERIFY: Scoring matrix strongly favors fp_close for travel type
  15. VERIFY: confidence_9 > confidence_1 (significantly)
  16. VERIFY: Tab 4 shows REAL convergence curve (not pre-seeded data)

CYCLE 10 — TRUST ASYMMETRY:
  17. Submit outcome: INCORRECT
  18. VERIFY: Weight delta ~20x larger than correct outcomes
  19. VERIFY: Trust drops sharply
  20. Analyze next travel alert
  21. VERIFY: recommended action CHANGES from fp_close
      (one mistake outweighs multiple correct outcomes — 20:1 asymmetry)
```

**What this test proves:**
- Channel A works: Decision nodes accumulate (steps 2, 10)
- Channel B works: Outcome markings accumulate (steps 7, 13)
- R4 works: f(t) from graph, not memory (step 5)
- R8 works: each cycle's output becomes next cycle's input (step 11)
- Eq. 4b works: W adapts to outcomes (steps 6, 18)
- P11 works: events emitted for all mutations (step 8)
- Asymmetry works: incorrect outcomes dominate (steps 18-21)


---

## 14. Semantic Accumulation — How New Data Propagates

### 14.1 New Entity Ingested

```
TI-2026-SG-CRED ingested → Neo4j (properties: geo, severity, TTPs)
    → [GraphMutated: domain="ThreatIntel"]
    → (v5.5) TI domain embedding stale → re-embed → E₃ gains row
    → Cross-attention TI × DecisionHistory includes new entity as KEY
    → If high similarity to existing patterns → discovery candidate
    → If validated → [:CALIBRATED_BY] written
    → (Fast Loop) Next alert: threat_intel_enrichment factor sees new entity
      → factor value changes → scoring produces different action probabilities
```

### 14.2 New Connection Created

```
jsmith promoted to CFO → :User node updated (role, role_changed_at)
    → [GraphMutated: domain="Organizational"]
    → (v5.5) Org embedding stale → re-embed → jsmith vector changes
    → Cross-attention Org × DecisionHistory: jsmith now similar to
      executive-targeted alert patterns
    → Discovery: "past auto-close for jsmith should be recalibrated
      given executive threat profile"
    → [:CALIBRATED_BY] written; if new dimension → W expands (R5)
    → Next jsmith alert: factors different, scoring different
```

### 14.3 New Graph Domain Added

```
domains/supply_chain/ activated → new entities (Supplier, PO, RiskAssessment)
    → Embedding: E₇ (new domain)
    → n domains: 6 → 7. Attention heads: 15 → 21 (+6 new)
    → New cross-domain sweeps: SC × TI, SC × Org, SC × DecisionHistory
    → Discovery surface expands super-quadratically
    → ALL domains benefit from supply chain intelligence
```


---

## 15. Open-Source Strategy & Public API

### 15.1 Three-Repository Architecture

The HuggingFace ecosystem parallel:

| HuggingFace | Our Architecture | License | Repo |
|---|---|---|---|
| `transformers` | `graph-attention-engine` | Apache 2.0 | THIS repo |
| `hub` + `accelerate` + `datasets` | `ci-platform` | Apache 2.0 | Companion |
| Model repos (bert-base, etc.) | `soc-copilot`, `s2p-copilot` | Proprietary | Domain repos |

### 15.2 GAE Boundary (This Repo)

| Component | In GAE? | Rationale |
|---|---|---|
| `gae/scoring.py` | ✓ | Eq. 4 — pure math |
| `gae/learning.py` | ✓ | Eq. 4b, 4c — pure math + numpy |
| `gae/factors.py` | ✓ | FactorComputer Protocol + `assemble_factor_vector()` |
| `gae/contracts.py` | ✓ | Demand-side: SchemaContract, EmbeddingContract, PropertySpec |
| `gae/primitives.py` | ✓ | Backend-swappable attention primitive |
| `gae/embeddings.py` | ✓ | Eq. 5 — entity embeddings (v5.5) |
| `gae/attention.py` | ✓ | Eq. 6, 7, 9 — cross-graph attention (v5.5) |
| `gae/discovery.py` | ✓ | Eq. 8a-8c — discovery extraction (v5.5) |
| `gae/convergence.py` | ✓ | Three failure mode detection |
| `gae/events.py` | ✓ | Event TYPE dataclasses (no bus) |
| `gae/store.py` | ✓ | JSON persistence (no Neo4j) |
| `examples/` | ✓ | Hello World domain (v5.0) |
| Event bus (async dispatch) | ✗ → ci-platform | Infrastructure, needs asyncio |
| DomainOntology, SchemaValidator | ✗ → ci-platform | Supply-side governance, needs Neo4j |
| UCL entity resolution | ✗ → ci-platform | Infrastructure, needs Neo4j |
| Agents (meta-prompt, safeguard) | ✗ → ci-platform | Infrastructure |
| SOC factors, seed, UI, routers | ✗ → soc-copilot | Domain expertise, proprietary |

### 15.3 Public API Surface

```python
# gae/__init__.py — what consumers can import
from gae.scoring import score_alert, ScoringResult
from gae.learning import LearningState, WeightUpdate, PendingValidation
from gae.factors import FactorComputer, assemble_factor_vector
from gae.contracts import SchemaContract, EmbeddingContract, PropertySpec
from gae.primitives import scaled_dot_product_attention, get_backend
from gae.events import DecisionMade, OutcomeVerified, GraphMutated, DiscoveryValidated
from gae.store import save_learning_state, load_learning_state
from gae.convergence import get_convergence_metrics

# v5.5 additions:
from gae.embeddings import PropertyEmbedder, compute_domain_embeddings
from gae.attention import cross_attention, multi_domain_attention
from gae.discovery import extract_discoveries, Discovery
```

### 15.4 The Moat After Open-Sourcing

The engine is the math that makes knowledge graphs compound. Open-sourcing the math does not give away the moat. The moat is:

1. **The graph data.** A customer's context graph, accumulated over months, contains firm-specific institutional judgment. Switching means starting from zero.
2. **Domain expertise.** The SOC domain module encodes deep security knowledge. Anyone can build a domain module; building a good one requires domain expertise.
3. **Time-to-value.** The product (GAE + platform + domain + connectors + UI + deployment) is a complete solution. GAE alone is a library.
4. **Network effects.** The meta-graph (v7.0 cross-tenant patterns) compounds across customers. Product-level.
5. **Community as moat amplifier.** If GAE becomes the standard for compounding intelligence systems, being the canonical product on the leading engine is a stronger position than being a proprietary black box.

### 15.5 The Commercial Model

| Component | Repo | License | Revenue Model |
|---|---|---|---|
| GAE (math library) | graph-attention-engine | Apache 2.0 | Free + community contributions |
| Platform (infrastructure) | ci-platform | Apache 2.0 | Free + community contributions |
| SOC Domain Module | soc-copilot | Proprietary | Subscription |
| S2P Domain Module | s2p-copilot | Proprietary | Subscription |
| Connectors | soc-copilot | Mix | Value-add |
| Frontend/UI | soc-copilot | Proprietary | Part of subscription |
| Deployment + Cloud | soc-copilot | Proprietary | Hosting / managed service |
| Multi-tenant + Partner | soc-copilot | Proprietary | Enterprise license |

### 15.6 Release Timing

**v5.0:** API hardened through v4.1 development. B3 gate check run. Example domain built.

**v5.5:** GAE + ci-platform open-sourced. Tiers 1-5 implemented. Cross-graph discovery working. README, API reference published.

**Why v5.5 and not v5.0:** At v5.0 the engine is "just" a scoring matrix with weight learning — functional but not distinctive. At v5.5, cross-graph attention and discovery make it unique.

### 15.7 Example Domain — "Hello World"

```python
# examples/minimal_domain/config.py (sketch)
from gae.factors import FactorComputer, assemble_factor_vector
from gae.contracts import SchemaContract

class FreshnessFactor(FactorComputer):
    name = "freshness"
    contract = SchemaContract(
        factor_name="freshness",
        required_labels=["Item"],
        required_relationships=[("Item", "CREATED_AT", "Timestamp")],
        required_properties={"Item": ["id", "age_days"]},
    )
    async def compute(self, item, neo4j):
        return min(item["age_days"] / 30.0, 1.0)

# 3 factors, 3 actions. Full loop in < 50 lines of config.
```

**Test:** `python -m examples.minimal_domain.run` → processes 5 items, prints scores, updates weights, shows convergence.


---

## 16. CalibrationProfile (`gae/calibration.py`)

### 16.1 Design Rationale

The v5 design hardcodes learning hyperparameters as module-level constants in `learning.py` (ALPHA=0.02, LAMBDA_NEG=20.0, EPSILON_DEFAULT=0.001). This prevents multi-domain usage — SOC needs 20:1 penalty asymmetry, but S2P procurement might need 5:1 (wrong approvals are costly but not $4.44M costly). CalibrationProfile makes these domain-configurable without touching GAE internals.

**Ownership split:**
- **GAE** defines CalibrationProfile (the dataclass) and LearningState (the consumer)
- **DomainConfig** (in ci-platform or copilot) returns a CalibrationProfile
- **SOCDomainConfig** provides SOC-specific values; a future S2PDomainConfig provides different ones

### 16.2 Interface

```python
# gae/calibration.py
"""
CalibrationProfile: domain-configurable hyperparameters for the GAE learning loop.

Core parameters (v4.5):
    learning_rate      α in Eq. 4b. Controls update magnitude per decision.
    penalty_ratio      λ_neg in Eq. 4b. Asymmetric penalty multiplier (20:1 for SOC).
    temperature        τ in Eq. 4. Softmax sharpness for action selection.
    decay_class_rates  Maps decay class names → ε values for Eq. 4c.
    discount_strength  A1 confirmation bias mitigation. 0.0 = disabled.

Extension point:
    extensions         Dict for experimental parameters. Graduate to named fields when stable.

Future candidates (documented, not committed):
    convergence_threshold      Minimum decisions before declaring convergence
    cold_start_window          Decisions during which learning rate is elevated
    factor_overrides           Per-factor learning rate or penalty overrides
    exploration_rate           Initial softmax temperature boost for cold start
"""

from dataclasses import dataclass, field
from typing import Any

# Decay class defaults — domain-configurable via CalibrationProfile
DEFAULT_DECAY_CLASS_RATES = {
    "permanent": 0.0001,    # Half-life ~6,930 decisions. Asset types, org structure.
    "standard": 0.001,      # Half-life ~693 decisions. Default.
    "campaign": 0.003,      # Half-life ~231 decisions. Threat campaigns, project phases.
    "transient": 0.01,      # Half-life ~69 decisions. Spot prices, breaking alerts.
}

@dataclass
class CalibrationProfile:
    """Domain-configurable hyperparameters for GAE learning loop.
    
    Every field has a sensible default. A domain that only cares about
    penalty_ratio passes CalibrationProfile(penalty_ratio=5.0) and gets
    standard defaults for everything else. New parameters added to future
    versions don't break existing consumers.
    """
    # Core learning parameters
    learning_rate: float = 0.02
    penalty_ratio: float = 20.0
    temperature: float = 0.25
    
    # Per-factor decay (A2): maps decay class → ε value
    # Domain copilot's SchemaContract declares each factor's decay_class.
    # CalibrationProfile maps that class name to a numeric rate.
    decay_class_rates: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DECAY_CLASS_RATES)
    )
    
    # Confirmation bias mitigation (A1)
    # 0.0 = disabled (v4.5 default). SOC tunes to ~0.3-0.5 at v5.0.
    # Formula: α_eff = α × (1 - discount_strength × max(P))
    discount_strength: float = 0.0
    
    # Extension point for experimental parameters
    extensions: dict[str, Any] = field(default_factory=dict)
    
    def get_decay_rate(self, decay_class: str) -> float:
        """Look up ε for a decay class. Falls back to 'standard' if unknown."""
        return self.decay_class_rates.get(
            decay_class, 
            self.decay_class_rates.get("standard", 0.001)
        )
    
    def validate(self) -> list[str]:
        """Return list of validation warnings (empty = valid)."""
        warnings = []
        if self.learning_rate <= 0 or self.learning_rate > 0.5:
            warnings.append(f"learning_rate {self.learning_rate} outside recommended range (0, 0.5]")
        if self.penalty_ratio < 1.0:
            warnings.append(f"penalty_ratio {self.penalty_ratio} < 1.0 means positive outcomes penalized more than negative")
        if self.temperature <= 0:
            warnings.append(f"temperature must be positive, got {self.temperature}")
        if self.discount_strength < 0 or self.discount_strength > 1.0:
            warnings.append(f"discount_strength {self.discount_strength} outside [0, 1]")
        return warnings


# Convenience constructors for known domains
def soc_calibration_profile() -> CalibrationProfile:
    """SOC domain defaults: high asymmetry, standard temperature."""
    return CalibrationProfile(
        learning_rate=0.02,
        penalty_ratio=20.0,       # One missed threat ≈ $4.44M avg breach cost
        temperature=0.25,
        discount_strength=0.0,    # Enable at v5.0 after evaluation
    )

def s2p_calibration_profile() -> CalibrationProfile:
    """S2P domain defaults: moderate asymmetry, softer temperature.
    NOTE: Placeholder — validate with real procurement data."""
    return CalibrationProfile(
        learning_rate=0.02,
        penalty_ratio=5.0,        # Wrong approval costly but not catastrophic
        temperature=0.4,          # Softer — procurement decisions less time-critical
        decay_class_rates={
            "permanent": 0.0001,  # Supplier master data
            "standard": 0.001,
            "campaign": 0.005,    # Commodity price cycles (faster than threat campaigns)
            "transient": 0.02,    # Spot pricing
        },
        discount_strength=0.0,
    )
```

### 16.3 Impact on LearningState (§8 Changes)

LearningState constructor changes from hardcoded constants to CalibrationProfile:

```python
# BEFORE (v5):
ALPHA = 0.02
LAMBDA_NEG = 20.0
EPSILON_DEFAULT = 0.001

class LearningState:
    discount_strength: float = 0.5  # hardcoded
    ...

# AFTER (v6):
class LearningState:
    """Persistent state for the weight matrix and its history."""
    W: np.ndarray
    n_actions: int
    n_factors: int
    factor_names: List[str]
    profile: CalibrationProfile = field(default_factory=CalibrationProfile)
    decision_count: int = 0
    history: List[WeightUpdate] = field(default_factory=list)
    # ... rest unchanged

    def __post_init__(self):
        # Build epsilon_vector from factor decay classes + profile rates
        if self.epsilon_vector is None:
            self.epsilon_vector = self._build_epsilon_vector()
    
    def _build_epsilon_vector(self) -> np.ndarray:
        """Map each factor's decay_class → ε via CalibrationProfile."""
        # Requires factor_decay_classes to be set (from domain schema or SchemaContracts)
        if hasattr(self, 'factor_decay_classes') and self.factor_decay_classes:
            return np.array([
                self.profile.get_decay_rate(dc) 
                for dc in self.factor_decay_classes
            ])
        return np.full(self.n_factors, self.profile.get_decay_rate("standard"))

    def update(self, action_index, action_name, outcome, f, 
               confidence_at_decision=None, decision_source="analyst"):
        """
        Eq. 4b + 4c. Now reads from self.profile instead of module constants.
        """
        # ...
        delta_t = 1.0 if outcome == +1 else self.profile.penalty_ratio  # was LAMBDA_NEG
        
        alpha = self.profile.learning_rate  # was ALPHA
        if outcome == +1 and confidence_at_decision is not None:
            confidence_discount = 1.0 - (self.profile.discount_strength * confidence_at_decision)
            alpha = self.profile.learning_rate * max(confidence_discount, 0.05)
        
        update_vector = alpha * outcome * f.flatten() * delta_t
        self.W[action_index, :] += update_vector
        self.W *= (1 - self.epsilon_vector)  # per-factor decay unchanged
        # ... rest unchanged
```

### 16.4 Impact on score_alert (§7 Changes)

```python
# BEFORE (v5):
def score_alert(f, W, actions, tau=0.25):

# AFTER (v6):
def score_alert(f, W, actions, tau=None, profile=None):
    """
    Eq. 4. Temperature from profile if provided, else tau param, else 0.25.
    """
    if tau is None:
        tau = profile.temperature if profile else 0.25
    # ... rest unchanged
```

### 16.5 Three-Layer Decay Design (Replaces §8.4)

v5 §8.4 showed a static table of per-factor decay rates. v6 replaces this with the three-layer design:

```
┌───────────────────────────────────────────────────────────┐
│ Layer 1: Domain Schema (domain expertise)                  │
│   "travel_match is campaign-class"                         │
│   "asset_criticality is permanent-class"                   │
│   Source: domain_schema.yaml in copilot repo               │
├───────────────────────────────────────────────────────────┤
│ Layer 2: CalibrationProfile (operational tuning)           │
│   "For SOC, campaign = 0.003, permanent = 0.0001"          │
│   "For S2P, campaign = 0.005, permanent = 0.0001"          │
│   Source: CalibrationProfile in DomainConfig                │
├───────────────────────────────────────────────────────────┤
│ Layer 3: GAE Learning Loop (math)                          │
│   factor → schema lookup → decay class → profile lookup    │
│   → ε value → Eq. 4c: W *= (1 - ε)                        │
│   Source: gae/learning.py                                   │
└───────────────────────────────────────────────────────────┘
```

Adding a new decay class is a schema + profile config change. No GAE code changes required.


## 17. Evaluation Framework (`gae/evaluation.py`)

### 17.1 Design Rationale

Three consumers need a shared evaluation format:
- **EVAL-1** (v5.0): Accuracy measurement against ground truth
- **SEED-2** (v5.0): Planted patterns in realistic data
- **B3 gate** (v4.5 Phase C): Discovery test cases

The format is a GAE-level capability so any domain copilot can define evaluation scenarios and run them through the standard eval pipeline.

### 17.2 Core Data Structures

```python
# gae/evaluation.py
"""
Evaluation framework: scenario definition, execution, and reporting.

Three-tier progression:
    Tier 1 (v4.5): Bernoulli oracle — proves learning mechanism works
    Tier 2 (v5.0): Constructed scenarios — proves product makes correct decisions
    Tier 3 (v6.0): Live analyst decisions — proves product works in production

Ground truth: Constructed from domain techniques × graph context combinations.
Each scenario has planted signals with deterministic correct answers.
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EvaluationScenario:
    """A complete, self-contained test case. Deterministic.
    
    Run it twice, get the same result. No external dependencies.
    The graph_context specifies what must be true for this scenario.
    The test harness ensures those conditions exist before running.
    """
    # Identity
    scenario_id: str              # "SOC-T1078-FP-01", "S2P-DUAL-SOURCE-01"
    description: str              # Human-readable
    
    # Classification
    domain: str                   # "soc", "s2p", "finserv"
    category: str                 # Domain-specific: "travel_anomaly", "spend_exception"
    technique_id: Optional[str]   # ATT&CK for SOC ("T1078"), N/A for S2P
    confidence_tier: str          # "high" | "medium" | "low"
                                  # high: clear-cut. medium: reasonable disagreement.
                                  # low: genuinely ambiguous — lower accuracy expected.
    
    # Input — what the graph looks like when this alert fires
    graph_context: dict           # {
                                  #   "entities": [...],
                                  #   "relationships": [...],
                                  #   "absent": [...]  ← conditions that must NOT exist
                                  # }
    alert: dict                   # Alert/trigger parameters
    
    # Expected output
    expected_action: str          # "suppress", "escalate", "investigate", "approve", "hold"
    expected_dominant_factors: list[str]  # Which factors should drive the decision.
                                          # Getting the right answer for wrong reasons
                                          # is a false pass.
    
    # For discovery scenarios (B3 gate)
    planted_relationship: Optional[dict] = None     # Cross-graph connection to find
    expected_confidence_shift: Optional[dict] = None # {"before": 0.89, "after_direction": "decrease"}
    
    # For compounding test scenarios
    learning_prerequisite: Optional[dict] = None    # {
                                                     #   "min_prior_decisions": 20,
                                                     #   "prior_categories": ["travel_anomaly"],
                                                     #   "prior_correctness_rate": 0.9
                                                     # }
                                                     # Test harness uses simulation to generate
                                                     # prerequisite history before running scenario.


@dataclass
class EvaluationResult:
    """Result of running one scenario."""
    scenario_id: str
    actual_action: str
    actual_confidence: float
    actual_dominant_factors: list[str]
    action_correct: bool
    dominant_factors_correct: bool    # Did the right factors drive the decision?
    confidence_tier: str
    factor_values: dict[str, float]  # Raw factor values for debugging


@dataclass
class EvaluationReport:
    """Aggregate report across a scenario set."""
    total: int
    action_accuracy: float            # % scenarios where action was correct
    factor_accuracy: float            # % scenarios where dominant factors were correct
    combined_accuracy: float          # % where BOTH action AND factors correct
    
    by_category: dict[str, float]     # Accuracy per alert/trigger category
    by_confidence_tier: dict[str, float]  # Accuracy by difficulty
    by_category: dict[str, float]     # Accuracy per category (ATT&CK technique for SOC, procurement type for S2P, etc.)
    
    # Compounding metrics (only populated if learning_prerequisite scenarios present)
    cold_start_accuracy: Optional[float] = None    # Accuracy on scenarios with no prerequisite
    post_learning_accuracy: Optional[float] = None  # Accuracy on scenarios with prerequisite
    learning_delta: Optional[float] = None          # post - cold_start (the compounding proof)
    
    # Discovery metrics (only populated if planted_relationship scenarios present)
    discovery_precision: Optional[float] = None
    discovery_recall: Optional[float] = None
    discovery_f1: Optional[float] = None
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Evaluation: {self.total} scenarios",
            f"  Action accuracy: {self.action_accuracy:.1%}",
            f"  Factor accuracy: {self.factor_accuracy:.1%}",
            f"  Combined: {self.combined_accuracy:.1%}",
            f"  By tier: high={self.by_confidence_tier.get('high', 0):.1%}, "
            f"medium={self.by_confidence_tier.get('medium', 0):.1%}, "
            f"low={self.by_confidence_tier.get('low', 0):.1%}",
        ]
        if self.learning_delta is not None:
            lines.append(f"  Learning delta: +{self.learning_delta:.1%} "
                        f"(cold={self.cold_start_accuracy:.1%} → "
                        f"post={self.post_learning_accuracy:.1%})")
        if self.discovery_f1 is not None:
            lines.append(f"  Discovery F1: {self.discovery_f1:.3f} "
                        f"(P={self.discovery_precision:.3f}, R={self.discovery_recall:.3f})")
        return "\n".join(lines)
```

### 17.3 Evaluation Runner (v5.0 — design only)

```python
def run_evaluation(
    scenarios: list[EvaluationScenario],
    scoring_fn,           # score_alert or equivalent
    learning_state,       # LearningState instance
    graph_setup_fn,       # Async function to seed graph_context for a scenario
    factor_compute_fn,    # Async function to compute factors for an alert
) -> EvaluationReport:
    """
    Run a scenario set and produce an EvaluationReport.
    
    For each scenario:
      1. If learning_prerequisite: run simulation to generate prerequisite history
      2. Seed graph_context (entities, relationships, absent conditions)
      3. Compute factors
      4. Score alert
      5. Compare actual vs expected
    
    Implementation at v5.0 (EVAL-1 prompt). Design specified here for
    format stability across SEED-2 and B3 consumers.
    """
    pass  # v5.0 implementation
```

### 17.4 Scenario Examples

See `design_decisions_v1.md` for two worked examples:
- `SOC-T1078-FP-01`: Singapore travel login, clear false positive (high confidence)
- `SOC-T1078-DISC-01`: Same scenario with role change + threat campaign (discovery variant)

**Scenario design guidance:**
- 30-40 scenarios for Tier 2, distributed across categories × confidence tiers
- Each category should have at least one high, one medium, one low confidence scenario
- The `absent` field is critical — it declares what must NOT exist in the graph
- `expected_dominant_factors` prevents false passes from correct action via wrong reasoning


## 18. Domain Schema Format (GAE Protocol Side)

### 18.1 Design Rationale

Factor metadata (decay class, temporal nature, required graph structure) currently lives in three places: SchemaContract, CalibrationProfile, and seed scripts. The domain schema format creates a single source of truth.

GAE defines the format (protocol). Copilots provide the content (domain-specific schemas). ci-platform validates and enforces (runtime governance).

**Ontology maturity levels:**
- Level 0 (v4.1): Implicit — exists in seed scripts and Cypher queries
- Level 1 (v5.0): Declared + validated — `domain_schema.yaml` + ContractChecker
- Level 2 (v6.0): Enforced — SchemaValidator gates all write paths

### 18.2 Schema Format Definition

```python
# gae/schema.py
"""
Domain schema format definition.

GAE defines what a valid schema looks like.
Copilots provide actual schemas as YAML or JSON.
ci-platform validates schemas against the live graph.

This module provides:
- DomainSchemaSpec: Python representation of a domain schema
- load_domain_schema(): Parse YAML/JSON into DomainSchemaSpec
- validate_schema_contracts(): Check SchemaContracts match schema
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class FactorSpec:
    """Schema entry for a factor."""
    name: str
    decay_class: str              # "permanent", "standard", "campaign", "transient"
    description: str
    required_labels: list[str]
    required_relationships: list[tuple[str, str, str]]  # (source, rel_type, target)
    required_properties: dict[str, list[str]]
    computes_from: str            # "graph_traversal" | "property_read" | "derived"

@dataclass
class EntitySpec:
    """Schema entry for an entity type (graph label)."""
    label: str
    properties: dict[str, dict]   # property_name → {"dtype": "float", "required": bool, ...}
    description: str

@dataclass
class RelationshipSpec:
    """Schema entry for a relationship type."""
    type: str
    source_label: str
    target_label: str
    properties: dict[str, dict]   # Optional relationship properties
    cardinality: str              # "one_to_one" | "one_to_many" | "many_to_many"
    description: str

@dataclass
class DomainSchemaSpec:
    """Complete domain schema. One per domain copilot."""
    domain: str                   # "soc", "s2p", "finserv"
    version: str
    description: str
    
    entities: list[EntitySpec]
    relationships: list[RelationshipSpec]
    factors: list[FactorSpec]
    
    # Actions this domain supports
    actions: list[dict]           # [{"name": "suppress", "description": "..."}, ...]
    
    # Decay class definitions (may extend standard set)
    decay_classes: dict[str, dict] = field(default_factory=dict)
    # e.g., {"campaign": {"description": "Active threat TTPs", "default_rate": 0.003}}
    
    def get_factor_decay_classes(self) -> dict[str, str]:
        """Map factor_name → decay_class for CalibrationProfile consumption."""
        return {f.name: f.decay_class for f in self.factors}
    
    def validate_against_contracts(self, contracts: list) -> list[str]:
        """Check that SchemaContracts are consistent with this schema."""
        warnings = []
        schema_labels = {e.label for e in self.entities}
        for contract in contracts:
            for label in contract.required_labels:
                if label not in schema_labels:
                    warnings.append(
                        f"Contract {contract.factor_name} requires label '{label}' "
                        f"not in schema"
                    )
        return warnings
```

### 18.3 Example Schema File (SOC — reference)

```yaml
# domains/soc/domain_schema.yaml
domain: soc
version: "1.0"
description: "Security Operations Center — alert triage and response"

entities:
  - label: User
    properties:
      id: {dtype: str, required: true}
      department: {dtype: str, required: true}
      risk_score: {dtype: float, required: false}
      role: {dtype: str, required: false}
    description: "Enterprise user identity"
    
  - label: Alert
    properties:
      id: {dtype: str, required: true}
      situation_type: {dtype: str, required: true}
      source_ip: {dtype: str, required: false}
      geo: {dtype: str, required: false}
      technique_id: {dtype: str, required: false}  # ATT&CK
    description: "Security alert from SIEM or detection system"

  - label: ThreatIntel
    properties:
      severity: {dtype: str, required: true}
      source: {dtype: str, required: true}
      confidence: {dtype: float, required: true}
      campaign: {dtype: str, required: false}
      technique: {dtype: str, required: false}
    description: "Threat intelligence indicator or campaign"

  # ... Asset, Device, TravelRecord, Decision, DataClass, TimeSlot

relationships:
  - type: HAS_TRAVEL
    source_label: User
    target_label: TravelRecord
    cardinality: one_to_many
    description: "User has an approved travel record"
    
  - type: DECIDED_ON
    source_label: Decision
    target_label: Alert
    cardinality: many_to_one
    description: "Decision was made on this alert"
    
  # ... STORES, ASSOCIATED_WITH, ACTIVE_AT, USES_DEVICE

factors:
  - name: travel_match
    decay_class: campaign
    description: "Does user have matching travel record for alert geography?"
    required_labels: [User, TravelRecord]
    required_relationships: [(User, HAS_TRAVEL, TravelRecord)]
    required_properties:
      User: [id]
      TravelRecord: [destination, date]
    computes_from: graph_traversal

  - name: asset_criticality
    decay_class: permanent
    description: "How critical is the accessed asset?"
    required_labels: [Asset, DataClass]
    required_relationships: [(Asset, STORES, DataClass)]
    required_properties:
      Asset: [id, criticality]
      DataClass: [sensitivity]
    computes_from: graph_traversal

  - name: pattern_history
    decay_class: standard
    description: "Historical base rate for this alert type — THE COMPOUNDING PROOF FACTOR"
    required_labels: [Decision, Alert]
    required_relationships: [(Decision, DECIDED_ON, Alert)]
    required_properties:
      Decision: [action, correct, outcome]
      Alert: [situation_type]
    computes_from: graph_traversal

  # ... threat_intel_enrichment (campaign), time_anomaly (standard), device_trust (standard)

actions:
  - name: false_positive_close
    description: "Close alert as false positive. No analyst action needed."
  - name: escalate_tier2
    description: "Escalate to Tier 2 analyst for investigation."
  - name: enrich_and_wait
    description: "Request additional context, hold for 24h."
  - name: escalate_incident
    description: "Immediate incident response. Highest severity."

decay_classes:
  permanent:
    description: "Organizational structure, asset classifications. Changes rarely."
    default_rate: 0.0001
  standard:
    description: "Baseline behavioral patterns. Default for most factors."
    default_rate: 0.001
  campaign:
    description: "Active threat campaigns, seasonal procurement patterns."
    default_rate: 0.003
  transient:
    description: "Breaking alerts, spot pricing. High volatility."
    default_rate: 0.01
```

### 18.4 Integration Flow

```
At startup (v5.0):
  1. Copilot loads domain_schema.yaml
  2. Parse → DomainSchemaSpec
  3. schema.get_factor_decay_classes() → {"travel_match": "campaign", ...}
  4. CalibrationProfile.get_decay_rate("campaign") → 0.003
  5. LearningState._build_epsilon_vector() → [0.003, 0.0001, 0.003, 0.001, 0.001, 0.001]
  6. ContractChecker validates SchemaContracts against live graph
  7. If mismatches: warnings logged, factors with unmet contracts return default values
```


## 19. Institutional Judgment Metrics (`gae/judgment.py`)

### 19.1 Design Rationale

"The system gets smarter" must be measurable, not just asserted. These metrics quantify the accumulated institutional judgment in terms a CISO (SOC) or CPO (S2P) can evaluate.

GAE computes abstract mathematical metrics. Domain copilots translate them into operational language.

### 19.2 Interface

```python
# gae/judgment.py
"""
Institutional Judgment metrics.

Three metrics quantify how the system's learned judgment differs from
a fresh install. The translation to operational language (analyst hours
saved, MTTR reduction, processing time saved) lives in the domain copilot.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class InstitutionalJudgmentMetrics:
    """Computed by GAE from LearningState and evaluation results."""
    
    # Metric 1: Prior Divergence × Accuracy Improvement
    prior_divergence: float          # ||W_current - W_initial|| (L2 norm)
    baseline_accuracy: float         # Accuracy at W_initial (evaluation scenarios)
    current_accuracy: float          # Accuracy at W_current
    accuracy_improvement: float      # current - baseline
    
    # Metric 2: Learning state
    decisions_processed: int
    categories_with_improvement: int # How many alert categories improved over baseline
    strongest_category: str          # Category with highest accuracy improvement
    weakest_category: str            # Category with lowest or negative improvement
    
    # Metric 3: Recovery behavior
    avg_recovery_decisions: float    # Average decisions to recover accuracy after an error
    
    # Judgment score: combined measure
    judgment_score: float            # prior_divergence × accuracy_improvement × decisions_factor
                                     # Normalized to [0, 100] scale
    
    @staticmethod
    def compute(
        W_initial: np.ndarray,
        learning_state,              # LearningState
        evaluation_results: dict,    # category → accuracy at current W
        baseline_results: dict,      # category → accuracy at W_initial
    ) -> 'InstitutionalJudgmentMetrics':
        """Compute all metrics from learning state and evaluation results."""
        W_current = learning_state.W
        divergence = float(np.linalg.norm(W_current - W_initial))
        
        baseline_acc = np.mean(list(baseline_results.values()))
        current_acc = np.mean(list(evaluation_results.values()))
        improvement = current_acc - baseline_acc
        
        category_deltas = {
            cat: evaluation_results.get(cat, 0) - baseline_results.get(cat, 0)
            for cat in set(evaluation_results) | set(baseline_results)
        }
        improved = sum(1 for d in category_deltas.values() if d > 0)
        strongest = max(category_deltas, key=category_deltas.get) if category_deltas else ""
        weakest = min(category_deltas, key=category_deltas.get) if category_deltas else ""
        
        # Recovery speed from learning history
        recovery_decisions = _compute_recovery_speed(learning_state.history)
        
        # Normalized judgment score [0, 100]
        decisions_factor = min(learning_state.decision_count / 100, 1.0)  # saturates at 100
        raw_score = divergence * max(improvement, 0) * decisions_factor
        judgment_score = min(raw_score * 1000, 100.0)  # scaling TBD from empirical data
        
        return InstitutionalJudgmentMetrics(
            prior_divergence=divergence,
            baseline_accuracy=float(baseline_acc),
            current_accuracy=float(current_acc),
            accuracy_improvement=float(improvement),
            decisions_processed=learning_state.decision_count,
            categories_with_improvement=improved,
            strongest_category=strongest,
            weakest_category=weakest,
            avg_recovery_decisions=recovery_decisions,
            judgment_score=judgment_score,
        )


def _compute_recovery_speed(history) -> float:
    """Average number of decisions to restore accuracy after an error."""
    if len(history) < 5:
        return 0.0
    errors = [i for i, h in enumerate(history) if h.outcome == -1]
    if not errors:
        return 0.0
    recovery_counts = []
    for err_idx in errors:
        # Count decisions until next 3 consecutive correct
        consecutive = 0
        for j in range(err_idx + 1, len(history)):
            if history[j].outcome == +1:
                consecutive += 1
                if consecutive >= 3:
                    recovery_counts.append(j - err_idx)
                    break
            else:
                consecutive = 0
    return float(np.mean(recovery_counts)) if recovery_counts else float(len(history))
```

### 19.3 Domain Translation (Copilot-Level — Reference)

```python
# soc-copilot: services/judgment_translator.py (REFERENCE — NOT part of GAE)
class SOCJudgmentTranslator:
    """Translate GAE metrics to CISO-meaningful language."""
    
    def translate(self, metrics: InstitutionalJudgmentMetrics) -> dict:
        return {
            "headline": (
                f"After {metrics.decisions_processed} triage decisions, "
                f"accuracy improved {metrics.accuracy_improvement:.0%} from baseline"
            ),
            "strongest_area": f"Best learning: {metrics.strongest_category} alerts",
            "operational_impact": {
                "analyst_decisions_automated": int(metrics.decisions_processed * metrics.current_accuracy),
                "estimated_hours_saved": self._hours_saved(metrics),
                "recovery_after_error": f"{metrics.avg_recovery_decisions:.0f} decisions",
            },
            "judgment_score": metrics.judgment_score,
            "judgment_score_explanation": (
                f"Score reflects {metrics.prior_divergence:.2f} divergence from generic priors "
                f"with {metrics.accuracy_improvement:.0%} accuracy gain across "
                f"{metrics.categories_with_improvement} alert categories"
            ),
        }
```

## 20. Ablation Framework (`gae/ablation.py`)

### 20.1 Design Rationale

Proving decisions are better than competing approaches requires ablation — running the same evaluation scenarios under degraded configurations and showing each architectural component's contribution. This is the standard methodology in ML research.

### 20.2 Interface

```python
# gae/ablation.py
"""
Ablation framework: prove each architectural component's value.

Four baselines, all runnable within the product:
  Static:        Fixed W at expert priors. Never learns.
  Flat learning: Symmetric reinforcement (no asymmetric penalty).
  No graph:      Random/uniform factor vector. W still learns.
  Full system:   Everything — the product.

Usage (v5.0):
  configs = [static_config(), flat_learning_config(), no_graph_config(), full_config()]
  report = run_ablation(scenarios, configs, ...)
"""

from dataclasses import dataclass, field
from typing import Optional
from .calibration import CalibrationProfile

@dataclass
class AblationConfig:
    """Configuration for one ablation run."""
    name: str
    description: str
    learning_enabled: bool = True
    asymmetric_penalty: bool = True
    graph_context_enabled: bool = True
    decay_enabled: bool = True
    calibration_profile: Optional[CalibrationProfile] = None
    
    def to_profile(self, base_profile: CalibrationProfile) -> CalibrationProfile:
        """Derive a CalibrationProfile for this ablation config."""
        if self.calibration_profile:
            return self.calibration_profile
        profile = CalibrationProfile(
            learning_rate=base_profile.learning_rate if self.learning_enabled else 0.0,
            penalty_ratio=base_profile.penalty_ratio if self.asymmetric_penalty else 1.0,
            temperature=base_profile.temperature,
            discount_strength=base_profile.discount_strength,
        )
        return profile


# Convenience constructors
def static_config() -> AblationConfig:
    return AblationConfig(
        name="static",
        description="Fixed W at expert priors. Never learns. Day 1000 = Day 1.",
        learning_enabled=False,
    )

def flat_learning_config() -> AblationConfig:
    return AblationConfig(
        name="flat_learning",
        description="Symmetric reinforcement. No 20:1 penalty. No risk awareness.",
        asymmetric_penalty=False,
    )

def no_graph_config() -> AblationConfig:
    return AblationConfig(
        name="no_graph",
        description="Random factor vectors. W learns from noise. No context.",
        graph_context_enabled=False,
    )

def full_config() -> AblationConfig:
    return AblationConfig(
        name="full_system",
        description="Full product: graph context + asymmetric learning + decay.",
    )


@dataclass
class AblationReport:
    """Comparative results across ablation configurations."""
    configs: list[AblationConfig]
    per_config_results: dict[str, 'EvaluationReport']  # config.name → report
    
    # Computed comparisons
    accuracy_deltas: dict[str, float]       # vs full_system
    recovery_deltas: dict[str, float]       # vs full_system
    confirmation_bias_results: dict[str, bool]  # Did each config catch the TP after 20 FPs?
    
    def summary(self) -> str:
        """Human-readable comparison."""
        lines = ["Ablation Results:"]
        full = self.per_config_results.get("full_system")
        for config in self.configs:
            result = self.per_config_results.get(config.name)
            if result:
                delta = self.accuracy_deltas.get(config.name, 0)
                lines.append(
                    f"  {config.name}: {result.action_accuracy:.1%} accuracy "
                    f"({delta:+.1%} vs full)"
                )
        return "\n".join(lines)
```


## 21. What's Built vs What's Next

### 21.1 GAE v0.1.0 (Tagged with v4.1)

| Module | Status | Tests | Notes |
|---|---|---|---|
| `factors.py` | ✅ Complete | 12 tests | FactorComputer Protocol + assemble_factor_vector |
| `scoring.py` | ✅ Complete | 15 tests | Eq. 4 with shape checks |
| `learning.py` | ✅ Complete | 18 tests | Eq. 4b, 4c. Hardcoded constants (to be refactored) |
| `convergence.py` | ✅ Complete | 8 tests | Three failure modes detected |
| `contracts.py` | ✅ Complete | 6 tests | SchemaContract + EmbeddingContract |
| `events.py` | ✅ Complete | 4 tests | Pure dataclasses |
| `store.py` | ✅ Complete | 5 tests | JSON persistence |
| `primitives.py` | ✅ Complete | 2 tests | NumPy backend only |
| `__init__.py` | ✅ Complete | — | Public API exports |

**Total:** 70 core tests + 107 generic domain tests = 177 passing.
**Measured results:** 28.6x asymmetric trust (exceeds 20:1 target), cold-start 2.5s, convergence threshold min-20.

### 21.2 GAE v4.5 Preamble (NEW — 2-3 prompts in GAE repo)

| Prompt | Creates | Tests | Gate |
|---|---|---|---|
| **GAE-CAL-1** | `gae/calibration.py` (CalibrationProfile dataclass + convenience constructors + validation). Refactor `learning.py` to accept CalibrationProfile. Refactor `scoring.py` to accept temperature from profile. | Profile creates with defaults. SOC profile matches current behavior. LearningState.update() uses profile.learning_rate, profile.penalty_ratio. score_alert uses profile.temperature. All 177 existing tests still pass. | All existing tests pass with default CalibrationProfile (backward compatible) |
| **GAE-CAL-2** | Update `learning.py` epsilon_vector construction from decay_class mapping. Update `contracts.py` SchemaContract to formalize decay_class field. | Per-factor decay rates match CalibrationProfile. Different decay classes produce different ε values. Test: two factors with different decay classes → epsilon_vector has different values. | Run 10-decision sequence, verify campaign-class factors decay faster than permanent-class |
| **GAE-CAL-3** (if needed) | `gae/schema.py` (DomainSchemaSpec, FactorSpec, load_domain_schema). May be deferred to v5.0 if v4.5 can use SchemaContract.decay_class directly. | Schema file parses. get_factor_decay_classes() returns correct mapping. validate_against_contracts() catches mismatches. | Parse SOC schema → correct decay classes for all 6 factors |

### 21.3 GAE v5.0 Prompts (Platform Breadth)

| Prompt | Creates | Tests | Dependency |
|---|---|---|---|
| **GAE-EVAL-1** | `gae/evaluation.py` (EvaluationScenario, EvaluationResult, EvaluationReport, run_evaluation). | 3 SOC scenarios execute correctly. Report by_category and by_confidence_tier populated. | CalibrationProfile (GAE-CAL-1) |
| **GAE-JUDG-1** | `gae/judgment.py` (InstitutionalJudgmentMetrics.compute). | Metrics compute from mock LearningState + evaluation results. judgment_score increases with more decisions. | GAE-EVAL-1 |
| **GAE-ABL-1** | `gae/ablation.py` (AblationConfig, AblationReport, run_ablation). | 4 configs × 3 scenarios. Full system > flat learning > static. | GAE-EVAL-1, GAE-JUDG-1 |
| **GAE-ENG-1** | API surface design: public exports in `__init__.py`, import lint rule, `EXPORTS.md`. | Lint rule catches any `from soc` or `from platform` import in gae/. | — |
| **GAE-ENG-2** | `examples/minimal_domain/` — Hello World DomainConfig (~50 lines). | Example runs full loop: factor → score → learn → converge. No Neo4j. | GAE-CAL-1 |
| **GAE-ENG-3** | Engine tests independent of SOC. Reorganize test suite so `pytest tests/` passes without SOC installed. | All 70 core tests pass in GAE repo alone. | GAE-ENG-1 |
| **GAE-DOC-1** | `docs/users_guide.md` — Calibration guidance, domain extension patterns, convergence monitoring, equation-to-code map. | — | All above |

### 21.4 GAE v5.5+ (Discovery + Evolution)

| Module | Status | Dependency |
|---|---|---|
| `embeddings.py` | Designed (§9). EmbeddingProvider protocol (v6 update). | Phase C results |
| `attention.py` | Designed (§10). | embeddings.py |
| `discovery.py` | Designed (§10.4). | attention.py |


## 22. Repository Structure (Consolidated)



```
graph-attention-engine/
├── gae/
│   ├── __init__.py              # Public API exports (updated: CalibrationProfile, evaluation)
│   ├── calibration.py           # NEW: CalibrationProfile + convenience constructors
│   ├── scoring.py               # UPDATED: accepts CalibrationProfile.temperature
│   ├── learning.py              # UPDATED: accepts CalibrationProfile, builds epsilon_vector from decay classes
│   ├── factors.py               # Unchanged
│   ├── convergence.py           # Unchanged
│   ├── contracts.py             # UPDATED: SchemaContract.decay_class formalized
│   ├── primitives.py            # Unchanged
│   ├── schema.py                # NEW: DomainSchemaSpec, FactorSpec, load_domain_schema
│   ├── evaluation.py            # NEW: EvaluationScenario, EvaluationReport, run_evaluation
│   ├── judgment.py              # NEW: InstitutionalJudgmentMetrics
│   ├── ablation.py              # NEW: AblationConfig, AblationReport, run_ablation
│   ├── embeddings.py            # Tier 4 (v5.5) — UPDATED: EmbeddingProvider protocol
│   ├── attention.py             # Tier 5 (v5.5)
│   ├── discovery.py             # Tier 5 (v5.5)
│   ├── events.py                # Unchanged
│   ├── store.py                 # Unchanged
│   └── types.py                 # Unchanged
├── tests/
│   ├── test_calibration.py      # NEW
│   ├── test_evaluation.py       # NEW (v5.0)
│   ├── test_judgment.py         # NEW (v5.0)
│   ├── test_ablation.py         # NEW (v5.0)
│   ├── test_schema.py           # NEW
│   ├── test_scoring.py          # Unchanged
│   ├── test_learning.py         # UPDATED: tests with CalibrationProfile
│   ├── test_contracts.py        # Unchanged
│   └── ...
├── examples/
│   └── minimal_domain/          # NEW (v5.0: GAE-ENG-2)
│       ├── config.py            # ~50-line DomainConfig
│       ├── domain_schema.yaml   # Example schema
│       └── run.py               # Full loop demo
├── docs/
│   ├── equations.md             # Unchanged
│   ├── users_guide.md           # NEW (v5.0: GAE-DOC-1)
│   └── EXPORTS.md               # NEW (v5.0: GAE-ENG-1)
├── pyproject.toml               # Dependencies: numpy>=1.24.0 + optional [embeddings]
├── LICENSE                      # Apache 2.0
└── README.md                    # UPDATED: CalibrationProfile usage example
```

**Optional dependency (v5.5):**
```toml
[project.optional-dependencies]
embeddings = ["sentence-transformers>=2.2.0", "torch>=2.0.0"]
```


## 23. Claude Code Prompt Sequence (GAE Repo Only)

> SOC copilot prompts → soc_copilot_design_v1. Platform prompts → ci_platform_design_v1.
> For the complete prompt sequence across all repos, see session_continuation_v17.

### 23.1 v4.1 GAE Foundation (7 prompts — GAE REPO)

| Prompt | Scope | Creates/Modifies | Test | Notes |
|---|---|---|---|---|
| GAE-0 | Repo scaffold: pyproject.toml, README, LICENSE, placeholders | All files | `from gae import __version__` works | numpy-only pyproject.toml |
| GAE-1a | Scoring matrix (Eq. 4) | `gae/__init__.py`, `gae/scoring.py` | Shape assertions, softmax sums to 1.0 | Pure math |
| GAE-1b | Weight learning (Eq. 4b, 4c) + W expansion (R5) | `gae/learning.py` | 20:1 ratio, clamp [-5,5] | Hardening data structures with no-op defaults (A1: discount=0.0, A2: uniform ε, A4: DimensionMetadata exists, C3: PendingValidation exists) |
| GAE-1c | State persistence (JSON, no Neo4j) | `gae/store.py` | Save/load roundtrip | JSON serialization of LearningState |
| GAE-1d | Attention primitive (NumPy + torch stub) | `gae/primitives.py` | Row-sum = 1.0 | v5.5 foundation |
| GAE-1e | Event type definitions (dataclasses, no bus) | `gae/events.py` | Import succeeds, types instantiate | NO async, NO bus — types only |
| GAE-2a-protocol | FactorComputer Protocol + `assemble_factor_vector()` | `gae/factors.py` | Protocol is domain-agnostic, assemble works with synthetic values | NO Neo4j dependency |

All prompts: Do NOT use debugger. Do NOT use git. No SOC knowledge. numpy-only.

### 23.2 v4.5 GAE Preamble (2-3 prompts — GAE REPO)

| Prompt | Creates | Tests | Gate |
|---|---|---|---|
| **GAE-CAL-1** | `gae/calibration.py` (CalibrationProfile dataclass + convenience constructors + validation). Refactor `learning.py` to accept CalibrationProfile. Refactor `scoring.py` to accept temperature from profile. | Profile creates with defaults. SOC profile matches current behavior. LearningState.update() uses profile.learning_rate, profile.penalty_ratio. score_alert uses profile.temperature. All 177 existing tests still pass. | All existing tests pass with default CalibrationProfile (backward compatible) |
| **GAE-CAL-2** | Update `learning.py` epsilon_vector construction from decay_class mapping. Update `contracts.py` SchemaContract to formalize decay_class field. | Per-factor decay rates match CalibrationProfile. Different decay classes produce different ε values. Test: two factors with different decay classes → epsilon_vector has different values. | Run 10-decision sequence, verify campaign-class factors decay faster than permanent-class |
| **GAE-CAL-3** (if needed) | `gae/schema.py` (DomainSchemaSpec, FactorSpec, load_domain_schema). May be deferred to v5.0 if v4.5 can use SchemaContract.decay_class directly. | Schema file parses. get_factor_decay_classes() returns correct mapping. validate_against_contracts() catches mismatches. | Parse SOC schema → correct decay classes for all 6 factors |

### 23.3 v5.0 GAE Additions (7 prompts — GAE REPO)

| Prompt | Creates | Tests | Dependency |
|---|---|---|---|
| **GAE-EVAL-1** | `gae/evaluation.py` (EvaluationScenario, EvaluationResult, EvaluationReport, run_evaluation). | 3 SOC scenarios execute correctly. Report by_category and by_confidence_tier populated. | CalibrationProfile (GAE-CAL-1) |
| **GAE-JUDG-1** | `gae/judgment.py` (InstitutionalJudgmentMetrics.compute). | Metrics compute from mock LearningState + evaluation results. judgment_score increases with more decisions. | GAE-EVAL-1 |
| **GAE-ABL-1** | `gae/ablation.py` (AblationConfig, AblationReport, run_ablation). | 4 configs × 3 scenarios. Full system > flat learning > static. | GAE-EVAL-1, GAE-JUDG-1 |
| **GAE-ENG-1** | API surface design: public exports in `__init__.py`, import lint rule, `EXPORTS.md`. | Lint rule catches any `from soc` or `from platform` import in gae/. | — |
| **GAE-ENG-2** | `examples/minimal_domain/` — Hello World DomainConfig (~50 lines). | Example runs full loop: factor → score → learn → converge. No Neo4j. | GAE-CAL-1 |
| **GAE-ENG-3** | Engine tests independent of SOC. Reorganize test suite so `pytest tests/` passes without SOC installed. | All 70 core tests pass in GAE repo alone. | GAE-ENG-1 |
| **GAE-DOC-1** | `docs/users_guide.md` — Calibration guidance, domain extension patterns, convergence monitoring, equation-to-code map. | — | All above |

### 23.4 v5.5 GAE Tiers 4-5 (7 prompts — GAE REPO)

| Prompt | Scope | Creates/Modifies | Test |
|---|---|---|---|
| EMB-1a | PropertyEmbeddingProvider + feature schema | `gae/embeddings.py` | Shapes match, L2 unit norm |
| EMB-1d | Embedding contracts + per-factor decay activation (A2) | `gae/contracts.py` extend | Coverage checks, decay_class per factor |
| ATT-1a | `cross_attention()` + `multi_domain_attention()` (Eq. 6, 9) | `gae/attention.py` | 15 heads for 6 domains, shapes match |
| ATT-1b | `extract_discoveries()` (Eq. 8a-8c) + soft expansion (A4) | `gae/discovery.py` | Threshold + top-K + margin filter chain |
| ATT-2a | Discovery-driven W expansion (R5 + A4 provisional dimensions) | `gae/learning.py` | W expands, provisional dims decay if unreinforced |
| ATT-2c | Recursive discovery + multi-head consolidation | `gae/attention.py` | Second sweep after discovery finds more |
| A1-ACTIVATE | Confidence-discounted α activation (A1 hardening) | `gae/learning.py` | discount_strength > 0 changes learning rate |

**Post v5.5:** README, API reference, quickstart. Open-source release.

### 23.5 Prompt Summary (GAE Repo Only)

| Version | GAE Prompts | Scope |
|---|---|---|
| v4.1 | 7 | Scoring, learning, persistence, primitives, events, factors protocol |
| v4.5 | 2-3 | CalibrationProfile, decay mapping, domain schema protocol |
| v5.0 | 7 | Evaluation, judgment, ablation, API surface, example domain, docs |
| v5.5 | 7 | Embeddings, attention, discovery, A1 activation |
| **Total** | **23-24** | Complete Tiers 1-5 + evaluation + open-source ready |

---

## 24. Equation-to-Code Traceability Matrix

| Equation | Blog Section | File | Function | Version | Scope |
|---|---|---|---|---|---|
| Eq. 4 | §3 Level 1 | `gae/scoring.py` | `score_alert()` | v5.0 | ENGINE |
| Eq. 4b | §3 Weight Update | `gae/learning.py` | `LearningState.update()` | v5.0 | ENGINE |
| Eq. 4c | §3 Decay | `gae/learning.py` | `LearningState.apply_decay()` | v5.0 | ENGINE |
| Eq. 5 | §4 Embeddings | `gae/embeddings.py` | `compute_domain_embeddings()` | v5.5 | ENGINE |
| Eq. 6 | §4 Cross-Attention | `gae/attention.py` | `cross_attention()` | v5.5 | ENGINE |
| Eq. 7a | §4 Logit Matrix | `gae/primitives.py` | `scaled_dot_product_attention()` | v5.5 | ENGINE |
| Eq. 7b | §4 Attention Matrix | `gae/primitives.py` | `scaled_dot_product_attention()` | v5.5 | ENGINE |
| Eq. 8a | §4 Discovery | `gae/discovery.py` | `extract_discoveries()` | v5.5 | ENGINE |
| Eq. 8b | §4 Discovery | `gae/discovery.py` | `extract_discoveries()` | v5.5 | ENGINE |
| Eq. 8c | §4 Discovery | `gae/discovery.py` | `extract_discoveries()` | v5.5 | ENGINE |
| Eq. 9 | §5 Multi-Domain | `gae/attention.py` | `multi_domain_attention()` | v5.5 | ENGINE |

**Every equation → one function → one file → one prompt. All ENGINE scope.**

A technical reviewer can grep for any equation number and find the implementation. At v5.5, all equations are verifiable in the open-source repository — no demo access required.

---

## 25. How GAE Transforms the Demo

> Full demo flow specification → soc_copilot_design_v1.md

### The Computation Transformation

```
BEFORE GAE:  hardcoded factors → if-else scoring → trust counter ±delta
AFTER GAE:   graph-traversal factors → Eq.4 scoring matrix → Eq.4b Hebbian learning
             with 20:1 asymmetry, per-factor decay, convergence monitoring
```

### Realization Sequence (All Repos)

```
v4.1  [build]    GAE: real scoring + learning + persistence + events
                  SOC: factor impls + router wiring + decision/outcome write-back
                  ──── COMPUTATION BOUNDARY ────
v5.0  [build]    GAE: contracts + eval framework + example domain
                  SOC: classification, narratives, seed, ROI dashboard
                  Platform: contract validation (startup hook)
                  ──── API BOUNDARY ────
v5.5  [build]    GAE: embeddings + attention + discovery (Tiers 4-5)
                  SOC: embedding endpoints + discovery dashboard
                  ──── OPEN-SOURCE BOUNDARY ────
                  GAE + Platform released: Apache 2.0
v6.0  [build]    Platform: agents + artifact evolution + ontology governance
                  SOC: autonomous loops, real connectors
                  ──── PRODUCTION BOUNDARY ────
v7.0  [plan]     Multi-tenant, community domains, cross-tenant meta-graph
```

---

---

## 26. Design Decisions Summary

| Decision | Rationale | Version |
|---|---|---|
| NumPy-only, zero external deps | 24 multiply-adds don't need PyTorch. Backend-swappable primitives for future | v4.1 |
| Three-repo architecture (GAE / platform / copilots) | GAE = transformers, platform = hub+accelerate, copilots = model repos. Structural enforcement of P12 | v4.1 |
| Demand-side contracts in GAE, supply-side in platform | GAE declares "I need X" (pure Python). Platform validates "X exists" (needs Neo4j) | v4.1 design, v5.0 impl |
| Event types in GAE, event bus in platform | Types are data (numpy-compatible). Bus is infrastructure (async) | v4.1 |
| FactorComputer Protocol in GAE, orchestrator in copilot | Protocol is abstract (pure Python). Orchestrator needs Neo4j | v4.1 |
| `assemble_factor_vector()` synchronous in GAE | Copilot does async work, passes values to GAE for numpy assembly | v4.1 |
| Hardening data structures with no-op defaults | A1/A2/A4/C3 structures exist from v4.1 (R9 stability). Behavior activates at v5.5-v6.0 | v4.1 |
| Factor queries traverse relationships (P10) | Property reads don't benefit from graph accumulation | v4.1 |
| f(t) stored in Decision nodes, not memory (R4) | Graph changes between decision and outcome. Recomputed f corrupts learning | v4.1 |
| Confidence-discounted learning rate (A1) | Discount α by prior confidence to reduce confirmation bias | v5.5 activate |
| Per-factor temporal decay (A2) | Permanent patterns shouldn't decay at campaign-specific rates | v5.5 activate |
| Soft discovery expansion with accelerated decay (A4) | False discoveries can expand W permanently. Provisional dims must earn their keep | v5.5 |
| Delayed outcome validation for autonomous decisions (C3) | Auto-close r(t)=+1 accumulates during dwell window. Defer learning until validated | v6.0 |
| Open-source GAE at v5.5, not v5.0 | At v5.0 GAE is just scoring + learning. At v5.5, discovery makes it distinctive | v5.5 |
| Artifact evolution in platform, not GAE | Pipeline infrastructure, not core math. GAE provides scoring primitives | v6.0 |

| CalibrationProfile replaces hardcoded constants | Multi-domain: SOC 20:1 ≠ S2P 5:1. Domain-configurable without touching GAE | v4.5 |
| Three-layer decay (schema → profile → loop) | Single source of truth for factor metadata. New decay class = config change, not code | v4.5 |
| EmbeddingProvider protocol (replaces PropertyEmbedder) | Design for Option B (property), test Option C (transformer). Community can contribute | v5.5 |
| Evaluation framework in GAE (not copilot) | Shared format across EVAL-1, SEED-2, B3 consumers. Domain-agnostic scenarios | v5.0 |
| Institutional judgment metrics in GAE | Abstract math metrics. Domain copilots translate to CISO/CPO language | v5.0 |
| Ablation framework in GAE | Standard ML methodology. Four baselines prove each component's value | v5.0 |

---

*Graph Attention Engine — Design & Architecture v7 | March 1, 2026*
*Consolidated from v5 (Feb 28) + v6 addendum (Mar 1)*
*Standalone library: `pip install graph-attention-engine` (Apache 2.0, numpy-only)*
*Three-repo architecture: GAE (math) → ci-platform (infrastructure) → domain copilots (expertise)*
*Requirements R1-R9 | 12 design principles (P1-P12) | Connectors C1-C5 in GAE, C6 in platform*
*CalibrationProfile: domain-configurable hyperparameters (α, λ_neg, τ, ε)*
*Hardening data structures: A1, A2, A4, C3 — present from v4.1, activated at v5.5-v6.0*
*GAE prompts: v4.1 (7) + v4.5 (2-3) + v5.0 (7) + v5.5 (7) = 23-24 total*
*Every equation → one function → one file → one prompt*
*Companion repos: ci-platform (v4.5+), soc-copilot (v4.1+)*
*Companion docs: soc_copilot_design_v1, ci_platform_design_v1, design_decisions_v1, session_continuation_v17*
*"The moat is the graph, not the model. The engine proves it."*
