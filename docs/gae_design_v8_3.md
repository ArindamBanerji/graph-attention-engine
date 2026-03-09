# Graph Attention Engine — Design & Architecture v8.3

**Date:** March 7, 2026
**Version:** 8.3 (v8.2 + Level 2/3 design specification + data preservation hooks)
**Status:** v4.5 tagged. **Scoring architecture SETTLED** (Eq. 4-final). GATE-OP passed. Planning v5.0 GAE: ProfileScorer integration.
**Repository:** graph-attention-engine (standalone, numpy-only)
**Scope:** v4.1 (Tiers 1-3 foundation), v4.5 (CalibrationProfile), **v5.0 (ProfileScorer + pluggable kernels + evaluation)**, v5.5 (Tiers 4-5 + meta-graph diagnostics + open-source release), **v7.0 (Level 2: GraphAttentionBridge — designed, not yet implemented)**, **v8.0 (Level 3: Cross-Domain Discovery — designed, not yet implemented)**
**Companion repos:** ci-platform (UCL, agents, event bus, governance), soc-copilot (SOC domain — proprietary), **cross-graph-experiments (25 experiments: 9 bridge + 5 validation + 11 OP/synthesis, 100+ charts)**
**Supersedes:** gae_design_v7.md, gae_design_v8.md, gae_design_v8_1.md, gae_design_v8_2.md, bridge_layer_design_v1.md, bridge_experiments_design_v1.md. Incorporates findings from bridge_experiments_design_v3.md and bridge_experiments_catalog_v7.md.

> **Changes from v7 → v8:**
> This is a major revision. The bridge experiments (9 experiments, March 2-4 2026) settled the scoring architecture. Five subsequent validation experiments (V1A–V3B) confirmed and refined the findings. Key changes:
>
> (1) **Tier 2 completely rewritten.** ScoringMatrix (Eq. 4, dot product) → ProfileScorer (Eq. 4-final, L2 distance). Profile centroids μ[c,a,:] replace weight matrix W. Category routing replaces shared scoring.
> (2) **Tier 3 substantially rewritten.** Hebbian W update (Eq. 4b) → centroid pull/push (Eq. 4b-final). Count-based decay replaces Ψ warmup. CalibrationProfile updated for profile parameters.
> (3) **New §4: Experimental Foundation.** Inline evidence for every architecture decision. 9 bridge experiments + 5 validation experiments summarized.
> (4) **New §5: The Ontological Architecture.** Profiles as compiled ontology. Three-layer design: ontology → meta-graph → profiles → scoring.
> (5) **Eq. 4 progression documented.** From published dot product (49.26%) through 7 intermediate formulations to settled L2 profiles (98.2%).
> (6) **Pluggable kernel API.** L2 default, Mahalanobis for mixed-scale, cosine alternative, dot product warned. From EXP-E1 evidence.
> (7) **Scaling validated.** 20×10×20 (200 centroids, 4000 params) at 99.9% oracle accuracy. From EXP-E2 evidence.
> (8) **G, Ψ, W eliminated.** GatingMatrix experimentally falsified. Warmup tensor Ψ eliminated as control (repositioned as diagnostic). Weight matrix W replaced by profile centroids μ.
> (9) **Causal loop diagrams updated.** Connector 2 (scoring) and Connector 3 (learning) revised for profile-based architecture.
> (10) **Design decisions expanded.** 10+ new decisions with experimental evidence.
> (11) **Requirement R5 updated.** Dynamic dimensionality via profile augmentation, not W expansion.
>
> Unchanged from v7: §1.1-1.2 core problem statement, §2 technology analysis foundations, §3 P1-P12, §4.x package structure, §6 Tier 1 factors, §9 Tier 4 embeddings, §11 Tier 6 artifacts, §12-§15 contracts/API/accumulation/open-source, §17-§20 evaluation/schema/judgment/ablation frameworks.

> **Changes from v8 → v8.1:**
> Five validation experiments (V1A–V3B) addressed independent reviewer concerns on the math paper. All five confirmed the concerns; all five had concrete fixes. Changes integrated into this document:
>
> (1) **τ default changed from 0.25 to 0.1.** V3B showed ECE=0.19 (poorly calibrated) at τ=0.25 vs ECE=0.036 (well calibrated) at τ=0.1. ProfileScorer code and CalibrationProfile default updated.
> (2) **Push update clipping documented with V2 data.** Clipping was already in code; now annotated with quantitative justification (escape at dec 6–12 under adversarial, 4,608× norms by dec 200).
> (3) **Tier 5: LayerNorm requirement added.** V1B showed enrichment norm explosion (2.9M× after 5 sweeps). Production form: E_i^{enriched} = Normalize(E_i + Σ CrossAttention(G_i, G_j)).
> (4) **New §4.6: Validation Experiments summary.** V1A (scaling b=2.11), V1B (norm explosion), V2 (push stability), V3A (baselines), V3B (calibration).
> (5) **Category routing clarified as MoE** (externally defined, mutually exclusive), not multi-head (learned, overlapping).
> (6) **Convergence caveat added.** Pull/push rule is heuristic (related to LVQ), not derived from explicit loss. Formal convergence analysis under non-stationary data is future work.

> **Changes from v8.2 → v8.3 (March 7, 2026 — Level 2/3 design specification):**
>
> (1) **§11 substantially expanded.** "Tiers 4-6 (Unchanged from v7)" replaced with full Level 2/3 design specification. Four subsections: §11.1 Tier 4 (Embeddings), §11.2 Level 2 (GraphAttentionBridge — v7 target), §11.3 Level 3 (Cross-Domain Discovery — v8 target), §11.4 Level 1 Data Preservation Hooks (what every domain must write for Level 2/3 to work).
> (2) **γ precisely defined.** γ is the cross-domain compounding exponent in `𝒮(n,t) ~ O(n^{2.11} · t^γ)`. Definition, measurement method, and EXP-G1 prerequisite specified in §11.3. γ≈1.5 is an estimate until EXP-G1 runs.
> (3) **GraphAttentionBridge interface specified.** Inputs, outputs, prerequisite interfaces (EmbeddingProvider, GraphSchema), LayerNorm requirement (V1B), and v7 implementation target in §11.2.
> (4) **Level 1 hooks made explicit.** §11.4 specifies the three write obligations every Level 1 domain must fulfill: DecisionRecord, OutcomeRecord, ProfileSnapshot. These are prerequisites for Level 2/3 to function. Each field is named and typed.
> (5) **Scope field updated.** v7.0 (Level 2) and v8.0 (Level 3) targets added to scope. Both marked "designed, not yet implemented."
> (6) **§17.4 updated.** v5.5+ scope now explicitly distinguishes v5.5 (Tiers 4-5 embedding + attention infrastructure) from v7 (GraphAttentionBridge) and v8 (cross-domain discovery).
> (7) **§18 repo structure updated.** `gae/bridge.py` (Level 2, v7) and `gae/discovery_engine.py` (Level 3, v8) added. `gae/graph_schema.py` (Level 1 hook contract) added.
> (8) **§19 equation traceability updated.** Level 2/3 equations (Eq. 6, 8a-8c, 9, γ-scaling) status updated from "Designed" to "Designed — v7/v8 target" with prerequisite links.
> (9) **§20.3 new design decisions.** Four decisions from Level 2/3 design added: γ requires EXP-G1 before any claim; Level 1 hooks are backward-compatible additions (not breaking changes); LayerNorm is a hard production requirement for Level 2 (V1B); cross-domain compounding is architectural design until EXP-G1 validates γ.
> (10) **Experiment count updated.** 24 → 25 (SYNTH-EXP-0 infrastructure build counted).

---
>
> (1) **Experiment count updated.** 9 bridge + 5 validation → 24 total (adds 10 OP/synthesis). GATE-OP passed (p=0.0008 at λ=0.5).
> (2) **§4.4 Validated table expanded.** OP series key findings added: operative window, τ_modifier rejection, checkpoint requirement, 100%-correctness threshold.
> (3) **§4.6 V3A condition tag added.** 94.78% tagged as synthetic centroidal data. Scaling claim qualified: b=2.11 validated by simulation; full t^γ compounding expression is a projection (EXP-G1 pending).
> (4) **§7.1 Loop 4 PROPOSAL added.** Synthesis/intelligence layer as fourth causal loop, gated by GATE-M/GATE-D/GATE-V. Explicitly separated from Loops 1-3.
> (5) **§9.3 update() architectural firewall documented.** update() has NO synthesis parameter — this is the Loop 2/Loop 4 firewall. μ NEVER updated using σ. σ flows through score() path only via the synthesis term in Eq. 4-synthesis (when active). This constraint is permanent.
> (6) **§10.3 Production requirements expanded.** TD-027 (centroid RESET before L2 deployment — no migration), TD-033 (checkpoint/rollback infrastructure), TD-034 (τ recalibration as hard gate before Loop 2 activates) added.
> (7) **§17.3 GAE-PROF-3 note added.** Centroid reset (TD-027) must be executed before LiveState → ProfileScorer migration.
> (8) **§19 Eq. 4-synthesis added** as PROPOSAL row (gated by GATE-M/GATE-V).
> (9) **§20.2 Four OP series design decisions added.** τ_modifier rejection, operative window, update() firewall, checkpoint infrastructure.

---

## 1. Requirements: Why This Document Exists

### 1.1 The Problem

The SOC Copilot's published math blog describes a precise computational system: cross-graph attention (Eq. 6), weight learning (Eq. 4b), entity embeddings (Eq. 5), discovery extraction (Eq. 8a-8c), and scoring matrices (Eq. 4). The demo codebase implements a subset. The GAE is the computational substrate that implements the math.

### 1.2 Requirements

**R1 — Equation traceability.** Every equation maps to exactly one function in exactly one file.

**R2 — Causal closure specification.** Outputs of one cycle become inputs to the next. Six causal connectors (§6) wire computation into closed loops.

**R3 — Event-driven architecture.** Three causal loops at different timescales trigger each other via events.

**R4 — Factor vector preservation.** Decision nodes store f(t) from the ORIGINAL decision. Recomputed f at outcome time would corrupt the learning signal.

**R5 — Dynamic dimensionality.** *(Updated in v8.)* When factor space changes, profile centroids μ must accommodate new dimensions. This happens through profile augmentation (adding factor dimensions to μ), not W expansion. The centroid update rule naturally handles new dimensions — new factor values pull centroids toward observed data. **Note:** EXP-D2 found zero significant factor interactions in SOC data, so augmentation is unlikely in practice. R5 is preserved for generality.

**R6 — Technology selection justification.** NumPy for v5.0 (24 multiply-adds per scoring). Backend-swappable for future scale.

**R7 — Claude Code prompt decomposition.** Build decomposed into single-file, tightly-scoped prompts.

**R8 — Accumulation channel completeness.** Every graph mutation produces a write-back read by downstream computation.

**R9 — Open-source API stability.** Public classes versioned and backward-compatible. Engine never imports from product code (P12).

---

## 2. Technology Analysis

### 2.1 What Transformer Attention Is

*(Unchanged from v7 — see §2.1-2.3 in gae_design_v7.md for full analysis.)*

Core insight preserved: HuggingFace BertSelfAttention is 5 lines of math + 245 lines of infrastructure. We take the math, build our own infrastructure.

### 2.2 What's Changed: The Kernel

The v7 analysis mapped our Eq. 4 to transformer dot-product attention. **The experiments proved this mapping is wrong for our data.**

| Component | v7 Mapping | v8 Mapping (Experimentally Validated) |
|---|---|---|
| Query | Factor vector f | Factor vector f (unchanged) |
| Key | Weight matrix W (shared) | Profile centroids μ[c,:,:] (per-category) |
| Kernel | Dot product (f · W^T) | **L2 distance (-\|\|f − μ\|\|²)** |
| Multi-head | None (shared W) | Category routing (n_c heads) |

**Why L2 beats dot product:** Dot product is magnitude-confounded. Factors with high mean values (device_trust ≈ 0.85) dominate regardless of discriminative value. L2 measures profile **shape** — "how different is this alert from each action's typical pattern?" — which is the right question.

**Evidence:** EXP-C1. Same data, same profiles, zero learning. Dot product: 61.0%. L2: 97.89%. The 36.89pp gap is the most important single finding.

### 2.3 Scale Analysis (Updated)

| Version | Operation | Size | Backend | Latency |
|---|---|---|---|---|
| v5.0 | L2 distance: \|\|f − μ[c,a,:]\|\|² | (1×6) vs (4×6) = 24 subtracts + squares | NumPy | <0.01ms |
| v5.0 (xlarge) | L2 distance at 20×10×20 | (1×20) vs (10×20) = 200 ops | NumPy | <0.1ms |
| v5.5 | Cross-graph attention Eq. 6 | (500×128) × (300×128) | NumPy | <10ms |
| v6.0+ | Scaled attention + meta-graph | Larger matrices | PyTorch optional | TBD |

**Decision unchanged:** v5.0 and v5.5 use NumPy only.

### 2.4 Why This Is Attention (Revised)

The system IS single-layer attention with:
- **RBF (distance) kernel** instead of dot-product kernel
- **Category-conditional key selection** (MoE routing — classifier selects expert, not learned multi-head projections)
- **No stacking** — depth comes from the meta-graph reflection layer (v6.0+)

This is well-established in the attention literature. RBF kernels appear in kernel attention methods, Gaussian process attention, and the "Performers" random-feature attention approximation. Our system is a concrete instance of distance-kernel attention applied to ontological prototypes.

---

## 3. Design Principles

**P1–P12:** *(Unchanged from v7.)* Equation traceability, shape checks, domain-agnostic, incremental replacement, NumPy only, backend-swappable, factor preservation, event-driven, no debugger/git, relationship traversal, event emission, engine/product boundary.

**P13 — Profiles are first-class computational constituents.** *(New in v8.)* Profile centroids μ[c,a,:] are not configuration, not initialization data, not parameters — they ARE the model. The configured profiles represent Day 1 domain knowledge (the ontology). Learning refines them through operational experience. They must be inspectable, auditable, and overridable by domain experts.

**P14 — Similarity function must match data characteristics.** *(New in v8.)* The kernel K in the scoring equation must be appropriate for the factor distribution. L2 is the default. Mahalanobis for mixed-scale data. The GAE must support pluggable kernels because different domains have different factor distributions. Evidence: EXP-E1.

**P15 — Warm start from domain knowledge is essential at scale.** *(New in v8.)* Cold-start profile learning degrades from 89.9% (5×4×6) to 72.7% (20×10×20). Domain-provided initial profiles are not optional for production deployments. DomainConfig MUST provide `get_profile_config()`. Evidence: EXP-E2.

---

## 4. Experimental Foundation

This section summarizes the 9 bridge experiments and 5 validation experiments that settled the scoring architecture, plus 10 OP/synthesis experiments that validated the operator framework. Bridge and OP details in `bridge_experiments_catalog_v7.md`; validation details in *Cross-Graph Attention v3, §10.4*. **Total: 24 completed experiments.**

### 4.1 The Journey (Summary)

| # | Experiment | Hypothesis | Result | Key Number |
|---|---|---|---|---|
| 1 | EXP-5 | GT oracle improves learning | PASS | 79.65% |
| 2 | EXP-A | Gating (G) breaks capacity ceiling | FAIL | 49.27% (+0.01pp) |
| 3 | EXP-A2 | Per-category W breaks ceiling | FAIL | 51.61% (sample starvation) |
| 4 | EXP-C1 | Centroid oracle with L2 | **PASS** | **97.89%** |
| 5 | EXP-B1 | Profile scoring with learning | **PASS** | **98.2% warm** |
| 6 | EXP-D1 | Cross-category transfer | Marginal | Config wins by 2-14pp |
| 7 | EXP-D2 | Factor interactions | None | 0 significant |
| 8 | EXP-E1 | Kernel generalization | L2 wins 2/3 | Pluggable kernels |
| 9 | EXP-E2 | Scale to 20×10×20 | PASS | 99.9% oracle |

### 4.2 The Equation Progression

Every intermediate formulation was tested. Each failure led to the next insight.

```
Eq. 4 (published):       P(a|f) = softmax(f · W^T / τ)
  → 49.26% on realistic profiles. Magnitude-confounded.

Eq. 4-G (bridge v1):     P(a|f,c) = softmax((G[c,:] ⊙ f) · W^T / τ)
  → 49.27%. Diagonal projection too weak.

Eq. 4-aug:               P(a|f,c) = softmax([f; e_c] · W^T / τ)
  → 51.03%. Bias-only, still dot product.

Eq. 4' (per-category W): P(a|f,c) = softmax(f · W_c^T / τ)
  → 51.61%. Sample starvation (200 decisions/category).

Eq. 4-centroid-dot:      P(a|f,c) = softmax(f · μ[c,:,:]^T / τ)
  → 61.00%. Right profiles, wrong kernel.

Eq. 4-centroid-cos:      P(a|f,c) = softmax(cos(f, μ[c,:,:]) / τ)
  → 96.42%. Shape-based. Getting there.

Eq. 4-final (SETTLED):   P(a|f,c) = softmax(-||f - μ[c,a,:]||² / τ)
  → 98.2% with learning, 97.89% without. SETTLED.
```

### 4.3 What Was Eliminated (with Evidence)

| Element | Why Eliminated | Evidence |
|---|---|---|
| W (weight matrix) | Dot product is magnitude-confounded | EXP-C1: dot=61%, L2=97.89% |
| G (gating matrix) | Diagonal projection too weak | EXP-A: G-learned = +0.01pp |
| Ψ (warmup control) | Count-based decay in centroid update suffices | EXP-B1: natural phase behavior |
| Dot product kernel | Magnitude-confounded on [0,1] data | EXP-C1: 36.89pp gap vs L2 |
| Per-category W | Sample starvation from uniform init | EXP-A2: 51.61% (worse than simplified) |
| Factor interactions | Factors independently informative | EXP-D2: 0 significant (all gain < 0.67) |

### 4.4 What Was Validated (with Evidence)

| Element | Why Validated | Evidence |
|---|---|---|
| L2 distance kernel | Best on 2/3 distributions | EXP-E1: 97.9% original, 97.8% normalized |
| Profile centroids as keys | 97.89% with zero learning | EXP-C1: centroid oracle *(synthetic, GT profiles)* |
| Warm start from domain profiles | Essential at scale | EXP-E2: 99.9% warm vs 72.7% cold at xlarge |
| Count-based learning rate decay | Natural phase behavior | EXP-B1: cold recovers to 90.7% |
| Noise resistance to 30% oracle error | Greedy L2 from good init is self-correcting | EXP-B1: 98.1% at noise=0.30 |
| Pluggable kernels | Mahalanobis wins on mixed-scale data | EXP-E1: 92.9% Maha vs 79.9% L2 on mixed |
| Scaling to 20×10×20 | Higher dims = more separation | EXP-E2: 97.9% → 99.9% |
| Scalar σ (synthesis bias) improves accuracy at operative λ | GATE-OP passed — benefit concentrated in acute phase | EXP-OP1-FINAL: delta=+0.0041, p=0.0008 at λ=0.5 |
| Operative window λ∈[0.5, 0.6] | Bonferroni-significant; λ>0.6 asymmetric risk | EXP-OP1-FINAL narrow sweep; EXP-OP2: harm scales faster than benefit above λ=0.6 |
| τ_modifier REJECTED — temperature must be fixed at τ=0.1 | τ_modifier degrades calibration ECE +0.138 | S-series experiments; confirmed in §4.6 V3B |
| Centroid checkpoint infrastructure required | TTL expiry alone does not repair harmful operator damage | EXP-OP2: C-exp lasting damage −0.0124 post-expiry (TD-033) |
| Only 100%-correct operators Bonferroni-significant | No tolerance for partial σ accuracy | EXP-OP2: zero-crossing between 25% and 50% cell accuracy |
| update() has NO synthesis parameter — Loop 2/4 firewall | μ updated by experience only; σ never flows into update() | Architectural constraint; EXP-S3: firewall Frobenius 0.0028 |

### 4.5 Kernel Comparison (EXP-E1)

| Kernel | Original [0,1] | Normalized | Mixed-Scale [0,1]+[0,100]+[0,0.01] |
|---|---|---|---|
| **L2** | **97.9%** | **97.8%** | 79.9% |
| Mahalanobis | 97.7% | 97.5% | **92.9%** |
| Cosine | 96.4% | 96.7% | 61.2% |
| Dot product | 61.0% | 90.8% | 41.9% |

**Guidance:** L2 default. Mahalanobis when factors have different natural units. Dot product only with pre-normalized data (warned). Cosine as alternative for normalized data.

### 4.6 Validation Experiments (V1A–V3B)

Five validation experiments addressed concerns raised by independent LLM reviewers (GPT-5, Gemini 3.1) on the math paper. Every concern was confirmed; every one had a concrete fix. These findings are incorporated into the code and defaults above.

| # | Experiment | Concern | Result | GAE Impact |
|---|---|---|---|---|
| V1A | Scaling extension (n=2–15) | 5-point fit inflated exponent | b = 2.11, CI [2.09, 2.14] (was 2.30) — *simulation validated* | Moat calculations updated. Note: full O(n^2.11·t^γ) compounding expression is a projection — γ≈1.5 estimated, not measured (EXP-G1 pending). |
| V1B | Norm tracking (Eq. 13) | Residual enrichment may explode norms | CONFIRMED: 2.9M× after 5 sweeps | **Tier 5: LayerNorm required after each sweep** |
| V2 | Push update stability | Push rule may drive centroids out of bounds | CONFIRMED: escape at dec 6–12 adversarial | **Clipping already in ProfileScorer.update()** |
| V3A | Baseline comparison | L2 centroid vs ML baselines | L2: 94.78% vs XGBoost: 92.24% — *synthetic centroidal data* | Architecture validated — L2 competitive |
| V3B | Calibration (ECE) | Confidence scores may be miscalibrated | ECE=0.19 at τ=0.25; ECE=0.036 at τ=0.1 | **τ default changed from 0.25 to 0.1** |

**V3A online learning finding:** With only 200 training samples, L2 centroid reaches 94.3% immediately while XGBoost needs 1300 additional labeled examples to reach 91.5%. This validates the profile-based architecture's data efficiency — the core advantage of warm-start centroid scoring over periodic batch retraining.

**Synthetic data caveat:** All experiments use synthetic data with centroidal class structure. Real SOC data validation — with heavy-tailed distributions, correlated factors, concept drift, and adversarial behavior — is the critical next step.

---

## 5. The Ontological Architecture

### 5.1 The Core Claim

**Ontologies (domain concepts) → Meta-graphs (computational structure) → Profiles (compiled neighborhoods) → Scoring (distance-kernel attention).**

Profiles are compiled ontology. The meta-graph is the source code. You edit the source (reflection-time), recompile (update profiles), and run the compiled version (decision-time). This is the compile-vs-interpret tradeoff.

### 5.2 Three-Layer Design

```
Layer 1: Ontology (static + slow evolution)
├── Concepts: categories, factors, actions, relationships
├── Expressed as: DomainConfig YAML (realistic_profiles)
├── Source: domain experts, industry standards, analyst training
├── Updated: manually, or by meta-graph discovery (v6.0+)
└── Timescale: weeks to months

Layer 2: Compiled Profiles (dynamic, per-decision refinement)
├── μ[c, a, :] = compiled ontological neighborhood for (category, action)
├── Initialized from Layer 1 config (Day 1 analyst knowledge) — P15
├── Refined by Eq. 4b-final (centroid pull/push with count decay)
├── Used at decision-time: P(a|f,c) = softmax(-K(f, μ[c,a,:]) / τ)
├── Inspectable, auditable, overridable (P13)
└── Timescale: per-decision updates (milliseconds)

Layer 3: Meta-Graph Reflection (v6.0+)
├── Full relational structure with learned edge weights
├── Built from Layers 1 + 2 + outcome history
├── Used for: drift detection, missing factor diagnosis, transfer learning
├── Updates Layer 1 when structural changes detected
└── Timescale: hours to days (offline batch)
```

### 5.3 The Bridge — Solved

**Original problem:** How does the SituationAnalyzer's category classification connect to the scoring math?

**Answer (experimentally validated):** The bridge is **ontological routing**. Category classification selects the appropriate profile set. Scoring computes distance between alert factors and each action's profile centroid within that set.

This is why G (gating) failed: the bridge is not a matrix multiplication. It's a **lookup** — category → profile set — mediated by the ontology. The SituationAnalyzer already does the hard work (classification). The bridge is just routing.

### 5.4 Three First-Class Computational Constituents

| Element | Shape | Role | Source |
|---|---|---|---|
| **f** | 1 × n_f | Alert factor vector (query) | Tier 1 factor computation |
| **μ** | n_c × n_a × n_f | Profile centroids (keys) | Compiled ontology + operational refinement |
| **c** | scalar | Category index (head selector) | SituationAnalyzer classification |

Temperature τ is a hyperparameter. The kernel K is a configuration choice. These are not first-class elements.

---

## 6. Package Structure — Three-Repository Architecture

*(§4 in v7. Unchanged in principle. Updated repo structure in §18.)*

Three repos: GAE (math, numpy-only, Apache 2.0) → ci-platform (infrastructure) → domain copilots (expertise).
Dependency arrow is strictly one-way: copilot → platform → GAE (P12).

---

## 7. Causal Architecture: Three Loops, Six Connectors

### 7.1 Three Causal Loops (Updated for Profile-Based Scoring)

```
╔══════════════════════════════════════════════════════════════════╗
║  FAST LOOP (per-alert, seconds)                                  ║
║                                                                  ║
║  Alert → [Factor Computation] → f (1×n_f)                       ║
║            ↑ reads graph              ↓                          ║
║            │             [SituationAnalyzer] → category c        ║
║            │                          ↓                          ║
║            │             [Profile Scoring] Eq. 4-final           ║
║            │             P(a|f,c) = softmax(-K(f, μ[c,:,:]) / τ) ║
║            │                          ↓                          ║
║            │             Action + Confidence + Distances          ║
║            │                          ↓                          ║
║            │             [LLM Narrative]                          ║
║            │                          ↓                          ║
║            │             Human/Auto Decision                      ║
║            │                          ↓                          ║
║            │             [Outcome Verification]                   ║
║            │                          ↓                          ║
║            │             [Profile Update] Eq. 4b-final            ║
║            │             μ[c,a,:] ← centroid pull/push           ║
║            │                          ↓                          ║
║            │             [Graph Write-Back]                       ║
║            │             Decision node + profile snapshot         ║
║            │                          │                          ║
║            └──────────────────────────┘                          ║
║                                                                  ║
║  CLOSURE: Graph is richer. Profiles are refined.                 ║
║  NEXT ALERT: factors from richer graph, scored by refined        ║
║  profiles → different (better) decision.                         ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║  SLOW LOOP (periodic, hours/daily) [v5.5]                        ║
║  (Unchanged from v7 — embeddings + cross-graph attention)        ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║  META LOOP (rare, structural) [v6.0+]                            ║
║                                                                  ║
║  REVISED from v7: Meta-graph reflection, not W expansion.        ║
║  Triggers:                                                       ║
║    (A) Ψ_anomaly[c] exceeds threshold → novel pattern detected  ║
║    (B) Ψ_stability[c] → drift detected → profile recalibration  ║
║    (C) Cross-category Ψ_separation declining → categories merging║
║                                                                  ║
║  Actions:                                                        ║
║    Reconstruct meta-graph from profiles + outcome history         ║
║    Discover: missing factors, structural changes, transfer opps  ║
║    Update: Layer 1 ontology → recompile Layer 2 profiles         ║
║                                                                  ║
║  Evidence: EXP-D1 (transfer marginal), EXP-D2 (no interactions) ║
║  → Meta-loop value is in drift/anomaly detection, not transfer.  ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║  SYNTHESIS LOOP (external intelligence) [PROPOSAL — v5.5+]      ║
║  GATED BY: GATE-M (math), GATE-D (pipeline), GATE-V (decisions) ║
║                                                                  ║
║  Loop 4 brings external intelligence and internal context        ║
║  into the scoring equation via synthesis bias σ[c,a].            ║
║                                                                  ║
║  Sources → Claims → σ[c,a] → Eq. 4-synthesis:                   ║
║    P(a|f,c,σ) = softmax(−(‖f − μ‖² + λ·σ[c,a]) / τ)           ║
║    λ=0: exact Eq. 4-final behavior (kill switch)                 ║
║    λ∈[0.5,0.6]: operative window (GATE-OP validated)             ║
║                                                                  ║
║  ARCHITECTURAL FIREWALL (PERMANENT):                             ║
║    μ (Loop 2) is NEVER updated using σ (Loop 4).                 ║
║    update() has NO synthesis parameter — by design.              ║
║    σ flows only through score(), never through update().         ║
║    Loop 2 learns from experience. Loop 4 informs at decision     ║
║    time. They are computationally independent.                   ║
║                                                                  ║
║  Temperature: τ is FIXED at τ=0.1 regardless of synthesis state.║
║  τ_modifier was tested and rejected (ECE degradation +0.138).    ║
║                                                                  ║
║  Status: GATE-OP PASSED (p=0.0008 at λ=0.5).                    ║
║  GATE-M (EXP-S2-REPRO at operative λ) and GATE-D pending.       ║
║  See intelligence_layer_design_v2 for full specification.        ║
╚══════════════════════════════════════════════════════════════════╝
```

### 7.2 Six Causal Connectors (Updated)

**Connector 1: Graph → Factors (read)** — *Unchanged from v7.*

**Connector 2: Factors × Profiles → Decision (compute)** — *REVISED*

| Attribute | v7 Specification | v8 Specification |
|---|---|---|
| Trigger | Factor vector f ready | Factor vector f ready + category c classified |
| Input | f, W, τ, actions | f, μ[c,:,:], τ, actions, kernel K |
| Output | Action probs, selected action, confidence | Action probs, selected action, confidence, **distances** |
| Mechanism | Eq. 4: softmax(f · W^T / τ) | **Eq. 4-final: softmax(-K(f, μ[c,:,:]) / τ)** |
| Causal dependency | W encodes all prior learning | **μ encodes compiled ontology + operational refinement. The profiles ARE the accumulated judgment.** |
| Implementation | `gae/scoring.py` → `score_alert()` | **`gae/profile_scorer.py` → `ProfileScorer.score()`** |

**Connector 3: Decision + Outcome → Profile Update (learn)** — *REVISED*

| Attribute | v7 Specification | v8 Specification |
|---|---|---|
| Mechanism | Eq. 4b: W[a,:] += α·r(t)·f(t)·δ(t) (Hebbian) | **Eq. 4b-final: centroid pull (correct) / push (incorrect) with count decay** |
| What changes | W rows (hyperplane normals) | **μ[c,a,:] centroids (prototypes)** |
| Implementation | `gae/learning.py` → `LearningState.update()` | **`gae/profile_scorer.py` → `ProfileScorer.update()`** |

**Connectors 4-6:** *Unchanged from v7.* (Discovery → graph write-back, embeddings, artifact evolution.)

---

## 8. Tier 1 — Factor Protocol & Assembly (`gae/factors.py`)

*(Unchanged from v7 §6. FactorComputer Protocol, assemble_factor_vector(), 6 SOC factor implementations. See gae_design_v7 §6 for full specification.)*

One note: the factor vector f is now the **query** in Eq. 4-final (distance-kernel attention), not the left operand in a dot product. The protocol is identical — only the downstream consumer (Tier 2) has changed.

---

## 9. Tier 2 — Profile-Based Scoring (`gae/profile_scorer.py`)

### 9.1 What This Implements

**Eq. 4-final:** P(a | f, c) = softmax(-K(f, μ[c,a,:]) / τ)

This replaces v7's `gae/scoring.py` (Eq. 4: f · W^T / τ). The old `scoring.py` is retained for backward compatibility (deprecated, removed at v5.5).

### 9.2 Why This Change (Experimental Evidence)

The published Eq. 4 uses dot product over a shared weight matrix W. This achieves **49.26%** on realistic SOC data (EXP-A). The root cause is magnitude confounding: factors with high mean values dominate the dot product regardless of discriminative value.

Profile-based scoring with L2 distance achieves **98.2%** on the same data (EXP-B1). The improvement comes from two changes: (1) per-category profile sets instead of shared W, and (2) L2 distance instead of dot product. EXP-C1 proved that the kernel change alone (with zero learning) achieves 97.89%.

### 9.3 Interface

```python
# gae/profile_scorer.py
"""
Tier 2: Profile-Based Scoring — Eq. 4-final from the experimental validation.

P(a | f, c) = softmax(-K(f, μ[c,a,:]) / τ)

f: factor vector (1 × n_f), computed by Tier 1
μ: profile centroids (n_c × n_a × n_f), compiled ontology
c: category index, from SituationAnalyzer
K: distance kernel (L2 default, pluggable)
τ: temperature scalar, from CalibrationProfile

Three first-class computational constituents: f (query), μ (keys), c (head selector).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from enum import Enum

class KernelType(Enum):
    L2 = "l2"                    # Default. -||f - μ||². Best for uniform-scale data.
    MAHALANOBIS = "mahalanobis"  # Scale-invariant. Needs covariance estimation.
    COSINE = "cosine"            # Shape-based. Alternative for normalized data.
    DOT = "dot"                  # WARNED. Only with pre-normalized data.


@dataclass
class ScoringResult:
    """Output of Eq. 4-final."""
    action_probabilities: np.ndarray   # shape (1, n_a) — softmax output
    selected_action: str               # argmax action name
    confidence: float                  # max probability
    distances: np.ndarray              # raw distance values (before softmax)
    factor_vector: np.ndarray          # f — preserved per R4
    category_index: int                # c — which profile set was used
    temperature: float                 # τ used
    kernel: str                        # which kernel was used


class ProfileScorer:
    """
    Profile-based scoring with pluggable distance kernels.
    
    The profile centroids μ[c, a, :] are the model. They represent
    "what does action a typically look like for category c?" Scoring
    asks "which action's profile is closest to this alert's factors?"
    
    Initialization:
        - Warm start: from DomainConfig.get_profile_config() (Day 1 knowledge)
        - Cold start: uniform 0.5 (viable for small scale, degrades at scale — P15)
    
    Learning:
        - Eq. 4b-final: centroid pull (correct) / push (incorrect)
        - Count-based decay: η_eff = η / (1 + n[c,a] * decay_rate)
        - Natural phase behavior: early updates large, late updates small
    
    Kernels:
        - L2 (default): -||f - μ||². Best for uniform-scale [0,1] data.
        - Mahalanobis: -(f-μ)^T Σ⁻¹ (f-μ). For mixed-scale data.
        - Cosine: cos(f, μ). For pre-normalized data.
        - Dot: f · μ. WARNED — only with normalization.
    
    Evidence: EXP-C1 (97.89% L2 oracle), EXP-B1 (98.2% warm),
              EXP-E1 (L2 wins 2/3), EXP-E2 (scales to 20×10×20).
    """
    
    def __init__(
        self,
        n_categories: int,
        n_actions: int,
        n_factors: int,
        actions: List[str],
        kernel: KernelType = KernelType.L2,
        profile: 'CalibrationProfile' = None,
    ):
        self.n_categories = n_categories
        self.n_actions = n_actions
        self.n_factors = n_factors
        self.actions = actions
        self.kernel = kernel
        
        # Profile hyperparameters from CalibrationProfile
        tau = profile.temperature if profile else 0.1  # V3B: τ=0.1 for calibrated confidence (ECE 0.036 vs 0.19 at 0.25)
        self.tau = tau
        self.eta = profile.extensions.get("eta", 0.05) if profile else 0.05
        self.eta_neg = profile.extensions.get("eta_neg", 0.05) if profile else 0.05
        self.decay_rate = profile.extensions.get("count_decay", 0.001) if profile else 0.001
        
        # Profile centroids — the model (P13)
        self.mu = np.full((n_categories, n_actions, n_factors), 0.5)
        
        # Per-(category, action) observation counts — for learning rate decay
        self.counts = np.zeros((n_categories, n_actions), dtype=int)
        
        # Covariance matrices — only for Mahalanobis kernel
        self.cov_inv = None  # shape (n_categories, n_actions, n_factors, n_factors)
        
        # Kernel warning
        if kernel == KernelType.DOT:
            import warnings
            warnings.warn(
                "Dot product kernel is magnitude-confounded on non-normalized data. "
                "EXP-E1 evidence: 61% (dot) vs 97.9% (L2) on [0,1] factors. "
                "Use L2 or normalize your factors first.",
                UserWarning
            )
    
    def init_from_config(self, profile_config: dict, categories: List[str]):
        """
        Load Day 1 ontology from DomainConfig (P15).
        
        profile_config: {category_name: {action_name: [factor_values]}}
        categories: ordered list of category names
        """
        for c, cat in enumerate(categories):
            if cat in profile_config:
                for a, act in enumerate(self.actions):
                    if act in profile_config[cat]:
                        self.mu[c, a, :] = np.array(
                            profile_config[cat][act], dtype=np.float64
                        )
    
    def score(self, f: np.ndarray, category_index: int) -> ScoringResult:
        """
        Eq. 4-final: P(a | f, c) = softmax(-K(f, μ[c,:,:]) / τ)
        
        Shape checks:
            f: (1, n_f) — factor vector from Tier 1
            μ[c,:,:]: (n_a, n_f) — profile set for this category
            output: (1, n_a) — action probability distribution
        """
        f_flat = f.flatten()
        c = category_index
        
        assert f_flat.shape == (self.n_factors,), \
            f"f must have {self.n_factors} factors, got {f_flat.shape}"
        assert 0 <= c < self.n_categories, \
            f"category_index {c} out of range [0, {self.n_categories})"
        
        # Compute kernel distances
        distances = self._compute_distances(f_flat, c)
        
        # Softmax with temperature
        neg_dist_scaled = -distances / self.tau
        neg_dist_scaled -= np.max(neg_dist_scaled)  # numerical stability
        probs = np.exp(neg_dist_scaled)
        probs /= probs.sum() + 1e-10
        
        assert abs(probs.sum() - 1.0) < 1e-6, "Probabilities must sum to 1"
        
        selected_idx = int(np.argmax(probs))
        return ScoringResult(
            action_probabilities=probs.reshape(1, -1),
            selected_action=self.actions[selected_idx],
            confidence=float(probs[selected_idx]),
            distances=distances,
            factor_vector=f.copy(),
            category_index=c,
            temperature=self.tau,
            kernel=self.kernel.value,
        )
    
    def _compute_distances(self, f: np.ndarray, c: int) -> np.ndarray:
        """Pluggable kernel computation. Returns distance array (n_actions,)."""
        mu_c = self.mu[c]  # (n_actions, n_factors)
        
        if self.kernel == KernelType.L2:
            return np.array([np.sum((f - mu_c[a]) ** 2) for a in range(self.n_actions)])
        
        elif self.kernel == KernelType.COSINE:
            f_norm = np.linalg.norm(f) + 1e-10
            return np.array([
                1.0 - np.dot(f, mu_c[a]) / (f_norm * np.linalg.norm(mu_c[a]) + 1e-10)
                for a in range(self.n_actions)
            ])
        
        elif self.kernel == KernelType.DOT:
            return np.array([-np.dot(f, mu_c[a]) for a in range(self.n_actions)])
        
        elif self.kernel == KernelType.MAHALANOBIS:
            if self.cov_inv is None:
                return self._compute_distances_l2(f, c)  # fallback
            return np.array([
                (f - mu_c[a]) @ self.cov_inv[c, a] @ (f - mu_c[a])
                for a in range(self.n_actions)
            ])
    
    def update(
        self,
        f: np.ndarray,
        category_index: int,
        action_index: int,
        correct: bool,
    ) -> dict:
        """
        Eq. 4b-final: Centroid pull (correct) / push (incorrect).
        
        Correct:   μ[c,a,:] += η_eff · (f - μ[c,a,:])
        Incorrect: μ[c,a,:] -= η_neg_eff · (f - μ[c,a,:])
        
        η_eff = η / (1 + n[c,a] · decay_rate)
        
        Count-based decay provides natural phase behavior:
        early updates are large (centroid finding), late updates
        refine (centroid calibration). This replaces Ψ warmup control.
        
        Evidence: EXP-B1 best configs —
          noise=0%: η=0.05, η_neg=0.05
          noise=15%: η=0.01, η_neg=0.05
        
        ARCHITECTURAL FIREWALL (PERMANENT — v8.2):
        This method has NO synthesis parameter and accepts NO σ input.
        This is the Loop 2 / Loop 4 separation boundary.
        μ[c,a,:] is ONLY updated from operational experience (f, correct).
        σ (synthesis bias) flows ONLY through score() via Eq. 4-synthesis.
        The two loops are computationally independent by design.
        Violating this constraint would corrupt the Loop 2 learning signal.
        EXP-S3 validates the firewall: Frobenius norm 0.0028 (0.28%).
        DO NOT add a synthesis parameter to this method.
        """
        f_flat = f.flatten()
        c, a = category_index, action_index
        
        decay = 1.0 / (1.0 + self.counts[c, a] * self.decay_rate)
        mu_before = self.mu[c, a, :].copy()
        
        if correct:
            eta_eff = self.eta * decay
            self.mu[c, a, :] += eta_eff * (f_flat - self.mu[c, a, :])
        else:
            eta_neg_eff = self.eta_neg * decay
            self.mu[c, a, :] -= eta_neg_eff * (f_flat - self.mu[c, a, :])
        
        # PRODUCTION REQUIREMENT (V2 validated): Without clipping, sustained incorrect
        # outcomes drive centroids outside [0,1] within 6-12 decisions under adversarial
        # conditions (norms reach 4,608× by decision 200). Normal operation (70% correct)
        # is safe (~1.6× norms), but clipping is required for all conditions.
        self.mu[c, a, :] = np.clip(self.mu[c, a, :], 0.0, 1.0)
        self.counts[c, a] += 1
        
        return {
            "category_index": c,
            "action_index": a,
            "correct": correct,
            "eta_effective": self.eta * decay if correct else self.eta_neg * decay,
            "mu_before": mu_before,
            "mu_after": self.mu[c, a, :].copy(),
            "delta_norm": float(np.linalg.norm(self.mu[c, a, :] - mu_before)),
            "count": int(self.counts[c, a]),
        }
    
    def get_diagnostics(self) -> dict:
        """
        Ψ diagnostic functions — computed from profile state (§5.4 in bridge doc).
        NOT control mechanisms. Monitoring only.
        """
        diagnostics = {}
        for c in range(self.n_categories):
            pairwise = []
            for a1 in range(self.n_actions):
                for a2 in range(a1 + 1, self.n_actions):
                    pairwise.append(np.linalg.norm(
                        self.mu[c, a1, :] - self.mu[c, a2, :]))
            diagnostics[c] = {
                "mean_separation": float(np.mean(pairwise)) if pairwise else 0.0,
                "min_separation": float(np.min(pairwise)) if pairwise else 0.0,
                "total_observations": int(np.sum(self.counts[c, :])),
            }
        return diagnostics
```

### 9.4 Backward Compatibility

```python
# gae/scoring.py — DEPRECATED, retained for backward compat until v5.5
# Original Eq. 4: P(a|f) = softmax(f · W^T / τ)
# Use ProfileScorer (gae/profile_scorer.py) for new code.

import warnings
def score_alert(f, W, actions, tau=None, profile=None):
    """DEPRECATED. Use ProfileScorer.score() instead."""
    warnings.warn("score_alert() uses dot-product scoring (49% on realistic data). "
                   "Use ProfileScorer with L2 distance (98.2%). "
                   "See gae_design_v8 §9.", DeprecationWarning, stacklevel=2)
    # ... existing implementation unchanged ...
```

---

## 10. Tier 3 — Profile Learning (`gae/profile_scorer.py`)

### 10.1 What This Implements

**Eq. 4b-final:** Centroid pull/push with count-based decay.

This replaces v7's Hebbian update (Eq. 4b: W[a,:] += α·r(t)·f(t)·δ(t)).

### 10.2 Why This Change

Hebbian learning discovers weight vectors from scratch. With 1000 decisions across 5 categories × 4 actions, each (category, action) pair gets ~50 decisions on average. Hebbian from uniform initialization cannot converge in 50 decisions (EXP-A2: 51.61%).

Profile learning starts from domain knowledge (configured centroids) and refines. The configured profiles already achieve 97.89% (EXP-C1). Learning adds +0.3pp by adapting to deployment-specific drift.

### 10.3 The Update Rule

```
Eq. 4b-final:
  On correct decision:
    μ[c, a, :] += η / (1 + n[c,a] · d) · (f - μ[c, a, :])
    
  On incorrect decision:
    μ[c, a, :] -= η_neg / (1 + n[c,a] · d) · (f - μ[c, a, :])
    
  μ[c, a, :] = clip(μ, 0, 1)
  n[c, a] += 1
  
Where:
  η = 0.05        base learning rate (correct decisions)
  η_neg = 0.05    base penalty rate (incorrect decisions)
  d = 0.001        count decay rate
  n[c, a]          observation count for this (category, action) pair
```

**Clipping is a production requirement, not an optimization.** Validation experiment V2 tested push update stability under 5 conditions. Under normal operation (70% correct), centroid norms remain bounded at ~1.6× — the push rule is safe. Under sustained adversarial conditions (100% incorrect), centroids escape [0, 1] within 6–12 decisions and reach norms of 4,608× by decision 200. With clipping, centroids remain bounded under all tested conditions including pure adversarial. Margin-based alternatives (apply push only when margin < threshold) do not prevent escape.

**Note on convergence:** The pull/push rule with count-based decay is a custom heuristic (structurally related to Learning Vector Quantization, Kohonen 1990), not a derived optimization of an explicit loss function. Formal convergence analysis under non-stationary data remains future work; empirically, the system converges reliably across all tested conditions.

**Production Requirements (v8.2 additions from OP series):**

**TD-027 — Centroid RESET before L2 deployment (not migration).** Any system running the deprecated ScoringMatrix (Eq. 4, dot product) has W-based state. W is a weight matrix; μ is a centroid array. There is no meaningful mapping between them — migrating W values into μ would produce incorrect centroids that would require hundreds of decisions to correct. The correct procedure is: reset μ to the configured DomainConfig profiles (Day 1 knowledge), discard W state, and let Loop 2 relearn from the new baseline. This is not data loss — the Day 1 profiles already achieve 97.89% accuracy (EXP-C1).

**TD-033 — Centroid checkpoint/rollback infrastructure required before Loop 4 activation.** EXP-OP2 demonstrated that a harmful operator (incorrect σ cells) leaves lasting centroid damage beyond TTL expiry: −0.0124 accuracy post-expiry, 35% never-recover within 400 decisions. The only repair mechanism is rolling back to a pre-operator checkpoint. This means ProfileScorer must support: `checkpoint()` (snapshot μ at operator start), `rollback(checkpoint_id)` (restore μ to snapshot). TTL expiry alone is insufficient — it removes σ but does not undo the centroid updates that occurred while σ was active. **Checkpoint every 50 decisions minimum.**

**TD-034 — τ recalibration is a hard prerequisite before Loop 2 activates in production.** τ=0.1 was validated on synthetic centroidal data (V3B: ECE=0.036). Real SOC data may have different factor distributions. The optimal τ for real data may differ. τ recalibration must be run on a sample of real alerts (minimum 200) before the first centroid update in production. This prevents operating with a miscalibrated confidence distribution from decision 1.

### 10.4 What Happened to v7 Hardening (A1, A2, A4, C3)

| Hardening | v7 Role | v8 Status |
|---|---|---|
| A1 (confidence discount) | α_eff = α × (1 - discount × confidence) | **Subsumed** by count-based decay. Count decay provides the same effect: early decisions (low count = low historical confidence) get large updates; late decisions (high count = established profile) get small updates. |
| A2 (per-factor decay) | ε_vector applies different decay rates per factor | **Deferred to v5.5.** Profile centroids don't decay toward zero — they drift toward observed data. Per-factor decay is less relevant when the model is centroid-based rather than weight-based. Can be added as differential η per factor if needed. |
| A4 (provisional dimensions) | New W columns provisional until reinforced | **Replaced** by profile augmentation. New factor dimensions added to μ via config change. No provisional/established state needed — the centroid update naturally calibrates new dimensions. |
| C3 (delayed validation) | Autonomous decisions deferred | **Preserved.** Autonomous decisions still need outcome validation before centroid updates. Implementation in ProfileScorer.update() via decision_source parameter. |

### 10.5 CalibrationProfile (Updated for Profiles)

```python
# gae/calibration.py — UPDATED for v8
@dataclass
class CalibrationProfile:
    """
    Domain-configurable hyperparameters for the GAE.
    
    v8 additions:
      - eta, eta_neg: profile learning rates (replace learning_rate for centroids)
      - count_decay: decay rate for count-based η reduction
      - kernel: default kernel type for this domain
    
    v7 fields preserved for backward compatibility:
      - learning_rate, penalty_ratio: used by deprecated ScoringMatrix
      - temperature: τ for softmax (used by both old and new scoring)
      - decay_class_rates: per-factor decay (v5.5)
      - discount_strength: A1 confirmation bias (subsumed by count decay)
    """
    # Profile learning parameters (v8 — Eq. 4b-final)
    temperature: float = 0.1  # V3B validated: ECE=0.036 at τ=0.1 vs ECE=0.19 at τ=0.25
    
    # v7 backward compat (deprecated, used by ScoringMatrix)
    learning_rate: float = 0.02
    penalty_ratio: float = 20.0
    
    # Per-factor decay (v5.5)
    decay_class_rates: dict = field(default_factory=lambda: {
        "permanent": 0.0001, "standard": 0.001,
        "campaign": 0.003, "transient": 0.01,
    })
    
    # A1 confirmation bias (subsumed by count decay in v8)
    discount_strength: float = 0.0
    
    # Extension point — v8 profile parameters live here until stable
    extensions: dict = field(default_factory=lambda: {
        "eta": 0.05,           # base learning rate (correct)
        "eta_neg": 0.05,       # base penalty rate (incorrect)
        "count_decay": 0.001,  # count-based decay rate
        "kernel": "l2",        # default kernel type
    })
```

---

## 11. Level 2/3 — Design Specification (v7/v8 Targets)

> **Status of this section:** Level 2 (GraphAttentionBridge) and Level 3 (Cross-Domain Discovery) are **designed but not implemented**. Implementation targets are v7.0 and v8.0 respectively. Level 1 (Tiers 1-3: ProfileScorer + learning) is fully specified and ships in v5.0. The design in this section determines what Level 1 must preserve now so Level 2/3 can be built later without breaking changes.

> **Why this section exists in v8.3:** The Level 2/3 design affects three other documents (math_synopsis, architecture_philosophy, soc_copilot_design). Those documents need a single authoritative source for the interface contracts, γ definition, and data hooks. This is that source.

---

### 11.1 Tier 4 — Entity Embeddings (`gae/embeddings.py`, v5.5)

Tier 4 computes per-entity embedding vectors from graph properties. These are the inputs to Level 2 cross-graph attention. They are implemented in v5.5 (before Level 2) so the embedding infrastructure and data are available when Level 2 ships in v7.

**EmbeddingProvider protocol:**

```python
# gae/embeddings.py (v5.5)
class EmbeddingProvider(Protocol):
    """
    Produces embedding vectors for graph entities.
    Used by Level 2 (GraphAttentionBridge) as input.
    """
    @property
    def embedding_dim(self) -> int: ...
    
    def embed_entity(self, entity_id: str, properties: dict) -> np.ndarray:
        """Returns embedding vector of shape (embedding_dim,)."""
        ...
    
    def embed_graph(self, graph: 'GraphSnapshot') -> np.ndarray:
        """
        Returns embedding matrix E of shape (n_entities, embedding_dim).
        Row i = embedding of entity i.
        """
        ...
```

**Two implementations (both v5.5):**
- `PropertyEmbedding`: Linear projection of entity property vector. No external model needed. Fast, interpretable.
- `TransformerEmbedding`: Sentence transformer on text properties. Richer but requires a model download.

**V1B production requirement:** Enrichment norms explode without LayerNorm after cross-graph attention (2.9M× after 5 sweeps). Normalization step is mandatory at Level 2. See §11.2.

---

### 11.2 Level 2 — GraphAttentionBridge (`gae/bridge.py`, v7.0 target)

**What this does:** Given embedding matrices from two knowledge graph domains, compute attention-weighted enrichment of one domain's entities using the other domain's context. This is cross-domain intelligence transfer — entities in the SOC graph learn from patterns in (e.g.) the vulnerability graph, and vice versa.

**What this does NOT do:** It does not replace or modify Level 1 scoring. It does not update profile centroids μ. It does not flow through ProfileScorer.update(). It is a separate enrichment path that produces enriched entity representations for use in downstream discovery (Level 3) and display (Tab 5).

**Equation (Eq. 6 — cross-graph attention):**

```
CrossAttention(G_i, G_j):
    Q_i = E_i · W_Q        # queries from source graph (n_i × d)
    K_j = E_j · W_K        # keys from target graph (n_j × d)  
    V_j = E_j · W_V        # values from target graph (n_j × d)
    
    A_ij = softmax(Q_i · K_j^T / √d)     # attention weights (n_i × n_j)
    R_ij = A_ij · V_j                     # attended values (n_i × d)
    
    return R_ij
    
Enrichment (Eq. 5 — with V1B LayerNorm, mandatory):
    E_i^enriched = LayerNorm(E_i + Σ_{j≠i} CrossAttention(G_i, G_j))
```

**V1B production requirement (hard constraint, not optional):**
Without LayerNorm, enrichment norms grow by 2.9M× after 5 sweeps. With LayerNorm, norms remain bounded. LayerNorm is applied after every enrichment sweep. This is not a tuning choice — it is required for numerical stability in production.

**Interface:**

```python
# gae/bridge.py (v7.0 — DESIGNED, NOT IMPLEMENTED)
@dataclass
class GraphSchema:
    """
    Registered schema for a domain's knowledge graph.
    Required for Level 2 cross-graph attention.
    Written by Level 1 domains via §11.4 hooks.
    """
    domain_id: str                      # e.g. "soc", "s2p", "vuln"
    entity_types: List[str]             # e.g. ["Alert", "Asset", "CVE"]
    relationship_types: List[str]       # e.g. ["AFFECTS", "ESCALATED_TO"]
    factor_names: List[str]             # factor dimensions in this domain
    embedding_dim: int                  # dimension of entity embeddings
    n_categories: int                   # C — number of categories in this domain
    n_actions: int                      # A — number of actions in this domain


class GraphAttentionBridge:
    """
    Level 2: Cross-graph attention enrichment.
    
    Computes Eq. 5 + Eq. 6: enriched entity embeddings by attending
    across multiple knowledge graph domains.
    
    Prerequisites:
        - EmbeddingProvider for each registered domain (v5.5)
        - GraphSchema registered for each domain (§11.4)
        - GraphSnapshot from each domain's Level 1 write-back (§11.4)
        - LayerNorm applied after every enrichment sweep (V1B)
    
    v7.0 implementation target.
    GATE prerequisite: EXP-G1 validates γ before cross-domain 
    compounding claims are made externally.
    
    Relationship to Level 1:
        - Reads: GraphSnapshot (written by Level 1 §11.4 hooks)
        - Does NOT write to: ProfileScorer.mu (firewall preserved)
        - Does NOT call: ProfileScorer.update()
        - Writes to: EnrichedEmbeddingStore (new, Level 2 only)
    """
    
    def register_domain(
        self,
        schema: GraphSchema,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        """Register a domain for cross-graph attention."""
        ...
    
    def enrich(
        self,
        source_domain: str,
        target_domains: List[str],
        snapshot: 'GraphSnapshot',
    ) -> np.ndarray:
        """
        Compute enriched embeddings for source_domain entities
        using context from target_domains.
        
        Returns E_enriched: shape (n_source_entities, embedding_dim).
        LayerNorm applied internally (V1B — mandatory).
        """
        ...
    
    def sweep(self, n_sweeps: int = 1) -> dict:
        """
        Run n_sweeps enrichment passes across all registered domains.
        
        Returns: {domain_id: enriched_embeddings, ...}
        
        Each sweep: for each domain, attend to all other domains,
        apply LayerNorm, store enriched embeddings.
        Slow loop — runs periodically (hours/daily), not per-alert.
        """
        ...
```

**What Level 2 enables:**
- Entities in the SOC graph can "see" patterns in the vulnerability graph and vice versa.
- An alert about CVE-2024-XXXX is scored with the benefit of vulnerability intelligence, not just SOC operational history.
- The enriched embeddings feed Level 3 (cross-domain discovery) and Tab 5 (synthesis briefing display).

**What Level 2 does not enable:**
- It does not improve ProfileScorer.score() directly. Enriched embeddings are not factor vectors (f). The scoring path (f → μ → action) is unchanged.
- It does not replace the synthesis layer (Loop 4). σ comes from claims parsing, not from graph attention.
- It does not operate in real-time. It runs as the slow loop (periodic, hours/daily).

---

### 11.3 Level 3 — Cross-Domain Discovery (`gae/discovery_engine.py`, v8.0 target)

**What this does:** Given enriched entity embeddings from multiple domains (produced by Level 2), discover cross-domain patterns — entities, relationships, or behavioral signatures that span more than one domain's knowledge graph. These discoveries are candidates for surfacing in the synthesis briefing (Tab 5) and for analyst review.

**What this does NOT do:** Level 3 does not make decisions. It does not modify centroids. It is a pattern-mining layer that produces candidate discoveries for human review, not autonomous actions.

**Equations (Eq. 8a-8c — cross-domain discovery):**

```
Discovery scoring (Eq. 8a):
    score(e_i, e_j) = σ(W_d · [e_i ; e_j ; e_i ⊙ e_j])
    
    e_i, e_j: enriched entity embeddings from different domains
    W_d: learned discovery weight vector
    [;]: concatenation, ⊙: elementwise product
    σ: sigmoid

Threshold (Eq. 8b):
    discovery(e_i, e_j) = 1  if score(e_i, e_j) > δ_discovery
                           0  otherwise

Pattern extraction (Eq. 8c):
    pattern = {entity_i, entity_j, score, relationship_type, confidence}
    discoveries = {pattern | score > δ_discovery, domains(i) ≠ domains(j)}
```

**The γ definition (precise):**

γ is the cross-domain compounding exponent in the system intelligence scaling expression:

```
𝒮(n, t) ~ O(n^{2.11} · t^γ)

Where:
    n = number of entities across all registered domains
    t = number of completed enrichment sweeps (each sweep = one slow-loop pass)
    b = 2.11 (validated by simulation, V1A: CI [2.09, 2.14])
    γ = cross-domain compounding exponent (ESTIMATED at ≈1.5, NOT YET MEASURED)

γ measures how marginal discovery yield grows with each additional enrichment sweep:
    marginal_yield(t) = 𝒮(n, t) - 𝒮(n, t-1) ~ O(n^{2.11} · γ · t^{γ-1})
    
    If γ = 1.0: marginal yield is constant — each sweep produces the same number
                of new discoveries regardless of t. No compounding.
    If γ > 1.0: marginal yield increases with t — later sweeps are more productive
                than earlier ones because enriched embeddings expose patterns invisible
                to raw embeddings. This is the compounding claim.
    If γ < 1.0: diminishing returns — earlier sweeps are most productive.
```

**γ is currently an estimate (≈1.5). It is not validated.** The compounding claim — that the system gets meaningfully richer with each sweep — depends on γ > 1.0 being measured, not assumed. EXP-G1 is the validation experiment.

**EXP-G1 prerequisite before any public compounding claim:**
EXP-G1 runs mock multi-domain enrichment sweeps (T=20 time steps) and measures marginal discovery yield vs t. If measured γ > 1.0 with p<0.05, the compounding claim is validated and the full `𝒮(n,t)` expression can be used in external communications. Until EXP-G1 runs, all language must use "projected" or "designed" qualification. See bridge_experiments_catalog_v7 §7.5 for full EXP-G1 spec.

**Interface:**

```python
# gae/discovery_engine.py (v8.0 — DESIGNED, NOT IMPLEMENTED)
@dataclass
class Discovery:
    """A single cross-domain pattern discovery."""
    entity_i_id: str        # entity from domain i
    domain_i: str
    entity_j_id: str        # entity from domain j (domain_j ≠ domain_i)
    domain_j: str
    score: float            # Eq. 8a output (0-1)
    relationship_type: str  # inferred relationship label
    confidence: float       # calibrated confidence
    sweep_number: int       # which enrichment sweep produced this
    timestamp: str


class DiscoveryEngine:
    """
    Level 3: Cross-domain pattern discovery.
    
    Prerequisites:
        - GraphAttentionBridge enrichment complete (Level 2)
        - EnrichedEmbeddingStore populated (Level 2 output)
        - Minimum 2 registered domains
        - δ_discovery threshold calibrated per domain pair
    
    v8.0 implementation target (after Level 2 validated in v7).
    GATE prerequisite: EXP-G1 must confirm γ > 1.0 before cross-domain
    compounding claims appear in any external document.
    """
    
    def discover(
        self,
        enriched_embeddings: dict,  # {domain_id: np.ndarray}
        delta: float = 0.7,         # discovery threshold
    ) -> List[Discovery]:
        """
        Run Eq. 8a-8c across all domain pairs.
        Returns discoveries sorted by score descending.
        """
        ...
    
    def measure_marginal_yield(
        self,
        sweep_history: List[List[Discovery]],
    ) -> dict:
        """
        Compute marginal yield per sweep for γ estimation.
        Returns: {sweep: n_new_discoveries, estimated_gamma: float}
        Used by EXP-G1.
        """
        ...
```

**Relationship between Level 2 and Level 3:**

```
Level 1 (v5.0):  Alert → f → ProfileScorer.score(f, c) → Action
                                    ↓ (write-back §11.4)
                              GraphSnapshot

Level 2 (v7):    GraphSnapshot → EmbeddingProvider → E
                 E_source + E_targets → CrossAttention → E_enriched
                 (LayerNorm applied — V1B)

Level 3 (v8):    E_enriched (multi-domain) → DiscoveryEngine → Discoveries
                 Discoveries → Tab 5 synthesis briefing
                 Discoveries → analyst review → confirmed patterns
```

---

### 11.4 Level 1 Data Preservation Hooks — What Every Domain Must Write

**Purpose:** Level 2 and Level 3 require specific data from Level 1's operational history. Every Level 1 domain implementation must write this data to the graph as part of its normal operation. These are not optional — without them, Level 2/3 cannot function when implemented in v7/v8.

**These are backward-compatible additions.** They extend what Level 1 writes; they do not change how Level 1 scores or learns. ProfileScorer.score() and ProfileScorer.update() are unchanged.

**Three write obligations:**

**Hook 1 — DecisionRecord (written on every scoring call):**

```python
@dataclass
class DecisionRecord:
    """
    Written to graph on every ProfileScorer.score() call.
    Required by Level 2 (entity embedding construction).
    Required by GATE-R (routing accuracy measurement).
    """
    entity_id: str              # the alert/entity being scored
    domain_id: str              # which domain ("soc", "s2p", etc.)
    category: str               # routing result from SituationAnalyzer
    category_index: int         # c — numeric category index
    action: str                 # selected action
    action_index: int           # a — numeric action index
    confidence: float           # max probability from softmax
    distances: List[float]      # raw distances to all action centroids (n_actions,)
    factor_vector: List[float]  # f — ALL factor values, named (n_factors,)
    factor_names: List[str]     # names corresponding to factor_vector positions
    timestamp: str              # ISO 8601
    decision_source: str        # "human" | "autonomous" | "assisted"
```

**Hook 2 — OutcomeRecord (written on every ProfileScorer.update() call):**

```python
@dataclass
class OutcomeRecord:
    """
    Written to graph on every ProfileScorer.update() call.
    Required by Level 2 (learning signal traceability).
    Required by compounding claim validation.
    """
    entity_id: str              # same entity as DecisionRecord
    domain_id: str
    decision_record_ref: str    # reference to the DecisionRecord this outcome resolves
    action: str                 # action that was taken
    correct: bool               # did the analyst confirm or override?
    mu_before: List[float]      # centroid snapshot before update (n_factors,)
    mu_after: List[float]       # centroid snapshot after update (n_factors,)
    delta_norm: float           # ||mu_after - mu_before||
    timestamp: str
```

**Hook 3 — ProfileSnapshot (written at checkpoint intervals):**

```python
@dataclass
class ProfileSnapshot:
    """
    Periodic snapshot of the full centroid array.
    Required by TD-033 (checkpoint/rollback).
    Required by Level 2 (embedding construction uses centroid positions).
    Required by EXP-G1 (γ measurement needs centroid state per sweep).
    """
    domain_id: str
    snapshot_id: str            # unique identifier for rollback reference
    mu: List[List[List[float]]] # full centroid array (n_categories × n_actions × n_factors)
    counts: List[List[int]]     # observation counts (n_categories × n_actions)
    decision_count: int         # total decisions at time of snapshot
    timestamp: str
    trigger: str                # "checkpoint_interval" | "operator_start" | "operator_end" | "manual"
```

**GraphSchema registration (written at domain initialization):**

```python
# Every domain DomainConfig must call this at startup:
def register_graph_schema(self) -> GraphSchema:
    """
    Returns the GraphSchema for this domain.
    Used by Level 2 GraphAttentionBridge domain registration.
    Called once at domain initialization.
    """
    return GraphSchema(
        domain_id=self.domain_id,
        entity_types=self.entity_types,        # from DomainConfig
        relationship_types=self.relationship_types,
        factor_names=self.factor_names,
        embedding_dim=128,                     # default; overridable
        n_categories=self.n_categories,
        n_actions=self.n_actions,
    )
```

**Compliance check:** The `ci-platform` StateManager will verify that DecisionRecord, OutcomeRecord, and ProfileSnapshot are being written correctly at domain startup. A domain that does not write Hook 1 and Hook 2 on every decision cycle will fail the PLAT-4 startup validation hook.

**Note on GATE-R:** Hook 1 (DecisionRecord) is also the data source for GATE-R (routing accuracy measurement). The `category` and `category_index` fields provide routing decisions. The ground-truth category labels are provided separately by analyst review. Without Hook 1 being written, GATE-R cannot run.

---

**PRODUCTION REQUIREMENT (V1B validated): Normalize embeddings after each enrichment sweep.** The residual enrichment equation E_i^{enriched} = E_i + Σ CrossAttention(G_i, G_j) without normalization produces embedding norms of 2.9 million× initial after 5 sweeps (~40× geometric growth per sweep). With LayerNorm or L2 re-normalization after each sweep, norms remain bounded. This follows standard transformer practice: residual connections paired with LayerNorm. The correct production form is: E_i^{enriched} = Normalize(E_i + Σ CrossAttention(G_i, G_j)).

**v8 note on Tier 5:** The original EXP-9 (discovery trigger validation) was designed for W-expansion discovery. With profile-based scoring, discovery is reframed as meta-graph reflection (§5, Layer 3). The discovery machinery still applies — it discovers cross-domain relationships. But the mechanism for incorporating discoveries into scoring is profile augmentation (add factor dimensions to μ), not W column expansion.

**Tier 6 — Artifact Evolution** (ci-platform repo, v6.0): Pipeline evolution via meta-prompt agents. *(See gae_design_v7 §11.)*

---

## 12-15. Contracts, API, Accumulation, Open-Source Strategy

*(Unchanged from v7 §12-§15. Demand-side contracts, public API pipeline, semantic accumulation channels, open-source strategy.)*

**One update to §13 (Public API):** The pipeline reference changes from:

```
v7: alert → factors → score_alert(f, W) → decision → outcome → W.update()
v8: alert → factors → SituationAnalyzer(f) → c → ProfileScorer.score(f, c) →
    decision → outcome → ProfileScorer.update(f, c, a, correct)
```

The additional step (category routing) is the bridge.

---

## 16. Evaluation, Schema, Judgment, Ablation Frameworks

*(§17-§20 in v7. Unchanged in structure. Updated notes:)*

**Evaluation (§17):** EvaluationScenario and run_evaluation() now use ProfileScorer instead of ScoringMatrix. The evaluation framework is scorer-agnostic — it calls score() and update() without knowing the kernel.

**Ablation (§20):** Updated baselines:
1. No learning (centroid_only — expected ~97.89% from EXP-C1)
2. No category routing (single profile set, shared across all categories)
3. Dot product kernel (expected ~61% from EXP-C1)
4. Random scoring (25% — unchanged)
5. **NEW:** Cold start (uniform initialization — expected ~90.7% at t=1000 from EXP-B1)

---

## 17. What's Built vs What's Next

### 17.1 GAE v0.1.0 (Tagged with v4.1)

*(Unchanged from v7 §21.1. 177 tests, 9 modules.)*

### 17.2 GAE v4.5 Preamble (DONE)

*(CalibrationProfile, per-factor decay. GAE-CAL-1/2 executed. 187 tests.)*

### 17.3 GAE v5.0 Prompts (REVISED for Profile-Based Scoring)

**Tier 0: Profile Integration (4 prompts — NEW)**

| Prompt | Creates | Gate |
|---|---|---|
| **GAE-PROF-1** | `gae/profile_scorer.py` — ProfileScorer class with pluggable kernels (L2, Mahalanobis, cosine, dot). KernelType enum. ScoringResult updated. init_from_config(). | Unit tests pass. L2 self-test > 95% on synthetic data. |
| **GAE-PROF-2** | Update `gae/scoring.py` — deprecate score_alert(), add forwarding to ProfileScorer. Update `gae/__init__.py` exports. | Existing tests pass (backward compat). New ProfileScorer in public API. |
| **GAE-PROF-3** | Update `gae/learning.py` — LearningState gets `profile_scorer` field. update() delegates to ProfileScorer.update() when available. Count-based decay. **TD-027 NOTE:** Include a `reset_to_config(profile_config)` method on ProfileScorer that reinitializes μ from DomainConfig profiles and zeros counts. This is the migration path from deprecated ScoringMatrix state — no W→μ migration, always reset to configured profiles. | Learning loop produces improving accuracy. 10-cycle compounding test passes. `reset_to_config()` correctly reinitializes μ from profile_config. |
| **GAE-PROF-4** | Wire through orchestrator. Category routing: SituationAnalyzer → c → μ[c,:,:]. DomainConfig.get_profile_config() hook. | End-to-end: alert → factors → category → score → learn. All tests pass. |

**Tier 1: Evaluation & API (from v7, updated)**

| Prompt | Creates | Gate |
|---|---|---|
| **GAE-EVAL-1** | `gae/evaluation.py` — uses ProfileScorer | Scenarios execute. Report populated. |
| **GAE-JUDG-1** | `gae/judgment.py` — profile convergence as judgment metric | judgment_score increases with decisions. |
| **GAE-ABL-1** | `gae/ablation.py` — 5 baselines including kernel comparison + cold start | Full system > all baselines. |
| **GAE-ENG-1** | API surface: ProfileScorer + ScoringResult + KernelType exported | Lint clean. |
| **GAE-ENG-2** | `examples/minimal_domain/` — uses ProfileScorer | Example runs full loop. |
| **GAE-DOC-1** | Users guide with kernel selection guidance | — |

**Total v5.0:** 10 prompts (4 profile + 6 evaluation/API).

### 17.4 GAE v5.5+ — Implementation Roadmap

**v5.5 (Tiers 4-5 embedding infrastructure + open-source release):**
- `gae/embeddings.py`: EmbeddingProvider protocol, PropertyEmbedding, TransformerEmbedding
- `gae/attention.py`: cross_attention() function (Eq. 6) — infrastructure only, not yet wired to GraphAttentionBridge
- `gae/discovery.py`: discovery extraction functions (Eq. 8a-8c) — infrastructure only
- `gae/events.py`: slow loop event triggers for periodic enrichment sweeps
- Ψ diagnostic functions fully activated
- Meta-graph reflection foundation (meta-loop triggers)
- Open-source release: Apache 2.0, public API stable, docs complete

**v7.0 (Level 2 — GraphAttentionBridge, designed in §11.2):**
- `gae/bridge.py`: GraphAttentionBridge class, domain registration, enrich(), sweep()
- `gae/graph_schema.py`: GraphSchema dataclass + DomainConfig integration
- Integration with v5.5 EmbeddingProvider and attention.py infrastructure
- LayerNorm applied after every sweep (V1B — mandatory)
- Prerequisite: all registered domains have Level 1 hooks (§11.4) writing DecisionRecord + OutcomeRecord + ProfileSnapshot
- Prerequisite: EXP-G1 completed (γ measured) before any cross-domain compounding claim is made externally

**v8.0 (Level 3 — Cross-Domain Discovery, designed in §11.3):**
- `gae/discovery_engine.py`: DiscoveryEngine class, discover(), measure_marginal_yield()
- Integration with Level 2 EnrichedEmbeddingStore
- δ_discovery calibration tooling
- Prerequisite: Level 2 validated in v7
- Prerequisite: EXP-G1 γ > 1.0 confirmed (compounding claim must be validated before Level 3 ships)

---

## 18. Repository Structure (Updated)

```
graph-attention-engine/
├── gae/
│   ├── __init__.py              # Public API: ProfileScorer, ScoringResult, KernelType,
│   │                            #   CalibrationProfile, LearningState, FactorComputer, etc.
│   ├── profile_scorer.py        # NEW: ProfileScorer (Eq. 4-final, pluggable kernels)
│   ├── calibration.py           # UPDATED: eta, eta_neg, count_decay in extensions
│   ├── scoring.py               # DEPRECATED: score_alert() forwards to ProfileScorer
│   ├── learning.py              # UPDATED: delegates to ProfileScorer.update() when available
│   ├── factors.py               # Unchanged
│   ├── convergence.py           # Unchanged
│   ├── contracts.py             # Unchanged
│   ├── primitives.py            # Unchanged
│   ├── evaluation.py            # UPDATED: uses ProfileScorer
│   ├── judgment.py              # UPDATED: profile convergence metrics
│   ├── ablation.py              # UPDATED: 5 baselines including kernel comparison
│   ├── embeddings.py            # Tier 4 (v5.5) — EmbeddingProvider protocol
│   ├── attention.py             # Tier 5 infrastructure (v5.5) — cross_attention() Eq. 6
│   ├── discovery.py             # Tier 5 infrastructure (v5.5) — Eq. 8a-8c functions
│   ├── graph_schema.py          # NEW (v7): GraphSchema dataclass — Level 1 hook contract
│   ├── bridge.py                # NEW (v7): GraphAttentionBridge — Level 2 (DESIGNED §11.2)
│   ├── discovery_engine.py      # NEW (v8): DiscoveryEngine — Level 3 (DESIGNED §11.3)
│   ├── events.py                # Unchanged
│   ├── store.py                 # UPDATED: ProfileScorer state persistence
│   └── types.py                 # Unchanged
├── tests/
│   ├── test_profile_scorer.py   # NEW: L2, Mahalanobis, cosine, dot tests
│   ├── test_calibration.py      # UPDATED
│   ├── test_scoring.py          # Unchanged (backward compat)
│   ├── test_learning.py         # UPDATED
│   ├── test_evaluation.py       # UPDATED
│   └── ...
├── examples/
│   └── minimal_domain/          # UPDATED: uses ProfileScorer
└── docs/
    ├── equations.md             # UPDATED: Eq. 4-final, 4b-final
    ├── users_guide.md           # NEW: kernel selection, profile config
    └── EXPORTS.md               # UPDATED
```

---

## 19. Equation-to-Code Traceability Matrix (Updated)

| Equation | Source | File | Function | Version | Status |
|---|---|---|---|---|---|
| **Eq. 4-final** | Experiments | `gae/profile_scorer.py` | `ProfileScorer.score()` | **v5.0** | **NEW** |
| **Eq. 4b-final** | Experiments | `gae/profile_scorer.py` | `ProfileScorer.update()` | **v5.0** | **NEW** |
| **Eq. 4-synthesis** *(PROPOSAL)* | intelligence_layer_design_v2 | `gae/profile_scorer.py` | `ProfileScorer.score()` extended with `sigma` param | **v5.5+ (gated by GATE-M/GATE-V)** | **PROPOSAL** — λ=0 reduces to Eq. 4-final exactly |
| Eq. 4 (published) | Blog §3 | `gae/scoring.py` | `score_alert()` | v4.1 | **DEPRECATED** |
| Eq. 4b (published) | Blog §3 | `gae/learning.py` | `LearningState.update()` | v4.1 | **DEPRECATED** |
| Eq. 4c (decay) | Blog §3 | `gae/learning.py` | `LearningState.apply_decay()` | v4.1 | Preserved (v5.5) |
| Eq. 5 (embeddings) | Blog §4 | `gae/embeddings.py` | `EmbeddingProvider.embed_graph()` | v5.5 | Infrastructure |
| Eq. 6 (cross-attention) | Blog §4, §11.2 | `gae/attention.py` (infra), `gae/bridge.py` (wired) | `cross_attention()` / `GraphAttentionBridge.enrich()` | v5.5 infra / **v7 wired** | Designed — LayerNorm mandatory (V1B) |
| Eq. 8a-8c (discovery) | Blog §4, §11.3 | `gae/discovery.py` (infra), `gae/discovery_engine.py` (wired) | `extract_discoveries()` / `DiscoveryEngine.discover()` | v5.5 infra / **v8 wired** | Designed — EXP-G1 prerequisite |
| Eq. 9 (multi-domain) | Blog §5, §11.3 | `gae/discovery_engine.py` | `DiscoveryEngine.discover()` multi-domain | **v8** | Designed — requires Level 2 validated |
| γ-scaling `𝒮(n,t) ~ O(n^{2.11}·t^γ)` | §11.3 | `gae/discovery_engine.py` | `measure_marginal_yield()` | **v8** | **γ≈1.5 ESTIMATED — EXP-G1 required before public claim** |

---

## 20. Design Decisions Summary (Expanded)

### 20.1 Decisions from v7 (Unchanged)

| Decision | Rationale | Version |
|---|---|---|
| NumPy-only, zero external deps | 24 multiply-adds don't need PyTorch | v4.1 |
| Three-repo architecture | Structural enforcement of P12 | v4.1 |
| FactorComputer Protocol in GAE | Protocol is abstract, orchestrator needs Neo4j | v4.1 |
| f(t) stored in Decision nodes (R4) | Graph changes between decision and outcome | v4.1 |
| CalibrationProfile replaces constants | Multi-domain: SOC 20:1 ≠ S2P 5:1 | v4.5 |
| Three-layer decay | Single source of truth for factor metadata | v4.5 |

### 20.2 Decisions from Experiments (NEW in v8)

| Decision | Evidence | Impact |
|---|---|---|
| **L2 distance as default kernel** | EXP-C1: L2=97.89%, dot=61.0%, cosine=96.42% | Replaces dot product in Eq. 4 |
| **Pluggable kernels (not hardcoded)** | EXP-E1: L2 wins 2/3, Mahalanobis wins mixed-scale | GAE supports 4 kernels |
| **Profile centroids as first-class** | EXP-C1: 97.89% with zero learning | Profiles ARE the model (P13) |
| **Warm start essential at scale** | EXP-E2: cold degrades 89.9%→72.7% | DomainConfig MUST provide profiles (P15) |
| **G (gating matrix) eliminated** | EXP-A: +0.01pp (negligible) | No GatingMatrix in GAE |
| **Ψ (warmup) eliminated as control** | EXP-B1: count-based decay suffices | Ψ repositioned as diagnostic |
| **Dot product warned** | EXP-C1: 61% on [0,1] data, EXP-E1: 90.8% only when normalized | Warning in KernelType.DOT |
| **Count-based decay replaces A1** | EXP-B1: natural phase behavior from 1/(1+n·d) | Simpler, no confidence computation needed |
| **Factors independently informative** | EXP-D2: 0 significant interactions | No factor augmentation needed for SOC |
| **Cross-category transfer marginal** | EXP-D1: config wins by 2-14pp | Transfer is v6.0+ nice-to-have |
| **Architecture scales to 20×10×20** | EXP-E2: 99.9% oracle, 200 centroids | No architectural limits at tested scales |
| **Cold start viable at small scale** | EXP-B1: 90.7% in 1000 decisions | Enables domains without expert profiles |
| **τ_modifier permanently rejected** | S-series: ECE degradation +0.138 at τ_mod≠1.0; V3B: τ=0.1 optimal | Temperature FIXED at τ=0.1. No τ_modifier field in CalibrationProfile. No synthesis condition changes τ. |
| **update() has no synthesis parameter (architectural firewall)** | EXP-S3: Loop 2 firewall Frobenius 0.0028; OP series conceptual design | μ updated by experience only. σ flows through score() only. The boundary is permanently enforced. |
| **Operative window λ∈[0.5, 0.6] for synthesis bias** | EXP-OP1-FINAL (Bonferroni); EXP-OP2 (asymmetry finding) | λ=0.5 safe default. λ>0.6: harmful operator damage scales 5:1 vs benefit. Never deploy λ>0.6 without operator quality controls. |
| **Centroid checkpoint/rollback required (TD-033)** | EXP-OP2: C-exp −0.0124 post-expiry, 35% never-recover | ProfileScorer needs checkpoint() + rollback(). Checkpoint every 50 decisions. TTL alone insufficient. |

### 20.3 Decisions from Level 2/3 Design (NEW in v8.3)

| Decision | Rationale | Impact |
|---|---|---|
| **Level 2/3 design lives in gae_design, not a separate doc** | Single authoritative source for interface contracts. Other documents (math_synopsis, architecture_philosophy, soc_copilot_design) reference this section. | Eliminates design fragmentation. Updates to Level 2/3 design are one change, not four. |
| **Level 1 data preservation hooks are backward-compatible** | DecisionRecord, OutcomeRecord, ProfileSnapshot are write-only additions. They do not change ProfileScorer.score() or update() behavior. | Level 1 domains can add hooks incrementally without breaking existing tests or API contracts. |
| **LayerNorm is a hard production requirement for Level 2 (V1B validated)** | Without LayerNorm, enrichment norms grow 2.9M× after 5 sweeps. This was confirmed in V1B. It is not a configurable option. | GraphAttentionBridge.enrich() always applies LayerNorm. No flag to disable it. |
| **γ requires EXP-G1 before any public compounding claim** | γ≈1.5 is an estimate from the scaling expression derivation. It has not been measured. If γ < 1.0, the compounding claim is wrong. If γ ≈ 1.0, the claim is overstated. | All external documents use "projected" language until EXP-G1 completes. The math expression `𝒮(n,t) ~ O(n^{2.11}·t^γ)` may be stated with γ as "estimated ≈1.5, EXP-G1 pending." |
| **Level 3 is a v8 target, not v7** | Level 3 requires validated Level 2 embeddings as input. Building Level 3 before Level 2 is proven would be building on an unvalidated substrate. v7 validates Level 2; v8 builds Level 3 on the validated foundation. | Level 3 claims (cross-domain discovery) are v8 target language in all documents. |
| **GraphAttentionBridge does NOT update μ** | The Loop 2 / Loop 4 firewall (EXP-S3, Frobenius 0.0028) is a permanent architectural constraint. Level 2 enrichment is a read-only enrichment path from the perspective of centroid state. | GraphAttentionBridge has no reference to ProfileScorer.update(). The firewall extends to Level 2. |

---

### A.1 Experiment Parameters

| Experiment | Alerts | Seeds | Decisions | Runtime |
|---|---|---|---|---|
| EXP-5 (oracle) | 1000 | 10 | 1000 | 45m local |
| EXP-A (capacity) | 1000 | 10 | 1000 | ~10m |
| EXP-C1 (centroid) | 10000 | 10 | 0 (oracle only) | <30s |
| EXP-B1 (profiles) | 1000 | 10×57 | 1000 | ~9m |
| EXP-D1 (transfer) | 500+200 | 10 | 700 | ~5m |
| EXP-D2 (interactions) | 10000 | 10 | 0 (MI analysis) | <2m |
| EXP-E1 (kernels) | 10000 | 10 | 1000 (Phase 2) | ~7m |
| EXP-E2 (scale) | 5000-10000 | 10 | 1000-10000 | ~8m |

### A.2 Chart Inventory (for Blog and Outreach)

| Chart | File | Best For |
|---|---|---|
| Waterfall 25%→98% | expC1_comparison_waterfall.png | Math blog, demo blurb |
| L2 vs dot vs cosine | expC1_method_comparison.png | Math blog §3 |
| Warm vs cold vs centroid | expB1_warm_vs_cold_vs_centroid.png | Compounding blog |
| Noise robustness | expB1_noise_robustness.png | Production viability argument |
| Kernel × distribution | expE1_kernel_x_distribution.png | Math blog, GAE docs |
| Dot vs L2 (normalization) | expE1_dot_vs_l2.png | GAE kernel guidance |
| Oracle scaling | expE2_oracle_scaling.png | "Transformers for graphs" claim |
| Scaling trend | expE2_scaling_trend.png | GAE docs |
| Transfer matrix | expD1_transfer_matrix.png | Meta-graph discussion |
| Single-factor MI | expD2_single_mi.png | Factor design validation |

All charts in `cross-graph-experiments/paper_figures/` as both PNG and PDF.

### A.3 Key Experimental Files

| File | Repo | Purpose |
|---|---|---|
| `src/models/profile_scorer.py` | cross-graph-experiments | Reference implementation (port to GAE) |
| `experiments/expB1_profile_scoring/results/summary.json` | cross-graph-experiments | 98.2% validation |
| `experiments/expE1_kernel_generalization/results/summary.json` | cross-graph-experiments | Kernel comparison data |
| `experiments/expE2_scale_test/results/summary.json` | cross-graph-experiments | Scale test data |
| `configs/default.yaml` | cross-graph-experiments | Profile configurations |
| `bridge_experiments_design_v3.md` | cross-graph-experiments | Full bridge experimental record |
| `cross_graph_attention_v3.md` | project docs | Math paper with validation experiments (V1A–V3B) |
| `experiments/validation/` | cross-graph-experiments | V1A–V3B validation experiment data |

---

*Graph Attention Engine — Design & Architecture v8.3 | March 7, 2026*
*Merged: gae_design_v7 + bridge experiment findings + ontological architecture + validation experiments + OP series findings + Level 2/3 design specification.*
*Scoring architecture SETTLED: Eq. 4-final with L2 distance kernel (98.2% warm — synthetic, GT profiles).*
*ProfileScorer replaces ScoringMatrix. Pluggable kernels (L2 default). τ=0.1 (calibrated, V3B). τ_modifier REJECTED.*
*Profiles are first-class computational constituents — compiled ontology (P13).*
*Ontology → meta-graph → profiles → distance-kernel attention.*
*25 experiments (9 bridge + 5 validation + 11 OP/synthesis). GATE-OP PASSED (p=0.0008 at λ=0.5).*
*G eliminated. Ψ eliminated. Dot product warned. τ_modifier rejected.*
*Production requirements: centroid clipping (V2), enrichment LayerNorm (V1B), τ=0.1 (V3B), centroid RESET before L2 deployment (TD-027), checkpoint infrastructure (TD-033), τ recalibration gate (TD-034).*
*L2 centroid beats XGBoost, RF, LogReg on same data (V3A — synthetic). Online learning 5pp advantage.*
*Scales to 20×10×20 (99.9%). Warm start essential at scale (P15).*
*Scaling exponent: b=2.11, CI [2.09, 2.14] — super-quadratic confirmed by simulation (V1A). t^γ compounding is projected (γ≈1.5 estimated — EXP-G1 pending).*
*update() has NO synthesis parameter — Loop 2/Loop 4 firewall is permanent. GraphAttentionBridge also does NOT update μ — firewall extends to Level 2.*
*Eq. 4-synthesis is a PROPOSAL (gated by GATE-M/GATE-V). λ=0 gives exact Eq. 4-final behavior.*
*Level 2 (GraphAttentionBridge): designed in §11.2, v7.0 target. Level 3 (Cross-Domain Discovery): designed in §11.3, v8.0 target.*
*γ in 𝒮(n,t) ~ O(n^{2.11}·t^γ): estimated ≈1.5. EXP-G1 required before any public compounding claim.*
*Every Level 1 domain must write DecisionRecord + OutcomeRecord + ProfileSnapshot (§11.4) for Level 2/3 to function.*
*v5.0: 10 prompts (4 profile + 6 eval/API). v5.5: Tiers 4-5 embedding + attention infrastructure. v7: GraphAttentionBridge. v8: DiscoveryEngine.*
*Every equation → one function → one file → one prompt.*
*"The profiles ARE the institutional judgment. Not weights — knowledge you can read, audit, and trust."*
