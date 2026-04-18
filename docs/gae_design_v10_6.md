# Graph Attention Engine — Design & Architecture v10.4

**Date:** April 8, 2026
**Version:** 10.6 (v10.5 + gae/experiments/ module spec §10.15: reproducibility scripts + domain examples + oracle separation demos; Block 3.2 concrete; 527 tests current)
**Authority:** claims_registry_v10.0 · MAP v5.20
**Status:** Phase 0 ✅ Phase 1 ✅ Phase 2 ✅ Phase 3 Priority 1 ✅. Loom demo v1 unblocked. DiagonalKernel validated (+13pp SOC, +7pp S2P). W2 flywheel validated (CLAIM-W2: +10.13pp, p=0.0002). Flywheel Health Monitor validated (CLAIM-OLS-01: 0% miss, p90≥50d). ReferralRules architecture validated (72.7% DR, 12% FPR). ShrinkageKernel deprioritized to v7.0.
**Repository:** graph-attention-engine (standalone, numpy-only, Apache 2.0)
**Scope:**
- v4.1 (Tiers 1-3 foundation)
- v4.5 (CalibrationProfile, per-factor decay)
- **v5.0 COMPLETE** (ProfileScorer + OracleProvider + Evaluation + Judgment + Ablation + Users Guide)
- **v6.0 COMPLETE** (Kernel framework + CovarianceEstimator + KernelSelector + asymmetric η + AMBER auto-pause + ReferralEngine + OLSMonitor + enrichment_advisor — 517 tests)
  - **v0.7.18-v0.7.20**: KERNELSEL-001 tiebreaker + raw_weights fix + Block 9.1-9.5 (CLAIM-66-70) — 527 tests
- v6.5 (GainScheduler, Fisher calendar, enforcement mode, OverrideDetector activation)
- v7.0 (Level 2: GraphAttentionBridge + ShrinkageKernel investigation)
- v8.0 (Level 3: Cross-Domain Discovery)

**Companion repos:**
- ci-platform (93 tests — connectors, onboarding, deployment qualification, entity resolution, PII redaction, SAML — Apache 2.0)
- soc-copilot (288 tests — SOC domain expertise, frozen ROI, hooks, shadow mode, PatternHistoryFactorComputer — proprietary)
- cross-graph-experiments (~130 experiments: bridge, validation, OP/synthesis, persona sweeps, factorial kernel studies, Phase 1 closure)

**Supersedes:** gae_design_v9, v9.1 + all prior versions.

---

> **Changes from v8.3 → v9 (March 9, 2026 — v5.0 complete):**
>
> (1) **§10 completely rewritten.** All GAE v5.0 Phase 6 modules now exist as live code with tests. Interfaces reflect actual implemented APIs, not design specs. Replaces the "Tier 0–1 prompt descriptions" with the actual shipped API contract.
> (2) **New §10.6: OracleProvider.** oracle.py is the ground-truth interface for the evaluation pipeline. Three implementations: OracleProvider protocol, GTAlignedOracle, BernoulliOracle. This module was not in v8.3.
> (3) **New §10.7: Evaluation.** evaluation.py: EvaluationScenario, run_evaluation(), EvaluationReport, compute_ece(). Actual field names confirmed from implementation.
> (4) **New §10.8: Judgment.** judgment.py: compute_judgment(), JudgmentResult, three confidence tiers (high ≥0.80, medium ≥0.50, discovery). Signed function signature confirmed.
> (5) **New §10.9: Ablation.** ablation.py: run_ablation(), AblationReport, factor importance ranking. accuracy_drop > 0 means factor helps.
> (6) **§10.4 ScoringResult field correction.** Field is `.probabilities` not `.scores`. Confirmed from implementation and users_guide.md.
> (7) **New §15: Open-Source Strategy.** First-class section on what it takes to make GAE a successful open-source project — not just "Apache 2.0 license" but README, contributing guide, PyPI, CI, issue templates, community, and the "why open-source" strategic argument.
> (8) **§17 (What's Built vs Next) substantially rewritten.** v5.0 marked COMPLETE with test counts. v5.5 next steps expanded. Phase 6 completion status added.
> (9) **§18 repo structure updated.** All v5.0 files marked ✅. v5.5 and v7/v8 targets clarified.
> (10) **Test count updated.** 187 (v4.1) → 211 (v5.0-alpha: Phase 1) → 235 (v5.0 Phase 6 GAE) → **243 (v5.0 TAGGED, post-SOC fixes).**
> (11) **Realistic accuracy numbers integrated.** math_synopsis_v6 50-seed validated numbers now present alongside synthetic numbers. Both are correct and serve different purposes.
>
> **Changes from v10.1 → v10.2 (March 25, 2026 — Phase 1 closure):**
>
> (29) **Header updated.** 478→517 tests. ci-platform 73→93. soc-copilot 252→288. Experiments ~100→~130. v6.0 marked COMPLETE.
>
> (30) **§9.5 DiagonalKernel gradient corrected (CRITICAL).** GAE-GRADIENT-001 bug fixed in v0.7.7: DiagonalKernel gradient was `W*(f-mu)`. Correct formula is `W/W.max()*(f-mu)`. The unnormalized gradient caused W to dominate in magnitude rather than direction, corrupting the learning signal under high-W factors. This was the root cause of the V-ENRICHMENT-NEGATIVE false UNSAFE result at GAE <0.7.7. §10.8 DiagonalKernel.compute_gradient() updated to match.
>
> (31) **§16 Production Constraints: two additions.** DiagonalKernel gradient constraint (W/W.max()*(f-mu) only) and η_neg guard (CalibrationProfile raises ValueError on η_neg ≥ 1.0, GAE 0.7.8 fix).
>
> (32) **§10.1 Module Overview updated.** 478→517 tests. convergence.py annotated: OLSMonitor (CUSUM on OLS, h=5.0 OLS scale, plateau-snapshot baseline, CLAIM-OLS-01). enrichment_advisor.py added (enrichment recommendation engine, validated on 5 profiles). VarQMonitor noted (logged only — Var(q) gating: PERMANENT HARD STOP, Bernoulli mixture theorem). Test file additions listed.
>
> (33) **§14.2 Design Decisions: Phase 1 entries added.** DiagonalKernel gradient fix, η_neg guard, W2 flywheel (CLAIM-W2), Flywheel Health Monitor (CLAIM-OLS-01), Var(q) hard stop, convergence calendar (CLAIM-CONV-01), two-tier poisoning (CLAIM-SK-01/LP-01), enrichment safety (CLAIM-65).
>
> (34) **§17.1a v6.0 marked COMPLETE.** Test progression extended to 517. Phase 1 session additions documented.
>
> (35) **§17.2 v6.0 open-source release items updated.** OLSMonitor, enrichment_advisor, VarQMonitor added.
>
> (36) **§18 Repo structure updated.** enrichment_advisor.py added. convergence.py annotation updated with OLSMonitor. VarQMonitor added to convergence.py. Test file additions listed.
>
> (37) **§15.2 test count updated.** 478→517.

> **Changes from v9.1 → v10 (March 21, 2026 — kernel architecture settled):**
>
> (15) **NEW §9.6: DiagonalKernel equation.** P(a|f,c) = softmax(−(f−μ)ᵀW(f−μ)/τ) where W=diag(1/σ²). Validated: +13.2pp SOC, +6.8pp S2P on heterogeneous noise. Corr(noise_ratio, advantage)=0.990 across 4 healthcare personas.
> (16) **§9.2 KernelType updated.** DIAGONAL added. MAHALANOBIS/COSINE deprecated. Kernel selection rule: noise_ratio > 1.5 → diagonal, else L2.
> (17) **§9.4 ProfileScorer interface expanded.** New parameters: `kernel` (ScoringKernel instance), `factor_mask` (deprecated by DiagonalKernel), `eta_override` (asymmetric η, P0 fix), `auto_pause_on_amber` (conservation AMBER → freeze learning).
> (18) **§9.5 Learning equation updated.** Asymmetric η: η_confirm=0.05, η_override=0.01. Prevents 13-27pp centroid degradation from low-quality overrides. Validated across 24 personas.
> (19) **NEW §10.9: gae/kernels.py.** ScoringKernel protocol, L2Kernel, DiagonalKernel. 28 tests.
> (20) **NEW §10.10: gae/covariance.py.** CovarianceEstimator with Ledoit-Wolf shrinkage + exponential decay. COLLECTS data at v6.0, does NOT affect scoring. 23 tests.
> (21) **NEW §10.11: gae/kernel_selector.py.** KernelSelector: Phase 2 rule-based + Phase 3 rolling 100-window comparison + Phase 4 empirical lock. 46 tests.
> (22) **NEW §10.12: gae/calibration.py additions.** compute_factor_mask(), mask_to_array(), compute_eta_override(), derive_theta_min(). Factor mask deprecated but present.
> (23) **§10.1 module overview updated.** Four new modules. 478 tests. v0.7.0 target.
> (24) **§4 experiment counts updated.** 25→~100 experiments. New series: persona sweeps (24 personas, 72 harness runs), V-MV-KERNEL factorial (216+144 cells), kernel deliverables.
> (25) **§14.2 new experiment-driven decisions.** DiagonalKernel default, factor mask deprecated, ShrinkageKernel deprioritized, asymmetric η, AMBER auto-pause, noise ceiling revised.
> (26) **§16 production constraints updated.** η_override=0.01, θ_min=0.467, AMBER auto-pause.
> (27) **§17 test progression updated.** 246→437 (+191). v6.0 section added.
> (28) **§18 repo structure updated.** Four new files: kernels.py, covariance.py, kernel_selector.py, test files.
>
> **v9.1 additions (March 12, 2026 — post-tag WIRING-1):**
>
> (12) **CentroidUpdate dataclass added.** `ProfileScorer.update()` now returns `Optional[CentroidUpdate]` with fields `delta_norm`, `category_index`, `action_index`, `before`, `after`. Enables the `centroid_delta_norm` wiring in SOC triage endpoint and Tab-3 centroid delta display.
> (13) **freeze()/unfreeze() added.** `ProfileScorer.freeze()` halts centroid updates (returns `None` from `update()`). `ProfileScorer.unfreeze()` resumes. `LearningState` exposes frozen status. Test count: 243 → **246**.
> (14) **Two-level institutional judgment framing added (§5, §9).** ProfileScorer is explicitly named as **Level 1 — Decision Intelligence** (what to decide; centroids μ; slow-moving, months). AgentEvolver (in SOC copilot) is **Level 2 — Deployment Intelligence** (how to operate; variant selection; moderate, weeks). GAE owns Level 1 only. Level 2 is implemented in the domain copilot layer.

---

## 1. Requirements: Why This Document Exists

### 1.1 The Problem

The SOC Copilot's published math blog describes a precise computational system: cross-graph attention (Eq. 6), weight learning (Eq. 4b), entity embeddings (Eq. 5), discovery extraction (Eq. 8a-8c), and scoring matrices (Eq. 4). The demo codebase implements a subset. The GAE is the computational substrate that implements the math — and must eventually stand alone as an open-source library that others can use to build their own compounding intelligence systems.

### 1.2 Requirements

**R1 — Equation traceability.** Every equation maps to exactly one function in exactly one file.

**R2 — Causal closure.** Outputs of one cycle become inputs to the next. Six causal connectors (§7) wire computation into closed loops.

**R3 — Event-driven architecture.** Three causal loops at different timescales trigger each other via events.

**R4 — Factor vector preservation.** Decision nodes store f(t) from the ORIGINAL decision. Recomputed f at outcome time would corrupt the learning signal.

**R5 — Dynamic dimensionality.** Profile centroids μ accommodate new factor dimensions via profile augmentation (adding dimensions to μ), not W expansion. EXP-D2 found zero significant factor interactions in SOC data, so augmentation is unlikely in practice — R5 is preserved for generality.

**R6 — Technology selection justification.** NumPy for v5.0 (24 multiply-adds per scoring). Backend-swappable for future scale.

**R7 — Single-concern prompt decomposition.** Build decomposed into single-file, tightly-scoped prompts. Each prompt is one concern.

**R8 — Accumulation channel completeness.** Every graph mutation produces a write-back read by downstream computation.

**R9 — Open-source API stability.** Public classes versioned and backward-compatible. Engine never imports from product code (P12). API surface is the contract with the open-source community — changes are breaking.

**R10 (NEW v9) — Self-contained documentation.** The library must be fully understandable without reading the SOC copilot or the math blog. A new engineer working on a procurement domain must be able to use GAE from README + users_guide.md alone.

---

## 2. Technology Analysis

### 2.1 What Transformer Attention Is

Core insight preserved: HuggingFace BertSelfAttention is 5 lines of math + 245 lines of infrastructure. We take the math, build our own infrastructure.

### 2.2 What Changed: The Kernel

The v7 analysis mapped Eq. 4 to transformer dot-product attention. **The experiments proved this mapping is wrong for our data.**

| Component | v7 Mapping | v9 Mapping (Validated) |
|---|---|---|
| Queries (Q) | f · W (linear projection) | f (factor vector directly) |
| Keys (K) | W rows | μ[c,a,:] (profile centroids) |
| Similarity | Q · K^T (dot product) | **−‖f − μ[c,a,:]‖² (L2 distance)** |
| Routing | Shared W across categories | **μ[c,:,:] per category (MoE)** |
| Learning | Hebbian W update | **Centroid pull/push** |

**Why this matters for open-source positioning:** Most graph ML libraries use dot-product attention (GNNs, transformers). GAE uses distance-kernel attention. This is a genuine technical differentiation — not just a parameter choice. EXP-C1 (36.89pp gap between L2 and dot product on the same data) makes this claim defensible to a technical audience.

### 2.3 Transformer Parallel (the "Rosetta Stone")

```
Transformer:  tokens attend to tokens  via dot-product similarity
GAE:          graph domains attend to  via distance-kernel similarity
              profile prototypes       to factor vectors
              
MoE parallel: category c = expert head selector (externally defined)
              μ[c,:,:] = expert parameters
              f = the query routed to expert c
```

This parallel makes GAE legible to engineers who know transformers. Use it in README, talks, and the math blog update.

---

## 3. Architectural Principles

*P1–P12 unchanged from v7. P13–P16 added in v8. Reproduced for completeness.*

| # | Principle | Implication |
|---|---|---|
| P1 | Exact equation traceability | One equation → one function → one file |
| P2 | Causal closure | Every output is someone's input |
| P3 | Event-driven decoupling | Loops communicate via events, not direct calls |
| P4 | Factor preservation | f(t) stored at decision time, never recomputed |
| P5 | Domain-agnostic engine | SOC logic cannot live in GAE |
| P6 | Technology justification | Every technology choice is documented |
| P7 | Prompt decomposition | One concern per Claude Code prompt |
| P8 | Accumulation completeness | Every graph mutation has a write-back |
| P9 | API stability | Breaking changes require major version bump |
| P10 | Separation of scoring and learning | score() never updates μ; update() never scores |
| P11 | CalibrationProfile as domain contract | Hyperparameters are domain choices, not engine choices |
| P12 | One-way dependency | GAE → zero. Domain copilot → GAE. Never reverse. |
| P13 | Profiles as first-class constituents | μ is the model, not W |
| P14 | Experimental evidence for architecture | No architecture decision without experimental basis |
| P15 | Warm start essential | DomainConfig MUST provide initial profiles |
| P16 | Synthesis firewall | update() has NO synthesis parameter — by design, forever |

---

## 4. Experimental Foundation

### 4.1 Experiment Summary (~175 total)

| Group | Count | Key Findings |
|---|---|---|
| Bridge (EXP-5, A, A2, B1, C1, D1, D2, E1, E2) | 9 | G falsified; dot product root cause; L2=97.89%; profiling approach validated |
| Validation (V1A, V1B, V2, V3A, V3B) | 5 | b=2.11; LayerNorm required; clipping required; XGBoost beaten; τ=0.1 |
| OP/Synthesis (S1-S3, OP1-OP4, others) | 11 | GATE-OP PASSED; τ_mod rejected; Loop 2/4 firewall validated; operative window λ∈[0.5, 0.6] |
| Block 5A harnesses (TD-034, PROD-5, B-A) | 3 | τ=0.08 optimal on SOC; ε=0.10 for convergence; +10.1pp Level 2 acceptance |
| Block 5B proxy (9 LLM-judge personas) | 9 | Calibration wrong for 8/9; 13-27pp degradation from overrides → P0 fix |
| Phase 1 sweeps (1C/1A/1B/2D/1D/2G) | 72 runs | Product boundaries: σ≤0.157, V≥30, η_override=0.01, B-A≥8, enrichment +7-13pp |
| Priority 1 validation (V-B3/V-B1/V-CL-RECOVER) | 9 | AMBER zone safe; recovery conservative; ceiling is three-variable |
| V-HC-CONFIG (mask + diagonal) | 5 | Mask halves degradation but insufficient; DiagonalKernel rescues healthcare (+3.7pp) |
| V-MV-KERNEL factorial | 360 cells | Uniform: all kernels identical. Heterogeneous: Diagonal +13.2pp SOC, +6.8pp S2P |
| Kernel deliverables + selector fixes | ~20 | Explanation A confirmed; shrinkage adds nothing; selector needs rolling window, ~250 decisions |
| SVM series (SVM-001 through 004b, 005) | 8 | CL-ECON-MEASURED 30.85 min/alert. CLAIM-62 +42.69pp. CLAIM-64 r=0.9669. FX-1 coverage. |
| V-CGA-FROZEN (Batch G) + V-SHADOW-SYNTHETIC | ~15 | CLAIM-59 54.4% faster convergence p<0.0001. Third compounding pathway confirmed. |
| Block 9 D-series + V-GATE-STABILITY | ~8 | CLAIM-66–70: per-analyst η, η cap (UNCONDITIONAL F=8.14), spike detector, category freeze, cap. |
| **Oracle separation (γ theorem, April 2026)** | **7** | **Batch F META-4 RETIRED (identifiability). Oracle separation Exp A/v6/v8/v11/v2/v3/Final. Binary prediction confirmed (ε=0.05: γ=0.71 ε=0.20: γ=1.03>1). Theorem established (4 LLMs). N_half gap documented. EXP-G1 design finalized.** |

### 4.2 Key Validated Numbers

| Metric | Value | Condition | Source |
|---|---|---|---|
| Zero-learning accuracy | 97.89% | Synthetic centroidal data, GT profiles | EXP-C1 |
| With-learning accuracy | 98.2% | Synthetic, warm-start, noise-free oracle | EXP-B1 |
| **Realistic static accuracy** | **71.7%** [71.4%, 71.9%] | **50 seeds, realistic distributions, combined categories** | **math_synopsis_v6** |
| **Realistic at decision 1,000** | **78.9%** [78.1%, 79.6%] | **50 seeds, realistic distributions** | **math_synopsis_v6** |
| **Auto-approve accuracy (≥0.90)** | **90.7%** [90.1%, 91.2%] | **50 seeds, realistic distributions** | **math_synopsis_v6** |
| Auto-approve coverage | 11.5% ± 0.70% | Combined realistic, 50 seeds | math_synopsis_v6 |
| Calibration ECE | 0.036 at τ=0.1 | Synthetic | V3B |
| L2 vs XGBoost | 94.78% vs 92.24% | Synthetic held-out data | V3A |
| Domain scaling exponent | b=2.11, CI [2.09, 2.14] | Simulation, R²=0.9999 | V1A |
| Max scale tested | 20×10×20 (4,000 params) | Synthetic, GT profiles | EXP-E2 |
| **DiagonalKernel lift (SOC)** | **+13.2pp** | **Heterogeneous noise (ratio 0.5-2.0×)** | **V-MV-KERNEL factorial** |
| **DiagonalKernel lift (S2P)** | **+6.8pp** | **Heterogeneous noise (ratio 0.6-1.8×)** | **V-MV-KERNEL factorial** |
| **DiagonalKernel at σ=0.22** | **+20-22pp over L2** | **Healthcare-like noise profile** | **V-HC-CONFIG diagonal** |
| **Healthcare rescue** | **+3.7pp (Diagonal) vs +0.3pp (L2)** | **σ_mean=0.22, 4 analysts, 15 seeds** | **V-HC-CONFIG** |
| **Corr(noise_ratio, advantage)** | **0.990** | **4 healthcare personas, ratio 1.6-4.6×** | **Selector fixes Ask 3** |
| **Shrinkage vs Diagonal gap** | **<1pp (both domains)** | **Explanation A confirmed** | **Kernel deliverables D2/D3** |

| **Third compounding pathway** | **54.4% faster convergence** | **p<0.0001, 26/30 seeds, 90-day simulation** | **CLAIM-59 (V-CGA-FROZEN, April 6, 2026)** |
| **Enrichment Day-1 lift (prod config)** | **+42.69pp** | **Enriched μ₀ + DiagonalKernel, 3 profiles** | **CLAIM-62 (SVM-003b)** |
| → enriched μ₀ initialization alone | +40.93pp | L2 kernel, N=30/profile | CLAIM-62 decomposition |
| **Fisher info: enrichment → η_eff** | **r=0.9669 empirical=analytical** | **GAE 0.7.20 required (raw_weights)** | **CLAIM-64 (SVM-004b)** |
| **Analyst time savings (SANS-calibrated)** | **30.85 min/alert** | **CI=[29.90,31.81], 30 personas** | **CL-ECON-MEASURED** |
| Per-industry ROI | $523K–$2.8M/year | Healthcare/Midmarket/FinServ | CL-ECON-MEASURED |

**Note on the two accuracy regimes:** The 97.89% / 98.2% numbers are centroidal synthetic (ground-truth profiles, oracle-initialized centroids). The 71.7% / 78.9% numbers are realistic (profiles estimated from domain knowledge, realistic factor distributions, 50 seeds). Both are correct — they measure different things. The centroidal numbers validate the scoring mechanism. The realistic numbers are the honest product claim. **Never mix these in the same sentence.**

**Note on DiagonalKernel numbers:** The +13.2pp / +6.8pp numbers are measured on heterogeneous noise (different σ per factor). With uniform noise across all factors, all kernels are mathematically identical. Every real deployment has heterogeneous noise — the diagonal advantage is real and consistent.

### 4.3 Architecture Decisions Made by Experiments

| Decision | Evidence | Status |
|---|---|---|
| L2 as cold-start kernel | EXP-C1: 36.89pp gap | Settled |
| **DiagonalKernel as v6.0 default** | **V-MV-KERNEL: +13.2pp SOC, +6.8pp S2P on heterogeneous noise. Corr=0.990.** | **Settled** |
| **Kernel selection: noise_ratio > 1.5** | **Selector fixes Ask 2: 4/4 correct. Explanation A confirmed.** | **Settled** |
| **ShrinkageKernel deprioritized** | **Deliverables D2/D3: off-diagonal adds <1pp. Noise ratio alone sufficient.** | **Deferred to v7.0** |
| **Factor quarantine mask deprecated** | **V-HC-CONFIG: mask hurt Day 1 by 6pp. DiagonalKernel supersedes.** | **Deprecated** |
| **Asymmetric η (P0 fix)** | **9-persona stress test: 13-27pp degradation. η_override=0.01 validated across 24 personas.** | **Settled, permanent** |
| **AMBER auto-pause** | **Three-judge consensus. Conservation AMBER → freeze learning.** | **Settled** |
| Pluggable kernels | EXP-E1: L2 wins 2/3 scenarios | Settled |
| Profile centroids (not W matrix) | EXP-C1, EXP-B1 | Settled |
| G (gating matrix) eliminated | EXP-A: +0.01pp | Settled |
| Count-based decay (not Ψ) | EXP-B1: natural phase behavior | Settled |
| τ = 0.1 (not 0.25) | V3B: ECE 0.19→0.036 | Settled |
| Clipping to [0,1] required | V2: escape at dec 6-12 | Settled |
| τ_modifier rejected | OP series: ECE +0.138 degradation | Settled, permanent |
| Loop 2/4 firewall | EXP-S3: Frobenius 0.0028 | Settled, permanent |
| Operative window λ∈[0.5, 0.6] | EXP-OP2 | Settled |
| Checkpoint/rollback required | EXP-OP2: 35% never recover | TD-033 requirement |
| LayerNorm required (Level 2) | V1B: 2.9M× norm explosion | Settled, permanent |
| **γ > 1 ANALYTICALLY PROVEN** | **Oracle separation + 4-LLM math poll (April 8, 2026). Binary simulation: ε=0.05 → γ=0.71<1 ✓; ε=0.20 → γ=1.03>1 ✓. Theorem: γ > 1 ⇔ ε_firm > ε_firm★ ≈ 0.125. Production ε_firm ∈ [0.15, 0.40]. Batch F META-4 RETIRED.** | **CC-21 Tier 2 (conditional). Settled.** |
| centroid_distance_to_canonical as convergence metric | N_half conflates centroid convergence with vector quality. dist(t) = ‖μ(t)−GT‖_F decreases monotonically in every oracle separation experiment (all seeds, all phases). | Hard architectural rule — permanent. |
| **Noise ceiling is kernel-dependent** | **V-B3 + V-MV-KERNEL. L2: σ≤0.157. Diagonal: σ≤0.25.** | **Settled** |
| **Noise ceiling is three-variable** | **V-B3: corruption vector is V×(1-q̄)×η, not σ alone** | **Settled** |
| **Third compounding pathway confirmed** | **CLAIM-59: graph enrichment independent of learning. 54.4% faster p<0.0001, 26/30 seeds.** | **Settled (April 6, 2026)** |
| **raw_weights vs weights distinction** | **CLAIM-64: use raw_weights (true 1/σ²) for η_eff. weights (pre-normalized) for scoring. Silent footgun in v0.7.19.** | **Settled, GAE 0.7.20 required** |
| **Innovation 10 CLOSED** | **CL-ECON-MEASURED UNCONDITIONAL: 30.85 min/alert, $523K–$2.8M/year (SVM-002b, SANS-calibrated)** | **Settled (March 26, 2026)** |

---

## 5. The Ontological Architecture

*(Unchanged from v8 §5. Reproduced for completeness.)*

**The deepest conceptual insight:** Profile centroids are compiled ontologies.

A domain ontology statement — "for insider_behavioral alerts, escalate when asset_criticality is high and pattern_history is low" — is compiled into a point in n_f-dimensional factor space:

```
μ[insider_behavioral, escalate, :] = [0.35, 0.85, 0.60, 0.15, 0.50, 0.30]
                                       ↑      ↑                 ↑
                                    travel  asset           pattern
                                    (low)  (HIGH)           (LOW)
```

Three consequences:

1. **The math does not understand the domain.** ProfileScorer computes pure L2 distance. It works identically for SOC alerts, procurement decisions, and financial compliance — only the centroid values change. This is why GAE can be domain-agnostic.

2. **Learning is geometric, not symbolic.** Verified outcomes pull/push centroids toward observed patterns. After 1,000 decisions, centroids have drifted from expert opinion toward empirical reality.

3. **The meta-graph is the moat.** Evolved centroids cannot be copied without running the decisions that evolved them. Initial centroids are opinion. Evolved centroids are evidence.

**Three-layer design:**

```
Layer 1: Domain Ontology (expert knowledge)
    → compiled into →
Layer 2: Meta-Graph (profile centroids μ — "compiled ontology")
    → scored via →
Layer 3: Mathematical Engine (ProfileScorer, distance kernel)
```

**Two levels of institutional judgment — GAE owns Level 1:**

ProfileScorer implements **Level 1 — Decision Intelligence**: the system learns *what to decide*. Per-category profile centroids μ[c,a,:] encode verified organizational judgment as geometric objects. Centroid updates from verified outcomes (Eq. 4b-final) are the mechanism. Slow-moving (months, hundreds of decisions). EXP-C1: 97.89% zero-learning accuracy; EXP-B1: 98.2% warm-start.

**Level 2 — Deployment Intelligence** (AgentEvolver) is implemented in the domain copilot layer, not in GAE. It learns *how to operate* — which prompt variants, framing, and context structure work in this deployment. Moderate-moving (weeks). GATE-OP confirmed the operative window (λ=0.5, p=0.0008). The GAE `bridge.py` (v7.0) and `discovery_engine.py` (v8.0) are **Level 2/3 architecture** in the cross-domain enrichment sense — distinct from the Decision/Deployment Intelligence naming. Both namings are in use; context disambiguates.

**The critical separation** (P16, permanent): `update()` has NO synthesis parameter. μ learns from verified operational experience only. Level 2 Deployment Intelligence signals never contaminate Level 1 centroid evolution. Architecturally enforced, not configurable.

---

## 6. Package Structure — Three-Repository Architecture

```
graph-attention-engine  (Apache 2.0 — THIS REPO)
    Pure math, numpy-only, zero external dependencies
    Exports: ProfileScorer, ScoringResult, KernelType,
             CalibrationProfile, FactorComputer, LearningState,
             OracleProvider, GTAlignedOracle, BernoulliOracle,
             EvaluationScenario, run_evaluation, EvaluationReport,
             compute_judgment, JudgmentResult,
             run_ablation, AblationReport
         ↑
ci-platform             (Apache 2.0)
    UCL, agents, event bus, governance, DomainConfig protocol
         ↑
soc-copilot             (Proprietary)
    SOC domain expertise, FactorComputer implementations,
    SOCDomainConfig, Neo4j seed data, connectors, UI
```

**The one-way dependency (P12) is the architectural contract.** GAE has zero imports from ci-platform or any domain copilot. If a GAE module needs something from a domain, it must be passed in as a parameter (e.g., CalibrationProfile). This is what makes GAE publishable as an independent open-source library.

---

## 7. Causal Architecture: Three Loops, Four Connectors

### 7.1 Four Causal Loops

```
╔══════════════════════════════════════════════════════════════════╗
║  LOOP 1: FAST LOOP (per-alert, seconds) — LIVE in v5.0          ║
║                                                                  ║
║  Alert → FactorComputers → f (1×n_f)                            ║
║            ↑ reads graph           ↓                             ║
║            │          SituationAnalyzer → category c            ║
║            │                       ↓                             ║
║            │          ProfileScorer.score(f, c)                  ║
║            │          Eq. 4-final: softmax(-K(f, μ[c,:,:])/τ)   ║
║            │                       ↓                             ║
║            │          ScoringResult: action, confidence,         ║
║            │          probabilities, distances, factor_vector    ║
║            │                       ↓                             ║
║            │          compute_judgment() → JudgmentResult        ║
║            │                       ↓                             ║
║            │          Human/Auto decision → Outcome              ║
║            │                       ↓                             ║
║            │          ProfileScorer.update(f, c, a, correct)     ║
║            │          Eq. 4b-final: centroid pull/push, clip     ║
║            │                       ↓                             ║
║            │          Graph Write-Back (Decision + Outcome)      ║
║            │                       │                             ║
║            └───────────────────────┘                             ║
║                                                                  ║
║  CLOSURE: Graph is richer. Profiles are refined.                 ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║  LOOP 2: SLOW LOOP (periodic, hours/daily) — v5.5 target        ║
║  Entity embeddings + cross-graph attention sweep                 ║
║  GraphAttentionBridge.sweep() — designed in §11                  ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║  LOOP 3: META LOOP (rare, structural) — v6.0+                   ║
║  Drift detection → profile recalibration → ontology update       ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║  LOOP 4: SYNTHESIS LOOP (external intelligence) — PROPOSAL       ║
║  GATED BY: GATE-M (math), GATE-D (pipeline), GATE-V (decisions) ║
║                                                                  ║
║  External claims → σ[c,a] → Eq. 4-synthesis                     ║
║  P(a|f,c,σ) = softmax(-(‖f-μ‖²+λ·σ[c,a])/(τ))                  ║
║  λ∈[0.5,0.6] operative window (GATE-OP passed, p=0.0008)        ║
║                                                                  ║
║  ARCHITECTURAL FIREWALL (PERMANENT):                             ║
║    ProfileScorer.update() has NO synthesis parameter.            ║
║    μ is NEVER updated using σ. σ flows through score() only.     ║
║    Temperature τ=0.1 REGARDLESS of synthesis state.              ║
║    τ_modifier permanently rejected (ECE +0.138 degradation).     ║
╚══════════════════════════════════════════════════════════════════╝
```

### 7.2 Four Causal Connectors

**Connector 1 — Graph → Factors:** FactorComputers traverse Neo4j → produce f ∈ [0,1]^n_f

**Connector 2 — Factors + Profiles → Decision:**
- Input: f (factor vector), μ[c,:,:] (category centroid slice), τ, actions
- Output: ScoringResult with action, confidence, probabilities, distances
- Mechanism: Eq. 4-final — `softmax(-K(f, μ[c,a,:]) / τ)` for each action a
- Kernel K: L2 default, pluggable via KernelType

**Connector 3 — Decision + Outcome → Profile Update:**
- Trigger: verified outcome (human review or auto-approve)
- Mechanism: Eq. 4b-final — centroid pull (correct) / push (incorrect), count-based decay, clip to [0,1]
- Asymmetry: penalty_ratio (20:1 SOC default) via CalibrationProfile

**Connector 4 — Level 1 Data Hooks → Level 2/3:**
- DecisionRecord, OutcomeRecord, ProfileSnapshot written by every domain
- These are the substrate Level 2 (GraphAttentionBridge) reads
- See §11.4 for the full specification

---

## 8. Tier 1 — Factor Protocol & Assembly (`gae/factors.py`)

**FactorComputer Protocol** (domain-agnostic):

```python
class FactorComputer(Protocol):
    """
    Computes a single normalized factor value [0,1] from graph data.
    Implementations live in domain repos (e.g., SOC TravelMatchFactor).
    GAE defines the interface only.
    """
    @property
    def factor_name(self) -> str: ...
    
    async def compute(self, alert: AlertContext, graph: GraphClient) -> float:
        """Returns value in [0.0, 1.0]. Never raises — returns 0.5 on error."""
        ...

def assemble_factor_vector(
    computers: List[FactorComputer],
    alert: AlertContext,
    graph: GraphClient,
) -> np.ndarray:
    """
    Runs all FactorComputers, returns f of shape (n_factors,).
    Order matches the order of computers list — must match centroid dimension order.
    """
```

**Important:** f is the QUERY in Eq. 4-final. The factor vector is not the answer — it is the question the engine scores against μ. The semantics: "how close is THIS ALERT to what each action looks like for this category?"

---

## 9. Tier 2 — Profile-Based Scoring (`gae/profile_scorer.py`)

**Role in the two-level architecture:** ProfileScorer is **Level 1 — Decision Intelligence**. It learns *what to decide* by maintaining per-category profile centroids that encode verified organizational judgment as geometry. See §5 for the full two-level framing. This section specifies the implementation contract.

### 9.1 Equation

**Eq. 4-final (L2 — cold-start fallback):**
```
P(a | f, c) = softmax(-K(f, μ[c,a,:]) / τ)

where K(f, μ[c,a,:]) = ‖f − μ[c,a,:]‖²   (L2 — used when noise_ratio ≤ 1.5)
```

**Eq. 4-diagonal (v6.0 default for noise_ratio > 1.5):**
```
P(a | f, c) = softmax(-(f − μ[c,a,:])ᵀ W (f − μ[c,a,:]) / τ)

where W = diag(1/σ²)  — per-factor inverse noise variance
      σ = per-factor noise from P28 deployment qualification

When W = I (identity): reduces to L2.
When σ is uniform: reduces to L2 (all weights equal).
When σ varies per factor: down-weights noisy dimensions, up-weights clean ones.
```

**Kernel selection rule (settled, Explanation A confirmed):**
```
noise_ratio = max(σ_per_factor) / min(σ_per_factor)
if noise_ratio > 1.5 → DiagonalKernel(weights=1/σ²)
else                  → L2Kernel
```

**Why not ShrinkageKernel (full Mahalanobis)?**
Deliverables D2 and D3 tested off-diagonal covariance: SOC gap 0.8pp, S2P gap -0.2pp.
Noise ratio alone drives the kernel advantage (Explanation A). Correlation density adds nothing measurable. ShrinkageKernel deprioritized to v7.0 research.

Three first-class constituents:
- **f** (query): factor vector, shape (n_f,), produced by FactorComputers
- **μ** (keys): profile centroids, shape (n_c × n_a × n_f), compiled ontology
- **c** (router): category index, from SituationAnalyzer (MoE head selector)

### 9.2 Kernel Architecture (v10 — settled)

**Three production kernels (two shipped, one collecting):**

```python
# gae/kernels.py — ScoringKernel protocol
class ScoringKernel(Protocol):
    def compute_distance(self, f: np.ndarray, mu: np.ndarray) -> np.ndarray: ...
    def compute_gradient(self, f: np.ndarray, mu: np.ndarray) -> np.ndarray: ...

class L2Kernel:           # ‖f−μ‖² — cold-start fallback
class DiagonalKernel:     # (f−μ)ᵀ W (f−μ) — v6.0 default for noise_ratio > 1.5
# ShrinkageKernel:        # deprioritized to v7.0 — off-diagonal adds <1pp
```

**KernelType enum (backward compat — deprecated in favor of kernel instances):**

```python
class KernelType(Enum):
    L2 = "l2"                    # Cold-start fallback. -||f - μ||².
    DIAGONAL = "diagonal"        # v6.0 DEFAULT. -(f-μ)ᵀ W (f-μ) where W=diag(1/σ²).
    MAHALANOBIS = "mahalanobis"  # DEPRECATED. Use DiagonalKernel — off-diagonal adds nothing.
    COSINE = "cosine"            # DEPRECATED. Use for pre-normalized embeddings only.
    DOT = "dot"                  # WARNED. 61% on [0,1] data. Legacy/comparison only.
```

**Kernel selection (KernelSelector — §10.11):**
- Phase 2 (COMPUTE): noise_ratio > 1.5 → diagonal, else L2. One parameter. No ρ_max.
- Phase 3 (SHADOW): all kernels scored simultaneously, rolling 100-decision window.
- Phase 4 (QUALIFY): lock winner after ~250 verified decisions.
- Ongoing: should_reconsider() on σ change, ρ change, covariance λ drop.

**Noise ceiling is kernel-dependent:**

| Kernel | GREEN | AMBER | RED |
|---|---|---|---|
| L2 | σ ≤ 0.105 | 0.105 < σ ≤ 0.157 | σ > 0.157 |
| Diagonal | σ ≤ 0.157 | 0.157 < σ ≤ 0.25 | σ > 0.25 |

### 9.3 ScoringResult

```python
@dataclass
class ScoringResult:
    """Output of Eq. 4-final. All fields are populated on every score() call."""
    action_probabilities: np.ndarray  # shape (n_a,) — softmax output
    selected_action: str              # argmax action name
    confidence: float                 # max probability value
    distances: np.ndarray             # raw distance values (before softmax), shape (n_a,)
    factor_vector: np.ndarray         # f — preserved per R4, shape (n_f,)
    category_index: int               # c — which profile set was used
    temperature: float                # τ used (always 0.1 in production)
    kernel: str                       # which kernel was used
    probabilities: np.ndarray         # alias for action_probabilities — primary field name
    
    # NOTE: The primary field name is .probabilities, not .scores.
    # This is confirmed in docs/users_guide.md §5.2.
```

### 9.4 ProfileScorer Interface

```python
class ProfileScorer:
    def __init__(
        self,
        mu: np.ndarray,          # shape (n_c, n_a, n_f) — MUST match factor order
        actions: List[str],      # action names, length n_a
        tau: float = 0.1,        # V3B validated. NEVER use 0.25.
        kernel_type: KernelType = KernelType.L2,  # backward compat (deprecated)
        calibration: Optional[CalibrationProfile] = None,
        # v6.0 additions:
        kernel: Optional[ScoringKernel] = None,   # L2Kernel() if None. DiagonalKernel(weights) for v6.0 default.
        factor_mask: Optional[np.ndarray] = None,  # DEPRECATED by DiagonalKernel. Binary 0/1 mask. Still functional.
        eta: float = 0.05,                         # base learning rate (confirm path)
        eta_neg: float = 0.05,                     # base penalty rate (incorrect outcomes)
        eta_override: Optional[float] = None,      # P0 FIX: attenuated η for override path. 0.01 canonical.
        min_confidence: float = 0.0,               # gate: update only if confidence > threshold
        auto_pause_on_amber: bool = False,         # conservation AMBER → freeze learning. Default False (backward compat).
    ) -> None: ...

    def score(
        self,
        f: np.ndarray,           # shape (n_f,) or (1, n_f)
        category_index: int,     # from SituationAnalyzer
        synthesis: None = None,  # ALWAYS None in v5.0/v6.0. Loop 4 PROPOSAL only.
    ) -> ScoringResult:
        """
        Score with kernel dispatch:
          1. Apply factor_mask if set (zero out masked dimensions)
          2. kernel.compute_distance(f_effective, mu_effective) → distances
          3. softmax(-distances / τ) → probabilities
        """
        ...

    def update(
        self,
        f: np.ndarray,           # original factor vector (R4 — never recomputed)
        category_index: int,
        action_index: int,       # gt_action_index for correct action
        correct: bool,           # was the recommendation correct?
        # NOTE: NO synthesis parameter — Loop 2/4 firewall, permanent (P16)
    ) -> Optional[CentroidUpdate]:
        """
        Update with kernel-aware gradient + safety gates:
          1. Check auto_pause_on_amber → early return if paused
          2. Check min_confidence gate
          3. kernel.compute_gradient(f, mu_single) → gradient direction
          4. Apply asymmetric η: η_confirm (correct) or η_override (incorrect)
          5. Apply factor_mask to gradient (masked dims don't change)
          6. Clip centroids to [0, 1]
        Returns CentroidUpdate with delta_norm, category, action, outcome.
        """
        ...

    # Conservation integration (v6.0):
    def set_conservation_status(self, status: str) -> None:
        """Called by ConservationMonitor. 'AMBER'/'RED' → pause, 'GREEN' → resume."""
        ...

    @property
    def conservation_status(self) -> str: ...

    @property
    def is_paused(self) -> bool: ...

    @property
    def centroids(self) -> np.ndarray:
        """Alias for self.mu. Shape (n_c, n_a, n_f)."""
        ...

    @property
    def update_gate_stats(self) -> Dict:
        """{'applied': N, 'gated': M} — tracks paused/gated updates."""
        ...

    # Existing methods (unchanged):
    def diagnostics(self) -> Dict[str, Any]: ...
    def checkpoint(self) -> Dict: ...
    def rollback(self, checkpoint: Dict) -> None: ...
    def reset_to_config(self, profile_config: Dict) -> None: ...
    def freeze(self) -> None: ...
    def unfreeze(self) -> None: ...
```

### 9.5 Learning Equation

**Eq. 4b-final (with asymmetric η — P0 fix):**
```
Confirm path (analyst accepts system recommendation):
  μ[c,a,:] ← μ[c,a,:] + η_confirm·K.gradient(f, μ[c,a,:])     (pull toward observed)
  η_confirm = 0.05 (clean signal, full learning rate)

Override path (analyst changes action):
  μ[c,a,:] ← μ[c,a,:] − η_override·K.gradient(f, μ[c,a,:])    (push away from wrong)
  η_override = 0.01 (noisy signal, attenuated 5×)

Where K.gradient is kernel-aware:
  L2Kernel:       gradient = (f − μ)
  DiagonalKernel: gradient = W/W.max() · (f − μ)    — normalized weights control direction, not magnitude
                  NOTE: W/W.max() NOT W*(f-mu). GAE-GRADIENT-001 fixed in v0.7.7.
                  W*(f-mu) was wrong: high-W factors dominated by magnitude, corrupting learning direction.
                  W/W.max()*(f-mu): weights in [0,1], direction preserved, all factors contribute correctly.

REQUIRED: all μ values clipped to [0, 1] after every update (V2 validated)
REQUIRED: masked dimensions (factor_mask) do NOT update (gradient zeroed)
REQUIRED: τ = 0.1 always — never modified by synthesis or any other condition
```

**Why asymmetric η (P0 fix):**
9-persona LLM-judge stress test found 13-27pp centroid degradation from realistic analyst
override quality (q̄=0.60-0.70). The override path carries noise from analyst errors.
η_override=0.01 attenuates this noise by 5×. Validated across 24 personas (1C quality sweep).
At q̄=0.57 (worst quality): +0.5pp with η_override=0.01 (no degradation). Without: -9pp.

**compute_eta_override() formula (gae/calibration.py):**
```python
def compute_eta_override(eta_confirm=0.05, mean_quality=0.75, quality_variance=0.02):
    """
    η* ∝ (2q̄-1) / (2σ²_q + signal). Directionally correct, ~2× overestimate vs empirical.
    Global default η_override=0.01 is the validated value. This formula is diagnostic.
    """
```

---

## 10. v5.0 Complete API Surface

### 10.1 Module Overview

All v5.0 modules shipped. v6.0 COMPLETE. 517 tests. v0.7.17.

```
gae/
├── profile_scorer.py    ✅ ProfileScorer, ScoringResult, KernelType, CentroidUpdate (updated v6.0: kernel, factor_mask, eta_override, auto_pause)
├── kernels.py           ✅ NEW v6.0: ScoringKernel protocol, L2Kernel, DiagonalKernel (28 tests)
│                            NOTE: DiagonalKernel.compute_gradient = W/W.max()*(f-mu) — see GAE-GRADIENT-001 fix
├── covariance.py        ✅ NEW v6.0: CovarianceEstimator, CovarianceSnapshot (23 tests) — COLLECTS only, does not score
├── kernel_selector.py   ✅ NEW v6.0: KernelSelector, KernelRecommendation (46 tests) — Phase 2/3/4 kernel selection
├── referral.py          ✅ NEW v6.0: ReferralEngine, ReferralRule, ReferralDecision, OverrideDetector (31 tests)
├── convergence.py       ✅ v2: EPSILON=0.10, safety_factor=2.0
│                            OLSMonitor: CUSUM on OLS, h=5.0 (OLS scale), plateau-snapshot baseline (CLAIM-OLS-01)
│                            VarQMonitor: LOGGED ONLY — Bernoulli mixture theorem = PERMANENT HARD STOP for gating
├── enrichment_advisor.py ✅ NEW Phase 1: enrichment recommendation engine — ranks factors by expected Day-1 gain
│                            Validated on 5 deployment profiles. Integrates with P28 Phase 2 report.
│                            Input: σ_profile from CovarianceEstimator. Output: factor priority ranking.
├── calibration.py       ✅ CalibrationProfile + compute_factor_mask, mask_to_array, compute_eta_override,
│                            derive_theta_min, check_conservation, compute_optimal_tau, compute_breach_window,
│                            estimate_fisher_information, predict_n_half
│                            η_neg guard: ValueError on η_neg ≥ 1.0 (v0.7.8 fix)
├── factors.py           ✅ FactorComputer protocol, assemble_factor_vector
├── learning.py          ✅ LearningState (delegates to ProfileScorer.update)
├── scoring.py           ⚠️ DEPRECATED — forwards to ProfileScorer (remove at v7.0)
├── oracle.py            ✅ OracleProvider, GTAlignedOracle, BernoulliOracle (NEW v5.0)
├── evaluation.py        ✅ run_evaluation, EvaluationScenario, EvaluationReport (NEW v5.0)
├── judgment.py          ✅ compute_judgment, JudgmentResult (NEW v5.0)
├── ablation.py          ✅ run_ablation, AblationReport (NEW v5.0)
├── fisher.py            ✅ estimate_fisher_information, predict_n_half, enrichment_multiplier
├── contracts.py         ✅ Unchanged
├── primitives.py        ✅ Unchanged
├── events.py            ✅ Unchanged
├── store.py             ✅ Updated: ProfileScorer state persistence
├── types.py             ✅ Unchanged
├── embeddings.py        📋 Tier 4 — v6.5
├── bridge.py            📋 Level 2 GraphAttentionBridge — v7.0
└── discovery_engine.py  📋 Level 3 — v8.0
```

### 10.2 CalibrationProfile

```python
@dataclass
class CalibrationProfile:
    """Domain-configurable hyperparameters for the scoring and learning pipeline."""
    
    # Core (validated)
    temperature: float = 0.1             # V3B: ECE=0.036 at τ=0.1. Never use 0.25.
    penalty_ratio: float = 20.0          # SOC default. 5.0 for procurement.
    
    # Learning rates (in extensions dict for backward compat — will move to top-level v5.5)
    extensions: dict = field(default_factory=lambda: {
        "eta": 0.05,           # base learning rate (correct outcomes)
        "eta_neg": 0.05,       # base penalty rate (incorrect outcomes)
        "count_decay": 0.001,  # count-based decay: η(n) = η₀/(1 + n·count_decay)
        "kernel": "l2",        # default kernel type
    })
    
    # Per-factor decay (v5.5)
    decay_class_rates: dict = field(default_factory=lambda: {
        "permanent": 0.0001, "standard": 0.001,
        "campaign": 0.003, "transient": 0.01,
    })
    
    # v7 backward compat (deprecated — used by ScoringMatrix only)
    learning_rate: float = 0.02          # DEPRECATED. Use extensions["eta"].
    
    # TODO v5.5: Add category_thresholds dict for per-category auto-approve thresholds
    # This enables 40%+ auto-approve coverage (v5.5-R1 from product_requirements_v1)
    # category_thresholds: dict = field(default_factory=dict)
```

### 10.3 LearningState

```python
@dataclass
class LearningState:
    """
    Wraps ProfileScorer and manages the learning loop.
    Handles bootstrap calibration, rollback, and checkpoint.
    """
    profile_scorer: ProfileScorer        # THE scoring mechanism (TD-029)
    calibration: CalibrationProfile
    decision_history: List[WeightUpdate]
    
    def update(
        self,
        f: np.ndarray,
        category_index: int,
        action_index: int,
        correct: bool,
    ) -> Optional[WeightUpdate]:
        """
        Delegates to ProfileScorer.update().
        Records update to history (for GATE-R, EXP-G1).
        Returns WeightUpdate with delta_applied, centroid snapshot.
        """
        ...
    
    def bootstrap_calibrate(
        self,
        n_decisions: int = 1200,
        seed: int = 42,
    ) -> None:
        """
        Bootstrap warm-start: generate synthetic decisions from current μ.
        Converges centroids before real data arrives.
        SOC: 1,200 decisions, converged=True, drift=0.0097 confirmed.
        """
        ...
```

### 10.4 OracleProvider (`gae/oracle.py`) — NEW v5.0

```python
class OracleProvider(Protocol):
    """
    Ground-truth provider for evaluation scenarios.
    The oracle answers "what is the correct action for this (f, c) pair?"
    Used in run_evaluation() to compute accuracy against a known ground truth.
    """
    def get_correct_action(
        self,
        f: np.ndarray,
        category_index: int,
    ) -> OracleResult:
        """Returns the ground-truth action and confidence in that ground truth."""
        ...

@dataclass
class OracleResult:
    """Ground truth for a single scoring event."""
    correct_action_index: int
    correct_action_name: str
    ground_truth_confidence: float   # how certain is the oracle about this ground truth?

class GTAlignedOracle:
    """
    Ground-truth oracle aligned with profile centroids.
    Correct answer = argmin L2 distance from f to μ[c,:,:].
    This is the oracle used in all synthetic experiments (EXP-C1, EXP-B1).
    Achieves 97.89% zero-learning accuracy because scoring and oracle agree on geometry.
    """
    def __init__(self, mu: np.ndarray, actions: List[str]) -> None: ...

class BernoulliOracle:
    """
    Probabilistic oracle: each action correct with probability p_correct.
    Used to simulate imperfect feedback (noise robustness testing).
    EXP-B1 showed 98.1% accuracy maintained at 30% oracle noise.
    """
    def __init__(self, base_oracle: OracleProvider, p_correct: float = 0.7) -> None: ...
```

### 10.5 Evaluation (`gae/evaluation.py`) — NEW v5.0

```python
@dataclass
class EvaluationScenario:
    """
    A single (f, c) → ground_truth_action test case.
    Human-authored scenarios (EVAL-1-SOC: 36 SOC scenarios across all 6 categories).
    """
    name: str                        # "credential_access_singapore_login"
    description: str
    f: np.ndarray                    # factor vector, shape (n_f,)
    category_index: int              # SituationAnalyzer output for this scenario
    expected_action: str             # ground truth action name
    expected_action_index: int

@dataclass
class EvaluationReport:
    """Output of run_evaluation()."""
    accuracy: float                  # fraction correct
    per_category_accuracy: Dict[str, float]
    confusion_matrix: np.ndarray     # shape (n_a, n_a)
    ece: float                       # Expected Calibration Error (compute_ece())
    n_scenarios: int
    scenarios_by_outcome: Dict[str, List[str]]  # "correct"/"incorrect" → scenario names

def run_evaluation(
    profile_scorer: ProfileScorer,
    scenarios: List[EvaluationScenario],
    oracle: OracleProvider,
    learn: bool = False,             # if True: call update() after each score
) -> EvaluationReport:
    """
    Evaluate ProfileScorer against provided scenarios.
    When learn=True: simulates online learning — each decision updates μ.
    """
    ...

def compute_ece(
    probabilities: np.ndarray,      # shape (n_scenarios, n_a)
    correct_indices: np.ndarray,    # shape (n_scenarios,)
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error.
    V3B validated: ECE=0.036 at τ=0.1 (well-calibrated).
    ECE=0.19 at τ=0.25 (poorly calibrated — 5× worse).
    """
    ...
```

### 10.6 Judgment (`gae/judgment.py`) — NEW v5.0

```python
@dataclass
class JudgmentResult:
    """
    Human-readable explanation of a scoring decision.
    Bridges the gap between raw ScoringResult and analyst-facing explanation.
    """
    action: str                      # selected action
    confidence: float                # confidence value
    confidence_tier: str             # "high" / "medium" / "discovery"
    top_factors: List[Tuple[str, float]]  # [(factor_name, value), ...] sorted by influence
    rationale: str                   # plain-English explanation template
    should_auto_approve: bool        # confidence >= auto_approve_threshold
    centroid_distance: float         # L2 distance to winning centroid
    margin: float                    # confidence − second-highest probability

def compute_judgment(
    scoring_result: ScoringResult,
    f: np.ndarray,
    mu: np.ndarray,
    category_index: int,
    factor_names: List[str],
    actions: List[str],
    auto_approve_threshold: float = 0.90,
) -> JudgmentResult:
    """
    Compute human-readable judgment from scoring result.
    
    Confidence tiers:
        high:      confidence >= 0.80 (system recommends; analyst can accept quickly)
        medium:    confidence >= 0.50 (system suggests; analyst should review)
        discovery: confidence <  0.50 (system uncertain; escalate to human review)
    """
    ...
```

### 10.7 Ablation (`gae/ablation.py`) — NEW v5.0

```python
@dataclass
class AblationReport:
    """Output of run_ablation()."""
    factor_importance: Dict[str, float]    # factor_name → accuracy_drop when removed
    full_accuracy: float                   # baseline accuracy with all factors
    per_factor_accuracy: Dict[str, float]  # accuracy without each factor
    top_factors: List[str]                 # sorted by importance (most important first)
    dead_factors: List[str]                # accuracy_drop <= 0 (removing helps or neutral)

def run_ablation(
    profile_scorer: ProfileScorer,
    scenarios: List[EvaluationScenario],
    factor_names: List[str],
) -> AblationReport:
    """
    Leave-one-factor-out ablation study.
    
    For each factor i:
        1. Zero out dimension i in every scenario's f
        2. Re-evaluate with ProfileScorer
        3. accuracy_drop = full_accuracy − accuracy_without_factor_i
        4. accuracy_drop > 0 → factor helps
        5. accuracy_drop <= 0 → factor is a "dead factor" (neutral or harmful)
    
    Note: EXP-D2 found 0 significant factor interactions in SOC data.
    Ablation is therefore independent-factor analysis, not interaction study.
    """
    ...
```

### 10.8 Kernels (`gae/kernels.py`) — NEW v6.0

```python
class ScoringKernel(Protocol):
    """Protocol for all scoring kernels. All kernels implement both methods."""
    def compute_distance(self, f: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """f: (d,), mu: (A, d) → distances: (A,)"""
        ...
    def compute_gradient(self, f: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """f: (d,), mu: (d,) single centroid → gradient: (d,)"""
        ...

class L2Kernel:
    """‖f−μ‖². Cold-start default. Equal weight to all dimensions."""
    def compute_distance(self, f, mu): return np.sum((f - mu)**2, axis=-1)
    def compute_gradient(self, f, mu): return f - mu

class DiagonalKernel:
    """(f−μ)ᵀ W (f−μ). v6.0 default for noise_ratio > 1.5.
    W = diag(weights). Typical: weights = 1/σ² normalized to [0,1].
    When weights are all 1.0: mathematically identical to L2Kernel.
    When weights are 0/1: equivalent to binary factor mask.
    """
    def __init__(self, weights: np.ndarray): ...  # shape (d,)
    def compute_distance(self, f, mu): return np.sum(self.weights * (f-mu)**2, axis=-1)
    def compute_gradient(self, f, mu):
        # GAE-GRADIENT-001 fix (v0.7.7): normalize before multiply.
        # W/W.max() keeps weights in [0,1] — direction preserved, magnitude bounded.
        # W*(f-mu) was WRONG: dominant weights corrupted gradient direction.
        w_max = self.weights.max()
        if w_max == 0:
            return f - mu  # fallback to L2 if all weights zero
        return (self.weights / w_max) * (f - mu)
```

**28 tests** covering: protocol compliance, L2 distance/gradient, DiagonalKernel init/distance/gradient, unit weights = L2, binary weights = factor mask, zero weights → uniform probabilities.

### 10.9 Covariance Estimator (`gae/covariance.py`) — NEW v6.0

```python
@dataclass
class CovarianceSnapshot:
    sigma: np.ndarray              # d×d covariance matrix
    sigma_inv: Optional[np.ndarray]  # d×d inverse (None if singular)
    correlation: np.ndarray         # d×d correlation matrix
    shrinkage_lambda: float        # 1.0=diagonal, 0.0=full sample
    condition_number: float        # κ(Σ̂) — high = unstable
    n_samples: int
    per_factor_sigma: np.ndarray   # d, diagonal variances

class CovarianceEstimator:
    """
    Online covariance estimation with Ledoit-Wolf optimal shrinkage
    and exponential decay for regime shift tracking.
    
    v6.0: COLLECTS data. Does NOT affect scoring.
    v7.0: May feed ShrinkageKernel (if v7.0 research validates it).
    
    One estimator per (category, action) pair.
    SOC: 6×4=24 estimators. S2P: 5×5=25 estimators.
    """
    def __init__(self, d: int, half_life_decisions: int = 300): ...
    def update(self, f: np.ndarray) -> None: ...
    def get_snapshot(self) -> CovarianceSnapshot: ...
    def get_change_rate(self, previous: CovarianceSnapshot) -> float: ...
```

**23 tests** covering: cold start (identity matrix), warm snapshots (symmetry, PSD, inverse, correlation bounds, shrinkage), per-factor sigma ordering, change rate.

### 10.10 Kernel Selector (`gae/kernel_selector.py`) — NEW v6.0

```python
@dataclass
class KernelRecommendation:
    recommended_kernel: str     # "l2" or "diagonal"
    confidence: float           # margin over runner-up
    scores: Dict[str, Dict]     # per-kernel agreement rates
    method: str                 # "empirical" (Phase 4) or "rule" (Phase 2)
    reason: str                 # human-readable explanation
    sufficient_data: bool       # True if ≥100 decisions

class KernelSelector:
    """
    Empirical kernel selection during shadow mode.
    
    Phase 2 (COMPUTE): noise_ratio > 1.5 → diagonal, else L2.
    Phase 3 (SHADOW): score every alert with all kernels. Rolling 100-decision window.
    Phase 4 (QUALIFY): lock the winner after ~250 verified decisions.
    Ongoing: should_reconsider() on σ/ρ/λ changes.
    
    Design principle: "Ship the selector, not a default."
    Each deployment gets what ITS data says is best.
    """
    MIN_DECISIONS_FOR_RECOMMENDATION = 100
    
    def __init__(self, d: int, sigma_per_factor: np.ndarray,
                 correlation_max: float = 0.0): ...
    def preliminary_recommendation(self) -> KernelRecommendation: ...  # Phase 2
    def record_comparison(self, factors, category_index, mu,
                          analyst_action_index, actions) -> Dict[str, int]: ...  # Phase 3
    def recommend(self) -> KernelRecommendation: ...  # Phase 4
    def should_reconsider(self, new_sigma=None, new_rho_max=None,
                          covariance_lambda=None) -> Optional[str]: ...  # Ongoing
    def get_comparison_summary(self) -> Dict: ...
    def reset_comparison(self) -> None: ...
```

**46 tests** covering: KernelScore dataclass, initialization, preliminary rules (uniform→L2, hetero→diagonal, high ρ→shrinkage), record comparison, accumulation, recommend (insufficient/sufficient data, picks highest agreement), summary, monitoring (σ change, ρ change, λ drop), reset.

### 10.11 Calibration Additions (`gae/calibration.py`) — v6.0 extensions

```python
# Factor mask (DEPRECATED by DiagonalKernel — still functional)
def compute_factor_mask(sigma_per_factor: Dict[str, float],
                        threshold: float = 0.20) -> Dict[str, bool]:
    """Binary include/exclude. sigma < threshold → True (include)."""

def mask_to_array(mask: Dict[str, bool],
                  factor_names: List[str] = None) -> np.ndarray:
    """Dict → float64 array (1.0=include, 0.0=exclude). SOC default factor order built-in."""

# Asymmetric η
def compute_eta_override(eta_confirm: float = 0.05,
                         mean_quality: float = 0.75,
                         quality_variance: float = 0.02) -> float:
    """η* ∝ (2q̄-1)/(2σ²_q+signal). Diagnostic — global default 0.01 is validated value."""

# Conservation
def derive_theta_min(eta: float = 0.05, n_half: float = 14.0,
                     t_max_days: float = 21.0) -> float:
    """θ_min = η × N_half² / T_max = 0.467. Canonical product promise: 3 weeks."""

def check_conservation(alpha: float, q: float, v: float,
                       theta_min: float = 0.467) -> str:
    """Returns 'GREEN', 'AMBER', or 'RED'."""

# Fisher information
def estimate_fisher_information(centroids: np.ndarray, tau: float = 0.1) -> float: ...
def predict_n_half(fisher: float, eta: float = 0.05) -> float: ...
def enrichment_multiplier(fisher_before: float, fisher_after: float) -> float: ...
```

### 10.12 Public API (`gae/__init__.py`) — updated v6.0

```python
# All symbols exported from gae/__init__.py as of v6.0
from gae import (
    # Core scoring
    ProfileScorer,
    ScoringResult,
    KernelType,
    CentroidUpdate,
    build_profile_scorer,
    
    # Kernels (NEW v6.0)
    L2Kernel,
    DiagonalKernel,
    
    # Covariance (NEW v6.0 — collects only)
    CovarianceEstimator,
    CovarianceSnapshot,
    
    # Kernel selection (NEW v6.0)
    KernelSelector,
    KernelRecommendation,
    
    # Configuration
    CalibrationProfile,
    
    # Calibration utilities (NEW/updated v6.0)
    compute_factor_mask,      # DEPRECATED by DiagonalKernel
    mask_to_array,            # DEPRECATED by DiagonalKernel
    compute_eta_override,
    derive_theta_min,
    check_conservation,
    
    # Factor protocol
    FactorComputer,
    assemble_factor_vector,
    
    # Learning
    LearningState,
    
    # Oracle
    OracleProvider,
    GTAlignedOracle,
    BernoulliOracle,
    OracleResult,
    
    # Evaluation
    EvaluationScenario,
    run_evaluation,
    EvaluationReport,
    compute_ece,
    
    # Judgment
    compute_judgment,
    JudgmentResult,
    
    # Ablation
    run_ablation,
    AblationReport,
    
    # Deprecated (remove at v7.0)
    # score_alert,  # Use ProfileScorer.score() instead
    # WeightUpdate, # Use CentroidUpdate instead
)
```

**Verification:**
```python
python -c "from gae import ProfileScorer, KernelType, OracleProvider, run_evaluation, compute_judgment, run_ablation"
# Must succeed with no errors.
```

---


### 10.13 Synthetic Data Generation (`gae/synthetic.py`) — NEW v10.4

> **HARD RULE: NO LLM DEPENDENCY.** `gae/synthetic.py` is pure numpy/scipy.
> Factor vectors are generated parametrically. Correctness is labeled from centroid
> geometry. LLM-based generation lives in cross-graph-experiments and is passed in
> as pre-computed `List[FactorVectorSample]`. GAE never calls an LLM. Never.

> **NO LLM DEPENDENCY.** `gae/synthetic.py` is pure numpy/scipy. It generates factor
> vectors parametrically (Gaussian distributions) and labels correctness from centroid
> distance. LLM-based factor vector generation lives in cross-graph-experiments and is
> passed to `OracleSeparationExperiment` as pre-computed `List[FactorVectorSample]`.

**Architecture rationale:** Synthetic data generation for GAE experiments lives in GAE,
not ci-platform. The oracle separation framework works directly with centroid distances
and OracleProvider (both already in GAE). It is useful for testing any GAE domain.
Overnight onboarding synthetic prep (industry-calibrated alert generation for new customers)
remains in ci-platform — that is deployment infrastructure, not math testing infrastructure.

**Module responsibility:** Generate realistic factor vectors PARAMETRICALLY (no LLM)
and oracle-label correctness from centroid geometry, for γ theorem validation, convergence
experiments, and domain testing. The oracle separation principle: correctness comes from
math (centroid distance to GT), never from LLM judgment. Factor vector distributions
are specified by sigma_profile — open-source users define their own noise characteristics.

```python
# gae/synthetic.py

@dataclass
class FactorVectorSample:
    """A single sampled factor vector with metadata."""
    f: np.ndarray               # shape (d,) — the factor vector in [0,1]^d
    regime: str                 # "cold_start" | "post_disruption" | "enriched"
    sigma_per_factor: np.ndarray  # σ for each factor at time of sampling
    generation_seed: int

class FactorVectorSampler:
    """
    Samples realistic factor vectors PARAMETRICALLY (numpy/Gaussian) for a GAE domain.

    NO LLM DEPENDENCY. This class never calls an LLM. It generates factor vectors
    from Gaussian distributions parameterized by sigma_profile. Correctness is always
    labeled by the oracle (centroid distance to GT), never by LLM judgment.

    If you want LLM-generated factor vectors: call your LLM externally, collect the
    output as List[FactorVectorSample], and pass it directly to OracleSeparationExperiment.
    GAE accepts pre-computed vectors from any source.

    Validated: Exp A (April 2026) confirmed parametric generation produces
    well-differentiated regime vectors (per-factor variance 0.077–0.089,
    regime differentiation confirmed across cold_start vs post_disruption).
    """
    def __init__(
        self,
        domain_config: DomainConfig,
        sigma_profile: np.ndarray,     # per-factor noise standard deviations
        regime_shift: float = 0.3,     # mean shift at disruption boundary
        seed: Optional[int] = None,
    ) -> None: ...

class OracleSeparationExperiment:
    """
    Oracle separation protocol for γ theorem validation.
    Phase 1: cold-start calibration from mu_0 toward GT_1.
    Phase 2: post-disruption re-convergence from mu_T1 toward GT_2 = GT_1 + Δ.

    CORRECTNESS IS DETERMINED BY CENTROID DISTANCE TO GT — NOT BY LLM JUDGMENT.
    Factor vectors can come from FactorVectorSampler (parametric, numpy) or from
    any external source (including LLM-generated, passed as pre-computed List).
    GAE does not call LLMs. GAE does not depend on LLMs.

    Validated: v8 (ε_sim=0.05 < threshold): γ=0.714 < 1 ✓
               v3 (ε_sim=0.20 > threshold): γ=1.033 > 1 ✓
    """
    def __init__(
        self,
        scorer: ProfileScorer,
        oracle: OracleProvider,
        epsilon_firm: float,            # ‖μ_0 − GT_1‖ — firm-specific deviation
        disruption_magnitude: float,    # ‖Δ‖ — shift applied to c_d categories
        disrupted_categories: List[int],
        window: int = 10,               # rolling window w for N_half
        theta: float = 0.85,            # accuracy threshold θ
    ) -> None: ...

    def run_phase1(self, factor_samples: List[FactorVectorSample]) -> Phase1Result: ...
    def run_phase2(self, factor_samples: List[FactorVectorSample],
                   phase1_result: Phase1Result) -> Phase2Result: ...
    def compute_gamma(self, r1: Phase1Result, r2: Phase2Result) -> GammaResult: ...

@dataclass
class GammaResult:
    n_half_1: Optional[int]    # None if Phase 1 DNF (N_half measurement gap)
    n_half_2: Optional[int]
    gamma: Optional[float]     # None if either phase DNF
    centroid_dist_phase1: List[float]  # dist(t) = ‖μ(t) − GT₁‖_F per decision
    centroid_dist_phase2: List[float]  # dist(t) = ‖μ(t) − GT₂‖_F per decision
    n_half_gap_detected: bool  # True if N_half fired before centroid converged
    note: str

class CanonicalCentroid:
    """
    Manages canonical centroids (ground truth) for oracle separation experiments.
    Distinct from the deployed system's centroids — these are the mathematical GT
    that the oracle uses to label correctness.

    NOT the same as BACKLOG-015 centroid_distance_to_canonical (which logs
    distance from the live deployed centroid to a pre-deployment canonical snapshot).
    """
    def __init__(self, domain_config: DomainConfig) -> None: ...

    @classmethod
    def from_ground_truth(cls, gt: np.ndarray) -> CanonicalCentroid: ...

    def apply_disruption(self, delta: np.ndarray,
                         categories: List[int]) -> CanonicalCentroid:
        """Returns a new CanonicalCentroid representing GT_2 = GT_1 + Δ."""
        ...
```

**Production integration note:** The three EXP-G1 log fields (centroid_distance_to_canonical,
pattern_history_value, alert_category_distribution) are logged in ci-platform's triage path —
not in `gae/synthetic.py`. `gae/synthetic.py` is for experiment design, not production logging.

**LLM boundary (hard rule):** `gae/synthetic.py` never calls an LLM. Never imports an LLM
client library. Never requires an API key. The oracle separation experiments in
cross-graph-experiments that used LLMs passed the LLM output (factor vectors) to the
oracle separation framework as pre-computed numpy arrays. That pattern is the contract:
LLM output goes IN as data; GAE does the math.

**Tests:** ≥5 tests covering: OracleSeparationExperiment produces γ<1 below threshold;
γ>1 above threshold; GammaResult.n_half_gap_detected fires when N_half crosses θ before
centroid has converged 80% of D(μ_0, GT); FactorVectorSampler variance within spec;
CanonicalCentroid.apply_disruption matches ‖Δ‖ exactly.

---

### 10.14 Convergence Analysis (`gae/convergence.py`) — NEW v10.4

**Module responsibility:** Model-independent convergence metrics for production deployment
monitoring and EXP-G1 measurement. Separates centroid convergence from N_half accuracy
threshold crossing.

```python
# gae/convergence.py

def centroid_distance_to_canonical(
    mu: np.ndarray,
    canonical: np.ndarray,
) -> float:
    """
    Frobenius distance between the current centroid tensor and the canonical (GT) snapshot.
    The model-independent convergence signal: decreases monotonically under production
    learning dynamics regardless of seed or factor vector quality.

    Replaces N_half as the primary γ measurement metric.
    Simulation finding (April 2026): N_half variance 27× (γ=13.4 vs γ=0.48) at 3 seeds
    vs centroid distance decreasing monotonically in every seed, every phase.

    Args:
        mu:        Current centroid tensor, shape (C, A, d)
        canonical: Canonical (ground truth) centroid, shape (C, A, d)
    Returns:
        float: ‖mu − canonical‖_F  (Frobenius norm)
    """
    return float(np.linalg.norm(mu.flatten() - canonical.flatten()))

def gamma_threshold(
    alpha_cat: float,
    delta_norm: float,
    theta: float = 0.85,
) -> float:
    """
    Computes the ε_firm threshold below which γ ≤ 1 (theorem).
    Production values: alpha_cat=2/6≈0.33, delta_norm≈0.25, theta=0.85 → 0.128.

    Args:
        alpha_cat:  Fraction of alert categories disrupted (c_d / C)
        delta_norm: Disruption magnitude ‖Δ‖
        theta:      Operational accuracy threshold (default 0.85)
    Returns:
        float: ε_firm★ — the threshold
    """
    return alpha_cat * delta_norm * theta / (theta - (1 - alpha_cat))

def phase2_effective_threshold(
    alpha_cat: float,
    theta: float = 0.85,
) -> float:
    """
    The effective accuracy required from disrupted categories for rolling-window
    to declare Phase 2 complete. p_d★ = (θ - (1 - α_cat)) / α_cat ≈ 0.55.
    Explains why Phase 2 is shorter than Phase 1: undisrupted categories carry
    the rolling window, reducing the effective target for disrupted categories.

    Args:
        alpha_cat:  Fraction of alert categories disrupted
        theta:      Operational accuracy threshold (default 0.85)
    Returns:
        float: p_d★ — the effective disrupted-category threshold
    """
    return (theta - (1 - alpha_cat)) / alpha_cat

@dataclass
class ConvergenceTrace:
    """
    Full convergence history for a deployment or experiment phase.
    Primary artifact of EXP-G1 and oracle separation experiments.
    """
    centroid_distances: List[float]  # dist(t) per verified decision
    rolling_accuracy: List[float]    # θ-rolling accuracy per decision
    n_half: Optional[int]            # decision count when rolling_accuracy ≥ θ
    centroid_converged_at: Optional[int]  # decision count when dist plateau
    n_half_gap: bool  # True if n_half fired before centroid plateau
    phase: str        # "phase1" | "phase2"
    epsilon_firm: Optional[float]
```

**Re-Convergence Theorem (formal statement for this module):**

```
Theorem (April 8, 2026): γ = N_half,1 / N_half,2 > 1 if and only if

    ε_firm > γ_threshold(α_cat, ‖Δ‖, θ) ≈ 0.128

Conditions:
  (1) Category-sparse disruption: c_d/C ≈ 0.33
  (2) Warm-started centroids: μ_T1 ≈ GT_1 at Phase 2 start
  (3) ε_firm = ‖μ_0 − GT_1‖ > 0.128

Three proof paths: Geometric | Dimensional | η₋ trap avoidance.
Confirmed: GPT-4.1, Claude Opus 4, Grok 3, Gemini (April 8, 2026).
Simulation: binary prediction correct in both directions.
Production ε_firm ∈ [0.15, 0.40]. Every real deployment clears the threshold.

Commercial claim: CC-21 (Tier 2 — conditional). EXP-G1 → Tier 1.
Reference: math_synopsis_v13 §3.2, claims_registry_v10 §B.5.
```

**Tests:** ≥4 tests covering: gamma_threshold returns 0.128 for production values;
phase2_effective_threshold returns ~0.55 for alpha_cat=2/6; centroid_distance_to_canonical
is zero for identical tensors; ConvergenceTrace.n_half_gap is True when rolling accuracy
crosses θ before centroid distance drops to 20% of initial.

---

### 10.15 Experiment Library (`gae/experiments/`) — NEW v10.6

**Architectural principle:** cross-graph-experiments is the lab notebook.
`gae/experiments/` is the published methods section.

**Three strict rules for everything in this directory:**
1. Apache 2.0 only — no API keys, no proprietary data, no SOC-specific configs
2. Depends only on `gae` public API and `numpy`. No other dependencies.
3. Marked clearly as research/examples, not production code (README states this)

**Module structure:**

```
gae/
  experiments/
    __init__.py
    README.md                  # "Research examples — not production code"
    evals/
      exp_c1.py                # Kernel gap: L2 vs dot product, 36.89pp
      exp_e2.py                # Scale test: 99.9% at 20×10×20 tensor
      factorial.py             # Factorial experiment framework (k×m×n, any domain)
    oracle_separation/
      example_validation.py   # Validate γ>1 for any domain — parametric vectors, no LLM
      convergence_demo.py     # centroid_distance_to_canonical vs N_half — shows why
                               #   N_half is noisy and centroid distance is reliable
    domains/
      medical_triage.py        # 4×4×5 domain: diagnose/treat/refer/monitor
      supply_chain.py          # 3×4×6 domain: approve/hold/expedite/cancel
      financial_approval.py    # 5×4×4 domain: approve/review/reject/escalate
```

---

**`evals/exp_c1.py` — The arXiv kernel gap experiment:**

```python
# gae/experiments/evals/exp_c1.py
"""
EXP-C1: L2 distance vs dot product on identical centroidal data.
Reproduces the 36.89pp result from Banerji (2026a).
Runtime: < 1 minute. No API keys required.
"""
from gae import ProfileScorer, DomainConfig, run_evaluation, GTAlignedOracle
import numpy as np

def run_exp_c1(C=6, A=4, d=6, n_decisions=500, seed=42):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0, 1, (C, A, d))
    config = DomainConfig(C=C, A=A, d=d)
    oracle = GTAlignedOracle(mu=mu, actions=config.actions)

    l2_scorer    = ProfileScorer(mu=mu.copy(), domain_config=config, kernel="l2")
    dot_scorer   = ProfileScorer(mu=mu.copy(), domain_config=config, kernel="dot")

    l2_result  = run_evaluation(l2_scorer,  oracle, n_decisions=n_decisions)
    dot_result = run_evaluation(dot_scorer, oracle, n_decisions=n_decisions)

    gap = l2_result.accuracy - dot_result.accuracy
    print(f"L2:  {l2_result.accuracy:.2%}")
    print(f"Dot: {dot_result.accuracy:.2%}")
    print(f"Gap: {gap:.2%}pp")  # Expected: ~36.89pp
    return gap

if __name__ == "__main__":
    run_exp_c1()
```

**`oracle_separation/example_validation.py` — γ theorem for any domain:**

```python
# gae/experiments/oracle_separation/example_validation.py
"""
Demonstrates oracle separation γ validation for any GAE domain.
Shows: ε_firm=0.05 (below threshold 0.125) → γ < 1
       ε_firm=0.20 (above threshold 0.125) → γ > 1
Reproduces the binary prediction from Banerji (2026a) §4.6.
"""
from gae.synthetic import OracleSeparationExperiment, FactorVectorSampler, CanonicalCentroid
from gae.convergence import gamma_threshold

def run_binary_prediction(domain_config, epsilon_firm, n_decisions=300, seed=42):
    threshold = gamma_threshold(alpha_cat=2/6, delta_norm=0.25, theta=0.85)
    print(f"ε_firm = {epsilon_firm:.3f} | threshold = {threshold:.3f} | "
          f"{'ABOVE (predict γ>1)' if epsilon_firm > threshold else 'BELOW (predict γ<1)'}")

    experiment = OracleSeparationExperiment(
        domain_config=domain_config,
        epsilon_firm=epsilon_firm,
        disrupted_categories=[0, 1],
        delta_norm=0.25,
    )
    result = experiment.run(n_decisions=n_decisions, seed=seed)
    print(f"γ = {result.gamma:.3f} | {'> 1 ✓' if result.gamma > 1 else '< 1 ✓'}")
    return result
```

**`domains/medical_triage.py` — Example non-SOC domain:**

```python
# gae/experiments/domains/medical_triage.py
"""
Medical triage domain: demonstrates GAE with entirely non-SOC actions and factors.
Categories: chest_pain, respiratory, neurological, trauma, other
Actions:    treat_immediately / admit / refer_specialist / discharge
Factors:    vital_signs / symptom_severity / lab_values / history_risk / response_to_treatment
"""
from gae import DomainConfig

MEDICAL_CONFIG = DomainConfig(
    categories=["chest_pain", "respiratory", "neurological", "trauma", "other"],
    actions=["treat_immediately", "admit", "refer_specialist", "discharge"],
    factor_names=["vital_signs", "symptom_severity", "lab_values",
                  "history_risk", "response_to_treatment"],
)
```

---

**What stays in cross-graph-experiments (not moved):**

| Cross-graph artifact | Why it stays |
|---|---|
| SOC-specific personas (factorial_soc_streams.json etc.) | Proprietary — characterizes our deployment parameter space |
| All result JSONs (Batches A–G, oracle sep runs) | Research data, not library code |
| LLM API calling infrastructure | Requires API keys; open-source users write their own |
| LLM-based factor vector generation scripts | LLM calls that produce List[FactorVectorSample] for oracle sep experiments. Output is passed to OracleSeparationExperiment — the GAE public API accepts any pre-computed vectors. |
| Our specific oracle sep runs (v6, v8, v11, v2, v3, Final) | Lab notebook entries, not reproducibility demonstrations |
| V-MV-KERNEL 390-cell data | Proprietary experiment data |

**The migration validation test:** If gae/synthetic.py and gae/convergence.py are
well-designed, the cross-graph-experiments oracle separation scripts should simplify to
thin wrappers calling the GAE library. If they can't be simplified, the abstractions
need revision. The migration is a correctness test for the new modules.

---

## 11. Level 2/3 — Design Specification (v7/v8 Targets)

*(Unchanged from v8.3 §11. Reproduced in brief. Full specification in v8.3.)*

### 11.1 Tier 4 — Entity Embeddings (`gae/embeddings.py`, v5.5)

EmbeddingProvider protocol. Two implementations: PropertyEmbedding (linear projection), TransformerEmbedding (sentence transformer). These are the inputs to Level 2.

### 11.2 Level 2 — GraphAttentionBridge (`gae/bridge.py`, v7.0 target)

Cross-graph attention enrichment. Given embedding matrices from two knowledge graph domains, compute attention-weighted enrichment. **V1B production requirement: LayerNorm after every enrichment sweep (mandatory, not optional).**

Equation:
```
E_i^enriched = LayerNorm(E_i + Σ_{j≠i} CrossAttention(G_i, G_j))
```

**Architectural constraint (permanent):** GraphAttentionBridge does NOT call ProfileScorer.update(). The Loop 2/4 firewall extends to Level 2.

### 11.3 Level 3 — Cross-Domain Discovery (`gae/discovery_engine.py`, v8.0 target)

Uses enriched embeddings from Level 2 to discover cross-domain connections. Outputs new entities, relationships, and scoring dimensions that feed back to Level 1 via profile augmentation.

**γ constraint:** 𝒮(n,t) ~ O(n^{2.11}·t^γ). The t^γ term requires EXP-G1 before any external claim. γ≈1.5 is estimated. Not measured.

### 11.4 Level 1 Data Preservation Hooks (every domain must write)

```python
# Every Level 1 domain writes these on every decision cycle (PLAT-4 validated):

@dataclass
class DecisionRecord:
    """Hook 1 — written at score time."""
    decision_id: str
    alert_id: str
    category: str
    category_index: int
    action: str
    action_index: int
    confidence: float
    factor_vector: List[float]      # MUST be serialized to list, not np.ndarray
    centroid_snapshot: List[List[List[float]]]  # μ[c,:,:] at decision time
    timestamp: str
    domain_id: str

@dataclass
class OutcomeRecord:
    """Hook 2 — written at outcome verification time."""
    decision_id: str                # links to DecisionRecord
    correct: bool
    verified_by: str                # "human" | "auto_approve" | "oracle"
    timestamp: str

@dataclass
class ProfileSnapshot:
    """Hook 3 — written on checkpoint (every 50 decisions when synthesis active)."""
    domain_id: str
    snapshot_id: str
    mu: List[List[List[float]]]     # full centroid array
    counts: List[List[int]]
    decision_count: int
    timestamp: str
    trigger: str                    # "checkpoint_interval" | "operator_start" | "manual"
```

---

## 12. Contracts, API Pipeline, Accumulation

### 12.1 Public API Pipeline (v5.0 → v5.5)

```
# v5.0 pipeline (LIVE):
alert → FactorComputers → f
f → SituationAnalyzer → c
f, c → ProfileScorer.score(f, c) → ScoringResult
ScoringResult → compute_judgment(...) → JudgmentResult
outcome → ProfileScorer.update(f, c, a, correct) → WeightUpdate

# Evaluation pipeline (LIVE):
scenarios, oracle → run_evaluation(scorer, scenarios, oracle) → EvaluationReport
scoring_results → run_ablation(scorer, scenarios, factor_names) → AblationReport

# v5.5 additions:
GraphSnapshot → EmbeddingProvider.embed_graph() → E (n_entities × d)
E_i, E_j → GraphAttentionBridge.enrich() → E_enriched (with LayerNorm)
```

### 12.2 Semantic Accumulation Channels

| Channel | What Accumulates | When |
|---|---|---|
| A — Profile centroids | μ[c,a,:] ← pull/push via Eq. 4b-final | Every verified outcome |
| B — Decision nodes | DecisionRecord written to graph | Every score() call |
| C — Outcome nodes | OutcomeRecord linked to Decision | Every verified outcome |
| D — Profile snapshots | ProfileSnapshot for rollback | Every 50 decisions (when synthesis active) |
| E — Enriched embeddings | Level 2 enrichment (v5.5+) | Periodic sweep |

### 12.3 Referral Routing — Independent of Action Scoring [NEW v10.1]

**File:** `gae/referral.py` (478 tests including 31 referral tests)

**Architecture:** Action routing (ProfileScorer, A=4) and referral routing
(ReferralEngine) are **independent, orthogonal decisions.** Both fire on every
alert. Referral is a VETO — overrides auto-approve at any confidence.

```
ProfileScorer answers: "What action fits this alert?" (A=4: escalate/investigate/suppress/monitor)
ReferralEngine answers: "Should a human see this alert?" (binary: refer or not)

An alert can be: high confidence suppress AND referred (executive account)
                 low confidence investigate AND not referred (normal ambiguity)
```

**Experimental validation (EXP-REFER-LAYERED, March 21):**
Confidence gate for referral: REJECTED (14% precision, 38.7% FPR — active harm).
Policy rules R1-R7: 72.7% DR, 12% FPR, 50.7% precision, 978 net min/100 alerts.
Rules strictly dominate confidence gate on every metric (+39.4pp DR, -22.8pp FPR).
Override learning (ML): deferred to v6.5 (24:1 class imbalance at <1500 decisions).

**Three-phase architecture:**

| Phase | Mechanism | Activation |
|---|---|---|
| Phase 1 (v6.0) | ReferralRules (deterministic, configurable, Day 1) | Always active |
| Phase 2 (v6.5) | + OverrideDetector (learned from analyst override history) | Data-gated: ≥50 production positives |
| Phase 3 (v7.0) | OverrideDetector retrains monthly | Production cadence |

**Components in `gae/referral.py`:**

| Component | Type | Purpose |
|---|---|---|
| `ReferralReason` | Enum (9 values) | R1-R7 (rule-based) + R8 (learned) + NONE |
| `ReferralDecision` | Dataclass | should_refer, reasons, rule_details, audit_summary |
| `ReferralRule` | Protocol | Domain-agnostic: rule_id, reason, evaluate(context) → (bool, dict) |
| `ReferralEngine` | Dataclass | Evaluates all rules. ANY firing → should_refer=True. Pure, no side effects. |
| `OverrideDetectorConfig` | Dataclass | min_positives=50, enabled=False (v6.0 default) |
| `OverrideDetector` | Class (stub) | Interface contract for v6.5. NotImplementedError on predict(). |

**Domain implementation:** SOC implements 7 rules (R1-R7) in `soc-copilot/services/referral_rules.py`.
S2P implements its own rules via `get_referral_rules()`. The protocol is domain-agnostic.

**SOC Rules (validated by EXP-REFER-COVERAGE):**

| Rule | Reason | Fires when | DR contribution |
|---|---|---|---|
| R1 | EXECUTIVE_ACCOUNT | identity_tier ∈ {executive, board, c_suite} | 100% of R1 alerts |
| R2 | RAPID_SUCCESSION | sequence_count ≥ 3 in window | 100% of R2 alerts |
| R3 | COMPLIANCE_MANDATE | insider_threat AND compliance_mode | 100% of R3 alerts |
| R4 | HIGH_VALUE_DATA | data_exfil AND criticality > 0.85 AND monitor/suppress | 42.6% (by design — see note) |
| R5 | ACTIVE_INCIDENT | incident_active flag | 100% of R5 alerts |
| R6 | NEW_ASSET | asset_age_days < 30 | 100% of R6 alerts |
| R7 | CROSS_CATEGORY | ≥2 categories for same user in 1 hour | 100% of R7 alerts |

**R4 detection note:** R4 has 42.6% detection because it requires Stage 1 to predict
monitor/suppress. When Stage 1 correctly escalates a suspicious data_exfiltration alert,
R4 doesn't fire — but the alert is already going to a human. This is by design.

**Referral problem decomposition (EXP-REFER-COVERAGE):**

| Category | % of referrals | Mechanism |
|---|---|---|
| Rule-expressible (R1-R6) | 65.5% | Policy rules (Day 1) |
| Context-dependent (R7, R9) | 13.8% | Graph queries + calendar |
| Emergent (R8, R10) | 20.7% | Override learning (v6.5+) |

**Key design properties:**

P-REF-1: Referral never modifies ProfileScorer scoring or centroids.
P-REF-2: Missing context data → rule doesn't fire (safe degradation, not false positive).
P-REF-3: Rules are inspectable, auditable, EU AI Act Art. 14 compliant.
P-REF-4: Customer configures rules during onboarding (thresholds overridable).
P-REF-5: OverrideDetector activates on data volume, not calendar (50+ positives).

---

## 13. Equation Traceability Matrix

> **v10.4 additions:** Eq. GAMMA-THEOREM, GAMMA-THRESH, GAMMA-P_D, GAMMA-DIST added.
> See `gae/convergence.py` for implementations.

| Equation | Paper Section | File | Function | Version | Status |
|---|---|---|---|---|---|
| **Eq. 4-final** | §3 | `gae/profile_scorer.py` | `ProfileScorer.score()` | v5.0 | ✅ LIVE |
| **Eq. 4b-final** | §3 | `gae/profile_scorer.py` | `ProfileScorer.update()` | v5.0 | ✅ LIVE |
| Eq. 4 (published) | §3 original | `gae/scoring.py` | `score_alert()` | v4.1 | ⚠️ DEPRECATED (v6.0) |
| Eq. 4b (published) | §3 original | `gae/learning.py` | `LearningState.update()` | v4.1 | ⚠️ DEPRECATED (v6.0) |
| Eq. 4c (decay) | §3 | `gae/learning.py` | `apply_decay()` | v4.1 | ✅ Preserved |
| **Eq. 4-synthesis** | §5 (PROPOSAL) | `gae/profile_scorer.py` | `ProfileScorer.score(synthesis=σ)` | v6.0 | 🔵 GATED (GATE-M) |
| Eq. 5 (embeddings) | §4 | `gae/embeddings.py` | `embed_graph()` | v5.5 | 📋 Designed |
| Eq. 6 (cross-attention) | §4 | `gae/bridge.py` | `GraphAttentionBridge.enrich()` | v7.0 | 📋 Designed |
| Eq. 8a-8c (discovery) | §4 | `gae/discovery_engine.py` | `discover()` | v8.0 | 📋 Designed |
| ECE computation | - | `gae/evaluation.py` | `compute_ece()` | v5.0 | ✅ LIVE |
| **ReferralEngine** | §12.3 | `gae/referral.py` | `ReferralEngine.evaluate()` | v6.0 | ✅ LIVE [NEW v10.1] |
| **Eq. GAMMA-THEOREM** | §3.2 (math_synopsis_v13) | `gae/convergence.py` | `gamma_threshold()` | v10.4 | ✅ IMPLEMENTED |
| **Eq. GAMMA-THRESH** | §3.2 | `gae/convergence.py` | `gamma_threshold()` | v10.4 | ✅ IMPLEMENTED |
| **Eq. GAMMA-P_D** | §3.2 | `gae/convergence.py` | `phase2_effective_threshold()` | v10.4 | ✅ IMPLEMENTED |
| **Eq. GAMMA-DIST** | §3.2 | `gae/convergence.py` | `centroid_distance_to_canonical()` | v10.4 | ✅ IMPLEMENTED |
| **OverrideDetector** | §12.3 | `gae/referral.py` | `OverrideDetector.predict()` | v6.5 | 🔵 STUB [NEW v10.1] |

---

## 14. Design Decisions Log

### 14.1 Original Design Decisions (v4.1)

*(See prior versions for full record. Summary:)*
NumPy-only (24 multiply-adds don't need PyTorch), three-repo architecture (enforce P12), FactorComputer Protocol in GAE (abstract interface), f(t) stored in Decision nodes (R4).

### 14.2 Experiment-Driven Decisions (v8+)

| Decision | Evidence | Impact |
|---|---|---|
| L2 as cold-start kernel | EXP-C1: 36.89pp gap | Settled |
| **DiagonalKernel as v6.0 default** | **V-MV-KERNEL: +13.2pp SOC, +6.8pp S2P** | **Settled** |
| **noise_ratio > 1.5 → diagonal** | **Selector Ask 2: 4/4 correct** | **Settled (one parameter, no ρ)** |
| **ShrinkageKernel deprioritized** | **D2/D3: off-diagonal <1pp** | **Deferred to v7.0** |
| **Factor mask deprecated** | **V-HC-CONFIG: mask hurt Day 1 by 6pp** | **DiagonalKernel supersedes** |
| **Asymmetric η (P0)** | **9 personas: 13-27pp degradation. 24 personas: validated.** | **η_override=0.01 permanent** |
| **AMBER auto-pause** | **Three-judge consensus** | **Conservation AMBER → freeze** |
| **DiagonalKernel gradient fix (GAE-GRADIENT-001)** | **V-ENRICHMENT-NEGATIVE false UNSAFE at GAE <0.7.7 — gradient W*(f-mu) was wrong. W/W.max()*(f-mu) confirmed correct.** | **W/W.max() is the only valid gradient. Gradient bug skepticism protocol added to governing principles.** |
| **η_neg ≥ 1.0 guard** | **RuntimeError at η_neg=1.0 produced FORBIDDEN ECE=0.49 result. ValueError added.** | **CalibrationProfile raises ValueError on η_neg ≥ 1.0. Test covers boundary at exactly 1.0.** |
| **W2 flywheel validated** | **V-TRIGGERED-EVOLUTION full: +10.13pp (CI=[+5.4,+14.9]pp, p=0.0002, N=30). Δ_dissimilar=0.00pp.** | **CLAIM-W2 UNCONDITIONAL. PatternHistoryFactorComputer in SOC. Graph compounds independently of centroids.** |
| **Flywheel Health Monitor** | **V-OLS-DETECT: 0% miss rate, p90≥50d lead time (adversarial + complacency, N=30/condition).** | **CLAIM-OLS-01 UNCONDITIONAL. CUSUM h=5.0 OLS scale. Plateau-snapshot baseline. Conservation law 4th role.** |
| **Var(q) gating: PERMANENT HARD STOP** | **V-MV-CONSERVATION-BIMODAL: Bernoulli mixture theorem. Var(Q_bimodal)=p̄(1-p̄) — identical to uniform at same mean.** | **Var(q) is LOGGED ONLY. No gating, no product claim, no further iterations.** |
| **Convergence calendar** | **V-MV-CONVERGENCE v2: MAE=1.55d. V has NO causal effect on N_half. q̄ dominant (coeff -3.28).** | **CLAIM-CONV-01 UNCONDITIONAL. Calendar shows decisions AND days separately.** |
| **Two-tier poisoning at A=4** | **EXP-S2-REPRO-A4 series: σ-perturbation mean 0.850pp at 20% (gate ≤1.0pp). Label poisoning mean 3.20pp at 20% adversarial (gate ≤5pp).** | **CLAIM-SK-01 + CLAIM-LP-01. Prior single gate (≤0.20pp) was A=5 geometry — A=4 recalibration required.** |
| **Enrichment safety (CLAIM-65)** | **V-ENRICHMENT-NEGATIVE v2 (GAE 0.7.8): SAFE. CI upper 1.1pp at adversarial multi-factor contamination. Mechanism: W=1/σ² auto-downweights bad enrichment.** | **CLAIM-65 UNCONDITIONAL. Source trust gate (0A-6) moved to Post-MVP enrichment provenance logging.** |
| Dot product warned | EXP-C1: 61% on [0,1] | KernelType.DOT emits warning |
| Profile centroids | EXP-C1: 97.89% zero-learning | W matrix eliminated |
| G eliminated | EXP-A: +0.01pp | No GatingMatrix anywhere |
| τ = 0.1 | V3B: ECE 0.19 → 0.036 | Hard default, never override |
| Clipping required | V2: escape at dec 6-12 | All updates clip to [0,1] |
| τ_modifier rejected | OP: ECE +0.138 | No τ_modifier field, ever |
| Loop 2/4 firewall | EXP-S3: Frobenius 0.0028 | update() has NO synthesis param |
| Checkpoint required | EXP-OP2: 35% never recover | TD-033: checkpoint every 50 |
| LayerNorm required | V1B: 2.9M× explosion | Level 2: LayerNorm non-negotiable |
| γ is estimated | Not measured | EXP-G1 before public claim |
| Realistic ≠ synthetic | math_synopsis_v6 50-seed | Two accuracy regimes documented |

### 14.3 Open-Source Strategy Decisions (v9 NEW)

| Decision | Rationale | Implementation |
|---|---|---|
| Apache 2.0 (not MIT or GPL) | Compatible with enterprise use; preserves attribution; permits relicensing in products | `LICENSE` file at repo root |
| Open math, proprietary domain expertise | The engine is a commodity; the SOC DomainConfig is the moat | GAE exports Protocol, not SOC implementation |
| Users guide as first-class deliverable | A library without good docs is unused | `docs/users_guide.md` shipped in v5.0 |
| Minimal example in repo | "Does this work?" answered in 3 minutes | `examples/minimal_domain/` runnable standalone |
| No SOC-specific code in GAE | Prevents leakage; keeps GAE genuinely domain-agnostic | P12 enforced by tests |
| Published math as credibility | Open-source adoption requires trust; equations are peer-reviewable | Blog + paper link in README |

---

## 15. Open-Source Strategy

### 15.1 Why Open-Source GAE

The strategic argument for open-sourcing the engine (not the domain expertise) has three parts:

**Trust.** The competitive moat is the graph — the accumulated institutional knowledge in profile centroids. Opening the math does not give away the moat. A competitor who forks GAE and deploys it at their customer starts at zero centroids. The customer's 10,000 verified decisions are not in the fork. Trust comes from the math being visible and peer-reviewable.

**Ecosystem.** If GAE becomes the standard library for compounding-intelligence systems, every new domain (procurement, AML, ITSM, clinical trials) that uses GAE is potential infrastructure for a Dakshineshwari domain copilot. Open-source is the distribution mechanism.

**Recruiting and credibility.** Strong technical talent evaluates open-source repos. A clean, tested, well-documented Apache 2.0 library with peer-reviewed math is a better hiring signal than a closed-source demo.

### 15.2 What a Successful Open-Source Release Requires

A GitHub repo is not an open-source project. An open-source project has the following (current status in brackets):

**Code quality:**
- [✅] Apache 2.0 LICENSE at repo root
- [✅] 517 tests, all passing
- [✅] NumPy-only — zero external dependency to install
- [✅] Type hints throughout
- [⚠️] Docstrings complete on public classes (need audit — v5.5)
- [📋] `py.typed` marker file (PEP 561 — v5.5)

**Documentation:**
- [✅] `docs/users_guide.md` — 289 lines, 8 sections, runnable code
- [✅] `examples/minimal_domain/` — canonical new domain example
- [📋] `README.md` — THE landing page. Currently minimal. Must be substantially rewritten for v5.5. (See §15.3)
- [📋] `docs/equations.md` — Eq. 4-final through Eq. 4b-final with links to paper (update for v5.5)
- [📋] `CHANGELOG.md` — v0.1.0 → v0.5.0 with breaking changes noted
- [📋] `docs/CONTRIBUTING.md` — how to submit a PR, coding standards, test requirements

**Packaging:**
- [✅] `pyproject.toml` with version 0.5.0
- [📋] PyPI publish: `pip install graph-attention-engine` (target: v5.5 release)
- [📋] GitHub Release v0.5.0 with notes (target: immediately after v5.5 open-source prep)
- [📋] `pip install graph-attention-engine` installs correctly and `from gae import ProfileScorer` works

**CI/CD:**
- [📋] GitHub Actions: `pytest` on push (every PR runs the full test suite)
- [📋] GitHub Actions: lint (ruff or flake8) on push
- [📋] GitHub Actions: type-check (mypy --strict on public API) on push
- [📋] Badge: [![Tests](passing)](actions) [![License: Apache 2.0](...)] in README

**Community:**
- [📋] GitHub Issues enabled with labeled templates: Bug Report, Feature Request, New Domain
- [📋] GitHub Discussions enabled (Q&A tab)
- [📋] `SECURITY.md` — how to report vulnerabilities (required for enterprise adoption)
- [📋] `CODE_OF_CONDUCT.md` — standard Contributor Covenant

**Discoverability:**
- [📋] GitHub Topics: `machine-learning`, `knowledge-graph`, `decision-intelligence`, `numpy`, `enterprise-ai`, `online-learning`
- [📋] PyPI classifiers: Python version, license, topic
- [📋] Link from math blog to repo
- [📋] Link from compounding intelligence blog to repo

### 15.3 The README Problem

The README is the most important document in the repo. It is the first thing every potential contributor, user, and evaluator reads. It determines whether they continue or close the tab.

**Current state:** Minimal. Does not tell the story.

**Required sections for v5.5 README:**

```markdown
# Graph Attention Engine

> Distance-kernel attention for enterprise decision intelligence.
> Learn from every verified decision. Profiles are the model. 
> Math is open. The accumulated judgment is yours.

## What This Is

A scoring library that implements compounding intelligence —
AI systems that get measurably better at their job over time.

Core equation: P(a|f,c) = softmax(−‖f − μ[c,a,:]‖² / τ)
- f: your situation encoded as a factor vector
- μ: domain expert knowledge compiled into profile centroids  
- c: situational category (Mixture-of-Experts routing)
- τ: temperature (validated at 0.1, ECE=0.036)

## 30-Second Quick Start

pip install graph-attention-engine

from gae import ProfileScorer, build_profile_scorer
import numpy as np

# 3 categories × 2 actions × 3 factors
mu = np.array([...])  # your domain expertise
scorer = build_profile_scorer(mu, actions=["approve", "escalate"])

f = np.array([0.8, 0.3, 0.6])  # this situation
result = scorer.score(f, category_index=0)
print(result.selected_action, result.confidence)

scorer.update(f, 0, result.action_index, correct=True)  # learn

## Why Not Dot Product? (The Key Insight)

[The 36.89pp gap chart — expC1_comparison_waterfall.png]
[1 paragraph: proximity vs alignment]

## Validated Numbers

| Metric | Value | Condition |
|---|---|---|
| Zero-learning accuracy | 97.89% | Synthetic centroidal, GT profiles |
| With-learning accuracy | 98.2% | Synthetic, warm-start |
| Realistic static accuracy | 71.7% | 50-seed realistic distributions |
| Realistic at 1k decisions | 78.9% | 50-seed realistic distributions |
| Calibration ECE | 0.036 | τ=0.1 (use this, never 0.25) |
| Domain scaling exponent | n^2.11 | Simulation, R²=0.9999 |

See math/blog for derivation. See experiments repo for raw data.

## Building a New Domain

[Link to docs/users_guide.md §6 — 6-step onboarding]

## Architecture

[The five-layer diagram]
[Link to the math blog]
[Link to the experiments repo]

## License

Apache 2.0 — use it in your products. The math is open.
The competitive moat is your accumulated graph, not the equations.
```

**Key principle for the README:** Serve three audiences simultaneously — (1) a developer evaluating whether to use GAE (30-second quick start), (2) a ML practitioner who wants to understand the approach (the dot product section + validated numbers), (3) a technical evaluator doing due diligence (architecture + experiments link).

### 15.4 The Helpdesk Example Problem

The current `examples/minimal_domain/` uses a "helpdesk" domain. This is technically correct but misses the opportunity to demonstrate a compelling use case. Consider adding a second example domain for v5.5:

- **`examples/procurement_approval/`** — S2P use case proof. 3 categories (price_variance, vendor_risk, contract_compliance), 3 actions (approve, escalate, flag), 4 factors. Demonstrates that GAE generalizes beyond security operations without any changes to the engine.

This example serves three purposes: (1) proves the platform claim, (2) gives procurement teams an on-ramp, (3) demonstrates that DomainConfig design is accessible to non-ML engineers.

### 15.5 Version Naming Strategy

| Version | PyPI Tag | GitHub Tag | Theme |
|---|---|---|---|
| v0.5.0 | 0.5.0 | v0.5.0 | ProfileScorer + full evaluation API (tagged) |
| **v0.7.0** | **0.7.0** | **v0.7.0** | **Kernel architecture + CovarianceEstimator + KernelSelector + open-source release prep** |
| v0.8.0 | 0.8.0 | v0.8.0 | GainScheduler + Fisher calendar + embeddings + open-source test suite (~900 tests, §15.6) + **gae/experiments/** (§10.15): exp_c1.py reproduces 36.89pp; oracle_separation/ demonstrates γ theorem; domain examples (medical, supply chain, financial) |
| v1.0.0 | 1.0.0 | v1.0.0 | GraphAttentionBridge (Level 2) — first major release |
| v2.0.0 | 2.0.0 | v2.0.0 | Cross-domain discovery (Level 3) |

**Why 0.x until GraphAttentionBridge?** The API is stable within 0.x (no breaking changes without a minor version bump) but the 1.0.0 signal means "this is the long-term supported API." That signal should coincide with the Level 2 implementation, which is when external developers can build the cross-domain compounding story themselves.

---

### 15.6 Open-Source Test Strategy [NEW v10.5]

**Context:** GAE is designed to be used as a domain-agnostic engine — the "transformers
for compounding intelligence." Current tests (527) verify correctness at nominal values
for the SOC domain. Open-source requires tests that verify INVARIANTS across the full
parameter space, for ANY domain, at boundary conditions not yet tested.

**Target: ~900 tests before open-source release (v0.8.0).** The gap is real but bounded.
~370 new tests across 11 categories plus infrastructure rework.

---

#### Category 1 — Numeric Stability (~5 now → 30 target, P0)

Most dangerous silent failure mode for open-source users: extreme inputs producing NaN,
overflow, or silent clipping violations.

**Test inventory:**
- Factor boundary: f = [0,0,...,0], f = [1,1,...,1], f alternating [0,1,0,1]
- Centroid boundary: μ already at 0 or 1, then updated — stays clipped
- 1000 incorrect updates: μ stays in [0,1] at all corners
- Temperature extremes: τ = 0.001 (near-zero), τ = 10.0 (near-uniform)
- τ invariant argmax: argmax same across all τ values
- DiagonalKernel weight extremes: W[i] ≈ 1e-10 (nearly masked), W = ones (must equal L2), W = [1000,1,1,1,1,1]
- softmax no overflow, no underflow, no NaN at extreme distances
- gradient norm bounded: no explosion on any input combination

---

#### Category 2 — Tensor Shape Variants (~3 now → 25 target, P0)

Open-source users will build every conceivable domain shape. Must prove GAE works for all.

**Test inventory:**
- Minimal: (1,1,1), (1,2,3), (2,2,2)
- Large: (20,10,50), (100,5,8), (6,4,100)
- Asymmetric: (12,2,6), (2,8,6)
- Batch scoring: shape (N,d) — verify batch == N individual calls
- Dtype variants: float32, float64; output dtype matches input dtype

---

#### Category 3 — API Contract / Backward Compatibility (~0 now → 25 target, P0)

**Critical.** Breaking an API field breaks every downstream user. These tests are the
regression gate for every future release.

**ScoringResult fields — all must always be present:**
action_probabilities, selected_action, confidence, distances, factor_vector, probabilities.

**ProfileScorer constructor — all kwargs optional except mu + actions.**

**Serialization round-trip:** score(f) before checkpoint == score(f) after restore.

**CentroidUpdate contract:** returns CentroidUpdate or None; delta_norm always positive.

**freeze/unfreeze:** frozen → update() returns None; score() unchanged; unfreeze → resumes.

**gae/convergence.py API contract (NEW v10.5):**
- `centroid_distance_to_canonical(mu, canonical)` — non-negative, zero for identical,
  symmetric (dist(A,B) == dist(B,A)), scales correctly with tensor size
- `gamma_threshold(alpha_cat, delta_norm, theta)` — returns ≈0.128 for production values;
  monotone in all three parameters; raises ValueError on invalid inputs
- `phase2_effective_threshold(alpha_cat, theta)` — returns ≈0.55 for standard values;
  raises when formula produces negative result (theta ≤ 1 - alpha_cat)

---

#### Category 4 — Domain-Agnostic Correctness (~0 now → 25 target, P1)

**This is the key open-source differentiator.** Tests must prove GAE works for domains
we've never built.

**Four test domains (all distinct from SOC):**
```
Medical:       diagnose / treat / refer / monitor (4 categories, 4 actions, 5 factors)
Supply chain:  approve / hold / expedite / cancel (3 categories, 4 actions, 6 factors)
Financial:     approve / review / reject / escalate (5 categories, 4 actions, 4 factors)
HR screening:  hire / waitlist / reject / reinterview (4 categories, 4 actions, 5 factors)
```

**For each domain, verify:**
1. Score changes meaningfully as factor values change (non-constant)
2. Centroid updates in the correct direction (toward f for correct action)
3. After 200 updates, accuracy > 70% (above random for any domain)
4. IKS increases from bootstrap to converged state
5. γ_threshold(alpha_cat, delta_norm) returns valid float — convergence.py works

**Cross-domain consistency:**
- Swapping action name strings doesn't change math (geometry is label-agnostic)
- Permuting categories doesn't change per-category scores

---

#### Category 5 — Learning Dynamics (~10 now → 40 target, P1)

**Asymmetric η must be proven, not asserted:**
- |Δμ_override| < |Δμ_confirm| for every input combination
- Exactly 5× attenuation at default settings (η_confirm=0.05, η_override=0.01)
- Ratio preserved after 1000 updates

**Convergence properties:**
- centroid_distance_to_canonical decreases monotonically over 200 correct updates
- centroid_distance does NOT decrease monotonically over 200 random updates
- N_half gap detection fires when accuracy crosses θ before centroid distance drops 20%

**Count-based decay:** update magnitude decreases over time, asymptotically (not to zero).

**Multi-category isolation:** update in category 0 does not affect category 1.

---

#### Category 6 — Kernel Protocol Compliance (~28 now → 50 target, P1)

**Every user-defined kernel must satisfy the protocol.** A compliance checker:

```python
def assert_kernel_protocol_compliance(kernel):
    # distance is non-negative, zero for identical vectors, symmetric
    # gradient points from mu toward f (for correct update direction)
    # gradient magnitude bounded — no explosion
    # gradient is zero when f == mu exactly
```

Run on: L2Kernel, DiagonalKernel. And any user-defined kernel fails loudly if protocol
violated.

**DiagonalKernel specific mathematical properties:**
- Reduces to L2 when all weights are uniform — **must be exact, not approximate**
- Zero weight removes factor influence entirely: W[i]=0 → factor i irrelevant to score
- `gradient_formula_exact`: W/W.max()*(f-mu), not W*(f-mu) — sign of this is a footgun
- `raw_weights` are true 1/σ² (not pre-normalized) — regression test from v0.7.20 fix
- `weights` (pre-normalized [0,1]) for scoring; `raw_weights` for η_eff calculations

---

#### Category 7 — Synthetic Data Range Tests (~5 now → 45 target, P1)

**Extended from original analysis to cover oracle separation parameter space (v10.5 addition):**

```python
# Original SOC parameter space (24 parametrized tests)
@pytest.mark.parametrize("sigma", [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
def test_learning_at_sigma(sigma): ...

@pytest.mark.parametrize("q_bar", [0.55, 0.60, 0.70, 0.80, 0.90, 0.95])
def test_learning_at_analyst_quality(q_bar): ...

@pytest.mark.parametrize("n_decisions", [10, 50, 100, 500, 1000, 5000])
def test_learning_at_volume(n_decisions): ...

# Oracle separation parameter space (NEW — 21 additional tests)
@pytest.mark.parametrize("epsilon_firm", [0.05, 0.10, 0.128, 0.15, 0.20, 0.30, 0.40])
def test_gamma_theorem_at_epsilon_firm(epsilon_firm):
    # Below 0.128: gamma < 1 (theorem predicts < 1)
    # Above 0.128: gamma > 1 (theorem predicts > 1)
    # At 0.128: gamma ≈ 1 (boundary case)

@pytest.mark.parametrize("alpha_cat", [1/6, 2/6, 3/6, 4/6])
def test_rolling_window_shortcut_at_alpha(alpha_cat):
    # p_d* = (theta - (1-alpha_cat)) / alpha_cat decreases as alpha increases

@pytest.mark.parametrize("noise_ratio", [1.0, 1.5, 2.0, 3.0, 5.0])
def test_diagonal_advantage_at_noise_ratio(noise_ratio): ...
```

**Torture test (unchanged from analysis — correct as specified):**
σ=0.30, q̄=0.60, N=2000. Asymmetric η prevents corruption. μ stays in [0,1].
Conservation law fires AMBER before damage accumulates. **Add: centroid_distance_to_canonical
decreases despite hostile conditions** — validates that the reliable metric holds even here.

---

#### Category 8 — Oracle and Evaluation Pipeline (~15 now → 40 target, P2)

**EXP-G1 related tests now included (from gae/synthetic.py):**
- centroid_distance_to_canonical decreases under correct learning
- OracleSeparationExperiment binary prediction: ε=0.05 → γ<1; ε=0.20 → γ>1
- GammaResult.n_half_gap_detected fires when N_half crosses θ before centroid plateau

---

#### Category 9 — Referral Engine (~31 now → 50 target, P2)

All rules return (False, {}) on empty context, not exception.
Referral VETO overrides high-confidence suppress — **key safety property, must be tested**.
Referral does not modify scoring result (P-REF-1 invariant).

---

#### Category 10 — `gae/convergence.py` Tests (0 now → 20 target, P0) [NEW v10.5]

**This is a new public API. Every function must have contract tests.**

```python
def test_centroid_distance_zero_for_identical():
def test_centroid_distance_nonnegative_always():
def test_centroid_distance_symmetric():
def test_centroid_distance_scales_with_tensor_size():
def test_centroid_distance_decreases_monotonically_under_learning():  # KEY claim
def test_gamma_threshold_production_values():    # ≈0.128 for (2/6, 0.25, 0.85)
def test_gamma_threshold_monotone_in_alpha():    # more disruption → higher threshold
def test_gamma_threshold_monotone_in_delta():    # bigger disruption → higher threshold
def test_gamma_threshold_invalid_inputs_raise():
def test_phase2_effective_threshold_standard():  # ≈0.55 for (2/6, 0.85)
def test_phase2_effective_threshold_invalid_raises():  # theta ≤ 1-alpha_cat
def test_convergence_trace_nhalf_gap_detected():
def test_convergence_trace_nhalf_gap_not_detected_when_centroid_converges_first():
def test_convergence_trace_full_construction():
```

---

#### Category 11 — `gae/synthetic.py` Tests (0 now → 20 target, P0) [NEW v10.5]

```python
def test_factor_vector_sampler_range_in_01():
def test_factor_vector_sampler_regime_differentiation():
def test_oracle_separation_phase1_converges():
def test_oracle_separation_phase2_converges():
def test_gamma_result_nhalf_gap_fires():
def test_canonical_centroid_apply_disruption_magnitude():  # ‖GT₂ − GT₁‖ = ‖Δ‖ exactly
def test_canonical_centroid_only_disrupted_categories_shift():
def test_binary_prediction_below_threshold():   # ε=0.05 → γ < 1
def test_binary_prediction_above_threshold():   # ε=0.20 → γ > 1
def test_oracle_separation_model_independence(): # same result across LLM models
```

---

#### Category 12 — Thread Safety (0 now → 10 target, P1) [NEW v10.5]

**This is the most dangerous silent failure mode for production open-source use.**
Concurrent `score()` calls are safe (reads μ). Concurrent `update()` calls must be
serialized or explicitly documented as unsafe.

```python
def test_concurrent_scoring_produces_consistent_results():
    # 100 threads calling score() simultaneously — all produce identical results
def test_concurrent_updates_are_safe_or_documented_unsafe():
    # Either: concurrent updates don't corrupt μ (locking)
    # Or: concurrent updates raise a clear ConcurrentUpdateError
def test_score_during_update_is_consistent():
    # score() called while update() is running — result is pre- or post-update, not corrupt
```

---

#### Test Infrastructure — Prerequisite (do before Categories 4–12)

The current test suite at 527 tests is at the limit of manageable flat organization.
Adding ~370 more tests without restructuring creates an unmaintainable test suite.

**Required before writing new tests:**

```
tests/
  conftest.py            # shared fixtures: standard domains, nominal scorer instances
  unit/
    test_scoring.py      # Category 1 + 2
    test_learning.py     # Category 5
    test_kernels.py      # Category 6
    test_api_contract.py # Category 3
    test_convergence.py  # Category 10
    test_synthetic.py    # Category 11
  parametric/
    test_ranges.py       # Category 7 — all parametrized tests together
    test_domains.py      # Category 4 — all four domain tests
  integration/
    test_oracle_eval.py  # Category 8
    test_referral.py     # Category 9
    test_thread_safety.py # Category 12
  benchmarks/
    bench_score.py       # score() latency at nominal + large tensor
    bench_update.py      # update() latency
    bench_memory.py      # memory footprint
```

**Shared fixtures (conftest.py):**
```python
@pytest.fixture
def nominal_soc_scorer(): ...  # 6×4×6 tensor, standard calibration
@pytest.fixture
def medical_scorer(): ...      # 4×4×5 tensor, medical domain
@pytest.fixture
def gt_oracle(nominal_scorer): ...
@pytest.fixture
def standard_factor_vector(): ...  # d=6, mid-range values
```

---

#### Summary Table

| Category | Current | Target | Priority |
|---|---|---|---|
| 1 — Numeric stability | ~5 | 30 | **P0** |
| 2 — Tensor shape variants | ~3 | 25 | **P0** |
| 3 — API contract | ~0 | 25 | **P0** |
| 10 — gae/convergence.py | 0 | 20 | **P0** |
| 11 — gae/synthetic.py | 0 | 20 | **P0** |
| 4 — Domain-agnostic | ~0 | 25 | P1 |
| 5 — Learning dynamics | ~10 | 40 | P1 |
| 6 — Kernel protocol | ~28 | 50 | P1 |
| 7 — Synthetic data range | ~5 | 45 | P1 |
| 12 — Thread safety | 0 | 10 | P1 |
| 8 — Oracle/evaluation | ~15 | 40 | P2 |
| 9 — Referral engine | ~31 | 50 | P2 |
| **Total new** | | **~370** | |
| + existing 527 | | | |
| **Grand total target** | | **~900** | |

**`gae/experiments/` development** runs in parallel with Block 3.7a test suite work.
The exp_c1.py and exp_e2.py scripts are essentially free — they're pure GAE public API
calls that already work. The oracle_separation/ examples validate gae/synthetic.py.
The domain examples validate Category 4 test domains. All three are produced together.

**Test infrastructure restructure** is prerequisite to Categories 4–12.
Write infrastructure first, then P0 categories, then P1, then P2.

---

## 16. Production Constraints (All Hard — Not Configurable)

These are constraints derived from experiments. They are not guidelines. Code that violates them is wrong.

| Constraint | Value | Evidence | What Fails Without It |
|---|---|---|---|
| τ (temperature) | Always 0.1 | V3B: ECE 0.19→0.036 | Miscalibrated confidence → wrong auto-approve decisions |
| Centroid clipping | [0.0, 1.0] | V2: escape at dec 6-12 | Adversarial escape; 4,608× norms by dec 200 |
| update() has no synthesis param | Permanent | EXP-S3: Frobenius 0.0028 | μ contaminated by external intelligence |
| LayerNorm in Level 2 | Mandatory | V1B: 2.9M× explosion | Embedding norms blow up after 5 sweeps |
| λ ≤ 0.6 for synthesis | Hard ceiling | EXP-OP2: harmful above 0.6 | Operator damage scales 5:1 vs benefit |
| Checkpoint every 50 (synthesis) | Required | EXP-OP2: 35% never recover | Irreversible centroid damage after bad synthesis operator |
| Factor vectors in [0.0, 1.0] | Hard contract | EXP-C1: L2 validated on [0,1] | L2 distances become meaningless at other scales |
| τ_modifier | REJECTED | OP series: ECE +0.138 | Never add this field; test that it doesn't exist |
| **η_override** | **0.01** | **9-persona stress test: 13-27pp degradation without** | **Centroid corruption from low-quality overrides** |
| **η_confirm** | **0.05** | **P0 fix validated across 24 personas** | **Under-learning from confirm path if reduced** |
| **θ_min** | **0.467** | **η=0.05, N_half=14, T_max=21 days** | **Conservation law floor too lenient at 0.434** |
| **AMBER auto-pause** | **Freeze on AMBER/RED** | **Three-judge consensus** | **Detection-without-response gap** |
| **DiagonalKernel gradient** | **W/W.max()*(f-mu) ONLY** | **GAE-GRADIENT-001 fix v0.7.7** | **W*(f-mu) corrupts learning direction — high-W factors dominate by magnitude not signal** |
| **η_neg guard** | **ValueError on η_neg ≥ 1.0** | **GAE 0.7.8 — η_neg=1.0 produced ECE=0.49 (FORBIDDEN)** | **η_neg=1.0 inverts the penalty signal — pushes centroid toward wrong action** |
| **DiagonalKernel for noise_ratio > 1.5** | **Default** | **V-MV-KERNEL: +13pp SOC, +7pp S2P** | **13-22pp accuracy loss on heterogeneous noise** |
| **ShrinkageKernel** | **NOT shipped at v6.0** | **D2/D3: off-diagonal <1pp** | **Complexity without benefit** |

---

## 17. What's Built vs What's Next

### 17.1 v5.0 — COMPLETE ✅ (246 tests including post-tag WIRING-1)

**GAE Phase 1 (ProfileScorer integration):**
- `gae/profile_scorer.py` — ProfileScorer, ScoringResult, KernelType (GAE-PROF-1)
- `gae/scoring.py` — deprecated, forwards to ProfileScorer (GAE-PROF-2)
- `gae/learning.py` — delegates to ProfileScorer.update() (GAE-PROF-3)
- Orchestrator wired: SituationAnalyzer → c → μ[c,:,:] (GAE-PROF-4)
- Oracle integration: OracleProvider protocol (GAE-ORACLE-1)

**GAE Phase 6 (Evaluation + API):**
- `gae/evaluation.py` — EvaluationScenario, run_evaluation(), EvaluationReport (GAE-EVAL-1)
- `gae/judgment.py` — compute_judgment(), JudgmentResult, confidence tiers (GAE-JUDG-1)
- `gae/ablation.py` — run_ablation(), AblationReport (GAE-ABL-1)
- `gae/__init__.py` — full public API surface, all symbols (GAE-ENG-1)
- `examples/minimal_domain/` — canonical onboarding example (GAE-ENG-2)
- `docs/users_guide.md` — 289 lines, 8 sections, runnable code (GAE-DOC-1)

**Test count progression (v5.0):**
```
v4.1 tagged:     177 tests
v4.5 preamble:   187 tests  (+CalibrationProfile)
v5.0-alpha:      211 tests  (+ProfileScorer)
v5.0 Phase 6:    235 tests  (+Evaluation/Judgment/Ablation)
v5.0 TAGGED:     243 tests  (+SOC integration fixes)
v5.0 post-tag:   246 tests  (+WIRING-1: CentroidUpdate, freeze/unfreeze)
```

### 17.1a v6.0 — COMPLETE ✅ (517 tests, v0.7.17)

*(See §17.1b below for v0.7.18–v0.7.20 additions: 517→527 tests)*

**Kernel architecture (settled March 21, 2026):**
- `gae/kernels.py` — ScoringKernel protocol, L2Kernel, DiagonalKernel (28 tests)
- `gae/covariance.py` — CovarianceEstimator with Ledoit-Wolf + exponential decay (23 tests)
- `gae/kernel_selector.py` — KernelSelector: Phase 2/3/4 empirical kernel selection (46 tests)
- `gae/referral.py` — ReferralEngine, ReferralRule protocol, OverrideDetector stub (31 tests)
- ProfileScorer: kernel parameter, factor_mask, eta_override, auto_pause_on_amber
- calibration.py: compute_factor_mask, mask_to_array, compute_eta_override
- convergence.py v2: EPSILON_DEFAULT=0.10, safety_factor=2.0
- fisher.py: estimate_fisher_information, predict_n_half, enrichment_multiplier

**Phase 1 session additions (March 25, 2026):**
- `gae/convergence.py` — OLSMonitor: CUSUM on OLS, h=5.0 (OLS scale), plateau-snapshot baseline (CLAIM-OLS-01 validated)
- `gae/convergence.py` — VarQMonitor: logged observability metric, NO gating logic (Bernoulli mixture theorem hard stop)
- `gae/enrichment_advisor.py` — enrichment recommendation engine, 5 deployment profiles validated
- DiagonalKernel gradient fix: W/W.max()*(f-mu) — GAE-GRADIENT-001, v0.7.7
- η_neg guard: ValueError on η_neg ≥ 1.0 — v0.7.8

**Key experimental validation:**
- V-MV-KERNEL factorial: 390 cells. DiagonalKernel +13.2pp SOC, +6.8pp S2P.
- V-HC-CONFIG with DiagonalKernel: healthcare rescued at σ=0.22 (+3.7pp vs L2 +0.3pp).
- EXP-REFER-LAYERED: Rules R1-R7 = 72.7% DR, 12% FPR. Confidence gate rejected for referral.
- V-TRIGGERED-EVOLUTION full: +10.13pp (p=0.0002, N=30) — CLAIM-W2 UNCONDITIONAL.
- V-OLS-DETECT: 0% miss, p90≥50d lead time — CLAIM-OLS-01 UNCONDITIONAL.
- V-ENRICHMENT-NEGATIVE v2 (GAE 0.7.8): SAFE, CI upper 1.1pp — CLAIM-65 UNCONDITIONAL.
- V-MV-CONSERVATION-BIMODAL: Bernoulli mixture theorem — Var(q) gating PERMANENT HARD STOP.

**Test count progression (v6.0):**
```
v5.0 post-tag:   246 tests
+WIRING-2:       251 tests  (+ProfileScorer bug fix, update() dual push/pull)
+Block 5A:       291 tests  (+convergence v2, calibration extensions)
+A=4 migration:  309 tests  (+A=4 actions, eta_override)
+Factor mask:    324 tests  (+compute_factor_mask, mask_to_array, scorer mask integration)
+Kernels:        383 tests  (+L2Kernel, DiagonalKernel, CovarianceEstimator, kernel integration)
+AMBER pause:    391 tests  (+auto_pause_on_amber, conservation status)
+KernelSelector: 437 tests  (+KernelSelector, KernelRecommendation, Phase 2/3/4)
+Rolling window: 447 tests  (+rolling 100-window comparison)
+ReferralEngine: 478 tests  (+ReferralEngine, ReferralRule, OverrideDetector stub)
+Gradient fix:   499 tests  (+GAE-GRADIENT-001 fix tests, enrichment_advisor validation)
+η_neg guard:    507 tests  (+η_neg ≥ 1.0 ValueError, boundary at exactly 1.0)
+OLSMonitor:     517 tests  (+OLSMonitor CUSUM, VarQMonitor, plateau-snapshot)
```

### 17.1b v0.7.18–v0.7.20 — COMPLETE ✅ (527 tests) [NEW v10.3]

**KERNELSEL-001 — KernelSelector tiebreaker (v0.7.18):**
- `gae/kernel_selector.py` — tiebreaker logic when Phase 2 rule and Phase 4 empirical disagree.
  Phase 4 (empirical) always overrides Phase 2 (rule) on genuine disagreement.
- +2 GAE tests. CLAIM-56 (KernelSelector 4/4 correct) confirmed on extended test suite.

**DiagonalKernel raw_weights fix (v0.7.19 → v0.7.20):**
- `gae/kernels.py` — DiagonalKernel.raw_weights property added, returns true 1/σ² values.
  Previous `.weights` property returned pre-normalized W/W.max() — correct for scoring,
  but wrong for Fisher information calculations (silent scale cancellation).
- DiagonalKernel.weights: pre-normalized [0,1] — USE FOR SCORING
- DiagonalKernel.raw_weights: true 1/σ² — USE FOR η_eff AND ENRICHMENT ROI (CLAIM-64)
- v0.7.19 had a raw_weights bug causing scale cancellation. v0.7.20 required for CLAIM-64.
- +1 GAE test: `kernel.raw_weights[i] == 1/sigma[i]**2` for all i (regression prevention).

**Block 9.1–9.5 v6.5 features (April 5, 2026):**
All five shipped as gated features. See §17.3 for full specs.

| Feature | Claim | Status | Tests |
|---|---|---|---|
| Per-analyst η weighting (Block 9.1) | CLAIM-66 CONDITIONAL (+0.86pp at q̄=0.80, r=0.975-1.000) | ✅ SHIPPED | +1 |
| η change-rate cap (Block 9.5) | CLAIM-67 UNCONDITIONAL (F=8.14, ±0.005/cadence) | ✅ SHIPPED | +1 |
| Spike detector deployment-specific (Block 9.2) | CLAIM-68 CONDITIONAL (p=0.010, NOT hardcoded 3σ) | ✅ SHIPPED | +1 |
| Category freeze on volume spikes (Block 9.3) | CLAIM-69 CONDITIONAL (65.5% spurious events blocked) | ✅ SHIPPED | +1 |
| Spike update cap 1.5× (Block 9.4) | CLAIM-70 CONDITIONAL (monotone learning during spikes) | ✅ SHIPPED | +1 |

**Test count progression (v0.7.18–v0.7.20):**
```
v0.7.17 (v6.0):  517 tests
+KERNELSEL-001:  519 tests  (+2: tiebreaker + regression)
+raw_weights:    520 tests  (+1: raw_weights regression)
+Block 9.1-9.5:  525 tests  (+5: one per feature)
+regression:     527 tests  (+2: BACKLOG-020 fixture + IKS stability)
```

### 17.1c Oracle Separation & γ Theorem (April 7–8, 2026) [NEW v10.4]

**Batch F META-4: RETIRED.** LLM competence prior dominates correctness labels at any
prompt sophistication. γ from LLM-generated personas is structurally unmeasurable.
Full record: `synthetic_data_generation_analysis_v2.md`.

**Oracle separation experiments (7 runs, April 2026):**

| Run | Setup | Key Finding |
|---|---|---|
| Exp A | Factor quality check | Oracle mechanism clean; regime differentiation validated |
| v6 | Cold init (μ₀ = uniform ± 0.15) | Phase 1 DNF — η₋ trap. N_half gap confirmed. |
| v8 | Expert init (ε_sim = 0.05) | γ = 0.714 < 1 ✓ (below theorem threshold) |
| v11 | GPT-5.4 probe (ε_sim = 0.05) | Floor effect: N_half = 10 (min possible). Model independence confirmed. |
| v2 | ε_sim = 0.20, wrong vector distribution | γ < 1 from distribution mismatch, not geometric failure |
| v3 | ε_sim = 0.20, corrected (cold_start both phases) | **γ = 1.033 > 1 ✓** (above theorem threshold) |
| Final | ε_sim = 0.35, 3 seeds | N_half variance 27× — simulation limit reached. Track closed. |

**Theorem established (April 8, 2026):** γ > 1 ↔ ε_firm > 0.125. Four LLMs independent.
Binary simulation confirmed in both directions. Simulation track CLOSED.

**New GAE modules added:**
- `gae/synthetic.py` — OracleSeparationExperiment, FactorVectorSampler, CanonicalCentroid (§10.13)
- `gae/convergence.py` — centroid_distance_to_canonical, gamma_threshold, phase2_effective_threshold (§10.14)

**Pending tests:** 5 for gae/synthetic.py + 4 for gae/convergence.py = 9 new tests before pilot Day 1.

---

### 17.2 v6.0 — COMPLETE ✅ (v0.7.17 shipped)

**Open-source prep (shipped with v0.7.x):**
- `README.md` complete rewrite (§15.3 spec) — DiagonalKernel in quick start
- `CONTRIBUTING.md` + `CODE_OF_CONDUCT.md` + `SECURITY.md`
- `CHANGELOG.md` v0.1.0 → v0.7.17 (includes kernel architecture + Phase 1 fixes)
- GitHub Actions: pytest + lint + mypy on push
- PyPI: `pip install graph-attention-engine` (v0.7.17)
- `examples/procurement_approval/` (§15.4)
- Docstring audit: all public classes fully documented
- `py.typed` marker (PEP 561)
- GitHub Issues templates (Bug, Feature, New Domain)

**Kernel architecture (SHIPPED v0.7.0):**
- DiagonalKernel as default for noise_ratio > 1.5
- KernelSelector for empirical kernel comparison during shadow mode
- CovarianceEstimator collecting data (research asset for v7.0)
- Factor quarantine mask deprecated (code present, docs say deprecated)

**Calibration and conservation (SHIPPED v0.7.x):**
- CalibrationProfile category_thresholds (enables 40%+ auto-approve coverage)
- compute_eta_override() for per-deployment η tuning
- AMBER auto-pause (conservation signal → freeze learning)
- derive_theta_min() canonical (θ_min=0.467, T_max=21)

**Referral routing (SHIPPED v0.7.0):**
- ReferralEngine: domain-agnostic protocol for post-scoring referral VETO
- ReferralRule protocol: pure functions, no ML, fully auditable
- OverrideDetector stub: interface for v6.5 learned referral patterns
- Experimental validation: rules 72.7% DR at 12% FPR (EXP-REFER-LAYERED)
- Confidence gate REJECTED for referral (14% precision = active harm)

**Phase 1 additions (SHIPPED v0.7.7–v0.7.17):**
- `gae/enrichment_advisor.py` — enrichment recommendation engine (validated on 5 profiles)
- `gae/convergence.py` OLSMonitor — CUSUM on OLS, h=5.0 OLS scale, plateau-snapshot (CLAIM-OLS-01)
- `gae/convergence.py` VarQMonitor — LOGGED ONLY, Bernoulli mixture theorem hard stop
- DiagonalKernel gradient fix: W/W.max()*(f-mu) — GAE-GRADIENT-001 v0.7.7
- η_neg guard: ValueError on η_neg ≥ 1.0 — v0.7.8

### 17.3 v6.5 — GainScheduler + Fisher Calendar + Embeddings + Override Learning

**Block 9.1–9.5 items: ✅ SHIPPED April 5, 2026 (in §17.1b above)**
Per-analyst η (CLAIM-66), η cap (CLAIM-67 UNCONDITIONAL), spike detector (CLAIM-68),
category freeze (CLAIM-69), spike update cap (CLAIM-70). All in GAE v0.7.20.

**Open-source test strategy (v10.5, April 8, 2026):**
Target ~900 tests (from current 527). 11 test categories + infrastructure.
New Block 3.7a: test suite restructure + P0/P1 categories.
See §15.6 for full category breakdown and test inventory.

**New deliverables from v10.4 (added April 8, 2026):**
- `gae/synthetic.py` — OracleSeparationExperiment, FactorVectorSampler, CanonicalCentroid, GammaResult (§10.13)
- `gae/convergence.py` — centroid_distance_to_canonical, gamma_threshold, phase2_effective_threshold, ConvergenceTrace (§10.14)
- 9 new tests needed (5 synthetic + 4 convergence) — P1 before pilot Day 1

**Remaining v6.5 items (not yet shipped):**
- `gae/embeddings.py` — EmbeddingProvider, PropertyEmbedding, TransformerEmbedding (Tier 4)
- `gae/graph_schema.py` — GraphSchema dataclass (§11.4)
- GainScheduler: periodic τ recalibration (~70 decisions/category), conservation-gated, σ-aware. Requires τ sweep (50 seeds) before implementation — Block 9.6 deferred.
- Fisher onboarding calendar (dynamic): per-category bars update as sources connect
- **OverrideDetector activation:** implement predict() (LogisticRegression on 19 features), retrain loop, data-gated at ≥50 production positives [NEW v10.1]
- Infrastructure for GraphAttentionBridge (writes hooks, no bridge logic yet)
- V1B LayerNorm: full test coverage before Level 2 ships
- Epistemic State Indicator (Adjustment G): per-decision kernel_type, shrinkage_lambda, centroid_age

### 17.4 v7.0 — GraphAttentionBridge (Level 2) + ShrinkageKernel Investigation

- `gae/bridge.py` — GraphAttentionBridge.register_domain(), enrich(), sweep()
- ShrinkageKernel investigation: does full Mahalanobis help at ρ>0.8? (research, not committed)
- Prerequisite: EmbeddingProvider for all domains (v6.5)
- Prerequisite: Level 1 data hooks writing correctly (PLAT-4)
- Prerequisite: EXP-G1 validates γ before cross-domain compounding claims
- LayerNorm is a hard production requirement (V1B — no configuration option)
- First major GAE release: v1.0.0

### 17.5 v8.0 — Cross-Domain Discovery (Level 3)

- `gae/discovery_engine.py` — DiscoveryEngine.discover()
- Prerequisite: GraphAttentionBridge validated (v7.0)
- Uses enriched embeddings as input
- Outputs new scoring dimensions → profile augmentation
- EXP-G1 must have passed and γ published

---

## 18. Repository Structure

```
graph-attention-engine/
├── gae/
│   ├── __init__.py              ✅ Full public API surface (v6.0 — expanded, OLSMonitor + enrichment_advisor)
│   ├── profile_scorer.py        ✅ ProfileScorer, ScoringResult, CentroidUpdate (v6.0: kernel, mask, eta_override, auto_pause)
│   ├── kernels.py               ✅ ScoringKernel, L2Kernel, DiagonalKernel (28 tests)
│   │                               NOTE: DiagonalKernel.compute_gradient = W/W.max()*(f-mu) — GAE-GRADIENT-001 fix v0.7.7
│   ├── covariance.py            ✅ CovarianceEstimator, CovarianceSnapshot (23 tests)
│   ├── kernel_selector.py       ✅ KernelSelector, KernelRecommendation (46 tests)
│   ├── referral.py              ✅ ReferralEngine, ReferralRule, OverrideDetector (31 tests)
│   ├── enrichment_advisor.py    ✅ NEW Phase 1: enrichment recommendation engine — factor priority ranking
│   │                               Input: σ_profile from CovarianceEstimator → Output: ranked factor list
│   │                               Validated on 5 deployment profiles. Feeds P28 Phase 2 report.
│   ├── calibration.py           ✅ CalibrationProfile + factor_mask + eta_override + conservation + fisher
│   │                               η_neg guard: ValueError on η_neg ≥ 1.0 (v0.7.8)
│   ├── factors.py               ✅ FactorComputer protocol, assemble_factor_vector
│   ├── learning.py              ✅ LearningState (delegates to ProfileScorer)
│   ├── oracle.py                ✅ OracleProvider, GTAlignedOracle, BernoulliOracle
│   ├── evaluation.py            ✅ run_evaluation, EvaluationReport, compute_ece
│   ├── judgment.py              ✅ compute_judgment, JudgmentResult
│   ├── ablation.py              ✅ run_ablation, AblationReport
│   ├── scoring.py               ⚠️ DEPRECATED — forwards to ProfileScorer (remove v7.0)
│   ├── convergence.py           ✅ v2: EPSILON=0.10, safety_factor=2.0
│   │                               OLSMonitor: CUSUM on OLS, h=5.0 (OLS scale), plateau-snapshot (CLAIM-OLS-01)
│   │                               VarQMonitor: logged only — PERMANENT HARD STOP for gating (Bernoulli mixture theorem)
│   ├── fisher.py                ✅ estimate_fisher_information, predict_n_half, enrichment_multiplier
│   ├── contracts.py             ✅ Unchanged
│   ├── primitives.py            ✅ Unchanged
│   ├── events.py                ✅ Unchanged
│   ├── store.py                 ✅ ProfileScorer state persistence
│   ├── types.py                 ✅ Unchanged
│   ├── embeddings.py            📋 Tier 4 — v6.5
│   ├── graph_schema.py          📋 Level 1 hooks contract — v6.5
│   ├── bridge.py                📋 Level 2 GraphAttentionBridge — v7.0
│   └── discovery_engine.py      📋 Level 3 — v8.0
├── tests/
│   ├── test_profile_scorer.py   ✅ L2, DiagonalKernel, mask, eta_override, AMBER pause, kernel integration
│   ├── test_kernels.py          ✅ ScoringKernel protocol, L2, Diagonal, gradient correctness (28 tests)
│   │                               Includes: W/W.max() gradient test, zero-weights fallback, gradient direction validation
│   ├── test_covariance.py       ✅ CovarianceEstimator (23 tests)
│   ├── test_kernel_selector.py  ✅ KernelSelector Phase 2/3/4 (46 tests)
│   ├── test_referral.py         ✅ ReferralEngine, OverrideDetector (31 tests)
│   ├── test_calibration.py      ✅ factor_mask, mask_to_array, eta_override, conservation, fisher
│   │                               η_neg guard: ValueError at ≥1.0, boundary at exactly 1.0
│   ├── test_convergence.py      ✅ OLSMonitor CUSUM (h=5.0), plateau-snapshot, VarQMonitor logged-only
│   ├── test_enrichment_advisor.py ✅ NEW Phase 1: factor ranking, 5 deployment profiles
│   ├── test_oracle.py           ✅
│   ├── test_evaluation.py       ✅
│   ├── test_judgment.py         ✅
│   ├── test_ablation.py         ✅
│   ├── test_learning.py         ✅
│   └── test_scoring.py          ✅ (backward compat)
├── examples/
│   ├── minimal_domain/          ✅ Helpdesk domain, runnable standalone
│   │   ├── config.py
│   │   ├── run_example.py
│   │   └── README.md
│   └── procurement_approval/    ✅ S2P domain (§15.4)
├── docs/
│   ├── users_guide.md           ✅ Updated for kernel architecture + OLSMonitor
│   ├── equations.md             ✅ Eq reference with paper links + gradient fix documented
│   └── EXPORTS.md               ✅ Full public API reference
├── .github/
│   ├── workflows/
│   │   ├── pytest.yml           ✅ CI — all 517 tests on push
│   │   └── lint.yml             ✅ CI — ruff + mypy
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.yml       ✅
│       └── new_domain.yml       ✅
├── README.md                    ✅ Rewritten (§15.3 spec, DiagonalKernel, CLAIM-W2 context)
├── CONTRIBUTING.md              ✅
├── CHANGELOG.md                 ✅ v0.1.0 → v0.7.17
├── SECURITY.md                  ✅
├── CODE_OF_CONDUCT.md           ✅
├── pyproject.toml               ✅ v0.7.20
└── LICENSE                      ✅ Apache 2.0
```

---

## Appendix A: Experiment Parameters

*(Unchanged from v8.3 Appendix A. See that version for full table.)*

## Appendix B: Chart Inventory

*(Unchanged from v8.3 Appendix B. Key chart for README: expC1_comparison_waterfall.png — the 25%→98% waterfall that visualizes the kernel finding.)*

---

*Graph Attention Engine — Design & Architecture v10.3 | April 6, 2026*
*Phase 0 ✅ Phase 1 ✅ Phase 2 ✅ Phase 3 Priority 1 ✅. 527 tests. v0.7.20.*
*v6.0 COMPLETE: DiagonalKernel (+13pp SOC, +7pp S2P), KernelSelector, CovarianceEstimator, ReferralEngine, OLSMonitor, enrichment_advisor.*
*CRITICAL FIX: DiagonalKernel gradient = W/W.max()*(f-mu) — GAE-GRADIENT-001 fix v0.7.7. η_neg guard v0.7.8.*
*W2 flywheel: CLAIM-W2 +10.13pp (p=0.0002). Third compounding pathway: CLAIM-59 54.4% faster (p<0.0001).*
*Enrichment Day-1 lift: CLAIM-62 +42.69pp. Fisher info path: CLAIM-64 r=0.9669 (raw_weights, GAE 0.7.20).*
*Economics: CL-ECON-MEASURED 30.85 min/alert, $523K–$2.8M/year. Innovation 10 CLOSED.*
*Flywheel Health Monitor: CLAIM-OLS-01 0% miss rate, p90≥50d lead time. CUSUM h=5.0 OLS scale.*
*Var(q) gating: PERMANENT HARD STOP (Bernoulli mixture theorem). Logged metric only.*
*Convergence calendar: CLAIM-CONV-01 MAE=1.55d. V no causal effect on N_half.*
*Poisoning resilience: CLAIM-SK-01 (0.850pp at 20% σ-perturbation) + CLAIM-LP-01 (3.20pp at 20% label).*
*Enrichment safety: CLAIM-65 <1.2pp at adversarial multi-factor contamination.*
*ShrinkageKernel deprioritized to v7.0. Off-diagonal adds <1pp in all current tests.*
*Two levels of institutional judgment: ProfileScorer = Level 1 (Decision Intelligence). AgentEvolver = Level 2 (Deployment Intelligence). GAE owns Level 1.*
*Three referral phases: Rules (v6.0) → OverrideDetector (v6.5, ≥50 positives) → Monthly retrain (v7.0).*
*Block 9.1-9.5 CLAIM-66-70 all shipped. CLAIM-67 (η rate cap) UNCONDITIONAL (F=8.14).*
*raw_weights (true 1/σ²) for η_eff. weights (pre-normalized) for scoring. Distinction is a hard architectural rule.*
*"The math is open. The moat is the accumulated graph. The gradient fix makes the flywheel real. The graph compounds while centroids wait. Recovery is not coincidence — γ > 1 is proven. Test every invariant before every release."*
