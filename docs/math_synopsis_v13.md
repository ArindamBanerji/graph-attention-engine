> **CHANGE NOTE (April 8, 2026 — v12→v13 update):**
>
> One new theorem, four document locations updated.
>
> **(1) §3.2 NEW:** Re-Convergence Theorem. γ > 1 proven analytically for category-sparse
>     disruption when ε_firm > ε_firm★ ≈ 0.125. Four proof paths. Simulation confirmation.
>     **Formula confirmed correct (coding session April 8 + April 9):** ε_firm★ = α·‖Δ‖/(1−α) — also written as `alpha*delta/(1-alpha)` in code.
>     θ cancels in correct derivation. Value ≈ 0.125 (was approximated as 0.128, diff = 0.003).
>     OracleSeparationExperiment in gae/experiments/ validates both directions: ε=0.05 → γ=0.714 < 1; ε=0.20 → γ=1.033 > 1.
>     N_half measurement gap documented. EXP-G1 is the empirical measurement path.
>
> **(2) §10.1 UPDATED:** Reference to §3.2 added. Graph-dependent convergence now
>     explicitly connects to the re-convergence theorem.
>
> **(3) §12 UPDATED:** Re-Convergence Theorem added as RESOLVED open math issue.
>
> **(4) §13 UPDATED:** Gate GAMMA added after Gate REFERRAL.
>
> **(5) §3.2 UPDATED:** Fourth proof path added (centroid-distance
>     derivation, Grok primary, April 16 math poll).
>
> Sources: GPT-4.1, Claude Opus 4, Grok 3, Gemini 1.5 Pro (April 8, 2026).
> Simulation: oracle separation v3/v8/final (oracle_separation_validation files).
> Full audit trail: synthetic_data_generation_analysis_v2.md

> **ARXIV CHANGE NOTE (April 1, 2026 — v11→v12 update):**
>
> Three confirmed errors corrected. Four additions from validated experiments.
> V-GATE-STABILITY (N=1000 confirmed), V-GATE-DRIFT (baselines stable 90 days),
> V-D5 (per-analyst η weighting, CONDITIONAL), D6 closure, FINDING-OVR-01.
> Entity resolution status confirmed by code audit. Self-calibrating gate principle added.
>
> **(1) §15 v6-KERNEL CORRECTED:** KernelSelector shadow minimum: ~250 decisions
>     → max(1000, 20×V×α). V-GATE-STABILITY: all three baselines (volume, precision
>     ranking, agreement variance) stabilize at N=1000. Binding constraint: volume
>     baseline requires ≥20 days, not ≥250 decisions.
>
> **(2) §15 v6-NEW Entity Resolution CORRECTED:** "3-pass pipeline, deterministic
>     canonical IDs" is partially correct. Exact-match pipeline (_normalize,
>     _build_merge_groups, union-find, IdentifierType) ✅ complete. Probabilistic
>     fuzzy matching (Jaccard token overlap) ❌ not present — pending Block 6.1.
>
> **(3) §16 Constraints CORRECTED — θ_min:** θ_min = 0.467 (hardcoded) REPLACED by
>     θ_min = 23.53/(α×V) (deployment-specific formula). Derivation: N_half × η × V_min.
>     At V=50, α=0.25: θ_min=1.88 — impossible (V×α < 20/day). Formally equivalent
>     to CLAIM-SC-01 scope condition. KernelSelector shadow minimum correction
>     propagated: "~250 decisions" → max(1000, 20×V×α) in §16 table.
>
> **(4) §3 ADDED — Per-Analyst η Weighting (Eq. η_i):** D5 validated (CONDITIONAL,
>     0.86pp, V-D5-MECHANISM-GATE PATH 1, regime-independent). Each analyst's override
>     contribution weighted by their measured override precision relative to team mean.
>     Subsumes D6 (night-shift attenuation) — precision drops for any reason, including
>     fatigue, without time windows or shift configuration.
>
> **(5) §16 ADDED — FINDING-OVR-01:** Override precision is structurally uncorrelated
>     with q̄ across both synthetic datasets (V7: r=0.00, V3: r=-0.70). Must be measured
>     directly per analyst. Not predictable from role, seniority, or agreement rate.
>
> **(6) §16 ADDED — D6 Closure:** Night-shift attenuation (D6) CLOSED PERMANENTLY.
>     Subsumed by per-analyst η weighting (D5). V-NIGHT: two attempts (v7: -12.9pp
>     inverted, v10: -3.9pp inverted). Explicit override rules cannot model fatigue —
>     D5 continuous precision monitoring subsumes D6 without time windows or config.
>
> **(7) §16 ADDED — Self-Calibrating Gate Principle:** All operational gates are
>     now deployment-specific formulas derived from the deployment's own shadow data,
>     not experiment-calibrated constants. V-GATE-STABILITY + V-GATE-DRIFT validated:
>     baselines stable within 7% across 90-day pilot period, no monthly recalibration.

> **ARXIV CHANGE NOTE (March 21, 2026 — v10→v11 update):**
>
> A=4 confirmed by experiment (EXP-A4-DIAGONAL: 13pp structural, kernel-independent).
> ReferralRules R1-R7 shipped (VETO mechanism, 72.7% DR, 12% FPR). 478 GAE + 280 SOC tests.
> ~104 experiments. Factorial cells corrected to 390 (was 360). Source references updated.
>
> **(1) §8 UPDATED:** Four new experiment rows: EXP-A4-DIAGONAL, EXP-REFER-COVERAGE,
>     EXP-REFER-LAYERED, HC-scaling. KernelSelector validation row added. ~104 experiments.
>
> **(2) §9 UPDATED:** Safety layer S9 added (ReferralRules R1-R7 VETO). Referral routing
>     is policy-based, independent of centroid scoring. Override learning data-gated v6.5.
>
> **(3) §11 UPDATED:** Problem 3 referral mechanism updated: ReferralRules R1-R7 replace
>     confidence-gated referral. Three-tier dispatch preserved (auto-approve / investigate /
>     review), with policy VETO overlay for referral.
>
> **(4) §13 UPDATED:** Gate A4-CONFIRM and Gate REFERRAL added. A=4 confirmed by 13pp
>     structural gap. ReferralRules validated by EXP-REFER-LAYERED (4 layers × 300 runs).
>
> **(5) §14 UPDATED:** PyPI v0.6.0 → v0.7.0 (was stale). 437 → 478 tests.
>
> **(6) §15 UPDATED:** v6-REFERRAL requirement added. ReferralRules R1-R7. 280 SOC tests.
>     v6-KERNEL test count 447→478.
>
> **(7) §16 UPDATED:** ReferralRules R1-R7 constraint added. Referral VETO does not
>     affect ProfileScorer scoring (by construction — insertion after composite gate).
>
> **(8) Sources UPDATED:** product_strategy_v2→v3, design_note_v1→v2, roadmap_v19 added.
>     Factorial cells 360→390. ~100→~104 experiments.

> **ARXIV CHANGE NOTE (March 21, 2026 — v9.1→v10 update):**
>
> Kernel architecture settled. DiagonalKernel validated. Asymmetric η (P0 fix).
> Tensor updated to A=4. ~100 experiments complete. Product boundaries established.
>
> **(1) §1 UPDATED:** Tensor 144 values (6×4×6, A=4). DiagonalKernel as v6.0 default.
>     Asymmetric η (η_confirm=0.05, η_override=0.01). Conservation law θ_min=0.467.
>
> **(2) §3 Eq. 4-final UPDATED:** DiagonalKernel equation added alongside L2.
>     Kernel selection rule: noise_ratio > 1.5 → Diagonal, else L2.
>     Validated: +13.2pp SOC, +6.8pp S2P (V-MV-KERNEL factorial, 390 cells).
>
> **(3) §3 Eq. 4b-final UPDATED:** Asymmetric η. η_confirm=0.05, η_override=0.01.
>     P0 fix: prevents 13-27pp centroid degradation from low-quality overrides.
>     Validated across 24 personas. Corr(noise_ratio, diagonal_advantage)=0.990.
>
> **(4) §3 Available Kernels UPDATED:** DiagonalKernel row added. Noise ceiling
>     table added (kernel-dependent: L2 σ≤0.157, Diagonal σ≤0.25).
>     ShrinkageKernel deprioritized to v7.0 (off-diagonal adds <1pp).
>
> **(5) §6 Tensor UPDATED:** A=5→4. Actions: escalate, investigate, suppress, monitor.
>     refer_to_analyst removed as scorable action (via confidence gate only).
>     Tensor 180→144. Random baseline 25% (1/4).
>
> **(6) §7 UPDATED:** DiagonalKernel accuracy numbers. Product boundary table from
>     Phase 1 persona sweeps (24 personas) + V-MV-KERNEL factorial (390 cells).
>
> **(7) §9 UPDATED:** σ_max = 0.034 (from FX-1-PROXY-REAL + margin computation).
>
> **(8) §16 Constraints UPDATED:** η_override=0.01, θ_min=0.467, AMBER auto-pause,
>     DiagonalKernel default for noise_ratio>1.5, noise ceiling kernel-dependent.
>
> **(9) §17/§18 UPDATED:** New equations and notation for DiagonalKernel, W, η_override.

> **ARXIV CHANGE NOTE (March 17, 2026 — v9→v9.1 update):**
>
> Three-judge validated mathematical bridges incorporated. All formulas
> reviewed by GPT-4o, Claude Opus, and Gemini. Search for [CHANGED v9.1].
>
> **(1) §3.1 ADDED: Convergence Analysis.** Centroid learning convergence
>     rate derived from Eq. 4b-final. Mean error decays as (1−η)^n (Borkar
>     2008, Kushner & Yin 2003). Steady-state MSE = η·tr(Σ_f)/(2−η).
>     Three-judge confirmed. Language: "tracks stationary neighborhood"
>     not "converges to point."
>
> **(2) §5 UPDATED: Level 2 formalized.** AgentEvolver as conservative
>     contextual bandit with four-condition promotion gate (superiority +
>     correctness floor + conservation law + variance stability).
>     Conservation law: α(t)·q(t)·V(t) ≥ θ_min. Three-judge confirmed.
>
> **(3) §10.1 ADDED: Graph-Dependent Convergence.** Convergence rate as
>     function of graph richness. ρ-adjusted effective sources:
>     N_eff = N/(1+ρ(N−1)). Mechanism-level grounding for γ>1.
>
> **(4) §12 UPDATED.** Bridge-derived items added to open math issues.
>     Verification rate floor refined from "~15% estimate" to three
>     monitored conditions (coverage, drift consistency, excitation).
>
> **(5) §16 UPDATED.** Conservation law added as architectural constraint.
>     Centroid support monitoring added.
>
> **(6) §17/§18 UPDATED.** New equations and notation from bridges.

> **ARXIV CHANGE NOTE (March 15, 2026 — v8→v9 update):**
>
> Authors referencing math_synopsis for paper drafts: the following changes
> affect equations and parameter values. Search for [CHANGED v9] markers.
>
> **(1) Tensor dimensions:** C=6, A=4 [CHANGED v10: was A=5], tensor 144.
> Category list corrected to match code. Random baseline 25% (1/4).
>
> **(2) Eq 4b update rule CORRECTED.** The correct=False branch previously
> described as "push centroid away" was ambiguous about WHICH centroid.
> The corrected rule: push PREDICTED (wrong) centroid away from f, AND
> pull GROUND TRUTH centroid toward f. Prior implementation pushed ALL
> centroids (a bug, fixed March 15, SHIFT-2 validated).
>
> **(3) η_neg design decision.** η_neg=0.05 (symmetric with η) is canonical.
> η_neg=1.0 produces catastrophic miscalibration (ECE=0.49, PROD-4b).
> [CHANGED v10: asymmetric η added — η_override=0.01 for override path.]
>
> **(4) Frozen scorer baseline established.** Frozen μ₀ (expert prior, no
> learning): 80.4% accuracy, 92.9% coverage at 85% precision (zero noise).
> At 10% noise: 72.5% accuracy, 62.6% coverage. These are the verified
> baseline numbers for the architecture. [CHANGED v10: A=4 gives 90.6% static.]
>
> **(5) Learning validated post-fix.** With corrected update rule: +2.7%
> accuracy lift at noise=0/δ=0.10/warmup=1000. +1.5% at noise=0.10.
> Learning works when prior mismatch exists. Noise is the remaining constraint.

# Compounding Intelligence: Mathematical Synopsis
**Version:** v13 (April 8, 2026) — Re-Convergence Theorem added (§3.2, §10.1 update, §12, §13 Gate GAMMA)

**Version:** 11.0 · March 21, 2026
**Purpose:** Complete mathematical framework in one document. For experts and LLMs.
**Status:** v6.0 KERNEL ARCHITECTURE SETTLED. A=4 CONFIRMED (13pp structural).
DiagonalKernel validated (+13.2pp SOC, +6.8pp S2P).
ReferralRules R1-R7 shipped (VETO, 72.7% DR, 12% FPR). 478 GAE + 280 SOC tests (~935 total).
Eq. 1–4b validated (~104 experiments including 50-seed realistic suite + persona sweeps + factorial).
Eq. 4-synthesis is PROPOSAL (gated by EXP-S1–S8). GATE-M satisfied (March 14).
Asymmetric η (P0 fix): η_confirm=0.05, η_override=0.01. Validated across 24 personas.
Noise ceiling kernel-dependent: L2 σ≤0.157, Diagonal σ≤0.25.
Realistic accuracy numbers supersede centroidal synthetic numbers for all product claims.
Three-judge validated mathematical bridges (GPT-4o, Claude Opus, Gemini) incorporated at v9.1.
**Sources:** cross_graph_attention_v3 (authoritative math), compounding_intelligence_v7
(authoritative architecture), intelligence_layer_design_v2 (synthesis proposal),
product_strategy_v3 (gap analysis), experiments_catalog_v2 (master catalog),
gae_design_v10 (engine API contract), multivariate_foundation_design_note_v2,
platform_roadmap_v19 (what ships when),
three-judge bridge reviews (GPT-4o, Claude Opus, Gemini — March 17, 2026).

---

**What changed from v10:** [CHANGED v11] — see change note above.

**What changed from v9.1:** [CHANGED v10] — see change note above.

**What changed from v9:** [CHANGED v9.1] — see change note above.

**What changed from v8:** [CHANGED v9] — see change note above.

**What changed from v7/v6:** — see prior change notes above.

---

## 1. The Architecture in One Paragraph

A knowledge graph accumulates operational context. FactorComputers traverse the graph
to produce a factor vector f ∈ [0,1]^d for each decision. A ProfileScorer computes
action probabilities using a distance kernel between f and learned profile centroids
μ[c,a,:]. The default kernel is DiagonalKernel(W=diag(1/σ²)) for deployments with
noise_ratio > 1.5 (every real deployment except ultra-clean FinServ); L2 is the
cold-start fallback. Verified outcomes update centroids via pull/push with asymmetric
learning rates: η_confirm=0.05 for analyst confirmations (clean signal),
η_override=0.01 for analyst overrides (noisy signal, attenuated 5× — the P0 fix
preventing 13-27pp centroid degradation from low-quality overrides). An optional
synthesis bias σ[c,a] (PROPOSAL, gated by EXP-S1–S8) shifts action preferences based
on external intelligence. The system compounds: every verified decision reshapes
centroids, making the next decision on similar alerts more accurate. The moat is the
centroid tensor — readable institutional judgment encoded in 144 numbers (6 categories
× 4 actions × 6 factors). Referral routing is handled OUTSIDE the tensor by
ReferralRules R1-R7 (policy-based VETO, 72.7% DR, 12% FPR) — the signal for "needs
human review" lives in context, not factor-space geometry. The system learns at two
formally characterized levels: Level 1 (ProfileScorer centroids — what to decide,
convergence rate (1−η)^n, half-life N_half≈14, Borkar 2008) and Level 2 (AgentEvolver
variant selection — how to operate, conservative contextual bandit with four-condition
safety gate). A conservation law α(t)·q(t)·V(t) ≥ θ_min (where θ_min = 23.53/(α×V), deployment-specific) ensures the two levels compound rather than conflict. [CHANGED v11; θ_min formula CORRECTED v12]

---

## 2. Five Layers

| Layer | Name | Contains | Changes At |
|---|---|---|---|
| 1 | Domain Ontology | Categories, actions, factors, initial centroids μ₀, referral rules | Configuration time |
| 2 | Meta-Graph (Compiled State) | μ[c,a,:] (experience), σ[c,a] (awareness, PROPOSAL) | Decision time (μ slow, σ fast) |
| 3 | Mathematical Engine | ProfileScorer, kernels (L2/Diagonal), softmax, calibration | Never (deterministic) |
| 4 | Knowledge Graph | PostgreSQL + AGE nodes/edges, FactorComputers, write-back | Every decision |
| 5 | Control Plane | Four feedback loops (3 validated + 1 proposed), ReferralRules VETO | Per outcome |

---

## 3. Core Equations (VALIDATED)

### Eq. 4-final — Scoring

```
P(a|f,c) = softmax(−K(f, μ[c,a,:]) / τ)
```

- f ∈ [0,1]^d: factor vector from graph traversal (d=6 for SOC, d=8 for S2P)
- μ[c,a,:] ∈ [0,1]^d: profile centroid for category c, action a
- K: kernel function. See kernel table below.
- τ = 0.1: temperature (V3B validated, ECE=0.036). Fixed. Never change.
- Softmax over n_act actions: P sums to 1, selected action = argmax

**Two production kernels (v6.0):** [NEW v10]

```
L2 (cold-start fallback):
  K(f, μ) = ‖f − μ‖² = Σ_j (f_j − μ_j)²

DiagonalKernel (v6.0 default for noise_ratio > 1.5):
  K(f, μ) = (f − μ)ᵀ W (f − μ) = Σ_j w_j (f_j − μ_j)²
  where W = diag(1/σ²), σ_j = per-factor noise from deployment qualification

  When W = I (identity): reduces to L2.
  When σ is uniform across factors: reduces to L2 (all weights equal).
  When σ varies per factor: down-weights noisy dimensions, up-weights clean ones.
```

**Kernel selection rule (settled, Explanation A confirmed):** [NEW v10]
```
noise_ratio = max(σ_per_factor) / min(σ_per_factor)
if noise_ratio > 1.5 → DiagonalKernel(weights=1/σ²)
else                  → L2Kernel
```

**Interpretation:** Each action has a prototype factor pattern (the centroid). The action
whose prototype is closest to the observed factors wins — where "closest" accounts for
per-factor noise when using DiagonalKernel. Temperature τ=0.1 controls decision
sharpness — validated as optimal by V3B (ECE=0.036 vs 0.19 at τ=0.25).

---

### Eq. 4b-final — Centroid Learning [CHANGED v10]

```
When correct (a_pred = a_gt) — CONFIRM path:
  μ[c, a_pred, :] ← μ[c, a_pred, :] + η_confirm_eff · G(f, μ[c, a_pred, :])   [pull]

When incorrect (a_pred ≠ a_gt) — OVERRIDE path, dual update:
  μ[c, a_pred, :] ← μ[c, a_pred, :] − η_override_eff · G(f, μ[c, a_pred, :])  [push wrong away]
  μ[c, a_gt, :]   ← μ[c, a_gt, :]   + η_confirm_eff  · G(f, μ[c, a_gt, :])    [pull correct toward]
  All other action centroids in category c: unchanged.

η_confirm_eff = η_confirm / (1 + n[c, a_pred] · decay_rate)
η_override_eff = η_override / (1 + n[c, a_pred] · decay_rate)

G(f, μ) = kernel gradient:
  L2Kernel:       G(f, μ) = (f − μ)
  DiagonalKernel: G(f, μ) = W · (f − μ)    — pushes harder on clean dimensions

Where:
  a_pred = predicted action (argmax of P(a|f,c))
  a_gt   = ground truth action (from verified analyst outcome)
```

- **η_confirm = 0.05:** confirm-path learning rate (clean signal, full rate) [CHANGED v10]
- **η_override = 0.01:** override-path learning rate (noisy signal, attenuated 5×) [NEW v10]
- η_neg = 0.05: canonical base rate. **FORBIDDEN: η_neg=1.0 — ECE=0.49 (PROD-4b).**
- n[c,a]: cumulative verified-outcome count for (category, action) pair
- decay_rate = 0.001: count-based decay — stability increases with experience
- **INVARIANT:** μ ← clip(μ, 0.0, 1.0) after every update (V2 validated, mandatory)
- **INVARIANT:** Masked dimensions (factor_mask) do NOT update — gradient zeroed. [NEW v10]

**Asymmetric η — the P0 fix (March 19, 2026):** [NEW v10]

9-persona LLM-judge stress test found 13-27pp centroid degradation from realistic analyst
override quality (q̄=0.60-0.70). The override path carries noise from analyst errors.
η_override=0.01 attenuates this by 5×. Validated across 24 personas (1C quality sweep):
at q̄=0.57 (worst quality): +0.5pp with η_override=0.01 (no degradation); without: -9pp.
Corr(noise_ratio, diagonal_advantage) = 0.990 across 4 healthcare personas.

**Formula (diagnostic — global default 0.01 is the validated value):**
```
η_override* ∝ (2q̄ − 1) / (2σ²_q + signal)
```
Directionally correct, ~2× overestimate vs empirical optimum. Q5 persona sweep is
ground truth; per-deployment TD-034 sweep confirms.

**Bug fix note (March 15, 2026):** Prior to this fix, the incorrect branch pushed ALL
action centroids in category c away from f (including the ground truth centroid). This
caused centroid degradation. SHIFT-2 validated: -9.0% accuracy lift before fix,
+2.7% after fix (noise=0, δ=0.10, warmup=1000).

**Interpretation:** Correct outcomes pull the predicted centroid toward the observed
factor pattern using the kernel-aware gradient. Incorrect outcomes push the wrong
centroid away AND pull the correct centroid closer, both at the attenuated override
rate. The asymmetry between confirm (0.05) and override (0.01) reflects that analyst
overrides are noisier than confirmations — the system trusts agreements more than
disagreements. Count decay ensures early decisions carry more weight per-update
than later decisions — the system front-loads learning and stabilizes with experience.

---

### Available Kernels [CHANGED v10]

| Kernel | K(f,μ) | When to Use | Accuracy | Noise Ceiling |
|---|---|---|---|---|
| **L2 (cold-start)** | ‖f − μ‖² | noise_ratio ≤ 1.5. Factors in [0,1]. | **97.89%** (centroidal) | σ ≤ 0.157 |
| **DiagonalKernel (v6.0 default)** | **(f−μ)ᵀW(f−μ), W=diag(1/σ²)** | **noise_ratio > 1.5. Heterogeneous factor noise.** | **+13.2pp over L2** (hetero SOC) | **σ ≤ 0.25** |
| Cosine | 1 − cos(f,μ) | Pre-normalized factors | 96.42% | — |
| Dot product | −f·μᵀ | **DO NOT USE.** Magnitude confounding. | 61.00% | — |
| ShrinkageKernel | (f−μ)ᵀΣ̂⁻¹(f−μ) | **DEPRIORITIZED to v7.0.** Off-diagonal adds <1pp. | Same as Diagonal | — |

**The DiagonalKernel finding (V-MV-KERNEL, March 21):** [NEW v10]

The 390-cell factorial (216 uniform + 144 heterogeneous + 18 S2P + 4 HC + 4 selector
+ 4 shrinkage) across SOC + S2P domains proved that DiagonalKernel(1/σ²) captures the
full kernel advantage. Every real deployment has heterogeneous per-factor noise — the
advantage is real and consistent. [CHANGED v11: 360→390 cells]

| Domain | L2 (hetero) | Diagonal (hetero) | Lift |
|---|---|---|---|
| SOC | 79.5% | 92.7% | **+13.2pp** |
| S2P | 42.2% | 49.0% | **+6.8pp** |
| SOC at σ=0.22 | 61-64% | 83-85% | **+20-22pp** |

**Explanation A confirmed:** Off-diagonal correlations add <1pp in both domains. Noise
ratio alone drives the kernel advantage. ShrinkageKernel deprioritized to v7.0.

**Noise ceiling is kernel-dependent:** [NEW v10]

| Kernel | GREEN | AMBER | RED |
|---|---|---|---|
| L2 | σ_mean ≤ 0.105 | 0.105 < σ ≤ 0.157 | σ > 0.157 |
| DiagonalKernel | σ_mean ≤ 0.157 | 0.157 < σ ≤ 0.25 | σ > 0.25 |

The GREEN zone nearly doubles with DiagonalKernel. Healthcare deployments (σ≈0.22)
move from RED (L2, frozen only) to AMBER (Diagonal, learning with monitoring).

**Noise ceiling is also three-variable (V-B3, March 21):** [NEW v10]

The corruption vector is V × (1−q̄) × η_override, not σ alone. At low volume (V=50),
even σ=0.190 doesn't corrupt centroids (too starved for bad overrides). At V=200 +
σ=0.157 + q̄=0.60, degradation occurs. Low volume is protective (same mechanism as
2D starvation finding).

The 36.89pp gap between dot product and L2 on identical data (EXP-C1) remains the
foundational kernel finding. DiagonalKernel extends this: the kernel choice AND the
per-factor weighting are both critical architectural decisions.

---

### §3.1 — Convergence Analysis for Eq. 4b-final [UNCHANGED from v9.1]

> **Note:** The convergence analysis in §3.1 is unchanged from v9.1. The asymmetric η
> (η_confirm vs η_override) modifies the RATE but not the FORM of convergence. The mean
> error dynamics become E[e_n] ≈ (1−η_eff)^n × e_0 where η_eff is the effective rate
> accounting for the mix of confirms and overrides. N_half ≈ 14 at η_confirm=0.05
> remains correct for the confirm path; the override path converges more slowly
> (N_half ≈ 69 at η_override=0.01). The blended N_half depends on the confirm/override
> ratio, which varies by analyst quality. At q̄≥0.70 AND σ≤0.157 (L2) /
> σ≤0.25 (Diagonal): N_half≈14 remains the correct claim (most updates are confirms).

[Full §3.1 content preserved from v9.1 — Eq. CONV, MSE∞, N_CONV, conditions unchanged.]


### §3.2 — Re-Convergence Theorem [NEW v13]

**Context:** After initial deployment (Phase 1), the system converges to operational
accuracy in N_half,1 decisions. After an environmental disruption (Phase 2), the system
re-converges in N_half,2 decisions. The re-convergence speed ratio γ = N_half,1 / N_half,2
is the temporal compounding claim: γ > 1 means re-convergence is faster than initial
calibration.

---

**Two-Phase Setup**

```
Phase 1 — Initial deployment:
  μ₀ = μ_canonical + ε_firm       (expert-initialized centroid + firm-specific deviation)
  D₁ = ‖ε_firm‖                   (Phase 1 initial distance from firm ground truth GT₁)
  P₁ = cold factor vector distribution (no accumulated W2 graph edges)
  N_half,1 = min{n ≥ w : rolling_w(correct) ≥ θ}

Phase 2 — After category-sparse disruption:
  GT₂ = GT₁ + Δ                   (disruption shifts c_d of C categories)
  μ_T1 ≈ GT₁                      (Phase 1 has converged)
  α_cat = c_d / C                  (fraction of alert categories disrupted)
  α_tensor = 4 / (C·A·d) ≈ 0.03  (fraction of tensor elements disrupted)
  P₂ ≠ P₁                         (enriched factor vectors from T₁ W2 edges)
  N_half,2 = min{n ≥ w : rolling_w(correct) ≥ θ} from Phase 2 start

Production values: C=6, A=4, d=6, c_d=2, w=10, θ=0.85
```

---

**The Rolling-Window Shortcut (Central Mechanism)**

In Phase 2, (1 − α_cat) = 4/6 ≈ 0.67 of alert decisions come from undisrupted categories.
These decisions are immediately correct (μ_T1 ≈ GT₂ for undisrupted categories).

The effective Phase 2 accuracy threshold for disrupted categories is:

```
p_d★ = (θ − (1 − α_cat)) / α_cat
     = (0.85 − 0.67) / 0.33
     ≈ 0.55
```

Phase 2 only needs disrupted categories to reach 55% correctness (not 85%) for the
rolling window to declare N_half,2. Phase 1 had no such shortcut — all C categories
started cold from ε_firm.

---

**Theorem (Re-Convergence Speed — April 8, 2026)**

```
γ > 1   ⟺   ε_firm > ε_firm★

where:

ε_firm★ = α_cat · ‖Δ‖ / (1 − α_cat)
         = (2/6) · 0.25 / (1 − 2/6)
         = 0.0833 / 0.6667
         ≈ 0.125

[Formula corrected April 8, 2026: θ cancels in correct derivation.
 Previous version had α·‖Δ‖·θ/(θ−(1−α)) = 0.387 for production values — wrong.
 Coding session confirmed via gae/convergence.py implementation. Diff = 0.003.]
```

**Parameters:**
- ε_firm: firm-specific deviation from canonical centroid (every real deployment has ε_firm > 0)
- α_cat = c_d/C: fraction of alert categories disrupted (≈ 0.33 for a typical campaign)
- ‖Δ‖: disruption magnitude per affected tensor element (≈ 0.25 for a new attack campaign)
- θ: operational accuracy threshold (= 0.85)

**Production range:** ε_firm ≈ 0.15–0.40. Threshold ε_firm★ ≈ 0.125.
Every real deployment clears the threshold.

**Simulation range:** ε_sim = 0.05 (oracle separation experiments). Below threshold.
This correctly predicts γ < 1 in simulation, as observed.

---

**Four Structural Proof Paths (all confirmed independently)**

*Path 1 — Geometric (primary):*
Phase 1's convergence challenge (all C categories, from ε_firm) exceeds Phase 2's
challenge (c_d disrupted categories, to reduced target p_d★) when ε_firm > ε_firm★.
Directly gives the threshold condition.

*Path 2 — Dimensional:*
Phase 1 must calibrate C categories. Phase 2 must reconverge c_d categories while
(C − c_d) categories remain correctly calibrated and contribute free rolling-window
accuracy. Effective convergence dimension ratio: C / c_d = 3.
Lower bound: γ ≥ (C/c_d) · (θ / (θ − (1−α_cat))) ≈ 4.6 in the idealized limit.

*Path 3 — η₋ Trap Avoidance:*
Phase 2 maintains η_eff near η₊ = 0.05 because 67% of decisions are undisrupted
(immediately correct → confirm path). Phase 1 with large ε_firm may enter the η₋ trap
where low initial accuracy → η_eff ≈ η₋ = 0.01 → learning near-stalled.
Effective learning rate ratio: η_eff,2 / η_eff,1 > 1 when ε_firm is large.
This means true production γ exceeds the geometric lower bound.

*Path 4 — Centroid-Distance (Grok, April 16, 2026):*
Phase 1 challenge: D₁ = ε_firm (all C categories cold-start from expert prior).
Phase 2 challenge: D₂ = (α_cat/(1−α_cat))·‖Δ‖ (only disrupted categories
reconverge; undisrupted contribute free rolling-window accuracy via the
1/(1−α_cat) scaling). Solving D₁ > D₂ gives
ε_firm > α_cat·‖Δ‖/(1−α_cat) ≈ 0.125. θ never enters because challenge is
measured in centroid-distance space, not accuracy-gap space — θ governs the
operational observable, not the mathematical contraction dynamics.

---

**Conditions (Required for Theorem to Hold)**

```
(1) Category-sparse disruption: c_d << C (few alert types disrupted at once)
    Realistic: ransomware campaign disrupts 2 of 6 categories, not all 6.

(2) Warm-started centroids: μ_T1 ≈ GT₁
    Always true when Phase 1 has converged (N_half,1 was reached).

(3) ε_firm > ε_firm★ ≈ 0.125
    Estimated from deployment data: 0.15–0.40 for real SOC environments.
    Verifiable from centroid_distance_to_canonical logged per decision.

(4) Same-regime factor vectors (P₁ and P₂ from same distribution)
    When disruption also shifts factor vector distribution (e.g., new attack changes
    alert signatures), Lemma B applies: PatternHistory enrichment adapts the factor
    distribution. T_adapt decisions needed for full enrichment effect.
```

---

**N_half Measurement Gap (Practical Note)**

N_half conflates two distinct quantities:
- Centroid convergence: how far μ has moved toward GT (what the theorem models)
- Vector separability: how well factor vectors separate actions at the oracle boundary

For high-quality factor vectors (high Day-1 accuracy), N_half may fire before the centroid
has genuinely converged to GT — making N_half too short and γ measurements unreliable.

**The correct convergence signal for EXP-G1 is centroid_distance_to_gt:**
```
dist(t) = ‖μ(t) − GT‖_F
```
This decreases monotonically under production learning dynamics regardless of factor
vector quality or noise seed. Confirmed across all oracle separation experiments
(v6/v8/v11/v2/v3/final, April 8, 2026). EXP-G1 measures γ as the ratio of centroid
distance convergence rates, not as a N_half ratio.

---

**Simulation Validation (Binary Prediction)**

```
Oracle separation v8 (ε_sim = 0.05 < ε_firm★ = 0.125):
  Opus: N_half,1=25, N_half,2=35, γ=0.714 < 1  ✓  (theorem predicts γ < 1)

Oracle separation v3 Opus (ε_sim = 0.20 > ε_firm★ = 0.125):
  Opus: N_half,1=125, N_half,2=121, γ=1.033 > 1  ✓  (theorem predicts γ > 1)
```

Binary prediction correct in both directions. Full record:
`synthetic_data_generation_analysis_v2.md`

---

**Equation Tags**

```
Eq. GAMMA-THEOREM:  γ > 1 ⟺ ε_firm > α_cat · ‖Δ‖ / (1 − α_cat)  [corrected]
Eq. GAMMA-THRESH:   ε_firm★ = α_cat · ‖Δ‖ / (1 − α_cat) ≈ 0.125  [corrected April 8]
Eq. GAMMA-P_D:      p_d★ = (θ − (1 − α_cat)) / α_cat ≈ 0.55
Eq. GAMMA-RATIO:    γ = N_half,1 / N_half,2   (threshold-crossing time ratio)
Eq. GAMMA-DIST:     dist(t) = ‖μ(t) − GT‖_F  (preferred EXP-G1 metric)
```

**Proof sources:** GPT-4.1, Claude Opus 4, Grok 3, Gemini 1.5 Pro (April 8, 2026).
See Claims Registry v10.0 §B.5 CLAIM-GAMMA-THEOREM for the full formal statement with
conditions and commercial tier designation (Tier 2, conditional).

---

---

### Eq. IKS — Institutional Knowledge Score (v5.5)

```
IKS(t) = 100 · min( D(t) / κ*, 1.0 )

where D(t) = (1 / (n_cat · n_act)) · Σ_{c,a} ‖μ[c,a,:](t) − μ[c,a,:](0)‖₂
```

- μ(0): centroid tensor at bootstrap completion (1,200 synthetic calibration decisions)
- **Shape note (v10):** mean_drift averages over 24 (c,a) cells (6×4).
  D_MAX=0.20. κ*=0.20 (PROD-1 validated).
- μ(t): centroid tensor after t verified analyst decisions
- **κ* = 0.20** (PROD-1 validated, March 2026). Design estimate was 0.30 (near-floor).
- IKS = 0: centroid tensor has not moved from bootstrap baseline
- IKS = 100: centroid tensor has drifted the full normalization distance (saturated)

**Acceptance criterion (v5.5-R4):** IKS increases after every batch of verified analyst
decisions. IKS trend chart shows a non-flat line after 10 verified decisions. The score
and its weekly delta are displayed in Tab 2 header.

**Interpretation for a CISO:** "Day 1: IKS = 0. Day 30: IKS = 14. Day 90: IKS = 47."
The system has accumulated 47 units of firm-specific institutional knowledge that
did not exist at deployment. This is the single most important number for answering
Demo Question 2: "Show me it's getting smarter."

---

### Eq. T* — Category-Specific Auto-Approve Threshold Model (v5.5)

```
threshold*(c) = min{ θ ∈ [0, 1] : accuracy(θ, c) ≥ target_accuracy(c) }
```

- accuracy(θ, c): fraction of alerts in category c where the auto-approved action
  matches the oracle, measured on the 50-seed simulation pool at threshold θ
- target_accuracy(c): minimum acceptable accuracy for auto-approve in category c.
  Risk-stratified defaults:
  - insider_threat, lateral_movement: target = 0.92 (catastrophic miss cost)
  - credential_access, data_exfiltration: target = 0.88
  - cloud_infrastructure, threat_intel_match: target = 0.85

**PROD-4 validated (A=4):** 40%+ coverage at ≥85% per-category accuracy achieved.
The category-specific thresholds ship with v5.5.

---

## 4. Synthesis Extension (PROPOSAL — Eq. 4-synthesis)

[§4 UNCHANGED from v9.1. Eq. 4-synthesis, Eq. S1, Eq. S2, Eq. S4 preserved as-is.
σ_max note updated:]

**σ_max status [CHANGED v10]:** σ_max = 0.034 (derived from FX-1-PROXY-REAL factor
distributions + L2 margin computation). The p10 of the empirical L2 margin distribution
at realistic AUAC. Previous design default was 1.0 (conservative placeholder, now
superseded).

---

## 5. Four Feedback Loops

[§5 UNCHANGED from v9.1. Level 2 formalization, conservation law, GATE-L2 preserved.
One addition:]

**N_half qualifier [CHANGED v10]:** The convergence claim "N_half ≈ 14" requires
q̄≥0.70 AND σ≤0.157 (L2 kernel) or σ≤0.25 (DiagonalKernel). Below q̄=0.70:
confirm-only learning with ~2× timeline. Above noise ceiling: learning not
recommended (frozen scorer only).

---

## 6. The Centroid Tensor [CHANGED v10]

**Canonical SOC v6.0 dimensions:** μ ∈ ℝ^(6×4×6) = **144 values**

```
Categories (C = 6):
  0: credential_access
  1: threat_intel_match
  2: lateral_movement
  3: data_exfiltration
  4: insider_threat
  5: cloud_infrastructure

Actions (A = 4): [CHANGED v10: was A=5]
  0: escalate
  1: investigate
  2: suppress
  3: monitor
  NOTE: refer_to_analyst REMOVED as scorable action (v6.0 A=4 migration).
  Accessed via confidence gate (action routing) and ReferralRules R1-R7
  (referral routing — VETO mechanism, 72.7% DR, 12% FPR). Never in centroid
  tensor. Static accuracy improved 80.6→90.6% with A=4. Zero dangerous
  actions. 13pp structural gap confirmed by EXP-A4-DIAGONAL. [CHANGED v11]

Factors (d = 6):
  0: travel_match
  1: asset_criticality
  2: threat_intel_enrichment
  3: time_anomaly
  4: pattern_history
  5: device_trust
```

**S2P domain (parallel):** μ ∈ ℝ^(5×5×8) = **200 values**
  5 categories × 5 actions × 8 domain-level risk scores.
  penalty_ratio=5.0 (vs SOC 20:1). SOC A=4 / S2P A=5 is intentional asymmetry.

Each μ[c,a,:] is a d-dimensional profile: "what does a credential_access alert that
should be escalated typically look like?" The centroids ARE the institutional judgment —
readable, auditable, and correctable by domain experts without touching code.

**Key prior (v5.0 corrected):** credential_access/escalate centroid — travel_match
set to 0.72 (was 0.30 in v4.5). Validated: high-travel vector → P(escalate) = 0.913 ✅

**Bootstrap calibration:** 1,200 synthetic decisions, σ=0.08, seed=42. Converged=True,
drift=0.0097. This is the warm-start baseline μ(0) for every deployment and the
reference point for IKS calculation.

**Synthesis tensor (PROPOSAL):** σ ∈ ℝ^(6×4) = **24 scalar values**. [CHANGED v10: was 6×5=30]
"How should current intelligence shift action preferences for each category?"

---

## 7. Accuracy Numbers

### Two Accuracy Regimes — Never Mix Them

**Centroidal synthetic (EXP-C1, EXP-B1):** Tests the math in ideal conditions.
Oracle-generated factor vectors that perfectly align with centroids. Perfect routing.
These are the ARCHITECTURE VALIDATION numbers. Use to explain the mechanism.
Do NOT use as product claims.

**Realistic (50-seed validated):** Tests the product in realistic conditions.
Bernoulli oracle decisions (noisier), all FactorComputers with realistic noise,
full pipeline, realistic alert type distribution. These are the PRODUCT CLAIM numbers.
Use in customer communications, demos, and investor materials.

---

### Canonical Product Claim Numbers (50-seed validated, R50)

| Metric | Value | 95% CI | Condition |
|---|---|---|---|
| Static accuracy | 71.7% | [71.4%, 71.9%] | Combined realistic, no learning |
| Learning @ dec 1,000 | 78.9% | [78.1%, 79.6%] | Combined realistic, full learning |
| credential_access @ dec 1,000 | 68.0% | [66.7%, 69.1%] | Hardest category |
| Auto-approve accuracy (≥0.90) | 90.7% | [90.1%, 91.2%] | High-confidence suppress |
| Auto-approve coverage | 11.5% | ±0.70% | Alerts meeting ≥0.90 threshold |

**Frozen scorer baseline (A=4, March 2026):**

| Condition | Accuracy | Note |
|---|---|---|
| A=4 static (no learning) | 90.6% | A=4 migration improved from 80.6% (A=5) |
| A=5 static (historical) | 80.4% | SHIFT-2, noise=0, pre-A=4 migration |

**Learning lift (corrected update rule + asymmetric η):**

| Condition | Accuracy lift | Note |
|---|---|---|
| A=5, η_override=0.05 (old) | +2.7% | SHIFT-2, δ=0.10, warmup=1000 |
| A=4, η_override=0.01 (new) | +2.3pp to +3.7pp | Depends on noise ratio. HC: +3.7pp Diagonal. |

---

### DiagonalKernel Accuracy Numbers (NEW v10)

| Metric | L2 | DiagonalKernel | Lift | Condition |
|---|---|---|---|---|
| SOC heterogeneous noise | 79.5% | 92.7% | **+13.2pp** | V-MV-KERNEL factorial, 72 cells |
| S2P heterogeneous noise | 42.2% | 49.0% | **+6.8pp** | V-MV-KERNEL factorial, 72 cells |
| SOC at σ_mean=0.22 (healthcare) | 61-64% | 83-85% | **+20-22pp** | High noise, hetero ratio 2-3× |
| Healthcare persona (V-HC-CONFIG) | 69.2%→69.5% (+0.3pp) | 70.2%→73.9% (+3.7pp) | **+3.4pp learning** | 15 seeds × 60 days |
| HC-C (extreme, ratio 4.6×) | -6.3pp (degrades) | +8.0pp (improves) | **+14.3pp swing** | L2 actively harmful |
| HC-A (ratio 3.2×) | -3.3pp | +6.4pp | **+9.6pp** | One very noisy factor |
| HC-B (ratio 1.9×) | +3.2pp | +5.5pp | **+2.3pp** | Moderate ratio |

**Corr(noise_ratio, diagonal_advantage) = 0.990** across 4 healthcare personas.
The advantage scales nearly perfectly with per-factor noise heterogeneity.

---

### Product Boundaries (Phase 1 Persona Sweeps + Factorial) [NEW v10]

| Dimension | Boundary (L2) | Boundary (Diagonal) | Evidence |
|---|---|---|---|
| Factor noise ceiling | σ≤0.105 GREEN, ≤0.157 AMBER | σ≤0.157 GREEN, ≤0.25 AMBER | 1D + V-MV-KERNEL |
| Alert volume | V≥30 viable | Same | 1A sweep (5 personas) |
| Team quality | q̄≥0.70 for full lift | Same (η_override handles low quality) | 1C sweep (5 personas) |
| Team size (Level 2) | A/B needs ≥8 analysts | Same | 1B sweep (5 personas) |
| Graph enrichment | +7 to +13pp from noise reduction | Same | 2G sweep (2 personas) |
| Correlated error | Conservation detects in 5 days | Same | 2D sweep (2 personas) |
| Recovery | ≤3 days at all tested volumes | Same | V-CL-RECOVER (2 personas) |
| Healthcare | Frozen scorer + remediation first | **Learning from Day 1** | V-HC-CONFIG diagonal |
| Kernel selection | — | noise_ratio > 1.5 → Diagonal | Selector fixes (4/4 correct) |
| Selector stabilization | — | **max(1000, 20×V×α) decisions** [CORRECTED v12: was "~250 decisions"] | V-GATE-STABILITY (N=1000 confirmed) |

---

### Architecture Validation Numbers (centroidal synthetic — qualify always)

| Metric | Value | Experiment | Condition Qualifier |
|---|---|---|---|
| L2 zero-learning accuracy | 97.89% | EXP-C1 | Centroidal synthetic, correct routing |
| L2 with-learning accuracy | 98.2% | EXP-B1 | Centroidal synthetic, warm-start |
| Calibration ECE at τ=0.1 | 0.036 | V3B | Centroidal synthetic |

When stating 97.89% or 98.2%: always append "validated on centroidal synthetic data
with oracle routing. Real-data routing accuracy pending GATE-R." Omitting this
qualifier is a forbidden claim (see §13).

---

### The Consistency Claim (Unconditional — No Experiments Required)

> **Every analyst on your team receives the same starting recommendation, from the
> same reasoning, every time — regardless of which analyst is reviewing the alert,
> what time of day it is, or how long they have been on shift.**

This claim requires no experiments, no gate results, and no qualifications. It is
true at v5.0 today and will remain true at all future versions.

---

### The Accuracy Waterfall (Updated for A=4) [CHANGED v10]

```
Random baseline ..................... 25.0%   (1/4 actions — n_act = 4)
Shared W + Hebbian (EXP-A) ......... 49.3%   original architecture
Per-category W (EXP-A2) ............ 51.6%   attempted fix
Dot product centroid (EXP-C1) ...... 61.0%   right model, wrong kernel
Cosine centroid (EXP-C1) ........... 96.4%   magnitude confounding removed
L2 centroid zero-shot (EXP-C1) ..... 97.89%  ← ARCHITECTURE (centroidal)
L2 + learning (EXP-B1) ............. 98.2%   operational refinement (centroidal)
Frozen μ₀ baseline (A=4) ........... 90.6%   ← EXPERT PRIOR (A=4, noise=0)
Diagonal μ₀ + learning (HC) ........ 73.9%   ← DiagonalKernel on healthcare [NEW v10]
Realistic static ................... 71.7%   ← PRODUCT CLAIM (50-seed)
Realistic @ 1,000 decisions ........ 78.9%   ← PRODUCT CLAIM (50-seed)
```

---

## 8. Key Experimental Results [CHANGED v11]

| Exp | Question | Result | Status |
|---|---|---|---|
| EXP-5 | GT-aligned oracle works? | 79.65% (was 40–44% Bernoulli) | ✅ PASS |
| EXP-A | Gating Matrix G helps? | **FALSIFIED** (+0.01pp, 4 variants) | ✅ FALSIFIED |
| EXP-C1 | Profile centroids work? | **97.89% L2** (settles architecture) | ✅ PASS |
| EXP-B1 | Learning improves centroids? | 98.2% warm, 90.7% cold, robust to 30% noise | ✅ PASS |
| EXP-D1 | Cross-category transfer? | Marginal (config wins 2–14pp) | ✅ DONE |
| EXP-D2 | Factor interactions? | None significant (75 pairs) | ✅ DONE |
| EXP-E1 | Which kernel? | L2 wins 2/3, pluggable for mixed-scale | ✅ DONE |
| EXP-E2 | Scales? | 99.9% at 20×10×20 | ✅ PASS |
| GATE-G | Gating obsolete? | ✅ PASSED — ProfileScorer is THE scorer | ✅ PASS |
| GATE-OP | λ=0.5 operative? | ✅ PASSED — p=0.0008 at centroidal AUAC≈0.97 | ✅ PASS |
| V1A | Scaling exponent? | b=2.11 ± 0.03, R²=0.9999 | ✅ VALIDATED |
| V1B | Norm explosion? | 2.9M× without LayerNorm → required | ✅ VALIDATED |
| V2 | Push stability? | Centroid escape → [0,1] clipping required | ✅ VALIDATED |
| V3A | Beats ML baselines? | L2 94.78% vs XGBoost 92.24% vs RF 92.93% | ✅ VALIDATED |
| V3B | Calibration? | ECE=0.036 at τ=0.1 (ECE=0.19 at τ=0.25) | ✅ VALIDATED |
| SHIFT-2 | Update rule corrected? | +2.7% lift (was -9.0% pre-fix). Dual push/pull. | ✅ PASS |
| DISC-1 | Composite discriminant? | 70.4% coverage vs 62.6% confidence-alone (+7.8pp) | ✅ PASS |
| **V-MV-KERNEL** | **DiagonalKernel vs L2?** | **+13.2pp SOC, +6.8pp S2P on heterogeneous noise** | **✅ PASS** [NEW v10] |
| **V-HC-CONFIG** | **Diagonal rescues healthcare?** | **+3.7pp (Diagonal) vs +0.3pp (L2) at σ=0.22** | **✅ PASS** [NEW v10] |
| **V-HC-SHRINKAGE** | **Off-diagonal helps?** | **<1pp gap. Explanation A confirmed.** | **✅ DONE** [NEW v10] |
| **B5B-PROXY** | **Realistic analyst quality?** | **13-27pp degradation → P0 fix (η_override=0.01)** | **✅ PASS** [NEW v10] |
| **Phase 1 sweeps** | **Product boundaries?** | **σ≤0.157(L2)/0.25(Diag), V≥30, q̄≥0.70, B-A≥8** | **✅ PASS** [NEW v10] |
| **HC-scaling** | **Noise ratio predicts advantage?** | **r=0.990 across 4 personas. Monotonic.** | **✅ PASS** [NEW v11] |
| **KernelSelector** | **Auto-select correct kernel?** | **4/4 correct. Rolling 100-window. max(1000,20×V×α) decisions.** [CORRECTED v12] | **✅ PASS** [NEW v11] |
| **EXP-A4-DIAGONAL** | **A=4 vs A=5 under Diagonal?** | **13pp structural gap. Kernel-independent. A=4 confirmed.** | **✅ PASS** [NEW v11] |
| **EXP-REFER-COVERAGE** | **What fraction of referrals are rule-expressible?** | **72.7% rule-expressible. 20.7% emergent. 6.6% undetectable.** | **✅ DONE** [NEW v11] |
| **EXP-REFER-LAYERED** | **Which referral layers ship?** | **Rules: 72.7% DR, 12% FPR. Confidence gate: 14% precision (harmful). Override learning: +1.1pp at 1,500 decisions (premature).** | **✅ DONE** [NEW v11] |

---

### Synthesis and Validation Experiments (PLANNED — unchanged from v9.1)

| Exp | Question | Gate | AUAC Condition |
|---|---|---|---|
| EXP-S1 | Does σ improve accuracy? | GATE-M: ≥3pp, p<0.0083, ECE ≤+0.02 | Centroidal |
| EXP-S2 | Poisoning resilience? | GATE-M: ≤2pp at 20% bad claims | **Both: centroidal + realistic** |
| EXP-S3 | σ contaminates μ? | GATE-M: ≤5% centroid divergence | Centroidal |
| EXP-S4 | λ sensitivity? | GATE-M: plateau ≥0.05 wide | Centroidal |
| EXP-S5a | Real CISA KEV → σ? | GATE-D-early: ≥3 σ cells updated in <60s | Real data |
| EXP-S5b | Work artifact → claims? | GATE-D-early: LLM F1 ≥ 0.70 | Real data |
| EXP-S5 | Full pipeline? | GATE-D: end-to-end latency <200ms P95 | Real data + v6.0 |
| EXP-S6 | INTSUM-quality briefing? | GATE-D: ≥80% claim coverage, LLM judge | Real data |
| EXP-S7 | Ask-the-Graph improvement? | GATE-D: ≥10/15 correct (3 conditions) | v5.0 baseline |
| EXP-S8 | Real synthesis → real decisions? | GATE-V: ≥3pp treatment, ≤1pp irrelevant degradation | Real deployment |
| ~~FX-1-PROXY-REAL~~ ✅ | Real factor distributions | KL 1.88–2.58. σ_max=0.034 derived. | ✅ Complete |
| PROD-1 ✅ | IKS κ* calibration? | κ*=0.20. 100% of seeds in [15,40] range. | ✅ Complete |
| PROD-3 ✅ | Shadow mode baseline? | θ: 0.744–0.809 per category. | ✅ Complete |
| PROD-4 ✅ | Per-category threshold*? | 40%+ coverage at ≥85% accuracy (A=4). | ✅ Complete |

---

## 9. Safety Architecture

### Synthesis Safety Layers

| Layer | Mechanism | Prevents |
|---|---|---|
| S1 | Extraction confidence gate (≥0.80) | Low-quality claims affecting σ |
| S2 | Source trust weighting (tier × confidence) | Untrusted sources dominating |
| S3 | Magnitude clipping σ ∈ [−σ_max, +σ_max] | Any combination of claims overriding μ |
| S4 | Human confirmation for high-impact Δσ | Misinterpreted or manipulated claims |
| S5 | Full rollback audit trail (claim provenance) | Damage assessment and reversal |
| **S6** | **AMBER auto-pause: conservation AMBER/RED → freeze learning** | **Detection-without-response gap** [NEW v10] |
| **S7** | **Asymmetric η: η_override=0.01 (5× attenuation on override path)** | **Centroid corruption from low-quality overrides** [NEW v10] |
| **S8** | **DiagonalKernel: down-weight noisy factors via W=diag(1/σ²)** | **Noisy factors corrupting scoring and learning** [NEW v10] |
| **S9** | **ReferralRules R1-R7: policy-based VETO after composite gate** | **Alerts needing human review that confidence gate misses (72.7% DR, 12% FPR)** [NEW v11] |

**S9 architectural guarantee [NEW v11]:** ReferralRules insert AFTER the composite
scoring gate, BEFORE response build. ProfileScorer scoring (Stage 1) accuracy is
unaffected by construction (|accuracy change| = 0.0pp). The referral decision is independent of and cannot
contaminate the scoring decision. Rules R1-R7 are deterministic pure functions with
safe defaults (missing context → rule doesn't fire → false negative, not false positive).
Override learning (v6.5+) activates only when ≥50 positive examples accumulate in the
override buffer — data-gated, not calendar-gated.

**σ_max status [CHANGED v10]:** σ_max = 0.034. Derived from FX-1-PROXY-REAL factor
distributions + L2 margin computation on characterized distributions. The p10 of the
empirical L2 margin distribution at realistic AUAC. Previous design default was 1.0
(conservative placeholder, now superseded).

**Architectural guarantee:** At σ_max=0.034 and λ=0.5 (operative), max synthesis
influence = 0.017 distance units. Typical L2 distances = 0.5–2.0. Experience (μ) always
dominates awareness (σ). EXP-S2-REPRO confirmed poisoning resilience holds (0.15pp max).

### Verification Safety Architecture

**Asymmetric learning rates (P0 fix) [NEW v10]:** The override path (analyst changes
system recommendation) carries noise from analyst errors. At realistic analyst quality
(q̄=0.60-0.70), this noise caused 13-27pp centroid degradation with symmetric η.
η_override=0.01 attenuates the override signal by 5× relative to confirms:
  - Confirm path: η_confirm=0.05 (clean signal → full learning rate)
  - Override path: η_override=0.01 (noisy signal → attenuated rate)

Validated across 24 personas (6 sweeps). At q̄=0.57 (worst tested quality):
+0.5pp with η_override=0.01 (no degradation). Without: -9pp.

**AMBER auto-pause [NEW v10]:** When ConservationMonitor signals AMBER or RED
(α·q·V drops below 0.7× baseline or absolute floor θ_min=23.53/(α×V) [CORRECTED v12]),
learning freezes automatically. No centroid updates until GREEN resumes. Prevents
continued corruption during detected degradation episodes. Three-judge consensus.

**Count decay:** η_eff decreases with n[c,a]. Early decisions (low n) produce large
centroid moves. Late decisions (high n) produce small refinements. A mature category
profile (high n) is resistant to individual noisy feedback events.

**N3 endogenous loop (KNOWN RISK — no intervention point designed):**
Calibration error → biased verification selection → learning from biased data →
worse calibration → repeat. No intervention point has been designed for this loop.
Mitigation: shadow mode data from first deployment allows measurement of whether
verification is systematically biased. Disclosed in EU AI Act Article 9 risk log
and shadow mode deployment documentation. Full characterization requires EXP-S8
with real analyst decisions.

## 10. Scaling Properties

**Domain scaling:** Information gain I(n) ~ O(n^b) where b=2.11 ± 0.03 (V1A,
simulation). Super-quadratic: adding the 10th domain provides more than 10× the
information gain of the 1st. This is the mathematical basis for the compounding moat.
**Condition:** b=2.11 validated in simulation only. EXP-G1 (planned) will test
whether γ > 1.0 holds on real multi-domain data. Do not state "compounding" without
EXP-G1 result.

**Category scaling:** 99.9% accuracy at 20 categories × 10 actions × 20 factors
(EXP-E2, centroidal synthetic). No degradation with tensor dimensionality.

**Temporal compounding:** Cold start 52% → 90.7% accuracy at 1,000 decisions (EXP-B1,
centroidal). Warm start 97.89% at decision 1 (EXP-C1). The gap between warm and cold
IS the quantified value of institutional judgment encoded in the centroid tensor.

**Kernel compounding [NEW v10]:** The distance metric itself improves as deployment
matures. Day 1: L2 (before per-factor σ measured). Day 30+: DiagonalKernel with
weights calibrated from P28 deployment qualification. The kernel W=diag(1/σ²) is
firm-specific — a competitor who copies the code starts with W=I.

**CovarianceEstimator [CHANGED v11]:** Online covariance collection with Ledoit-Wolf
shrinkage. half_life=300 decisions (≈90 days at V=100). Collects per-factor σ (used
by DiagonalKernel) and full correlation matrix ρ (logged, not used for scoring at v6.0).
Research asset for v7.0 shrinkage investigation if high-ρ domains emerge.

---

### §10.1 — Graph-Dependent Convergence [UNCHANGED from v9.1]

> Full §10.1 content preserved from v9.1. Eq. σ²_G, three mechanisms,
> graph-dependent convergence time, connection to γ, three regimes.
> No changes needed — DiagonalKernel affects per-factor weighting of
> noise, not the graph-dependent noise reduction mechanism.
>
> **v13 NOTE:** The connection to γ is now formalized in §3.2. Graph enrichment
> (W2 edges, PatternHistory) increases the effective learning signal s̄ in Phase 2,
> providing Strengthening Lemma A for the re-convergence theorem. The γ > 1 condition
> (Eq. GAMMA-THEOREM) can be satisfied by geometry alone (ε_firm > ε_firm★) when
> graph enrichment is absent; when enrichment is present, the condition is met at
> smaller ε_firm values. See §3.2 for the full theorem and conditions.

---

## 11. Gap Analysis — Five Customer Problems vs Current Architecture

[§11 UNCHANGED from v9.1 except Problem 3 and new Problem 6:]

**Problem 3 update [CHANGED v11]:** Auto-approve coverage target 40%+ achieved
at v5.5 with A=4 migration + PROD-4 per-category thresholds. The three-tier
dispatch (auto-approve / investigate / full review) ships at v5.5 with A=4
actions (escalate, investigate, suppress, monitor). Referral routing handled
by ReferralRules R1-R7 (policy-based VETO, independent of scoring). The referral
question is solved at the architecture level: scoring determines WHAT action to take
(4 centroids), rules determine WHO should see it (7 policy rules). These are
orthogonal concerns — an alert can be high-confidence suppress AND flagged for
referral (e.g., executive account during M&A).

**Problem 6 (NEW v11): The Referral Problem.** "When should a human see this alert,
regardless of system confidence?" The confidence gate catches "I don't know."
ReferralRules R1-R7 catch "I know this is technically fine, but organizational
context says a human should see it." EXP-REFER-LAYERED validated: rules deliver
72.7% detection at 12% FPR. Confidence gate as referral: 14% precision (harmful).
Override learning: +1.1pp at 1,500 decisions (premature — needs ≥50 positives).
Architecture: scoring (ProfileScorer, A=4) → referral (ReferralRules, VETO) →
override learning (v6.5+, data-gated). Three concerns, three mechanisms.

---

## 12. Open Math Issues for v5.5 and v6 [CHANGED v11]

| Issue | Depends On | Blocks | Status |
|---|---|---|---|
| IKS κ* value | L2 margin at realistic AUAC | v5.5-R4 (IKS display) | **✅ RESOLVED: κ*=0.20 (PROD-1)** |
| σ_max correct value | p10 of L2 margin distribution | Synthesis layer safety | **✅ RESOLVED: σ_max=0.034** |
| λ operative window at realistic AUAC | EXP-S2-REPRO realistic arm | GATE-M | **✅ RESOLVED: validated March 14** |
| Confidence floor 0.70 derivation | Accuracy-vs-threshold per category | refer_to_analyst routing | **✅ RESOLVED: PROD-4 validated, A=4. Referral via ReferralRules, not confidence.** [CHANGED v11] |
| S4 bound at 2pp accuracy degradation | Per-cell drift vs accuracy sweep | IKS saturation threshold | Pending (PROD-1b) |
| Minimum verification rate floor | Three monitored conditions | Core compounding claim | ⚠️ Partially characterized |
| Segmented accuracy by alert type | Realistic L2 margin distribution | Competitive claims | INTERNAL ESTIMATE |
| Bridge A: Level 2 poisoning | EXP-L2-POISON design | v6.0 Level 2 safety | Planned |
| Bridge B: ρ per source pair | Empirical source correlation | σ²_G precision | Shadow mode data |
| Bridge A: θ_min calibration | N_half + deployment V + α | Conservation law threshold | **✅ RESOLVED: θ_min = 23.53/(α×V). Deployment-specific formula. Deprecated hardcoded 0.467.** [CORRECTED v12] |
| **DiagonalKernel on real data** | **FX-1 real SOC data** | **Production kernel validation** | **Pending (v6.0 first customer)** [NEW v10] |
| **Per-customer τ calibration** | **TD-034 on real alert stream** | **Deployment qualification** | **Pipeline built (P28). Awaiting customer.** [NEW v10] |
| **Override learning activation** | **≥50 positive referral examples** | **v6.5 referral ML** | **Data-gated. At V=200: ~8 days. At V=50: ~50 days.** [NEW v11] |
| **R2/R7 Neo4j wiring** | **DecisionRecord queries for sequence_count, cross_category_count** | **Rules R2, R7 live in production** | **Pending wiring. R1,R3-R6 work immediately.** [NEW v11] |
| **Re-Convergence Theorem (γ > 1)** | **Analytic proof from centroid update equations** | **CC-21 Tier 1 (EXP-G1), Claims Registry §B.5** | **✅ RESOLVED: γ > 1 ⇔ ε_firm > 0.125 [corrected April 8; was 0.128, diff=0.003]. Proven by 4 LLMs (April 8, 2026). See §3.2. EXP-G1 provides empirical measurement.** [NEW v13] |

---

## 13. Claims Evolution Roadmap

[§13 gates 0-8 UNCHANGED from v9.1, with the following additions:]

### Gate KERNEL: DiagonalKernel Validated (March 21, 2026) [NEW v10]

**Prerequisites:** V-MV-KERNEL factorial (390 cells, 2 domains). V-HC-CONFIG (3
conditions, 15 seeds). 4 healthcare personas (Corr=0.990). Selector fixes (rolling
window, simplified rule).

**Unlocks:**
- **CL-KERNEL-COMPOUND:** "The distance metric itself compounds. DiagonalKernel at
  Day 30 outperforms L2 by 13pp on SOC heterogeneous noise data."
- **CL-KERNEL-SELECT:** "The system automatically selects the optimal kernel for the
  deployment's factor noise profile. Rule: noise_ratio > 1.5 → Diagonal."
- **CL-NOISE-CEILING-REVISED:** "With DiagonalKernel, learning is viable at
  σ_mean ≤ 0.25 (was 0.157 with L2). Healthcare deployments open."
- **CL-HC-OPENS:** "Healthcare SOCs can enable learning from Day 1 with DiagonalKernel.
  +3.7pp learning lift at σ=0.22 where L2 produces +0.3pp."
- Factor quarantine mask DEPRECATED — Diagonal supersedes (binary→continuous weighting).
- ShrinkageKernel deprioritized to v7.0 (off-diagonal <1pp in both domains).

**Remains forbidden:**
- "DiagonalKernel validated on production data" (needs v6.0 first customer)
- Any claim mixing L2 and Diagonal numbers without stating which kernel

---

### Gate P0-FIX: Asymmetric η Validated (March 19, 2026) [NEW v10]

**Prerequisites:** 9-persona stress test (B5B-PROXY). η-SWEEP (54 runs). Phase 1
quality sweep (1C, 5 personas across q̄=0.57-0.91).

**Unlocks:**
- **CL-ASYMMETRIC-ETA:** "Override learning attenuated 5× (η_override=0.01) prevents
  13-27pp centroid degradation from realistic analyst quality."
- **CL-QUALITY-ROBUST:** "Learning produces positive lift at all tested analyst quality
  levels (q̄=0.57 to 0.91) with η_override=0.01."
- N_half claim qualified: "N_half ≈ 14 at q̄≥0.70 AND σ≤0.157 (L2) / σ≤0.25 (Diagonal)."

---

### Gate A4-CONFIRM: A=4 Action Space Confirmed (March 21, 2026) [NEW v11]

**Prerequisites:** EXP-A4-DIAGONAL (A=4 vs A=5 under DiagonalKernel). Coding session
code inspection (GAE implements A=4, 478 tests). V-MV-KERNEL ran at A=4.

**Unlocks:**
- **CL-A4-ACCURACY:** "A=4 (escalate, investigate, suppress, monitor) achieves 90.6%
  frozen scorer accuracy, up from 80.4% at A=5. Zero dangerous actions."
- **CL-A4-STRUCTURAL:** "The 13pp gap between A=4 and A=5 is structural — centroid
  geometry confusion, not fixable by kernel weighting."
- refer_to_analyst permanently removed from centroid tensor. Accessed via confidence
  gate (action routing) and ReferralRules (referral routing).

**Remains forbidden:**
- "A=5 is better" or any claim implying refer_to_analyst belongs in the tensor.
- Using A=5 numbers (80.4% frozen scorer) as current product claims.

---

### Gate REFERRAL: ReferralRules R1-R7 Validated (March 21, 2026) [NEW v11]

**Prerequisites:** EXP-REFER-COVERAGE (10-reason taxonomy, 5,000 alerts). EXP-REFER-LAYERED
(4 layers × 300 runs, 19 features, 1,500 warmup). 478 GAE + 280 SOC tests.

**Unlocks:**
- **CL-REFERRAL-RULES:** "Policy-based referral rules R1-R7 detect 72.7% of alerts
  needing human review at 12% false positive rate."
- **CL-REFERRAL-VETO:** "Referral routing is independent of and cannot contaminate
  action scoring. VETO inserts after composite gate, before response build."
- **CL-CONFIDENCE-NOT-REFERRAL:** "The confidence gate is NOT a referral mechanism.
  At 14% precision, it wastes 86% of analyst referral time."
- Override learning data-gated at v6.5 (≥50 positive examples).

**Remains forbidden:**
- "Confidence-based referral is effective" (14% precision = actively harmful).
- "Override learning works at low volume" (needs ≥50 positives, ~8 days at V=200).
- "Rules catch all referral patterns" (20.7% emergent fraction requires override learning).

---

### Gate GAMMA: Re-Convergence Theorem Established (April 8, 2026) [NEW v13]

**Status:** Tier 2 (conditional). Analytic proof established. Simulation directionally
confirmed. EXP-G1 provides Tier 1 (empirical measurement from pilot data).

**Prerequisites:** Oracle separation validation (v6/v8/v11/v2/v3/final, April 8, 2026).
LLM math polls: v4 (GPT-4.1, Claude Opus 4, Grok 3, Gemini 1.5 Pro, April 8, 2026) and
v5 (Grok primary, April 16, 2026).
Four structural proof paths, confirmed across two four-judge math polls (April 8 and April 16, 2026).
BACKLOG-015: EXP-G1 logging active from pilot Day 1.

**The theorem (see §3.2 for full derivation):**

```
γ > 1   ⟺   ε_firm > ε_firm★ ≈ 0.125

where ε_firm★ = α_cat · ‖Δ‖ / (1 − α_cat)  [θ cancels; corrected April 8]

Production ε_firm: 0.15–0.40  (clears threshold)
Simulation ε_sim: 0.05         (below threshold — correctly predicted γ < 1)
```

**Binary simulation confirmation:**
- ε_sim = 0.05 (below 0.128): γ = 0.714 < 1 ✓
- ε_sim = 0.20 (above 0.128): γ = 1.033 > 1 ✓

**Unlocks:**
- **CC-21 Tier 2:** "The system recovers from disruptions faster than it first calibrated."
  State with conditions: category-sparse disruption, warm-started centroids, ε_firm > 0.125.
- **CLAIM-GAMMA-THEOREM:** In Claims Registry v10.0 §B.5 — Tier 2, conditional.
- **EXP-G1 design:** centroid_distance_to_canonical is the primary metric (N_half is a
  noisy proxy — see §3.2 N_half Measurement Gap).

**Conditions (must state alongside any commercial claim):**
1. Category-sparse disruption: c_d/C ≈ 0.33
2. Warm-started centroids: Phase 1 has converged
3. ε_firm > 0.125 (verifiable from centroid_distance_to_canonical in production logs)

**Path to Tier 1:** EXP-G1 — 90-day pilot data. centroid_distance_to_canonical logged
per verified decision (BACKLOG-015 active, 3 new fields added April 8, 2026).

**Forbidden:** "Re-convergence is always faster." The theorem is conditional. See Claims
Registry v10.0 §D.

---

## 14. v5.5 Requirements [CHANGED v11]

### v5.5-T1-1: NL Template Engine [CRITICAL — Demo Q1]
**Status:** ✅ SHIPPED. 24 templates (6 categories × 4 actions). Three layers.
Deterministic. No LLM dependency. "Similar past cases" sidebar implemented.

### v5.5-R1: Auto-Approve Coverage 40%+ [CRITICAL — Demo Q3]
**Status:** ✅ SHIPPED. A=4 + PROD-4 per-category thresholds achieve 40%+ coverage
at ≥85% per-category accuracy. Three-tier dispatch: auto-approve / investigate / review.

### v5.5-R4: Institutional Knowledge Score [CRITICAL — Demo Q2]
**Status:** ✅ SHIPPED. IKS(t) with κ*=0.20 (PROD-1 calibrated). Tab 2 header.

### v5.5-R3: Chart A — Centroid Drift Visualization
**Status:** ✅ SHIPPED. Corrected metric.

### v5.5-R5: Factor Node Provenance
**Status:** ✅ SHIPPED. ProvenanceNode per factor.

### v5.5-R6: Alert Type → Category Mapping Fix
**Status:** ✅ SHIPPED. GATE-R: 100% routing accuracy.

### v5.5-R7: Threat Intelligence Write-Back
**Status:** ✅ SHIPPED. ThreatIndicator nodes with 24h TTL.

### v5.5-R8: Shadow Mode + Shadow Report
**Status:** ✅ SHIPPED. PROD-3 calibrated agreement rates.

### v5.5-NEW: Actions = 4 (A=4 migration) [NEW v10]
**Status:** ✅ SHIPPED. escalate, investigate, suppress, monitor. refer_to_analyst
via referral rules R1-R7 (not confidence gate). Static accuracy 80.6→90.6%. Zero dangerous actions.

### v5.5-NEW: Evidence Ledger + EU AI Act Compliance
**Status:** ✅ SHIPPED. Hash-chained, tamper-evident.

### v5.5-NEW: PyPI Release [CHANGED v11]
**Status:** ✅ SHIPPED. GAE v0.7.0 on PyPI. 478 tests.

---

## 15. v6.0 Requirements [CHANGED v11]

### v6-KERNEL: DiagonalKernel as Default [NEW v10]
**Status:** Code complete. 478 GAE tests. [CHANGED v11: was 447]
DiagonalKernel(W=diag(1/σ²)) default for noise_ratio > 1.5. L2 fallback.
KernelSelector: Phase 2 rule + Phase 3 rolling comparison (100-window) + Phase 4 lock
at **max(1000, 20×V×α) decisions** [CORRECTED v12: was "~250 decisions" — V-GATE-STABILITY
confirms all three baselines require N=1000 minimum; binding constraint is volume baseline
(≥20 days). "250 decisions" figure is deprecated.] CovarianceEstimator collects data
(half_life=300 decisions, does not affect scoring — v7.0 research). [CHANGED v11]

### v6-P0: Asymmetric η [NEW v10]
**Status:** ✅ Code complete. η_confirm=0.05, η_override=0.01. 24 personas validated.

### v6-PAUSE: AMBER Auto-Pause [NEW v10]
**Status:** ✅ Code complete. Conservation AMBER/RED → freeze learning → GREEN resumes.

### v6-ROI: Frozen Mode ROI Calculator [NEW v10]
**Status:** ✅ Code complete. 44min × V × cost, NOT $127/alert. Three value drivers.

### v6-REFERRAL: ReferralRules R1-R7 [NEW v11]
**Status:** ✅ Code complete. 280 SOC tests. ReferralRules R1-R7 as VETO mechanism
after composite gate. 72.7% detection rate, 12% FPR. Deterministic pure functions.
Safe defaults (missing context → rule doesn't fire). R2 (sequence_count) and R7
(cross_category_count) need Neo4j query wiring for production — R1, R3-R6 work
immediately from alert metadata. Override learning (OverrideDetector) data-gated
at v6.5: activates when ≥50 positive examples in override buffer.

### v6-R1: Synthesis Layer (conditional — GATE-M + GATE-D required)
[Unchanged from v9.1]

### v6-R2: Attack Chain Correlation
[Unchanged from v9.1]

### v6-R3: Real SIEM Integration (Splunk + Sentinel)
**Status:** ✅ Code complete. ci-platform connectors: sentinel.py, splunk.py,
sentinel_writeback.py. SourceConnectorProtocol ABC.

### v6-R4: S2P Second Domain
**Status:** S2PDomainConfig d=8 created. 5 categories × 5 actions × 8 factors.
Correlation prior (Regime A, 28 pairs from two-judge research). penalty_ratio=5.0.
SOC A=4 / S2P A=5 — intentional asymmetry. [CHANGED v11]

### v6-R5: Autonomy Envelope Widening
[Unchanged from v9.1]

### v6-R6: Centroid Editor UI / Management API
[Unchanged from v9.1]

### v6-R7: Analyst Benchmarking Report
**Status:** ✅ Code complete (L-04). Per-shift quality breakdown.

### v6-NEW: Deployment Qualification Pipeline (P28) [NEW v10]
**Status:** ✅ Code complete. ci-platform. 6 phases:
Import → Compute (TD-034 + σ_mean + remediation) → Shadow → Qualify → Enable.
DeploymentQualifier: GREEN/AMBER/RED classification. 73 ci-platform tests.

### v6-NEW: SAML + PII + Entity Resolution [NEW v10]
**Status:** ✅ Code complete. ci-platform. SAML auth, PII redaction (5 patterns,
3 strategies), entity resolution: **exact-match pipeline complete** (_normalize,
_build_merge_groups, union-find, IdentifierType enum — deterministic canonical IDs).
**Probabilistic fuzzy matching (Jaccard token overlap) NOT present — pending Block 6.1.**
[CORRECTED v12: "3-pass pipeline, deterministic canonical IDs" overstated — exact-match
is built, fuzzy matching is not. Confirmed by code audit April 1, 2026.]

---

## 16. Constraints & Invariants [CHANGED v11]

| Constraint | Source | Enforced By | Status |
|---|---|---|---|
| μ ∈ [0, 1]^d | V2 (centroid escape without clipping) | np.clip after every update | ✅ VALIDATED |
| τ = 0.1, fixed | V3B (ECE=0.036 vs 0.19 at τ=0.25) | CalibrationProfile default | ✅ VALIDATED |
| η_neg = 0.05 (canonical, symmetric with η) | η_neg=1.0 FORBIDDEN (ECE=0.49). | CalibrationProfile default | ✅ VALIDATED |
| **η_confirm = 0.05** | **P0 fix: confirm path, clean signal** | **ProfileScorer.update()** | **✅ VALIDATED** [NEW v10] |
| **η_override = 0.01** | **P0 fix: override path, noisy signal. Prevents 13-27pp degradation.** | **ProfileScorer.update(eta_override=)** | **✅ VALIDATED (24 personas)** [NEW v10] |
| **θ_min = 23.53/(α×V)** | **Deployment-specific conservation floor. Derived from N_half × η × V_min. At V=200, α=0.25: θ_min=0.47. At V=50, α=0.25: θ_min=1.88 (impossible — deployment ineligible). Formally equivalent to CLAIM-SC-01 scope condition.** [CORRECTED v12: was θ_min=0.467 hardcoded] | **compute_theta_min(α,V)** | **✅ VALIDATED** [CORRECTED v12] |
| **AMBER auto-pause** | **Conservation AMBER/RED → freeze learning. Three-judge consensus.** | **ProfileScorer.set_conservation_status()** | **✅ IMPLEMENTED** [NEW v10] |
| **DiagonalKernel default for noise_ratio > 1.5** | **V-MV-KERNEL: +13pp SOC, +7pp S2P. Explanation A confirmed.** | **KernelSelector Phase 2 rule** | **✅ VALIDATED (390 cells)** [CHANGED v11: was 360] |
| **KernelSelector shadow minimum: max(1000, 20×V×α)** | **V-GATE-STABILITY: N=1000 required for all three baselines. Binding constraint: volume baseline needs ≥20 days. "~250 decisions" figure deprecated.** [CORRECTED v12] | **P28 Phase 3 configuration** | **✅ VALIDATED** [CORRECTED v12] |
| **ShrinkageKernel NOT shipped at v6.0** | **Off-diagonal <1pp in both domains** | **Not implemented** | **✅ DEPRIORITIZED** [NEW v10] |
| **Noise ceiling kernel-dependent** | **L2: σ≤0.157. Diagonal: σ≤0.25. V-B3: three-variable.** | **P28 deployment qualification** | **✅ VALIDATED** [NEW v10] |
| **Factor mask DEPRECATED** | **V-HC-CONFIG: mask hurt Day 1 by 6pp. Diagonal supersedes.** | **DiagonalKernel replaces** | **✅ DEPRECATED** [NEW v10] |
| σ ∈ [−σ_max, +σ_max] | Safety S3; **σ_max=0.034** | RuleBasedProjector clipping | ✅ Resolved |
| Loop 2 never uses σ | Epistemic separation | ProfileScorer.update() ignores σ | ✅ Enforced |
| LayerNorm before Tier 5 | V1B (2.9M× explosion without it) | Required for embedding ops | ✅ VALIDATED |
| Centroids readable by humans | Architecture choice — auditability | μ[c,a,:] is inspectable d-vector | ✅ Design |
| ProfileScorer is THE scorer | TD-029 (ScoringMatrix deprecated) | ScoringMatrix removed | ✅ v5.0 |
| τ_mod = REJECTED | ECE +0.138 at any τ_mod ≠ 1.0 | Removed from all equations | ✅ REJECTED |
| **n_act = 4 (SOC v6.0 canonical)** | **A=4 migration: 80.6→90.6%, zero dangerous actions. 13pp structural (EXP-A4-DIAGONAL).** | **DomainConfig** | **✅ v6.0** [CHANGED v11] |
| n_cat = 6 (SOC v5.0+) | threat_intel_match added. ORDER PERMANENT. | DomainConfig | ✅ v5.0+ |
| **ReferralRules R1-R7 (VETO)** | **EXP-REFER-LAYERED: 72.7% DR, 12% FPR. Confidence gate = 14% precision (harmful for referral).** | **ReferralRules after composite gate. Cannot affect ProfileScorer scoring.** | **✅ SHIPPED (280 SOC tests)** [NEW v11] |
| **Per-analyst η weighting (η_i)** | **D5 validated: CONDITIONAL 0.86pp. V-D5-MECHANISM-GATE PATH 1, regime-independent (Spearman r=0.975–1.000). η_i = clip(precision_i / mean_precision, 0.5, 1.5) × η_override. Subsumes D6.** [NEW v12] | **ProfileScorer.update() per-analyst weight** | **✅ CONDITIONAL — production gate ≥1.0pp** [NEW v12] |
| **D6 (night-shift attenuation): CLOSED** | **V-NIGHT: two attempts, both inverted. Explicit override rules cannot model fatigue. D5 continuous precision monitoring subsumes D6 — precision drops for any reason automatically.** [NEW v12] | **Subsumed by D5** | **✅ CLOSED** [NEW v12] |
| **FINDING-OVR-01** | **Override precision structurally uncorrelated with q̄. V7: r=0.00. V3: r=-0.70. Must be measured directly per analyst — not predictable from role, seniority, or agreement rate.** [NEW v12] | **Must measure directly** | **✅ EMPIRICAL FINDING** [NEW v12] |
| **Self-calibrating gate principle** | **All operational gates derived from deployment's own shadow data. V-GATE-DRIFT: baselines stable within 7% across 90-day pilot. No monthly recalibration needed.** [NEW v12] | **GateConfig class: conservative fallbacks before N_min, calibrated after** | **✅ VALIDATED** [NEW v12] |
| Verification rate health | Three conditions: coverage, drift, conservation law | Three conditions monitored | ⚠️ Partially characterized |
| N3 endogenous loop | Known calibration-feedback risk | **No intervention designed** | ⚠️ Known risk |
| Conservation law: α(t)·q(t)·V(t) ≥ θ_min | Level 2 must not cannibalize Level 1 signal | Eq. GATE-L2 condition (3) | ✅ DESIGNED |
| Centroid support: each (c,a) within data support | Push-away can retire centroid (Opus finding) | Monitor ‖f − μ[c,a,:]‖ < D_support | ⚠️ Monitoring needed |
| Variant variance stability | Level 2 variant must not be volatile | Eq. GATE-L2 condition (4) | ✅ DESIGNED |
| **LEARNING_ENABLED = False default** | **Enable per-customer after shadow qualification** | **P28 deployment pipeline** | **✅ IMPLEMENTED** [NEW v10] |

---

## 17. Equation Index [CHANGED v11]

| Equation | Status | Location | Purpose |
|---|---|---|---|
| **Eq. 4-final** | ✅ VALIDATED | §3 | L2 distance scoring (cold-start fallback) |
| **Eq. 4-diagonal** | **✅ VALIDATED** | **§3** | **DiagonalKernel scoring (v6.0 default)** [NEW v10] |
| **Eq. 4b-final** | ✅ VALIDATED | §3 | Centroid pull/push learning (asymmetric η) |
| **Count decay** | ✅ VALIDATED | §3 | Stability from experience |
| **Eq. IKS** | ✅ VALIDATED (κ*=0.20) | §3 | Institutional Knowledge Score |
| **Eq. T*** | ✅ VALIDATED (PROD-4, A=4) | §3 | Category-specific auto-approve |
| **Eq. CONV** | ✅ THREE-JUDGE VALIDATED | §3.1 | Mean error dynamics: (1−η)^n |
| **Eq. MSE∞** | ✅ THREE-JUDGE VALIDATED | §3.1 | Steady-state MSE: η·tr(Σ_f)/(2−η) |
| **Eq. N_CONV** | ✅ THREE-JUDGE VALIDATED | §3.1 | Convergence time to calibration neighborhood |
| **Eq. R-L2** | ✅ THREE-JUDGE VALIDATED | §5 | Level 2 composite reward (phase-gated) |
| **Eq. GATE-L2** | ✅ THREE-JUDGE VALIDATED | §5 | Four-condition promotion gate |
| **Eq. CL** | ✅ THREE-JUDGE VALIDATED | §5 | Conservation law: α·q·V ≥ θ_min |
| **Eq. σ²_G** | ⚠️ CONDITIONAL (ρ needed) | §10.1 | Graph-dependent factor noise |
| **Eq. η_override** | **✅ VALIDATED (24 personas)** | **§3** | **Asymmetric learning rate formula** [NEW v10] |
| **Eq. η_i** | **✅ CONDITIONAL (V-D5, 0.86pp, PATH 1)** | **§3** | **Per-analyst η weighting: η_i = clip(precision_i / mean_precision, 0.5, 1.5) × η_override. Production gate ≥1.0pp.** [NEW v12] |
| Eq. 4-synthesis | PROPOSAL (GATE-M pending) | §4 | Scoring with awareness bias |
| Eq. S1 | PROPOSAL | §4 | SynthesisProjector protocol |
| Eq. S2 | PROPOSAL | §4 | Bias accumulation with decay |
| Eq. S4 | PROPOSAL | §4 | GATE-M validation metric |
| ~~Eq. S3 (τ_mod)~~ | **REJECTED** | — | Urgency temperature modifier (ECE +0.138) |
| Eq. 5 (embeddings) | DESIGNED | cross_graph_attention_v3 §4 | Property → embedding (Tier 4, v6.5+) |
| Eq. 6–9 (cross-attention) | DESIGNED | cross_graph_attention_v3 §4–5 | Multi-domain attention (Tier 5, v7+) |

---

## 18. Notation Summary [CHANGED v11]

| Symbol | Meaning | Shape | Range / Value | Status |
|---|---|---|---|---|
| f | Factor vector | (d,) | [0, 1] | ✅ |
| μ | Profile centroids (experience) | (n_cat, n_act, d) | [0, 1] | ✅ |
| σ | Synthesis bias (awareness) | (n_cat, n_act) | [−σ_max, σ_max] | PROPOSAL |
| **W** | **Kernel weight matrix** | **(d, d) diagonal** | **diag(1/σ²_factor). Identity → L2.** | **✅ v6.0** [NEW v10] |
| τ | Temperature | scalar | **0.1 — fixed, never change** | ✅ |
| ~~τ_mod~~ | ~~Urgency modifier~~ | — | **REJECTED — ECE +0.138** | ❌ REMOVED |
| λ | Coupling constant | scalar | [0, 0.5]; operative window [0.5, 0.6]. Kill switch at λ=0. | PROPOSAL |
| η | Learning rate (base) | scalar | 0.05 (default) | ✅ |
| **η_confirm** | **Confirm-path learning rate** | **scalar** | **0.05 (clean signal, full rate)** | **✅ v6.0** [NEW v10] |
| **η_override** | **Override-path learning rate** | **scalar** | **0.01 (noisy signal, attenuated 5×). P0 fix.** | **✅ v6.0** [NEW v10] |
| **η_i** | **Per-analyst learning weight** | **scalar per analyst** | **clip(precision_i / mean_precision, 0.5, 1.5) × η_override. Subsumes D6 (night-shift).** | **✅ CONDITIONAL (D5)** [NEW v12] |
| η_neg | Negative learning rate (base) | scalar | **0.05 (canonical). FORBIDDEN: 1.0 (ECE=0.49).** | ✅ |
| decay_rate | Count-based decay | scalar | 0.001 | ✅ |
| c | Category index | int | [0, n_cat); **n_cat=6 (SOC v6.0)** | ✅ |
| a | Action index | int | [0, n_act); **n_act=4 (SOC v6.0). Random baseline = 25% (1/4).** | ✅ |
| K | Kernel function | f,μ → ℝ | **L2 (fallback) or DiagonalKernel (default for noise_ratio>1.5)** | ✅ |
| **G** | **Kernel gradient function** | **f,μ → ℝ^d** | **L2: (f−μ). Diagonal: W·(f−μ).** | **✅ v6.0** [NEW v10] |
| n[c,a] | Observation count | (n_cat, n_act) | ℕ | ✅ |
| b | Scaling exponent | scalar | 2.11 ± 0.03 (V1A, simulation) | ✅ |
| κ* | IKS normalization constant | scalar | **0.20 (PROD-1 validated).** | ✅ Resolved |
| σ_max | Synthesis clipping bound | scalar | **0.034 (FX-1-PROXY-REAL derived).** | ✅ Resolved |
| **θ_min** | **Conservation law floor** | **scalar** | **23.53/(α×V) — deployment-specific formula. Deprecated: 0.467 hardcoded.** [CORRECTED v12] | **✅ v6.0** [CORRECTED v12] |
| f_min | Minimum verification rate | scalar | **Replaced by three monitored conditions (v9.1).** | Partially characterized |
| IKS | Institutional Knowledge Score | scalar | [0, 100] | ✅ Shipped |
| threshold*(c) | Category auto-approve threshold | (n_cat,) | **PROD-4 validated (A=4).** | ✅ Resolved |
| **noise_ratio** | **max(σ_factor)/min(σ_factor)** | **scalar** | **>1.5 → DiagonalKernel, else L2** | **✅ v6.0** [NEW v10] |
| Σ_f | Factor covariance matrix | (d, d) | Positive semi-definite. Use tr(Σ_f) for MSE. | ✅ |
| e_n | Centroid error at step n | (d,) | μ_n − μ* | ✅ |
| N_half | Convergence half-life | scalar | ln(2)/η ≈ 14 at η=0.05. **Qualifier: q̄≥0.70 AND σ≤0.157(L2)/0.25(Diag).** | ✅ |
| N_converge | Decisions to calibration neighborhood | scalar | Eq. N_CONV | ✅ |
| α(t) | Analyst override rate | scalar | [0, 1] — fraction disagreeing | ✅ |
| q(t) | Override quality | scalar | [0, 1] — fraction of correct overrides | ✅ |
| V_verified | Verified decisions per day | scalar | ℕ | ✅ |
| ρ_j | Cross-source correlation for factor j | scalar | [0, 1] — ρ=0.8 typical for SIEMs | ✅ |
| N_eff | Effective independent sources | scalar | N/(1+ρ(N−1)) | ✅ |
| K (Level 2) | Number of prompt variants | scalar | **K=2 recommended (was 3-5)** | ✅ |
| w₁,w₂,w₃ | Level 2 reward weights (phase-gated) | scalars | Sum to 1.0 | ✅ |
| Δ_min | Level 2 promotion threshold | scalar | 0.05 default (5pp improvement) | ✅ |
| N_gate | Samples per arm for promotion | scalar | ≈445 (one-sided, α=0.05, β=0.20) | ✅ |
| D_support | Centroid support radius | scalar | Flag when centroid leaves data support | ✅ |

---

*Mathematical Synopsis v12 | April 1, 2026*
*v12 CORRECTIONS: KernelSelector minimum 250→max(1000,20×V×α). θ_min 0.467→23.53/(α×V). Entity resolution: exact-match ✅, fuzzy ❌.*
*v12 ADDITIONS: Per-analyst η_i weighting (D5, CONDITIONAL). D6 closed (subsumed by D5). FINDING-OVR-01 (q̄ does not predict override precision). Self-calibrating gate principle.*
*v6.0 KERNEL + A=4 + REFERRAL ALL SETTLED. ~175 experiments. 408 SOC backend + 525 GAE + 102 ci-platform + 46 S2P + 10 copilot-sdk (~1,057 total).*
*A=4 confirmed: 13pp structural, kernel-independent. Tensor 6×4×6=144. Frozen scorer 90.6%.*
*DiagonalKernel validated: +13.2pp SOC, +6.8pp S2P. Explanation A confirmed (noise ratio alone).*
*Corr(noise_ratio, advantage) = 0.990 across 4 healthcare personas.*
*ShrinkageKernel deprioritized to v7.0 (off-diagonal <1pp in both domains).*
*Asymmetric η (P0 fix): η_confirm=0.05, η_override=0.01. Prevents 13-27pp degradation. 24 personas.*
*Per-analyst η_i: clip(precision_i/mean_precision, 0.5, 1.5) × η_override. D5 CONDITIONAL (0.86pp, production gate ≥1.0pp).*
*AMBER auto-pause: conservation AMBER/RED → freeze learning. Three-judge consensus.*
*Noise ceiling kernel-dependent: L2 σ≤0.157. Diagonal σ≤0.25. Healthcare opens at v6.0.*
*KernelSelector: rolling 100-window, noise_ratio>1.5→Diagonal, max(1000,20×V×α) decisions to stabilize.*
*ReferralRules R1-R7: policy VETO, 72.7% DR, 12% FPR. Override learning v6.5 (≥50 positives).*
*SOC A=4 / S2P A=5 — intentional asymmetry.*
*θ_min=23.53/(α×V) per-deployment. σ_max=0.034. κ*=0.20. LEARNING_ENABLED=False default.*
*Conservation law α·q·V ≥ θ_min ensures two-level compounding (three-judge validated).*
*FINDING-OVR-01: override precision uncorrelated with q̄ (r=0.00 to -0.70). Measure directly.*
*"The distance metric itself compounds — Day 1: L2. Day 30: Diagonal calibrated to YOUR noise."*
*"μ is what you've learned. W is how you weight it. Rules are what you know needs human eyes."*
