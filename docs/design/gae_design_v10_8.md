# Graph Attention Engine — Design & Architecture v10.8

**Date:** April 29, 2026
**Version:** 10.8 (v10.7 + Framework v4 integration: TwoPhaseScorer, batch pipeline, defense-in-depth safety, profile state terminology)
**Authority:** claims_registry_v10.0 · MAP v5.50 · framework v4 (April 29, 2026, post-judge-review)
**Status:** Phase 0 ✅ Phase 1 ✅ Phase 2 ✅ Phase 3 Priority 1 ✅. UNI-DK-01 v5.3 closed (PUBLISH). KernelSelector architecture settled: **Option A (NR-rule) PRIMARY**. Conservation-law q operational (rolling verified accuracy). DK calibration addressed architecturally. **Framework v4 integrated: TwoPhaseScorer + batch pipeline + defense-in-depth safety + profile state terminology. ~115 compounding experiments. 5-judge review.**
**Repository:** graph-attention-engine (standalone, numpy-only, Apache 2.0)
**Scope:**
- v4.1 (Tiers 1-3 foundation)
- v4.5 (CalibrationProfile, per-factor decay)
- **v5.0 COMPLETE** (ProfileScorer + OracleProvider + Evaluation + Judgment + Ablation + Users Guide)
- **v6.0 COMPLETE** (Kernel framework + CovarianceEstimator + KernelSelector + asymmetric η + AMBER auto-pause + ReferralEngine + OLSMonitor + enrichment_advisor — 517 tests)
  - **v0.7.18-v0.7.20**: KERNELSEL-001 tiebreaker + raw_weights fix + Block 9.1-9.5 (CLAIM-66-70) — 884 tests
  - **v0.7.21-v0.7.22-pre**: compute_per_factor_n_half + ProfileScorer.for_soc factory + compute_eta_override worst_case_quality parameter + η_override RuntimeWarning + τ/isfinite guards + consumer contract tests + γ-threshold 0.128→0.125 cleanup
- v6.5 (GainScheduler, Fisher calendar, enforcement mode, OverrideDetector activation, **TwoPhaseScorer + batch pipeline (~550 lines new code + ~200 tests), BatchCompositionPolicy, PromotionGate, BatchHistory, NoveltyTracker with 4 failure modes, profile state terminology alignment**)
- v7.0 (Level 2: GraphAttentionBridge + ShrinkageKernel investigation)
- v8.0 (Level 3: Cross-Domain Discovery)

**Companion repos:**
- ci-platform (93 tests — connectors, onboarding, deployment qualification, entity resolution, PII redaction, SAML — Apache 2.0)
- soc-copilot (288 tests — SOC domain expertise, frozen ROI, hooks, shadow mode, PatternHistoryFactorComputer — proprietary)
- cross-graph-experiments (~180 experiments after UNI-DK-01 v5.3: bridge, validation, OP/synthesis, persona sweeps, factorial kernel studies, Phase 1 closure, UNI-DK-01 v5.3 characterization surface)

**Supersedes:** gae_design_v9, v9.1, v10.0-v10.6 + all prior versions.

---

> **Changes from v10.7 → v10.8 (April 29, 2026 — Framework v4 integration):**
>
> Driven by framework v4 (post-judge-review, 5 LLM judges) and ~115 compounding experiments. No existing v10.7 content invalidated. All additions are forward-looking (v6.5 implementation scope).
>
> **Category A — New sections:**
> (A1) §9.10 TwoPhaseScorer architecture. Phase 1/Phase 2, per-(c,a) state, LearningStrategy interface, ProfileScorer modifications. Experimental evidence: K26 (ρ_variance 18/18), K31 (coord descent), K32 (DK robustness), K33 (centroid divergence), K34 (concentration).
> (A2) §9.11 Batch Pipeline. 7 lifecycle steps, 3 new interfaces. Defense in depth: shrinkage + promotion gate + rollback. Error budget derivation from E1, F7, RATE-5.
> (A3) §9.12 NoveltyTracker. Interface, default implementation (d_nn), 4 failure modes with architectural responses (K38).
> (A4) §12.5 Batch pipeline consumer contracts (4 additions).
> (A5) §4 Experiment table updated (~295 total experiments).
>
> **Category B — Modified sections (15 edits):**
> (B1) Header. (B2) §3 defense-in-depth principle. (B3) §5 profile state framing. (B4) §9.1 w̃ equation. (B5) §9.4 new methods. (B6) §9.5 Phase 2 batch learning. (B7) §10.1 new modules. (B8) §13 equation traceability. (B9) §14.2 11 new design decisions. (B10) §16 5 new production constraints. (B11) §17 v6.5 implementation. (B12) §18 new files. (B13) §10.12 new exports. (B14) ResetPolicy factor-version drift.
>
> **Category C — Terminology (4 mechanical passes):**
> (C1) "Sufficient statistics" → "profile state" (preserve in surrogate). (C2) DK → "discriminative precision weights" (not inverse variances). (C3) Fisher → "Layer A/B" (motivation, not proof). (C4) Channels → "mechanistically distinct, statistically coupled."
>
> **Authority:** framework v4 (April 29, 2026), claims v6.

---

> **Changes from v10.6 → v10.7 (April 19, 2026 — UNI-DK-01 v5.3 incorporation + Codex drift cleanup + v0.7.21/v0.7.22-pre code catch-up):**
>
> Driven by three categories of change:
> (A) Code shipped in v0.7.21 + v0.7.22-pre that v10.6 didn't document.
> (B) Architecture decisions from UNI-DK-01 v5.3 (April 19, 2026) — KernelSelector architecture inverted (rule-based PRIMARY), conservation q-source changed, DK calibration finding.
> (C) Accumulated corrections surfaced by Codex drift analysis (12 items).
>
> **Category A — Code catch-up (v0.7.21 + v0.7.22-pre):**
>
> (A1) **§10.4 compute_per_factor_n_half() documented.** gae/convergence.py:63-72. Returns per-factor N_half under DiagonalKernel. Complements the scalar compute_n_half() for heterogeneous-σ deployments. Public API.
>
> (A2) **§9.1 ProfileScorer.for_soc() factory documented.** gae/profile_scorer.py:284-300. Convenience constructor that sets SOC canonical defaults: eta_override=0.01, auto_pause_on_amber=True, tau=0.1. Note: Constructor-level eta_override default remains None (backward compatibility); for_soc() injects 0.01. The canonical value for SOC deployments is 0.01.
>
> (A3) **§10.11 compute_eta_override() worst_case_quality parameter documented.** gae/calibration.py:354+. New optional parameter lets callers specify worst-case analyst quality explicitly rather than relying on mean_quality alone. Diagnostic formula — global default η_override=0.01 remains the validated shipping value.
>
> (A4) **§9.1 η_override RuntimeWarning documented.** gae/profile_scorer.py:703. If η_override is used outside [0.005, 0.02] range, a RuntimeWarning fires. Prevents accidental drift from canonical value in production.
>
> (A5) **§9.1 safety guards documented.** gae/profile_scorer.py:326, 332, 648. τ guard (raises on τ ≤ 0 or τ > 10) and isfinite guards (raise on non-finite f or μ values after clip). Defense against numerical instability.
>
> (A6) **§9.3 ScoringResult fields corrected.** v10.6 documented fields `action_probabilities`, `selected_action`, `confidence`, `distances`, `factor_vector`, `category_index`, `temperature`, `kernel`, `probabilities`. Live code at gae/profile_scorer.py:77-101 has ONLY: `action_index`, `action_name`, `probabilities`, `distances`, `confidence`. **Doc corrected to match code.** Other fields the v10.6 doc invented (`action_probabilities` alias, `factor_vector`, `category_index`, `temperature`, `kernel`) are removed; consumers use `probabilities` (not `action_probabilities`), and the scorer preserves f/category/tau/kernel internally via other APIs.
>
> (A7) **§9.2 Phase 0 instrumentation documented.** gae/kernel_selector.py:38-40, 272-280. KernelScore dataclass includes `cumulative_analyst_prob: float = 0.0` (shadow-mode analyst-action probability tracking for future log-likelihood metrics). Documented for completeness.
>
> (A8) **§10.14 γ-threshold 0.128→0.125 cleanup.** gae/convergence.py:1087-1100, :1109. Formula is `alpha_cat * delta_norm / (1 - alpha_cat)` (θ cancels; production values → 0.125). v10.6 had 6 sites with 0.128 (Codex drift report item 3). All updated to 0.125 for consistency with math_synopsis v14 and code.
>
> (A9) **§10.11 θ_min formula documented (Grok form).** gae/calibration.py:151-183. `θ_min = η × N_half² / T_max`. At η=0.05, N_half=14, T_max=21 days: θ_min ≈ 0.467. This is the deployment-specific conservation floor. Matches live code and v10.6 — no change, but v10.6 had ambiguous narrative; v10.7 clarifies.
>
> (A10) **§9.5 DiagonalKernel gradient docstring verified.** gae/kernels.py:199-200. `w_max = max(self.weights.max(), 1e-9); return (self.weights / w_max) * (f - mu)`. Formula W/W.max()·(f-μ) remains canonical (GAE-GRADIENT-001 fix from v0.7.7). v10.6 was correct on the formula itself; v10.7 adds the exact code form.
>
> (A11) **§10.4 compute_n_half() exact formula documented.** gae/convergence.py:49-64. `N_half = ln(2) / ln(1/(1-η))`. At η=0.05: N_half ≈ **13.51** (discrete-time exact), NOT 13.86 (continuous approximation ln(2)/η). v10.6 rounded to 14; v10.7 preserves the rounding for general use but also exposes the exact 13.51 for precision-sensitive callers.
>
> (A12) **§12 Consumer contract tests documented.** tests/test_consumer_contracts.py:14-190. 12 test functions covering the stability commitments for downstream consumers (soc-copilot, ci-platform). Public API surface contracts.
>
> **Category B — v5.3 architecture decisions:**
>
> (B1) **§9.2 KernelSelector architecture INVERTED.** Was: "Phase 2 rule-based + Phase 3 rolling 100-window comparison + Phase 4 empirical lock." Now: "Phase 2 rule-based is PRIMARY. Phase 3 shadow comparison collects data for MONITORING only — it does NOT become the selector. No Phase 4 empirical lock."
>
> Rationale (from UNI-DK-01 v5.3 E1/E3/E4):
>   - Rule-based selection: 100% correct at every informative NR in v5.3 (91/91, 2/2, 49/49, 127/127 cells). Combined with V-MV-KERNEL HC-personas 4/4, two independent validations.
>   - Data-driven (trimmed_ll): below chance at NR ≥ 3.0 (0.37–0.53 correctness).
>   - mean_conf-based winner selection: E4 winner_stability_rate=1.000, dk_final_share=0.000 — mean_conf ALWAYS selects L2, never DK, even at NR=5.0 where DK has +7.67pp accuracy advantage. Mechanism: DK's sharpened softmax produces lower mean_conf than L2 despite higher accuracy (see DK calibration finding, B3).
>   - Confidence-based kernel comparison is fundamentally untenable given DK's calibration properties. NR-rule sidesteps the issue entirely.
>
> **Option A (NR-rule) ships for v6.0.** Option B (accuracy-comparison holdout) retained as v6.1 contingency if P28 σ estimation proves unreliable in the field. Option C (calibrated-confidence Platt scaling) archived — no scenario requires it.
>
> (B2) **§10.10 KernelSelector class semantics changed.** `recommend()` still exists but is now documented as a monitoring output, not a selection authority. The production scorer uses `preliminary_recommendation()` (the rule) for selection. `record_comparison()` continues to accumulate shadow data for drift detection and future re-evaluation. MIN_DECISIONS_FOR_RECOMMENDATION=100 retained as the threshold at which monitoring output is emitted (not at which selection switches). Live code value at gae/kernel_selector.py:104 is unchanged; the semantic reinterpretation is documentary — the selection never switches based on record_comparison data in v6.0.
>
> (B3) **NEW §9.6 — DK Calibration Properties.** UNI-DK-01 v5.3 characterized DK's calibration at scale. ECE rises from 0.055 at NR=1.0 to 0.42 at NR=5.0; L2 stays 0.04–0.06. Mechanism: W-concentration (W normalized to max=1) reduces effective dimensionality, sharpens softmax, produces overconfidence on the decisive subset. Property of inverse-variance weighting, not implementation defect.
>
> **Architectural consequence: DK confidence outputs are NOT safe for direct downstream consumption.** Production touchpoints that needed confidence now use alternative signals:
>   - **Conservation-law q:** rolling verified accuracy (§9.8 new, §10.11 updated).
>   - **AMBER auto-pause trigger:** rolling verified accuracy (§9.7 updated, previously documented as confidence-based).
>   - **Analyst triage ranking:** softmax entropy / confidence gap (§9.9 new, spec for SOC Copilot).
>   - **KernelSelector:** NR-rule (§9.2, §10.10). Not confidence.
>
> The architectural pattern: "prediction vs estimation channels." DK is a good prediction channel (accuracy ↑). DK is a poor estimation channel (calibration ↓). Consumers needing estimation use signals that don't depend on raw DK max_p.
>
> (B4) **§9.8 NEW — Conservation-law q operational definition.** q(t) = rolling verified accuracy over last 400 decisions. Replaces previous implicit definition (per-decision confidence). Mathematically equivalent for well-calibrated scorers; robust for miscalibrated ones. Cross-references math_synopsis v14 §5, §9. Derivation check: no v13 derivation step required q = confidence specifically; substitution preserves all guarantees and strengthens EU AI Act Article 14 oversight argument.
>
> (B5) **§9.7 UPDATED — AMBER auto-pause signal source.** Trigger = rolling verified accuracy drops below (baseline × 0.9) or absolute floor. State-level transition (pause AUTO zone, route to human review), not per-decision filter. Resume condition: rolling accuracy recovers above threshold for ≥100 consecutive decisions. v10.6 was ambiguous on signal source; v10.7 is explicit.
>
> **Drift note on auto-pause enforcement:** Codex drift item 14 verified that live code (gae/profile_scorer.py:184-188, :560, :565-568, :662-675) does enforce pause: `if self._paused_by_conservation: ... outcome='paused_conservation'`. centroid_updates are blocked. There is NO doc-code gap here. The coding-session brief flagged "Not wired in SOC backend (DRIFT finding). Fix planned (SOC-Q3)." — that's a SOC BACKEND wiring issue (soc-copilot consuming the paused status), not a GAE issue. Documented as a cross-repo coordination item in §16, not a GAE design gap.
>
> (B6) **§9.9 NEW — Analyst Triage Ranking Specification.** Product-design decision from UNI-DK-01 v5.3 §10.7.1. SOC Copilot's triage UI ranks decisions by softmax entropy (ascending) or confidence gap top_p − second_p (descending) — equivalent for argmax decisions. Raw max_p shown as secondary display with "uncalibrated — DK" badge when DK is routed. Rationale: entropy/gap captures decisiveness without calibration dependency; distance-based ranking was considered and rejected (ranks near-boundary decisions as high-confidence, opposite of triage intent).
>
> **Category C — Accumulated corrections from Codex drift analysis:**
>
> (C1) **γ-threshold consistency swept.** All 0.128 references replaced with 0.125 (6 sites flagged by Codex item 3: lines 252, 1332-1348, 1386-1405, 1484-1492, 1741-1742, 2018, 2109-2113, 2157-2160, 2434). Math_synopsis v14 §3.2 formula ε_firm★ = α_cat · ‖Δ‖ / (1 − α_cat) ≈ 0.125 (θ cancels, corrected April 8) is canonical. Live code gae/convergence.py:1109 `return alpha_cat * delta_norm / (1 - alpha_cat)` matches.
>
> (C2) **N_half per-factor under DiagonalKernel documented.** v10.6 treats N_half as scalar ≈14 throughout. Live code gae/convergence.py:67-72 exposes `compute_per_factor_n_half()` for heterogeneous-σ deployments under DK. §10.4 updated to show both scalar (rounded 14) and per-factor forms, with the exact discrete-time value 13.51 noted.
>
> (C3) **Fourth proof path for γ theorem documented (Grok, April 16).** Centroid-distance derivation. Complements the three structural proof paths in math_synopsis v14 §3.2. Documented as supplementary evidence in §10.14.
>
> (C4) **"~250 decisions" retired.** Three sites (docs lines 192, 513, 1035). Replaced with "MIN_DECISIONS_FOR_RECOMMENDATION=100 monitoring threshold; selection never switches from NR-rule in v6.0." Historical note: v10.6 had carried "max(1000, 20×V×α) decisions" from v12 math_synopsis as a selection-lock threshold. Under v10.7's architecture inversion, there is no selection lock — the rule ships as PRIMARY at every deployment and doesn't switch based on shadow comparison data. The max(1000, 20×V×α) figure is retired from this document. (Math_synopsis v14 continues to cite it as a V-GATE-STABILITY baseline for other gates, which is separate.)
>
> (C5) **Design doc refs v5/v8.3 already updated.** Codex item 17 confirmed no remaining stale references. GAE-02 completed this work in a previous pass.
>
> (C6) **Stale version references cleaned.** Codex item 16 flagged v0.7.0, v0.7.20 references. v10.7 updates to current shipped version range (v0.7.18–v0.7.22-pre) with the v0.7.21 and v0.7.22-pre additions listed in the scope block above.
>
> (C7) **Test counts updated.** 884 tests → 884 tests (12 new consumer contract tests, Codex item 10). GAE test progression reflected in §15 and §17.
>
> **Cross-document coordination:**
> - math_synopsis v14: updated in parallel. Defines q = rolling verified accuracy (§5, §9), KernelSelector Option A architecture (§3, §16), DK characterization surface (§3.3, §7). gae_design v10.7 cross-references math_synopsis v14 for canonical math.
> - innovation_note v6: updated separately. Customer-facing curve-based citation ("up to +13.2pp" + "characterized at NR=5.0 mean-σ=0.175").
> - claims_registry v5: updated separately. B3 row adds v5.3 as second-source citation; Gate UNI-DK-01 added.
> - soc_copilot design v5.7+: will inherit §9.9 triage ranking spec; document update scheduled separately.
> - architecture_philosophy v4.3: will inherit §9.8 q definition from math_synopsis v14; document update scheduled separately.
> - production paper / arxiv paper: optional calibration footnote in next revision; non-gating for v6.0 ship.
>
> **Summary:** v10.6 → v10.7 is a significant but scoped update. Categories A and C are mostly mechanical doc-catches-up-to-code cleanup. Category B is the substantive architectural change — KernelSelector architecture inversion and DK calibration finding — which changes the design, not just the documentation of existing code.

> **Changes from v10.5 → v10.6 (April 8, 2026):**
>
> Post-v10.5 additions carried forward from v10.6 preamble (unchanged):
> - Block 3.2 concrete examples in gae/experiments/ (oracle separation scripts, domain examples, reproducibility scripts)
> - 517→527 test count
> - KERNELSEL-001 tiebreaker, raw_weights fix, Block 9.1–9.5 (CLAIM-66–70)
> - Source file references in §10.15 updated

> **Changes from v10.4 → v10.5 (April 6, 2026):**
>
> Carried forward:
> - §4 experiment table: ~104→~145 experiments (post-Phase 2 closure)
> - §14.2 decisions log: Phase 2 closure entries (DRIFT-DATES, claim-registry alignment)

> **Changes from v10.3 → v10.4 (April 2, 2026):**
>
> Carried forward:
> - §10.15 gae/experiments/ module spec (reproducibility scripts, domain examples, oracle separation demos)
> - §4 experiment table: ~100→~104 experiments
> - 517 test count current

> **Changes from v10.2 → v10.3 (March 28, 2026):**
>
> Carried forward:
> - §17.1b v0.7.18–v0.7.20 additions (517→884 tests: KERNELSEL-001 tiebreaker, raw_weights fix, Block 9.1–9.5 with CLAIM-66–70)
> - §18 pyproject.toml v0.7.20

> **Changes from v10.1 → v10.2 (March 25, 2026 — Phase 1 closure):**
>
> (29) Header updated. 478→517 tests. ci-platform 73→93. soc-copilot 252→288. Experiments ~100→~130. v6.0 marked COMPLETE.
>
> (30) §9.5 DiagonalKernel gradient corrected (CRITICAL). GAE-GRADIENT-001 bug fixed in v0.7.7: DiagonalKernel gradient was `W*(f-mu)`. Correct formula is `W/W.max()*(f-mu)`.
>
> (31) §16 Production Constraints: two additions. DiagonalKernel gradient constraint (W/W.max()*(f-mu) only) and η_neg guard (CalibrationProfile raises ValueError on η_neg ≥ 1.0, GAE 0.7.8 fix).
>
> (32) §10.1 Module Overview updated. convergence.py: OLSMonitor. enrichment_advisor.py added. VarQMonitor noted.
>
> (33) §14.2 Design Decisions: Phase 1 entries added.
>
> (34) §17.1a v6.0 marked COMPLETE. Test progression extended to 517.
>
> (35) §17.2 v6.0 open-source release items updated.
>
> (36) §18 Repo structure updated.
>
> (37) §15.2 test count updated. 478→517.

> **Changes from v9.1 → v10 (March 21, 2026 — kernel architecture settled):**
>
> (15) NEW §9.6: DiagonalKernel equation. P(a|f,c) = softmax(−(f−μ)ᵀW(f−μ)/τ) where W=diag(1/σ²). Validated: +13.2pp SOC, +6.8pp S2P on heterogeneous noise. Corr(noise_ratio, advantage)=0.990 across 4 healthcare personas.
>
> (16) §9.2 KernelType updated. DIAGONAL added. MAHALANOBIS/COSINE deprecated. Kernel selection rule: noise_ratio > 1.5 → diagonal, else L2.
>
> (17) §9.4 ProfileScorer interface expanded. New parameters: `kernel`, `factor_mask`, `eta_override`, `auto_pause_on_amber`.
>
> (18) §9.5 Learning equation updated. Asymmetric η: η_confirm=0.05, η_override=0.01.
>
> (19) NEW §10.9: gae/kernels.py. ScoringKernel protocol, L2Kernel, DiagonalKernel. 28 tests.
>
> (20) NEW §10.10: gae/covariance.py. CovarianceEstimator with Ledoit-Wolf shrinkage.
>
> (21) NEW §10.11: gae/kernel_selector.py. KernelSelector.
>
> (22) NEW §10.12: gae/calibration.py additions.
>
> (23) §10.1 module overview updated.
>
> (24) §4 experiment counts updated.
>
> (25) §14.2 new experiment-driven decisions.
>
> (26) §16 production constraints updated.
>
> (27) §17 test progression updated.
>
> (28) §18 repo structure updated.

> **Changes from v8.3 → v9 (March 9, 2026 — v5.0 complete):**
>
> (1) §10 completely rewritten. All GAE v5.0 Phase 6 modules now exist as live code with tests.
>
> (2) New §10.6: OracleProvider. oracle.py is the ground-truth interface for the evaluation pipeline.
>
> (3) New §10.7: Evaluation. evaluation.py: EvaluationScenario, run_evaluation(), EvaluationReport, compute_ece().
>
> (4) New §10.8: Judgment. judgment.py: compute_judgment(), JudgmentResult, three confidence tiers.
>
> (5) New §10.9: Ablation. ablation.py: run_ablation(), AblationReport, factor importance ranking.
>
> (6) §10.4 ScoringResult field correction. Field is `.probabilities` not `.scores`.
>
> (7) New §15: Open-Source Strategy.
>
> (8) §17 (What's Built vs Next) substantially rewritten.
>
> (9) §18 repo structure updated.
>
> (10) Test count updated.
>
> (11) Realistic accuracy numbers integrated.

> **v9.1 additions (March 12, 2026 — post-tag WIRING-1):**
>
> (12) CentroidUpdate dataclass added.
>
> (13) freeze()/unfreeze() added.
>
> (14) Two-level institutional judgment framing added (§5, §9).

---

## 1. Requirements: Why This Document Exists

[§1 UNCHANGED from v10.6. Purpose: single source of truth for GAE design and implementation. Gates-first philosophy. Traceability from math_synopsis through equations to code. Downstream consumers (soc-copilot, ci-platform) commit against this contract.]

The document answers three questions for each GAE component:
1. What does it do? (equations, interfaces)
2. Why does it do that? (experimental foundation, design decisions)
3. What does it guarantee? (constraints, invariants, test coverage)

**v10.7 note:** Scope extends to cover UNI-DK-01 v5.3 architectural decisions (KernelSelector, q operational definition, DK calibration response) and v0.7.21/v0.7.22-pre code catch-up. Cross-references to math_synopsis v14 for canonical math.

---

## 2. Technology Analysis

[§2 UNCHANGED from v10.6. numpy-only foundation, no framework dependencies. Apache 2.0 license. Standalone design with opt-in integrations (soc-copilot, ci-platform). Python 3.10+ target. No GPU requirement — all operations on CPU with vectorized numpy.]

**Dependency surface:** numpy, scipy (stats only), pandas (optional — evaluation reports only). No torch, no tensorflow, no jax.

**v10.7 note:** No new dependencies added. v0.7.21/v0.7.22-pre changes use existing numpy/scipy surface.

---

## 3. Architectural Principles

[§3 UNCHANGED from v10.6. Six principles: mathematical foundations first; compound learning via centroid geometry; audit trail over black boxes; rule-based safety over statistical aggregates; test-gated shipping; open source trust.]

**v10.7 reinforcement of rule-based-safety principle:** UNI-DK-01 v5.3's KernelSelector findings strengthen this principle. Confidence-based kernel comparison (Options B/C in early design) depended on aggregate statistics (mean confidence, mean log-likelihood) that turned out to be misleading for DK. The NR-rule (Option A) is a deterministic function of measured per-factor σ — no aggregation, no calibration dependency, no way for the selector to drift. Rule > statistical aggregate again.

**v10.8 seventh principle — Defense in depth over single-mechanism safety.** Shrinkage provides mathematical guardrail. Promotion gate provides operational guardrail. Rollback provides recovery. No single mechanism is the safety guarantee; the combination is. (Framework v4 §3.6, 5-judge consensus.)

---

## 4. Experimental Foundation

[§4 UNCHANGED from v10.6 except for experiment count update.]

| Category | Experiments | Current | Key findings |
|---|---|---|---|
| Architecture validation (v4.1–v5.0) | EXP-5, EXP-A, EXP-C1, EXP-B1, EXP-D1, EXP-D2, EXP-E1, EXP-E2, etc. | ~30 | ProfileScorer = THE scorer. L2 97.89% centroidal. |
| Cross-graph experiments | CGT-1 through CGT-80+ | ~130 | W2 edges + attention. b=2.11 scaling exponent. |
| Kernel deliverables | V-MV-KERNEL (390 cells), V-HC-CONFIG, V-HC-SHRINKAGE, B5B-PROXY, Phase 1 sweeps, HC-scaling, KernelSelector, EXP-A4-DIAGONAL, EXP-REFER-COVERAGE, EXP-REFER-LAYERED, **UNI-DK-01 v5.3** | ~25 | DiagonalKernel +13.2pp peak (V-MV-KERNEL-HET) + characterized +7.67pp at NR=5.0 (UNI-DK-01 v5.3, 1500 cells). DK ECE 0.42 at NR=5.0. A=4 confirmed. ReferralRules 72.7% DR. |
| Phase 1 closure | W2 flywheel (CLAIM-W2), OLSMonitor (CLAIM-OLS-01), V-ENRICHMENT series (CLAIM-65), CLAIM-66-70 | ~20 | Flywheel validated. Enrichment safety. Tiebreaker fix. Raw weights fix. |
| Oracle separation / γ-theorem validation | Oracle separation v6/v8/v11/v2/v3/final | ~15 | γ > 1 ⟺ ε_firm > 0.125 (4-LLM validated, April 8, 2026). |
| **Total** | | **~180** | **v6.0 COMPLETE. UNI-DK-01 v5.3 closed April 19, 2026.** |

**UNI-DK-01 v5.3 detail (NEW v10.7):** 1500 cells (5 NR × 5 q̄ × 30 seeds × 2 kernels). Controlled-parameterization characterization of DK advantage at fixed mean-σ=0.175. All four pre-registered checks PASS (D1 monotonicity, D2 NR=5.0 ≥ 5pp, D3 cold-start ≥50% at NR=5.0, D4 q̄-std ≤1pp). D5 dropped as tautological. Supersedes Deliverable 1 cumulative-averaging decomposition (retracted). Full data at `DRIVE_BASE/uni_dk_01/uni_dk_01_v5_*.json`. See math_synopsis v14 §3.3 for full decomposition and methodology; claims_registry v5 Gate UNI-DK-01 for claim inventory; this document §9.6 for DK calibration finding consequences.

---

## 5. The Ontological Architecture

[§5 UNCHANGED from v10.6. Two-level institutional judgment framing (Level 1 — Decision Intelligence = ProfileScorer; Level 2 — Deployment Intelligence = AgentEvolver in soc-copilot). Compiled knowledge via centroid tensor. Four loops, three connectors.]

**v10.7 note:** Conservation law q in the invariant α·q·V ≥ θ_min is now operationally defined as rolling verified accuracy over last 400 decisions (see §9.8 new, §10.11 updated). This strengthens the Level 1 ↔ Level 2 compounding guarantee: conservation operates on a calibration-independent signal, so it remains valid when DK ships as v6.0 default.

**v10.8 — Profile state terminology (framework v4 + 5-judge consensus):**

The GAE's knowledge of each (category, action) class is encoded as the Order-0 and Order-1 PROFILE STATE of a diagonal scoring model.

**Order-0 profile state** (T₀, the centroid tensor) encodes WHERE each pattern cluster is. This is the MLE of the class-conditional mean under the Gaussian surrogate. It transfers across deployments (+28pp, EXP-19) because class centers are domain-structural.

**Order-1 profile state** (T₁, the DK precision weights) encodes WHICH DIMENSIONS MATTER. These are discriminative precision weights — deployment-specific parameters within the diagonal-kernel scoring family, estimated to maximize classification accuracy. They do NOT transfer (-5.6pp, ORDER-3) because per-dimension importance is firm-specific.

Phase 1 estimates T₀ (means). Phase 2 estimates T₁|T₀ (precision weights) via James-Stein shrinkage. The transition is governed by PhasePolicy. The Fisher-inspired asymmetry (Layer A: Gaussian surrogate motivates; Layer B: empirical ρ_variance ≈ +0.35 confirms) explains why Phase 2 can persist after Phase 1 saturates.

---

## 6. Package Structure — Three-Repository Architecture

[§6 UNCHANGED from v10.6. gae (Apache 2.0, numpy-only), ci-platform (Apache 2.0, connectors + deployment), soc-copilot (proprietary, SOC domain expertise). Dependency direction: soc-copilot → ci-platform → gae. No circular dependencies.]

---

## 7. Causal Architecture: Three Loops, Four Connectors

[§7 UNCHANGED from v10.6. Loop 1 (Perception → Prediction), Loop 2 (Action → Feedback → Learning), Loop 3 (Synthesis, PROPOSAL), Loop 4 (Awareness, PROPOSAL). Connectors: FactorProtocol, CategoryRouter, ScorerInterface, OracleProvider.]

**v10.7 note:** No architectural changes to loops or connectors. The conservation-law q operational change (confidence → rolling verified accuracy) is an implementation refinement within Loop 2 → Loop 3 monitoring, not an architectural change to the loops themselves.

---

## 8. Tier 1 — Factor Protocol & Assembly (`gae/factors.py`)

[§8 UNCHANGED from v10.6. FactorProtocol. Factor identity preservation (R4). FactorComputer registry. SOC and S2P factor sets. Factor mask DEPRECATED in favor of DiagonalKernel continuous weighting.]

---


## 9. Tier 2 — Profile-Based Scoring (`gae/profile_scorer.py`)

**Role in the two-level architecture:** ProfileScorer is **Level 1 — Decision Intelligence**. It learns *what to decide* by maintaining per-category profile centroids that encode verified organizational judgment as geometry. See §5 for the full two-level framing. This section specifies the implementation contract.

**v10.7 changes at a glance:**
- §9.1 API corrections (for_soc factory, ScoringResult fields corrected against live code, eta_override semantics clarified, safety guards documented).
- §9.2 KernelSelector architecture **INVERTED** — rule-based PRIMARY, data-driven MONITOR only.
- §9.3 ScoringResult fields corrected to match live code (5 fields, not 9).
- §9.4 ProfileScorer interface expanded with v0.7.21/v0.7.22-pre additions.
- §9.5 Learning equation unchanged substantively; compute_eta_override() signature updated with worst_case_quality parameter.
- §9.6 NEW — DK Calibration Properties (prediction vs estimation channels).
- §9.7 NEW — AMBER Auto-Pause Signal Source (rolling verified accuracy).
- §9.8 NEW — Conservation-Law q Operational Definition.
- §9.9 NEW — Analyst Triage Ranking Specification (softmax entropy / confidence gap).

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

**Kernel selection rule (v6.0 settled — Option A):**
```
noise_ratio = max(σ_per_factor) / min(σ_per_factor)
if noise_ratio > 1.5 → DiagonalKernel(weights=1/σ²)
else                  → L2Kernel
```

**Why rule-based, not confidence-based (v10.7 clarification):** UNI-DK-01 v5.3 E1/E3 tested confidence-based kernel comparison (mean_conf, mean_ll, trimmed_ll) at scale across 1500 cells. Rule-based selection: 100% correct at every informative NR. Data-driven (trimmed_ll): below chance at NR ≥ 3.0. Root cause: DK's sharpened softmax produces miscalibrated confidence (ECE 0.42 at NR=5.0, see §9.6). The NR-rule depends only on measured per-factor σ from P28 qualification — no calibration dependency. Two validations: V-MV-KERNEL HC-personas 4/4 + UNI-DK-01 v5.3 E3 100%. See math_synopsis v14 §3 for the canonical reasoning; see §9.2 for implementation details.

**Why not ShrinkageKernel (full Mahalanobis)?**
Deliverables D2 and D3 tested off-diagonal covariance: SOC gap 0.8pp, S2P gap -0.2pp. Noise ratio alone drives the kernel advantage (Explanation A). Correlation density adds nothing measurable. ShrinkageKernel deprioritized to v7.0 research.

Three first-class constituents:
- **f** (query): factor vector, shape (n_f,), produced by FactorComputers
- **μ** (keys): profile centroids, shape (n_c × n_a × n_f), compiled ontology
- **c** (router): category index, from SituationAnalyzer (MoE head selector)

### 9.2 Kernel Architecture (v10.7 — settled with architecture inversion)

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

**Kernel selection (KernelSelector — §10.10) — v10.7 architecture:**

```
Phase 2 (COMPUTE, PRIMARY): noise_ratio > 1.5 → diagonal, else L2.
  One parameter. No ρ_max. This is the production selector.

Phase 3 (SHADOW, MONITOR only): all kernels scored simultaneously, rolling 100-decision
  window. Data is COLLECTED for monitoring (agreement-rate divergence, drift detection,
  future re-evaluation) but does NOT become the selector.
  MIN_DECISIONS_FOR_RECOMMENDATION=100 is the threshold at which monitoring output is
  emitted (via KernelSelector.recommend()), not at which selection switches.

No Phase 4 empirical lock. Selection never switches from the NR-rule in v6.0.

Ongoing: should_reconsider() on σ change, ρ change, covariance λ drop — triggers
  a new call to preliminary_recommendation() with updated σ. Still rule-based.
```

**Why the architecture inverted (v10.7):**

Previous versions documented "Phase 4 empirical lock after ~250 verified decisions" — confidence-based comparison would choose the kernel with highest rolling agreement rate. UNI-DK-01 v5.3 demonstrated this is untenable:

1. **mean_conf systematically fails for DK.** E4 winner_stability_rate=1.000 at every (NR, window) combination; dk_final_share=0.000 everywhere — including NR=5.0 where DK has a +7.67pp accuracy advantage. mean_conf comparison ALWAYS selects L2, never DK. Mechanism: DK's W-weighting concentrates scoring on low-σ factors, reducing effective dimensionality, sharpening the softmax. DK has HIGHER max_p variance across decisions but LOWER mean max_p than L2's more-uniform distribution. The mean doesn't track accuracy.

2. **Data-driven trimmed_ll fails at high NR.** E3 data-driven selection: 0.527, 1.000 (n=2), 0.449, 0.370 at NR=1.5/2.0/3.0/5.0. Below chance at NR=3.0 and NR=5.0 — exactly where DK would be the correct choice. Root cause: DK's overconfident predictions inflate or suppress LL inconsistently relative to accuracy.

3. **Rule-based selection is 100% correct.** E3 rule-based: 1.000, 1.000, 1.000, 1.000 across all informative NRs (91+2+49+127=269 cells). Combined with historical V-MV-KERNEL HC-personas 4/4, two independent validations.

**Option A (NR-rule) ships. Option B (accuracy-comparison holdout) retained as v6.1 contingency if P28 σ estimation proves unreliable in the field — that would require adding holdout management, which is not a v6.0 scope item. Option C (calibrated-confidence Platt scaling) archived — no scenario requires it.**

**Noise ceiling is kernel-dependent:**

| Kernel | GREEN | AMBER | RED |
|---|---|---|---|
| L2 | σ ≤ 0.105 | 0.105 < σ ≤ 0.157 | σ > 0.157 |
| Diagonal | σ ≤ 0.157 | 0.157 < σ ≤ 0.25 | σ > 0.25 |

The GREEN zone nearly doubles with DiagonalKernel. Healthcare deployments (σ≈0.22) move from RED (L2, frozen only) to AMBER (Diagonal, learning with monitoring).

**Phase 0 instrumentation (v0.7.21/v0.7.22-pre, NEW v10.7):**

`gae/kernel_selector.py:38-40, :272-280` — KernelScore dataclass includes `cumulative_analyst_prob: float = 0.0`. This is shadow-mode instrumentation: when a verified analyst action is recorded, both L2 and DK's probability-of-analyst-action are accumulated. Enables log-likelihood metrics for future re-evaluation (not used for v6.0 selection — selection is the NR-rule). Surfaced in `get_comparison_summary()` output.

### 9.3 ScoringResult [CORRECTED v10.7 against live code]

**v10.7 correction:** v10.6 documented ScoringResult with 9 fields (`action_probabilities`, `selected_action`, `confidence`, `distances`, `factor_vector`, `category_index`, `temperature`, `kernel`, `probabilities`). Live code at `gae/profile_scorer.py:77-101` has ONLY 5 fields. The v10.6 doc was wrong. v10.7 matches code:

```python
@dataclass
class ScoringResult:
    """Output of Eq. 4-final. All fields are populated on every score() call.

    Reference: gae/profile_scorer.py:77-101
    """
    action_index: int                  # argmax action index into scorer.actions
    action_name: str                   # scorer.actions[action_index] — the human-readable name
    probabilities: np.ndarray          # shape (n_a,) — softmax output, sums to 1
    distances: np.ndarray              # raw distance values (before softmax), shape (n_a,)
    confidence: float                  # max probability value = probabilities.max()

    # NOTE: The probability field name is .probabilities (primary and only name —
    # no .action_probabilities alias). Consumers must use .probabilities.
    #
    # NOTE: Factor vector, category index, temperature, and kernel used are NOT
    # returned on the ScoringResult. Consumers that need them must access them
    # via other scorer APIs (scorer.tau, scorer.kernel, etc.) or preserve them
    # at the call site.
    #
    # NOTE: factor_vector preservation for the R4 requirement (factor identity
    # preserved across scoring and learning) is handled by ProfileScorer.update()
    # taking f as an explicit parameter — the scorer does not cache or return f
    # via ScoringResult.
```

**Downstream consumer contract:**

Consumers that previously used `.action_probabilities` must use `.probabilities`. Consumers that accessed `.factor_vector`, `.category_index`, `.temperature`, or `.kernel` via ScoringResult must fetch them from the scorer instance or preserve them at the call site. `tests/test_consumer_contracts.py:14-190` locks this contract (12 tests covering `action_index`, `action_name`, `probabilities` shape, `distances` shape, `confidence`, and the absence of removed fields).

### 9.4 ProfileScorer Interface [UPDATED v10.7]

**v10.7 additions:**
- `for_soc()` classmethod factory for SOC-canonical defaults.
- `eta_override` semantics clarified: constructor default is `None` (backward compatible), but production use should set 0.01 — `for_soc()` does this automatically.
- Safety guards: τ range check, isfinite checks on f and μ post-clip.
- η_override RuntimeWarning when outside [0.005, 0.02].

```python
class ProfileScorer:
    def __init__(
        self,
        mu: np.ndarray,                        # shape (n_c, n_a, n_f)
        actions: List[str],                    # action names, length n_a
        kernel: KernelType = KernelType.L2,    # or KernelType.DIAGONAL
        profile: Optional[CalibrationProfile] = None,  # sets eta/tau/decay
        categories: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        eta_override: Optional[float] = None,  # P0 fix: 0.01 for SOC
        factor_mask: Optional[np.ndarray] = None,
        scoring_kernel: Optional[ScoringKernel] = None,
        auto_pause_on_amber: bool = False,
    ) -> None: ...

    # NOTE: eta, eta_neg, tau are NOT direct constructor kwargs.
    # They come through CalibrationProfile or are set internally.
    # Consumer contract test (test_consumer_contracts.py) verifies 
    # that ProfileScorer(eta=0.05) raises TypeError.
    # Use ProfileScorer.for_soc() for SOC canonical defaults.

    @classmethod
    def for_soc(cls, mu, actions=None, **kwargs):
        """
        Canonical SOC factory — NEW v0.7.21 (gae/profile_scorer.py:284-300).

        Sets SOC canonical defaults:
          - eta_override = 0.01 (P0 fix canonical)
          - auto_pause_on_amber = True (conservation-aware)
          - tau = 0.1 (V3B validated)

        Any kwargs override these defaults — e.g., for_soc(mu, actions, eta_override=0.008)
        is legal but will emit a RuntimeWarning if eta_override is outside [0.005, 0.02].

        Purpose: eliminates the config-setting boilerplate that soc-copilot
        repeated across tests and deployment code. One-line construction for
        the production SOC scorer:

            scorer = ProfileScorer.for_soc(mu, actions=SOC_ACTIONS_V6)

        For non-SOC domains, use the constructor directly with explicit kwargs.
        """
        ...

    def score(
        self,
        f: np.ndarray,           # shape (n_f,) or (1, n_f)
        category_index: int,     # from SituationAnalyzer
        synthesis: None = None,  # ALWAYS None in v5.0/v6.0. Loop 4 PROPOSAL only.
    ) -> ScoringResult:
        """
        Score with kernel dispatch:
          1. Validate f (isfinite check — gae/profile_scorer.py:332)
          2. Apply factor_mask if set (zero out masked dimensions)
          3. kernel.compute_distance(f_effective, mu_effective) → distances
          4. softmax(-distances / τ) → probabilities
          5. Return ScoringResult(action_index, action_name, probabilities, distances, confidence)
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
          1. Check auto_pause_on_amber → early return with outcome='paused_conservation'
             if _paused_by_conservation is set (gae/profile_scorer.py:184-188, :662-675)
          2. Check min_confidence gate
          3. kernel.compute_gradient(f, mu_single) → gradient direction
          4. Apply asymmetric η: η_confirm (correct) or η_override (incorrect)
          5. Apply factor_mask to gradient (masked dims don't change)
          6. Clip centroids to [0, 1]
          7. Validate μ post-clip (isfinite check — gae/profile_scorer.py:648)
        Returns CentroidUpdate with delta_norm, category, action, outcome.
        """
        ...

    # Conservation integration (v6.0, CONFIRMED ENFORCED v10.7):
    def set_conservation_status(self, status: str) -> None:
        """
        Called by ConservationMonitor. 'AMBER'/'RED' → pause, 'GREEN' → resume.

        When paused, subsequent update() calls return CentroidUpdate with
        outcome='paused_conservation' and no centroid modification occurs.
        Reference: gae/profile_scorer.py:560, :565-568, :662-675.

        v10.7 cross-repo note: GAE enforces pause at the ProfileScorer layer.
        soc-copilot must consume the pause signal (via is_paused or the
        update outcome field) to route subsequent alerts to human review.
        That SOC backend wiring is tracked separately as SOC-Q3 — it is
        NOT a GAE design gap.
        """
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
        """{'applied': N, 'gated': M, 'paused_conservation': P} — tracks paused/gated updates.
        v10.7: added paused_conservation counter."""
        ...

    # Existing methods (unchanged):
    def diagnostics(self) -> Dict[str, Any]: ...
    def checkpoint(self) -> Dict: ...
    def rollback(self, checkpoint: Dict) -> None: ...
    def reset_to_config(self, profile_config: Dict) -> None: ...
    def freeze(self) -> None: ...
    def unfreeze(self) -> None: ...
```

**η_override guard (v10.7):** `gae/profile_scorer.py:703` emits `RuntimeWarning` if `eta_override` is set outside `[0.005, 0.02]`. Prevents accidental drift from the canonical 0.01 validated value. Deployments needing values outside this range must explicitly suppress the warning and justify in their deployment log.

**Safety guards (v10.7):**
- `tau` guard: `gae/profile_scorer.py:326` raises `ValueError` if `tau <= 0` or `tau > 10`. Prevents catastrophically miscalibrated softmax.
- `isfinite(f)` guard: `gae/profile_scorer.py:332` raises `ValueError` if any factor value is NaN or Inf. Prevents silent propagation through softmax.
- `isfinite(μ)` guard: `gae/profile_scorer.py:648` raises `ValueError` if clipping produces non-finite values (should never happen; defense in depth).

### 9.5 Learning Equation

**Eq. 4b-final (with asymmetric η — P0 fix):**
```
Confirm path (analyst accepts system recommendation):
  μ[c,a,:] ← μ[c,a,:] + η_confirm·K.gradient(f, μ[c,a,:])     (pull toward observed)
  η_confirm = 0.05 (clean signal, full learning rate)

Override path (analyst changes action):
  μ[c,a_pred,:] ← μ[c,a_pred,:] − η_override·K.gradient(f, μ[c,a_pred,:])  (push wrong away)
  μ[c,a_gt,:]   ← μ[c,a_gt,:]   + η_confirm·K.gradient(f, μ[c,a_gt,:])    (pull correct toward)
  η_override = 0.01 (noisy signal, attenuated 5×)

Where K.gradient is kernel-aware:
  L2Kernel:       gradient = (f − μ)
  DiagonalKernel: gradient = W/W.max() · (f − μ)    — normalized weights control direction, not magnitude
                  NOTE: W/W.max() NOT W*(f-mu). GAE-GRADIENT-001 fixed in v0.7.7.
                  W*(f-mu) was wrong: high-W factors dominated by magnitude, corrupting learning direction.
                  W/W.max()*(f-mu): weights in [0,1], direction preserved, all factors contribute correctly.
                  Live code: gae/kernels.py:199-200. `w_max = max(self.weights.max(), 1e-9)`
                                                     `return (self.weights / w_max) * (f - mu)`

REQUIRED: all μ values clipped to [0, 1] after every update (V2 validated)
REQUIRED: masked dimensions (factor_mask) do NOT update (gradient zeroed)
REQUIRED: τ = 0.1 always — never modified by synthesis or any other condition
```

**Why asymmetric η (P0 fix):**
9-persona LLM-judge stress test found 13-27pp centroid degradation from realistic analyst override quality (q̄=0.60-0.70). The override path carries noise from analyst errors. η_override=0.01 attenuates this noise by 5×. Validated across 24 personas (1C quality sweep). At q̄=0.57 (worst quality): +0.5pp with η_override=0.01 (no degradation). Without: -9pp.

**UNI-DK-01 v5.3 confirmation (v10.7):** At fixed mean-σ=0.175 across NR={1.0, 1.5, 2.0, 3.0, 5.0}, positive learning gap is observed at every NR level (+0.19pp to +3.49pp across the range). Previously reported "negative learning at low NR" from cumulative-averaging v4.2 measurement is **retracted as a measurement artifact**. See math_synopsis v14 §3.3 for full characterization.

**compute_eta_override() formula (v0.7.22-pre):**
```python
def compute_eta_override(
    eta_confirm: float = 0.05,
    mean_quality: float = 0.75,
    quality_variance: float = 0.02,
    worst_case_quality: Optional[float] = None,  # NEW v0.7.22-pre (gae/calibration.py:354)
) -> float:
    """
    η* ∝ (2q̄-1) / (2σ²_q + signal). Directionally correct, ~2× overestimate vs empirical.
    Global default η_override=0.01 is the validated shipping value. This formula is
    diagnostic — use it to check whether a deployment's characteristics justify the
    default, not to override the default in production.

    NEW v0.7.22-pre: `worst_case_quality` parameter lets callers specify the lower
    bound of analyst quality explicitly (typically the 10th percentile of analyst
    agreement rate from shadow mode data). If provided, the formula uses
    min(mean_quality, worst_case_quality) as the effective quality input —
    produces a more conservative η_override estimate for heterogeneous teams.

    Example usage (deployment qualification, NOT production):
        eta_diag = compute_eta_override(
            mean_quality=0.82,      # team mean agreement rate from shadow
            quality_variance=0.04,  # team variance
            worst_case_quality=0.65 # 10th-percentile analyst
        )
        # Compare eta_diag to shipped 0.01. If eta_diag < 0.005, file a ticket
        # (deployment has worse-than-validated-range quality).
    """
    ...
```

### 9.6 DiagonalKernel Calibration Properties [NEW v10.7]

**Context.** UNI-DK-01 v5.3 (April 19, 2026) characterized DK's calibration properties at scale (1500 cells, 5 NR levels × 5 q̄ levels × 30 seeds × 2 kernels). This is the first scaled characterization of DK calibration in the experiment history — prior experiments (V-MV-KERNEL-HET, V-HC-CONFIG, HC-scaling) measured accuracy only.

**The finding.** DK's accuracy advantage does NOT translate to calibrated confidence. Expected Calibration Error at τ=0.1:

| NR | L2 ECE | DK ECE | Ratio |
|---|---|---|---|
| 1.0 | 0.055 | 0.055 | 1.0× |
| 1.5 | 0.053 | 0.148 | 2.8× |
| 2.0 | 0.051 | 0.223 | 4.4× |
| 3.0 | 0.046 | 0.325 | 7.0× |
| 5.0 | 0.041 | **0.420** | **10.4×** |

L2 is well-calibrated and mildly improves with NR (0.04–0.06). DK is severely miscalibrated and degrades rapidly with NR (0.06 → 0.42).

**Interpretation.** When DK reports confidence 90%, the actual accuracy on those decisions is far below 90% — off by up to 42pp on average at NR=5.0.

**Mechanism.** DK's weighted-distance scoring with W = diag(1/σ²) normalized so max weight = 1. At NR=5.0, σ profile = [0.058, 0.105, 0.152, 0.198, 0.245, 0.292]. W normalized = [1.00, 0.31, 0.15, 0.09, 0.06, 0.04]. Four of six factors contribute <15% of the weight of the dominant factor. DK's effective dimensionality is ~3 (factors with w > 0.1). Reduced effective dimensionality sharpens the softmax: predictions concentrate toward 0 or 1 rather than spread across actions.

This is a property of inverse-variance weighting, not an implementation defect. Any weighted-distance kernel with concentrated weights exhibits the same pattern.

**The "prediction vs estimation channels" pattern (v10.7 architectural framing):**

DiagonalKernel is:
- A **good prediction channel**. Its argmax is more accurate than L2's at NR > 1 (up to +7.67pp at NR=5.0, +13.2pp peak at V-MV-KERNEL-HET σ_level=0.30).
- A **poor estimation channel**. Its confidence values (max probability, full probability distribution, log-likelihoods) are miscalibrated as NR rises.

Production components that need prediction can consume DK's action choice (`ScoringResult.action_index`, `.action_name`) directly. Production components that need estimation (calibrated probability, confidence threshold, likelihood-based selection) must use alternative signals:

| Consumer | Needs | v6.0 signal |
|---|---|---|
| KernelSelector | Kernel choice | NR-rule (§9.2), not confidence |
| Conservation-law q | Decision quality aggregate | Rolling verified accuracy (§9.8) |
| AMBER auto-pause | Degradation detection | Rolling verified accuracy (§9.7) |
| Analyst triage UI | Decision ordering | Softmax entropy / confidence gap (§9.9) |
| Auto-approve gate | Per-decision reliability | Raw `confidence` + category-specific threshold (unchanged from v5.5 — uses L2 when routed, acceptable under current threshold calibration; see note below) |
| Downstream ML ensembles | Soft probabilities | CURRENTLY NONE — consumers needing DK soft probabilities must run Platt calibration on their own holdout. Not shipped at v6.0. |

**Auto-approve note:** The v5.5 auto-approve threshold `threshold*(c)` (§10.12 Eq. T*) was calibrated on L2 data with τ=0.1 (ECE=0.036, V3B validated). Under v6.0 with DK routed at NR > 1.5, thresholds calibrated for L2 may under-select (DK's sharpened distribution pushes more decisions above the threshold). Two responses possible: (1) re-calibrate per-category thresholds under DK using PROD-4b data, or (2) leave thresholds conservative and accept slightly reduced auto-approve coverage in heterogeneous-noise deployments. Decision deferred to first v6.0 customer PROD-4b exercise — flagged in §16.

**Why this wasn't caught earlier.** V-MV-KERNEL-HET (the original 390-cell experiment that established "+13.2pp SOC") measured accuracy, not calibration. The historical claim was "DK is more accurate" (true). The implicit assumption was "accurate prediction → reliable confidence" (false for weighted-distance kernels). UNI-DK-01 v5.3 is the first experiment in this thread to measure ECE for both kernels at scale. The finding is new because nobody asked the question before — not a bug.

**Commercial framing (v10.7).** B3 (DiagonalKernel) claim in claims_registry remains valid. The accuracy advantage is real and robust across two independent experiments (V-MV-KERNEL-HET 390 cells + UNI-DK-01 v5.3 1500 cells = 1890 cells total). The calibration finding is a deployment-engineering concern, not a commercial-claim concern — customer-facing accuracy claims are unaffected because no customer-facing claim cites raw DK confidence values. See innovation_note v6 §Innovation 4 for the curve-based citation pattern.

### 9.7 AMBER Auto-Pause: Signal Source [UPDATED v10.7]

**v10.7 change:** Auto-pause trigger signal changed from per-decision confidence to rolling verified accuracy. The mechanism (state-level transition, freeze learning, route to human review) is unchanged; the trigger is different.

**Signal.** Rolling verified accuracy over the last 400 decisions (≈100 verified events at 25% verify rate — standard error ≈ 3.6pp at q≈0.85).

**Trigger condition (AMBER):** Rolling accuracy drops below `(baseline_accuracy × 0.9)` where baseline is computed from the first 400 verified decisions after GREEN deployment state is reached. Deployment-specific, not a hardcoded constant.

**Trigger condition (RED):** Rolling accuracy drops below absolute floor:
- Safety-critical categories (insider_threat, lateral_movement): 0.70
- Standard categories (credential_access, data_exfiltration, cloud_infrastructure, threat_intel_match): 0.60

**State transition.** When AMBER or RED is signaled:
1. ProfileScorer enters `_paused_by_conservation=True` state.
2. Subsequent `update()` calls return CentroidUpdate with `outcome='paused_conservation'`. Centroid updates are blocked (`gae/profile_scorer.py:184-188, :662-675`).
3. `ScoringResult` still returns normally — the scorer continues to produce action recommendations. The SOC BACKEND decides what to do with them.
4. soc-copilot routes decisions to human review instead of auto-acting (SOC-Q3 wiring — tracked separately; see §9.4 cross-repo note).

**Resume condition (GREEN).** Rolling accuracy recovers above AMBER threshold for ≥100 consecutive decisions. ConservationMonitor calls `set_conservation_status('GREEN')`; scorer exits paused state; learning resumes.

**Why rolling accuracy, not confidence (v10.7):**
1. **Directly measures degradation.** Accuracy drop is the thing we care about; confidence drop is a proxy whose quality depends on calibration.
2. **Kernel-independent.** Works identically for L2 and DK. No calibration dependency.
3. **Single monitoring pipeline.** Same signal powers the conservation-law q (§9.8). One rolling-accuracy computation, multiple consumers.
4. **Robust to DK miscalibration.** At NR=5.0 where DK confidence is off by 42pp (§9.6), a confidence-based trigger would either fire at the wrong threshold or miss genuine accuracy degradation. Rolling accuracy is immune.

**What per-decision confidence gating still DOES NOT do:**

Auto-pause is a SYSTEM state change, not a per-decision filter. If the product needs per-decision uncertainty gating *within* the AUTO zone (e.g., "auto-act on this decision unless confidence is too low"), that's a separate mechanism. For that use case, use softmax entropy / confidence gap (§9.9). Mixing state-level auto-pause with per-decision uncertainty gating into a single "confidence threshold" was the design mistake in v10.6.

### 9.8 Conservation-Law q: Operational Definition [NEW v10.7]

**The invariant.** Conservation law (math_synopsis v14 §5, Eq. CL):
```
α(t) · q(t) · V(t) ≥ θ_min
```
where:
- α(t): analyst override rate among verified decisions — fraction of verified decisions where analyst disagreed with system recommendation
- q(t): decision quality — the operational definition is the subject of this section
- V(t): verified decisions per day
- θ_min: deployment-specific floor = η × N_half² / T_max (Grok form, §10.11)

**q operational definition (v10.7):**
```
q(t) = (1 / N_v(t)) × Σ_{i ∈ W(t)} 𝟙[prediction_i = verified_action_i]

where:
  W(t) = last 400 decisions at time t (rolling window, aligned with α and V)
  N_v(t) = number of verified decisions in W(t) (≈100 at 25% verify rate)
  𝟙[·] = indicator of prediction-action match on verified events only
```

q is a **rolling verified accuracy over the last 400 decisions**. At 25% verify rate × 400 decisions = ~100 verified events per window. Standard error on q at n=100, p≈0.85: √(0.85 × 0.15 / 100) ≈ 3.6pp.

**Why not per-decision confidence:**
- DK's confidence is miscalibrated at high NR (ECE 0.42 at NR=5.0, §9.6). A confidence-based q would operate on noisy inputs whenever DK is routed.
- Confidence is an *estimate* of quality; verified accuracy *is* quality on the verified subsample — more direct operationalization.
- Strengthens EU AI Act Article 14 oversight argument: accuracy evidence is more defensible than confidence-trajectory evidence for "effective oversight."
- Aggregate nature matches α and V (both rolling aggregates over the same window). A confidence-derived q was implicitly per-decision; the aggregate semantics were never clean.

**Why the substitution preserves the v13 derivation:**

The v13 conservation-law derivation (math_synopsis §5, Eq. CL) established q as "per-decision quality" semantically. No derivation step required q to be confidence specifically. Every derivation step that uses q operates on its semantic meaning as a quality measurement. Rolling verified accuracy satisfies that semantic role equivalently (or more directly) than confidence.

The math_synopsis v14 §5 update notes that no derivation step required re-derivation under this substitution. If a future audit (codex, external review) identifies a step that implicitly required q = confidence, the substitution would TIGHTEN the guarantee — the invariant becomes about literal accuracy × volume rather than confidence × volume. No step has been identified.

**Implementation.** A `ConservationMonitor.compute_q()` method computes rolling verified accuracy from the decision log. The monitor already maintains the rolling window for α and V; q is added as a third rolling quantity. No new storage, no new pipeline.

**Cross-references:**
- math_synopsis v14 §5 for the mathematical derivation.
- math_synopsis v14 §9 for the safety-architecture integration.
- math_synopsis v14 §16 for the constraint table entry.
- §10.11 below for the `check_conservation()` API signature.

### 9.9 Analyst Triage Ranking [NEW v10.7]

**Context.** SOC Copilot surfaces decisions to analysts in some priority order. Under DK's miscalibration finding (§9.6), raw max_p is unreliable for ordering. v10.7 specifies the GAE-level ranking signal; soc-copilot design v5.7+ will implement the UI.

**Ranking signal:** Softmax entropy (ascending) OR confidence gap top_p − second_p (descending). These are equivalent for argmax decisions and rank decisions by *decisiveness* rather than by raw peak probability.

```python
# In SOC Copilot's triage UI construction:

def triage_rank(scoring_result: ScoringResult) -> float:
    """
    Return a rank key for triage ordering. Lower = higher priority for review
    (less decisive predictions should be reviewed first). Sort ASCENDING.

    Uses softmax entropy — calibration-independent, robust to DK sharpening.
    """
    p = scoring_result.probabilities
    # Add small epsilon for numerical stability on exact-0 probabilities
    p_safe = np.maximum(p, 1e-12)
    entropy = -np.sum(p_safe * np.log(p_safe))
    return -entropy  # negate so ascending sort puts highest-entropy first

# Alternative (equivalent ordering for argmax):
def triage_rank_gap(scoring_result: ScoringResult) -> float:
    """Confidence gap top_p - second_p. Sort ASCENDING (smaller gap = higher priority)."""
    p_sorted = np.sort(scoring_result.probabilities)[::-1]  # descending
    return p_sorted[0] - p_sorted[1]
```

**Why entropy / gap, not max_p:**
- **Available per-decision**, no calibration infrastructure needed.
- **Correctly monotone for triage.** Entropy measures distribution spread; low entropy = decisive prediction (trust it), high entropy = ambiguous prediction (review it). Gap measures argmax vs runner-up margin; same interpretation.
- **Less affected by DK sharpening than raw max_p.** DK's sharpened distribution has higher peaks (raw max_p) AND bigger gaps — the direction of the effect matches triage intent (sharper decisions really ARE more decisive). The mis-calibration affects the absolute values of max_p relative to frequency, but NOT the ranking between decisions.
- **Works for both kernels.** L2 and DK produce meaningful entropy / gap values; no kernel-specific logic.

**Why not distance-based triage:**

An alternative was considered: rank by distance to nearest centroid (low = close to known pattern, high = unfamiliar). Rejected because distance and decisiveness rank *different things*:

| Decision archetype | Distance | Top-prob gap | What's happening | Triage priority |
|---|---|---|---|---|
| Near correct centroid | low | large | Confident and right | Low (trust it) |
| Far from all centroids | high | small | Unfamiliar pattern; low confidence | High (review) |
| Between two centroids | low | *small* | Near class boundary; ambiguous | **Highest** (actively misleadable) |

Sort-by-distance would rank near-boundary decisions (archetype 3) at the TOP of the "trust" list. But those are exactly where the system is most likely to be wrong — the decisions that flip with small noise. Entropy/gap correctly puts them at the top of the "review" list. Under DK's W-concentration, this mis-triage would be worse, not better.

**UI display recommendation.**

Primary ordering: entropy (or gap) — computed per-decision, no calibration.

Secondary display column: raw `confidence` (= `probabilities.max()`) shown alongside the entropy-based rank. When DK is routed (KernelSelector chose Diagonal), display a badge next to the confidence value: "uncalibrated — DK". This communicates to the analyst that the confidence number is a raw model output, not a calibrated probability. When L2 is routed, no badge — L2 confidence at τ=0.1 is well-calibrated (ECE=0.036, V3B).

**Future enhancement (post-v6.0):** If Platt scaling or isotonic regression is added as an opt-in calibration layer (Option C from the KernelSelector architecture discussion), the badge would read "calibrated" and the confidence value would be post-hoc mapped. That's a v6.5+ workstream if the first-customer data indicates the uncalibrated display is causing analyst confusion.

**Cross-references:**
- UNI-DK-01 v5.3 comprehensive results impact document §10.7.1 for the full rationale.
- soc_copilot_design v5.7+ for the UI implementation spec (inheriting this GAE specification).
- math_synopsis v14 §3.3 for the ECE mechanism that motivates this spec.

---

### 9.10 TwoPhaseScorer Architecture [NEW v10.8]

**Experimental foundation:** ~115 experiments across 6 phases (framework v4, post-judge-review).

**Phase 1 (MEAN_CONVERGENCE):** Centroid learning saturates. K33: centroids reach minimum GT distance at N≈200, then DIVERGE (N=200: ||μ-μ*||=0.755, acc=86%; N=2000: ||μ-μ*||=0.942, acc=80%). K1: ρ_mean ≈ -0.07 at convergence — updates become ANTI-ALIGNED with GT. Cause: label-noise contamination at 15% noise (framework v4 + Opus/GPT5.5). DEFAULT: DecisionCountPolicy(n=200).

**Phase 2 (VARIANCE_LEARNING):** Second-order metric structure persists. K26: ρ_variance ≈ +0.35 at ALL 18 measured checkpoints (3 seeds × 6 checkpoints, N=500-4000). Fisher-inspired asymmetry: Layer A (Gaussian surrogate: I_σ² = 1/(2σ⁴) > 0 always, dispersion remains learnable) + Layer B (deployed estimator confirms same qualitative pattern). Improvement trajectory (REPARAM-2, ORDER-1): +3.2pp at N=500, +5.4pp at N=4000.

**DK estimation — why coordinate descent:** K31: direct variance estimation (w=1/σ̂²) FAILS (65-76%). Coordinate descent succeeds (80-88%). Simulated factors have SIMILAR per-dimension variance, but DISCRIMINATIVE importance differs enormously. DK weights are classification-optimal axis importances, NOT inverse variances. DEFAULT: CoordinateDescentEstimator(n_rounds=5, max_per_cat=400, candidates=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]).

**DK concentration:** K34: effective scoring dimensionality drops from ~5.5 to ~4.0. Weight range [0.1, 5.0] = 50× importance ratio. DK is automatic feature selection.

**DK robustness to T₀ quality:** K32: +3.2pp at good centroids, +52pp at random. Phase 1 contributes only 0.3pp to DK effectiveness.

**Shrinkage safety:** K29: at α=0.5, 0/21 checkpoints below centroid (3 seeds × 7 checkpoints). At α=1.0 (pure DK): 3/21 below. DEFAULT: FixedAlpha(0.5). James-Stein adaptive deferred to v7.0.

**Per-(c,a) state:** Each (category, action) pair has independent phase, n_decisions, alpha, last_dk_estimation, novelty_rate, freeze_point.

**LearningStrategy interface:**

```python
class LearningStrategy(Protocol):
    def on_decision(self, f, category, action, correct, state) -> Outcome: ...
    def should_freeze(self, state) -> bool: ...
    def should_reestimate(self, state) -> bool: ...

class TwoPhaseStrategy:
    phase_policy: PhasePolicy          # DecisionCountPolicy(n=200) DEFAULT
    dk_estimator: DKEstimator          # CoordinateDescentEstimator DEFAULT
    shrinkage_schedule: ShrinkageSchedule  # FixedAlpha(0.5) DEFAULT
    novelty_tracker: NoveltyTracker    # NearestNeighborNovelty(max_look=300) DEFAULT

class ContinuousStrategy:
    """Legacy. Current production. α=0 always. No DK. No batch pipeline."""
```

**ProfileScorer modification (ONE LINE for scoring):**

```python
# In score(): compute effective weights
if learning_strategy and pair_state.phase == 'VARIANCE_LEARNING':
    w_tilde = alpha * w_dk + (1 - alpha)  # shrinkage interpolation
    kernel = DiagonalKernel(weights=w_tilde)
else:
    kernel = self._kernel  # existing (L2 or DK from P28)
```

**New observability APIs:** get_phase(), get_alpha(), get_novelty_rate(), get_dk_weights().

**New factory:** ProfileScorer.for_soc_twophase(mu, actions) — SOC with TwoPhaseStrategy defaults.

**Backward compatibility:** ProfileScorer(mu, actions) with no learning_strategy → ContinuousStrategy. α=0, no DK, no batch pipeline. IDENTICAL to v10.7. Zero downstream changes.

---

### 9.11 Batch Pipeline [NEW v10.8]

**Error budget (drives the pipeline's purpose):**

- REFERENCE 1 (Bayes floor, E1): 89.8% accuracy with zero noise. ~10% irreducible factor overlap.
- REFERENCE 2 (Production centroid): ~80-82% accuracy. Gap to floor: ~8-10pp.
- REFERENCE 3 (Oracle noise, F7): at 15% noise, centroid drops ~4pp.
- BOUNDARY error: ~8-10pp (Channel 1 DK + Channel 2 factor precision). Channel substitution at σ≈0.15 (RATE-5): each ~+2pp, combined ~+4pp.
- NOISE error: ~4-5pp (Channel 3 label quality). IRREDUCIBLE: ~10pp (factor overlap).

**7 lifecycle steps:** (1) Buffer verified decisions. (2) Compute novelty (§9.12). (3) Check composition (min novel fraction, category coverage, total count, flag rate). (4) Run DK estimation (coordinate descent). (5) Validate on holdout (promotion gate). (6) Promote or reject. (7) Record in BatchHistory (enables rollback).

**Three interfaces:**

```python
class BatchCompositionPolicy(Protocol):
    def should_estimate(self, batch_stats: BatchStats) -> bool: ...
    # Default: min_novel_fraction=0.10, min_category_coverage=3,
    #          min_total_count=50, max_flag_rate=0.30

class PromotionGate(Protocol):
    def should_promote(self, current_weights, new_weights,
                       holdout_data, centroids) -> PromotionDecision: ...
    # Default: epsilon=1.0pp, delta=2.0pp, n_min=20, holdout=20%
    # Non-inferiority check. Engineering safeguard, not hypothesis test.

class BatchHistory:
    def record(self, batch_id, weights_before, weights_after,
               promotion_decision, metrics) -> None: ...
    def rollback_to(self, batch_id) -> WeightState: ...
    # Rollback is idempotent. No data loss.
```

**Safety architecture (defense in depth):**

Layer 1: SHRINKAGE (mathematical guardrail). w̃ = α × w_DK + (1-α). At α=0.5: 0/21 below centroid (K29). James-Stein guarantees total squared error reduction on weights. NOTE: NOT a guarantee on classification accuracy — mathematical guardrail complemented by operational ones.

Layer 2: PROMOTION GATE (operational guardrail). No weights deploy without holdout non-inferiority. Catches cases where shrinkage alone doesn't prevent accuracy loss.

Layer 3: ROLLBACK (recovery). Instant revert to any prior promoted state. Also: set α=0 at any time → pure centroid. Instant.

---

### 9.12 NoveltyTracker [NEW v10.8]

**Experimental calibration:** K38: novelty-triggered re-estimation matches fixed-schedule accuracy with 12-25% fewer re-estimations. Primary value is as a CHANGE DETECTOR (novelty spikes signal distribution change before accuracy drops), not computational savings. RATE-6: d_nn correlation with DK improvement: r = +0.43.

**Interface:**

```python
class NoveltyTracker(Protocol):
    def compute_novelty(self, f, category, action, prior) -> float: ...
    def get_novelty_rate(self, category, action) -> float: ...

class NearestNeighborNovelty:
    """DEFAULT. d_nn to nearest prior decision in same (c,a) pair."""
    # Cost: max_look × D = 300 × 6 = 1800 multiplications. <1ms.
    # Storage: 24 pairs × 300 × D = 43,200 floats = 346KB.
```

**Four failure modes (framework v4 §4.3):**

1. **Collision:** Two novel patterns collide in factor space. RESPONSE: measure novelty vs EXISTING prior, not vs other new decisions.
2. **Gradual drift:** All vectors shift slowly. d_nn is small. RESPONSE: aggregate drift monitor alongside per-decision novelty. Conservation (LAGGING) catches accuracy impact.
3. **Factor-version change:** Factor computer update shifts distributions WITHOUT environmental change. False novelty spike. RESPONSE: ResetPolicy detects factor_schema_version changes, suppresses novelty-triggered re-estimation, reduces α, runs canary comparison.
4. **Correct detection:** Genuinely novel pattern. Works correctly. No response needed.

---


## 10. v5.0 Complete API Surface

### 10.1 Module Overview [UPDATED v10.7]

All v5.0 modules shipped. v6.0 COMPLETE. **884 tests** (v0.7.21: 870 at tag + 12 consumer contracts + 2 Phase 0 = 884). Current version: v0.7.22-pre.

```
gae/
├── profile_scorer.py    ✅ ProfileScorer, ScoringResult, KernelType, CentroidUpdate
│                            v6.0 additions: kernel, factor_mask, eta_override, auto_pause
│                            v0.7.21+ additions: ProfileScorer.for_soc() factory
│                                                τ + isfinite guards (lines 326, 332, 648)
│                                                η_override RuntimeWarning [0.005, 0.02] (line 703)
├── kernels.py           ✅ ScoringKernel protocol, L2Kernel, DiagonalKernel (28 tests)
│                            DiagonalKernel.compute_gradient = W/W.max()*(f-mu)
│                            Reference: gae/kernels.py:199-200 for the canonical form
├── covariance.py        ✅ CovarianceEstimator, CovarianceSnapshot (23 tests) — COLLECTS only, does not score
├── kernel_selector.py   ✅ KernelSelector, KernelRecommendation, KernelScore (46 tests)
│                            v10.7: Phase 2 rule-based PRIMARY. Phase 3 shadow = MONITORING only.
│                            No Phase 4 empirical lock. MIN_DECISIONS_FOR_RECOMMENDATION=100 (monitoring threshold).
│                            cumulative_analyst_prob instrumentation (lines 38-40, 272-280)
├── referral.py          ✅ ReferralEngine, ReferralRule, ReferralDecision, OverrideDetector (31 tests)
├── convergence.py       ✅ EPSILON=0.10, safety_factor=2.0
│                            compute_n_half(): exact discrete-time formula (lines 49-64) — N_half ≈ 13.51 at η=0.05
│                            compute_per_factor_n_half(): per-factor N_half under DiagonalKernel (lines 67-72) — NEW v0.7.21
│                            gamma_threshold(): ε_firm★ ≈ 0.125 (line 1087+) — formula θ cancels
│                            OLSMonitor: CUSUM on OLS, h=5.0 OLS scale, plateau-snapshot baseline (CLAIM-OLS-01)
│                            VarQMonitor: LOGGED ONLY — Bernoulli mixture theorem = PERMANENT HARD STOP for gating
├── enrichment_advisor.py ✅ Enrichment recommendation engine — ranks factors by expected Day-1 gain
│                            Validated on 5 deployment profiles. Integrates with P28 Phase 2 report.
├── calibration.py       ✅ CalibrationProfile + compute_factor_mask, mask_to_array,
│                            compute_eta_override [v0.7.22-pre: worst_case_quality parameter],
│                            derive_theta_min (Grok form: η × N_half² / T_max = 0.467),
│                            check_conservation, compute_optimal_tau, compute_breach_window,
│                            estimate_fisher_information, predict_n_half
│                            η_neg guard: ValueError on η_neg ≥ 1.0 (v0.7.8 fix)
├── factors.py           ✅ FactorComputer protocol, assemble_factor_vector
├── learning.py          ✅ LearningState (delegates to ProfileScorer.update)
├── scoring.py           ⚠️ DEPRECATED — forwards to ProfileScorer (remove at v7.0)
├── oracle.py            ✅ OracleProvider, GTAlignedOracle, BernoulliOracle
├── evaluation.py        ✅ run_evaluation, EvaluationScenario, EvaluationReport
├── judgment.py          ✅ compute_judgment, JudgmentResult
├── ablation.py          ✅ run_ablation, AblationReport
├── fisher.py            ✅ estimate_fisher_information, predict_n_half, enrichment_multiplier
├── contracts.py         ✅ Unchanged
├── primitives.py        ✅ Unchanged
├── events.py            ✅ Unchanged
├── store.py             ✅ ProfileScorer state persistence
├── types.py             ✅ Unchanged
├── embeddings.py        📋 Tier 4 — v6.5
├── bridge.py            📋 Level 2 GraphAttentionBridge — v7.0
└── discovery_engine.py  📋 Level 3 — v8.0

tests/
├── test_consumer_contracts.py  ✅ NEW v0.7.22-pre: 12 tests locking the downstream consumer contract
│                                   ScoringResult field stability, ProfileScorer.for_soc behavior,
│                                   eta_override warning bounds, auto-pause outcome semantics
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

[§10.2 unchanged from v10.6. No v10.7 changes.]

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

[§10.3 unchanged from v10.6. No v10.7 changes.]

### 10.4 OracleProvider (`gae/oracle.py`) — NEW v5.0

[§10.4 OracleProvider / OracleResult / GTAlignedOracle / BernoulliOracle UNCHANGED from v10.6. Interface contract preserved; no v10.7 changes.]

### 10.5 Evaluation (`gae/evaluation.py`) — NEW v5.0

[§10.5 EvaluationScenario / EvaluationReport / run_evaluation / compute_ece UNCHANGED from v10.6. ECE computation used in §9.6 DK calibration characterization; no API changes needed.]

### 10.6 Judgment (`gae/judgment.py`) — NEW v5.0

[§10.6 JudgmentResult / compute_judgment UNCHANGED from v10.6. Three confidence tiers (high/medium/discovery) preserved.]

### 10.7 Ablation (`gae/ablation.py`) — NEW v5.0

[§10.7 AblationReport / run_ablation UNCHANGED from v10.6.]

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

    UNI-DK-01 v5.3 characterization (April 2026):
      - +7.67pp asymptotic at NR=5.0 mean-σ=0.175 (1500 cells)
      - Monotone in NR: 0.00pp at NR=1.0 (mathematical identity) to +7.67pp at NR=5.0
      - ECE rises 0.055→0.42 across NR=1.0→5.0 (calibration degrades; see §9.6)
    """
    def __init__(self, weights: np.ndarray):
        """
        weights: (d,) — typically 1/σ² per factor.
                 Raw (un-normalized) weights are stored; normalization happens in gradient.
        """
        self.weights = np.asarray(weights, dtype=np.float64)

    def compute_distance(self, f, mu):
        """
        Distance uses raw (un-normalized) weights. Reference: gae/kernels.py:compute_distance.
        """
        return np.sum(self.weights * (f - mu)**2, axis=-1)

    def compute_gradient(self, f, mu):
        """
        Gradient uses NORMALIZED weights (W/W.max()). Reference: gae/kernels.py:199-200.

        GAE-GRADIENT-001 fix (v0.7.7): normalize before multiply.
          W/W.max() keeps weights in [0,1] — direction preserved, magnitude bounded.
          W*(f-mu) was WRONG: dominant weights corrupted gradient direction.

        Guard against numerical issues:
          w_max = max(self.weights.max(), 1e-9)
          return (self.weights / w_max) * (f - mu)

        When w_max is pathologically small (all weights ≈ 0), the guard prevents
        division by zero; the gradient effectively scales down to zero, which
        freezes learning for that update — correct behavior for a degenerate
        kernel.
        """
        w_max = max(self.weights.max(), 1e-9)
        return (self.weights / w_max) * (f - mu)
```

**28 tests** covering: protocol compliance, L2 distance/gradient, DiagonalKernel init/distance/gradient, unit weights = L2, binary weights = factor mask, zero weights → uniform probabilities, GAE-GRADIENT-001 regression test (W/W.max() vs W direct).

[v10.7 note: v10.6 documented the gradient formula correctly. v10.7 adds the explicit code form from gae/kernels.py:199-200 and the `w_max = max(..., 1e-9)` guard rationale. The UNI-DK-01 v5.3 characterization results are summarized in the docstring with a pointer to §9.6 for calibration properties.]

### 10.9 Covariance Estimator (`gae/covariance.py`) — NEW v6.0

[§10.9 CovarianceEstimator / CovarianceSnapshot UNCHANGED from v10.6. half_life_decisions=300 preserved; 23 tests preserved. v6.0 COLLECTS data only, does not affect scoring. v7.0 may feed ShrinkageKernel if off-diagonal value emerges.]

### 10.10 Kernel Selector (`gae/kernel_selector.py`) — REWRITTEN v10.7

**v10.7 architecture:** Rule-based PRIMARY, data-driven MONITOR only. No empirical lock. See §9.2 for the full rationale.

```python
@dataclass
class KernelRecommendation:
    recommended_kernel: str     # "l2" or "diagonal"
    confidence: float           # margin over runner-up (monitoring context) or rule-threshold margin (rule context)
    scores: Dict[str, Dict]     # per-kernel agreement rates from shadow comparison — MONITORING, not selection
    method: str                 # "rule" (Phase 2, PRIMARY) or "monitoring" (Phase 3, diagnostic)
                                # v10.6 had "empirical" (Phase 4) — removed in v10.7 (no empirical lock)
    reason: str                 # human-readable explanation
    sufficient_data: bool       # True if ≥100 decisions — threshold for monitoring output, NOT for selection


@dataclass
class KernelScore:
    """Per-kernel tracking during shadow comparison.
    Reference: gae/kernel_selector.py:30-40."""
    kernel_name: str
    total_decisions: int = 0
    agreements: int = 0
    disagreements: int = 0
    cumulative_confidence: float = 0.0
    cumulative_analyst_prob: float = 0.0  # v0.7.22-pre instrumentation

    # Properties (not fields):
    # agreement_rate -> agreements / total_decisions
    # mean_confidence -> cumulative_confidence / total_decisions
    # mean_analyst_action_prob -> cumulative_analyst_prob / total_decisions

    # NOTE: The rolling boolean buffer is stored on KernelSelector 
    # as self._buffers[kernel_name], NOT on KernelScore. KernelScore 
    # holds cumulative counts; the rolling window is a selector-level 
    # data structure used by recommend() for windowed monitoring.


class KernelSelector:
    """
    Kernel selector (v10.7 architecture).

    PRIMARY selection: preliminary_recommendation() — rule-based, uses measured σ.
      noise_ratio = max(σ_per_factor) / min(σ_per_factor)
      noise_ratio > 1.5 → diagonal
      noise_ratio ≤ 1.5 → l2

    MONITORING: record_comparison() accumulates shadow-mode KernelScore data.
      get_comparison_summary() returns the monitoring view (rolling agreement rates,
      cumulative analyst-probability, etc.) for drift detection and future
      re-evaluation. recommend() returns a KernelRecommendation with
      method="monitoring" once ≥100 decisions have been recorded — this is an
      OBSERVATION, not a selection authority.

    In v6.0, the production scorer uses preliminary_recommendation() only.
    Selection never switches based on shadow comparison data.

    Change handling: should_reconsider() on σ change, ρ change, covariance λ drop —
    triggers a fresh preliminary_recommendation() call with updated σ. Still
    rule-based. If σ shifts enough to flip the noise_ratio > 1.5 threshold,
    the selector's rule output changes; otherwise the rule is stable.

    Design principle: "Ship the rule. Use shadow data for drift detection, not for
    automated switching. In v6.0, rule-based has two independent validations —
    V-MV-KERNEL HC-personas 4/4 and UNI-DK-01 v5.3 E3 100% (269 informative cells).
    Confidence-based comparison was found unreliable (v5.3 §10.6)."

    Reference: math_synopsis v14 §3.3 for the UNI-DK-01 v5.3 characterization
    that drives this architecture.

    Version history:
      v0.7.0       First shipped (Phase 2/3/4 empirical lock architecture)
      v0.7.18-20   KERNELSEL-001 tiebreaker, raw_weights fix
      v0.7.21+     cumulative_analyst_prob instrumentation added
      v10.7 doc    architecture inverted (Phase 4 empirical lock removed from spec)
                   Code change: semantic reinterpretation, not structural.
                   recommend() still works but is documented as monitoring-only.
    """

    # Monitoring threshold — decisions required before recommend() emits output.
    # NOT a selection threshold (v10.7). Selection is always rule-based.
    MIN_DECISIONS_FOR_RECOMMENDATION: int = 100

    def __init__(
        self,
        d: int,
        sigma_per_factor: np.ndarray,
        correlation_max: float = 0.0,
    ) -> None: ...

    def preliminary_recommendation(self) -> KernelRecommendation:
        """
        PRIMARY SELECTION in v6.0.
        Rule-based recommendation from measured σ. Used at deployment time.

        Returns KernelRecommendation with method="rule".
        """
        ...

    def record_comparison(
        self,
        factors: np.ndarray,
        category_index: int,
        mu: np.ndarray,
        analyst_action_index: int,
        actions: List[str],
    ) -> Dict[str, int]:
        """
        MONITORING: Record shadow-mode comparison data.
        Scores the same (f, c) with BOTH L2 and DiagonalKernel; records agreement
        with analyst action for each. Accumulates into KernelScore per-kernel.

        Data accumulated here does NOT drive selection in v6.0. It enables:
          - Drift detection (is DK's agreement rate declining over time?)
          - Future re-evaluation (if Option B holdout-based selector is later added)
          - Monitoring dashboards (shadow comparison visible to operators)

        Returns counts dict: {"l2": N_l2_agree, "diagonal": N_dk_agree}.
        """
        ...

    def recommend(self) -> KernelRecommendation:
        """
        MONITORING output (was empirical selection in v10.6; reinterpreted v10.7).

        Returns a KernelRecommendation with method="monitoring" indicating which
        kernel has higher rolling agreement rate in shadow data. This is an
        OBSERVATION for operators, not a selection authority. The production
        scorer does not consume this to switch kernels.

        If total_decisions < MIN_DECISIONS_FOR_RECOMMENDATION (100), returns
        a fallback recommendation with sufficient_data=False and method="rule"
        (echoing preliminary_recommendation()).

        v6.0 note: this method's return value should be surfaced in monitoring
        dashboards with a label like "Shadow agreement rate observation" so
        operators don't confuse it with selection authority. If shadow
        monitoring shows sustained disagreement with the rule's choice over
        ≥1000 verified decisions, raise a ticket for human review — that's
        the signal that Option B (accuracy-comparison holdout) might need to
        be activated as a v6.1 enhancement.
        """
        ...

    def should_reconsider(
        self,
        new_sigma: Optional[np.ndarray] = None,
        new_rho_max: Optional[float] = None,
        covariance_lambda: Optional[float] = None,
    ) -> Optional[str]:
        """
        Triggers fresh rule evaluation on σ/ρ/λ changes.

        If new_sigma changes noise_ratio across the 1.5 threshold, the kernel
        switches. Returns a change description string if a switch is recommended,
        else None. The caller (typically ConservationMonitor or the deployment
        orchestrator) decides whether to actually switch kernels.

        Still rule-based in v10.7 — does not consult shadow comparison data.
        """
        ...

    def get_comparison_summary(self) -> Dict:
        """
        Returns the full monitoring view: per-kernel rolling agreement rates,
        cumulative_confidence, cumulative_analyst_prob, total decisions,
        disagreement rate between kernels, etc. For dashboards.
        """
        ...

    def reset_comparison(self) -> None:
        """Clear shadow-mode accumulators. Used on configuration change."""
        ...
```

**46 tests** covering: KernelScore dataclass (with cumulative_analyst_prob), initialization, preliminary rules (uniform σ → L2, heterogeneous σ → diagonal, high ρ → consider shrinkage [deferred to v7.0]), record comparison, accumulation, recommend (insufficient data returns rule-based fallback; sufficient data returns monitoring observation), summary (all KernelScore fields), monitoring (σ change triggers reconsideration, ρ change triggers, λ drop triggers), reset.

**v10.7 test additions (indirect):** the 12 consumer contract tests in `tests/test_consumer_contracts.py` include 2 tests locking the KernelRecommendation `method` field values ("rule" | "monitoring") and the reinterpretation of the MIN_DECISIONS_FOR_RECOMMENDATION threshold (monitoring, not selection).

**Retired from the v10.6 spec (v10.7):**
- "Phase 4 (QUALIFY): lock winner after ~250 verified decisions" — retired. No empirical lock exists.
- "method: 'empirical' (Phase 4)" — retired. method is "rule" or "monitoring".
- "max(1000, 20×V×α) decisions" as a selection-lock threshold — retired from this document. Math_synopsis v14 retains the figure as a V-GATE-STABILITY baseline for other purposes.
- "Phase 4 (empirical) always overrides Phase 2 (rule) on genuine disagreement" — retired. The rule is never overridden by shadow data in v6.0.

### 10.11 Calibration Additions (`gae/calibration.py`) — v6.0 extensions [UPDATED v10.7]

```python
# Factor mask (DEPRECATED by DiagonalKernel — still functional for backward compat)
def compute_factor_mask(
    sigma_per_factor: Dict[str, float],
    threshold: float = 0.20,
) -> Dict[str, bool]:
    """Binary include/exclude. sigma < threshold → True (include).
    DEPRECATED: use DiagonalKernel continuous weighting instead."""

def mask_to_array(
    mask: Dict[str, bool],
    factor_names: List[str] = None,
) -> np.ndarray:
    """Dict → float64 array (1.0=include, 0.0=exclude). SOC default factor order built-in.
    DEPRECATED: use DiagonalKernel continuous weighting instead."""


# Asymmetric η [UPDATED v10.7 signature — worst_case_quality parameter added v0.7.22-pre]
def compute_eta_override(
    eta_confirm: float = 0.05,
    mean_quality: float = 0.75,
    quality_variance: float = 0.02,
    worst_case_quality: Optional[float] = None,  # NEW v0.7.22-pre, gae/calibration.py:354
) -> float:
    """
    Diagnostic formula for η_override. Global default η_override=0.01 is the validated
    shipping value — this function is used for deployment characterization, not for
    overriding the shipped default in production.

        η* ∝ (2q̄ − 1) / (2σ²_q + signal)

    Directionally correct, ~2× overestimate vs empirical optimum.

    New worst_case_quality parameter (v0.7.22-pre):
      If provided, the formula uses min(mean_quality, worst_case_quality) as the
      effective quality input. Produces a more conservative η_override estimate
      for heterogeneous teams (e.g., team mean q̄=0.82 but 10th-percentile analyst
      at 0.65).

    Example (deployment qualification, NOT production override):
        eta_diagnostic = compute_eta_override(
            mean_quality=0.82,
            quality_variance=0.04,
            worst_case_quality=0.65,  # 10th-percentile analyst from shadow data
        )
        if eta_diagnostic < 0.005:
            # Deployment has worse-than-validated-range quality.
            # File a ticket; review deployment qualification.
            ...
        # Ship 0.01 anyway — the diagnostic is informational, not authoritative.
    """
    ...


# Conservation [UPDATED v10.7 — q definition documented explicitly]
def derive_theta_min(
    eta: float = 0.05,
    n_half: float = 14.0,
    t_max_days: float = 21.0,
) -> float:
    """
    θ_min = η × N_half² / T_max.

    At η=0.05, N_half=14, T_max=21 days: θ_min ≈ 0.467.

    Grok form (v11+): This is the deployment-specific conservation floor.
    At V=200, α=0.25: θ_min × (α × V)^-1 → feasible.
    At V=50, α=0.25: θ_min × (α × V)^-1 = 1.88 → impossible (deployment ineligible).

    Formally equivalent to CLAIM-SC-01 scope condition.
    Reference: gae/calibration.py:151-183.

    v10.7 note: This formula and canonical value (0.467) are UNCHANGED from v10.6
    and live code. The narrative in some v10.6 sections was ambiguous about whether
    the formula used η_confirm or η_override — it's η (confirm path = 0.05), which
    is the effective rate for undisrupted decisions. This clarification is
    documentary only, no formula change.
    """
    return eta * n_half ** 2 / t_max_days


def check_conservation(
    alpha: float,        # analyst override rate — fraction disagreeing with system
    q: float,            # decision quality — see v10.7 operational definition below
    v: float,            # verified decisions per day
    theta_min: float = 0.467,
) -> str:
    """
    Check the conservation invariant α × q × V ≥ θ_min.
    Returns 'GREEN', 'AMBER', or 'RED'.

    GREEN: α × q × V ≥ θ_min  (system is compounding; learning is safe)
    AMBER: (0.7 × θ_min) ≤ α × q × V < θ_min  (degradation detected; auto-pause triggers)
    RED:   α × q × V < 0.7 × θ_min  (severe degradation; auto-pause + operator alert)

    v10.7 operational definition of q:
      q(t) = rolling verified accuracy over the last 400 decisions
           = (1 / N_v(t)) × Σ_{i ∈ W(t)} 𝟙[prediction_i = verified_action_i]

      where W(t) = last 400 decisions, N_v(t) ≈ 100 at 25% verify rate.

      q is a rolling verified accuracy, NOT per-decision confidence. The v10.6
      spec was ambiguous on q's source; v10.7 explicitly defines it as rolling
      verified accuracy. See §9.8 for the full rationale; math_synopsis v14 §5
      for the derivation preservation under this substitution.

      Rationale:
        - DK's confidence is miscalibrated at high NR (ECE 0.42 at NR=5.0, §9.6).
          Confidence-based q would be unreliable under DK routing.
        - Rolling accuracy is calibration-independent and kernel-agnostic.
        - Matches the aggregate nature of α and V (also rolling over same window).
        - Strengthens EU AI Act Article 14 oversight argument (accuracy is more
          defensible than confidence trajectory).

    Implementation note: callers provide q as a rolling-accuracy value, computed
    by ConservationMonitor.compute_q() or equivalent. This function does not
    compute q internally.
    """
    product = alpha * q * v
    if product >= theta_min:
        return 'GREEN'
    elif product >= 0.7 * theta_min:
        return 'AMBER'
    else:
        return 'RED'


# Fisher information (UNCHANGED from v10.6)
def estimate_fisher_information(centroids: np.ndarray, tau: float = 0.1) -> float: ...
def predict_n_half(fisher: float, eta: float = 0.05) -> float: ...
def enrichment_multiplier(fisher_before: float, fisher_after: float) -> float: ...
```

### 10.12 Public API (`gae/__init__.py`) — updated v10.7

```python
# All symbols exported from gae/__init__.py as of v6.0 / v0.7.22-pre
from gae import (
    # Core scoring
    ProfileScorer,
    ScoringResult,
    KernelType,
    CentroidUpdate,
    build_profile_scorer,

    # Kernels (v6.0)
    L2Kernel,
    DiagonalKernel,

    # Covariance (v6.0 — collects only)
    CovarianceEstimator,
    CovarianceSnapshot,

    # Kernel selection (v6.0, architecture clarified v10.7)
    KernelSelector,
    KernelRecommendation,
    KernelScore,              # NEW public export v10.7 (shadow-mode monitoring dataclass)

    # Configuration
    CalibrationProfile,

    # Calibration utilities (v6.0 / v10.7)
    compute_factor_mask,      # DEPRECATED by DiagonalKernel
    mask_to_array,            # DEPRECATED by DiagonalKernel
    compute_eta_override,     # UPDATED v10.7: worst_case_quality parameter
    derive_theta_min,         # Grok form: η × N_half² / T_max
    check_conservation,       # q now defined as rolling verified accuracy (v10.7)

    # Convergence (v10.4/v10.7)
    compute_n_half,           # UPDATED v10.7: exact discrete-time formula
    compute_per_factor_n_half, # NEW v0.7.21: per-factor N_half under DK
    centroid_distance_to_canonical,
    gamma_threshold,          # UPDATED v10.7: returns ≈0.125 for production values (was 0.128)
    phase2_effective_threshold,
    ConvergenceTrace,

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
python -c "from gae import ProfileScorer, KernelType, OracleProvider, run_evaluation, compute_judgment, run_ablation, KernelScore, compute_per_factor_n_half"
# Must succeed with no errors. v0.7.22-pre surface.
```

**Consumer contract tests:**
```bash
python -m pytest tests/test_consumer_contracts.py -v
# 12 tests, all must pass. Locks ScoringResult fields, for_soc factory behavior,
# eta_override warning bounds, auto-pause outcome semantics, KernelRecommendation
# method field values.
```

### 10.13 Synthetic Data Generation (`gae/synthetic.py`) — NEW v10.4

[§10.13 UNCHANGED from v10.6. Synthetic centroid generation, event stream generation, seed management for reproducibility. Used by oracle separation experiments, UNI-DK-01 v5.3, and downstream benchmarks.]

### 10.14 Convergence Analysis (`gae/convergence.py`) — REWRITTEN v10.7

**Module responsibility:** Model-independent convergence metrics for production deployment monitoring and EXP-G1 measurement. Separates centroid convergence from N_half accuracy threshold crossing.

**v10.7 changes:**
- `compute_n_half()` exact discrete-time formula documented (N_half ≈ 13.51 at η=0.05, NOT 13.86 continuous approximation).
- `compute_per_factor_n_half()` documented (NEW v0.7.21).
- `gamma_threshold()` returns ≈0.125 (v10.6 had 0.128 in 6 sites — all corrected).
- Fourth proof path for γ theorem added (centroid-distance derivation from Grok, April 16).

```python
# gae/convergence.py

def compute_n_half(eta: float = 0.05) -> float:
    """
    Convergence half-life: decisions needed for mean centroid error to decay by 1/2.

    EXACT discrete-time formula (gae/convergence.py:49-64):
      N_half = ln(2) / ln(1 / (1 - η))

    At η=0.05: N_half ≈ 13.51.

    This is the exact value for discrete-time centroid updates
    μ_{n+1} = μ_n + η (f − μ_n), which is the actual update rule in
    ProfileScorer.update(). Each confirm step reduces the centroid error
    by a factor of (1 − η), so after n steps the error is (1 − η)^n × e_0.
    Half-life solves (1 − η)^N = 1/2 → N = ln(2) / ln(1 / (1 − η)).

    CONTINUOUS-TIME approximation (for comparison):
      N_half ≈ ln(2) / η ≈ 13.86 at η=0.05

    Difference: 0.35 decisions (3 parts per hundred). Small but measurable.
    Precision-sensitive callers (e.g., deployment calendar estimators,
    tight-budget experiments) should use the exact 13.51.

    v10.7 note: This formula and exact value are documented in the design doc
    for the first time. Previous versions rounded to 14 throughout. The
    rounded value remains correct for general narrative use (N_half ≈ 14)
    but the exact 13.51 is exposed for precision-sensitive callers.
    """
    return float(np.log(2) / np.log(1 / (1 - eta)))


def compute_per_factor_n_half(
    sigma_per_factor: np.ndarray,
    eta: float = 0.05,
) -> np.ndarray:
    """
    Per-factor N_half under DiagonalKernel [NEW v0.7.21].
    Reference: gae/convergence.py:67-72.

    Under DK with W = diag(1/σ²) normalized to max=1, each factor's effective
    learning rate is (w_i/w_max) × η. High-w factors (low σ) learn at nearly
    the full rate; low-w factors (high σ) learn slowly.

    Per-factor N_half:
      w_norm_i = (1/σ_i²) / max_j (1/σ_j²)
      η_eff_i  = w_norm_i × η
      N_half_i = ln(2) / ln(1 / (1 − η_eff_i))

    For DK at NR=5.0 with σ = [0.058, 0.105, 0.152, 0.198, 0.245, 0.292]:
      w_norm = [1.00, 0.31, 0.15, 0.09, 0.06, 0.04]
      η_eff  = [0.050, 0.0154, 0.0073, 0.0044, 0.0029, 0.0020]
      N_half = [13.5, 44.7, 94.2, 157.2, 236.4, 345.9]

    At low-σ factors, N_half ≈ 14 (same as scalar L2).
    At high-σ factors, N_half grows to 300+ decisions. This is correct:
    noisy factors should learn slowly (their inputs are less reliable).

    For L2 deployments (uniform W=I), this function returns a constant array
    of length d with every entry equal to scalar compute_n_half(eta).

    Use cases:
      - Deployment qualification: predict per-factor convergence calendar
      - EXP-G1 instrumentation: log per-factor centroid_distance trajectories
      - Diagnostic: identify factors that are "slow-learning" in the current σ regime

    NOT used for scoring or learning — informational API only.
    """
    ...


def centroid_distance_to_canonical(
    mu: np.ndarray,
    canonical: np.ndarray,
) -> float:
    """
    Frobenius distance between the current centroid tensor and the canonical (GT) snapshot.
    Model-independent convergence signal: decreases monotonically under production
    learning dynamics regardless of seed or factor vector quality.

    Replaces N_half as the primary γ measurement metric.
    Simulation finding (April 2026): N_half variance 27× (γ=13.4 vs γ=0.48) at 3 seeds
    vs centroid distance decreasing monotonically in every seed, every phase.
    """
    return float(np.linalg.norm(mu.flatten() - canonical.flatten()))


def gamma_threshold(
    alpha_cat: float,
    delta_norm: float,
    theta: float = 0.85,  # accepted for backward compat; does NOT appear in the formula (v10.7 correction)
) -> float:
    """
    Computes the ε_firm threshold below which γ ≤ 1 (re-convergence theorem).

    v10.7 formula (CORRECTED from v10.6):
      ε_firm★ = α_cat × ‖Δ‖ / (1 − α_cat)

    Production values: alpha_cat=2/6≈0.33, delta_norm≈0.25 → 0.125.
    (Previously documented as 0.128 in 6 sites; v10.7 swept all to 0.125.)

    Reference: gae/convergence.py:1109.
      return alpha_cat * delta_norm / (1 - alpha_cat)

    Note on theta parameter: theta is accepted by the function signature for
    backward compatibility, but theta CANCELS in the correct derivation and
    does not appear in the formula. Passing a non-default theta has no effect.
    A future release may deprecate and remove the parameter; downstream
    consumers should not rely on theta in gamma_threshold() calls.

    Derivation summary (full proof in math_synopsis v14 §3.2):
      The re-convergence theorem requires ε_firm > ε_firm★ for γ > 1.
      The threshold is derived from the requirement that Phase 2's effective
      accuracy target (p_d★ from phase2_effective_threshold) is reached faster
      than Phase 1's full-category calibration from ε_firm.

    Production: ε_firm ∈ [0.15, 0.40]. Every real deployment clears 0.125.
    Simulation: ε_sim = 0.05 (below threshold; correctly predicts γ < 1).
    """
    return alpha_cat * delta_norm / (1 - alpha_cat)


def phase2_effective_threshold(
    alpha_cat: float,
    theta: float = 0.85,
) -> float:
    """
    The effective accuracy required from disrupted categories for rolling-window
    to declare Phase 2 complete. p_d★ = (θ - (1 - α_cat)) / α_cat ≈ 0.55.
    Explains why Phase 2 is shorter than Phase 1: undisrupted categories carry
    the rolling window, reducing the effective target for disrupted categories.
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

**Re-Convergence Theorem (formal statement for this module — v10.7):**

```
Theorem (April 8, 2026, four-proof-path confirmed April 16, 2026):

  γ = N_half,1 / N_half,2 > 1   ⟺   ε_firm > γ_threshold(α_cat, ‖Δ‖) ≈ 0.125

Conditions:
  (1) Category-sparse disruption: c_d/C ≈ 0.33
  (2) Warm-started centroids: μ_T1 ≈ GT_1 at Phase 2 start
  (3) ε_firm = ‖μ_0 − GT_1‖ > 0.125

Four proof paths:
  (i)   Geometric — Phase 1's convergence challenge (all C categories, from ε_firm)
        exceeds Phase 2's (c_d categories, reduced target p_d★) when ε_firm > ε_firm★.
  (ii)  Dimensional — Phase 2 has effective convergence dimension ratio C/c_d = 3,
        giving a lower bound γ ≥ (C/c_d) × (θ / (θ − (1 − α_cat))) ≈ 4.6 in the
        idealized limit.
  (iii) η₋ trap avoidance — Phase 2 maintains η_eff near η_confirm = 0.05 because
        67% of decisions are undisrupted; Phase 1 may enter the η_override ≈ 0.01
        regime under large ε_firm.
  (iv)  Centroid-distance (Grok, April 16, 2026) — Direct derivation from
        dist(t) = ‖μ(t) − GT‖_F trajectory. Under the category-sparse assumption,
        Phase 2's distance-decay rate exceeds Phase 1's at the same t because
        the distance decay operates on fewer "active" centroid subspaces.
        This is the cleanest proof for EXP-G1 measurement (uses the same metric
        the experiment actually logs).

Confirmed: GPT-4.1, Claude Opus 4, Grok 3, Gemini 1.5 Pro (April 8, 2026);
  Grok 3 centroid-distance path (April 16, 2026).
Simulation: binary prediction correct in both directions
  (ε_sim=0.05 → γ=0.714 < 1; ε_sim=0.20 → γ=1.033 > 1).
Production ε_firm ∈ [0.15, 0.40]. Every real deployment clears 0.125.

Commercial claim: CC-21 (Tier 2 — conditional). EXP-G1 → Tier 1.
Reference: math_synopsis v14 §3.2, claims_registry v5 §B.5.
```

**Tests:** ≥4 core tests covering: `gamma_threshold` returns ≈0.125 for production values (α_cat=2/6, Δ=0.25); `gamma_threshold` ignores theta parameter (regression against v10.6 theta-dependent formula); `phase2_effective_threshold` returns ~0.55 for alpha_cat=2/6; `centroid_distance_to_canonical` is zero for identical tensors; `ConvergenceTrace.n_half_gap` is True when rolling accuracy crosses θ before centroid distance drops to 20% of initial. Additional test for `compute_per_factor_n_half` consistency (uniform σ → constant array equal to scalar `compute_n_half`; heterogeneous σ → strictly monotone array in factor noise).

### 10.15 Experiment Library (`gae/experiments/`) — NEW v10.6

[§10.15 UNCHANGED from v10.6. Three strict rules (Apache 2.0 only, numpy+gae deps only, research/examples not production). Module structure (evals/, oracle_separation/, domains/). EXP-C1 reproduction script, oracle separation validation, domain examples (medical_triage, supply_chain, financial_approval).]

**v10.7 note:** No new experiment additions planned for v10.7 scope. UNI-DK-01 v5.3 remains in cross-graph-experiments (not published as an example) because it depends on the full v5.3 parameterization which is research-specific rather than generic example material. If a future v10.8+ wants to expose UNI-DK-01 as a reproducible public example, that would live under `gae/experiments/kernel_characterization/` with the three strict rules applied.

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

[v10.7 note on γ: The re-convergence theorem (γ > 1 ⟺ ε_firm > 0.125) is a DIFFERENT γ from the scaling exponent in 𝒮(n,t). The re-convergence γ is the ratio of phase-specific convergence half-lives (N_half,1/N_half,2); the scaling γ is a temporal compounding exponent for cross-domain discovery. Do not conflate. The re-convergence γ is established (§10.14, math_synopsis v14 §3.2); the scaling γ requires EXP-G1.]

### 11.4 Level 1 Data Preservation Hooks (every domain must write)

[§11.4 DecisionRecord / OutcomeRecord / ProfileSnapshot dataclasses UNCHANGED from v10.6. PLAT-4 validated. Every Level 1 domain writes these on every decision cycle.]

---

## 12. Contracts, API Pipeline, Accumulation

### 12.1 Public API Pipeline (v5.0 → v5.5)

```
# v5.0 pipeline (LIVE):
alert → FactorComputers → f
f → SituationAnalyzer → c
f, c → ProfileScorer.score(f, c) → ScoringResult
ScoringResult → compute_judgment(...) → JudgmentResult
outcome → ProfileScorer.update(f, c, a, correct) → CentroidUpdate

# v6.0 additions [v10.7 updated]:
deployment σ → KernelSelector.preliminary_recommendation() → L2 or DiagonalKernel (RULE-based)
shadow data (MONITORING only) → KernelSelector.record_comparison() → accumulated KernelScore
                                → KernelSelector.recommend() → monitoring observation (not selection)
rolling decisions → ConservationMonitor.compute_q() → q (rolling verified accuracy)
α × q × V → check_conservation(theta_min) → GREEN/AMBER/RED
AMBER/RED → ProfileScorer.set_conservation_status() → freeze learning
ScoringResult → triage_rank() (softmax entropy) → UI ordering [NEW v10.7]

# Evaluation pipeline (LIVE):
scenarios, oracle → run_evaluation(scorer, scenarios, oracle) → EvaluationReport
scoring_results → run_ablation(scorer, scenarios, factor_names) → AblationReport

# v5.5 additions:
GraphSnapshot → EmbeddingProvider.embed_graph() → E (n_entities × d)
E_i, E_j → GraphAttentionBridge.enrich() → E_enriched (with LayerNorm)
```

### 12.2 Semantic Accumulation Channels

[§12.2 table UNCHANGED from v10.6: Channel A profile centroids, B decision nodes, C outcome nodes, D profile snapshots, E enriched embeddings. Update cadence and triggers preserved.]

**v10.7 addition to Channel B:** DecisionRecord now includes the kernel used for that decision (`"l2"` or `"diagonal"`) as a convenience field for downstream analysis. This is not a new semantic channel; it's metadata on existing records. Allows post-hoc kernel-specific accuracy analysis without re-running the KernelSelector rule against logged σ.

### 12.3 Referral Routing — Independent of Action Scoring [v10.1, UNCHANGED v10.7]

[§12.3 full content UNCHANGED from v10.6. File: `gae/referral.py` (31 tests). Architecture: ProfileScorer and ReferralEngine independent, orthogonal. Referral is VETO. Confidence gate REJECTED for referral (14% precision, 38.7% FPR). Policy rules R1-R7: 72.7% DR, 12% FPR. Three-phase architecture (v6.0 Rules / v6.5 OverrideDetector / v7.0 monthly retrain). Components in `gae/referral.py`. SOC Rules R1-R7 table. R4 detection note. Referral problem decomposition (65.5% rule-expressible + 13.8% context-dependent + 20.7% emergent). Five design properties P-REF-1 through P-REF-5.]

### 12.4 Consumer Contract (NEW v10.7)

`tests/test_consumer_contracts.py` (12 tests) locks the downstream consumer surface. These are the commitments that soc-copilot and ci-platform rely on — any GAE change that breaks these tests requires a coordinated downstream release.

Locked contracts:
1. **ScoringResult fields** — at minimum these 5 fields, all required: `action_index: int`, `action_name: str`, `probabilities: np.ndarray`, `distances: np.ndarray`, `confidence: float`. No removals, no renames, no type changes on existing fields. Additions of NEW fields (e.g., entropy, confidence_gap) are non-breaking because all construction uses keyword arguments (verified by Codex — no positional ScoringResult construction exists in the repo). New fields must have defaults so existing consumers are unaffected.
2. **probabilities shape** — always `(n_a,)`, sum to 1 within floating-point tolerance, all values in `[0, 1]`.
3. **distances shape** — always `(n_a,)`, same length as `probabilities`.
4. **confidence value** — exactly equals `probabilities.max()`.
5. **ProfileScorer.for_soc()** — returns a configured scorer with `eta_override=0.01`, `auto_pause_on_amber=True`, `tau=0.1`. Kwargs override defaults. Signature: `for_soc(cls, mu, actions=None, **kwargs)`.
6. **η_override RuntimeWarning** — fires for values outside `[0.005, 0.02]`. Does NOT fire for default 0.01 or boundary values 0.005, 0.02.
7. **τ guard** — raises `ValueError` on `tau <= 0` or `tau > 10`.
8. **isfinite guards** — raise `ValueError` on non-finite f or μ values.
9. **Auto-pause outcome** — `CentroidUpdate` returned from `update()` while paused has `outcome='paused_conservation'`; no μ modification.
10. **KernelRecommendation.method** — one of `"rule"` or `"monitoring"` (v10.7 change; `"empirical"` removed).
11. **MIN_DECISIONS_FOR_RECOMMENDATION** — stable integer `100`; threshold for monitoring output, not selection.
12. **Public exports** — all symbols listed in §10.12 `gae/__init__.py` export block are importable.

**Breakage policy:** A change to any of these 12 contracts is a major-version break. soc-copilot and ci-platform pin to compatible GAE ranges; a GAE release that breaks any contract bumps the major version and publishes a migration note.

**v10.7 retired contracts from v10.6:** Callers of `ScoringResult.action_probabilities` (old alias name), `.factor_vector`, `.category_index`, `.temperature`, or `.kernel` via the result object will receive `AttributeError`. Those fields never existed in code — v10.6 documented them in error. See §9.3 correction note.

---

## 13. Equation Traceability Matrix [UPDATED v10.7]

| Equation | Paper Section | File | Function | Version | Status |
|---|---|---|---|---|---|
| **Eq. 4-final (L2)** | §3 | `gae/profile_scorer.py` + `gae/kernels.py:L2Kernel` | `ProfileScorer.score()` via kernel | v5.0 | ✅ LIVE |
| **Eq. 4-diagonal (DK)** | §3 | `gae/profile_scorer.py` + `gae/kernels.py:DiagonalKernel` | `ProfileScorer.score()` via kernel | v6.0 | ✅ LIVE |
| **Eq. 4b-final (dual push/pull + asymmetric η)** | §3 | `gae/profile_scorer.py` | `ProfileScorer.update()` | v6.0 | ✅ LIVE |
| Eq. 4 (published v4.1) | §3 original | `gae/scoring.py` | `score_alert()` | v4.1 | ⚠️ DEPRECATED (remove v7.0) |
| Eq. 4b (published v4.1) | §3 original | `gae/learning.py` | `LearningState.update()` | v4.1 | ⚠️ DEPRECATED — delegates to ProfileScorer |
| Eq. 4c (decay) | §3 | `gae/learning.py` | `apply_decay()` | v4.1 | ✅ Preserved |
| **Eq. 4-synthesis** | §5 (PROPOSAL) | `gae/profile_scorer.py` | `ProfileScorer.score(synthesis=σ)` | v6.0 | 🔵 GATED (GATE-M) |
| Eq. 5 (embeddings) | §4 | `gae/embeddings.py` | `embed_graph()` | v5.5 | 📋 Designed |
| Eq. 6 (cross-attention) | §4 | `gae/bridge.py` | `GraphAttentionBridge.enrich()` | v7.0 | 📋 Designed |
| Eq. 8a-8c (discovery) | §4 | `gae/discovery_engine.py` | `discover()` | v8.0 | 📋 Designed |
| ECE computation | - | `gae/evaluation.py` | `compute_ece()` | v5.0 | ✅ LIVE |
| **ReferralEngine** | §12.3 | `gae/referral.py` | `ReferralEngine.evaluate()` | v6.0 | ✅ LIVE |
| **Eq. GAMMA-THEOREM** | §3.2 (math_synopsis v14) | `gae/convergence.py` | `gamma_threshold()` | v10.4 | ✅ IMPLEMENTED |
| **Eq. GAMMA-THRESH** | §3.2 | `gae/convergence.py` | `gamma_threshold()` (returns 0.125) | v10.4 [CORRECTED v10.7] | ✅ IMPLEMENTED |
| **Eq. GAMMA-P_D** | §3.2 | `gae/convergence.py` | `phase2_effective_threshold()` | v10.4 | ✅ IMPLEMENTED |
| **Eq. GAMMA-DIST** | §3.2 | `gae/convergence.py` | `centroid_distance_to_canonical()` | v10.4 | ✅ IMPLEMENTED |
| **Eq. DK-DECOMP** (NEW v10.7) | §3.3 (math_synopsis v14) | *(analysis only; not a runtime function)* | N/A — characterization methodology | v10.7 | ✅ DOCUMENTED |
| **Eq. DK-ECE** (NEW v10.7) | §3.3 | `gae/evaluation.py:compute_ece` applied to DK outputs | (documentation of DK calibration properties — §9.6) | v10.7 | ✅ DOCUMENTED |
| **Eq. q-OPERATIONAL** (NEW v10.7) | §5, §9 (math_synopsis v14) | `gae/calibration.py:check_conservation` + ConservationMonitor.compute_q (soc-copilot) | q = rolling verified accuracy over last 400 decisions | v10.7 | ✅ SPECIFIED |
| **Eq. η_i (per-analyst)** | §3 (math_synopsis v14) | `gae/profile_scorer.py` | `ProfileScorer.update(analyst_precision=)` | v6.5 | 🔵 CONDITIONAL (D5: 0.86pp) |
| **compute_per_factor_n_half** (NEW v10.7) | — | `gae/convergence.py:67-72` | `compute_per_factor_n_half()` | v0.7.21 | ✅ IMPLEMENTED |
| **compute_n_half exact** | — | `gae/convergence.py:49-64` | `compute_n_half()` (returns 13.51 at η=0.05) | v0.7.21 | ✅ IMPLEMENTED |
| **OverrideDetector** | §12.3 | `gae/referral.py` | `OverrideDetector.predict()` | v6.5 | 🔵 STUB |

---

## 14. Design Decisions Log [UPDATED v10.7]

### 14.1 Original Design Decisions (v4.1)

*(See prior versions for full record. Summary:)*
NumPy-only (24 multiply-adds don't need PyTorch), three-repo architecture (enforce P12), FactorComputer Protocol in GAE (abstract interface), f(t) stored in Decision nodes (R4).

### 14.2 Experiment-Driven Decisions (v8+)

| Decision | Evidence | Impact |
|---|---|---|
| L2 as cold-start kernel | EXP-C1: 36.89pp gap | Settled |
| **DiagonalKernel as v6.0 default** | **V-MV-KERNEL: +13.2pp SOC, +6.8pp S2P (peak). UNI-DK-01 v5.3: +7.67pp at NR=5.0 characterized (1500 cells).** | **Settled (curve-based citation)** |
| **noise_ratio > 1.5 → diagonal (KernelSelector Option A)** | **V-MV-KERNEL HC-personas 4/4 + UNI-DK-01 v5.3 E3 100% (269 cells)** | **Settled. Rule-based PRIMARY in v6.0.** |
| **KernelSelector architecture: rule-based PRIMARY, data-driven MONITOR only** | **UNI-DK-01 v5.3 E1/E3/E4: confidence-based comparison untenable. mean_conf never picks DK; trimmed_ll below chance at NR≥3.** | **Settled v10.7. No Phase 4 empirical lock. Option B retained as v6.1 contingency. Option C archived.** [NEW v10.7] |
| **ShrinkageKernel deprioritized** | **D2/D3: off-diagonal <1pp** | **Deferred to v7.0** |
| **Factor mask deprecated** | **V-HC-CONFIG: mask hurt Day 1 by 6pp** | **DiagonalKernel supersedes** |
| **Asymmetric η (P0)** | **9 personas: 13-27pp degradation. 24 personas: validated. UNI-DK-01 v5.3 confirms positive learning at every NR (+0.19pp to +3.49pp).** | **η_override=0.01 permanent. "Negative learning at low NR" retracted as v4.2 measurement artifact.** [UPDATED v10.7] |
| **AMBER auto-pause signal = rolling verified accuracy** | **UNI-DK-01 v5.3: DK confidence miscalibrated at high NR (ECE 0.42 at NR=5.0). Confidence-based trigger unreliable under DK routing.** | **Signal source changed from confidence to accuracy. State-level trigger, not per-decision filter. Same signal as conservation-law q. §9.7.** [NEW v10.7] |
| **Conservation-law q = rolling verified accuracy over last 400 decisions** | **Same DK calibration finding. v13 derivation preserved under substitution — no step required q=confidence specifically.** | **Operational definition formalized. §9.8, §10.11 check_conservation. Math_synopsis v14 §5.** [NEW v10.7] |
| **Analyst triage ranking = softmax entropy / confidence gap** | **Distance-based triage mis-ranks boundary decisions. Raw max_p miscalibrated under DK. Entropy/gap capture decisiveness without calibration dependency.** | **§9.9 spec. Inherited by soc-copilot design v5.7+.** [NEW v10.7] |
| **DK calibration: prediction vs estimation channels** | **UNI-DK-01 v5.3: DK ECE 0.42 at NR=5.0 vs L2 0.04–0.06. Property of inverse-variance weighting, not defect.** | **Architectural pattern: DK outputs consumed as prediction (argmax, action_index) directly; estimation consumers (auto-pause, q, triage) use alternative signals. §9.6.** [NEW v10.7] |
| **ScoringResult contract: 5 fields** | **Code audit (Codex drift item 5, April 19, 2026): v10.6 documented 9 fields; live code has 5. `action_probabilities` alias, `factor_vector`, `category_index`, `temperature`, `kernel` never existed.** | **Doc corrected v10.7. 12 consumer contract tests lock the surface. Breakage policy: major-version break.** [NEW v10.7] |
| **ProfileScorer.for_soc() factory** | **Eliminates config boilerplate in soc-copilot tests/deployment. SOC canonical defaults encoded once.** | **NEW v0.7.21. `for_soc(cls, mu, actions=None, **kwargs)` sets eta_override=0.01, auto_pause_on_amber=True, tau=0.1.** [NEW v10.7] |
| **η_override warning bounds [0.005, 0.02]** | **P0 fix canonical = 0.01. Deployments outside this range deserve scrutiny, not silent acceptance.** | **NEW v0.7.22-pre. RuntimeWarning fires at gae/profile_scorer.py:703. Warning suppressible but must be justified in deployment log.** [NEW v10.7] |
| **γ-threshold value = 0.125** | **Correct formula: α_cat · ‖Δ‖ / (1 − α_cat) (θ cancels). v10.6 had 0.128 in 6 sites from earlier theta-dependent derivation.** | **All 6 sites swept to 0.125. gae/convergence.py:1109 is canonical reference. Math_synopsis v14 §3.2 matches.** [CORRECTED v10.7] |
| **Fourth γ-theorem proof path: centroid-distance** | **Grok independent derivation (April 16, 2026). Uses the same metric EXP-G1 logs — cleanest proof for empirical validation.** | **Added as path (iv) in §10.14 theorem statement. Three structural paths (geometric, dimensional, η₋-trap) + one empirical-metric path.** [NEW v10.7] |
| **compute_n_half exact formula 13.51** | **Discrete-time: N_half = ln(2)/ln(1/(1-η)). Continuous approximation 13.86 was from ln(2)/η.** | **gae/convergence.py:49-64. Rounded N_half=14 remains correct for narrative; exact 13.51 exposed for precision-sensitive callers.** [NEW v10.7] |
| **compute_per_factor_n_half public API** | **Under DK with heterogeneous W, per-factor N_half varies 10-100× across factors. Needed for deployment calendar estimation and EXP-G1 diagnostics.** | **NEW v0.7.21. gae/convergence.py:67-72. Informational API, not used for scoring/learning.** [NEW v10.7] |
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

[§14.3 UNCHANGED from v10.6. Apache 2.0 / open math proprietary domain / users guide / minimal example / no SOC-specific code in GAE / published math as credibility.]

---

## 15. Open-Source Strategy

[§15 UNCHANGED from v10.6 except for test count in §15.2.]

### 15.1 Why Open-Source GAE

[§15.1 UNCHANGED. Three-part argument: engine is a commodity, domain expertise is the moat, open math is peer-reviewable credibility.]

### 15.2 Repository README, CONTRIBUTING, CHANGELOG, etc.

[§15.2 UNCHANGED except: test count updated to **884** (v0.7.22-pre). README quick-start: DiagonalKernel introduced with the noise_ratio > 1.5 rule. CLAIM-W2 / CLAIM-64 / CLAIM-65 cited in README context section.]

### 15.3 README Specification

[§15.3 UNCHANGED from v10.6.]

### 15.4 examples/procurement_approval/

[§15.4 UNCHANGED from v10.6. S2P 5×5×8 domain, runnable standalone, full README.]

### 15.5 Docstrings

[§15.5 UNCHANGED.]

### 15.6 Open-Source Test Strategy

[§15.6 UNCHANGED except: current test count 884 (was 527). Target remains ~900 for full open-source release quality. Block 3.7a test categories.]

---

## 16. Production Constraints (All Hard — Not Configurable) [UPDATED v10.7]

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
| **η_override** | **0.01** | **9-persona stress test: 13-27pp degradation without. UNI-DK-01 v5.3: positive learning at every NR with 0.01.** | **Centroid corruption from low-quality overrides** [UPDATED v10.7] |
| **η_override warning range** | **[0.005, 0.02]** | **Canonical 0.01 ±2× tolerance. Outside this range is experimental.** | **RuntimeWarning at gae/profile_scorer.py:703. Suppressible but must be justified.** [NEW v10.7] |
| **η_confirm** | **0.05** | **P0 fix validated across 24 personas + UNI-DK-01 v5.3** | **Under-learning from confirm path if reduced** |
| **θ_min** | **0.467 (at η=0.05, N_half=14, T_max=21)** | **Grok form: η × N_half² / T_max** | **Conservation law floor too lenient at lower values** |
| **Conservation-law q source** | **Rolling verified accuracy over last 400 decisions** | **UNI-DK-01 v5.3: DK confidence miscalibrated at high NR (ECE 0.42). Confidence-based q unreliable under DK routing.** | **Conservation invariant operates on noisy inputs; auto-pause fires incorrectly; EU AI Act Article 14 argument weakened.** [NEW v10.7] |
| **AMBER auto-pause trigger signal** | **Rolling verified accuracy drop below (baseline × 0.9) or absolute floor (0.60 standard / 0.70 safety-critical)** | **Same DK calibration finding. Same rolling signal as q.** | **State-level, not per-decision. Resume after ≥100 consecutive decisions above threshold.** [NEW v10.7] |
| **AMBER auto-pause mechanism** | **Freeze on AMBER/RED. Centroid updates return CentroidUpdate(outcome='paused_conservation'); no μ modification.** | **Three-judge consensus. Enforcement confirmed at gae/profile_scorer.py:184-188, :662-675 (Codex item 14, April 19, 2026).** | **Detection-without-response gap. Continued learning during degradation episode.** [UPDATED v10.7] |
| **Cross-repo auto-pause wiring (SOC-Q3)** | **soc-copilot must consume scorer.is_paused or update outcome='paused_conservation' to route decisions to human review** | **GAE ProfileScorer enforces pause (above). SOC backend must respect it.** | **GAE-level enforcement without SOC-level action = scorer halts learning but soc-copilot continues auto-acting. Tracked as SOC-Q3, NOT a GAE design gap.** [NEW v10.7] |
| **DiagonalKernel gradient** | **W/W.max()*(f-mu) ONLY** | **GAE-GRADIENT-001 fix v0.7.7. Live reference: gae/kernels.py:199-200.** | **W*(f-mu) corrupts learning direction — high-W factors dominate by magnitude not signal** |
| **η_neg guard** | **ValueError on η_neg ≥ 1.0** | **GAE 0.7.8 — η_neg=1.0 produced ECE=0.49 (FORBIDDEN)** | **η_neg=1.0 inverts the penalty signal — pushes centroid toward wrong action** |
| **DiagonalKernel for noise_ratio > 1.5** | **Default (KernelSelector Option A, rule-based PRIMARY)** | **V-MV-KERNEL 390 cells +13.2pp peak + UNI-DK-01 v5.3 1500 cells +7.67pp characterized. Rule validation: 4/4 + 100%.** | **13-22pp accuracy loss on heterogeneous noise. Confidence-based selection (mean_conf, trimmed_ll) tested and rejected.** [UPDATED v10.7] |
| **KernelSelector architecture** | **Rule-based PRIMARY (preliminary_recommendation). Shadow data (record_comparison) is MONITORING only. No Phase 4 empirical lock.** | **UNI-DK-01 v5.3 E1/E3/E4 — confidence-based comparison untenable given DK calibration properties.** | **Empirical lock based on mean_conf would always select L2 even at NR=5.0 where DK is +7.67pp better. recommend() surfaced as selection authority is a bug.** [NEW v10.7] |
| **DK confidence outputs NOT safe for direct downstream consumption** | **Production components needing estimation (auto-pause, q, triage) use alternative signals.** | **UNI-DK-01 v5.3: DK ECE 0.42 at NR=5.0 vs L2 0.04. Property of inverse-variance weighting, not defect.** | **Downstream components consuming raw DK max_p behave incorrectly under DK routing. Architectural pattern: DK outputs are prediction channels, not estimation channels. §9.6.** [NEW v10.7] |
| **ScoringResult fields** | **Minimum 5 required: action_index, action_name, probabilities, distances, confidence. No removals/renames. New fields (entropy, confidence_gap) non-breaking with defaults.** | **Live code gae/profile_scorer.py:77-101 + consumer contract tests (v0.7.22-pre, 12 tests).** | **Consumers breaking = major-version break. Additions are non-breaking (keyword construction verified by Codex).** [UPDATED v10.7] |
| **ShrinkageKernel** | **NOT shipped at v6.0** | **D2/D3: off-diagonal <1pp** | **Complexity without benefit** |
| **Auto-approve threshold recalibration under DK** | **Deferred to first v6.0 customer PROD-4b exercise** | **v5.5 threshold*(c) was calibrated for L2 at τ=0.1 (ECE=0.036). Under DK at NR>1.5, sharpened distributions may shift threshold-crossing frequencies.** | **Possible over- or under-selection in AUTO zone. Impact bounded; either re-calibrate per-category under DK or accept conservative coverage in heterogeneous deployments.** [FLAGGED v10.7] |

---

## 17. What's Built vs What's Next [UPDATED v10.7]

### 17.1 v5.0 — COMPLETE ✅ (246 tests including post-tag WIRING-1)

[§17.1 v5.0 content UNCHANGED from v10.6. ProfileScorer integration, Oracle, Evaluation, Judgment, Ablation, Users Guide, minimal_domain example.]

### 17.1a v6.0 — COMPLETE ✅ (517 tests, v0.7.17)

[§17.1a v6.0 content UNCHANGED from v10.6. Kernel architecture, CovarianceEstimator, KernelSelector, ReferralEngine, enrichment_advisor, convergence v2, OLSMonitor. Experimental validation bullets preserved.]

### 17.1b v0.7.18–v0.7.20 — COMPLETE ✅ (884 tests) [NEW v10.3]

[§17.1b UNCHANGED from v10.6. KERNELSEL-001 tiebreaker, DiagonalKernel raw_weights fix, Block 9.1–9.5 (CLAIM-66-70).]

### 17.1c Oracle Separation & γ Theorem (April 7–8, 2026) [NEW v10.4]

[§17.1c UNCHANGED from v10.6 except: fourth proof path (Grok centroid-distance, April 16) added to theorem summary. See §10.14 for the four-path presentation.]

### 17.1d v0.7.21–v0.7.22-pre — COMPLETE ✅ (884 tests) [NEW v10.7]

**ProfileScorer API additions:**
- `ProfileScorer.for_soc(cls, mu, actions=None, **kwargs)` classmethod factory
  (gae/profile_scorer.py:284-300). SOC canonical defaults: eta_override=0.01,
  auto_pause_on_amber=True, tau=0.1.
- η_override RuntimeWarning on values outside [0.005, 0.02] (line 703).
- τ guard: ValueError on tau ≤ 0 or tau > 10 (line 326).
- isfinite guards on f (line 332) and μ post-clip (line 648).
- ScoringResult documented correctly (5 fields, matches live code — Codex drift item 5 closed).

**Convergence additions:**
- `compute_per_factor_n_half(sigma_per_factor, eta)` (gae/convergence.py:67-72).
  Returns per-factor N_half under DiagonalKernel.
- `compute_n_half()` exact discrete-time formula documented (lines 49-64).
  N_half ≈ 13.51 at η=0.05 (was rounded to 14).
- `gamma_threshold()` returns ≈0.125 for production values (line 1087+).
  Formula α_cat · ‖Δ‖ / (1 − α_cat); theta parameter ignored (cancels in derivation).
  6 doc sites swept from 0.128 to 0.125 (Codex drift item 3 closed).

**Calibration additions:**
- `compute_eta_override(..., worst_case_quality=None)` parameter
  (gae/calibration.py:354). Diagnostic formula produces more conservative
  η_override estimate when worst_case_quality is provided.
- `check_conservation(alpha, q, v, theta_min=0.467)` docstring updated:
  q is rolling verified accuracy over last 400 decisions (v10.7 operational
  definition). Formula and default values unchanged.

**KernelSelector architectural reinterpretation (code unchanged, doc + semantics):**
- `KernelRecommendation.method` values: "rule" | "monitoring" (was "empirical" | "rule").
- `recommend()` documented as monitoring output, not selection authority.
- `MIN_DECISIONS_FOR_RECOMMENDATION=100` reinterpreted as monitoring threshold.
- cumulative_analyst_prob instrumentation exposed via KernelScore (lines 38-40, 272-280).

**UNI-DK-01 v5.3 closure (April 19, 2026):**
- 1500-cell characterization of DK advantage surface at mean-σ=0.175.
- All four pre-registered checks PASS (D1-D4). D5 dropped.
- DK ECE finding characterized at scale (0.055→0.42 across NR=1.0→5.0).
- Full data at DRIVE_BASE/uni_dk_01/uni_dk_01_v5_*.json.
- Supersedes Deliverable 1 cumulative-averaging decomposition (retracted).

**Consumer contract tests (NEW v0.7.22-pre):**
- `tests/test_consumer_contracts.py` — 12 tests locking the downstream surface.
- See §12.4 for contract enumeration.

**Test count progression (v0.7.21–v0.7.22-pre):**
```
v0.7.20 (v10.3):         884 tests
+for_soc factory:        529 tests  (+2: factory construction + kwargs override)
+η_override warning:     530 tests  (+1: warning boundaries [0.005, 0.02])
+τ + isfinite guards:    533 tests  (+3: τ range, f isfinite, μ isfinite)
+compute_per_factor:     534 tests  (+1: per-factor N_half consistency)
+γ threshold 0.125:      535 tests  (+1: theta-independence regression)
+worst_case_quality:     536 tests  (+1: conservative η_override estimate)
+method="monitoring":    537 tests  (+1: KernelRecommendation value regression)
+consumer contracts:     884 tests  (+12 in test_consumer_contracts.py
                                       +2 Phase 0 instrumentation
                                       +~345 accumulated since v0.7.20)
```

*Reconciliation note: the jump from 527 to 884 includes tests added across
v0.7.21 (870 at tag) + 12 consumer contracts (882) + 2 Phase 0 instrumentation (884).
Full breakdown in tests/test_consumer_contracts.py:14-190.*

---

### 17.2 v6.0 — COMPLETE ✅ (v0.7.22-pre shipped)

[§17.2 UNCHANGED from v10.6 except: "pyproject.toml version v0.7.22-pre" (was v0.7.17/v0.7.20). CHANGELOG entries for v0.7.21 and v0.7.22-pre added. Open-source prep items all shipped.]

### 17.3 v6.5 — GainScheduler + Fisher Calendar + Embeddings + Override Learning

[§17.3 UNCHANGED from v10.6. Block 9.1–9.5 shipped (CLAIM-66-70). Embeddings, GainScheduler, Fisher calendar, OverrideDetector activation remain as v6.5 scope.]

### 17.4 v7.0 — GraphAttentionBridge (Level 2) + ShrinkageKernel Investigation

[§17.4 UNCHANGED from v10.6.]

### 17.5 v8.0 — Cross-Domain Discovery (Level 3)

[§17.5 UNCHANGED from v10.6.]

---

## 18. Repository Structure [UPDATED v10.7]

```
graph-attention-engine/
├── gae/
│   ├── __init__.py              ✅ Full public API surface (v6.0 + v10.7 additions)
│   ├── profile_scorer.py        ✅ ProfileScorer, ScoringResult, CentroidUpdate
│   │                               v6.0: kernel, mask, eta_override, auto_pause
│   │                               v0.7.21+: for_soc() factory, τ/isfinite guards, η_override warning
│   ├── kernels.py               ✅ ScoringKernel, L2Kernel, DiagonalKernel (28 tests)
│   │                               DiagonalKernel.compute_gradient = W/W.max()*(f-mu) — GAE-GRADIENT-001 fix v0.7.7
│   ├── covariance.py            ✅ CovarianceEstimator, CovarianceSnapshot (23 tests)
│   ├── kernel_selector.py       ✅ KernelSelector, KernelRecommendation, KernelScore (46 tests)
│   │                               v10.7 architecture: rule-based PRIMARY, shadow = MONITORING
│   │                               cumulative_analyst_prob instrumentation (lines 38-40)
│   ├── referral.py              ✅ ReferralEngine, ReferralRule, OverrideDetector (31 tests)
│   ├── enrichment_advisor.py    ✅ Enrichment recommendation engine (Phase 1)
│   ├── calibration.py           ✅ CalibrationProfile + factor_mask + eta_override + conservation + fisher
│   │                               η_neg guard: ValueError on η_neg ≥ 1.0 (v0.7.8)
│   │                               compute_eta_override worst_case_quality parameter (v0.7.22-pre)
│   ├── factors.py               ✅ FactorComputer protocol, assemble_factor_vector
│   ├── learning.py              ✅ LearningState (delegates to ProfileScorer)
│   ├── oracle.py                ✅ OracleProvider, GTAlignedOracle, BernoulliOracle
│   ├── evaluation.py            ✅ run_evaluation, EvaluationReport, compute_ece
│   ├── judgment.py              ✅ compute_judgment, JudgmentResult
│   ├── ablation.py              ✅ run_ablation, AblationReport
│   ├── scoring.py               ⚠️ DEPRECATED — forwards to ProfileScorer (remove v7.0)
│   ├── synthetic.py             ✅ OracleSeparationExperiment, FactorVectorSampler, CanonicalCentroid (v10.4)
│   ├── convergence.py           ✅ OLSMonitor (CLAIM-OLS-01), VarQMonitor (logged only)
│   │                               compute_n_half exact 13.51 (lines 49-64)
│   │                               compute_per_factor_n_half (lines 67-72, v0.7.21)
│   │                               gamma_threshold returns 0.125 (lines 1087-1109, corrected v10.7)
│   │                               centroid_distance_to_canonical, phase2_effective_threshold
│   ├── experiments/             ✅ gae/experiments/ — published research examples (v10.6)
│   │   ├── README.md            ✅ "Research examples — not production code"
│   │   ├── evals/               ✅ exp_c1.py, exp_e2.py, factorial.py
│   │   ├── oracle_separation/   ✅ example_validation.py, convergence_demo.py
│   │   └── domains/             ✅ medical_triage.py, supply_chain.py, financial_approval.py
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
│   ├── test_profile_scorer.py   ✅ L2, DK, mask, eta_override, AMBER pause, kernel integration
│   │                               v0.7.21+: for_soc factory, τ/isfinite guards, η_override warning
│   ├── test_kernels.py          ✅ ScoringKernel protocol, L2, Diagonal, gradient correctness (28 tests)
│   ├── test_covariance.py       ✅ CovarianceEstimator (23 tests)
│   ├── test_kernel_selector.py  ✅ KernelSelector rule + monitoring (46 tests)
│   │                               v10.7: method="rule"|"monitoring" regression tests
│   ├── test_referral.py         ✅ ReferralEngine, OverrideDetector (31 tests)
│   ├── test_calibration.py      ✅ factor_mask, eta_override, conservation, fisher
│   │                               v0.7.22-pre: worst_case_quality parameter
│   ├── test_convergence.py      ✅ OLSMonitor, VarQMonitor, N_half per-factor, γ threshold 0.125
│   ├── test_enrichment_advisor.py ✅ factor ranking, 5 deployment profiles (Phase 1)
│   ├── test_synthetic.py        ✅ Oracle separation fixtures (v10.4)
│   ├── test_consumer_contracts.py ✅ NEW v0.7.22-pre: 12 tests locking downstream surface
│   │                                   ScoringResult fields, for_soc behavior, η_override bounds,
│   │                                   auto-pause outcome, KernelRecommendation.method values
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
│   ├── EXPORTS.md               ✅ Full public API reference
│   └── gae_design_v10_7.md      ✅ THIS DOCUMENT (v10.7, April 19, 2026)
├── .github/
│   ├── workflows/
│   │   ├── pytest.yml           ✅ CI — all 884 tests on push
│   │   └── lint.yml             ✅ CI — ruff + mypy
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.yml       ✅
│       └── new_domain.yml       ✅
├── README.md                    ✅ Rewritten (§15.3 spec, DiagonalKernel, CLAIM-W2 context)
├── CONTRIBUTING.md              ✅
├── CHANGELOG.md                 ✅ v0.1.0 → v0.7.22-pre (v10.7 adds v0.7.21, v0.7.22-pre entries)
├── SECURITY.md                  ✅
├── CODE_OF_CONDUCT.md           ✅
├── pyproject.toml               ✅ v0.7.22-pre
└── LICENSE                      ✅ Apache 2.0
```

---

## Appendix A: Experiment Parameters

*(Unchanged from v8.3 Appendix A. See that version for full table. v10.7 addendum:
UNI-DK-01 v5.3 parameters — NR ∈ {1.0, 1.5, 2.0, 3.0, 5.0}, q̄ ∈ {0.50, 0.65, 0.75, 0.85, 0.95},
30 seeds × 2 kernels × 5 NR × 5 q̄ = 1500 cells, mean-σ=0.175 fixed,
frozen no-learning scorer for cold-start measurement, 400-decision rolling window for asymptotic.)*

## Appendix B: Chart Inventory

*(Unchanged from v8.3 Appendix B. Key chart for README: expC1_comparison_waterfall.png — the 25%→98% waterfall that visualizes the kernel finding. v10.7 addition:
UNI-DK-01 v5.3 decomposition surface chart — cold-start vs learning contribution at each NR level. Canonical source: DRIVE_BASE/uni_dk_01/uni_dk_01_v5_decomposition.json.)*

---

*Graph Attention Engine — Design & Architecture v10.7 | April 19, 2026*
*Phase 0 ✅ Phase 1 ✅ Phase 2 ✅ Phase 3 Priority 1 ✅. 884 tests. v0.7.22-pre.*
*v6.0 COMPLETE + UNI-DK-01 v5.3 closed. KernelSelector architecture settled (Option A rule-based PRIMARY).*
*v10.7 ADDITIONS: KernelSelector architecture inversion — rule-based PRIMARY, data-driven MONITOR only (§9.2, §10.10).*
*v10.7 ADDITIONS: DK calibration finding — ECE 0.055→0.42 across NR=1.0→5.0 (§9.6). Prediction vs estimation channels.*
*v10.7 ADDITIONS: Conservation-law q = rolling verified accuracy over last 400 decisions (§9.8, §10.11).*
*v10.7 ADDITIONS: AMBER auto-pause signal = rolling verified accuracy (§9.7). State-level, not per-decision.*
*v10.7 ADDITIONS: Analyst triage ranking = softmax entropy / confidence gap (§9.9). Inherited by soc-copilot v5.7+.*
*v10.7 ADDITIONS: Fourth γ-theorem proof path — centroid-distance derivation (Grok, April 16, 2026) (§10.14).*
*v10.7 ADDITIONS: ProfileScorer.for_soc() factory, η_override warning, τ/isfinite guards (v0.7.21/v0.7.22-pre).*
*v10.7 ADDITIONS: compute_per_factor_n_half (v0.7.21), compute_eta_override worst_case_quality (v0.7.22-pre).*
*v10.7 ADDITIONS: Consumer contract tests (12 tests, tests/test_consumer_contracts.py, §12.4).*
*v10.7 CORRECTIONS: ScoringResult 5 fields (not 9 as v10.6 documented). γ threshold 0.128→0.125 (6 sites).*
*v10.7 CORRECTIONS: N_half exact 13.51 exposed (rounded 14 preserved for narrative use).*
*v10.7 CORRECTIONS: compute_n_half formula documented (discrete-time, not continuous approximation).*
*v10.7 RETIRED: "Phase 4 empirical lock" language. "~250 decisions" threshold. method="empirical" value.*
*DiagonalKernel validated twice: V-MV-KERNEL-HET +13.2pp peak (390 cells) + UNI-DK-01 v5.3 +7.67pp characterized (1500 cells). Curve-based citation.*
*DK calibration: ECE 0.42 at NR=5.0. Architecturally addressed at every production touchpoint (§9.6 + §9.7 + §9.8 + §9.9).*
*Confidence-based kernel selection rejected: mean_conf always picks L2, mean_ll below chance at NR≥3. Rule-based 100% correct (269 cells) + historical 4/4.*
*Cross-repo coordination: GAE enforces pause at ProfileScorer layer. SOC backend wiring (SOC-Q3) tracked separately.*
*θ_min=0.467 Grok form (η × N_half² / T_max). Conservation-law q = rolling verified accuracy (v10.7 operational definition).*
*γ theorem: γ > 1 ⟺ ε_firm > 0.125. Four proof paths (geometric + dimensional + η₋-trap + centroid-distance).*
*W2 flywheel: CLAIM-W2 +10.13pp (p=0.0002). Fisher information path: CLAIM-64 r=0.9669 (raw_weights, GAE 0.7.20).*
*Flywheel Health Monitor: CLAIM-OLS-01 0% miss, p90≥50d lead time. Var(q) gating: PERMANENT HARD STOP.*
*Convergence calendar: CLAIM-CONV-01 MAE=1.55d. Poisoning resilience: CLAIM-SK-01 + CLAIM-LP-01. Enrichment safety: CLAIM-65.*
*ShrinkageKernel deprioritized to v7.0 (off-diagonal <1pp in both domains tested).*
*Two levels of institutional judgment: ProfileScorer = Level 1 (Decision Intelligence). AgentEvolver = Level 2 (Deployment Intelligence). GAE owns Level 1.*
*Three referral phases: Rules (v6.0) → OverrideDetector (v6.5, ≥50 positives) → Monthly retrain (v7.0).*
*Block 9.1-9.5 CLAIM-66-70 all shipped. CLAIM-67 (η rate cap) UNCONDITIONAL (F=8.14).*
*raw_weights (true 1/σ²) for η_eff. weights (pre-normalized) for scoring. Distinction is a hard architectural rule.*
*ScoringResult has 5 required fields (action_index, action_name, probabilities, distances, confidence) + optional additions (entropy, confidence_gap). Consumer contract tests lock this.*
*"The math is open. The moat is the accumulated graph. The gradient fix makes the flywheel real. The graph compounds while centroids wait. Recovery is not coincidence — γ > 1 is proven. Prediction and estimation are different channels, and we route accordingly. Test every invariant before every release."*

---


## v10.8 Addendum — Framework v4 Modifications

*The following modifications apply to existing sections. Collected here for clean application. Cross-reference with the v10.8 change preamble (Category B + C items).*

### B4: §9.1 Equation — Effective Weights Under TwoPhaseStrategy

Under TwoPhaseStrategy, the effective weights used in scoring are:

    w̃ᵢ = α × w_DK_i + (1 - α) × 1.0

At α=0 (Phase 1 or ContinuousStrategy): w̃ = 1 for all i. Pure centroid scoring. Hyperplane boundaries. At α>0 (Phase 2): shrinkage-optimal interpolation between Order 0 (uniform) and Order 1 (DK discriminative). Quadric boundaries. DiagonalKernel ALREADY accepts per-dimension weights — one-line change to score().

### B5: §9.4 ProfileScorer Interface — New Methods

New __init__ parameter: `learning_strategy: Optional[LearningStrategy] = None` (default: ContinuousStrategy = legacy).

New methods: `get_phase(category_index) → str`, `get_alpha(category_index) → float`, `get_novelty_rate(category_index) → float`, `get_dk_weights(category_index) → Optional[np.ndarray]`.

New factory: `ProfileScorer.for_soc_twophase(mu, actions)` — SOC with TwoPhaseStrategy defaults.

### B6: §9.5 Learning Equation — Phase 2 Batch Learning

Under TwoPhaseStrategy Phase 2, centroid updates are FROZEN for (c,a) pairs that have transitioned. Verified decisions are buffered for batch re-estimation (§9.11). The DKEstimator runs periodically on accumulated data. New weights pass through composition check → estimation → validation → promotion before deployment. Individual decisions do NOT update the deployed scorer directly. The batch is the unit of deployed improvement.

### B7: §10.1 Module Overview — New Modules

```
├── two_phase.py           📋 v6.5 — TwoPhaseStrategy, PhasePolicy,
│                              DKEstimator, ShrinkageSchedule, policies
├── batch_pipeline.py      📋 v6.5 — BatchCompositionPolicy, PromotionGate,
│                              BatchHistory, stats, decisions, state
├── novelty.py             📋 v6.5 — NoveltyTracker, NearestNeighborNovelty,
│                              failure mode detection
├── test_two_phase.py      📋 v6.5 (~100 lines)
├── test_batch_pipeline.py 📋 v6.5 (~50 lines)
├── test_novelty.py        📋 v6.5 (~50 lines)
```

### B8: §13 Equation Traceability — New Rows

| Equation | Source | File | Function | Version | Status |
|---|---|---|---|---|---|
| Eq. SHRINKAGE | framework v4 §2.2 | gae/two_phase.py | compute_w_tilde() | v6.5 | 📋 SPECIFIED |
| Eq. COORD-DESCENT | framework v4 §5.3 | gae/two_phase.py | CoordinateDescentEstimator.estimate() | v6.5 | 📋 SPECIFIED |
| Eq. PROMOTION-GATE | framework v4 §3.6 | gae/batch_pipeline.py | PromotionGate.should_promote() | v6.5 | 📋 SPECIFIED |
| Eq. NOVELTY-DNN | framework v4 §4.3 | gae/novelty.py | NearestNeighborNovelty.compute_novelty() | v6.5 | 📋 SPECIFIED |
| Eq. BATCH-COMPOSITION | framework v4 §3.6 | gae/batch_pipeline.py | BatchCompositionPolicy.should_estimate() | v6.5 | 📋 SPECIFIED |

### B9: §14.2 Design Decisions — New Entries

| Decision | Evidence | Status |
|---|---|---|
| TwoPhaseScorer | Framework v4, 5-judge, ~115 experiments. Phase 1 saturates (K33). Phase 2 persists (K26). Batch pipeline with promotion gate. | Settled v10.8. v6.5. |
| Batch pipeline defense in depth | Framework v4 §3.6. Shrinkage + promotion gate + rollback. | Settled v10.8. |
| DK = discriminative metric learning | K31: direct variance MLE fails (65-76%). Coordinate descent succeeds (80-88%). Weights ≠ inverse variances. | Settled v10.8. |
| Fisher-inspired asymmetry (Layer A/B) | Layer A: Gaussian surrogate motivates. Layer B: ρ_variance ≈ +0.35 at 18/18. Fisher is MOTIVATION, not proof. | Settled v10.8. |
| Profile state terminology | Framework v4 + 5-judge. T₀/T₁ are "profile state," not "sufficient statistics." Model misspecification. | Settled v10.8. |
| Phase 1 label-noise contamination | K33 + Opus/GPT5.5. Centroids diverge at N>200. Freezing limits contamination. Quality-sensitive update is OPEN. | Open. |
| DK concentrates dimensionality | K34: eff_dim ~5.5 → ~4.0. 50× importance ratio. Automatic feature selection. | Measured v10.8. |
| DK robust to T₀ quality | K32: +3.2pp at good, +52pp at random. Phase 1 contributes 0.3pp to DK effectiveness. | Measured v10.8. |
| DK gain increases with noise | K35: +1.5pp at 5%, +8.1pp at 50%. DK and label quality complementary. | Measured v10.8. |
| Shifts hurt DK | K36: -2.7pp at δ=0.25. Need shift detection + reset. ResetPolicy threshold. | Measured v10.8. |
| Full accumulation beats windowing | K37: at δ<0.15 pre-shift helps; at δ>0.20 accumulation hurts. ResetPolicy threshold. | Measured v10.8. |

### B10: §16 Production Constraints — New Entries

| Constraint | Why | Violation consequence |
|---|---|---|
| Batch pipeline: promotion required | Framework v4 §3.6. Defense in depth. | Scoring update without promotion = bug. |
| Phase 1 freeze: per-(c,a) | K33 + framework v4 §3.2. | Simultaneous freeze = premature for rare categories. |
| Factor-version drift: α reduction | Framework v4 §3.4. ResetPolicy. | Stale DK weights on shifted distributions. |
| DK weights: NOT inverse variances | K31 + 5-judge consensus. | Incorrect statistical claims. |
| Coordinate descent: the estimator | K31: direct MLE 65-76%, coord descent 80-88%. | Using direct variance MLE = broken DK. |

### B11: §17 — v6.5 TwoPhaseScorer Implementation Plan

**~550 lines new code + ~200 lines tests. ~2 weeks.**

PHASE A (~1 week): Core scorer extension. PhasePolicy + DecisionCountPolicy (15 lines). DKEstimator + CoordinateDescentEstimator (60 lines). ShrinkageSchedule + FixedAlpha (10 lines). ProfileScorer phase state + buffer + reestimate_dk() (50 lines). score() w_tilde modification (5 lines). Tests (100 lines). Gate: 884+ existing + new pass.

PHASE B (~3 days): Batch pipeline + novelty. NoveltyTracker + NearestNeighborNovelty (25 lines). BatchCompositionPolicy + PromotionGate + BatchHistory (50 lines). Checkpoint serialization (20 lines). Tests (50 lines). Gate: backward compat.

PHASE C (~3 days): SOC integration. Triage re-estimation trigger (20 lines). Tab 3/4 UI: phase indicator, α display (80 lines). Gate: E2E pass.

PHASE D (~1 day): Platform + SDK. ci-platform scheduling (25 lines). copilot-sdk health API (10 lines). s2p config (15 lines). Gate: all repos pass.

Storage: buffer N×(D+2) = 640KB at N=10000. Novelty 346KB. Computation: DK re-estimation ~2-5s at N=4000. Novelty <1ms.

Migration: old checkpoints → phase=PHASE_2, α=0. Rollback: α=0 → pure centroid, instant.

### B13: §10.12 Public API — New Exports (v6.5)

```python
# Two-phase learning
TwoPhaseStrategy, ContinuousStrategy, DecisionCountPolicy,
CoordinateDescentEstimator, FixedAlpha, LinearRampAlpha,

# Batch pipeline
BatchCompositionPolicy, DefaultBatchComposition, PromotionGate,
DefaultPromotionGate, BatchHistory, BatchStats, PromotionDecision, WeightState,

# Novelty
NoveltyTracker, NearestNeighborNovelty,
```

### B14: ResetPolicy — Factor-Version Drift

New trigger: `factor_version_changed`. ON trigger: suppress novelty-triggered re-estimation during revalidation, reduce α to max(α - 0.3, 0), run canary comparison (old vs new factor distributions), resume after canary non-inferiority. Rationale: DK weights estimated on pre-enrichment vectors may not be valid for post-enrichment. Largest system-level risk (5-judge review, framework v4 §3.4).

### A4: §12.5 Batch Pipeline Consumer Contracts

13. ProfileScorer with TwoPhaseStrategy: get_phase() returns 'MEAN_CONVERGENCE' | 'VARIANCE_LEARNING'. get_alpha() returns float [0,1]. get_novelty_rate() returns float [0,1]. get_dk_weights() returns ndarray (A,D) or None.

14. PromotionDecision: promote (bool), reason (str), metrics (Dict) always present. metrics contains 'accuracy_delta', 'worst_category_delta'.

15. BatchHistory: rollback_to(batch_id) returns exact WeightState. Rollback idempotent.

16. Legacy compatibility: ProfileScorer with no LearningStrategy → ContinuousStrategy. IDENTICAL to v10.7. Zero downstream changes.

### A5: §4 Experiment Table — New Row

| Compounding framework | EXP-1 through K38, RATE series, ORDER series, CRITICAL, CENTROID, ROADMAP-RERUN | ~115 | Two-phase validated. Fisher asymmetry 18/18. Shrinkage safety 21/21. DK concentrates dimensionality. DK robust to T₀. Label quality complementary. Shifts hurt at δ>0.20. Framework v4 (post-judge-review). |

**Updated total: ~180 → ~295 experiments.**

### C: Terminology Notes (apply throughout)

**C1:** "Sufficient statistics" → "profile state" in deployed-scorer context. PRESERVE in explicit Gaussian surrogate context only.

**C2:** DK weights: add "discriminative precision weights" qualifier. NOT "inverse variances" in deployed context. (K31: coordinate descent 80-88% vs direct variance MLE 65-76%.)

**C3:** Fisher information: add "Layer A/B" framing. Layer A = Gaussian surrogate motivates. Layer B = empirical ρ_variance ≈ +0.35 confirms. Fisher is MOTIVATION, not proof.

**C4:** "Independent channels" → "mechanistically distinct, statistically coupled channels targeting separable error sources."

---

## Cross-Document Coordination (v10.8)

After v10.8 is applied, the following documents need updates:

- math_synopsis v14 → v15: Layer A/B Fisher, profile state terminology, batch-level claims, error budget
- claims_registry v5 → v6: batch-level compounding claim, shrinkage/concentration/robustness claims, NOT-claims
- architecture_philosophy v4.3 → v4.4: two-phase learning, conservation under DK, defense in depth, profile state
- SOC copilot design v5.7 → v5.8: Tab 3/4 updates, batch pipeline in triage, PromotionGate
- MAP v5.50+: H-05-H-09 framework implementation items

---

*v10.8: TwoPhaseScorer. Batch pipeline. Defense in depth. Profile state. ~115 experiments. 5-judge review.*
*DK = discriminative precision weights (K31). Fisher-inspired (Layer A/B). Coordinate descent > direct MLE.*
*Phase 1 saturates (K33 N≈200). Phase 2 persists (K26 ρ≈+0.35 at 18/18). Shrinkage safe (K29 0/21). DK robust (K32 +52pp at random).*
*~550 lines + ~200 tests. Zero downstream changes. ContinuousStrategy = default.*

