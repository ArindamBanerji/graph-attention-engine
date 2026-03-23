# Graph Attention Engine — Project Structure
**Last Updated:** March 23, 2026
**Version:** v0.7.0 (branch: v5.0-dev)
**PyPI:** `pip install graph-attention-engine`
**License:** Apache 2.0
**Tests:** 478 passing
**Purpose:** Domain-agnostic compounding intelligence library. ProfileScorer + kernel-based distance learning + referral routing. Used by SOC Copilot and any other copilot built on the ci-platform framework.

---

## Architecture in One Sentence

A ProfileScorer learns institutional judgment by pulling centroid vectors toward verified analyst decisions using a kernel-weighted distance metric that adapts to per-factor noise.

---

## Critical Design Decisions (read before changing anything)

| Decision | Value | Why |
|---|---|---|
| A=4 actions (SOC) | escalate, investigate, suppress, monitor | A=5 caused 13pp accuracy drop (EXP-A4-DIAGONAL). `refer_to_analyst` is NOT a scorable action — it is a referral routing decision (ReferralEngine). |
| DiagonalKernel default | noise_ratio > 1.5 → diagonal | +13.2pp SOC, +6.8pp S2P on heterogeneous noise. Off-diagonal correlations add <1pp (validated, ShrinkageKernel deprioritized). |
| η_override = 0.01 | asymmetric learning rate | η_confirm=0.05 (clean signal), η_override=0.01 (noisy signal). Prevents 13-27pp centroid degradation from low-quality analyst overrides (P0 fix, validated 24 personas). |
| θ_min = 0.467 | conservation floor | T_max=21 days, η=0.05, N_half=14. All harnesses must use this value. |
| Referral is VETO | independent of scoring | ReferralEngine fires on every alert AFTER ProfileScorer. Any rule firing overrides auto-approve regardless of confidence. NEVER put refer_to_analyst in the centroid tensor. |
| μ ∈ [0,1]^d | centroid clipping | np.clip after every update. Invariant must hold on every update path. |
| No σ in update() | Loop 2 firewall | σ (synthesis layer) NEVER flows into ProfileScorer.update(). Frobenius norm 0.0028 — permanent constraint. |
| LEARNING_ENABLED=False | default off | Enable per-customer after shadow qualification (P28 pipeline). |

---

## Directory Tree

```
graph-attention-engine-v50/
├── gae/
│   ├── __init__.py
│   ├── profile_scorer.py       # ProfileScorer — core scoring + learning engine
│   ├── kernels.py              # L2Kernel, DiagonalKernel, CovarianceEstimator
│   ├── kernel_selector.py      # KernelSelector — rolling 100-window, Phase 2/3/4
│   ├── referral.py             # ReferralEngine, ReferralRule protocol, OverrideDetector stub
│   ├── calibration.py          # CalibrationProfile, compute_eta_override, derive_theta_min
│   ├── covariance.py           # CovarianceEstimator (collects only — NOT for scoring)
│   ├── evaluation.py           # run_evaluation, run_ablation, EvaluationReport
│   ├── factors.py              # FactorComputer protocol, assemble_factor_vector
│   ├── judgment.py             # compute_judgment, JudgmentResult
│   ├── learning.py             # LearningState, WeightUpdate, PendingValidation
│   ├── scoring.py              # score_alert, ScoringResult (backward compat)
│   ├── contracts.py            # SchemaContract, EmbeddingContract
│   ├── schema.py               # DomainSchemaSpec
│   └── fisher.py               # estimate_fisher_information, predict_n_half
├── tests/
│   ├── test_profile_scorer.py
│   ├── test_kernels.py
│   ├── test_kernel_selector.py
│   ├── test_referral.py        # 31 tests — ReferralEngine, OverrideDetector stub
│   ├── test_calibration.py
│   ├── test_covariance.py
│   ├── test_evaluation.py
│   ├── test_ablation.py
│   ├── test_learning.py
│   └── test_scoring.py
├── docs/
│   └── gae_design_v10.md       # Authoritative design document (v10.1)
├── examples/
│   └── minimal_domain/         # Helpdesk domain, runnable standalone
├── pyproject.toml              # v0.7.0
└── LICENSE                     # Apache 2.0
```

---

## Key Modules

### profile_scorer.py
`ProfileScorer` — the core engine.
- `score(f, c)` → `ScoringResult` — computes P(a|f,c) = softmax(−d(f,μ[c,a,:])/τ) using active kernel
- `update(f, c, a, correct, is_override)` — pull/push centroid update with asymmetric η
- `set_kernel(kernel)` — inject DiagonalKernel after P28 Phase 4 locks it
- `set_conservation_status(status)` — AMBER/RED → freezes learning (AMBER auto-pause)
- `checkpoint()` / `rollback(snapshot)` — state management (TD-033)
- Shape: `(C, A, d)` — C categories, A actions, d factors. SOC: (6,4,6)=144. S2P: (5,5,8)=200.
- **DO NOT** iterate `range(len(ACTIONS_LIST))` over counts — use `scorer.counts.shape[1]`

### kernels.py
- `L2Kernel` — cold start default (before P28 measures per-factor σ)
- `DiagonalKernel(weights)` — W=diag(1/σ²). Default when noise_ratio > 1.5.
- `CovarianceEstimator` — **collects** full covariance data, does NOT score. Research asset for v7.0 shrinkage.
- **ShrinkageKernel**: deprioritized to v7.0 — off-diagonal adds <1pp in both domains

### kernel_selector.py
`KernelSelector` — empirically selects kernel during shadow mode.
- Phase 2: `noise_ratio > 1.5 → diagonal, else l2` (rule-based, fires at P28 Phase 2)
- Phase 3: scores every alert with both kernels, rolling 100-decision window
- Phase 4: `lock()` at 250 verified decisions — switches scorer to winning kernel
- Validated: 4/4 correct on known deployment personas

### referral.py
`ReferralEngine` — evaluates all rules, ANY firing → `should_refer=True`.
- `ReferralRule` protocol — `rule_id`, `reason`, `evaluate(context) → (bool, dict)`
- `ReferralDecision` dataclass — `should_refer`, `reasons`, `rule_details`, `audit_summary`
- `OverrideDetector` — **stub only** (v6.5). Raises `NotImplementedError` on `predict()`.
- **P-REF-1:** ReferralEngine NEVER modifies ProfileScorer scoring or centroids.
- **P-REF-2:** Missing context → rule doesn't fire (safe degradation, returns False).
- Activation: v6.5, data-gated at ≥50 production override positives.

### calibration.py
- `CalibrationProfile` — τ, η_confirm, η_override, learning_enabled, auto_pause_on_amber
- `compute_eta_override(q_bar, sigma_q)` — per-deployment η tuning
- `derive_theta_min(eta, n_half, t_max)` → 0.467 canonical
- `check_conservation(alpha, q, V, theta_min)` → GREEN/AMBER/RED
- `compute_breach_window(...)`, `compute_transfer_prior(...)`, `check_meta_conservation(...)`

---

## Noise Ceiling (kernel-dependent)

| Kernel | GREEN | AMBER | RED |
|---|---|---|---|
| L2 | σ_mean ≤ 0.105 | ≤ 0.157 | > 0.157 |
| DiagonalKernel | σ_mean ≤ 0.157 | ≤ 0.25 | > 0.25 |

Noise ceiling is three-variable: σ × V × q̄ (V-B3 finding). The table above is σ_mean under equal V and q̄.

---

## Imports Used by SOC Copilot

```python
from gae.profile_scorer import ProfileScorer, build_profile_scorer
from gae.kernels import L2Kernel, DiagonalKernel, CovarianceEstimator
from gae.kernel_selector import KernelSelector, KernelRecommendation
from gae.referral import ReferralEngine, ReferralRule, ReferralDecision, OverrideDetector
from gae.calibration import CalibrationProfile, compute_eta_override, derive_theta_min
from gae.evaluation import run_evaluation, run_ablation, EvaluationReport
from gae.factors import FactorComputer, assemble_factor_vector
from gae.judgment import compute_judgment, JudgmentResult
from gae.learning import LearningState, WeightUpdate
```

---

## What NOT to Do

- **Never** put `refer_to_analyst` in the centroid tensor (A=4, not A=5)
- **Never** pass σ values into `ProfileScorer.update()` — Loop 2 firewall
- **Never** use `ShrinkageKernel` for scoring — deprioritized, off-diagonal <1pp
- **Never** use `len(ACTIONS_LIST)` to index `scorer.counts` — use `scorer.counts.shape[1]`
- **Never** set `η_neg = 1.0` — FORBIDDEN (ECE=0.49). Canonical: η_neg = 0.05
- **Never** enable learning before P28 shadow qualification
- **Never** clip centroids to anything other than [0.0, 1.0]

---

## Test Count Progression

```
v5.0 post-tag:   246
+Block 5A:       291
+A=4 migration:  309
+Kernels:        383
+KernelSelector: 447
+ReferralEngine: 478  ← current
```

---

*graph-attention-engine-v50 · v0.7.0 · Apache 2.0 · 478 tests · March 23, 2026*
