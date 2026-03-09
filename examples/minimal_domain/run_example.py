"""
GAE minimal end-to-end example — IT helpdesk ticket classifier.

Demonstrates how to onboard a new domain in 5 steps:
  1. Define domain config
  2. Build ProfileScorer with domain centroids
  3. Run evaluation on scenarios
  4. Compute judgment for one scenario
  5. Run ablation study

Run with:
  python examples/minimal_domain/run_example.py
"""

import sys
import numpy as np

from gae import (
    ProfileScorer,
    build_profile_scorer,
    EvaluationScenario,
    run_evaluation,
    compute_judgment,
    run_ablation,
)

# ─────────────────────────────────────────────────────────────────────
# STEP 1 — Define domain config
# ─────────────────────────────────────────────────────────────────────
print("=== STEP 1: Define domain config ===")

DOMAIN      = "helpdesk"
CATEGORIES  = ["hardware", "software", "network"]
ACTIONS     = ["auto_resolve", "assign_tier1", "assign_tier2", "escalate"]
FACTOR_NAMES = ["urgency", "recurrence", "affected_users", "asset_value"]

N_CAT = len(CATEGORIES)   # 3
N_ACT = len(ACTIONS)       # 4
N_FAC = len(FACTOR_NAMES)  # 4

cat_idx = {name: i for i, name in enumerate(CATEGORIES)}
act_idx = {name: i for i, name in enumerate(ACTIONS)}

print(f"  Domain:      {DOMAIN}")
print(f"  Categories:  {CATEGORIES}")
print(f"  Actions:     {ACTIONS}")
print(f"  Factors:     {FACTOR_NAMES}")

# ─────────────────────────────────────────────────────────────────────
# STEP 2 — Build ProfileScorer with domain centroids
# ─────────────────────────────────────────────────────────────────────
print("\n=== STEP 2: Build ProfileScorer ===")

# Start with neutral default (0.3) for all centroids
mu = np.full((N_CAT, N_ACT, N_FAC), 0.3, dtype=np.float64)

# hardware centroids
mu[cat_idx["hardware"], act_idx["escalate"],     :] = [0.9, 0.8, 0.7, 0.9]
mu[cat_idx["hardware"], act_idx["auto_resolve"],  :] = [0.1, 0.1, 0.1, 0.2]
mu[cat_idx["hardware"], act_idx["assign_tier1"],  :] = [0.3, 0.4, 0.2, 0.3]
mu[cat_idx["hardware"], act_idx["assign_tier2"],  :] = [0.5, 0.5, 0.4, 0.5]

# software centroids
mu[cat_idx["software"], act_idx["assign_tier1"],  :] = [0.5, 0.4, 0.3, 0.4]
mu[cat_idx["software"], act_idx["auto_resolve"],  :] = [0.2, 0.1, 0.1, 0.1]
mu[cat_idx["software"], act_idx["assign_tier2"],  :] = [0.6, 0.5, 0.4, 0.5]
mu[cat_idx["software"], act_idx["escalate"],      :] = [0.8, 0.7, 0.6, 0.8]

# network centroids
mu[cat_idx["network"],  act_idx["escalate"],      :] = [0.8, 0.7, 0.9, 0.6]
mu[cat_idx["network"],  act_idx["auto_resolve"],  :] = [0.1, 0.2, 0.1, 0.1]
mu[cat_idx["network"],  act_idx["assign_tier1"],  :] = [0.4, 0.4, 0.5, 0.3]
mu[cat_idx["network"],  act_idx["assign_tier2"],  :] = [0.5, 0.5, 0.6, 0.4]

scorer = ProfileScorer(mu=mu, actions=ACTIONS)
print(f"  ProfileScorer built: shape {mu.shape}  "
      f"(categories={N_CAT}, actions={N_ACT}, factors={N_FAC})")

# ─────────────────────────────────────────────────────────────────────
# STEP 3 — Run evaluation on 6 scenarios
# ─────────────────────────────────────────────────────────────────────
print("\n=== STEP 3: Run evaluation ===")

scenarios = [
    # hardware — high urgency, recurrence, asset_value → escalate
    EvaluationScenario(
        scenario_id="H1", domain=DOMAIN, category="hardware",
        category_index=cat_idx["hardware"],
        factors=np.array([0.9, 0.8, 0.7, 0.9]),
        expected_action="escalate",
        expected_action_index=act_idx["escalate"],
        description="Critical hardware failure — server down",
    ),
    # hardware — low signal → auto_resolve
    EvaluationScenario(
        scenario_id="H2", domain=DOMAIN, category="hardware",
        category_index=cat_idx["hardware"],
        factors=np.array([0.1, 0.1, 0.1, 0.2]),
        expected_action="auto_resolve",
        expected_action_index=act_idx["auto_resolve"],
        description="Printer paper jam — self-service",
    ),
    # software — medium urgency → assign_tier1
    EvaluationScenario(
        scenario_id="S1", domain=DOMAIN, category="software",
        category_index=cat_idx["software"],
        factors=np.array([0.5, 0.4, 0.3, 0.4]),
        expected_action="assign_tier1",
        expected_action_index=act_idx["assign_tier1"],
        description="Application crash — single user",
    ),
    # software — low signal → auto_resolve
    EvaluationScenario(
        scenario_id="S2", domain=DOMAIN, category="software",
        category_index=cat_idx["software"],
        factors=np.array([0.2, 0.1, 0.1, 0.1]),
        expected_action="auto_resolve",
        expected_action_index=act_idx["auto_resolve"],
        description="Password reset request",
    ),
    # network — high urgency, many affected → escalate
    EvaluationScenario(
        scenario_id="N1", domain=DOMAIN, category="network",
        category_index=cat_idx["network"],
        factors=np.array([0.8, 0.7, 0.9, 0.6]),
        expected_action="escalate",
        expected_action_index=act_idx["escalate"],
        description="Network outage — entire floor affected",
    ),
    # network — low signal → auto_resolve
    EvaluationScenario(
        scenario_id="N2", domain=DOMAIN, category="network",
        category_index=cat_idx["network"],
        factors=np.array([0.1, 0.2, 0.1, 0.1]),
        expected_action="auto_resolve",
        expected_action_index=act_idx["auto_resolve"],
        description="Slow WiFi — single device",
    ),
]

report = run_evaluation(scorer, scenarios)

print(f"  Scenarios:        {report.n_scenarios}")
print(f"  Correct:          {report.n_correct}")
print(f"  Overall accuracy: {report.accuracy:.1%}")
print(f"  ECE:              {report.ece:.4f}")
print("  Per-category accuracy:")
for cat, acc in report.by_category.items():
    print(f"    {cat:10s}: {acc:.1%}")
print("  Per-scenario results:")
for r in report.scenario_results:
    status = "OK" if r["correct"] else "XX"
    print(f"    [{status}] {r['scenario_id']}: predicted={r['predicted_action']!r:14s} "
          f"expected={r['expected_action']!r:14s} conf={r['confidence']:.3f}")

# ─────────────────────────────────────────────────────────────────────
# STEP 4 — Compute judgment for first scenario
# ─────────────────────────────────────────────────────────────────────
print("\n=== STEP 4: Compute judgment (scenario H1) ===")

first_scenario = scenarios[0]
first_result   = scorer.score(first_scenario.factors, first_scenario.category_index)

judgment = compute_judgment(
    scoring_result=first_result,
    f=first_scenario.factors,
    mu=mu,
    category_index=first_scenario.category_index,
    factor_names=FACTOR_NAMES,
    actions=ACTIONS,
)

print(f"  Scenario:         {first_scenario.scenario_id} - {first_scenario.description}")
print(f"  Recommended:      {judgment.action}")
print(f"  Confidence:       {judgment.confidence:.3f} ({judgment.confidence_tier})")
print(f"  Auto-approvable:  {judgment.auto_approvable}")
print(f"  Dominant factors: {judgment.dominant_factors}")
print(f"  Rationale:        {judgment.rationale}")
print("  All action scores:")
for action, score in judgment.action_scores.items():
    print(f"    {action:16s}: {score:.4f}")

# ─────────────────────────────────────────────────────────────────────
# STEP 5 — Run ablation study
# ─────────────────────────────────────────────────────────────────────
print("\n=== STEP 5: Ablation study ===")

ablation = run_ablation(scorer, scenarios, FACTOR_NAMES)

print(f"  Baseline accuracy: {ablation.baseline_accuracy:.1%}")
print(f"  Most important factor:  {ablation.most_important}")
print(f"  Least important factor: {ablation.least_important}")
print("  Factor importance ranking (most -> least):")
for r in ablation.results:
    bar = "#" * max(0, int(r.accuracy_drop * 20))
    print(f"    #{r.importance_rank} {r.factor_name:20s} "
          f"drop={r.accuracy_drop:+.4f}  ablated={r.ablated_accuracy:.1%}  {bar}")

print("\n=== Done ===")
sys.exit(0)
