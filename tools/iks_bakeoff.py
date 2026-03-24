"""
tools/iks_bakeoff.py — V-IKS-BAKEOFF simulation.

Evaluates four IKS anchor options under P28 Warmstart across three deployment
profiles. Standalone script — does NOT modify any gae/ files.

Run:
    python tools/iks_bakeoff.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

N_DECISIONS = [0, 14, 50, 100, 200, 500]
ETA = 0.05


def simulate_iks(
    mu0_dist_to_mustar: float,
    anchor_dist_to_mustar: float,
    d_max: float,
    n_decisions_list: list[int] = N_DECISIONS,
    eta: float = ETA,
) -> list[float]:
    """
    Simulate IKS trajectory as centroid converges toward μ*.

    Centroid starts at mu0_dist_to_mustar from μ* and decays geometrically.
    Anchor is fixed at anchor_dist_to_mustar from μ*.
    Both centroid and anchor lie on the same ray from μ*.

    IKS(n) = 100 × min(|current_dist − anchor_dist| / D_MAX, 1.0)
    """
    results = []
    for n in n_decisions_list:
        current_dist_to_mustar = mu0_dist_to_mustar * (1 - eta) ** n
        dist_to_anchor = abs(current_dist_to_mustar - anchor_dist_to_mustar)
        iks = 100.0 * min(dist_to_anchor / d_max, 1.0)
        results.append(round(iks, 1))
    return results


# ---------------------------------------------------------------------------
# Deployment profiles
# ---------------------------------------------------------------------------

PROFILES = {
    "profile1": {
        "label": "Standard (no Warmstart)",
        "mu0_dist": 0.20,         # standard prior distance to μ*
        "mu0_standard_dist": 0.20,
    },
    "profile2": {
        "label": "Enriched Warmstart (22% closer)",
        "mu0_dist": 0.156,        # enriched μ₀ distance to μ*
        "mu0_standard_dist": 0.20,
    },
    "profile3": {
        "label": "Greenfield enriched (35% closer)",
        "mu0_dist": 0.130,        # enriched μ₀ distance to μ*
        "mu0_standard_dist": 0.20,
    },
}

# E_score for Option C (fixed enrichment score, never changes after deployment)
# E_score = 100 × (1 - sigma_mean_post / sigma_mean_pre)
SIGMA_PRE = 0.19
SIGMA_POST = 0.10
E_SCORE = round(100.0 * (1.0 - SIGMA_POST / SIGMA_PRE), 1)  # = 47.4


# ---------------------------------------------------------------------------
# Option A — Dual-anchor
# ---------------------------------------------------------------------------
# anchor = standard expert prior (un-enriched μ₀, always 0.20 from μ*)
# mu_live = enriched μ₀ (22% closer to μ*)
# D_MAX = 0.20 (unchanged)
#
# For Profile 2:
#   IKS(0) = 100 × |0.156 - 0.20| / 0.20 = 100 × 0.044 / 0.20 = 22.0

def trajectories_option_a() -> dict:
    D_MAX = 0.20
    ANCHOR_DIST = 0.20  # standard prior always 0.20 from μ*
    trajs = {}
    for pname, p in PROFILES.items():
        iks = simulate_iks(
            mu0_dist_to_mustar=p["mu0_dist"],
            anchor_dist_to_mustar=ANCHOR_DIST,
            d_max=D_MAX,
        )
        trajs[pname] = {"decisions": N_DECISIONS, "iks": iks}
    return trajs


# ---------------------------------------------------------------------------
# Option B — Single enriched anchor + recalibrated D_MAX
# ---------------------------------------------------------------------------
# anchor = enriched μ₀ (per deployment)
# D_MAX = 0.156 (= 0.20 × 0.78, calibrated for 22%-closer case)
# IKS(0) = 0 for all profiles ✓
# IKS(convergence) = 100 for profiles where anchor_dist ≤ D_MAX; may cap at
#                    < 100 for profile3 (anchor_dist = 0.130 < 0.156 → 83.3)

def trajectories_option_b() -> dict:
    D_MAX = 0.156
    trajs = {}
    for pname, p in PROFILES.items():
        anchor_dist = p["mu0_dist"]  # enriched μ₀ is the anchor
        iks = simulate_iks(
            mu0_dist_to_mustar=p["mu0_dist"],
            anchor_dist_to_mustar=anchor_dist,
            d_max=D_MAX,
        )
        trajs[pname] = {"decisions": N_DECISIONS, "iks": iks}
    return trajs


# ---------------------------------------------------------------------------
# Option C — Separate scores (IKS + static Enrichment Score)
# ---------------------------------------------------------------------------
# IKS: same as Option B (anchor = enriched μ₀, D_MAX = 0.156)
# E_score: fixed at deployment, never changes
#   E_score = 100 × (1 - sigma_mean_post / sigma_mean_pre) = 47.4

def trajectories_option_c() -> dict:
    # IKS trajectories identical to Option B
    iks_trajs = trajectories_option_b()
    # Augment with fixed E_score column
    trajs = {}
    for pname, t in iks_trajs.items():
        trajs[pname] = {
            "decisions": t["decisions"],
            "iks": t["iks"],
            "e_score": [E_SCORE] * len(N_DECISIONS),
        }
    return trajs


# ---------------------------------------------------------------------------
# Option D — Deployment-normalized D_MAX
# ---------------------------------------------------------------------------
# anchor = enriched μ₀
# D_MAX = ‖μ* − enriched_μ₀‖ estimated from P28 (= mu0_dist per deployment)
# IKS(0) = 0 ✓, IKS(convergence) = 100 ✓ by construction
# Cross-deployment IKS comparisons NOT directly meaningful (D_MAX varies)

def trajectories_option_d() -> dict:
    trajs = {}
    for pname, p in PROFILES.items():
        anchor_dist = p["mu0_dist"]
        d_max = p["mu0_dist"]  # deployment-normalized
        iks = simulate_iks(
            mu0_dist_to_mustar=p["mu0_dist"],
            anchor_dist_to_mustar=anchor_dist,
            d_max=d_max,
        )
        trajs[pname] = {"decisions": N_DECISIONS, "iks": iks}
    return trajs


# ---------------------------------------------------------------------------
# Criterion evaluation
# ---------------------------------------------------------------------------

def evaluate_c1(trajs: dict, option: str) -> str:
    """
    C1 — IKS semantics: IKS(0)=0 AND IKS(convergence)≈100 for all profiles?
    PASS: both met for all profiles
    PARTIAL: at least one profile or one condition fails
    FAIL: neither met
    """
    all_zero_start = True
    all_full_converge = True

    for pname, t in trajs.items():
        iks_list = t["iks"]
        if iks_list[0] != 0.0:
            all_zero_start = False
        # convergence = IKS at n=500 (last entry)
        if iks_list[-1] < 99.0:
            all_full_converge = False

    if all_zero_start and all_full_converge:
        return "PASS"
    elif not all_zero_start and not all_full_converge:
        return "FAIL"
    else:
        return "PARTIAL"


def evaluate_c2(option: str) -> str:
    """
    C2 — Cross-deployment comparability: same IKS value = same meaning?
    COMPARABLE / PARTIAL / NOT COMPARABLE
    """
    return {
        "A": "COMPARABLE",     # same D_MAX=0.20, same anchor_dist=0.20 always
        "B": "PARTIAL",        # fixed D_MAX=0.156 but IKS max differs by profile
        "C": "PARTIAL",        # IKS same as B; E_score adds deployment-specific info
        "D": "NOT COMPARABLE", # D_MAX varies per deployment
    }[option]


def evaluate_c3(option: str) -> str:
    """
    C3 — Enrichment benefit visibility: is 22-35% head start visible to CISO?
    VISIBLE / PARTIAL / INVISIBLE
    """
    return {
        "A": "VISIBLE",   # IKS(0)=22 for P2, IKS(0)=35 for P3 — head start explicit
        "B": "INVISIBLE", # IKS always starts at 0; head start absorbed into D_MAX
        "C": "VISIBLE",   # E_score=47.4 explicitly communicates enrichment benefit
        "D": "INVISIBLE", # IKS always starts at 0 and ends at 100 — head start hidden
    }[option]


def evaluate_c4(option: str) -> str:
    """
    C4 — Implementation complexity.
    LOW: <20 lines / MEDIUM: 20-50 / HIGH: >50 or structural changes
    """
    return {
        "A": "LOW",    # store standard prior as anchor; single iks.py update
        "B": "LOW",    # store enriched μ₀ + recalibrated D_MAX; single iks.py update
        "C": "MEDIUM", # two metrics; need σ logging pipeline for E_score
        "D": "MEDIUM", # D_MAX estimation from P28 data per deployment; extra step
    }[option]


def evaluate_c5(option: str) -> tuple[str, str]:
    """
    C5 — CISO one-sentence explanation.
    Returns (CLEAR/AMBIGUOUS/CONFUSING, sentence)
    Option C has two sub-scores; use 'C_IKS' or 'C_Escore' for sub-sentences.
    For option 'C', returns the combined clarity score and both sentences joined.
    """
    sentences = {
        "A": (
            "CLEAR",
            "Your adaptation score starts at 22 because the enrichment data already "
            "closed 22% of the learning gap before your first alert was processed.",
        ),
        "B": (
            "AMBIGUOUS",
            "Your adaptation score measures how far your model has learned since "
            "deployment, normalized to the enriched starting point.",
        ),
        "C": (
            "CLEAR",
            "Your adaptation score measures learning progress since deployment; "
            "your enrichment score (fixed at 47) reflects the head-start gained "
            "from historical data loaded before go-live.",
        ),
        "C_IKS": (
            "CLEAR",
            "Your adaptation score measures how completely your model has learned "
            "your environment since deployment began.",
        ),
        "C_Escore": (
            "CLEAR",
            "Your enrichment score of 47 reflects how much tighter your baseline "
            "predictions became by loading historical data before go-live.",
        ),
        "D": (
            "CLEAR",
            "Your adaptation score measures the fraction of your full learning "
            "journey completed, normalized to your specific deployment conditions.",
        ),
    }
    return sentences[option]


# ---------------------------------------------------------------------------
# Architectural constraint checks (analysis only — no code modification)
# ---------------------------------------------------------------------------

ARCH_CONSTRAINTS = {
    "no_sigma_in_update": {
        "A": True,   # only μ distance, no σ
        "B": True,
        "C": True,   # E_score computed separately, never passed to update()
        "D": True,
    },
    "anchor_freeze_rule": {
        "A": True,   # standard prior written once to iks_bootstrap_soc.json
        "B": True,   # enriched μ₀ written once
        "C": True,
        "D": True,   # enriched μ₀ + D_MAX written once per deployment
    },
    "no_type_branching": {
        "A": True,   # standard prior is always the anchor, same code path
        "B": True,   # enriched = standard for non-enriched; no branching needed
        "C": True,
        "D": True,   # D_MAX estimated from P28 data uniformly
    },
}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_bakeoff() -> dict:
    traj_a = trajectories_option_a()
    traj_b = trajectories_option_b()
    traj_c = trajectories_option_c()
    traj_d = trajectories_option_d()

    iks_at_zero_p2_a = traj_a["profile2"]["iks"][0]  # should be 22.0

    options_data = {}

    for opt, trajs in [("A", traj_a), ("B", traj_b), ("C", traj_c), ("D", traj_d)]:
        c1 = evaluate_c1(trajs, opt)
        c2 = evaluate_c2(opt)
        c3 = evaluate_c3(opt)
        c4 = evaluate_c4(opt)
        c5_score, c5_sentence = evaluate_c5(opt)

        options_data[opt] = {
            "iks_at_zero_profile2": trajs["profile2"]["iks"][0],
            "trajectories": {
                pname: {
                    "decisions": t["decisions"],
                    "iks": t["iks"],
                    **({"e_score": t["e_score"]} if "e_score" in t else {}),
                }
                for pname, t in trajs.items()
            },
            "criteria": {
                "C1": c1,
                "C2": c2,
                "C3": c3,
                "C4": c4,
                "C5_score": c5_score,
                "C5_sentence": c5_sentence,
            },
        }

    # Recommendation: Option A
    # C1 PARTIAL is informative (IKS(0)=22 = head-start magnitude),
    # not a defect. Only option with COMPARABLE (C2) + VISIBLE (C3) + LOW (C4).
    # All architectural constraints satisfied.
    recommendation = "A"
    justification = (
        "Option A (Dual-anchor) is recommended. C1 is PARTIAL because IKS(0)=22 for "
        "enriched deployments — but this is a FEATURE: the CISO immediately sees the "
        "percentage of the adaptation journey already closed by enrichment. All profiles "
        "converge to IKS=100 (C1 convergence condition met). Option A is the only "
        "option achieving COMPARABLE cross-deployment semantics (C2) because D_MAX=0.20 "
        "and the anchor are identical for all deployments, making IKS=67 mean the same "
        "thing regardless of whether enrichment was used. Enrichment benefit is VISIBLE "
        "(C3): IKS(0)=22 for P28 Warmstart and IKS(0)=35 for greenfield directly encode "
        "the head-start. Complexity is LOW (C4): a single frozen standard-prior anchor "
        "in iks_bootstrap_soc.json, no estimation step. The CISO sentence is CLEAR (C5). "
        "Options B and C cap at IKS=83 for greenfield (C1 PARTIAL, wrong direction). "
        "Option D passes C1 but sacrifices cross-deployment comparability (C2 NOT "
        "COMPARABLE) and hides enrichment benefit entirely (C3 INVISIBLE)."
    )

    result = {
        "options": options_data,
        "recommendation": recommendation,
        "justification": justification,
        "architectural_constraints": ARCH_CONSTRAINTS,
        "e_score_fixed": E_SCORE,
        "e_score_sigma_pre": SIGMA_PRE,
        "e_score_sigma_post": SIGMA_POST,
    }
    return result


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_results(result: dict) -> None:
    opts = result["options"]
    n_dec = N_DECISIONS

    print("V-IKS-BAKEOFF Results:")
    print()

    # Criterion scores table
    print("Criterion scores:")
    hdr = f"{'Option':<8} | {'C1(semantics)':<15} | {'C2(compare)':<18} | {'C3(visible)':<10} | {'C4(complex)':<12} | {'C5(clarity)':<12}"
    print(hdr)
    print("-" * len(hdr))
    for opt in ["A", "B", "C", "D"]:
        cr = opts[opt]["criteria"]
        print(
            f"{opt:<8} | {cr['C1']:<15} | {cr['C2']:<18} | {cr['C3']:<10} | {cr['C4']:<12} | {cr['C5_score']:<12}"
        )

    print()

    # Profile 2 trajectory table
    print("IKS trajectories — Profile 2 (enriched Warmstart, 22% closer):")
    e_fixed = result["e_score_fixed"]
    hdr2 = f"{'Dec':<5} | {'Opt-A':>7} | {'Opt-B':>7} | {'Opt-C(IKS)':>10} | {'Opt-C(E_score)':>14} | {'Opt-D':>7}"
    print(hdr2)
    print("-" * len(hdr2))
    for i, n in enumerate(n_dec):
        a_iks = opts["A"]["trajectories"]["profile2"]["iks"][i]
        b_iks = opts["B"]["trajectories"]["profile2"]["iks"][i]
        c_iks = opts["C"]["trajectories"]["profile2"]["iks"][i]
        d_iks = opts["D"]["trajectories"]["profile2"]["iks"][i]
        print(
            f"{n:<5} | {a_iks:>7.1f} | {b_iks:>7.1f} | {c_iks:>10.1f} | {e_fixed:>14.1f} | {d_iks:>7.1f}"
        )

    print()

    # Profile 3 trajectory table
    print("IKS trajectories — Profile 3 (greenfield enriched, 35% closer):")
    print(hdr2)
    print("-" * len(hdr2))
    for i, n in enumerate(n_dec):
        a_iks = opts["A"]["trajectories"]["profile3"]["iks"][i]
        b_iks = opts["B"]["trajectories"]["profile3"]["iks"][i]
        c_iks = opts["C"]["trajectories"]["profile3"]["iks"][i]
        d_iks = opts["D"]["trajectories"]["profile3"]["iks"][i]
        print(
            f"{n:<5} | {a_iks:>7.1f} | {b_iks:>7.1f} | {c_iks:>10.1f} | {e_fixed:>14.1f} | {d_iks:>7.1f}"
        )

    print()

    # CISO sentences
    print("CISO-facing sentences:")
    c5_a_score, c5_a = evaluate_c5("A")
    c5_b_score, c5_b = evaluate_c5("B")
    c5_c_iks_score, c5_c_iks = evaluate_c5("C_IKS")
    c5_c_e_score, c5_c_e = evaluate_c5("C_Escore")
    c5_d_score, c5_d = evaluate_c5("D")
    print(f"Option A:       {c5_a}")
    print(f"Option B:       {c5_b}")
    print(f"Option C IKS:   {c5_c_iks}")
    print(f"Option C E_score: {c5_c_e}")
    print(f"Option D:       {c5_d}")

    print()

    print(f"Recommended option: {result['recommendation']}")
    print(f"Justification: {result['justification']}")

    print()

    print("Architectural constraints satisfied:")
    ac = result["architectural_constraints"]
    for constraint, per_option in ac.items():
        vals = "  ".join(f"{o}:{v}" for o, v in per_option.items())
        print(f"  {constraint:<25} {vals}")

    print()
    print("Raw numbers for roadmap session review.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_bakeoff()

    # Save JSON
    out_path = Path(__file__).parent / "iks_bakeoff_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print_results(result)
    print()
    print(f"Results saved to {out_path}")
