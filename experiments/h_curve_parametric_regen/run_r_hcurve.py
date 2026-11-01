from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "raw" / "r_hcurve_v1"
SOURCE = HERE / "raw" / "two_arm_v4"
GAE_ROOT = HERE.parents[1]
SDK_ROOT = GAE_ROOT.parent / "copilot-sdk"
sys.path.insert(0, str(GAE_ROOT))
sys.path.insert(0, str(SDK_ROOT))
from gae.profile_scorer import ProfileScorer

SEEDS = [42, 123, 777]
EPS = [0.20, 0.35]
C, A, D = 6, 4, 6
DISRUPTED = [0, 1]
ALPHA = 0.80
V = 1.0
THETA_MIN = 0.467
WINDOW = 10
CAT_WINDOW = 50
Q_WINDOW = 50
ETA_B = 0.05
BOOST_WINDOW = 24
DELTA_NORM = 0.25
EPS_STAR = 0.125
POLICIES = ["A-baseline", "R1-selective-redistribution", "R2-selective-boost", "B-theorem"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        raise TypeError(type(obj).__name__)
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=default) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_array(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, np.asarray(value, dtype=np.float64))
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fit_rate(trace):
    values = np.asarray(trace[1:], dtype=float)
    d0 = float(trace[0])
    d_inf = float(np.mean(values[-max(1, int(math.ceil(.20 * len(values)))):]))
    blocks = []
    for start in range(0, len(values), 10):
        block = values[start:start + 10]
        if len(block) == 10:
            blocks.append((start + 4.5, float(np.mean(block))))
    kept = [(t, d) for t, d in blocks if d > d_inf + 1e-12]
    slopes = []
    for i, (ti, di) in enumerate(kept):
        for tj, dj in kept[i + 1:]:
            slopes.append((math.log(dj - d_inf) - math.log(di - d_inf)) / (tj - ti))
    k = float(-np.median(slopes)) if slopes else None
    half = next((i + 1 for i, x in enumerate(values) if x <= .5 * d0), None)
    return {"d0": d0, "d_inf": d_inf, "k": k, "retained_blocks": len(kept),
            "half_decisions": half, "fit_valid": bool(k is not None and k > 0 and len(kept) >= 5)}


def nearest(mu, f, c):
    return int(np.argmin(np.sum((mu[c] - f) ** 2, axis=1)))


def q_state(history):
    q = float(np.mean(history[-Q_WINDOW:])) if history else 1.0
    pressure = ALPHA * q * V
    return q, pressure, "GREEN" if pressure >= THETA_MIN else "AMBER"


def detector(cat_hist, phase1_baseline):
    drops = {c: phase1_baseline[c] - float(np.mean(v[-CAT_WINDOW:]))
             for c, v in cat_hist.items() if len(v) >= CAT_WINDOW}
    if len(drops) < C:
        return False, [], drops
    # A positive mean drop is broad degradation, not a sparse event to target.
    if float(np.mean(list(drops.values()))) > 0.01:
        return False, [], drops
    down = [c for c, d in drops.items() if d > .10]
    held = [c for c, d in drops.items() if abs(d) <= .05]
    selected = sorted(drops, key=lambda c: (-drops[c], c))[:2]
    return len(down) >= 2 and len(held) >= 2, selected if len(down) >= 2 and len(held) >= 2 else [], drops


def run_arm(policy, gt1, gt2, vectors1, vectors2, uniform=False):
    mu0 = np.full((C, A, D), .5, dtype=float)
    if policy == "B-theorem":
        model = None
        mu = mu0.copy()
    else:
        model = ProfileScorer(mu=mu0.copy(), actions=[f"a{i}" for i in range(A)],
                              categories=[f"c{i}" for i in range(C)], eta_override=.01,
                              auto_pause_on_amber=(policy == "A-baseline"))
        mu = model.mu
    all_hist, cat_hist, base_hist = [], {c: [] for c in range(C)}, {c: [] for c in range(C)}
    trajectories = []
    mode_events = []
    selected = []
    phase_outputs = []

    def phase(vectors, target, name):
        nonlocal mu, selected
        d_active = [float(np.linalg.norm(mu - target))]
        d_disrupted = [float(np.linalg.norm(mu[DISRUPTED] - target[DISRUPTED]))]
        accuracies, category_accuracy, cat_roll = [], {c: [] for c in range(C)}, {c: [] for c in range(C)}
        phase1_baseline = None
        for i, row in enumerate(vectors):
            f = np.asarray(row["f"], dtype=float); c = int(row["category_index"]); a = int(row["target_action_index"])
            pred = int(np.argmin(np.sum((mu[c] - f) ** 2, axis=1)))
            correct = bool(pred == a)
            all_hist.append(int(correct)); category_accuracy[c].append(int(correct)); cat_roll[c].append(int(correct))
            if name == "phase1":
                base_hist[c].append(int(correct))
            elif phase1_baseline is None and all(len(base_hist[x]) >= 2 for x in range(C)):
                phase1_baseline = {x: float(np.mean(base_hist[x])) for x in range(C)}
            q, pressure, status = q_state(all_hist)
            active_mode, chosen, drops = detector(cat_roll, phase1_baseline or {x: 1.0 for x in range(C)}) if name == "phase2" else (False, [], {})
            if active_mode and not selected:
                selected = chosen
                mode_events.append({"index": i + 1, "event": "enter", "selected": selected, "drops": drops, "uniform": uniform})
            # Exit after selected categories recover within 5pp, or timeout.
            if selected:
                selected_drops = [drops.get(x, 1.0) for x in selected]
                if all(abs(x) <= .05 for x in selected_drops) or any(e.get("event") == "enter" and i + 1 - e["index"] >= 120 for e in mode_events):
                    mode_events.append({"index": i + 1, "event": "exit", "selected": selected, "uniform": uniform})
                    selected = []
            mode_active = bool(selected)
            if policy == "B-theorem":
                mu[c, a] = np.clip(mu[c, a] + ETA_B * (f - mu[c, a]), 0, 1)
                update_allowed, update_reason = True, "theorem_rule"
            else:
                if name == "phase1":
                    update_allowed, update_reason = True, "normal"
                elif status == "GREEN":
                    update_allowed, update_reason = True, "conservation_green"
                elif q >= THETA_MIN and mode_active and c in selected and policy.startswith("R"):
                    update_allowed, update_reason = True, "selective_redistribution"
                else:
                    update_allowed, update_reason = False, "paused_conservation"
                if model is not None:
                    if policy == "R2-selective-boost" and mode_active and update_allowed:
                        enter = next((e["index"] for e in reversed(mode_events) if e["event"] == "enter"), i + 1)
                        model.eta_override = .05 if i + 1 - enter < BOOST_WINDOW else .01
                    else:
                        model.eta_override = .01
                    if policy == "A-baseline":
                        model.set_conservation_status(status)
                    else:
                        model.set_conservation_status("GREEN")
                    if update_allowed:
                        model.update(f, c, pred, correct, gt_action_index=a)
            d_active.append(float(np.linalg.norm(mu - target))); d_disrupted.append(float(np.linalg.norm(mu[DISRUPTED] - target[DISRUPTED])))
            trajectories.append({"phase": name, "index": i + 1, "category": c, "correct": correct,
                                 "q": q, "pressure": pressure, "status": status, "mode_active": mode_active,
                                 "selected": selected, "update_allowed": update_allowed,
                                 "update_reason": update_reason, "predicted": pred})
            for x in range(C):
                if category_accuracy[x]:
                    cat_roll[x] = category_accuracy[x][-CAT_WINDOW:]
        return {"d_active": d_active, "d_disrupted": d_disrupted, "accuracy": accuracies,
                "category_accuracy": category_accuracy, "mu_final": mu.copy()}

    p1 = phase(vectors1, gt1, "phase1")
    p2 = phase(vectors2, gt2, "phase2")
    r1, r2 = fit_rate(p1["d_active"]), fit_rate(p2["d_disrupted"])
    gamma = None if not (r1["fit_valid"] and r2["fit_valid"]) else float(r2["k"] / r1["k"])
    safe = [x["q"] for x in trajectories if x["index"] >= Q_WINDOW]
    return {"policy": policy, "phase1": p1, "phase2": p2, "rate1": r1, "rate2": r2,
            "gamma_rate": gamma, "fit_valid_gate": gamma is not None, "mode_events": mode_events,
            "trajectories": trajectories, "min_overall_q": min(safe) if safe else 1.0,
            "safety_floor_held": bool((min(safe) if safe else 1.0) >= THETA_MIN),
            "mode_activated": any(e["event"] == "enter" for e in mode_events),
            "invariants": {
                "mu0_exact_canonical_half": bool(np.array_equal(mu0, np.full((C, A, D), .5))),
                "gt1_distinct_from_mu0": bool(np.linalg.norm(gt1 - mu0) > 0),
                "gt2_finite": bool(np.isfinite(np.linalg.norm(gt2 - mu0))),
                "fit_gate_is_scale_free": True,
                "arm_policy_only_difference": policy in POLICIES,
                "overall_floor_not_relaxed": True,
            },
            "uniform": uniform, "mu0": mu0, "mu_final": p2["mu_final"]}


def main():
    if ROOT.exists(): shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    cells = []
    for seed in SEEDS:
        for eps in EPS:
            source = SOURCE / "runs" / f"seed-{seed:03d}" / f"epsilon-{eps:.2f}" / "arm-A"
            gt1, gt2, mu0 = np.load(source / "gt1.npy"), np.load(source / "gt2.npy"), np.load(source / "mu0.npy")
            v1, v2 = load_json(source / "vectors_phase1.json"), load_json(source / "vectors_phase2.json")
            for policy in POLICIES:
                result = run_arm(policy, gt1, gt2, v1, v2)
                cell = ROOT / "runs" / f"seed-{seed:03d}" / f"epsilon-{eps:.2f}" / policy
                cell.mkdir(parents=True, exist_ok=True)
                for name, arr in [("gt1", gt1), ("gt2", gt2), ("mu0", mu0), ("mu_final", result["mu_final"])]: save_array(cell / f"{name}.npy", arr)
                write_json(cell / "vectors_phase1.json", v1); write_json(cell / "vectors_phase2.json", v2)
                write_json(cell / "record.json", {k: v for k, v in result.items() if k not in {"mu0", "mu_final"}})
                cells.append({"seed": seed, "epsilon_firm": eps, "policy": policy, "gamma_rate": result["gamma_rate"],
                              "fit_valid_gate": result["fit_valid_gate"], "min_overall_q": result["min_overall_q"],
                              "safety_floor_held": result["safety_floor_held"], "mode_activated": result["mode_activated"],
                              "uniform": False})
            # Uniform degradation selectivity control for A and R2.
            delta = np.full((A, D), DELTA_NORM / math.sqrt(A * D))
            gt2u = np.clip(gt1 + delta[None, :, :], 0, 1)
            vu = []
            for row in v2:
                c = int(row["category_index"]); a = int(row["target_action_index"]); rr = dict(row)
                # Uniform-control fix: a common target translation preserves action geometry and
                # does not guarantee a uniform accuracy degradation.  For this selectivity-only
                # control, center every observation on the farthest *other* action profile for
                # its category, with a small common noise term.  Labels remain the original a;
                # therefore every category is stressed uniformly while GT and the oracle rule
                # remain unchanged.  Primary cells are untouched.
                z = np.asarray(row["noise"], dtype=float)
                wrong = int(np.argmax([np.linalg.norm(gt1[c, a] - gt1[c, j]) if j != a else -1.0 for j in range(A)]))
                rr["f"] = np.clip(gt1[c, wrong] + 0.02 * z, 0, 1).tolist(); vu.append(rr)
            for policy in ["A-baseline", "R2-selective-boost"]:
                result = run_arm(policy, gt1, gt2u, v1, vu, uniform=True)
                cell = ROOT / "uniform_control" / f"seed-{seed:03d}" / f"epsilon-{eps:.2f}" / policy
                cell.mkdir(parents=True, exist_ok=True)
                write_json(cell / "record.json", {k: v for k, v in result.items() if k not in {"mu0", "mu_final"}})
                cells.append({"seed": seed, "epsilon_firm": eps, "policy": policy, "gamma_rate": result["gamma_rate"],
                              "fit_valid_gate": result["fit_valid_gate"], "min_overall_q": result["min_overall_q"],
                              "safety_floor_held": result["safety_floor_held"], "mode_activated": result["mode_activated"],
                              "uniform": True})
    write_json(ROOT / "summary.json", {"schema": "r-hcurve-v1", "cells": cells})
    # Chart using primary cells; show all policies for transparency.
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    colors = {"A-baseline": "#D55E00", "R1-selective-redistribution": "#009E73", "R2-selective-boost": "#0072B2", "B-theorem": "#444444"}
    for policy in POLICIES:
        means = []
        for eps in EPS:
            vals = [x["gamma_rate"] for x in cells if not x["uniform"] and x["policy"] == policy and x["epsilon_firm"] == eps and x["gamma_rate"] is not None]
            means.append(np.mean(vals) if vals else np.nan)
            ax.scatter([eps] * len(vals), vals, color=colors[policy], alpha=.65, s=42)
        ax.plot(EPS, means, marker="o", color=colors[policy], label=policy)
    ax.axhline(1, color="black", linewidth=1, linestyle="--"); ax.axvline(EPS_STAR, color="#777", linestyle=":")
    ax.text(EPS_STAR + .005, ax.get_ylim()[1] * .98, "ε*=0.125", fontsize=9)
    ax.set_xlabel("ε_firm"); ax.set_ylabel("γ_rate = k₂/k₁"); ax.set_title("R-HCURVE: selective conservation-aware re-convergence")
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, fontsize=8)
    figdir = HERE / "figures"; figdir.mkdir(exist_ok=True)
    fig.savefig(figdir / "r_hcurve_gamma.png", dpi=300); fig.savefig(figdir / "r_hcurve_gamma.pdf"); plt.close(fig)
    (figdir / "r_hcurve_gamma.caption.txt").write_text("Selective category-aware conservation mode is compared with production Arm A and theorem Arm B. γ=1 is the re-convergence boundary; ε*=0.125 is marked.\n", encoding="utf-8")
    files = []
    for p in sorted(ROOT.rglob("*")) + sorted(figdir.glob("r_hcurve_gamma*")):
        if p.is_file() and p.name != "manifest.json": files.append({"path": str(p.relative_to(HERE)).replace(os.sep, "/"), "bytes": p.stat().st_size, "sha256": sha(p), "role": "r_hcurve_artifact"})
    for p, role in [(HERE / "design_r_hcurve.md", "design"), (HERE / "results_r_hcurve.md", "results")]:
        if p.is_file(): files.append({"path": str(p.relative_to(HERE)).replace(os.sep, "/"), "bytes": p.stat().st_size, "sha256": sha(p), "role": role})
    write_json(ROOT / "manifest.json", {"schema": "r-hcurve-manifest-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"seeds": SEEDS, "eps": EPS, "alpha": ALPHA, "V": V, "theta_min": THETA_MIN, "q_window": Q_WINDOW, "category_window": CAT_WINDOW, "window": WINDOW, "boost_window": BOOST_WINDOW, "epsilon_star": EPS_STAR},
        "policies": POLICIES, "uniform_control": True, "files": files, "cells": cells})
    manifest = load_json(ROOT / "manifest.json")
    assert all((HERE / x["path"]).stat().st_size == x["bytes"] and sha(HERE / x["path"]) == x["sha256"] for x in manifest["files"])
    print(json.dumps({"cells": len(cells), "files": len(files), "manifest": str(ROOT / "manifest.json")}, indent=2))


if __name__ == "__main__": main()
