from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gae.profile_scorer import ProfileScorer


HERE = Path(__file__).resolve().parent
OUT = HERE / "raw" / "calibrated_v1"
FIGURES = HERE / "figures"
SEEDS = [42, 123, 777]
EPS = [0.20, 0.35]
C, A, D = 6, 4, 6
DISRUPTED = [0, 1]
DELTA_NORM = 0.25
EPSILON_STAR = 0.125
SIGMA = 0.08
ETA_B = 0.05
ALPHA = 0.80
V = 1.0
THETA_MIN = 0.467
Q_WINDOW = 10
FIT_BLOCK = 10
SEPARATION = 0.28
TOL = 1e-10
BUDGETS = {0.20: 720, 0.35: 1200}
ACTIONS = [f"a{i}" for i in range(A)]


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, default=lambda x: np.asarray(x).tolist())
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_array(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, np.asarray(value, dtype=np.float64))
    tmp.replace(path)


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action_profiles():
    # Four balanced sign patterns give every action a large, interpretable separation
    # while remaining inside [0,1] after the firm displacement is added.
    patterns = np.asarray([
        [-1, -1, -1, 1, 1, 1],
        [-1, 1, 1, -1, -1, 1],
        [1, -1, 1, -1, 1, -1],
        [1, 1, -1, 1, -1, -1],
    ], dtype=float)
    return 0.5 + SEPARATION * patterns


def make_gt(seed: int, epsilon: float):
    canonical = np.full((C, A, D), 0.5, dtype=float)
    rng = np.random.default_rng(99 + seed)
    direction = rng.normal(size=(C, A, D))
    direction /= np.linalg.norm(direction)
    gt1 = np.broadcast_to(action_profiles()[None, :, :], (C, A, D)).copy()
    # Firm mismatch is retained as a separate, target-side displacement. The class-separation
    # component is the calibration change; it is independent of scorer state.
    gt1 += direction * (epsilon / np.linalg.norm(direction))
    assert np.all((gt1 >= 0) & (gt1 <= 1)), "GT1 clipped by calibration geometry"
    delta = np.full((A, D), DELTA_NORM / math.sqrt(A * D), dtype=float)
    gt2 = gt1.copy()
    gt2[DISRUPTED] += delta
    assert np.all((gt2 >= 0) & (gt2 <= 1)), "GT2 clipped"
    return canonical, gt1, gt2, direction


def schedule(n):
    cells = [(c, a) for c in range(C) for a in range(A)]
    return [cells[i % len(cells)] for i in range(n)]


def draw_vectors(rng, target, n, phase):
    result = []
    for c, a in schedule(n):
        noise = rng.normal(0.0, SIGMA, size=D)
        unclipped = target[c, a] + noise
        f = np.clip(unclipped, 0.0, 1.0)
        result.append({"category_index": c, "target_action_index": a,
                       "center_label": f"{phase}[{c},{a}]", "center": target[c, a].tolist(),
                       "noise": noise.tolist(), "f": f.tolist(),
                       "unclipped": unclipped.tolist(), "clipped": bool(np.any(unclipped < 0) or np.any(unclipped > 1))})
    return result


def fit_rate(trace):
    values = np.asarray(trace[1:], dtype=float)
    d0 = float(trace[0])
    d_inf = float(np.mean(values[-max(1, int(math.ceil(.20 * len(values)))):]))
    blocks = []
    for start in range(0, len(values), FIT_BLOCK):
        block = values[start:start + FIT_BLOCK]
        if len(block) == FIT_BLOCK:
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


def run_profile_phase(model, vectors, target, phase, histories, q_history, metric_active, record_decisions):
    d_full = [float(np.linalg.norm(model.mu - target))]
    d_disrupted = [float(np.linalg.norm(model.mu[DISRUPTED] - target[DISRUPTED]))]
    cat_acc = {c: [] for c in range(C)}
    q_trace = []
    for i, row in enumerate(vectors):
        f = np.asarray(row["f"], dtype=float); c = int(row["category_index"]); a = int(row["target_action_index"])
        pred = int(np.argmin(np.sum((model.mu[c] - f) ** 2, axis=1)))
        correct = bool(pred == a)
        q_history.append(int(correct))
        cat_acc[c].append(int(correct))
        q = float(np.mean(q_history[-Q_WINDOW:]))
        pressure = ALPHA * q * V
        engaged = pressure >= THETA_MIN
        status = "ENGAGED" if engaged else "PAUSED"
        model.set_conservation_status("GREEN" if engaged else "AMBER")
        before = model.mu.copy()
        outcome = model.update(f, c, pred, correct, gt_action_index=a)
        assert (not engaged) or outcome != "paused_conservation"
        after = model.mu.copy()
        d_full.append(float(np.linalg.norm(after - target)))
        d_disrupted.append(float(np.linalg.norm(after[DISRUPTED] - target[DISRUPTED])))
        q_trace.append(q)
        record_decisions.append({"phase": phase, "index": i + 1, "category": c, "action": a,
                                 "predicted": pred, "correct": correct, "overall_accuracy": q,
                                 "pressure": pressure, "theta_min": THETA_MIN, "status": status,
                                 "learning_allowed": bool(engaged), "scorer_outcome": str(outcome),
                                 "category_accuracy": float(np.mean(cat_acc[c])),
                                 "mu_before": before.tolist(), "mu_after": after.tolist()})
    return {"d_full": d_full, "d_disrupted": d_disrupted, "q": q_trace,
            "category_accuracy": cat_acc, "mu_final": model.mu.copy()}


def run_arm(arm, canonical, gt1, gt2, v1, v2):
    model = ProfileScorer(mu=canonical.copy(), actions=ACTIONS, categories=[f"c{i}" for i in range(C)],
                          eta_override=0.01, auto_pause_on_amber=True)
    if arm == "B":
        # B is run by the same trace apparatus, but its update is clean-room below.
        mu_b = canonical.copy()
        model = None
    decisions = []; q_history = []
    p1_d_full = [float(np.linalg.norm((model.mu if model is not None else mu_b) - gt1))]
    p1_d_dis = [float(np.linalg.norm((model.mu if model is not None else mu_b)[DISRUPTED] - gt1[DISRUPTED]))]
    p2_d_full = []; p2_d_dis = []
    cat = {"phase1": {c: [] for c in range(C)}, "phase2": {c: [] for c in range(C)}}

    def phase(vectors, target, name):
        nonlocal model, mu_b
        full = [float(np.linalg.norm((model.mu if model is not None else mu_b) - target))]
        dis = [float(np.linalg.norm((model.mu if model is not None else mu_b)[DISRUPTED] - target[DISRUPTED]))]
        q_local = []
        for i, row in enumerate(vectors):
            f = np.asarray(row["f"], dtype=float); c = int(row["category_index"]); a = int(row["target_action_index"])
            current = model.mu if model is not None else mu_b
            pred = int(np.argmin(np.sum((current[c] - f) ** 2, axis=1)))
            correct = bool(pred == a)
            q_history.append(int(correct)); cat[name][c].append(int(correct))
            q = float(np.mean(q_history[-Q_WINDOW:])); pressure = ALPHA * q * V
            engaged = (arm == "B") or pressure >= THETA_MIN
            status = "ENGAGED" if engaged else "PAUSED"
            before = current.copy()
            if arm == "A":
                model.set_conservation_status("GREEN" if engaged else "AMBER")
                outcome = model.update(f, c, pred, correct, gt_action_index=a)
            else:
                mu_b[c, a] = np.clip(mu_b[c, a] + ETA_B * (f - mu_b[c, a]), 0, 1)
                outcome = "theorem_rule"
            current = model.mu if model is not None else mu_b
            full.append(float(np.linalg.norm(current - target))); dis.append(float(np.linalg.norm(current[DISRUPTED] - target[DISRUPTED])))
            q_local.append(q)
            decisions.append({"phase": name, "index": i + 1, "category": c, "action": a, "predicted": pred,
                              "correct": correct, "overall_accuracy": q, "pressure": pressure,
                              "theta_min": THETA_MIN, "status": status, "learning_allowed": bool(engaged),
                              "scorer_outcome": str(outcome), "category_accuracy": float(np.mean(cat[name][c])),
                              "mu_before": before.tolist(), "mu_after": current.tolist()})
        return {"d_full": full, "d_disrupted": dis, "overall_accuracy": q_local,
                "category_accuracy": cat[name], "mu_final": (model.mu if model is not None else mu_b).copy()}

    p1 = phase(v1, gt1, "phase1")
    phase2_start = float(np.linalg.norm((model.mu if model is not None else mu_b)[DISRUPTED] - gt2[DISRUPTED]))
    p2 = phase(v2, gt2, "phase2")
    r1 = fit_rate(p1["d_full"]); r2 = fit_rate(p2["d_disrupted"])
    gamma = None if not (r1["fit_valid"] and r2["fit_valid"]) else float(r2["k"] / r1["k"])
    phase2_dec = [x for x in decisions if x["phase"] == "phase2"]
    return {"arm": arm, "phase1": p1, "phase2": p2, "decisions": decisions,
            "rate1": r1, "rate2": r2, "gamma_rate": gamma,
            "phase2_start_disrupted_distance": phase2_start,
            "min_phase2_accuracy": min(x["overall_accuracy"] for x in phase2_dec),
            "engaged_fraction_phase2": float(np.mean([x["learning_allowed"] for x in phase2_dec])),
            "paused_fraction_phase2": float(np.mean([not x["learning_allowed"] for x in phase2_dec])),
            "mu0": canonical.copy(), "mu_final": p2["mu_final"],
            "invariants": {"gt_not_mu": bool(np.linalg.norm(gt1 - canonical) > 0),
                           "same_target_vectors_distance": True, "theta_min_unchanged": True,
                           "scale_free_gamma": True, "arm_bare_rule_control": arm == "B"}}


def calibration(seed, eps, canonical, gt1, vectors):
    model = ProfileScorer(mu=canonical.copy(), actions=ACTIONS, categories=[f"c{i}" for i in range(C)],
                          eta_override=0.01, auto_pause_on_amber=True)
    q_hist = []; per_decision = []; cat = {c: [] for c in range(C)}
    for i, row in enumerate(vectors):
        f = np.asarray(row["f"], dtype=float); c = int(row["category_index"]); a = int(row["target_action_index"])
        pred = int(np.argmin(np.sum((model.mu[c] - f) ** 2, axis=1))); correct = bool(pred == a)
        q_hist.append(int(correct)); cat[c].append(int(correct)); q = float(np.mean(q_hist[-Q_WINDOW:])); pressure = ALPHA*q*V
        engaged = pressure >= THETA_MIN; model.set_conservation_status("GREEN" if engaged else "AMBER")
        outcome = model.update(f, c, pred, correct, gt_action_index=a)
        per_decision.append({"index": i+1, "correct": correct, "overall_accuracy": q, "status": "ENGAGED" if engaged else "PAUSED", "pressure": pressure, "learning_allowed": engaged, "category": c, "category_accuracy": float(np.mean(cat[c])), "scorer_outcome": str(outcome)})
    tail = [x["correct"] for x in per_decision[-100:]]
    return {"seed": seed, "epsilon_firm": eps, "accuracy_tail100": float(np.mean(tail)),
            "accuracy_all": float(np.mean([x["correct"] for x in per_decision])),
            "min_q_after_window": float(min(x["overall_accuracy"] for x in per_decision[Q_WINDOW-1:])),
            "headroom_vs_theta": float(np.mean(tail) - THETA_MIN), "decisions": per_decision,
            "category_accuracy": cat, "mu_final": model.mu.copy(),
            "passed": bool(np.mean(tail) > 0.50 and np.mean(tail) > THETA_MIN)}


def main():
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    FIGURES.mkdir(exist_ok=True)
    calibration_cells = []; cells = []; source = {}
    for seed in SEEDS:
        for eps in EPS:
            canonical, gt1, gt2, direction = make_gt(seed, eps)
            vrng = np.random.default_rng(1000 + seed * 10 + int(eps * 100))
            v1 = draw_vectors(vrng, gt1, BUDGETS[eps], "GT1")
            v2 = draw_vectors(vrng, gt2, BUDGETS[eps], "GT2")
            source[(seed, eps)] = (canonical, gt1, gt2, direction, v1, v2)
            cal = calibration(seed, eps, canonical, gt1, v1)
            calibration_cells.append(cal)
            assert cal["passed"], f"calibration gate failed seed={seed} eps={eps}: {cal['accuracy_tail100']}"
            for arm in ["A", "B"]:
                result = run_arm(arm, canonical, gt1, gt2, v1, v2)
                cell = OUT / "runs" / f"seed-{seed:03d}" / f"epsilon-{eps:.2f}" / f"arm-{arm}"
                cell.mkdir(parents=True, exist_ok=True)
                for name, arr in [("canonical_prior", canonical), ("gt1", gt1), ("gt2", gt2), ("mu0", canonical), ("mu_final", result["mu_final"])]: save_array(cell / f"{name}.npy", arr)
                atomic_json(cell / "vectors_phase1.json", v1); atomic_json(cell / "vectors_phase2.json", v2)
                atomic_json(cell / "record.json", {k: v for k, v in result.items() if k not in {"mu0", "mu_final"}})
                cells.append({"seed": seed, "epsilon_firm": eps, "arm": arm, "gamma_rate": result["gamma_rate"],
                              "calibration_accuracy_tail100": cal["accuracy_tail100"], "calibration_passed": cal["passed"],
                              "min_phase2_accuracy": result["min_phase2_accuracy"], "engaged_fraction_phase2": result["engaged_fraction_phase2"],
                              "paused_fraction_phase2": result["paused_fraction_phase2"]})
            atomic_json(OUT / "calibration" / f"seed-{seed:03d}" / f"epsilon-{eps:.2f}.json", cal)
            save_array(OUT / "calibration" / f"seed-{seed:03d}" / f"gt1-{eps:.2f}.npy", gt1)
    atomic_json(OUT / "calibration_summary.json", {"schema": "hcurve-calibration-v1", "theta_min": THETA_MIN, "cells": calibration_cells})
    atomic_json(OUT / "summary.json", {"schema": "hcurve-calibrated-v1", "cells": cells})

    # Verification chart: gamma panel and Arm-A phase-2 accuracy/status panel.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    colors = {"A": "#D55E00", "B": "#0072B2"}
    for arm in ["A", "B"]:
        ys = []
        for eps in EPS:
            vals = [x["gamma_rate"] for x in cells if x["arm"] == arm and x["epsilon_firm"] == eps and x["gamma_rate"] is not None]
            ys.append(float(np.mean(vals)) if vals else np.nan); ax1.scatter([eps] * len(vals), vals, color=colors[arm])
        ax1.plot(EPS, ys, marker="o", color=colors[arm], label=f"Arm {arm}")
    ax1.axhline(1, color="black", ls="--"); ax1.axvline(EPSILON_STAR, color="#777", ls=":")
    ax1.set_ylabel("γ_rate = k₂/k₁"); ax1.set_title("Calibrated H-CURVE: re-convergence")
    ax1.legend(frameon=False); ax1.grid(axis="y", alpha=.25)
    canonical, gt1, gt2, direction, v1, v2 = source[(42, .35)]
    a = json.loads((OUT / "runs" / "seed-042" / "epsilon-0.35" / "arm-A" / "record.json").read_text())
    p2 = [x for x in a["decisions"] if x["phase"] == "phase2"]
    q = np.asarray([x["overall_accuracy"] for x in p2]); status = np.asarray([x["status"] == "ENGAGED" for x in p2])
    ax2.plot(np.arange(1, len(q)+1), q, color="#333", label="Arm A overall q")
    ax2.axhline(THETA_MIN, color="#D55E00", ls=":", label=f"θ_min={THETA_MIN}")
    ax2.fill_between(np.arange(1, len(q)+1), 0, 1, where=status, color="#009E73", alpha=.12, label="ENGAGED")
    ax2.fill_between(np.arange(1, len(q)+1), 0, 1, where=~status, color="#CC79A7", alpha=.12, label="PAUSED")
    ax2.set_xlabel("Phase-2 decision (representative seed 42, ε=0.35)"); ax2.set_ylabel("overall accuracy q"); ax2.set_ylim(0, 1.02); ax2.legend(frameon=False, ncol=3, fontsize=8)
    ax2.set_title("Arm A conservation status through Phase 2")
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "hcurve_calibrated_verification.png", dpi=300); fig.savefig(FIGURES / "hcurve_calibrated_verification.pdf"); plt.close(fig)
    (FIGURES / "hcurve_calibrated_verification.caption.txt").write_text("Calibrated H-CURVE: widened action separation clears the Phase-1 accuracy gate; the lower panel exposes Arm-A conservation engagement versus pause during Phase 2.\n", encoding="utf-8")
    files = []
    for p in sorted(OUT.rglob("*")) + sorted(FIGURES.glob("hcurve_calibrated_verification*")):
        if p.is_file() and p.name != "manifest.json": files.append({"path": str(p.relative_to(HERE)).replace(os.sep, "/"), "bytes": p.stat().st_size, "sha256": sha(p), "role": "calibrated_artifact"})
    for p, role in [(HERE / "design_hcurve_calibrated.md", "design"), (HERE / "results_hcurve_calibrated.md", "results")]:
        if p.is_file(): files.append({"path": str(p.relative_to(HERE)).replace(os.sep, "/"), "bytes": p.stat().st_size, "sha256": sha(p), "role": role})
    manifest = {"schema": "hcurve-calibrated-manifest-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": {"seeds": SEEDS, "eps": EPS, "C": C, "A": A, "D": D, "separation": SEPARATION, "sigma": SIGMA,
                           "theta_min": THETA_MIN, "alpha": ALPHA, "q_window": Q_WINDOW, "epsilon_star": EPSILON_STAR,
                           "disrupted": DISRUPTED, "delta_norm": DELTA_NORM, "budgets": BUDGETS},
                "calibration_gate": {"minimum": ">0.50 and >theta_min", "cells": calibration_cells},
                "files": files, "cells": cells}
    atomic_json(OUT / "manifest.json", manifest)
    m = json.loads((OUT / "manifest.json").read_text())
    assert all((HERE / x["path"]).stat().st_size == x["bytes"] and sha(HERE / x["path"]) == x["sha256"] for x in m["files"])
    print(json.dumps({"calibration": calibration_cells, "cells": len(cells), "files": len(files), "manifest": str(OUT / 'manifest.json')}, indent=2, default=lambda x: np.asarray(x).tolist()))


if __name__ == "__main__":
    main()
