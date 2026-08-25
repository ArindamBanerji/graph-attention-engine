from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GAE_ROOT = HERE.parents[1]
sys.path.insert(0, str(GAE_ROOT))

from gae.profile_scorer import ProfileScorer
from run_two_arm_v4 import (
    ACTIONS, A, C, DISRUPTED, EPSILONS, EPSILON_STAR, MIN_RETAINED_BLOCKS,
    SEEDS, TOL, fit_rate, nearest_action, write_json_atomic, save_array,
)

ROOT = HERE / "raw" / "two_arm_v4"
OUT = HERE / "raw" / "arm_c_v4"
POLICIES = ["C1_symmetric_rate", "C2_shift_triggered_boost", "C3_production_theorem_rate"]
C_EPSILONS = [0.20, 0.35]
WINDOW = 10
BOOST_DECISIONS = 24


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_vectors(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def distances(mu, target, mask):
    diff = mu - target
    return float(np.linalg.norm(diff)), float(np.linalg.norm(diff[mask])), float(np.linalg.norm(diff[DISRUPTED]))


def run_policy(policy: str, mu0: np.ndarray, gt1: np.ndarray, gt2: np.ndarray,
               vectors1: list[dict], vectors2: list[dict]):
    if policy == "C1_symmetric_rate":
        model = ProfileScorer(mu=mu0.copy(), actions=ACTIONS,
                              categories=[f"c{i}" for i in range(C)],
                              eta_override=0.05)
        conservation_mode = "not_paused_initial_green"
    elif policy == "C3_production_theorem_rate":
        model = ProfileScorer(mu=mu0.copy(), actions=ACTIONS,
                              categories=[f"c{i}" for i in range(C)],
                              eta_override=0.05, auto_pause_on_amber=True)
        conservation_mode = "auto_pause_enabled_no_transition_signal_in_standalone_harness"
    else:
        model = ProfileScorer(mu=mu0.copy(), actions=ACTIONS,
                              categories=[f"c{i}" for i in range(C)],
                              eta_override=0.01)
        conservation_mode = "not_paused_initial_green"

    boost_triggered = False
    boost_start = None
    boost_remaining = 0
    phase1_errors = []
    phase2_errors = []
    trigger_reason = None

    def phase(vectors, target, phase_name):
        nonlocal boost_triggered, boost_start, boost_remaining, trigger_reason
        counts = np.zeros((C, A), dtype=int)
        active = np.zeros((C, A), dtype=bool)
        metric_mask = np.ones((C, A), dtype=bool)
        full0, active0, disrupted0 = distances(model.mu, target, metric_mask)
        full_trace, active_trace, disrupted_trace = [full0], [active0], [disrupted0]
        active_counts = [0]
        decisions = []
        rolling = []
        errors = phase1_errors if phase_name == "phase1" else phase2_errors
        for index, row in enumerate(vectors):
            f = np.asarray(row["f"], dtype=np.float64)
            c = int(row["category_index"])
            a = int(row["target_action_index"])
            predicted = nearest_action(model.mu, f, c)
            correct = predicted == a
            if phase_name == "phase2" and policy == "C2_shift_triggered_boost":
                recent = errors[-WINDOW:]
                baseline_error = float(np.mean(phase1_errors)) if phase1_errors else 0.0
                if (not boost_triggered and len(recent) == WINDOW and
                        float(np.mean(recent)) >= max(0.50, baseline_error + 0.15)):
                    boost_triggered = True
                    boost_start = index + 1
                    boost_remaining = BOOST_DECISIONS
                    trigger_reason = {
                        "detector": "phase2 rolling error spike",
                        "window": WINDOW,
                        "baseline_phase1_error": baseline_error,
                        "phase2_window_error": float(np.mean(recent)),
                    }
                if boost_remaining > 0:
                    model.eta_override = 0.05
                else:
                    model.eta_override = 0.01
            model.update(f, c, predicted, correct, gt_action_index=a)
            if phase_name == "phase2" and boost_remaining > 0:
                boost_remaining -= 1
            errors.append(0 if correct else 1)
            counts[c, a] += 1
            active[c, a] = True
            full, active_d, disrupted_d = distances(model.mu, target, metric_mask)
            full_trace.append(full); active_trace.append(active_d); disrupted_trace.append(disrupted_d)
            active_counts.append(int(active.sum()))
            rolling.append(float(1.0 - np.mean(errors[-WINDOW:])))
            decisions.append({
                "index": index + 1, "category_index": c, "target_action_index": a,
                "predicted_action_index": predicted, "correct": correct,
                "eta_override_used": float(model.eta_override),
            })
        return {
            "phase": phase_name, "mu_initial": None, "mu_final": model.mu.copy(),
            "d_full": full_trace, "d_active": active_trace, "d_disrupted": disrupted_trace,
            "active_counts": active_counts, "rolling_accuracy": rolling,
            "decisions": decisions, "counts": counts.tolist(), "active": active.tolist(),
        }

    p1 = phase(vectors1, gt1, "phase1")
    phase2_start = float(np.linalg.norm(model.mu[DISRUPTED] - gt2[DISRUPTED]))
    p2 = phase(vectors2, gt2, "phase2")
    r1 = fit_rate(p1["d_active"])
    r2 = fit_rate(p2["d_disrupted"])
    valid = bool(r1["k"] is not None and r1["k"] > 0 and r1["retained_blocks"] >= MIN_RETAINED_BLOCKS and
                 r2["k"] is not None and r2["k"] > 0 and r2["retained_blocks"] >= MIN_RETAINED_BLOCKS)
    gamma = None if not valid else float(r2["k"] / r1["k"])
    n1, n2 = r1["half_decisions"], r2["half_decisions"]
    gamma_half = None if n1 is None or n2 is None else float(n1 / n2)
    return {
        "policy": policy, "phase1": p1, "phase2": p2,
        "phase1_rate": r1, "phase2_disrupted_rate": r2,
        "fit_valid_gate": valid, "gamma_rate": gamma, "n_half_phase1": n1,
        "n_half_phase2": n2, "gamma_half": gamma_half,
        "f1_fired": None if gamma is None else bool((gamma > 1) != (float(current_epsilon) > EPSILON_STAR)),
        "f2_fired": None if gamma is None or gamma_half is None else bool((gamma > 1) != (gamma_half > 1)),
        "f3_fired": not valid, "phase2_start_disrupted_distance": phase2_start,
        "conservation_mode": conservation_mode, "shift_triggered": boost_triggered,
        "shift_trigger_start": boost_start, "shift_trigger_reason": trigger_reason,
    }


def main():
    global current_epsilon
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    cells = []
    for seed in SEEDS:
        for epsilon in C_EPSILONS:
            current_epsilon = epsilon
            source = ROOT / "runs" / f"seed-{seed:03d}" / f"epsilon-{epsilon:.2f}" / "arm-A"
            gt1 = np.load(source / "gt1.npy")
            gt2 = np.load(source / "gt2.npy")
            mu0 = np.load(source / "mu0.npy")
            vectors1 = load_vectors(source / "vectors_phase1.json")
            vectors2 = load_vectors(source / "vectors_phase2.json")
            for policy in POLICIES:
                safe = policy.replace("_", "-")
                cell = OUT / "runs" / f"seed-{seed:03d}" / f"epsilon-{epsilon:.2f}" / safe
                cell.mkdir(parents=True, exist_ok=True)
                record = run_policy(policy, mu0, gt1, gt2, vectors1, vectors2)
                save_array(cell / "canonical_prior.npy", mu0)
                save_array(cell / "gt1.npy", gt1)
                save_array(cell / "gt2.npy", gt2)
                save_array(cell / "mu0.npy", mu0)
                save_array(cell / "mu_phase1_final.npy", record["phase1"]["mu_final"])
                save_array(cell / "mu_phase2_final.npy", record["phase2"]["mu_final"])
                write_json_atomic(cell / "vectors_phase1.json", vectors1)
                write_json_atomic(cell / "vectors_phase2.json", vectors2)
                phase1_json = dict(record["phase1"])
                phase2_json = dict(record["phase2"])
                phase1_json["mu_initial"] = None
                phase1_json["mu_final"] = None
                phase2_json["mu_initial"] = None
                phase2_json["mu_final"] = None
                write_json_atomic(cell / "distance_trajectories.json", {
                    "phase1": phase1_json, "phase2": phase2_json,
                    "phase1_rate": record["phase1_rate"],
                    "phase2_disrupted_rate": record["phase2_disrupted_rate"],
                })
                policy_json = dict(record)
                policy_json["phase1"] = phase1_json
                policy_json["phase2"] = phase2_json
                write_json_atomic(cell / "policy_record.json", policy_json)
                cells.append({k: v for k, v in record.items() if k not in {"phase1", "phase2"}} |
                             {"seed": seed, "epsilon_firm": epsilon})
    summary = {"schema_version": "hcurve-arm-c-v4", "created_utc": datetime.now(timezone.utc).isoformat(),
               "source_vectors": str(ROOT), "cells": cells, "policies": POLICIES}
    write_json_atomic(OUT / "summary.json", summary)
    files = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append({"path": str(path.relative_to(OUT)).replace(os.sep, "/"), "bytes": path.stat().st_size,
                          "sha256": sha256_file(path), "role": "arm_c_policy_artifact"})
    manifest = {"schema_version": "hcurve-arm-c-v4", "created_utc": datetime.now(timezone.utc).isoformat(),
                "runner_entry_point": str(Path(__file__).resolve()),
                "policy_variants": POLICIES, "epsilon_firm": C_EPSILONS,
                "detector": "No production change-point detector found; C2 uses conditional experiment-local rolling error spike",
                "frozen_apparatus_source": str(ROOT), "files": files, "cells": cells}
    write_json_atomic(OUT / "manifest.json", manifest)
    for item in files:
        path = OUT / item["path"]
        assert path.exists() and path.stat().st_size == item["bytes"] and sha256_file(path) == item["sha256"]
    assert len(cells) == len(SEEDS) * len(C_EPSILONS) * len(POLICIES)
    print(json.dumps({"cells": len(cells), "manifest": str(OUT / "manifest.json"), "hashes": "valid"}, indent=2))


if __name__ == "__main__":
    main()
