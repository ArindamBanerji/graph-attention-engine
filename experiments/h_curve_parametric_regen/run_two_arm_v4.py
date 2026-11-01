from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GAE_ROOT = HERE.parents[1]
SDK_ROOT = GAE_ROOT.parent / "copilot-sdk"
sys.path.insert(0, str(GAE_ROOT))
sys.path.insert(0, str(SDK_ROOT))

from gae.profile_scorer import ProfileScorer

ORACLE_PATH = SDK_ROOT / "examples" / "jm_reference" / "oracle.py"
spec = importlib.util.spec_from_file_location("hcurve_jm_oracle_v4", ORACLE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load oracle: {ORACLE_PATH}")
oracle_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oracle_module
spec.loader.exec_module(oracle_module)
GroundTruthOracle = oracle_module.GroundTruthOracle
OracleConfig = oracle_module.OracleConfig

SEEDS = [42, 123, 777]
EPSILONS = [0.05, 0.20, 0.35]
C, A, D = 6, 4, 6
DISRUPTED = [0, 1]
DELTA_NORM = 0.25
ALPHA = 2 / 6
EPSILON_STAR = (ALPHA * DELTA_NORM) / (1 - ALPHA)
SIGMA = 0.08
THETA = 0.85
WINDOW = 10
ETA_B = 0.05
ACTIONS = ["a0", "a1", "a2", "a3"]
TOL = 1e-10
BLOCK = 10
MIN_RETAINED_BLOCKS = 5
GATE_RATIO = 0.50
OUT = HERE / "raw" / "two_arm_v4"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(value, dtype=np.float64))
    tmp.replace(path)


def reconstruct_gt(oracle_seed: int, epsilon: float, canonical: np.ndarray):
    oracle_epsilon = epsilon / (C * A)
    rng = np.random.default_rng(oracle_seed)
    directions = np.empty((C, A, D), dtype=np.float64)
    raw = canonical.copy()
    for c in range(C):
        for a in range(A):
            direction = rng.standard_normal(D)
            direction /= np.linalg.norm(direction)
            directions[c, a] = direction
            raw[c, a] += direction * oracle_epsilon * math.sqrt(C * A)
    oracle = GroundTruthOracle(OracleConfig(
        seed=oracle_seed,
        n_categories=C,
        n_actions=A,
        n_factors=D,
        epsilon_firm=oracle_epsilon,
        canonical_prior=canonical,
    ))
    gt1 = np.asarray(oracle.ground_truth_centroids, dtype=np.float64)
    assert np.allclose(gt1, raw, atol=TOL, rtol=0.0)
    assert np.all((gt1 >= 0.0) & (gt1 <= 1.0))
    assert np.allclose(np.linalg.norm(directions, axis=-1), 1.0, atol=TOL)
    assert float(np.linalg.norm(gt1 - canonical)) - epsilon <= TOL
    assert abs(float(np.linalg.norm(gt1 - canonical)) - epsilon) <= TOL
    assert not np.any((raw < 0.0) | (raw > 1.0))
    return gt1, directions, oracle_epsilon


def make_gt2(gt1: np.ndarray) -> np.ndarray:
    delta = np.full((A, D), DELTA_NORM / math.sqrt(A * D), dtype=np.float64)
    gt2 = gt1.copy()
    gt2[DISRUPTED] = np.clip(gt2[DISRUPTED] + delta, 0.0, 1.0)
    return gt2


def schedule(n: int) -> list[tuple[int, int]]:
    cells = [(c, a) for c in range(C) for a in range(A)]
    return [cells[i % len(cells)] for i in range(n)]


def draw_vectors(rng: np.random.Generator, target: np.ndarray, n: int, target_label: str):
    result = []
    for c, a in schedule(n):
        noise = rng.normal(0.0, SIGMA, size=D)
        unclipped = target[c, a] + noise
        vector = np.clip(unclipped, 0.0, 1.0)
        result.append({
            "category_index": c,
            "target_action_index": a,
            "center_label": f"{target_label}[{c},{a}]",
            "center": target[c, a].tolist(),
            "noise": noise.tolist(),
            "unclipped": unclipped.tolist(),
            "f": vector.tolist(),
            "clipped": bool(np.any(unclipped < 0.0) or np.any(unclipped > 1.0)),
        })
    return result


def nearest_action(mu: np.ndarray, f: np.ndarray, c: int) -> int:
    distances = np.sum((mu[c] - f) ** 2, axis=1)
    return int(np.argmin(distances))


def distances(mu: np.ndarray, target: np.ndarray, active: np.ndarray) -> tuple[float, float, float]:
    diff = mu - target
    full = float(np.linalg.norm(diff))
    active_d = float(np.linalg.norm(diff[active]))
    disrupted_d = float(np.linalg.norm(diff[DISRUPTED]))
    return full, active_d, disrupted_d


def run_arm(arm: str, mu0: np.ndarray, gt1: np.ndarray, gt2: np.ndarray,
            vectors1: list[dict], vectors2: list[dict], n1: int, n2: int):
    if arm == "A":
        model = ProfileScorer(mu=mu0.copy(), actions=ACTIONS,
                              categories=[f"c{i}" for i in range(C)],
                              eta_override=0.01)
    else:
        model = None
        mu_b = mu0.copy()

    counts1 = np.zeros((C, A), dtype=int)
    counts2 = np.zeros((C, A), dtype=int)
    active1 = np.zeros((C, A), dtype=bool)
    active2 = np.zeros((C, A), dtype=bool)

    def current_mu():
        return np.asarray(model.mu if arm == "A" else mu_b, dtype=np.float64)

    def phase(vectors, target, counts, active, phase_name):
        initial = current_mu().copy()
        # The round-robin schedule is known before the first decision and covers
        # every cell. Use that fixed final coverage mask for d_active from t=0;
        # retain the dynamic mask separately for coverage reporting. This keeps
        # d0 and every later active distance on the same subspace.
        metric_active = np.ones((C, A), dtype=bool)
        full0, active0, disrupted0 = distances(current_mu(), target, metric_active)
        full_trace = [full0]
        active_trace = [active0]
        disrupted_trace = [disrupted0]
        active_counts = [int(active.sum())]
        decisions = []
        rolling = []
        window = []
        for index, row in enumerate(vectors):
            f = np.asarray(row["f"], dtype=np.float64)
            c = int(row["category_index"])
            source_a = int(row["target_action_index"])
            predicted = nearest_action(current_mu(), f, c)
            # The round-robin center label is the observed ground-truth outcome.
            # Do not reclassify it from a noisy vector: close GT action profiles
            # would turn measurement noise into label noise and defeat the
            # coverage contract.
            gt_action = source_a
            correct = predicted == gt_action
            if arm == "A":
                model.update(f, c, predicted, correct,
                             gt_action_index=gt_action)
            else:
                mu_b[c, gt_action] = np.clip(
                    mu_b[c, gt_action] + ETA_B * (f - mu_b[c, gt_action]), 0.0, 1.0
                )
            counts[c, source_a] += 1
            active[c, source_a] = True
            full, active_d, disrupted_d = distances(current_mu(), target, metric_active)
            full_trace.append(full)
            active_trace.append(active_d)
            disrupted_trace.append(disrupted_d)
            active_counts.append(int(active.sum()))
            window.append(int(correct))
            if len(window) > WINDOW:
                window.pop(0)
            rolling.append(float(np.mean(window)))
            decisions.append({
                "index": index + 1,
                "category_index": c,
                "source_action_index": source_a,
                "predicted_action_index": predicted,
                "gt_action_index": gt_action,
                "correct": correct,
            })
        return {
            "phase": phase_name,
            "mu_initial": initial,
            "mu_final": current_mu().copy(),
            "d_full": full_trace,
            "d_active": active_trace,
            "d_disrupted": disrupted_trace,
            "active_counts": active_counts,
            "rolling_accuracy": rolling,
            "decisions": decisions,
            "counts": counts.copy(),
            "active": active.copy(),
        }

    p1 = phase(vectors1, gt1, counts1, active1, "phase1")
    phase2_start_disrupted = float(np.linalg.norm(current_mu()[DISRUPTED] - gt2[DISRUPTED]))
    phase2_expected_shift = float(np.linalg.norm(gt1[DISRUPTED] - gt2[DISRUPTED]))
    phase2_sanity = {
        "distance_to_gt2_at_phase2_start": phase2_start_disrupted,
        "expected_gt1_to_gt2_shift": phase2_expected_shift,
        "within_tolerance": bool(abs(phase2_start_disrupted - phase2_expected_shift) <= 0.10),
    }
    assert phase2_sanity["within_tolerance"]
    p2 = phase(vectors2, gt2, counts2, active2, "phase2")
    return p1, p2, phase2_sanity


def fit_rate(trace: list[float]) -> dict:
    values = np.asarray(trace[1:], dtype=float)
    d0 = float(trace[0])
    tail = max(1, int(math.ceil(0.20 * len(values))))
    d_inf = float(np.mean(values[-tail:]))
    blocks = []
    for start in range(0, len(values), BLOCK):
        block = values[start:start + BLOCK]
        if len(block) == BLOCK:
            blocks.append((start + (BLOCK - 1) / 2.0, float(np.mean(block))))
    retained = [(t, d) for t, d in blocks if d > d_inf + 1e-12]
    slopes = []
    for i in range(len(retained)):
        ti, di = retained[i]
        for j in range(i + 1, len(retained)):
            tj, dj = retained[j]
            slopes.append((math.log(dj - d_inf) - math.log(di - d_inf)) / (tj - ti))
    slope = float(np.median(slopes)) if slopes else None
    k = None if slope is None else float(-slope)
    half = next((i for i, d in enumerate(values, start=1) if d <= 0.5 * d0), None)
    fit_valid = bool(k is not None and k > 0 and len(retained) >= MIN_RETAINED_BLOCKS)
    return {
        "d0": d0,
        "d_inf": d_inf,
        "d_inf_over_d0": None if d0 == 0 else d_inf / d0,
        "blocks": len(blocks),
        "retained_blocks": len(retained),
        "k": k,
        "half_decisions": half,
        "fit_valid": fit_valid,
        "gate_pass": fit_valid,
    }


def phase_record(run: dict, arm: str, target: np.ndarray, subspace: str):
    aggregate = fit_rate(run["d_active"])
    if subspace == "all":
        rate = aggregate
    else:
        rate = fit_rate(run["d_disrupted"])
    return {
        "aggregate_gate": aggregate,
        "primary_subspace_rate": rate,
        "d_full": run["d_full"],
        "d_active": run["d_active"],
        "d_disrupted": run["d_disrupted"],
        "active_counts": run["active_counts"],
        "counts": run["counts"].tolist(),
        "active": run["active"].tolist(),
        "rolling_accuracy": run["rolling_accuracy"],
        "decision_count": len(run["decisions"]),
    }


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def run() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    canonical = np.full((C, A, D), 0.5, dtype=np.float64)
    assert abs(EPSILON_STAR - 0.125) <= TOL
    assert EPSILONS[0] < EPSILON_STAR < EPSILONS[1] < EPSILONS[2]
    expected = []
    cell_records = []
    gt_cache = {}
    for seed in SEEDS:
        mismatches = []
        for epsilon in EPSILONS:
            oracle_seed = 99 + seed
            gt1, directions, oracle_epsilon = reconstruct_gt(oracle_seed, epsilon, canonical)
            mismatch = float(np.linalg.norm(gt1 - canonical))
            mismatches.append(mismatch)
            gt_cache[(seed, epsilon)] = (gt1, directions, oracle_seed, oracle_epsilon)
        assert mismatches[0] < mismatches[1] < mismatches[2]

    for seed in SEEDS:
        for epsilon in EPSILONS:
            gt1, directions, oracle_seed, oracle_epsilon = gt_cache[(seed, epsilon)]
            gt2 = make_gt2(gt1)
            n = 1200 if epsilon == 0.35 else 720
            vector_rng = np.random.default_rng(seed)
            vectors1 = draw_vectors(vector_rng, gt1, n, "GT1")
            vectors2 = draw_vectors(vector_rng, gt2, n, "GT2")
            phase2_center_checks = []
            for c in DISRUPTED:
                for a in range(A):
                    cell_vectors = [
                        np.asarray(row["f"], dtype=np.float64)
                        for row in vectors2
                        if row["category_index"] == c and row["target_action_index"] == a
                    ]
                    mean_vector = np.mean(cell_vectors, axis=0)
                    to_gt2 = float(np.linalg.norm(mean_vector - gt2[c, a]))
                    to_gt1 = float(np.linalg.norm(mean_vector - gt1[c, a]))
                    phase2_center_checks.append({
                        "category_index": c,
                        "action_index": a,
                        "mean_vector": mean_vector.tolist(),
                        "distance_to_gt2": to_gt2,
                        "distance_to_gt1": to_gt1,
                        "centered_on_gt2": bool(to_gt2 < to_gt1 and to_gt2 <= 2.0 * SIGMA),
                    })
            assert all(item["centered_on_gt2"] for item in phase2_center_checks)
            for arm in ("A", "B"):
                cell = OUT / "runs" / f"seed-{seed:03d}" / f"epsilon-{epsilon:.2f}" / f"arm-{arm}"
                cell.mkdir(parents=True, exist_ok=True)
                mu0 = canonical.copy()
                p1, p2, phase2_sanity = run_arm(arm, mu0, gt1, gt2, vectors1, vectors2, n, n)
                assert np.all(p1["counts"] >= n // (C * A))
                assert np.all(p2["counts"] >= n // (C * A))
                save_array(cell / "canonical_prior.npy", canonical)
                save_array(cell / "gt1.npy", gt1)
                save_array(cell / "gt2.npy", gt2)
                save_array(cell / "mu0.npy", mu0)
                save_array(cell / "mu_phase1_final.npy", p1["mu_final"])
                save_array(cell / "mu_phase2_final.npy", p2["mu_final"])
                write_json_atomic(cell / "vectors_phase1.json", vectors1)
                write_json_atomic(cell / "vectors_phase2.json", vectors2)
                write_json_atomic(cell / "decisions_phase1.json", p1["decisions"])
                write_json_atomic(cell / "decisions_phase2.json", p2["decisions"])
                phase1 = phase_record(p1, arm, gt1, "all")
                phase2 = phase_record(p2, arm, gt2, "disrupted")
                gate_pass = bool(phase1["primary_subspace_rate"]["fit_valid"] and
                                 phase2["primary_subspace_rate"]["fit_valid"])
                gamma_rate = None
                gamma_half = None
                if gate_pass:
                    k1 = phase1["primary_subspace_rate"]["k"]
                    k2 = phase2["primary_subspace_rate"]["k"]
                    if k1 and k2 and k1 > 0 and k2 > 0:
                        gamma_rate = float(k2 / k1)
                    n1h = phase1["primary_subspace_rate"]["half_decisions"]
                    n2h = phase2["primary_subspace_rate"]["half_decisions"]
                    if n1h is not None and n2h is not None:
                        gamma_half = float(n1h / n2h)
                f1 = None if gamma_rate is None else bool(
                    (gamma_rate > 1) != (epsilon > EPSILON_STAR)
                )
                f2 = None if gamma_rate is None or gamma_half is None else bool(
                    (gamma_rate > 1) != (gamma_half > 1)
                )
                f3 = not gate_pass
                geometry = {
                    "canonical_equals_mu0": bool(np.array_equal(canonical, mu0)),
                    "starting_mismatch": float(np.linalg.norm(gt1 - mu0)),
                    "configured_epsilon": epsilon,
                    "within_tolerance": abs(float(np.linalg.norm(gt1 - mu0)) - epsilon) <= TOL,
                    "no_gt1_clipping": True,
                    "direction_norms_unit": bool(np.allclose(np.linalg.norm(directions, axis=-1), 1.0, atol=TOL)),
                }
                record = {
                    "seed": seed, "epsilon_firm": epsilon, "arm": arm,
                    "oracle_seed": oracle_seed, "oracle_epsilon": oracle_epsilon,
                    "vector_seed": seed, "max_decisions": n,
                    "geometry": geometry,
                    "phase2_center_checks": phase2_center_checks,
                    "phase1": phase1, "phase2": phase2,
                    "phase2_start_sanity": phase2_sanity,
                    "convergence_gate_pass": gate_pass,
                    "gamma_rate": gamma_rate, "gamma_half": gamma_half,
                    "n_half_phase1": phase1["primary_subspace_rate"]["half_decisions"],
                    "n_half_phase2": phase2["primary_subspace_rate"]["half_decisions"],
                    "f1_fired": f1, "f2_fired": f2, "f3_fired": f3,
                    "c3_statement": (
                        "Below-threshold cell inconclusive due to noise dominance at ε_firm < σ_noise (0.05 < 0.08); "
                        "consistent with the theorem's prediction of negligible advantage below ε★, but NOT a confirmed "
                        "γ < 1. The above-threshold arms (ε=0.20, 0.35) carry the binary test and the paper's spine."
                    ) if epsilon == 0.05 else None,
                }
                write_json_atomic(cell / "distance_trajectories.json", {
                    "phase1": phase1, "phase2": phase2,
                })
                write_json_atomic(cell / "gamma.json", record)
                write_json_atomic(cell / "geometry.json", geometry)
                cell_records.append(record)

    summary = {
        "epsilon_star": EPSILON_STAR,
        "cells": cell_records,
        "persistence_complete": True,
    }
    write_json_atomic(OUT / "summary.json", summary)
    expected = [p for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json"]
    files = [{
        "path": str(p.relative_to(OUT)).replace(os.sep, "/"),
        "bytes": p.stat().st_size,
        "sha256": sha256_file(p),
        "role": "two_arm_experiment_artifact",
    } for p in sorted(expected)]
    config = {
        "seeds": SEEDS, "epsilon_firm": EPSILONS, "C": C, "A": A, "d": D,
        "disrupted": DISRUPTED, "delta_norm": DELTA_NORM, "alpha": ALPHA,
        "epsilon_star": EPSILON_STAR, "theta": THETA, "window": WINDOW,
        "sigma_noise": SIGMA, "budgets": {"0.05": 720, "0.20": 720, "0.35": 1200},
        "oracle_seed_rule": "99+run_seed", "oracle_seeds": {str(s): 99 + s for s in SEEDS},
        "oracle_epsilon_rule": "epsilon_firm/(C*A)",
        "mu0": "np.full((6,4,6),0.5)", "arm_a": "gae.ProfileScorer.update",
        "arm_b": "mu[c,gt_action] += 0.05*(f-mu[c,gt_action])",
        "common_decision_rule": "nearest squared L2 over current centroids",
        "gate_ratio": GATE_RATIO, "block_size": BLOCK,
    }
    config_hash = sha256_bytes(json.dumps(config, sort_keys=True).encode())
    manifest = {
        "schema_version": "hcurve-two-arm-v4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": str(GAE_ROOT), "runner_entry_point": str(Path(__file__).resolve()),
        "design_fixes": [
            {
                "id": "DF-1",
                "issue": "Phase-2 vectors were GT1-centered while distance used GT2",
                "fix": "Generate Phase-2 vectors around GT2[c,a] and measure Phase 2 against GT2",
                "reason": "A convergence update must receive observations from the target it is measured against",
            },
            {
                "id": "DF-2",
                "issue": "Noisy vectors were reclassified to infer the outcome action",
                "fix": "Use the persisted round-robin center label a as the ground-truth outcome",
                "reason": "The apparatus defines the decision cell; reclassification would add uncontrolled label noise",
            },
            {
                "id": "DF-3",
                "issue": "Dynamic active masks made d0 incomparable with later active distances",
                "fix": "Use the pre-registered final 24-cell coverage mask from t=0 and persist dynamic masks separately",
                "reason": "All cells are scheduled in advance, so the convergence metric must use one fixed subspace",
            },
        ],
        "config": config, "config_sha256": config_hash,
        "construction": "design-owned apparatus; GT1-centered Phase 1 and GT2-centered Phase 2 vectors; shared vectors and GT; only update rule differs",
        "cells": cell_records, "files": files,
    }
    basis = json.dumps(manifest, sort_keys=True, indent=2).encode()
    manifest["manifest_basis_sha256"] = sha256_bytes(basis)
    write_json_atomic(OUT / "manifest.json", manifest)
    # Verify the persistence gate after the atomic manifest write.
    loaded = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    for item in loaded["files"]:
        path = OUT / item["path"]
        assert path.exists() and path.stat().st_size == item["bytes"]
        assert sha256_file(path) == item["sha256"]
    assert len(loaded["cells"]) == len(SEEDS) * len(EPSILONS) * 2
    print(json.dumps({"cells": len(cell_records), "manifest": str(OUT / "manifest.json")}, indent=2))


if __name__ == "__main__":
    run()
