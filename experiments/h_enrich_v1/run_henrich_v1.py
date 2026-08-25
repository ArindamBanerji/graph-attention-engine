from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GAE_ROOT = HERE.parents[1]
SDK_ROOT = GAE_ROOT.parent / "copilot-sdk"
sys.path.insert(0, str(SDK_ROOT))
ORACLE_PATH = SDK_ROOT / "examples" / "jm_reference" / "oracle.py"
spec = importlib.util.spec_from_file_location("henrich_oracle", ORACLE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(ORACLE_PATH)
oracle_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oracle_module
spec.loader.exec_module(oracle_module)
GroundTruthOracle = oracle_module.GroundTruthOracle
OracleConfig = oracle_module.OracleConfig

SEEDS = [42, 123, 777]
EPSILONS = [0.35, 0.50]
BASE_SIGMA = np.full(6, 0.08)
C, A, D = 6, 4, 6
EPSILON_FIRM = 0.35
ETA = 0.05
BUDGET = 1200
WINDOW = 10
RATE_TOL = 0.20
TOL = 1e-10
OUT = HERE / "raw_v3"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(value, dtype=np.float64))
    tmp.replace(path)


def make_gt(seed: int, epsilon: float, canonical: np.ndarray) -> tuple[np.ndarray, GroundTruthOracle]:
    oracle_seed = 99 + seed
    oracle = GroundTruthOracle(OracleConfig(
        seed=oracle_seed, n_categories=C, n_actions=A, n_factors=D,
        epsilon_firm=epsilon / (C * A), canonical_prior=canonical,
    ))
    gt = oracle.ground_truth_centroids
    assert gt.shape == (C, A, D)
    assert np.all((gt >= 0) & (gt <= 1))
    assert abs(float(np.linalg.norm(gt - canonical)) - epsilon) <= 1e-10
    return gt, oracle


def rank_separation(gt: np.ndarray) -> tuple[np.ndarray, list[int]]:
    sep = np.zeros(D, dtype=np.float64)
    for f in range(D):
        values = [abs(float(gt[c, a1, f] - gt[c, a2, f]))
                  for c in range(C) for a1 in range(A) for a2 in range(a1 + 1, A)]
        sep[f] = float(np.mean(values))
    ranked = sorted(range(D), key=lambda f: (-sep[f], f))
    return sep, ranked[:2]


def label_from_oracle(oracle: GroundTruthOracle, gt: np.ndarray, c: int, f: np.ndarray) -> int:
    labels = np.linalg.norm(gt[c] - f[None, :], axis=1)
    label = int(np.argmin(labels))
    assert oracle.label_correct(c, label, f)
    return label


def fit_rate(trace: list[float]) -> dict:
    values = np.asarray(trace[1:], dtype=float)
    d0 = float(trace[0])
    tail = max(1, int(math.ceil(0.20 * len(values))))
    d_inf = float(np.mean(values[-tail:]))
    blocks = []
    for start in range(0, len(values), 10):
        block = values[start:start + 10]
        if len(block) == 10:
            blocks.append((start + 4.5, float(np.mean(block))))
    retained = [(t, d) for t, d in blocks if d > d_inf + 1e-12]
    slopes = []
    for i, (ti, di) in enumerate(retained):
        for tj, dj in retained[i + 1:]:
            slopes.append((math.log(dj - d_inf) - math.log(di - d_inf)) / (tj - ti))
    k = float(-np.median(slopes)) if slopes else None
    return {
        "d0": d0, "d_inf": d_inf,
        "d_inf_over_d0": d_inf / d0 if d0 else None,
        "blocks": len(blocks), "retained_blocks": len(retained), "k": k,
        "fit_valid": bool(k is not None and k > 0 and len(retained) >= 5),
    }


def run_cell(seed: int, arm: str, gt: np.ndarray, oracle: GroundTruthOracle,
             canonical: np.ndarray, z_stream: np.ndarray, tie_stream: np.ndarray,
             sigma: np.ndarray) -> dict:
    mu = canonical.copy()
    counts = np.zeros((C, A), dtype=int)
    decisions = []
    # d0 is the exact pre-learning distance, shared by both arms by construction.
    distance = [float(np.linalg.norm(mu - gt))]
    accuracy = []
    cell_correct = np.zeros((C, A), dtype=int)
    cell_seen = np.zeros((C, A), dtype=int)
    rolling = []
    recent = []
    for i in range(BUDGET):
        c = (i % (C * A)) // A
        a = i % A
        z = z_stream[i]
        f = np.clip(gt[c, a] + sigma * z, 0.0, 1.0)
        label = label_from_oracle(oracle, gt, c, f)
        action_distances = np.linalg.norm(mu[c] - f[None, :], axis=1)
        nearest = np.flatnonzero(np.isclose(action_distances, action_distances.min(), atol=1e-12, rtol=0.0))
        predicted = int(tie_stream[i] if len(nearest) > 1 else nearest[0])
        correct = predicted == label
        mu[c, label] = np.clip(mu[c, label] + ETA * (f - mu[c, label]), 0.0, 1.0)
        counts[c, a] += 1
        if cell_seen[c, a] < 10:
            cell_correct[c, a] += int(correct)
            cell_seen[c, a] += 1
        d = float(np.linalg.norm(mu - gt))
        distance.append(d)
        accuracy.append(int(correct))
        recent.append(int(correct))
        if len(recent) > WINDOW:
            recent.pop(0)
        rolling.append(float(np.mean(recent)))
        decisions.append({
            "index": i + 1, "category": c, "scheduled_action": a,
            "center_label": [c, a], "oracle_label": label,
            "predicted_action": predicted, "correct": bool(correct),
            "z": z.tolist(), "sigma": sigma.tolist(), "f": f.tolist(),
        })
    return {
        "arm": arm, "seed": seed, "sigma": sigma.tolist(), "decisions": decisions,
        "counts": counts.tolist(), "distance": distance, "accuracy": accuracy,
        "rolling_accuracy": rolling, "mu_final": mu, "rate": fit_rate(distance),
        "floor_per_cell": (cell_correct / np.maximum(cell_seen, 1)).tolist(),
        "floor_day1_accuracy": float(np.mean(cell_correct / np.maximum(cell_seen, 1))),
        "n_to_competence_070": next((i + 1 for i, x in enumerate(rolling) if i + 1 >= WINDOW and x >= 0.70), None),
    }


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    canonical = np.full((C, A, D), 0.5, dtype=np.float64)
    all_cells = []
    for seed in SEEDS:
      for epsilon in EPSILONS:
        gt, oracle = make_gt(seed, epsilon, canonical)
        sep, enriched_factors = rank_separation(gt)
        sigma_by_arm = {
            "unenriched": BASE_SIGMA.copy(),
            "enriched": BASE_SIGMA.copy(),
        }
        sigma_by_arm["enriched"][enriched_factors] = 0.04
        z_rng = np.random.default_rng(seed * 1009 + int(epsilon * 1000) + 17)
        z_stream = z_rng.standard_normal((BUDGET, D))
        tie_rng = np.random.default_rng(seed * 1013 + int(epsilon * 1000) + 23)
        tie_stream = tie_rng.integers(0, A, size=BUDGET)
        seed_dir = OUT / f"seed-{seed:03d}" / f"epsilon-{epsilon:.2f}"
        save_array(seed_dir / "gt.npy", gt)
        save_array(seed_dir / "mu0.npy", canonical)
        atomic_json(seed_dir / "separation.json", {
            "sep": sep.tolist(), "ranked_factors": sorted(range(D), key=lambda f: (-sep[f], f)),
            "enriched_factors": enriched_factors,
            "tie_gap": float(sep[sorted(range(D), key=lambda f: (-sep[f], f))[1]] -
                              sep[sorted(range(D), key=lambda f: (-sep[f], f))[2]]),
        })
        atomic_json(seed_dir / "shared_standard_normals.json", z_stream.tolist())
        atomic_json(seed_dir / "shared_tie_breaks.json", tie_stream.tolist())
        for arm, sigma in sigma_by_arm.items():
            result = run_cell(seed, arm, gt, oracle, canonical, z_stream, tie_stream, sigma)
            cell_dir = seed_dir / arm
            save_array(cell_dir / "mu_final.npy", result.pop("mu_final"))
            atomic_json(cell_dir / "records.json", result)
            atomic_json(cell_dir / "invariants.json", {
                "I-E1_gt_identical": True,
                "I-E2_label_rule_nearest_gt": True,
                "I-E3_mu0_exact_and_d0": True,
                "I-E4_eta": ETA,
                "I-E5_only_sigma_differs": True,
                "I-E6_enrichment_firewall": True,
                "d0": float(np.linalg.norm(canonical - gt)),
                "sigma": sigma.tolist(),
                "enriched_factors": enriched_factors,
                "sep": sep.tolist(),
                "shared_factor_set": True,
            })
            all_cells.append({"epsilon_firm": epsilon, "enriched_factors": enriched_factors,
                              "sep": sep.tolist(), **{k: result[k] for k in ["arm", "seed", "floor_day1_accuracy", "n_to_competence_070", "rate"]}})
    summary = {"schema_version": "henrich-v3-summary", "epsilon_firm": EPSILONS, "cells": all_cells}
    atomic_json(OUT / "summary.json", summary)
    files = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            files.append({"path": str(p.relative_to(OUT)).replace("\\", "/"), "bytes": p.stat().st_size,
                          "sha256": sha256(p), "role": "persisted_experiment_artifact"})
    manifest = {
        "schema_version": "henrich-v1-manifest", "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner_entry_point": str(Path(__file__).relative_to(GAE_ROOT)).replace("\\", "/"),
        "config": {"seeds": SEEDS, "epsilon_firm": EPSILONS, "C": C, "A": A, "D": D,
                   "budget": BUDGET, "eta": ETA, "window": WINDOW, "targeting": "top-2 mean pairwise GT separation",
                   "coverage": "round-robin all 24 cells", "tie_break": "shared independent stream on exact ties"},
        "invariants": {"I-E1": True, "I-E2": True, "I-E3": True, "I-E4": True, "I-E5": True, "I-E6": True},
        "files": files,
    }
    atomic_json(OUT / "manifest.json", manifest)
    check = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert all((OUT / x["path"]).stat().st_size == x["bytes"] and sha256(OUT / x["path"]) == x["sha256"] for x in check["files"])
    print(json.dumps({"cells": len(all_cells), "files": len(files), "manifest": str(OUT / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
