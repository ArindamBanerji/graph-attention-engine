from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GAE_ROOT = HERE.parents[1]
SDK_ROOT = GAE_ROOT.parent / "copilot-sdk"
sys.path.insert(0, str(GAE_ROOT))
sys.path.insert(0, str(SDK_ROOT))

_oracle_path = SDK_ROOT / "examples" / "jm_reference" / "oracle.py"
_oracle_spec = importlib.util.spec_from_file_location("hcurve_jm_oracle", _oracle_path)
if _oracle_spec is None or _oracle_spec.loader is None:
    raise ImportError(f"Cannot load oracle module: {_oracle_path}")
_oracle_module = importlib.util.module_from_spec(_oracle_spec)
sys.modules[_oracle_spec.name] = _oracle_module
_oracle_spec.loader.exec_module(_oracle_module)
GroundTruthOracle = _oracle_module.GroundTruthOracle
OracleConfig = _oracle_module.OracleConfig
from gae.profile_scorer import ProfileScorer
from gae.synthetic import CanonicalCentroid, FactorVectorSampler, OracleSeparationExperiment

SEEDS = [42, 123, 777]
EPSILONS = [0.05, 0.20, 0.35]
C, A, D = 6, 4, 6
DELTA_NORM = 0.25
ALPHA_DISRUPT = 2 / 6
THETA = 0.85
WINDOW = 10
SIGMA_NOISE = 0.08
ACTIONS = ["escalate", "investigate", "suppress", "monitor"]
RAW_ROOT = HERE / "raw"
TOL = 1e-10


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def array_payload(path: Path, label: str, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    meta = {"label": label, "shape": list(array.shape), "dtype": str(array.dtype)}
    write_json(path.with_suffix(".meta.json"), meta)


def json_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def reconstruct_raw_gt(oracle_seed: int, oracle_epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(oracle_seed)
    canonical = np.full((C, A, D), 0.5, dtype=float)
    directions = np.empty((C, A, D), dtype=float)
    scale = math.sqrt(C * A)
    raw = canonical.copy()
    for category in range(C):
        for action in range(A):
            direction = rng.standard_normal(D)
            direction /= max(float(np.linalg.norm(direction)), 1e-10)
            directions[category, action] = direction
            raw[category, action] += direction * oracle_epsilon * scale
    return raw, directions


def summarize_vectors(samples, gt1: np.ndarray, mu0: np.ndarray) -> dict:
    rows = []
    for sample in samples:
        vector = np.asarray(sample.f, dtype=float)
        gt_flat = gt1.reshape(-1, D)
        mu_flat = mu0.reshape(-1, D)
        gt_distances = np.linalg.norm(gt_flat - vector, axis=1)
        mu_distances = np.linalg.norm(mu_flat - vector, axis=1)
        rows.append(
            {
                "regime": sample.regime,
                "generation_seed": int(sample.generation_seed),
                "values": vector.tolist(),
                "sigma_per_factor": np.asarray(sample.sigma_per_factor).tolist(),
                "min_distance_to_gt1": float(gt_distances.min()),
                "mean_distance_to_gt1": float(gt_distances.mean()),
                "min_distance_to_mu0": float(mu_distances.min()),
                "mean_distance_to_mu0": float(mu_distances.mean()),
            }
        )
    by_regime = defaultdict(list)
    for row in rows:
        by_regime[row["regime"]].append(row)
    summary = {}
    for regime, regime_rows in by_regime.items():
        for key in ("min_distance_to_gt1", "min_distance_to_mu0"):
            values = np.asarray([r[key] for r in regime_rows], dtype=float)
            summary.setdefault(regime, {})[key] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
                "n": int(len(values)),
            }
    return {"rows": rows, "by_regime": summary}


def half_distance(distances: list[float]) -> int | None:
    if not distances:
        return None
    initial = float(distances[0])
    for index, distance in enumerate(distances[1:], start=2):
        if float(distance) <= 0.5 * initial:
            return index
    return None


def monotonicity(distances: list[float]) -> dict:
    violations = [
        {"index": i, "previous": float(distances[i - 1]), "current": float(distances[i])}
        for i in range(1, len(distances))
        if float(distances[i]) > float(distances[i - 1])
    ]
    return {"monotone_nonincreasing": not violations, "violations": violations}


def run() -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    epsilon_star = (ALPHA_DISRUPT * DELTA_NORM) / (1 - ALPHA_DISRUPT)
    assert abs(epsilon_star - 0.125) <= TOL
    assert EPSILONS[0] < epsilon_star < EPSILONS[1] < EPSILONS[2]

    canonical_prior = np.full((C, A, D), 0.5, dtype=float)
    cell_summaries = []
    expected_paths = []

    # Construction gate: build every seed's GT family and verify geometry/order first.
    gt_by_seed = {}
    for run_seed in SEEDS:
        oracle_seed = 99 + run_seed
        mismatches = []
        for epsilon in EPSILONS:
            oracle_epsilon = epsilon / (C * A)
            raw_gt, directions = reconstruct_raw_gt(oracle_seed, oracle_epsilon)
            oracle = GroundTruthOracle(
                OracleConfig(
                    seed=oracle_seed,
                    n_categories=C,
                    n_actions=A,
                    n_factors=D,
                    epsilon_firm=oracle_epsilon,
                    canonical_prior=canonical_prior,
                )
            )
            gt1 = oracle.ground_truth_centroids
            assert np.array_equal(canonical_prior, canonical_prior)
            assert np.allclose(gt1, raw_gt, atol=TOL, rtol=0.0), "GT construction/clipping mismatch"
            assert np.all((gt1 >= 0.0) & (gt1 <= 1.0))
            assert np.allclose(np.linalg.norm(directions, axis=-1), 1.0, atol=TOL)
            assert np.allclose(gt1, canonical_prior + (raw_gt - canonical_prior), atol=TOL)
            mismatch = float(np.linalg.norm(gt1 - canonical_prior))
            assert abs(mismatch - epsilon) <= TOL, f"raw mismatch {mismatch} != {epsilon}"
            mismatches.append(mismatch)
            gt_by_seed[(run_seed, epsilon)] = (gt1, directions, oracle_seed, oracle_epsilon)
        assert mismatches[0] < mismatches[1] < mismatches[2], mismatches

    for run_seed in SEEDS:
        vector_seed = run_seed
        sampler = FactorVectorSampler(
            d=D,
            sigma_profile=np.full(D, SIGMA_NOISE, dtype=float),
            seed=vector_seed,
        )
        for epsilon in EPSILONS:
            gt1, directions, oracle_seed, oracle_epsilon = gt_by_seed[(run_seed, epsilon)]
            gt2 = gt1.copy()
            delta = np.full((A, D), DELTA_NORM / math.sqrt(A * D), dtype=float)
            gt2[[0, 1]] = np.clip(gt2[[0, 1]] + delta, 0.0, 1.0)

            max_decisions = 600 if epsilon == 0.35 else 300
            phase1_samples = sampler.sample("cold_start", max_decisions)
            phase2_samples = sampler.sample("cold_start", max_decisions)

            cell_dir = RAW_ROOT / "runs" / f"seed-{run_seed:03d}" / f"epsilon-{epsilon:.2f}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            mu0 = canonical_prior.copy()
            scorer = ProfileScorer(mu=mu0.copy(), actions=ACTIONS)
            experiment = OracleSeparationExperiment(
                scorer=scorer,
                canonical_gt1=CanonicalCentroid.from_ground_truth(gt1),
                epsilon_firm=epsilon,
                disruption_magnitude=DELTA_NORM,
                disrupted_categories=[0, 1],
                alpha_cat=ALPHA_DISRUPT,
                window=WINDOW,
                theta=THETA,
                max_decisions=max_decisions,
            )
            phase1 = experiment.run_phase1(phase1_samples)
            mu_phase1_final = phase1.mu_final.copy()
            phase2 = experiment.run_phase2(phase2_samples, phase1)
            mu_phase2_final = scorer.centroids.copy()
            gamma = experiment.compute_gamma(phase1, phase2)

            array_payload(cell_dir / "centroids_canonical_prior.npy", "canonical_prior", canonical_prior)
            array_payload(cell_dir / "centroids_mu0.npy", "mu0", mu0)
            array_payload(cell_dir / "centroids_gt1.npy", "gt1", gt1)
            array_payload(cell_dir / "centroids_gt2.npy", "gt2", gt2)
            array_payload(cell_dir / "centroids_mu_phase1_final.npy", "mu_phase1_final", mu_phase1_final)
            array_payload(cell_dir / "centroids_mu_phase2_final.npy", "mu_phase2_final", mu_phase2_final)

            phase1_vectors = summarize_vectors(phase1_samples, gt1, mu0)
            phase2_vectors = summarize_vectors(phase2_samples, gt1, mu0)
            write_json(cell_dir / "vectors_phase1.json", phase1_vectors)
            write_json(cell_dir / "vectors_phase2.json", phase2_vectors)

            d1 = [float(x) for x in phase1.trace.centroid_distances]
            d2 = [float(x) for x in phase2.trace.centroid_distances]
            n1d, n2d = half_distance(d1), half_distance(d2)
            gamma_dist = None if n1d is None or n2d is None else float(n1d / n2d)
            mono1, mono2 = monotonicity(d1), monotonicity(d2)
            distance_record = {
                "phase1": {"target": "gt1", "distances": d1, "n_half_dist": n1d, **mono1},
                "phase2": {"target": "gt2", "distances": d2, "n_half_dist": n2d, **mono2},
            }
            write_json(cell_dir / "distance_trajectories.json", distance_record)

            vector_summary = {
                "phase1": phase1_vectors["by_regime"],
                "phase2": phase2_vectors["by_regime"],
                "starting_mismatch_raw": float(np.linalg.norm(gt1 - mu0)),
                "starting_mismatch_normalized": float(np.linalg.norm(gt1 - mu0) / math.sqrt(gt1.size)),
                "gt1_mu0_nonzero": bool(np.linalg.norm(gt1 - mu0) > 0.0),
                "inter_action_spacing_gt1_mean": float(np.mean([
                    np.linalg.norm(gt1[c, a] - gt1[c, b])
                    for c in range(C) for a in range(A) for b in range(a + 1, A)
                ])),
            }
            write_json(cell_dir / "vector_distance_summary.json", vector_summary)

            gamma_record = {
                "run_seed": run_seed,
                "vector_seed": vector_seed,
                "oracle_seed": oracle_seed,
                "epsilon_firm_configured": epsilon,
                "oracle_epsilon": oracle_epsilon,
                "starting_mismatch_raw": float(np.linalg.norm(gt1 - mu0)),
                "gamma_dist": gamma_dist,
                "n_half_dist_phase1": n1d,
                "n_half_dist_phase2": n2d,
                "gamma_nhalf": json_float(gamma.gamma),
                "n_half_1": gamma.n_half_1,
                "n_half_2": gamma.n_half_2,
                "phase1_dnf": phase1.dnf,
                "phase2_dnf": phase2.dnf,
                "epsilon_star": epsilon_star,
                "theorem_prediction": "gamma_gt_1" if epsilon > epsilon_star else "gamma_lt_1",
                "f1_fired": gamma_dist is not None and ((gamma_dist > 1) != (epsilon > epsilon_star)),
                "f2_fired": gamma_dist is not None and gamma.gamma is not None and ((gamma_dist > 1) != (gamma.gamma > 1)),
                "f3_fired": not mono1["monotone_nonincreasing"] or not mono2["monotone_nonincreasing"],
                "censored": n1d is None or n2d is None,
                "note": gamma.note,
            }
            write_json(cell_dir / "gamma.json", gamma_record)

            for path in cell_dir.rglob("*"):
                if path.is_file():
                    expected_paths.append(path)
            cell_summaries.append(gamma_record)

    # Write the scorer-free summary before manifest enumeration so it is hashed too.
    summary = {
        "epsilon_star": epsilon_star,
        "cells": cell_summaries,
        "non_centroidality": [],
        "persistence": {
            "manifest": str(RAW_ROOT / "manifest.json"),
            "all_expected_cells": len(cell_summaries) == len(SEEDS) * len(EPSILONS),
            "scorer_free_recompute": True,
        },
    }
    for seed in SEEDS:
        for epsilon in EPSILONS:
            cell_dir = RAW_ROOT / "runs" / f"seed-{seed:03d}" / f"epsilon-{epsilon:.2f}"
            for phase in ("phase1", "phase2"):
                payload = json.loads((cell_dir / f"vectors_{phase}.json").read_text(encoding="utf-8"))
                summary["non_centroidality"].append({
                    "seed": seed, "epsilon": epsilon, "phase": phase,
                    "by_regime": payload["by_regime"],
                })
    write_json(RAW_ROOT / "run_summary.json", summary)

    files = []
    for path in sorted(RAW_ROOT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append({
                "path": str(path.relative_to(RAW_ROOT)).replace(os.sep, "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": "experiment_artifact",
            })
    manifest = {
        "schema_version": "hcurve-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": str(GAE_ROOT),
        "runner_entry_point": str(Path(__file__).resolve()),
        "frozen_config": {
            "seeds": SEEDS,
            "epsilon_firm": EPSILONS,
            "C": C, "A": A, "d": D,
            "disrupted_categories": [0, 1],
            "alpha_disrupt": ALPHA_DISRUPT,
            "delta_norm": DELTA_NORM,
            "theta": THETA,
            "sigma_noise": SIGMA_NOISE,
            "window": WINDOW,
            "max_decisions": {"0.05": 300, "0.20": 300, "0.35": 600},
            "mu0_construction": "exact canonical_prior=np.full((6,4,6),0.5)",
            "mu0_seed_rule": "7*run_seed (unused; exact canonical mu0 has no RNG)",
            "vector_seed_rule": "run_seed",
            "oracle_seed_rule": "99+run_seed",
            "oracle_seeds": {"42": 141, "123": 222, "777": 876},
            "oracle_epsilon_rule": "epsilon_firm/(C*A)",
            "epsilon_star": epsilon_star,
        },
        "construction_gate": {
            "canonical_prior_identical_to_mu0": True,
            "raw_mismatch_tolerance": TOL,
            "no_clipping_required": True,
            "per_seed_epsilon_order_required": True,
        },
        "cells": cell_summaries,
        "files": files,
    }
    basis = dict(manifest)
    manifest["manifest_sha256_basis"] = sha256_bytes(
        json.dumps(basis, indent=2, sort_keys=True, ensure_ascii=False).encode()
    )
    manifest["files"].append({
        "path": "manifest.json",
        "bytes": 0,
        "sha256": manifest["manifest_sha256_basis"],
        "role": "manifest-self-hash-over-basis-without-self-entry",
    })
    self_entry = manifest["files"][-1]
    for _ in range(4):
        encoded = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").replace("\n", "\r\n").encode()
        actual_size = len(encoded)
        if self_entry["bytes"] == actual_size:
            break
        self_entry["bytes"] = actual_size
    temp = RAW_ROOT / "manifest.json.tmp"
    write_json(temp, manifest)
    temp.replace(RAW_ROOT / "manifest.json")

    summary["persistence"]["artifact_file_count"] = len(files)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
