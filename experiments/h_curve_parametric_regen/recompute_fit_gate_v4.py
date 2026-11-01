from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "raw" / "two_arm_v4"
MANIFEST = ROOT / "manifest.json"
OUT = ROOT / "fit_gate_recompute.json"
EPSILON_STAR = 0.125
SEEDS = [42, 123, 777]
EPSILONS = [0.05, 0.20, 0.35]
ARMS = ["A", "B"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_atomic(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cells = []
    for seed in SEEDS:
        for epsilon in EPSILONS:
            for arm in ARMS:
                cell = ROOT / "runs" / f"seed-{seed:03d}" / f"epsilon-{epsilon:.2f}" / f"arm-{arm}"
                trajectories = json.loads((cell / "distance_trajectories.json").read_text(encoding="utf-8"))
                p1 = trajectories["phase1"]["primary_subspace_rate"]
                p2 = trajectories["phase2"]["primary_subspace_rate"]
                valid1 = p1["k"] is not None and p1["k"] > 0 and p1["retained_blocks"] >= 5
                valid2 = p2["k"] is not None and p2["k"] > 0 and p2["retained_blocks"] >= 5
                valid = bool(valid1 and valid2)
                gamma = None if not valid else float(p2["k"] / p1["k"])
                n1 = p1["half_decisions"]
                n2 = p2["half_decisions"]
                gamma_half = None if n1 is None or n2 is None else float(n1 / n2)
                f1 = None if gamma is None else bool((gamma > 1.0) != (epsilon > EPSILON_STAR))
                f2 = None if gamma is None or gamma_half is None else bool((gamma > 1.0) != (gamma_half > 1.0))
                cells.append({
                    "seed": seed,
                    "epsilon_firm": epsilon,
                    "arm": arm,
                    "fit_valid_phase1": valid1,
                    "fit_valid_phase2": valid2,
                    "fit_valid_gate": valid,
                    "k_phase1_all_cells": p1["k"],
                    "k_phase2_disrupted_cells": p2["k"],
                    "gamma_rate": gamma,
                    "n_half_phase1": n1,
                    "n_half_phase2": n2,
                    "gamma_half": gamma_half,
                    "d_inf_over_d0_phase1": p1["d_inf_over_d0"],
                    "d_inf_over_d0_phase2": p2["d_inf_over_d0"],
                    "f1_fired": f1,
                    "f2_fired": f2,
                    "f3_fired": not valid,
                })
    result = {
        "schema_version": "hcurve-fit-gate-recompute-v4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "persisted distance_trajectories.json only; no scorer imported; no rerun",
        "gate": {"type": "fit_validity_only", "k_gt": 0.0, "min_retained_blocks": 5,
                 "absolute_plateau_gate_used": False},
        "cells": cells,
    }
    write_atomic(OUT, result)
    entry = {
        "path": str(OUT.relative_to(ROOT)).replace(os.sep, "/"),
        "bytes": OUT.stat().st_size,
        "sha256": sha256_file(OUT),
        "role": "scorer_free_fit_validity_gamma_recompute",
    }
    manifest["design_fixes"] = [*manifest.get("design_fixes", []), {
        "id": "DF-4",
        "issue": "Absolute d_inf/d0 gate censored valid scale-free rate comparisons",
        "fix": "Gate only positive decay fit with at least five retained blocks; keep plateau descriptive",
        "reason": "Gamma is a ratio of rates and must not be censored by an a priori absolute depth",
    }]
    manifest["fit_gate_recompute"] = {
        "path": entry["path"],
        "absolute_plateau_gate_used": False,
        "scorer_free": True,
    }
    manifest["files"] = [item for item in manifest.get("files", []) if item["path"] != entry["path"]]
    manifest["files"].append(entry)
    write_atomic(MANIFEST, manifest)
    # Fail closed on every listed artifact after adding the recomputation.
    for item in manifest["files"]:
        path = ROOT / item["path"]
        assert path.exists()
        assert path.stat().st_size == item["bytes"]
        assert sha256_file(path) == item["sha256"]
    assert len(cells) == 18
    print(json.dumps({"cells": len(cells), "fit_gate_recompute": str(OUT), "hashes": "valid"}, indent=2))


if __name__ == "__main__":
    main()
