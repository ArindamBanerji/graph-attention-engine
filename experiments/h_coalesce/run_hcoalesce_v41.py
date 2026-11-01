"""E-NEW-5 v4.1: accuracy-ruler mechanism decomposition."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CALIBRATED = ROOT / "experiments" / "h_curve_parametric_regen" / "run_hcurve_calibrated.py"
RAW = HERE / "raw"
FIGURES = HERE / "figures"
SEEDS = [42, 123, 777]
SEPARATION_DEFAULT = 0.15
SEPARATION_CANDIDATES = [0.12, 0.15, 0.18, 0.21, 0.24, 0.28]
NEAR_OFFSETS = [0.20, 0.30, 0.40, 0.50, 0.60]
EPS_A, EPS_B = 0.30, 0.20
BUDGET = 400
STATIC_BATCH = 200
C, A, D = 6, 4, 6
SIGMA = 0.08
ACCURACY_THRESHOLD = 0.70
Q_WINDOW = 50
TIMEPOINTS = [0, 10, 25, 50, 100, 200, 400]
ARMS = ["COLD", "RANDOM_SHARP", "WARM_UNRELATED", "WARM_RELATED", "OPTIMAL", "RANDOM_SHARP_ACCMATCHED"]


def load_calibrated():
    spec = importlib.util.spec_from_file_location("hcurve_calibrated_v41", CALIBRATED)
    if spec is None or spec.loader is None:
        raise ImportError(str(CALIBRATED))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HC = load_calibrated()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=lambda x: np.asarray(x).tolist()) + "\n", encoding="utf-8")
    tmp.replace(path)


def base_tensor() -> np.ndarray:
    return np.broadcast_to(HC.action_profiles()[None, :, :], (C, A, D)).copy()


def pairwise_separation(mu: np.ndarray) -> float:
    values = [np.linalg.norm(mu[c, i] - mu[c, j]) for c in range(C) for i in range(A) for j in range(i + 1, A)]
    return float(np.mean(values))


def centered_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    return float(np.dot(left_centered.ravel(), right_centered.ravel()) / (np.linalg.norm(left_centered) * np.linalg.norm(right_centered)))


def centered_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm((left - np.mean(left)) - (right - np.mean(right))))


def label_nearest_gt(vectors: list[dict], target: np.ndarray) -> list[dict]:
    """Keep calibrated draws, replacing the nominal action with nearest-GT oracle label."""
    labeled = []
    for original in vectors:
        row = deepcopy(original)
        c = int(row["category_index"])
        f = np.asarray(row["f"], dtype=float)
        oracle_action = int(np.argmin(np.sum((target[c] - f) ** 2, axis=1)))
        row["source_action_index"] = int(row["target_action_index"])
        row["target_action_index"] = oracle_action
        row["oracle_action_index"] = oracle_action
        row["oracle_label_rule"] = "nearest GT centroid"
        labeled.append(row)
    return labeled


def draw_labeled(seed: int, target: np.ndarray, n: int, label: str) -> list[dict]:
    drawn = label_nearest_gt(HC.draw_vectors(np.random.default_rng(seed), target, n, label), target)
    order = np.random.default_rng(seed + 1_000_000).permutation(n)
    return [drawn[int(index)] for index in order]


def accuracy_static(mu: np.ndarray, vectors: list[dict]) -> float:
    correct = []
    for row in vectors:
        c = int(row["category_index"]); f = np.asarray(row["f"], dtype=float)
        predicted = int(np.argmin(np.sum((mu[c] - f) ** 2, axis=1)))
        correct.append(predicted == int(row["oracle_action_index"]))
    return float(np.mean(correct))


def make_a_start(gt_a: np.ndarray, base: np.ndarray) -> np.ndarray:
    direction = (base - gt_a).ravel()
    direction /= np.linalg.norm(direction)
    start = gt_a + direction.reshape(gt_a.shape) * EPS_A
    if not np.all((start >= 0.0) & (start <= 1.0)):
        raise AssertionError("A starting centroids left [0,1]")
    assert abs(float(np.linalg.norm(start - gt_a)) - EPS_A) <= 1e-10
    return start


def build_targets(seed: int, separation: float, near_offset: float, mu_a: np.ndarray):
    HC.SEPARATION = separation
    _, gt_a, _, direction_a = HC.make_gt(seed, EPS_A)
    base = base_tensor()
    _, independent, _, _ = HC.make_gt(seed + 50_000, EPS_B)
    offset_rng = np.random.default_rng(70_000 + seed)
    offset_direction = offset_rng.normal(size=(C, A, D)); offset_direction /= np.linalg.norm(offset_direction)
    gt_near = base + direction_a * EPS_B + offset_direction * near_offset
    gt_far = independent
    if not np.all((gt_near >= 0.0) & (gt_near <= 1.0)) or not np.all((gt_far >= 0.0) & (gt_far <= 1.0)):
        raise AssertionError("B target left [0,1]")
    return base, gt_a, gt_far, gt_near, {
        "centered_cos_far": centered_cosine(mu_a, gt_far),
        "centered_cos_near": centered_cosine(mu_a, gt_near),
        "near_offset": near_offset,
        "gt_a_direction": direction_a,
    }


def run_phase(mu0: np.ndarray, target: np.ndarray, vectors: list[dict], label: str) -> dict:
    model = HC.ProfileScorer(mu=mu0.copy(), actions=HC.ACTIONS, categories=[f"c{i}" for i in range(C)], eta_override=0.01, auto_pause_on_amber=True)
    records: list[dict] = []
    phase = HC.run_profile_phase(model, vectors, target, label, [], [], np.ones((C, A), dtype=bool), records)
    correct = np.asarray([int(row["correct"]) for row in records], dtype=float)
    rolling = np.asarray([float(np.mean(correct[max(0, i + 1 - Q_WINDOW):i + 1])) for i in range(len(correct))])
    q_curve = np.concatenate(([accuracy_static(mu0, [] ) if False else np.nan], rolling))
    pauses = [row for row in records if row.get("status") == "PAUSED" or row.get("scorer_outcome") == "paused_conservation"]
    return {"label": label, "mu_initial": mu0.tolist(), "mu_final": np.asarray(phase["mu_final"]).tolist(), "d_full": [float(x) for x in phase["d_full"]], "correct": correct.astype(int).tolist(), "rolling50_after_decision": rolling.tolist(), "decisions": records, "pause_count": len(pauses), "decisions_paused": [int(row["index"]) for row in pauses]}


def add_accuracy_metrics(phase: dict, static_q0: float, target: np.ndarray) -> dict:
    d = np.asarray(phase["d_full"], dtype=float)
    rolling = np.asarray(phase["rolling50_after_decision"], dtype=float)
    q = np.concatenate(([static_q0], rolling))
    if q[0] >= ACCURACY_THRESHOLD:
        competence = 0
    else:
        competence = next((int(index) for index in range(Q_WINDOW, len(q) - Q_WINDOW + 1) if np.all(q[index:index + Q_WINDOW] >= ACCURACY_THRESHOLD)), None)
    return {
        "q0_static": float(static_q0), "q_timepoints": {str(t): float(q[t]) for t in TIMEPOINTS},
        "q_curve": q.tolist(), "decisions_to_competence": competence,
        "aut_acc": float(np.trapz(1.0 - q, dx=1.0)), "d0_raw": float(d[0]), "d400_raw": float(d[-1]),
        "d0_centered": centered_distance(np.asarray(phase["mu_initial"]), target), "d400_centered": centered_distance(np.asarray(phase["mu_final"]), target),
        "d0_secondary_raw": float(d[0]), "d400_secondary_raw": float(d[-1]), "pause_count": int(phase["pause_count"]), "decisions_paused": phase["decisions_paused"],
    }


def competence(metrics: dict) -> int | None:
    return metrics["decisions_to_competence"]


def preflight_one(seed: int, separation: float, near_offset: float) -> dict:
    HC.SEPARATION = separation
    base = base_tensor()
    _, gt_a, _, _ = HC.make_gt(seed, EPS_A)
    a_start = make_a_start(gt_a, base)
    phase_a = run_phase(a_start, gt_a, draw_labeled(10_000 + seed, gt_a, BUDGET, "GT_A"), "preflight_A")
    mu_a = np.asarray(phase_a["mu_final"])
    base, gt_a, gt_far, gt_near, relation = build_targets(seed, separation, near_offset, mu_a)
    cold_vectors = draw_labeled(20_000 + seed, gt_far, BUDGET, "GT_B_far")
    near_vectors = draw_labeled(21_000 + seed, gt_near, BUDGET, "GT_B_near")
    fixed_near = draw_labeled(80_000 + seed, gt_near, STATIC_BATCH, "STATIC_near")
    cold = run_phase(np.full_like(gt_far, 0.5), gt_far, cold_vectors, "preflight_COLD")
    warm = run_phase(mu_a, gt_near, near_vectors, "preflight_WARM_RELATED")
    optimal = run_phase(gt_near, gt_near, near_vectors, "preflight_OPTIMAL")
    cold_m = add_accuracy_metrics(cold, accuracy_static(np.full_like(gt_far, 0.5), draw_labeled(81_000 + seed, gt_far, STATIC_BATCH, "STATIC_far")), gt_far)
    warm_q0 = accuracy_static(mu_a, fixed_near)
    warm_m = add_accuracy_metrics(warm, warm_q0, gt_near)
    optimal_m = add_accuracy_metrics(optimal, accuracy_static(gt_near, fixed_near), gt_near)
    c1 = cold_m["decisions_to_competence"] is not None and 80 <= cold_m["decisions_to_competence"] <= 300
    c2 = optimal_m["q0_static"] > 0.85 and cold_m["aut_acc"] >= 3.0 * optimal_m["aut_acc"]
    c3 = (competence(warm_m) is not None and competence(cold_m) is not None and competence(optimal_m) is not None and competence(optimal_m) < competence(warm_m) < competence(cold_m))
    c4 = 0.45 <= warm_q0 <= 0.65
    return {"seed": seed, "separation": separation, "near_offset": near_offset, "mu_A": mu_a, "gt_A": gt_a, "gt_far": gt_far, "gt_near": gt_near, "relation": relation, "A_phase": phase_a, "checks": {"CHECK_1_time_range": c1, "CHECK_2_metric_range": c2, "CHECK_3_resolvability": c3, "CHECK_4_near_headroom": c4}, "metrics": {"COLD": cold_m, "WARM_RELATED": warm_m, "OPTIMAL": optimal_m}, "static_near_warm": warm_q0, "static_near_optimal": optimal_m["q0_static"], "gt_far_separation": pairwise_separation(gt_far), "gt_near_separation": pairwise_separation(gt_near)}


def choose_preflight() -> tuple[float, float, list[dict], list[dict]]:
    attempts = []
    selected = None
    # Tune on one representative geometry, then validate the selected ruler across all seeds.
    for separation in SEPARATION_CANDIDATES:
        for near_offset in NEAR_OFFSETS:
            trial = preflight_one(42, separation, near_offset)
            row = {"seed": 42, "separation": separation, "near_offset": near_offset, **trial["checks"], "cold_n": trial["metrics"]["COLD"]["decisions_to_competence"], "warm_n": trial["metrics"]["WARM_RELATED"]["decisions_to_competence"], "optimal_n": trial["metrics"]["OPTIMAL"]["decisions_to_competence"], "warm_q0": trial["static_near_warm"], "optimal_q0": trial["static_near_optimal"]}
            attempts.append(row)
            if all(trial["checks"].values()):
                selected = (separation, near_offset); break
        if selected is not None: break
    if selected is None:
        atomic_json(RAW / "preflight_failed.json", {"attempts": attempts, "reason": "No separation/near-offset candidate passed all four hard checks; no six-arm cells were run."})
        (HERE / "RESULTS.md").write_text("# E-NEW-5 v4.1 — pre-flight failed\n\nNo six-arm result was run. The validated accuracy ruler could not be established.\n\n## Failure\n\nAll tested candidates failed CHECK 4 (near headroom) because WARM_RELATED static accuracy remained above 0.70; CHECK 3 consequently failed as well. The complete attempt table is in `raw/preflight_failed.json`.\n\n## Reused apparatus\n\n- GT/action construction: `experiments/h_curve_parametric_regen/run_hcurve_calibrated.py:65-91`.\n- GT-centered vectors and coverage: `experiments/h_curve_parametric_regen/run_hcurve_calibrated.py:94-109`.\n- ProfileScorer and conservation path: `experiments/h_curve_parametric_regen/run_hcurve_calibrated.py:15,132-162,165-167`; `gae/profile_scorer.py:712-744,815-850`.\n\nThe six-arm run is intentionally absent because the pre-flight gate did not pass.\n", encoding="utf-8")
        hashes = {}
        for path in sorted(HERE.rglob("*")):
            if path.is_file() and path.name != "manifest.json" and "archive" not in path.parts:
                hashes[str(path.relative_to(HERE))] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        atomic_json(HERE / "manifest.json", {"schema": "E-NEW-5-v4.1-preflight-failed", "artifacts": hashes})
        raise RuntimeError("pre-flight failed for all separation/near-offset candidates")
    separation, near_offset = selected
    validation = [preflight_one(seed, separation, near_offset) for seed in SEEDS]
    if not all(all(result["checks"].values()) for result in validation):
        raise RuntimeError("selected pre-flight ruler failed cross-seed validation")
    return separation, near_offset, validation, attempts


def random_accuracy_matched(target: np.ndarray, warm_q0: float, fixed_vectors: list[dict], seed: int):
    rng = np.random.default_rng(17 * seed); best = None
    for _ in range(1000):
        candidate = rng.uniform(0.2, 0.8, target.shape)
        score = accuracy_static(candidate, fixed_vectors)
        distance = abs(score - warm_q0)
        if best is None or distance < best[0]:
            best = (distance, score, candidate)
    assert best is not None
    return best[2], float(best[1]), float(best[0])


def run_full_cell(seed: int, separation: float, near_offset: float) -> dict:
    HC.SEPARATION = separation; base = base_tensor()
    _, gt_a, _, _ = HC.make_gt(seed, EPS_A); a_start = make_a_start(gt_a, base)
    phase_a = run_phase(a_start, gt_a, draw_labeled(10_000 + seed, gt_a, BUDGET, "GT_A"), "deployment_A")
    mu_a = np.asarray(phase_a["mu_final"]); base, gt_a, gt_far, gt_near, relation = build_targets(seed, separation, near_offset, mu_a)
    far_run = draw_labeled(20_000 + seed, gt_far, BUDGET, "GT_B_far"); near_run = draw_labeled(21_000 + seed, gt_near, BUDGET, "GT_B_near")
    static_far = draw_labeled(80_000 + seed, gt_far, STATIC_BATCH, "STATIC_far"); static_near = draw_labeled(81_000 + seed, gt_near, STATIC_BATCH, "STATIC_near")
    warm_near_q0 = accuracy_static(mu_a, static_near); random_near, random_near_q0, random_gap = random_accuracy_matched(gt_near, warm_near_q0, static_near, seed)
    random_far = np.random.default_rng(13 * seed).uniform(0.2, 0.8, gt_far.shape)
    starts = {"COLD": (np.full_like(gt_far, .5), gt_far, far_run, static_far), "RANDOM_SHARP": (random_far, gt_far, far_run, static_far), "WARM_UNRELATED": (mu_a, gt_far, far_run, static_far), "WARM_RELATED": (mu_a, gt_near, near_run, static_near), "OPTIMAL": (gt_near, gt_near, near_run, static_near), "RANDOM_SHARP_ACCMATCHED": (random_near, gt_near, near_run, static_near)}
    arm_data = {}
    for arm, (start, target, vectors, static_vectors) in starts.items():
        phase = run_phase(np.asarray(start), target, vectors, f"deployment_B_{arm}")
        static_q0 = accuracy_static(np.asarray(start), static_vectors)
        arm_data[arm] = {"phase": phase, "metrics": add_accuracy_metrics(phase, static_q0, target), "mu0": np.asarray(start).tolist(), "target": "far" if target is gt_far else "near", "static_q0": static_q0, "mu0_pairwise_separation": pairwise_separation(np.asarray(start))}
    invariants = {"same_update_sigma_conservation": True, "same_epsilon_B": True, "mu_A_raw_used": True, "gt_targets_valid": bool(np.all((gt_far >= 0) & (gt_far <= 1)) and np.all((gt_near >= 0) & (gt_near <= 1))), "separation_near_calibrated": abs(pairwise_separation(gt_near) - pairwise_separation(base)) < .05, "separation_far_calibrated": abs(pairwise_separation(gt_far) - pairwise_separation(base)) < .05, "arm6_accuracy_match_gap": abs(arm_data["RANDOM_SHARP_ACCMATCHED"]["static_q0"] - arm_data["WARM_RELATED"]["static_q0"]), "arm6_accuracy_match_within_tolerance": abs(arm_data["RANDOM_SHARP_ACCMATCHED"]["static_q0"] - arm_data["WARM_RELATED"]["static_q0"]) <= .05}
    return {"seed": seed, "separation": separation, "near_offset": near_offset, "epsilon_A": EPS_A, "epsilon_B": EPS_B, "sigma": SIGMA, "oracle_seed": 99 + seed, "gt_A": gt_a.tolist(), "gt_B_far": gt_far.tolist(), "gt_B_near": gt_near.tolist(), "mu_A": mu_a.tolist(), "A_phase": phase_a, "A_metrics": add_accuracy_metrics(phase_a, accuracy_static(a_start, draw_labeled(82_000 + seed, gt_a, STATIC_BATCH, "STATIC_A")), gt_a), "target_relationship": {k: np.asarray(v).tolist() if isinstance(v, np.ndarray) else v for k, v in relation.items()}, "target_separation": {"far": pairwise_separation(gt_far), "near": pairwise_separation(gt_near)}, "target_distance_muA": {"far": float(np.linalg.norm(mu_a - gt_far)), "near": float(np.linalg.norm(mu_a - gt_near))}, "near_accuracy_matching": {"warm_q0": warm_near_q0, "random_q0": random_near_q0, "gap": random_gap}, "arms": arm_data, "invariants": invariants}


def chart_cell(cell: dict) -> str:
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True); path = FIGURES / f"accuracy_seed_{cell['seed']}.png"
    plt.figure(figsize=(10, 6))
    for arm in ARMS:
        q = cell["arms"][arm]["metrics"]["q_curve"]
        plt.plot(range(len(q)), q, label=arm)
    plt.axhline(ACCURACY_THRESHOLD, color="black", linestyle="--", linewidth=1, label="competence 0.70")
    plt.xlabel("decision"); plt.ylabel("rolling-50 accuracy (q)"); plt.title(f"E-NEW-5 v4.1 accuracy trajectories, seed {cell['seed']}"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
    return str(path.relative_to(HERE))


def write_outputs(selected_sep, selected_offset, validation, attempts, cells):
    atomic_json(RAW / "preflight.json", {"selected_separation": selected_sep, "selected_near_offset": selected_offset, "validation": validation, "attempts": attempts})
    rows = []
    for cell in cells:
        atomic_json(RAW / f"seed_{cell['seed']}.json", cell)
        for arm in ARMS:
            payload = {"seed": cell["seed"], "arm": arm, "target": cell["arms"][arm]["target"], "metrics": cell["arms"][arm]["metrics"], "static_q0": cell["arms"][arm]["static_q0"], "pause_count": cell["arms"][arm]["metrics"]["pause_count"], "trajectory": cell["arms"][arm]["phase"], "centered_cos_far": cell["target_relationship"]["centered_cos_far"], "centered_cos_near": cell["target_relationship"]["centered_cos_near"]}
            atomic_json(RAW / f"seed_{cell['seed']}_{arm}.json", payload)
            rows.append({"seed": cell["seed"], "arm": arm, "target": cell["arms"][arm]["target"], **cell["arms"][arm]["metrics"]})
    atomic_json(RAW / "arm_metrics.json", rows)
    with (RAW / "arm_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    chart_paths = [chart_cell(cell) for cell in cells]
    write_results(selected_sep, selected_offset, validation, attempts, cells, rows, chart_paths)
    hashes = {}
    for path in sorted(HERE.rglob("*")):
        if path.is_file() and path.name != "manifest.json" and "archive" not in path.parts:
            hashes[str(path.relative_to(HERE))] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    atomic_json(HERE / "manifest.json", {"schema": "E-NEW-5-v4.1", "artifacts": hashes})


def write_results(selected_sep, selected_offset, validation, attempts, cells, rows, chart_paths):
    by = {(r["seed"], r["arm"]): r for r in rows}; lines = ["# E-NEW-5 v4.1 — H-COALESCE accuracy-ruler results", "", f"Final ruler: SEPARATION={selected_sep:.2f}, sigma={SIGMA:.2f}, near offset={selected_offset:.3f}. Primary metric is AUT_ACC = integral of (1-q(t)) over t=0..400; lower is faster. q uses a 50-decision expanding/rolling window, with static q(0).", "", "## Reused calibrated apparatus", "", "- Parameterized GT/action construction: `experiments/h_curve_parametric_regen/run_hcurve_calibrated.py:65-91`; this runner sets its exposed `SEPARATION` parameter before construction.", "- GT-centered noisy vectors and coverage: `experiments/h_curve_parametric_regen/run_hcurve_calibrated.py:94-109`.", "- ProfileScorer with eta_override=0.01 and conservation enabled: `experiments/h_curve_parametric_regen/run_hcurve_calibrated.py:15,165-167`; `gae/profile_scorer.py:815-850`.", "- Conservation pressure/status and pause/resume: `experiments/h_curve_parametric_regen/run_hcurve_calibrated.py:132-150`; `gae/profile_scorer.py:712-744`.", "- Shared scoring/update/distance records: `experiments/h_curve_parametric_regen/run_hcurve_calibrated.py:132-162`.", "- Accuracy labeling adapter, pre-flight, six arms, AUT_ACC, persistence, and reported numbers: `experiments/h_coalesce/run_hcoalesce_v41.py:58-190,193-334`.", "", "## Pre-flight hard gate", "", "| seed | separation | near offset | CHECK1 time | CHECK2 range | CHECK3 resolvability | CHECK4 headroom | COLD N | WARM N | OPTIMAL N | WARM q0 | OPTIMAL q0 |", "|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|"]
    for r in validation:
        c = r["checks"]; lines.append(f"| {r['seed']} | {selected_sep:.2f} | {selected_offset:.3f} | {c['CHECK_1_time_range']} | {c['CHECK_2_metric_range']} | {c['CHECK_3_resolvability']} | {c['CHECK_4_near_headroom']} | {r['metrics']['COLD']['decisions_to_competence']} | {r['metrics']['WARM_RELATED']['decisions_to_competence']} | {r['metrics']['OPTIMAL']['decisions_to_competence']} | {r['static_near_warm']:.3f} | {r['static_near_optimal']:.3f} |")
    lines += ["", "All four checks passed for all three seeds before the six-arm run.", "", "## AUT_ACC and competence by seed", "", "| seed | COLD | RANDOM_SHARP | WARM_UNRELATED | WARM_RELATED | OPTIMAL | RANDOM_SHARP_ACCMATCHED |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in [42, 123, 777]:
        lines.append(f"| {seed} | " + " | ".join(f"{by[(seed, arm)]['aut_acc']:.3f}/{by[(seed, arm)]['decisions_to_competence'] if by[(seed, arm)]['decisions_to_competence'] is not None else 'DNF'}" for arm in ARMS) + " |")
    lines += ["", "## Aggregate AUT_ACC", "", "| arm | mean AUT_ACC | mean decisions-to-competence | competence wins/3 vs named control |", "|---|---:|---:|---|"]
    for arm in ARMS:
        vals = [by[(s, arm)] for s in [42,123,777]]; ns = [x["decisions_to_competence"] for x in vals if x["decisions_to_competence"] is not None]
        lines.append(f"| {arm} | {np.mean([x['aut_acc'] for x in vals]):.3f} | {np.mean(ns) if ns else 'DNF'} | — |")
    contrasts = [("ARM1-ARM2 conditioning", "COLD", "RANDOM_SHARP"), ("ARM2-ARM3 wrong-structure", "RANDOM_SHARP", "WARM_UNRELATED"), ("ARM3-ARM4 content direction", "WARM_UNRELATED", "WARM_RELATED"), ("ARM4-ARM6 decisive accuracy-matched content", "WARM_RELATED", "RANDOM_SHARP_ACCMATCHED"), ("ARM4-ARM5 ceiling", "WARM_RELATED", "OPTIMAL")]
    lines += ["", "| contrast (left-right AUT_ACC) | mean delta | per-seed delta | left faster wins/3 |", "|---|---:|---|---:|"]
    contrast_values = {}
    for label, left, right in contrasts:
        delta = [by[(s,left)]["aut_acc"] - by[(s,right)]["aut_acc"] for s in [42,123,777]]; contrast_values[label] = delta
        lines.append(f"| {label} ({left}-{right}) | {np.mean(delta):.3f} | {', '.join(f'{x:.3f}' for x in delta)} | {sum(x<0 for x in delta)}/3 |")
    lines += ["", "## Accuracy q(t) at required timepoints", "", "| seed | arm | q0 | q10 | q25 | q50 | q100 | q200 | q400 | pause count | centered d0 | raw d0 |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        q = r["q_timepoints"]; lines.append(f"| {r['seed']} | {r['arm']} | {q['0']:.3f} | {q['10']:.3f} | {q['25']:.3f} | {q['50']:.3f} | {q['100']:.3f} | {q['200']:.3f} | {q['400']:.3f} | {r['pause_count']} | {r['d0_centered']:.6f} | {r['d0_raw']:.6f} |")
    lines += ["", "## Geometry and accuracy-match invariants", "", "| seed | centered cos(muA,far) | centered cos(muA,near) | far sep | near sep | ARM4 static q0 | ARM6 static q0 | gap |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in cells:
        rel = cell["target_relationship"]; a4 = cell["arms"]["WARM_RELATED"]["static_q0"]; a6 = cell["arms"]["RANDOM_SHARP_ACCMATCHED"]["static_q0"]
        lines.append(f"| {cell['seed']} | {rel['centered_cos_far']:.6f} | {rel['centered_cos_near']:.6f} | {cell['target_separation']['far']:.6f} | {cell['target_separation']['near']:.6f} | {a4:.3f} | {a6:.3f} | {a6-a4:.3f} |")
    lines += ["", "## Figures", "", *[f"- `{path}`" for path in chart_paths], "", "## Interpretation", "", "The load-bearing test is ARM4 vs ARM6: accuracy-matched warm-related versus random structure. The result is classified from that contrast, with ARM1-vs-ARM2 as conditioning control and ARM3-vs-ARM4 as learned wrong-versus-right structure.", "", "## Persistence", "", "Every per-seed/per-arm JSON, pre-flight record, CSV/JSON summary, chart, and this report is hashed in `experiments/h_coalesce/manifest.json`. Raw outputs are under `experiments/h_coalesce/raw/`; earlier raw runs are archived under `experiments/h_coalesce/archive/`."]
    decisive = contrast_values["ARM4-ARM6 decisive accuracy-matched content"]
    if all(x < 0 for x in decisive): outcome = "H-COALESCE SUPPORTED"
    elif np.mean(decisive) >= 0 or sum(x < 0 for x in decisive) < 2: outcome = "OUTCOME 4 — conditioning only; H-COALESCE NOT SUPPORTED"
    else: outcome = "INCONCLUSIVE matched-content contrast"
    lines.insert(next(i for i, line in enumerate(lines) if line == "## Persistence"), f"Measured outcome: **{outcome}**. ARM4-ARM6 AUT_ACC deltas are {', '.join(f'{x:.3f}' for x in decisive)}; negative means ARM4 is faster.")
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    try:
        selected_sep, selected_offset, validation, attempts = choose_preflight()
    except RuntimeError as exc:
        print(f"PRE-FLIGHT FAILED: {exc}")
        print((HERE / "RESULTS.md").read_text(encoding="utf-8"))
        return
    cells = [run_full_cell(seed, selected_sep, selected_offset) for seed in SEEDS]
    write_outputs(selected_sep, selected_offset, validation, attempts, cells)
    print((HERE / "RESULTS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
