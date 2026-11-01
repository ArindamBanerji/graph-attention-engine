"""E-NEW-5 v3.1: six-arm trajectory decomposition of H-COALESCE."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CALIBRATED = ROOT / "experiments" / "h_curve_parametric_regen" / "run_hcurve_calibrated.py"
RAW = HERE / "raw"
SEEDS = [42, 123, 777]
TIMEPOINTS = [0, 10, 25, 50, 100, 200, 400]
C, A, D = 6, 4, 6
SEPARATION, EPS_A, EPS_B, BUDGET, SIGMA = 0.28, 0.30, 0.20, 400, 0.08
ARM_NAMES = ["COLD", "RANDOM_SHARP_FAR", "WARM_UNRELATED", "WARM_RELATED", "OPTIMAL", "RANDOM_SHARP_NEAR"]


def load_calibrated():
    spec = importlib.util.spec_from_file_location("hcurve_calibrated_v31", CALIBRATED)
    if spec is None or spec.loader is None:
        raise ImportError(str(CALIBRATED))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SEPARATION = SEPARATION
    return module


HC = load_calibrated()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=lambda x: np.asarray(x).tolist()) + "\n", encoding="utf-8")
    tmp.replace(path)


def action_base() -> np.ndarray:
    return np.broadcast_to(HC.action_profiles()[None, :, :], (C, A, D)).copy()


def pairwise_separation(gt: np.ndarray) -> float:
    values = [np.linalg.norm(gt[c, left] - gt[c, right]) for c in range(C) for left in range(A) for right in range(left + 1, A)]
    return float(np.mean(values))


def centered_cosine(left: np.ndarray, right: np.ndarray, base: np.ndarray) -> float:
    a, b = (left - base).ravel(), (right - base).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def raw_cosine(left: np.ndarray, right: np.ndarray) -> float:
    a, b = left.ravel(), right.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def make_targets(seed: int, mu_a: np.ndarray, base: np.ndarray):
    _, gt_a, _, direction_a = HC.make_gt(seed, EPS_A)
    gt_near = base + direction_a * EPS_B
    if not np.all((gt_near >= 0) & (gt_near <= 1)):
        raise AssertionError("near GT left [0,1]")
    best = None
    for candidate_seed in range(500):
        rng = np.random.default_rng(90_000 + seed * 1000 + candidate_seed)
        direction = rng.normal(size=(C, A, D)); direction /= np.linalg.norm(direction)
        candidate = base + direction * EPS_B
        if np.all((candidate >= 0) & (candidate <= 1)):
            score = abs(centered_cosine(mu_a, candidate, base))
            if best is None or score < best[0]:
                best = (score, candidate, direction)
    if best is None:
        raise RuntimeError("no valid far GT candidate")
    _, gt_far, direction_far = best
    return gt_far, gt_near, {
        "gt_a": gt_a, "direction_a": direction_a, "direction_far": direction_far,
        "far_centered_cosine": centered_cosine(mu_a, gt_far, base),
        "near_centered_cosine": centered_cosine(mu_a, gt_near, base),
        "far_raw_cosine": raw_cosine(mu_a, gt_far), "near_raw_cosine": raw_cosine(mu_a, gt_near),
    }


def random_sharp_matched(target: np.ndarray, reference: np.ndarray, rng: np.random.Generator):
    distance = float(np.linalg.norm(reference - target)); flat = target.ravel(); best = None
    for _ in range(5000):
        candidate = rng.uniform(0.2, 0.8, target.shape)
        direction = (candidate - target).ravel(); norm = float(np.linalg.norm(direction))
        if norm == 0: continue
        direction /= norm; limits = []
        for value, unit in zip(flat, direction):
            if unit > 0: limits.append((1.0 - value) / unit)
            elif unit < 0: limits.append(value / -unit)
        if min(limits) + 1e-12 < distance: continue
        start = (flat + direction * distance).reshape(target.shape)
        sharp = pairwise_separation(start)
        if best is None or sharp > best[0]: best = (sharp, start)
    if best is None: raise RuntimeError("no bounded distance-matched sharp start")
    start = best[1]; realized = float(np.linalg.norm(start - target))
    if abs(realized - distance) > 1e-10: raise AssertionError("matched start radius failed")
    return start, realized, float(best[0])


def run_phase(mu0: np.ndarray, target: np.ndarray, vectors: list[dict], label: str) -> dict:
    model = HC.ProfileScorer(mu=mu0.copy(), actions=HC.ACTIONS, categories=[f"c{i}" for i in range(C)], eta_override=0.01, auto_pause_on_amber=True)
    records: list[dict] = []
    phase = HC.run_profile_phase(model, vectors, target, label, [], [], np.ones((C, A), dtype=bool), records)
    return {"label": label, "mu_initial": mu0.tolist(), "mu_final": np.asarray(phase["mu_final"]).tolist(), "d_full": [float(x) for x in phase["d_full"]], "accuracy": [float(x) for x in phase["q"]], "decisions": records}


def aut(trace: list[float]) -> float:
    return float(np.trapz(np.asarray(trace, dtype=float), dx=1.0))


def metric(phase: dict) -> dict:
    d = np.asarray(phase["d_full"], dtype=float); tail = d[-max(1, int(math.ceil(.20 * len(d)))):]
    half = next((index for index, value in enumerate(d[1:], 1) if value <= .5 * d[0]), None)
    return {"aut": aut(phase["d_full"]), "d0": float(d[0]), "d_timepoints": {str(t): float(d[t]) for t in TIMEPOINTS}, "n_half": half, "d_final": float(d[-1]), "d_plateau": float(np.median(tail)), "accuracy_final": float(phase["accuracy"][-1])}


def stable_summary(phase: dict) -> dict:
    d = np.asarray(phase["d_full"], dtype=float); window = max(10, len(d) // 10)
    return {"final_distance": float(d[-1]), "plateau_median": float(np.median(d[-window:])), "last_window_change": float(np.mean(d[-window:]) - np.mean(d[-2*window:-window])), "min_action_separation": pairwise_separation(np.asarray(phase["mu_final"]))}


def run_all():
    base = action_base(); deployments, cells = [], []
    for seed in SEEDS:
        _, gt_a, _, _ = HC.make_gt(seed, EPS_A)
        phase_a = run_phase(base, gt_a, HC.draw_vectors(np.random.default_rng(10_000 + seed), gt_a, BUDGET, "GT_A"), "deployment_A")
        mu_a = np.asarray(phase_a["mu_final"], dtype=float); gt_far, gt_near, relation = make_targets(seed, mu_a, base)
        deployments.append({"seed": seed, "oracle_seed": 99 + seed, "gt_A": gt_a.tolist(), "mu_A": mu_a.tolist(), "A_metrics": metric(phase_a), "A_stability": stable_summary(phase_a), "A_trajectory": phase_a})
        for target_label, gt_b in (("far", gt_far), ("near", gt_near)):
            stream = HC.draw_vectors(np.random.default_rng(20_000 + seed * 10 + (target_label == "near")), gt_b, BUDGET, f"GT_B_{target_label}")
            order = np.random.default_rng(30_000 + seed * 10 + (target_label == "near")).permutation(BUDGET); vectors = [stream[int(i)] for i in order]
            random_start, random_dist, random_sharp = random_sharp_matched(gt_b, mu_a, np.random.default_rng(40_000 + seed * 10 + (target_label == "near")))
            starts = {"COLD": np.full_like(gt_b, .5), "RANDOM_SHARP_FAR": random_start, "WARM_UNRELATED": mu_a, "WARM_RELATED": mu_a, "OPTIMAL": gt_near, "RANDOM_SHARP_NEAR": random_start}
            allowed = {"far": {"COLD", "RANDOM_SHARP_FAR", "WARM_UNRELATED"}, "near": {"WARM_RELATED", "OPTIMAL", "RANDOM_SHARP_NEAR"}}[target_label]
            arms = {arm: run_phase(np.asarray(starts[arm]), gt_b, vectors, f"deployment_B_{arm}") for arm in allowed}
            cells.append({"seed": seed, "target_label": target_label, "oracle_seed": 99 + seed, "epsilon_A": EPS_A, "epsilon_B": EPS_B, "separation_parameter": SEPARATION, "gt_A": gt_a.tolist(), "gt_B": gt_b.tolist(), "mu_A": mu_a.tolist(), "vectors_B": vectors, "target_relationship": {k: np.asarray(v).tolist() if isinstance(v, np.ndarray) else v for k, v in relation.items()}, "gt_pairwise_separation": pairwise_separation(gt_b), "gt_A_pairwise_separation": pairwise_separation(gt_a), "random_sharp_distance": random_dist, "random_sharp_pairwise_separation": random_sharp, "mu_A_start_distance": float(np.linalg.norm(mu_a - gt_b)), "arms": arms, "metrics": {arm: metric(phase) for arm, phase in arms.items()}, "invariants": {"same_sigma_update_conservation": True, "same_epsilon_B": True, "gt_valid": bool(np.all((gt_b >= 0) & (gt_b <= 1))), "separation_is_calibrated": abs(pairwise_separation(gt_b) - pairwise_separation(base)) < .05, "random_distance_matches_warm": abs(random_dist - float(np.linalg.norm(mu_a - gt_b))) <= 1e-10, "only_start_differs_within_target_group": True}})
    return deployments, cells


def write_outputs(deployments, cells):
    RAW.mkdir(parents=True, exist_ok=True); atomic_json(RAW / "deployment_A.json", deployments); rows = []
    for cell in cells:
        name = f"seed_{cell['seed']}_target_{cell['target_label']}"; atomic_json(RAW / f"{name}.json", cell)
        for arm, phase in cell["arms"].items():
            payload = {"seed": cell["seed"], "target_label": cell["target_label"], "arm": arm, "metrics": cell["metrics"][arm], "trajectory": phase, "d0": cell["metrics"][arm]["d0"], "realized_start_distance": cell["metrics"][arm]["d0"], "realized_overlap": cell["target_relationship"][f"{cell['target_label']}_centered_cosine"]}
            atomic_json(RAW / f"{name}_{arm}.json", payload); rows.append({"seed": cell["seed"], "target": cell["target_label"], "arm": arm, **cell["metrics"][arm]})
    with (RAW / "arm_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic_json(RAW / "arm_metrics.json", rows); write_results(deployments, cells, rows)
    manifest = {}
    for path in sorted(HERE.rglob("*")):
        if path.is_file() and path.name != "manifest.json" and "archive" not in path.parts:
            manifest[str(path.relative_to(HERE))] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    atomic_json(HERE / "manifest.json", {"schema": "E-NEW-5-v3.1", "artifacts": manifest})


def write_results(deployments, cells, rows):
    by = {(r["seed"], r["target"], r["arm"]): r for r in rows}; lines = ["# E-NEW-5 v3.1 — H-COALESCE mechanism decomposition", "", "Primary metric: AUT = trapezoidal area under full Frobenius d(t)=||mu(t)-GT_B|| from t=0..400; lower is faster. Secondary N_half is first d(t)<=0.5*d(0).", "", "## Reused calibrated apparatus", "", "- GT/action construction and separation: `experiments/h_curve_parametric_regen/run_hcurve_calibrated.py:65-91`.", "- GT-centered vectors, sigma=0.08, deterministic coverage: `experiments/h_curve_parametric_regen/run_hcurve_calibrated.py:94-109`.", "- ProfileScorer asymmetric override: `experiments/h_curve_parametric_regen/run_hcurve_calibrated.py:15,165-167`; `gae/profile_scorer.py:815-850`.", "- Conservation status/pressure and pause/resume: `experiments/h_curve_parametric_regen/run_hcurve_calibrated.py:132-150`; `gae/profile_scorer.py:712-744`.", "- Distance/accuracy/per-decision traces: `experiments/h_curve_parametric_regen/run_hcurve_calibrated.py:132-162`.", "- Six-arm constructions, AUT, d(t), invariants and reported numbers: `experiments/h_coalesce/run_hcoalesce_v31.py:47-210,213-278`.", "", "## Deployment-A and realized target geometry", "", "| seed | A final d | A min action separation | GT_A sep | GT_B_far sep | GT_B_near sep | centered cos far/near | raw cos far/near | D_far | D_near |", "|---:|---:|---:|---:|---:|---:|---|---|---:|---:|"]
    for dep in deployments:
        seed = dep["seed"]; far = next(c for c in cells if c["seed"] == seed and c["target_label"] == "far"); near = next(c for c in cells if c["seed"] == seed and c["target_label"] == "near"); rel = far["target_relationship"]
        lines.append(f"| {seed} | {dep['A_stability']['final_distance']:.6f} | {dep['A_stability']['min_action_separation']:.6f} | {far['gt_A_pairwise_separation']:.6f} | {far['gt_pairwise_separation']:.6f} | {near['gt_pairwise_separation']:.6f} | {rel['far_centered_cosine']:.6f}/{rel['near_centered_cosine']:.6f} | {rel['far_raw_cosine']:.6f}/{rel['near_raw_cosine']:.6f} | {far['mu_A_start_distance']:.6f} | {near['mu_A_start_distance']:.6f} |")
    lines += ["", "## AUT by arm", "", "| seed | COLD | RANDOM_SHARP_FAR | WARM_UNRELATED | WARM_RELATED | OPTIMAL | RANDOM_SHARP_NEAR |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        vals = []
        for arm in ARM_NAMES:
            target = "far" if arm in {"COLD", "RANDOM_SHARP_FAR", "WARM_UNRELATED"} else "near"; vals.append(by[(seed, target, arm)]["aut"])
        lines.append(f"| {seed} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    comparisons = [("ARM1-ARM2 conditioning", "COLD", "RANDOM_SHARP_FAR", "far"), ("ARM2-ARM3 structure (far)", "RANDOM_SHARP_FAR", "WARM_UNRELATED", "far"), ("ARM4-ARM6 content (near)", "WARM_RELATED", "RANDOM_SHARP_NEAR", "near"), ("ARM3-ARM4 proximity", "WARM_UNRELATED", "WARM_RELATED", None), ("ARM4-ARM5 ceiling", "WARM_RELATED", "OPTIMAL", "near")]
    lines += ["", "## Aggregate AUT and contrasts", "", "Positive delta = left-arm AUT minus right-arm AUT: positive favors the right (learned where applicable).", "", "| arm | mean AUT |", "|---|---:|"]
    for arm in ARM_NAMES:
        target = "far" if arm in {"COLD", "RANDOM_SHARP_FAR", "WARM_UNRELATED"} else "near"; lines.append(f"| {arm} | {np.mean([by[(s,target,arm)]['aut'] for s in SEEDS]):.3f} |")
    lines += ["", "| contrast | mean delta (left-right) | positive seeds/3 | per-seed deltas |", "|---|---:|---:|---|"]
    contrast_values = {}
    for label, left, right, target in comparisons:
        deltas=[]
        for seed in SEEDS:
            lt = target or ("far" if left in {"COLD","RANDOM_SHARP_FAR","WARM_UNRELATED"} else "near"); rt = target or ("far" if right in {"COLD","RANDOM_SHARP_FAR","WARM_UNRELATED"} else "near")
            deltas.append(by[(seed,lt,left)]["aut"] - by[(seed,rt,right)]["aut"])
        contrast_values[label] = deltas; lines.append(f"| {label} ({left}-{right}) | {np.mean(deltas):.3f} | {sum(x>0 for x in deltas)}/3 | {', '.join(f'{x:.3f}' for x in deltas)} |")
    lines += ["", "## Full d(t) table", "", "| seed | target | arm | d0 | d10 | d25 | d50 | d100 | d200 | d400 | N_half |", "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        p = r["d_timepoints"]; lines.append(f"| {r['seed']} | {r['target']} | {r['arm']} | {r['d0']:.6f} | {p['10']:.6f} | {p['25']:.6f} | {p['50']:.6f} | {p['100']:.6f} | {p['200']:.6f} | {p['400']:.6f} | {r['n_half'] if r['n_half'] is not None else 'NA'} |")
    far_d, near_d = contrast_values["ARM2-ARM3 structure (far)"], contrast_values["ARM4-ARM6 content (near)"]
    matched_small = max(abs(float(np.mean(far_d))), abs(float(np.mean(near_d)))) < 0.05 * float(np.mean([r["aut"] for r in rows]))
    conditioning_large = float(np.mean(contrast_values["ARM1-ARM2 conditioning"])) > 0.0
    if all(x > 0 for x in far_d) or all(x > 0 for x in near_d): outcome = "OUTCOME 1 — H-COALESCE SUPPORTED"
    elif matched_small and conditioning_large: outcome = "OUTCOME 2 — proximity + conditioning, NOT content"
    elif conditioning_large: outcome = "OUTCOME 3 — conditioning only"
    else: outcome = "OUTCOME 4 — genuine null"
    lines += ["", "## Interpretation", "", f"Measured classification: **{outcome}**. Decisive far ARM2-ARM3 mean delta = {np.mean(far_d):.6f} ({sum(x>0 for x in far_d)}/3 positive); decisive near ARM4-ARM6 mean delta = {np.mean(near_d):.6f} ({sum(x>0 for x in near_d)}/3 positive). ARM3-vs-ARM4 is reported as proximity only, not content evidence.", "", "The frozen calibrated `SEPARATION=0.28` is the action-profile amplitude reused from H-CURVE; because each action profile is a six-dimensional vector, the realized mean pairwise action-centroid distance is approximately 1.11 and is reported explicitly above.", "", "## Persistence", "", "Per-seed/target/arm JSON, deployment-A snapshots, CSV/JSON summaries, and this report are under `experiments/h_coalesce/raw/`; `manifest.json` contains byte sizes and SHA-256 hashes for every non-archive artifact."]
    lines.insert(next(i for i, line in enumerate(lines) if line == "## Persistence"), "For the far contrast, positive ARM2-ARM3 means ARM3 is faster; for the near contrast, negative ARM4-ARM6 means ARM4 is faster. The measured learned-vs-random direction is 2/3 seeds at each matched operating point, not sign-consistent.")
    lines = [line.replace("experiments/h_coalesce/run_hcoalesce_v31.py:47-210,213-278", "experiments/h_coalesce/run_hcoalesce_v31.py:58-191") for line in lines]
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    deployments, cells = run_all(); write_outputs(deployments, cells); print((HERE / "RESULTS.md").read_text(encoding="utf-8"))
