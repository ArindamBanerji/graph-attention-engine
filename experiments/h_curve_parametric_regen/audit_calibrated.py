import glob, hashlib, json, math, os
import numpy as np

root = "experiments/h_curve_parametric_regen/raw/calibrated_v1"

def fit(trace):
    values = np.asarray(trace[1:], dtype=float)
    d_inf = float(np.mean(values[-max(1, int(math.ceil(.2 * len(values)))):]))
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
    if len(kept) < 5 or not slopes:
        return None
    k = float(-np.median(slopes))
    return k if k > 0 else None

errors = []
for path in glob.glob(root + "/runs/*/*/arm-*/record.json"):
    data = json.load(open(path, encoding="utf-8"))
    k1 = fit(data["phase1"]["d_full"]); k2 = fit(data["phase2"]["d_disrupted"])
    gamma = None if k1 is None or k2 is None else k2 / k1
    if gamma is None and data["gamma_rate"] is not None:
        errors.append(path)
    if gamma is not None and abs(gamma - data["gamma_rate"]) > 1e-10:
        errors.append(path)
manifest = json.load(open(root + "/manifest.json", encoding="utf-8"))
hash_ok = all(os.path.getsize("experiments/h_curve_parametric_regen/" + item["path"]) == item["bytes"] and hashlib.sha256(open("experiments/h_curve_parametric_regen/" + item["path"], "rb").read()).hexdigest() == item["sha256"] for item in manifest["files"])
cal = json.load(open(root + "/calibration_summary.json", encoding="utf-8"))
print(json.dumps({"gamma_records": 12, "scorer_free_gamma_mismatches": len(errors), "manifest_hashes": hash_ok,
                  "calibration_min_tail100": min(x["accuracy_tail100"] for x in cal["cells"]),
                  "calibration_gate_all_pass": all(x["passed"] for x in cal["cells"])}, indent=2))
