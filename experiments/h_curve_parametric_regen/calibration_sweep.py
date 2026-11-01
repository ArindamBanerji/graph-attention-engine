import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location("hc", "experiments/h_curve_parametric_regen/run_hcurve_calibrated.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for sep in [0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.18, 0.20]:
    mod.SEPARATION = sep
    values = []
    for seed in mod.SEEDS:
        for eps in mod.EPS:
            canonical, gt1, _, _ = mod.make_gt(seed, eps)
            rng = np.random.default_rng(1000 + seed * 10 + int(eps * 100))
            vectors = mod.draw_vectors(rng, gt1, mod.BUDGETS[eps], "GT1")
            values.append(mod.calibration(seed, eps, canonical, gt1, vectors)["accuracy_tail100"])
    print(sep, [round(x, 3) for x in values], round(float(np.mean(values)), 3))
