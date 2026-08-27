from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_v2"
FIG = ROOT / "figures_v2"
FIG.mkdir(parents=True, exist_ok=True)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    cells = load(RAW / "summary.json")["cells"]
    eps = [0.35, 0.50]
    arms = ["unenriched", "enriched"]
    labels = {"unenriched": "UNENRICHED σ=.08", "enriched": "ENRICHED σ=.04 on [0,1]"}
    colors = {"unenriched": "#0072B2", "enriched": "#D55E00"}
    fig, (floor_ax, rate_ax) = plt.subplots(1, 2, figsize=(12.2, 5.8))
    x = np.arange(len(eps))
    width = .28
    means = {}
    for ai, arm in enumerate(arms):
        means[arm] = {}
        for xi, e in enumerate(eps):
            rows = [r for r in cells if r["arm"] == arm and r["epsilon_firm"] == e]
            floor = [r["floor_day1_accuracy"] for r in rows]
            kvals = [r["rate"]["k"] for r in rows]
            means[arm][e] = {"floor": float(np.mean(floor)), "k": float(np.mean(kvals))}
            px = x[xi] + (-width / 2 if ai == 0 else width / 2)
            floor_ax.scatter(np.full(3, px), floor, color=colors[arm], s=52, edgecolor="white",
                             linewidth=.55, label=f"{labels[arm]} seeds" if xi == 0 else None)
            floor_ax.scatter([px], [np.mean(floor)], color=colors[arm], marker="D", s=88,
                             edgecolor="black", linewidth=.65, zorder=5)
            rate_ax.scatter(np.full(3, px), kvals, color=colors[arm], s=52, edgecolor="white",
                            linewidth=.55, label=f"{labels[arm]} seeds" if xi == 0 else None)
            rate_ax.scatter([px], [np.mean(kvals)], color=colors[arm], marker="D", s=88,
                            edgecolor="black", linewidth=.65, zorder=5)
    for xi, e in enumerate(eps):
        baseline = means["unenriched"][e]["k"]
        lo, hi = sorted([baseline * .8, baseline * 1.2])
        rate_ax.fill_between([x[xi] - .42, x[xi] + .42], [lo, lo], [hi, hi],
                             color="#009E73", alpha=.14, zorder=0)
        rate_ax.hlines(baseline, x[xi] - .42, x[xi] + .42, color="#333", linestyle="--", linewidth=1)
    floor_ax.set_xticks(x, ["ε=.35", "ε=.50"])
    floor_ax.set_ylabel("floor accuracy (mean of per-cell decisions 1–10)")
    floor_ax.set_title("Floor: leveling is ε-dependent")
    floor_ax.set_ylim(0.45, .80)
    floor_ax.grid(axis="y", alpha=.25)
    floor_ax.legend(fontsize=8, loc="lower right")
    rate_ax.set_xticks(x, ["ε=.35", "ε=.50"])
    rate_ax.set_ylabel("fitted k (positive decay rate)")
    rate_ax.set_title("Rate: k equality band is ε-dependent")
    rate_ax.grid(axis="y", alpha=.25)
    rate_ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("H-ENRICH v2: headroom test at two firm-mismatch levels", fontsize=14, y=.98)
    fig.text(.5, .015, "Diamonds are three-seed means; green bands show unenriched k ±20% at each ε.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=[0, .05, 1, .94])
    png = FIG / "henrich_v2_floor_rate.png"
    pdf = FIG / "henrich_v2_floor_rate.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    caption = (
        "H-ENRICH v2 headroom test. At ε=.35, enrichment lifts the per-cell first-10 floor by "
        "+0.83 percentage points but increases k by 20.4%, just outside the ±20% leveling band. "
        "At ε=.50, k remains invariant (ratio .915) but the floor is 1.11 points lower under "
        "enrichment. The leveling split is therefore ε-dependent, not a uniform clean second engine."
    )
    cap = FIG / "henrich_v2_floor_rate.caption.txt"
    cap.write_text(caption + "\n", encoding="utf-8")
    meta = FIG / "chart_metadata.json"
    meta.write_text(json.dumps({
        "schema_version": "henrich-v2-chart-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scorer_imported": False, "rerun": False,
        "inputs": ["raw_v2/summary.json", "raw_v2/seed-*/epsilon-*/unenriched/records.json",
                   "raw_v2/seed-*/epsilon-*/enriched/records.json"],
        "outputs": [png.name, pdf.name, cap.name], "rate_band": "unenriched k ±20% per epsilon",
    }, indent=2) + "\n", encoding="utf-8")
    manifest_path = RAW / "manifest.json"
    manifest = load(manifest_path)
    paths = {"../figures_v2/henrich_v2_floor_rate.png": png,
             "../figures_v2/henrich_v2_floor_rate.pdf": pdf,
             "../figures_v2/henrich_v2_floor_rate.caption.txt": cap,
             "../figures_v2/chart_metadata.json": meta}
    manifest["files"] = [f for f in manifest["files"] if f["path"] not in paths]
    for rel, path in paths.items():
        raw = path.read_bytes()
        manifest["files"].append({"path": rel, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                                   "role": "verification_chart_artifact"})
    manifest["verification_chart"] = {"png": "../figures_v2/henrich_v2_floor_rate.png",
                                       "pdf": "../figures_v2/henrich_v2_floor_rate.pdf",
                                       "caption": "../figures_v2/henrich_v2_floor_rate.caption.txt"}
    fd, tmp = tempfile.mkstemp(prefix="manifest.", suffix=".tmp", dir=manifest_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, manifest_path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    check = load(manifest_path)
    valid = all((manifest_path.parent / f["path"]).exists() and
                (manifest_path.parent / f["path"]).stat().st_size == f["bytes"] and
                hashlib.sha256((manifest_path.parent / f["path"]).read_bytes()).hexdigest() == f["sha256"]
                for f in check["files"])
    print(json.dumps({"png": str(png), "pdf": str(pdf), "caption": str(cap),
                      "manifest_files": len(check["files"]), "hashes_valid": valid}, indent=2))
    if not valid: raise SystemExit(1)


if __name__ == "__main__":
    main()
