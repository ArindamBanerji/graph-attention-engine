from __future__ import annotations
import hashlib, json, os, tempfile
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_v3"
FIG = ROOT / "figures_v3"
FIG.mkdir(parents=True, exist_ok=True)

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def main():
    cells = load(RAW / "summary.json")["cells"]
    eps = [0.35, 0.50]
    colors = {"unenriched": "#0072B2", "enriched": "#D55E00"}
    names = {"unenriched": "UNENRICHED", "enriched": "ENRICHED"}
    fig, (fa, ka) = plt.subplots(1, 2, figsize=(12.4, 5.9))
    x = np.arange(2); means = {}
    for arm in colors:
        means[arm] = {}
        for xi, e in enumerate(eps):
            rows = [r for r in cells if r["arm"] == arm and r["epsilon_firm"] == e]
            floors = [r["floor_day1_accuracy"] for r in rows]
            ks = [r["rate"]["k"] for r in rows]
            means[arm][e] = (float(np.mean(floors)), float(np.mean(ks)))
            px = x[xi] + (-.16 if arm == "unenriched" else .16)
            fa.scatter([px] * 3, floors, color=colors[arm], s=55, edgecolor="white", linewidth=.6, label=f"{names[arm]} seeds" if xi == 0 else None)
            fa.scatter([px], [np.mean(floors)], color=colors[arm], marker="D", s=88, edgecolor="black", linewidth=.7, zorder=5)
            ka.scatter([px] * 3, ks, color=colors[arm], s=55, edgecolor="white", linewidth=.6, label=f"{names[arm]} seeds" if xi == 0 else None)
            ka.scatter([px], [np.mean(ks)], color=colors[arm], marker="D", s=88, edgecolor="black", linewidth=.7, zorder=5)
    for xi, e in enumerate(eps):
        base = means["unenriched"][e][1]; lo, hi = sorted([base * .8, base * 1.2])
        ka.fill_between([x[xi] - .42, x[xi] + .42], [lo, lo], [hi, hi], color="#009E73", alpha=.14)
        ka.hlines(base, x[xi] - .42, x[xi] + .42, color="#333", linestyle="--", linewidth=1)
    fa.set_xticks(x, ["ε=.35", "ε=.50"]); fa.set_ylabel("aggregate floor accuracy\n(mean of first 10 decisions per cell)"); fa.set_title("Targeted enrichment: aggregate floor"); fa.set_ylim(.50, .75); fa.grid(axis="y", alpha=.25); fa.legend(fontsize=8, loc="lower right")
    ka.set_xticks(x, ["ε=.35", "ε=.50"]); ka.set_ylabel("fitted k (positive centroid-distance decay)"); ka.set_title("Rate invariance (green = baseline ±20%)"); ka.grid(axis="y", alpha=.25); ka.legend(fontsize=8, loc="lower right")
    fig.suptitle("H-ENRICH v3: decision-relevant factor targeting", fontsize=14, y=.98)
    fig.text(.5, .015, "Diamonds are means; points are the three seed geometries. Enriched factors are top-2 by GT separation per seed/ε.", ha="center", fontsize=8.3)
    fig.tight_layout(rect=[0, .05, 1, .94])
    png = FIG / "henrich_v3_floor_rate.png"; pdf = FIG / "henrich_v3_floor_rate.pdf"
    fig.savefig(png, dpi=300); fig.savefig(pdf); plt.close(fig)
    cap = FIG / "henrich_v3_floor_rate.caption.txt"
    cap.write_text("H-ENRICH v3. Targeting the top two per-factor GT separation coordinates gives aggregate floor diffs of +1.25pp, −0.42pp, +4.58pp at ε=.35 and −1.25pp, −2.08pp, −0.42pp at ε=.50. The n=3 screen is sign-inconsistent at both points and is NULL for this apparatus; the production +5pp result remains Tier-2 observed-in-production evidence.\n", encoding="utf-8")
    meta = FIG / "chart_metadata.json"
    meta.write_text(json.dumps({"schema_version":"henrich-v3-chart-v1","created_utc":datetime.now(timezone.utc).isoformat(),"scorer_imported":False,"rerun":False,"inputs":["raw_v3/summary.json","raw_v3/seed-*/epsilon-*/separation.json","raw_v3/seed-*/epsilon-*/unenriched/records.json","raw_v3/seed-*/epsilon-*/enriched/records.json"],"outputs":[png.name,pdf.name,cap.name],"rate_band":"unenriched k ±20% per epsilon"}, indent=2)+"\n", encoding="utf-8")
    mp = RAW / "manifest.json"; manifest = load(mp)
    paths = {"../figures_v3/henrich_v3_floor_rate.png":png,"../figures_v3/henrich_v3_floor_rate.pdf":pdf,"../figures_v3/henrich_v3_floor_rate.caption.txt":cap,"../figures_v3/chart_metadata.json":meta}
    manifest["files"] = [f for f in manifest["files"] if f["path"] not in paths]
    for rel, path in paths.items():
        raw = path.read_bytes(); manifest["files"].append({"path":rel,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"role":"verification_chart_artifact"})
    manifest["verification_chart"] = {"png":"../figures_v3/henrich_v3_floor_rate.png","pdf":"../figures_v3/henrich_v3_floor_rate.pdf","caption":"../figures_v3/henrich_v3_floor_rate.caption.txt"}
    fd, tmp = tempfile.mkstemp(prefix="manifest.", suffix=".tmp", dir=mp.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as h: json.dump(manifest,h,indent=2,sort_keys=True); h.write("\n")
        os.replace(tmp,mp)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    check=load(mp); valid=all((mp.parent/f["path"]).exists() and (mp.parent/f["path"]).stat().st_size==f["bytes"] and hashlib.sha256((mp.parent/f["path"]).read_bytes()).hexdigest()==f["sha256"] for f in check["files"])
    print(json.dumps({"png":str(png),"pdf":str(pdf),"caption":str(cap),"manifest_files":len(check["files"]),"hashes_valid":valid},indent=2))
    if not valid: raise SystemExit(1)

if __name__ == "__main__": main()
