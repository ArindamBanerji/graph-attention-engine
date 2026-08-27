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
RAW = ROOT / "raw"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    cells = load(RAW / "summary.json")["cells"]
    arms = ["unenriched", "enriched"]
    labels = {"unenriched": "UNENRICHED", "enriched": "ENRICHED"}
    colors = {"unenriched": "#0072B2", "enriched": "#D55E00"}
    fig, (floor_ax, rate_ax) = plt.subplots(1, 2, figsize=(11.8, 5.8))
    x = np.arange(2)
    for arm in arms:
        rows = [r for r in cells if r["arm"] == arm]
        floor = [r["floor_day1_accuracy"] for r in rows]
        kvals = [r["rate"]["k"] for r in rows]
        floor_ax.scatter(np.full(3, x[arms.index(arm)]), floor, color=colors[arm], s=62,
                         marker="o", edgecolor="white", linewidth=.6, label=f"{labels[arm]} seeds")
        floor_ax.scatter([x[arms.index(arm)]], [np.mean(floor)], color=colors[arm], marker="D",
                         s=100, edgecolor="black", linewidth=.7, zorder=5)
        rate_ax.scatter(np.full(3, x[arms.index(arm)]), kvals, color=colors[arm], s=62,
                        marker="x", linewidth=2.1, label=f"{labels[arm]} k (invalid fits)")
        rate_ax.scatter([x[arms.index(arm)]], [np.mean(kvals)], color=colors[arm], marker="D",
                        s=100, edgecolor="black", linewidth=.7, zorder=5)
    floor_ax.set_xticks(x, ["UNENRICHED\nσ=.08 all factors", "ENRICHED\nσ=.04 factors 0–1"])
    floor_ax.set_ylabel("day-1 accuracy (mean decisions 1–10)")
    floor_ax.set_title("Floor: enrichment does not lift accuracy")
    floor_ax.set_ylim(0, .65)
    floor_ax.grid(axis="y", alpha=.25)
    floor_ax.legend(fontsize=8, loc="upper right")
    un_mean = float(np.mean([r["rate"]["k"] for r in cells if r["arm"] == "unenriched"]))
    lo, hi = sorted([un_mean * .8, un_mean * 1.2])
    rate_ax.axhspan(lo, hi, color="#009E73", alpha=.14, label="unenriched mean ±20%")
    rate_ax.axhline(un_mean, color="#333", linestyle="--", linewidth=1.2)
    rate_ax.set_xticks(x, ["UNENRICHED\nσ=.08 all factors", "ENRICHED\nσ=.04 factors 0–1"])
    rate_ax.set_ylabel("k (H-CURVE distance fit; negative = no decay)")
    rate_ax.set_title("Rate: no valid positive decay fit")
    rate_ax.grid(axis="y", alpha=.25)
    rate_ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("H-ENRICH: floor-versus-rate separation", fontsize=14, y=.98)
    fig.text(.5, .015, "Diamonds are means; × marks estimates from fits that failed the positive-decay validity gate.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=[0, .05, 1, .94])
    png = FIG / "henrich_floor_rate.png"
    pdf = FIG / "henrich_floor_rate.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    caption = (
        "H-ENRICH verification at ε_firm=0.20. Lowering noise on factors 0–1 did not lift the "
        "day-1 accuracy floor (mean 0.300 enriched versus 0.367 unenriched). All six distance "
        "trajectories produced negative k estimates and failed the positive-decay validity gate, "
        "so pure rate invariance is not estimable in this apparatus. n=3 is a geometric robustness "
        "check, not powered inference."
    )
    cap = FIG / "henrich_floor_rate.caption.txt"
    cap.write_text(caption + "\n", encoding="utf-8")
    metadata = {
        "schema_version": "henrich-chart-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scorer_imported": False, "rerun": False,
        "inputs": ["raw/summary.json", "raw/seed-*/unenriched/records.json", "raw/seed-*/enriched/records.json"],
        "outputs": [p.name for p in [png, pdf, cap]],
        "rate_reference": "unenriched mean with ±20% band; × values are fit-invalid negative estimates",
    }
    meta = FIG / "chart_metadata.json"
    meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    # Add chart artifacts to the existing manifest and verify every listed hash.
    manifest_path = RAW / "manifest.json"
    manifest = load(manifest_path)
    paths = {"../figures/henrich_floor_rate.png": png,
             "../figures/henrich_floor_rate.pdf": pdf,
             "../figures/henrich_floor_rate.caption.txt": cap,
             "../figures/chart_metadata.json": meta}
    manifest["files"] = [f for f in manifest["files"] if f["path"] not in paths]
    for rel, path in paths.items():
        raw = path.read_bytes()
        manifest["files"].append({"path": rel, "bytes": len(raw),
                                   "sha256": hashlib.sha256(raw).hexdigest(),
                                   "role": "verification_chart_artifact"})
    manifest["verification_chart"] = {"png": "../figures/henrich_floor_rate.png",
                                      "pdf": "../figures/henrich_floor_rate.pdf",
                                      "caption": "../figures/henrich_floor_rate.caption.txt"}
    fd, tmp = tempfile.mkstemp(prefix="manifest.", suffix=".tmp", dir=manifest_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, manifest_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    check = load(manifest_path)
    valid = all((manifest_path.parent / f["path"]).exists() and
                (manifest_path.parent / f["path"]).stat().st_size == f["bytes"] and
                hashlib.sha256((manifest_path.parent / f["path"]).read_bytes()).hexdigest() == f["sha256"]
                for f in check["files"])
    print(json.dumps({"png": str(png), "pdf": str(pdf), "caption": str(cap),
                      "manifest_files": len(check["files"]), "hashes_valid": valid}, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
