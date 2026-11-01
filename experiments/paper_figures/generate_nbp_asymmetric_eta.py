"""Generate the asymmetric-eta chart from the recorded G1 sweep checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "cross-graph-experiments" / "experiments" / "block5b_proxy" / "results" / "eta_override_sweep.json"
OUTPUT = Path(__file__).resolve().parent / "nbp_asymmetric_eta.png"


def main() -> None:
    with SOURCE.open(encoding="utf-8") as handle:
        data = json.load(handle)
    g1 = data["results"]["G1"]
    days = np.array([1.0, 30.0, 60.0])
    symmetric = np.array([g1["0.05"][f"acc_day{int(day)}"] for day in days]) * 100.0
    asymmetric = np.array([g1["0.01"][f"acc_day{int(day)}"] for day in days]) * 100.0

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(2048 / 300, 1152 / 300), dpi=300, facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.axhspan(float(symmetric.min()), 100, color="#e74c3c", alpha=0.08, zorder=0)
    ax.axhspan(float(asymmetric.min()), 100, color="#27ae60", alpha=0.08, zorder=0)
    ax.plot(days, symmetric, "o--", color="#7f8c8d", linewidth=3, markersize=10, label=r"Symmetric $eta=0.05$")
    ax.plot(days, asymmetric, "o-", color="#e67e22", linewidth=3, markersize=10, label=r"Asymmetric $eta_{override}=0.01$")
    for x, y in zip(days, symmetric):
        ax.annotate(f"{y:.1f}%", (x, y), xytext=(0, -18), textcoords="offset points", ha="center", color="#e8e8e8", fontsize=9)
    for x, y in zip(days, asymmetric):
        ax.annotate(f"{y:.1f}%", (x, y), xytext=(0, 10), textcoords="offset points", ha="center", color="#e8e8e8", fontsize=9)
    gap = float(asymmetric[-1] - symmetric[-1])
    ax.annotate(f"{gap:+.1f} percentage points\nat day 60", xy=(60, asymmetric[-1]), xytext=(42, 95),
                arrowprops={"arrowstyle": "->", "color": "#27ae60", "lw": 2}, color="#27ae60",
                ha="center", va="center", fontsize=9)
    ax.set_title("Asymmetric Learning-Rate Operating Envelope", color="#e8e8e8", pad=12, weight="bold", fontsize=14)
    ax.text(0.5, 1.01, "Recorded G1 accuracy checkpoints under confirm/override learning rates",
            transform=ax.transAxes, ha="center", color="#e8e8e8", fontsize=10)
    ax.set_xlabel("Elapsed experiment time (days)", color="#e8e8e8", fontsize=10)
    ax.set_ylabel("Accuracy (%)", color="#e8e8e8", fontsize=10)
    ax.set_xlim(0, 65)
    ax.set_ylim(82, 98)
    ax.grid(True, color="#7f8c8d", alpha=0.25)
    ax.tick_params(colors="#e8e8e8")
    for spine in ax.spines.values():
        spine.set_color("#7f8c8d")
    ax.legend(facecolor="#1a1a2e", edgecolor="#7f8c8d", labelcolor="#e8e8e8", loc="lower left")
    ax.text(0.01, 0.02, "Source: block5b_proxy/results/eta_override_sweep.json · G1 checkpoints",
            transform=ax.transAxes, color="#e8e8e8", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, facecolor="#1a1a2e")
    plt.close(fig)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
