"""Generate the discovery-scaling chart from the recorded 11-point CSV."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "cross-graph-experiments" / "experiments" / "exp3_multidomain_scaling" / "results" / "extended_scaling_data.csv"
FIT = ROOT.parent / "cross-graph-experiments" / "experiments" / "exp3_multidomain_scaling" / "results" / "extended_scaling_fit.json"
OUTPUT = Path(__file__).resolve().parent / "eq_scaling.png"


def main() -> None:
    values: dict[int, list[float]] = defaultdict(list)
    with DATA.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values[int(row["n_domains"])].append(float(row["total_discoveries"]))
    x = np.array(sorted(values), dtype=float)
    y = np.array([np.mean(values[int(n)]) for n in x], dtype=float)
    with FIT.open(encoding="utf-8") as handle:
        fit = json.load(handle)["extended_range"]
    exponent = float(fit["fitted_exponent"])
    r_squared = float(fit["r_squared"])
    coefficient = float(fit["a"])
    x_fit = np.linspace(x.min(), x.max(), 300)
    y_fit = coefficient * x_fit**exponent

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(2048 / 300, 1152 / 300), dpi=300, facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.loglog(x, y, "o", color="#e67e22", markersize=11, label="Recorded mean discoveries")
    ax.loglog(x_fit, y_fit, color="#e67e22", linewidth=3, label=fr"Fit: $D(n)={coefficient:.1f}n^{{{exponent:.2f}}}$")
    ax.fill_between(x_fit, y_fit * 0.92, y_fit * 1.08, color="#27ae60", alpha=0.12, label="±8% visual band")
    ax.set_title("Discovery Scaling Across Domains", color="#e8e8e8", pad=12, weight="bold", fontsize=14)
    ax.text(0.5, 1.01, f"11 recorded domain-count points · exponent b={exponent:.2f} · R²={r_squared:.4f}",
            transform=ax.transAxes, ha="center", color="#e8e8e8", fontsize=10)
    ax.set_xlabel("Number of domains (count)", color="#e8e8e8", fontsize=10)
    ax.set_ylabel("Total discoveries (count)", color="#e8e8e8", fontsize=10)
    ax.grid(True, which="both", color="#7f8c8d", alpha=0.25)
    ax.tick_params(colors="#e8e8e8")
    for spine in ax.spines.values():
        spine.set_color("#7f8c8d")
    ax.legend(facecolor="#1a1a2e", edgecolor="#7f8c8d", labelcolor="#e8e8e8", loc="upper left")
    ax.text(0.01, 0.02, "Source: exp3_multidomain_scaling/results/extended_scaling_data.csv and extended_scaling_fit.json",
            transform=ax.transAxes, color="#e8e8e8", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, facecolor="#1a1a2e")
    plt.close(fig)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
