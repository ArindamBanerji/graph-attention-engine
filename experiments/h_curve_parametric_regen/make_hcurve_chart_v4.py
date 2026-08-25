"""Generate the stand-alone H-CURVE paper figure from persisted JSON only."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
EPS = [0.05, 0.20, 0.35]
COLORS = {"A": "#0072B2", "B": "#D55E00", "C1": "#009E73", "C2": "#CC79A7"}
MARKERS = {"A": "o", "B": "s", "C1": "^", "C2": "P"}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def gamma_rows():
    fit = load(RAW / "two_arm_v4" / "fit_gate_recompute.json")["cells"]
    rows = [{"arm": r["arm"], "epsilon": r["epsilon_firm"], "seed": r["seed"],
             "gamma": r["gamma_rate"] if r["fit_valid_gate"] else None} for r in fit]
    names = {"C1_symmetric_rate": "C1", "C2_shift_triggered_boost": "C2"}
    for r in load(RAW / "arm_c_v4" / "summary.json")["cells"]:
        if r["policy"] in names:
            rows.append({"arm": names[r["policy"]], "epsilon": r["epsilon_firm"],
                         "seed": r["seed"],
                         "gamma": r["gamma_rate"] if r["fit_valid_gate"] else None})
    return rows


def plot_arm(ax, rows, arm, label):
    subset = [r for r in rows if r["arm"] == arm]
    for eps in EPS:
        cell = [r for r in subset if r["epsilon"] == eps]
        valid = [r["gamma"] for r in cell if r["gamma"] is not None]
        if valid:
            ax.scatter([eps] * len(valid), valid, color=COLORS[arm], marker=MARKERS[arm],
                       s=48, alpha=0.82, edgecolor="white", linewidth=0.5, zorder=4)
            ax.scatter([eps], [np.mean(valid)], color=COLORS[arm], marker="D", s=76,
                       edgecolor="black", linewidth=0.65, zorder=6)
        # Censored is a qualitative status, shown below the numeric range and never as gamma=0.
        if any(r["gamma"] is None for r in cell):
            ax.scatter([eps], [-0.16], color=COLORS[arm], marker="x", s=75,
                       linewidth=2.0, zorder=5)
    means = [(eps, float(np.mean([r["gamma"] for r in subset
                                  if r["epsilon"] == eps and r["gamma"] is not None])))
             for eps in EPS if any(r["epsilon"] == eps and r["gamma"] is not None for r in subset)]
    if means:
        ax.plot([x for x, _ in means], [y for _, y in means], color=COLORS[arm],
                linewidth=2.0, marker="D", markersize=6, label=label, zorder=5)


def trajectory(path: Path, phase: str, key: str):
    return np.asarray(load(path)[phase][key], dtype=float)


def main():
    rows = gamma_rows()
    fig, (ax, tx) = plt.subplots(1, 2, figsize=(13.5, 6.2),
                                 gridspec_kw={"width_ratios": [1.35, 1]})

    for arm, label in [("A", "A production"), ("B", "B theorem"),
                       ("C1", "C1 symmetric rate"), ("C2", "C2 shift-boost")]:
        plot_arm(ax, rows, arm, label)
    ax.axhspan(1.0, 2.4, color="#D9F0D3", alpha=0.48, zorder=0)
    ax.axhline(1.0, color="#222", linestyle="--", linewidth=1.25, zorder=2)
    ax.axvline(0.125, color="#555", linestyle=":", linewidth=1.25, zorder=2)
    ax.text(0.128, 2.28, r"$\epsilon_\star=0.125$", rotation=90, va="top", ha="left", fontsize=9)
    ax.text(0.345, 2.22, "re-convergence faster", ha="right", fontsize=9, color="#27632a")
    ax.set_xticks(EPS)
    ax.set_xticklabels(["0.05", "0.20", "0.35"])
    ax.set_xlabel(r"firm mismatch $\epsilon_{firm}$")
    ax.set_ylabel(r"$\gamma_{rate}=k_2/k_1$ (higher = faster re-convergence)")
    ax.set_title("Primary: rate ratio by arm")
    ax.set_ylim(-0.24, 2.4)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax.text(0.05, -0.215, "× = censored (not a numeric zero)", ha="center", fontsize=8)

    paths = {
        "A": ROOT / "raw/two_arm_v4/runs/seed-042/epsilon-0.35/arm-A/distance_trajectories.json",
        "B": ROOT / "raw/two_arm_v4/runs/seed-042/epsilon-0.35/arm-B/distance_trajectories.json",
        "C2": ROOT / "raw/arm_c_v4/runs/seed-042/epsilon-0.35/C2-shift-triggered-boost/distance_trajectories.json",
    }
    for arm, path in paths.items():
        d1 = trajectory(path, "phase1", "d_active")
        d2 = trajectory(path, "phase2", "d_disrupted")
        x1 = np.arange(len(d1))
        x2 = np.arange(len(d1), len(d1) + len(d2))
        tx.plot(x1, d1, color=COLORS[arm], linewidth=1.15, alpha=0.75, label=f"{arm} phase 1")
        tx.plot(x2, d2, color=COLORS[arm], linewidth=1.45, linestyle="--", label=f"{arm} phase 2")
    tx.axvline(len(d1), color="#777", linewidth=0.9, linestyle=":")
    tx.text(len(d1) + 12, tx.get_ylim()[1] * 0.98, "phase 2", fontsize=8, va="top")
    tx.set_title("Companion: representative trajectories\nseed 42, ε = 0.35")
    tx.set_xlabel("decision index (phase 2 dashed)")
    tx.set_ylabel("active-cell distance")
    tx.grid(alpha=0.25)
    tx.legend(fontsize=7.5, ncol=2, loc="upper right")

    fig.suptitle("H-CURVE: theorem re-convergence is not realized by production policy", fontsize=14, y=0.98)
    fig.text(0.5, 0.015,
             "Three independent firm-deviation geometries; diamonds are means. C3 omitted because C3 = C1.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=[0, 0.045, 1, 0.95])
    png = FIG / "hcurve_gamma_by_epsilon.png"
    pdf = FIG / "hcurve_gamma_by_epsilon.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    caption = (
        "H-CURVE verification. The theorem reference (Arm B) shows re-convergence above the "
        "threshold more often than not, whereas production (Arm A) does not; neither tested "
        "policy variant recovers the binary effect consistently (C1 symmetric rate, C2 shift-boost, "
        "and C3=C1). The gap is structural in this apparatus. n=3 provides geometric robustness, "
        "not powered population inference. ε=0.05 cells without a valid positive decay fit are "
        "shown as censored markers, never as zero."
    )
    (FIG / "hcurve_gamma_by_epsilon.caption.txt").write_text(caption + "\n", encoding="utf-8")
    metadata = {
        "schema_version": "hcurve-verification-chart-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scorer_imported": False,
        "rerun": False,
        "inputs": [
            "raw/two_arm_v4/fit_gate_recompute.json",
            "raw/arm_c_v4/summary.json",
            "raw/two_arm_v4/runs/seed-042/epsilon-0.35/arm-A/distance_trajectories.json",
            "raw/two_arm_v4/runs/seed-042/epsilon-0.35/arm-B/distance_trajectories.json",
            "raw/arm_c_v4/runs/seed-042/epsilon-0.35/C2-shift-triggered-boost/distance_trajectories.json",
        ],
        "series": ["A production", "B theorem", "C1 symmetric rate", "C2 shift-boost"],
        "c3_note": "C3 omitted from the plot because persisted C3 values are identical to C1.",
        "censoring": "x markers below the numeric range; no censored value is assigned gamma=0",
        "outputs": [p.name for p in [png, pdf, FIG / "hcurve_gamma_by_epsilon.caption.txt"]],
    }
    (FIG / "chart_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"png": str(png), "pdf": str(pdf), "caption": str(FIG / 'hcurve_gamma_by_epsilon.caption.txt')}, indent=2))


if __name__ == "__main__":
    main()
