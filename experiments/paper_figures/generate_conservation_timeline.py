"""Generate the conservation timeline from the recorded self-compute runs."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "self_compute_v3" / "raw" / "summary.json"
OUTPUT = Path(__file__).with_name("conservation_timeline.png")
BG = "#1a1a2e"
TEXT = "#e8e8e8"
AMBER = "#e67e22"
GRAY = "#7f8c8d"
GREEN = "#27ae60"
RED_ORANGE = "#e74c3c"


def load_engagement() -> tuple[np.ndarray, list[str]]:
    """Return the recorded Arm C engagement indicators, one row per seed."""
    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = []
    labels = []
    for seed in sorted(raw, key=int):
        status = raw[seed]["C"]["status"]
        rows.append(np.asarray([1.0 if item == "ENGAGED" else 0.0 for item in status]))
        labels.append(f"seed {seed}")
    return np.vstack(rows), labels


def main() -> None:
    rows, labels = load_engagement()
    x = np.arange(1, rows.shape[1] + 1)
    aggregate = rows.mean(axis=0)

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT})
    fig, ax = plt.subplots(figsize=(2048 / 300, 1152 / 300), dpi=300)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for row, label in zip(rows, labels):
        ax.plot(x, row, color=GRAY, alpha=0.42, linewidth=0.8, label=label)
    ax.plot(x, aggregate, color=AMBER, linewidth=2.3, label="mean across seeds")
    ax.axhspan(0.5, 1.0, color=GREEN, alpha=0.10)
    ax.axhspan(0.0, 0.5, color=RED_ORANGE, alpha=0.08)
    ax.axhline(0.5, color=TEXT, linestyle=(0, (4, 4)), linewidth=0.8, alpha=0.65)
    ax.text(0.99, 0.88, "engaged-state operating zone", transform=ax.transAxes,
            ha="right", color=GREEN, fontsize=8)
    ax.text(0.99, 0.12, "paused-state operating zone", transform=ax.transAxes,
            ha="right", color=RED_ORANGE, fontsize=8)
    ax.set_title("Conservation Law Transition", loc="left", fontsize=14, pad=18, color=TEXT, weight="bold")
    ax.text(0.0, 1.02, "Recorded engagement state across the Arm C decision timeline",
            transform=ax.transAxes, fontsize=9, color=TEXT)
    ax.set_xlabel("Decision index (unitless)", fontsize=10, labelpad=8, color=TEXT)
    ax.set_ylabel("Engaged-state share (0–1)", fontsize=10, labelpad=8, color=TEXT)
    ax.set_xlim(1, rows.shape[1])
    ax.set_ylim(-0.05, 1.05)
    ax.grid(color=GRAY, alpha=0.18, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color(GRAY)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8)
    fig.text(0.99, 0.035, "Source: experiments/self_compute_v3/raw/summary.json; status fields, Arm C",
             ha="right", fontsize=7, color=GRAY)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.83, bottom=0.18)
    fig.savefig(OUTPUT, dpi=300, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    main()
