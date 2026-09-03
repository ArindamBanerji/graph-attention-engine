"""Generate the measured gamma phase view from the two-arm raw runs."""

from pathlib import Path
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PATTERN = str(ROOT / "experiments" / "h_curve_parametric_regen" / "raw" / "two_arm_v4" / "runs" / "*" / "epsilon-*" / "arm-*" / "gamma.json")
OUTPUT = Path(__file__).with_name("gamma_phase_diagram.png")
BG = "#1a1a2e"
TEXT = "#e8e8e8"
AMBER = "#e67e22"
GRAY = "#7f8c8d"
GREEN = "#27ae60"
RED_ORANGE = "#e74c3c"


def load_records() -> list[tuple[str, float, str, float]]:
    records = []
    for filename in sorted(glob.glob(PATTERN)):
        data = json.loads(Path(filename).read_text(encoding="utf-8"))
        gamma = next((data.get(key) for key in ("gamma", "gamma_final", "gamma_rate")
                      if data.get(key) is not None), None)
        if gamma is None:
            continue
        arm = os.path.basename(os.path.dirname(filename))
        epsilon_name = os.path.basename(os.path.dirname(os.path.dirname(filename)))
        seed_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(filename))))
        records.append((seed_name, float(epsilon_name.removeprefix("epsilon-")), arm, float(gamma)))
    return records


def main() -> None:
    records = load_records()
    if not records:
        raise RuntimeError("No measured gamma values found in the raw experiment files")
    epsilons = sorted({item[1] for item in records})
    arms = sorted({item[2] for item in records})
    x_positions = {value: index for index, value in enumerate(epsilons)}
    y_positions = {value: index for index, value in enumerate(arms)}
    values = np.asarray([item[3] for item in records])

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT})
    fig, ax = plt.subplots(figsize=(2048 / 300, 1152 / 300), dpi=300)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axvspan(-0.5, len(epsilons) - 0.5, color=GREEN, alpha=0.07)
    offsets = np.linspace(-0.12, 0.12, max(1, len(records)))
    for offset, (seed, epsilon, arm, gamma) in zip(offsets, records):
        ax.scatter(x_positions[epsilon], y_positions[arm] + offset, s=220, color=AMBER,
                   edgecolor=TEXT, linewidth=0.8, zorder=3)
        ax.text(x_positions[epsilon], y_positions[arm] + offset, f"{gamma:.2f}",
                ha="center", va="center", fontsize=8, color=BG, weight="bold", zorder=4)
        ax.annotate(seed.removeprefix("seed-"),
                    (x_positions[epsilon], y_positions[arm] + offset), xytext=(26, 0),
                    textcoords="offset points", ha="center", fontsize=7, color=GRAY)
    ax.set_title("Re-convergence γ by ε★", loc="left", fontsize=14, pad=18, color=TEXT, weight="bold")
    ax.text(0.0, 1.02, "Measured phase view; cells without a gamma estimate remain unfilled",
            transform=ax.transAxes, fontsize=9, color=TEXT)
    ax.set_xlabel("Configured ε (unitless)", fontsize=10, labelpad=8, color=TEXT)
    ax.set_ylabel("Arm (categorical)", fontsize=10, labelpad=8, color=TEXT)
    ax.set_xticks(range(len(epsilons)), [f"{value:.2f}" for value in epsilons])
    ax.set_yticks(range(len(arms)), arms)
    ax.set_xlim(-0.5, len(epsilons) - 0.5)
    ax.set_ylim(-0.6, len(arms) - 0.4)
    ax.grid(color=GRAY, alpha=0.18, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color(GRAY)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.text(0.99, 0.04, f"{len(records)} measured cells; {len(epsilons)} ε settings represented",
            transform=ax.transAxes, ha="right", fontsize=8, color=GREEN)
    fig.text(0.99, 0.025,
             "Source: experiments/h_curve_parametric_regen/raw/two_arm_v4/runs/*/epsilon-*/arm-*/gamma.json",
             ha="right", fontsize=7, color=GRAY)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.83, bottom=0.18)
    fig.savefig(OUTPUT, dpi=300, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    main()
