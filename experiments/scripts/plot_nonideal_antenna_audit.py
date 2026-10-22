from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "runs" / "nonideal_antenna_14000_14499.json"
FIGURES = BASE.parent / "figures"
BEAMWIDTHS = (65.0, 90.0, 120.0)
SERIES = (
    ("no_negative_update", "No negative update", "#4D4D4D", "o", "-"),
    ("full_pair_naive", "Original pair, 999 m", "#D55E00", "o", "-"),
    ("spatial_q2_naive", "Original spatial q=2, 999 m", "#0072B2", "o", "-"),
    (
        "full_pair_conservative",
        "Worst-angle radius (pair/q=2 identical)",
        "#009E73",
        "s",
        "--",
    ),
)


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.2,
            "figure.dpi": 160,
            "savefig.dpi": 320,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7), constrained_layout=True)

    ax = axes[0]
    edge = [
        data["beamwidth_metadata"][str(value)]["edge_guaranteed_radius_m"]
        for value in BEAMWIDTHS
    ]
    ax.plot(BEAMWIDTHS, edge, color="#E69F00", marker="o", linewidth=2)
    ax.axhline(999.0, color="#777777", linestyle="--", linewidth=1.2)
    ax.text(118.0, 970.0, "Original 999 m", ha="right", va="top", color="#666666")
    ax.set_title("(a) Worst-angle guaranteed radius")
    ax.set_xlabel("Half-power beamwidth (deg)")
    ax.set_ylabel("Radius (m)")
    ax.set_xticks(BEAMWIDTHS)
    ax.set_ylim(0, 1050)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for key, label, color, marker, linestyle in SERIES:
        values = [
            data["summary"][str(width)][key]["mean_observed_time_s"]
            for width in BEAMWIDTHS
        ]
        ax.plot(
            BEAMWIDTHS,
            values,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )
    ax.set_title("(b) Observed mission time")
    ax.set_xlabel("Half-power beamwidth (deg)")
    ax.set_ylabel("Mean time (s)")
    ax.set_xticks(BEAMWIDTHS)
    ax.grid(alpha=0.25)

    ax = axes[2]
    for key, label, color, marker, linestyle in SERIES:
        values = [
            100 * data["summary"][str(width)][key]["fallback_rate"]
            for width in BEAMWIDTHS
        ]
        ax.plot(
            BEAMWIDTHS,
            values,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
        )
    ax.set_title("(c) Safe fallback usage")
    ax.set_xlabel("Half-power beamwidth (deg)")
    ax.set_ylabel("Fallback rate (%)")
    ax.set_xticks(BEAMWIDTHS)
    ax.set_ylim(-2, 104)
    ax.grid(alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.09),
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    png = FIGURES / "nonideal_antenna_audit.png"
    pdf = FIGURES / "nonideal_antenna_audit.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
