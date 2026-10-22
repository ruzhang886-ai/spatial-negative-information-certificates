from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "runs" / "correlated_shadowing_13000_13499.json"
FIGURES = BASE.parent / "figures"
SCALES = (10.0, 25.0, 50.0, 100.0)
METHODS = (
    "no_negative_information",
    "full_pair",
    "temporal_repeat_q2",
    "spatial_multicover_q1",
    "spatial_multicover_q2",
)
LABELS = {
    "no_negative_information": "No negative update",
    "full_pair": "Ideal pair",
    "temporal_repeat_q2": "Temporal repeat, q=2",
    "spatial_multicover_q1": "Spatial certificate, q=1",
    "spatial_multicover_q2": "Spatial certificate, q=2",
}
COLORS = {
    "no_negative_information": "#4D4D4D",
    "full_pair": "#D55E00",
    "temporal_repeat_q2": "#CC79A7",
    "spatial_multicover_q1": "#009E73",
    "spatial_multicover_q2": "#0072B2",
}
LINESTYLES = {
    "no_negative_information": "-",
    "full_pair": "-",
    "temporal_repeat_q2": "--",
    "spatial_multicover_q1": "-",
    "spatial_multicover_q2": "-",
}
MARKERS = {
    "no_negative_information": "o",
    "full_pair": "o",
    "temporal_repeat_q2": "x",
    "spatial_multicover_q1": "o",
    "spatial_multicover_q2": "o",
}


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
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), constrained_layout=True)

    ax = axes[0]
    for method in METHODS:
        values = [
            100 * data["summary"][str(scale)][method]["unsafe_rate"]
            for scale in SCALES
        ]
        ax.plot(
            SCALES,
            values,
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=3.6 if method == "full_pair" else 1.8,
            alpha=0.55 if method == "full_pair" else 1.0,
            color=COLORS[method],
            label=LABELS[method],
        )
    ax.set_title("(a) Unsafe exclusions")
    ax.set_xlabel("Shadowing correlation length (m)")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(SCALES)
    ax.set_ylim(bottom=-0.35)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for method in METHODS:
        values = [
            data["summary"][str(scale)][method]["mean_observed_time_s"]
            for scale in SCALES
        ]
        ax.plot(
            SCALES,
            values,
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=3.6 if method == "full_pair" else 1.8,
            alpha=0.55 if method == "full_pair" else 1.0,
            color=COLORS[method],
        )
    ax.set_title("(b) Observed mission time")
    ax.set_xlabel("Shadowing correlation length (m)")
    ax.set_ylabel("Mean time (s)")
    ax.set_xticks(SCALES)
    ax.grid(alpha=0.25)

    ax = axes[2]
    q2 = "spatial_multicover_q2"
    exceeded = [
        100 * data["summary"][str(scale)][q2]["budget_exceedance_rate"]
        for scale in SCALES
    ]
    unsafe = [
        100 * data["summary"][str(scale)][q2]["unsafe_rate"]
        for scale in SCALES
    ]
    fallback = [
        100 * data["summary"][str(scale)][q2]["fallback_rate"]
        for scale in SCALES
    ]
    ax.plot(
        SCALES,
        exceeded,
        marker="s",
        linewidth=1.8,
        color="#E69F00",
        label="Budget exceeded",
    )
    ax.plot(
        SCALES,
        unsafe,
        marker="o",
        linewidth=1.8,
        color=COLORS[q2],
        label="Unsafe exclusion",
    )
    ax.plot(
        SCALES,
        fallback,
        marker="^",
        linewidth=1.8,
        color="#7A7A7A",
        label="Safe fallback",
    )
    ax.set_title("(c) Spatial q=2 diagnostics")
    ax.set_xlabel("Shadowing correlation length (m)")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(SCALES)
    ax.set_ylim(bottom=-0.5)
    ax.grid(alpha=0.25)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.09),
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    png = FIGURES / "correlated_shadowing_validation.png"
    pdf = FIGURES / "correlated_shadowing_validation.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
