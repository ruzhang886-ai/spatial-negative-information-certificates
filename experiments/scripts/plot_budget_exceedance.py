from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / "runs" / "budget_exceedance_12000_12499.json"
OUTPUT_DIR = BASE.parent / "figures"

COLORS = {
    "full_pair": "#D55E00",
    "temporal_repeat_q2": "#CC79A7",
    "spatial_multicover_q1": "#009E73",
    "spatial_multicover_q2": "#0072B2",
}
LABELS = {
    "full_pair": "Ideal pair",
    "temporal_repeat_q2": "Temporal repeat, q=2",
    "spatial_multicover_q1": "Spatial certificate, q=1",
    "spatial_multicover_q2": "Spatial certificate, q=2",
}


def main() -> None:
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    probabilities = list(payload["miss_probabilities"])
    summary = payload["summary"]
    methods = list(COLORS)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for method in methods:
        unsafe = [
            summary[f"{p:.2f}"][method]["unsafe_rate"]
            for p in probabilities
        ]
        time = [
            summary[f"{p:.2f}"][method]["mean_observed_time_s"]
            for p in probabilities
        ]
        axes[0].plot(
            probabilities,
            unsafe,
            marker="o",
            linewidth=2,
            color=COLORS[method],
            label=LABELS[method],
        )
        axes[2].plot(
            probabilities,
            time,
            marker="o",
            linewidth=2,
            color=COLORS[method],
            label=LABELS[method],
        )

    q2_exceeded = [
        summary[f"{p:.2f}"]["spatial_multicover_q2"]["budget_exceedance_rate"]
        for p in probabilities
    ]
    q2_unsafe = [
        summary[f"{p:.2f}"]["spatial_multicover_q2"]["unsafe_rate"]
        for p in probabilities
    ]
    axes[1].plot(
        probabilities,
        q2_exceeded,
        marker="s",
        linewidth=2,
        color="#E69F00",
        label="At least one certificate exceeds q=2",
    )
    axes[1].plot(
        probabilities,
        q2_unsafe,
        marker="o",
        linewidth=2,
        color=COLORS["spatial_multicover_q2"],
        label="Unsafe exclusion",
    )

    axes[0].set_title("Unsafe-exclusion risk")
    axes[0].set_ylabel("Fraction of runs")
    axes[1].set_title("Budget exceedance and observed failure")
    axes[1].set_ylabel("Fraction of runs")
    axes[2].set_title("Observed mission time")
    axes[2].set_ylabel("Mean time (s)")
    for axis in axes:
        axis.set_xlabel("Persistent-cell miss probability")
        axis.grid(alpha=0.22)
        axis.set_xlim(min(probabilities), max(probabilities))
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"budget_exceedance_risk.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
