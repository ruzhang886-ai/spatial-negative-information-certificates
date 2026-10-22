from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parents[1]
RUNS = BASE / "runs"
FIGURES = BASE.parent / "figures"


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["thresholds"]


def main() -> None:
    development = load(RUNS / "rmdirectional_external_analysis.json")
    holdout = load(RUNS / "rmdirectional_external_holdout_analysis.json")
    thresholds = np.array([entry["threshold_db"] for entry in holdout])

    blue = "#3B6FB6"
    orange = "#D9822B"
    light_blue = "#8FB1DA"
    light_orange = "#E8B278"

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), dpi=180)
    ax = axes[0]
    hold_pair = np.array([entry["pair_all_negative_rate"] for entry in holdout])
    hold_q2 = np.array([entry["q2_all_negative_rate"] for entry in holdout])
    pair_ci = np.array([entry["pair_rate_cluster_bootstrap_95ci"] for entry in holdout])
    q2_ci = np.array([entry["q2_rate_cluster_bootstrap_95ci"] for entry in holdout])
    ax.errorbar(
        thresholds,
        100 * hold_pair,
        yerr=np.vstack((100 * (hold_pair - pair_ci[:, 0]), 100 * (pair_ci[:, 1] - hold_pair))),
        color=blue,
        marker="o",
        linewidth=2.0,
        capsize=4,
        label="Pair, holdout",
    )
    ax.errorbar(
        thresholds,
        100 * hold_q2,
        yerr=np.vstack((100 * (hold_q2 - q2_ci[:, 0]), 100 * (q2_ci[:, 1] - hold_q2))),
        color=orange,
        marker="s",
        linewidth=2.0,
        capsize=4,
        label=r"Spatial $q=2$, holdout",
    )
    ax.plot(
        thresholds,
        [100 * entry["pair_all_negative_rate"] for entry in development],
        color=light_blue,
        marker="o",
        linestyle="--",
        linewidth=1.6,
        label="Pair, development",
    )
    ax.plot(
        thresholds,
        [100 * entry["q2_all_negative_rate"] for entry in development],
        color=light_orange,
        marker="s",
        linestyle="--",
        linewidth=1.6,
        label=r"Spatial $q=2$, development",
    )
    ax.set_ylabel("All-negative rate (%)")
    ax.set_xlabel("Reception threshold (dB)")
    ax.set_xticks(thresholds)
    ax.grid(alpha=0.22)
    ax.text(-0.13, 1.03, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    ax.plot(
        thresholds,
        [100 * entry["pair_evaluable_rate"] for entry in holdout],
        color=blue,
        marker="o",
        linewidth=2.0,
    )
    ax.plot(
        thresholds,
        [100 * entry["q2_evaluable_rate"] for entry in holdout],
        color=orange,
        marker="s",
        linewidth=2.0,
    )
    ax.set_ylabel("Executable certificate rate (%)")
    ax.set_xlabel("Reception threshold (dB)")
    ax.set_xticks(thresholds)
    ax.set_ylim(45, 75)
    ax.grid(alpha=0.22)
    ax.text(-0.13, 1.03, "(b)", transform=ax.transAxes, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.6)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"rmdirectional_external_validation.{suffix}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
