from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parents[1]
CONTINUOUS_DATA = BASE / "runs" / "continuous_gain_certificate_14700_14719.json"
POLICY_DATA = BASE / "runs" / "gain_aware_policy_14900_14999.json"
FIGURES = BASE.parent / "figures"
BEAMWIDTHS = (65.0, 90.0, 120.0)


def main() -> None:
    continuous = json.loads(CONTINUOUS_DATA.read_text(encoding="utf-8"))
    policy = json.loads(POLICY_DATA.read_text(encoding="utf-8"))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.4,
            "figure.dpi": 160,
            "savefig.dpi": 320,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(BEAMWIDTHS))
    width = 0.34
    pair_certified = [
        continuous["summary"][str(bw)]["pair"]["statuses"]["certified"]
        for bw in BEAMWIDTHS
    ]
    pair_counter = [
        continuous["summary"][str(bw)]["pair"]["statuses"]["counterexample"]
        for bw in BEAMWIDTHS
    ]
    q2_certified = [
        continuous["summary"][str(bw)]["spatial_q2"]["statuses"]["certified"]
        for bw in BEAMWIDTHS
    ]
    q2_counter = [
        continuous["summary"][str(bw)]["spatial_q2"]["statuses"]["counterexample"]
        for bw in BEAMWIDTHS
    ]
    ax.bar(
        x - width / 2,
        pair_certified,
        width,
        color="#0072B2",
        label="Continuously certified",
    )
    ax.bar(
        x - width / 2,
        pair_counter,
        width,
        bottom=pair_certified,
        color="#D55E00",
        label="Counterexample found",
    )
    ax.bar(
        x + width / 2,
        q2_certified,
        width,
        color="#0072B2",
        hatch="//",
    )
    ax.bar(
        x + width / 2,
        q2_counter,
        width,
        bottom=q2_certified,
        color="#D55E00",
        hatch="//",
    )
    for index in x:
        ax.text(index - width / 2, 20.5, "pair", ha="center", va="bottom", fontsize=8)
        ax.text(index + width / 2, 20.5, "q=2", ha="center", va="bottom", fontsize=8)
    ax.set_title("(a) Continuous-state audit (20 cases)")
    ax.set_xlabel("Half-power beamwidth (deg)")
    ax.set_ylabel("Candidate sets")
    ax.set_xticks(x, [str(int(value)) for value in BEAMWIDTHS])
    ax.set_ylim(0, 23)
    ax.grid(axis="y", alpha=0.25)

    series = (
        ("no_negative_update", "No negative update", "#4D4D4D", "o"),
        ("full_pair_gain_aware", "Gain-aware pair", "#E69F00", "o"),
        ("spatial_q2_gain_aware", "Gain-aware spatial q=2", "#009E73", "s"),
    )
    ax = axes[1]
    for key, label, color, marker in series:
        values = [
            policy["summary"][str(bw)][key]["mean_observed_time_s"]
            for bw in BEAMWIDTHS
        ]
        ax.plot(
            BEAMWIDTHS,
            values,
            label=label,
            color=color,
            marker=marker,
            linewidth=1.9,
        )
    ax.set_title("(b) Observed mission time (100 cases)")
    ax.set_xlabel("Half-power beamwidth (deg)")
    ax.set_ylabel("Mean time (s)")
    ax.set_xticks(BEAMWIDTHS)
    ax.grid(alpha=0.25)

    ax = axes[2]
    for key, label, color, marker in series:
        values = [
            100 * policy["summary"][str(bw)][key]["fallback_rate"]
            for bw in BEAMWIDTHS
        ]
        ax.plot(
            BEAMWIDTHS,
            values,
            color=color,
            marker=marker,
            linewidth=1.9,
        )
    ax.set_title("(c) Safe fallback usage")
    ax.set_xlabel("Half-power beamwidth (deg)")
    ax.set_ylabel("Fallback rate (%)")
    ax.set_xticks(BEAMWIDTHS)
    ax.set_ylim(-3, 105)
    ax.grid(alpha=0.25)

    handles_a, labels_a = axes[0].get_legend_handles_labels()
    handles_b, labels_b = axes[1].get_legend_handles_labels()
    fig.legend(
        handles_a + handles_b,
        labels_a + labels_b,
        loc="outside upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.10),
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    png = FIGURES / "gain_aware_certificate_audit.png"
    pdf = FIGURES / "gain_aware_certificate_audit.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
