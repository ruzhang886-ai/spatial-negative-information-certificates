from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "runs" / "holdout_10000_10499.json"
FIGURES = BASE.parent / "figures"

SCALES = (10.0, 25.0, 50.0, 100.0)
METHODS = (
    "no_negative_information",
    "full_pair",
    "spatial_multicover_q1",
    "spatial_multicover_q2",
)
LABELS = {
    "no_negative_information": "No negative update",
    "full_pair": "Ideal pair",
    "spatial_multicover_q1": "Spatial q=1",
    "spatial_multicover_q2": "Spatial q=2",
}
COLORS = {
    "no_negative_information": "#4D4D4D",
    "full_pair": "#D55E00",
    "spatial_multicover_q1": "#56B4E9",
    "spatial_multicover_q2": "#009E73",
}


def bootstrap_saving(rows, scale: float, samples: int = 10000) -> tuple[float, float, float]:
    baseline = {
        row["seed"]: row
        for row in rows
        if row["miss_correlation_m"] == scale
        and row["policy"] == "no_negative_information"
    }
    robust = {
        row["seed"]: row
        for row in rows
        if row["miss_correlation_m"] == scale
        and row["policy"] == "spatial_multicover_q2"
    }
    savings = np.asarray(
        [
            baseline[seed]["time_s"] - robust[seed]["time_s"]
            for seed in sorted(baseline)
            if robust[seed]["completed"] and not robust[seed]["unsafe_exclusion"]
        ],
        dtype=float,
    )
    rng = np.random.default_rng(20261015 + int(scale))
    indices = rng.integers(0, len(savings), size=(samples, len(savings)))
    bootstrap = savings[indices].mean(axis=1)
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return float(savings.mean()), float(low), float(high)


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    rows = data["runs"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "figure.dpi": 160,
            "savefig.dpi": 320,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.55), constrained_layout=True)
    x = np.arange(len(SCALES), dtype=float)

    # (a) Unsafe exclusion rate with Wilson intervals.
    ax = axes[0]
    offsets = np.linspace(-0.24, 0.24, len(METHODS))
    for offset, method in zip(offsets, METHODS):
        rates, lower, upper = [], [], []
        for scale in SCALES:
            metric = data["summary"][str(scale)][method]
            rate = 100 * metric["unsafe_rate"]
            low, high = metric["unsafe_rate_wilson95"]
            rates.append(rate)
            lower.append(max(0.0, rate - 100 * low))
            upper.append(max(0.0, 100 * high - rate))
        ax.errorbar(
            x + offset,
            rates,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            markersize=5,
            capsize=2.5,
            color=COLORS[method],
            label=LABELS[method],
        )
    ax.set_title("(a) Unsafe exclusions")
    ax.set_ylabel("Rate (%) with Wilson 95% CI")
    ax.set_xticks(x, [str(int(scale)) for scale in SCALES])
    ax.set_xlabel("Correlation scale (m)")
    ax.set_ylim(bottom=-0.45)
    ax.grid(axis="y", alpha=0.25)

    # (b) Paired time saving against the safe baseline.
    ax = axes[1]
    means, lows, highs = zip(*(bootstrap_saving(rows, scale) for scale in SCALES))
    means = np.asarray(means)
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    ax.errorbar(
        x,
        means,
        yerr=np.vstack([means - lows, highs - means]),
        fmt="o-",
        linewidth=1.8,
        markersize=5.5,
        capsize=3,
        color=COLORS["spatial_multicover_q2"],
    )
    ax.axhline(0, color="#777777", linewidth=0.9, linestyle="--")
    ax.set_title("(b) Time saved by spatial q=2")
    ax.set_ylabel("Paired mean saving (s), bootstrap 95% CI")
    ax.set_xticks(x, [str(int(scale)) for scale in SCALES])
    ax.set_xlabel("Correlation scale (m)")
    ax.grid(axis="y", alpha=0.25)

    # (c) Safe fallback explains the large-scale time cost.
    ax = axes[2]
    for method in (
        "no_negative_information",
        "spatial_multicover_q1",
        "spatial_multicover_q2",
    ):
        rates = [
            100 * data["summary"][str(scale)][method]["fallback_rate"]
            for scale in SCALES
        ]
        ax.plot(
            x,
            rates,
            marker="o",
            linewidth=1.8,
            markersize=5,
            color=COLORS[method],
            label=LABELS[method],
        )
    ax.set_title("(c) Safe fallback usage")
    ax.set_ylabel("Fallback rate (%)")
    ax.set_xticks(x, [str(int(scale)) for scale in SCALES])
    ax.set_xlabel("Correlation scale (m)")
    ax.set_ylim(bottom=-1)
    ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    png = FIGURES / "holdout_reliability_time.png"
    pdf = FIGURES / "holdout_reliability_time.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
