from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "runs" / "gain_parameter_sensitivity_15100_15109.json"
FIGURES = BASE.parent / "figures"


def key(beamwidth: float, attenuation: float, exponent: float) -> str:
    return f"bw={beamwidth:.1f}|amax={attenuation:.1f}|n={exponent:.1f}"


def status_text(result: dict) -> str:
    rates = (
        ("C", result["certification_rate"]),
        ("X", result["counterexample_rate"]),
        ("U", result["unresolved_rate"]),
    )
    return "/".join(f"{name}{100 * value:.0f}" for name, value in rates if value > 0)


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    beamwidths = [float(value) for value in data["beamwidths_deg"]]
    attenuations = [float(value) for value in data["maximum_attenuations_db"]]
    exponents = [float(value) for value in data["path_loss_exponents"]]
    summary = data["summary"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.4,
            "figure.dpi": 160,
            "savefig.dpi": 320,
        }
    )
    fig, axes = plt.subplots(
        len(attenuations),
        4,
        figsize=(15.2, 7.2),
        constrained_layout=True,
    )

    all_runtime_ms = []
    for attenuation in attenuations:
        for exponent in exponents:
            for beamwidth in beamwidths:
                cell = summary[key(beamwidth, attenuation, exponent)]
                all_runtime_ms.extend(
                    [
                        1000 * cell["pair"]["median_elapsed_s"],
                        1000 * cell["spatial_q2"]["median_elapsed_s"],
                    ]
                )
    runtime_norm = LogNorm(vmin=max(20.0, min(all_runtime_ms)), vmax=max(all_runtime_ms))

    for row, attenuation in enumerate(attenuations):
        matrices: list[np.ndarray] = []
        annotations: list[list[list[str]]] = []
        for policy_name in ("pair", "spatial_q2"):
            certification = np.zeros((len(exponents), len(beamwidths)))
            labels: list[list[str]] = []
            for y, exponent in enumerate(exponents):
                label_row = []
                for x, beamwidth in enumerate(beamwidths):
                    result = summary[key(beamwidth, attenuation, exponent)][policy_name]
                    certification[y, x] = 100 * result["certification_rate"]
                    label_row.append(status_text(result))
                labels.append(label_row)
            matrices.append(certification)
            annotations.append(labels)

        for policy_name in ("pair", "spatial_q2"):
            runtime = np.zeros((len(exponents), len(beamwidths)))
            labels = []
            for y, exponent in enumerate(exponents):
                label_row = []
                for x, beamwidth in enumerate(beamwidths):
                    result = summary[key(beamwidth, attenuation, exponent)][policy_name]
                    value = 1000 * result["median_elapsed_s"]
                    runtime[y, x] = value
                    label_row.append(f"{value:.0f}")
                labels.append(label_row)
            matrices.append(runtime)
            annotations.append(labels)

        for col, (matrix, labels) in enumerate(zip(matrices, annotations)):
            ax = axes[row, col]
            if col < 2:
                image = ax.imshow(matrix, vmin=0, vmax=100, cmap="viridis", aspect="auto")
            else:
                image = ax.imshow(matrix, norm=runtime_norm, cmap="magma", aspect="auto")
            for y in range(len(exponents)):
                for x in range(len(beamwidths)):
                    value = matrix[y, x]
                    if col < 2:
                        text_color = "white" if value < 55 else "#1A1A1A"
                    else:
                        text_color = "white" if runtime_norm(value) < 0.58 else "#1A1A1A"
                    ax.text(
                        x,
                        y,
                        labels[y][x],
                        ha="center",
                        va="center",
                        fontsize=7.2,
                        color=text_color,
                    )
            ax.set_xticks(range(len(beamwidths)), [str(int(value)) for value in beamwidths])
            ax.set_yticks(range(len(exponents)), [f"{value:g}" for value in exponents])
            ax.set_xlabel("Half-power beamwidth (deg)")
            if col == 0:
                ax.set_ylabel(f"Path-loss exponent\n($A_{{max}}={attenuation:g}$ dB)")
            else:
                ax.tick_params(labelleft=False)
            if row == 0:
                titles = (
                    "Pair: certification outcome",
                    "Spatial q=2: certification outcome",
                    "Pair: median verifier time (ms)",
                    "Spatial q=2: median verifier time (ms)",
                )
                ax.set_title(titles[col], pad=8)

    certification_map = axes[0, 0].images[0]
    runtime_map = axes[0, 2].images[0]
    cert_bar = fig.colorbar(
        certification_map,
        ax=axes[:, :2],
        location="bottom",
        shrink=0.72,
        pad=0.08,
        aspect=38,
    )
    cert_bar.set_label("Continuously certified cases (%)   [C: certified, X: counterexample, U: unresolved]")
    runtime_bar = fig.colorbar(
        runtime_map,
        ax=axes[:, 2:],
        location="bottom",
        shrink=0.72,
        pad=0.08,
        aspect=38,
    )
    runtime_bar.set_label("Median continuous-verifier wall-clock time (ms, logarithmic scale)")

    FIGURES.mkdir(parents=True, exist_ok=True)
    png = FIGURES / "gain_parameter_sensitivity.png"
    pdf = FIGURES / "gain_parameter_sensitivity.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
