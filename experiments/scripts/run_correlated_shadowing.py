from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.policies import run_policy
from src.simulator import ModelConfig, generate_scenario


START_SEED = 13000
SEEDS = 500
MISS_PROBABILITY = 0.10
SCALES_M = (10.0, 25.0, 50.0, 100.0)
METHODS = (
    "no_negative_information",
    "full_pair",
    "temporal_repeat_q2",
    "spatial_multicover_q1",
    "spatial_multicover_q2",
)
OUTPUT = BASE / "runs" / "correlated_shadowing_13000_13499.json"


def average(values):
    return statistics.mean(values) if values else None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return [center - half, center + half]


def summarize(rows: list[dict]) -> dict:
    completed = [row for row in rows if row["completed"]]
    unsafe = sum(row["unsafe_exclusion"] for row in rows)
    exceeded = sum(row["budget_exceeded_certificates"] > 0 for row in rows)
    unsafe_inside_budget = sum(
        row["unsafe_exclusion"]
        and row["budget_exceeded_certificates"] == 0
        for row in rows
    )
    completed_times = [row["time_s"] for row in completed]
    return {
        "runs": len(rows),
        "completed": len(completed),
        "completion_rate": len(completed) / len(rows),
        "unsafe_exclusions": unsafe,
        "unsafe_rate": unsafe / len(rows),
        "unsafe_rate_wilson95": wilson_interval(unsafe, len(rows)),
        "runs_with_budget_exceedance": exceeded,
        "budget_exceedance_rate": exceeded / len(rows),
        "unsafe_without_recorded_exceedance": unsafe_inside_budget,
        "mean_time_completed_s": average(completed_times),
        "median_time_completed_s": (
            float(np.median(completed_times)) if completed_times else None
        ),
        "mean_observed_time_s": average([row["time_s"] for row in rows]),
        "mean_measures": average([row["measures"] for row in rows]),
        "fallback_rate": average([row["fallback_used"] for row in rows]),
    }


def main() -> None:
    rows = []
    for scale in SCALES_M:
        model = ModelConfig(
            miss_probability=MISS_PROBABILITY,
            miss_mode="correlated_shadowing",
            miss_correlation_m=scale,
            shadowing_features=512,
        )
        for seed in range(START_SEED, START_SEED + SEEDS):
            scenario = generate_scenario(seed, model)
            for policy in METHODS:
                row = run_policy(scenario, policy, model=model)
                row.pop("events", None)
                row["miss_probability"] = MISS_PROBABILITY
                row["miss_correlation_m"] = scale
                rows.append(row)

    summary = {}
    for scale in SCALES_M:
        summary[str(scale)] = {}
        for policy in METHODS:
            subset = [
                row
                for row in rows
                if row["miss_correlation_m"] == scale
                and row["policy"] == policy
            ]
            summary[str(scale)][policy] = summarize(subset)

    payload = {
        "label": "continuous_correlated_shadowing_external_validity",
        "seed_range": [START_SEED, START_SEED + SEEDS - 1],
        "miss_probability": MISS_PROBABILITY,
        "miss_mode": "correlated_shadowing",
        "target_correlation": "exp(-distance/correlation_length)",
        "shadowing_features": 512,
        "scales_m": SCALES_M,
        "methods": METHODS,
        "summary": summary,
        "runs": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
