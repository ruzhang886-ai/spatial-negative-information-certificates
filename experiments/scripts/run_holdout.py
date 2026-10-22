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


START_SEED = 10000
SEEDS = 500
MISS_PROBABILITY = 0.10
SCALES_M = (10.0, 25.0, 50.0, 100.0)
METHODS = (
    "no_negative_information",
    "full_pair",
    "spatial_multicover_q1",
    "spatial_multicover_q2",
)
OUTPUT = BASE / "runs" / "holdout_10000_10499.json"


def average(values):
    return statistics.mean(values) if values else None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [center - half, center + half]


def summarize(subset: list[dict]) -> dict:
    completed = [row for row in subset if row["completed"]]
    unsafe = sum(row["unsafe_exclusion"] for row in subset)
    times = [row["time_s"] for row in completed]
    return {
        "runs": len(subset),
        "completed": len(completed),
        "completion_rate": len(completed) / len(subset),
        "completion_rate_wilson95": wilson_interval(len(completed), len(subset)),
        "unsafe_exclusions": unsafe,
        "unsafe_rate": unsafe / len(subset),
        "unsafe_rate_wilson95": wilson_interval(unsafe, len(subset)),
        "mean_time_completed_s": average(times),
        "median_time_completed_s": float(np.median(times)) if times else None,
        "p90_time_completed_s": float(np.percentile(times, 90)) if times else None,
        "mean_observed_time_until_completion_or_failure_s": average(
            [row["time_s"] for row in subset]
        ),
        "mean_measures": average([row["measures"] for row in subset]),
        "fallback_rate": average([row["fallback_used"] for row in subset]),
    }


def paired_time(rows: list[dict], scale: float, left: str, right: str) -> dict:
    by_key = {(row["seed"], row["policy"]): row for row in rows if row["miss_correlation_m"] == scale}
    differences = []
    for seed in range(START_SEED, START_SEED + SEEDS):
        a = by_key[(seed, left)]
        b = by_key[(seed, right)]
        if a["completed"] and b["completed"] and not a["unsafe_exclusion"] and not b["unsafe_exclusion"]:
            differences.append(a["time_s"] - b["time_s"])
    return {
        "pair": f"{left}_minus_{right}",
        "eligible_pairs": len(differences),
        "mean_difference_s": average(differences),
        "median_difference_s": float(np.median(differences)) if differences else None,
        "left_faster_rate": average([difference < 0 for difference in differences]),
    }


def main() -> None:
    rows = []
    for scale in SCALES_M:
        model = ModelConfig(
            miss_probability=MISS_PROBABILITY,
            miss_mode="persistent_location",
            miss_correlation_m=scale,
        )
        for seed in range(START_SEED, START_SEED + SEEDS):
            scenario = generate_scenario(seed, model)
            for policy in METHODS:
                row = run_policy(scenario, policy, model=model)
                row["miss_probability"] = MISS_PROBABILITY
                row["miss_correlation_m"] = scale
                rows.append(row)

    summary = {}
    paired = {}
    for scale in SCALES_M:
        summary[str(scale)] = {}
        for policy in METHODS:
            subset = [
                row
                for row in rows
                if row["miss_correlation_m"] == scale and row["policy"] == policy
            ]
            summary[str(scale)][policy] = summarize(subset)
        paired[str(scale)] = [
            paired_time(rows, scale, "spatial_multicover_q2", "no_negative_information"),
            paired_time(rows, scale, "spatial_multicover_q2", "spatial_multicover_q1"),
        ]

    payload = {
        "label": "frozen_algorithm_independent_holdout",
        "seed_range": [START_SEED, START_SEED + SEEDS - 1],
        "miss_probability": MISS_PROBABILITY,
        "miss_mode": "persistent_location",
        "scales_m": SCALES_M,
        "methods": METHODS,
        "summary": summary,
        "paired_time_comparisons": paired,
        "runs": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "paired": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
