from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.policies import run_policy
from src.simulator import ModelConfig, generate_scenario


METHODS = (
    "no_negative_information",
    "full_pair",
    "spatial_multicover_q1",
    "spatial_multicover_q2",
)


def average(values):
    return statistics.mean(values) if values else None


def summarize(rows, scale: float) -> dict:
    result = {}
    for policy in METHODS:
        subset = [
            row
            for row in rows
            if row["policy"] == policy and row["miss_correlation_m"] == scale
        ]
        completed = [row for row in subset if row["completed"]]
        result[policy] = {
            "runs": len(subset),
            "completed": sum(row["completed"] for row in subset),
            "completion_rate": average([float(row["completed"]) for row in subset]),
            "unsafe_exclusions": sum(row["unsafe_exclusion"] for row in subset),
            "mean_time_completed_s": average([row["time_s"] for row in completed]),
            "mean_observed_time_until_completion_or_failure_s": average(
                [row["time_s"] for row in subset]
            ),
            "mean_measures": average([row["measures"] for row in subset]),
            "fallback_rate": average([row["fallback_used"] for row in subset]),
            "mean_pair_cuts": average([row["pair_cuts"] for row in subset]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--start-seed", type=int, default=2000)
    parser.add_argument("--miss-probability", type=float, default=0.10)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=(10.0, 25.0, 50.0, 100.0),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "runs" / "correlation_scale_scan.json",
    )
    args = parser.parse_args()

    rows = []
    for scale in args.scales:
        model = ModelConfig(
            miss_probability=args.miss_probability,
            miss_mode="persistent_location",
            miss_correlation_m=scale,
        )
        for seed in range(args.start_seed, args.start_seed + args.seeds):
            scenario = generate_scenario(seed, model)
            for policy in METHODS:
                row = run_policy(scenario, policy, model=model)
                row["miss_probability"] = args.miss_probability
                row["miss_correlation_m"] = scale
                rows.append(row)

    summary = {str(scale): summarize(rows, scale) for scale in args.scales}
    payload = {
        "label": "development_correlation_scale_scan_not_for_publication",
        "miss_mode": "persistent_location",
        "miss_probability": args.miss_probability,
        "scales_m": args.scales,
        "seed_range": [args.start_seed, args.start_seed + args.seeds - 1],
        "summary": summary,
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
