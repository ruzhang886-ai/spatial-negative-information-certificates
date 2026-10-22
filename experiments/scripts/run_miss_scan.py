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
    "repeated_pair_q1",
)


def mean(values):
    return statistics.mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=BASE / "runs" / "miss_scan.json")
    args = parser.parse_args()

    rows = []
    for probability in (0.0, 0.01, 0.05, 0.10):
        model = ModelConfig(miss_probability=probability)
        for seed in range(args.start_seed, args.start_seed + args.seeds):
            scenario = generate_scenario(seed, model)
            for policy in METHODS:
                row = run_policy(scenario, policy, model=model)
                row["miss_probability"] = probability
                rows.append(row)

    summary = {}
    for probability in (0.0, 0.01, 0.05, 0.10):
        summary[str(probability)] = {}
        for policy in METHODS:
            subset = [
                row
                for row in rows
                if row["policy"] == policy and row["miss_probability"] == probability
            ]
            completed = [row for row in subset if row["completed"]]
            summary[str(probability)][policy] = {
                "runs": len(subset),
                "completed": sum(row["completed"] for row in subset),
                "completion_rate": mean([float(row["completed"]) for row in subset]),
                "unsafe_exclusions": sum(row["unsafe_exclusion"] for row in subset),
                "mean_time_completed_s": mean([row["time_s"] for row in completed]),
                "mean_observed_time_until_completion_or_failure_s": mean(
                    [row["time_s"] for row in subset]
                ),
                "mean_measures": mean([row["measures"] for row in subset]),
                "fallback_rate": mean([row["fallback_used"] for row in subset]),
                "mean_pair_cuts": mean([row["pair_cuts"] for row in subset]),
            }

    payload = {
        "label": "development_miss_scan_not_for_publication",
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
