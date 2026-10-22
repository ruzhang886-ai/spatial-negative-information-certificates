from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.policies import CORE_POLICIES, run_policy
from src.simulator import ModelConfig, generate_scenario


def summarize(rows: list[dict]) -> dict:
    result = {}
    for policy in CORE_POLICIES:
        subset = [row for row in rows if row["policy"] == policy]
        result[policy] = {
            "runs": len(subset),
            "completed": sum(row["completed"] for row in subset),
            "unsafe_exclusions": sum(row["unsafe_exclusion"] for row in subset),
            "mean_time_s": statistics.mean(row["time_s"] for row in subset),
            "median_time_s": statistics.median(row["time_s"] for row in subset),
            "max_time_s": max(row["time_s"] for row in subset),
            "mean_distance_m": statistics.mean(row["distance_m"] for row in subset),
            "mean_measures": statistics.mean(row["measures"] for row in subset),
            "fallback_rate": statistics.mean(row["fallback_used"] for row in subset),
            "mean_pair_cuts": statistics.mean(row["pair_cuts"] for row in subset),
            "mean_early_cancellations": statistics.mean(
                row["early_cancellations"] for row in subset
            ),
        }
    baseline = {row["seed"]: row for row in rows if row["policy"] == "no_negative_information"}
    paired = {}
    for policy in CORE_POLICIES[1:]:
        differences = [
            baseline[row["seed"]]["time_s"] - row["time_s"]
            for row in rows
            if row["policy"] == policy
        ]
        paired[policy] = {
            "mean_baseline_minus_policy_s": statistics.mean(differences),
            "median_baseline_minus_policy_s": statistics.median(differences),
            "faster_cases": sum(value > 1e-9 for value in differences),
            "slower_cases": sum(value < -1e-9 for value in differences),
        }
    return {"by_policy": result, "paired_vs_no_negative": paired}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=BASE / "runs" / "pilot.json")
    args = parser.parse_args()

    model = ModelConfig()
    rows = []
    for seed in range(args.start_seed, args.start_seed + args.seeds):
        scenario = generate_scenario(seed, model)
        for policy in CORE_POLICIES:
            rows.append(run_policy(scenario, policy, model=model))

    payload = {
        "label": "pilot_only_not_for_publication",
        "model": model.__dict__,
        "seed_range": [args.start_seed, args.start_seed + args.seeds - 1],
        "summary": summarize(rows),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
