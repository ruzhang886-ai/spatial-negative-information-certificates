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


DEFAULT_START_SEED = 14900
DEFAULT_SEEDS = 100
BEAMWIDTHS_DEG = (65.0, 90.0, 120.0)
VARIANTS = (
    ("no_negative_update", "no_negative_information"),
    ("full_pair_naive", "full_pair"),
    ("full_pair_gain_aware", "gain_aware_full_pair"),
    ("spatial_q2_naive", "spatial_multicover_q2"),
    ("spatial_q2_gain_aware", "gain_aware_spatial_q2"),
)


def model_config(beamwidth: float) -> ModelConfig:
    return ModelConfig(
        miss_probability=0.0,
        antenna_mode="parabolic_halfplane",
        antenna_half_power_beamwidth_deg=beamwidth,
        antenna_max_attenuation_db=30.0,
        path_loss_exponent=2.5,
        reliable_pair_radius_m=999.0,
    )


def summarize(rows):
    completed = [row for row in rows if row["completed"]]
    status_counts = {}
    for row in rows:
        for event in row.get("events", []):
            if event.get("action") == "gain_certificate_checked":
                status = event["status"]
                status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "runs": len(rows),
        "completed": len(completed),
        "completion_rate": len(completed) / len(rows),
        "unsafe_exclusions": sum(row["unsafe_exclusion"] for row in rows),
        "unsafe_rate": statistics.mean(row["unsafe_exclusion"] for row in rows),
        "mean_observed_time_s": statistics.mean(row["time_s"] for row in rows),
        "mean_completed_time_s": (
            statistics.mean(row["time_s"] for row in completed)
            if completed
            else None
        ),
        "mean_measures": statistics.mean(row["measures"] for row in rows),
        "mean_pair_cuts": statistics.mean(row["pair_cuts"] for row in rows),
        "fallback_rate": statistics.mean(row["fallback_used"] for row in rows),
        "gain_certificate_status_counts": status_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "runs" / "gain_aware_policy_14900_14999.json",
    )
    args = parser.parse_args()

    rows = []
    for beamwidth in BEAMWIDTHS_DEG:
        model = model_config(beamwidth)
        scenarios = {
            seed: generate_scenario(seed, model)
            for seed in range(args.start_seed, args.start_seed + args.seeds)
        }
        for variant, policy in VARIANTS:
            for seed, scenario in scenarios.items():
                row = run_policy(scenario, policy, model=model)
                row["variant"] = variant
                row["beamwidth_deg"] = beamwidth
                rows.append(row)

    summary = {}
    for beamwidth in BEAMWIDTHS_DEG:
        summary[str(beamwidth)] = {}
        for variant, _ in VARIANTS:
            subset = [
                row
                for row in rows
                if row["beamwidth_deg"] == beamwidth
                and row["variant"] == variant
            ]
            summary[str(beamwidth)][variant] = summarize(subset)

    payload = {
        "label": "gain_aware_continuous_certificate_policy_audit",
        "seed_range": [args.start_seed, args.start_seed + args.seeds - 1],
        "beamwidths_deg": BEAMWIDTHS_DEG,
        "variants": [name for name, _ in VARIANTS],
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
