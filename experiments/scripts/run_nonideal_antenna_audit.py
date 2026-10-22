from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.policies import run_policy
from src.simulator import (
    ModelConfig,
    antenna_range_factor,
    generate_scenario,
)


START_SEED = 14000
SEEDS = 500
BEAMWIDTHS_DEG = (65.0, 90.0, 120.0)
PATH_LOSS_EXPONENT = 2.5
MAX_ATTENUATION_DB = 30.0
VARIANTS = (
    ("no_negative_update", "no_negative_information", "naive"),
    ("full_pair_naive", "full_pair", "naive"),
    ("spatial_q2_naive", "spatial_multicover_q2", "naive"),
    ("full_pair_conservative", "full_pair", "conservative"),
    ("spatial_q2_conservative", "spatial_multicover_q2", "conservative"),
)
OUTPUT = BASE / "runs" / "nonideal_antenna_14000_14499.json"


def average(values):
    return statistics.mean(values) if values else None


def summarize(rows: list[dict]) -> dict:
    completed = [row for row in rows if row["completed"]]
    return {
        "runs": len(rows),
        "completed": len(completed),
        "completion_rate": len(completed) / len(rows),
        "unsafe_exclusions": sum(row["unsafe_exclusion"] for row in rows),
        "unsafe_rate": average([row["unsafe_exclusion"] for row in rows]),
        "mean_time_completed_s": average([row["time_s"] for row in completed]),
        "mean_observed_time_s": average([row["time_s"] for row in rows]),
        "mean_measures": average([row["measures"] for row in rows]),
        "mean_pair_cuts": average([row["pair_cuts"] for row in rows]),
        "fallback_rate": average([row["fallback_used"] for row in rows]),
    }


def physical_config(beamwidth: float, reliable_radius: float) -> ModelConfig:
    return ModelConfig(
        miss_probability=0.0,
        antenna_mode="parabolic_halfplane",
        antenna_half_power_beamwidth_deg=beamwidth,
        antenna_max_attenuation_db=MAX_ATTENUATION_DB,
        path_loss_exponent=PATH_LOSS_EXPONENT,
        reliable_pair_radius_m=reliable_radius,
    )


def edge_radius(beamwidth: float) -> float:
    probe_config = physical_config(beamwidth, reliable_radius=999.0)
    factor = antenna_range_factor(
        (0.0, 1.0),
        (0.0, 0.0),
        (1.0, 0.0),
        probe_config,
    )
    return probe_config.min_reception_m * factor


def main() -> None:
    rows = []
    metadata = {}
    for beamwidth in BEAMWIDTHS_DEG:
        conservative_radius = max(1.0, edge_radius(beamwidth) - 1.0)
        metadata[str(beamwidth)] = {
            "edge_guaranteed_radius_m": edge_radius(beamwidth),
            "conservative_certificate_radius_m": conservative_radius,
        }
        scenario_config = physical_config(beamwidth, reliable_radius=999.0)
        scenarios = {
            seed: generate_scenario(seed, scenario_config)
            for seed in range(START_SEED, START_SEED + SEEDS)
        }
        for variant, policy, radius_mode in VARIANTS:
            reliable_radius = (
                999.0 if radius_mode == "naive" else conservative_radius
            )
            model = physical_config(beamwidth, reliable_radius)
            for seed, scenario in scenarios.items():
                row = run_policy(scenario, policy, model=model)
                row.pop("events", None)
                row["variant"] = variant
                row["beamwidth_deg"] = beamwidth
                row["certificate_radius_m"] = reliable_radius
                rows.append(row)

    summary = {}
    for beamwidth in BEAMWIDTHS_DEG:
        summary[str(beamwidth)] = {}
        for variant, _, _ in VARIANTS:
            subset = [
                row
                for row in rows
                if row["beamwidth_deg"] == beamwidth
                and row["variant"] == variant
            ]
            summary[str(beamwidth)][variant] = summarize(subset)

    payload = {
        "label": "nonideal_antenna_theorem_audit",
        "seed_range": [START_SEED, START_SEED + SEEDS - 1],
        "beamwidths_deg": BEAMWIDTHS_DEG,
        "path_loss_exponent": PATH_LOSS_EXPONENT,
        "max_attenuation_db": MAX_ATTENUATION_DB,
        "antenna_pattern": "min(12*(off_axis/hpbw)^2, max_attenuation_db)",
        "variants": [variant for variant, _, _ in VARIANTS],
        "beamwidth_metadata": metadata,
        "summary": summary,
        "runs": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"metadata": metadata, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
