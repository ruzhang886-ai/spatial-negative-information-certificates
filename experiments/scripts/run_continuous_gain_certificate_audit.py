from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.gain_aware_verifier import certify_gain_aware_multiplicity
from src.policies import choose_pair_candidate, choose_spatial_candidate
from src.simulator import CostConfig, Mission, ModelConfig, generate_scenario


DEFAULT_START_SEED = 14700
DEFAULT_SEEDS = 20
BEAMWIDTHS_DEG = (65.0, 90.0, 120.0)


def model_config(beamwidth: float) -> ModelConfig:
    return ModelConfig(
        miss_probability=0.0,
        antenna_mode="parabolic_halfplane",
        antenna_half_power_beamwidth_deg=beamwidth,
        antenna_max_attenuation_db=30.0,
        path_loss_exponent=2.5,
        reliable_pair_radius_m=999.0,
    )


def serialize(result):
    return {
        "status": result.status,
        "required_multiplicity": result.required_multiplicity,
        "cells_visited": result.cells_visited,
        "cells_certified": result.cells_certified,
        "cells_discarded_as_infeasible": result.cells_discarded_as_infeasible,
        "maximum_queue_size": result.maximum_queue_size,
        "unresolved_position_radius_m": result.unresolved_position_radius_m,
        "unresolved_orientation_halfwidth_deg": (
            result.unresolved_orientation_halfwidth_deg
        ),
        "unresolved_source_center": result.unresolved_source_center,
        "unresolved_boresight_center_deg": (
            result.unresolved_boresight_center_deg
        ),
        "counterexample_source": result.counterexample_source,
        "counterexample_boresight_deg": result.counterexample_boresight_deg,
        "counterexample_detection_count": result.counterexample_detection_count,
    }


def audit(candidate, model, *, spatial, args):
    if candidate is None:
        return None
    if spatial:
        polygon = candidate.certificate.base.far_polygon
        history = candidate.certificate.base.origin
        points = candidate.ordered_points
        required = 3
    else:
        polygon = candidate.certificate.far_polygon
        history = candidate.certificate.origin
        points = candidate.ordered_points
        required = 1
    return serialize(
        certify_gain_aware_multiplicity(
            polygon,
            history,
            points,
            required,
            model,
            geometric_visibility_multiplicity=required,
            position_tolerance_m=args.position_tolerance_m,
            orientation_tolerance_deg=args.orientation_tolerance_deg,
            max_cells=args.max_cells,
            counterexample_position_subdivisions=10,
            counterexample_orientation_step_deg=2.0,
        )
    )


def summarize(rows, key):
    available = [row[key] for row in rows if row[key] is not None]
    statuses = {name: 0 for name in ("certified", "counterexample", "unresolved")}
    for result in available:
        statuses[result["status"]] += 1
    return {
        "scenarios": len(rows),
        "candidate_available": len(available),
        "statuses": statuses,
        "mean_cells_visited": (
            statistics.mean(result["cells_visited"] for result in available)
            if available
            else None
        ),
        "max_cells_visited": (
            max(result["cells_visited"] for result in available)
            if available
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--position-tolerance-m", type=float, default=2.0)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=0.5)
    parser.add_argument("--max-cells", type=int, default=300000)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "runs" / "continuous_gain_certificate_14700_14719.json",
    )
    args = parser.parse_args()

    rows = []
    for beamwidth in BEAMWIDTHS_DEG:
        model = model_config(beamwidth)
        for seed in range(args.start_seed, args.start_seed + args.seeds):
            scenario = generate_scenario(seed, model)
            pair_mission = Mission(scenario, "full_pair", model, CostConfig())
            pair_candidate = choose_pair_candidate(pair_mission)
            spatial_mission = Mission(
                scenario, "spatial_multicover_q2", model, CostConfig()
            )
            spatial_candidate = choose_spatial_candidate(spatial_mission, q=2)
            rows.append(
                {
                    "seed": seed,
                    "beamwidth_deg": beamwidth,
                    "pair": audit(
                        pair_candidate, model, spatial=False, args=args
                    ),
                    "spatial_q2": audit(
                        spatial_candidate, model, spatial=True, args=args
                    ),
                }
            )

    summary = {}
    for beamwidth in BEAMWIDTHS_DEG:
        subset = [row for row in rows if row["beamwidth_deg"] == beamwidth]
        summary[str(beamwidth)] = {
            "pair": summarize(subset, "pair"),
            "spatial_q2": summarize(subset, "spatial_q2"),
        }
    payload = {
        "label": "continuous_gain_aware_interval_certificate_audit",
        "seed_range": [args.start_seed, args.start_seed + args.seeds - 1],
        "beamwidths_deg": BEAMWIDTHS_DEG,
        "position_tolerance_m": args.position_tolerance_m,
        "orientation_tolerance_deg": args.orientation_tolerance_deg,
        "max_cells": args.max_cells,
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
