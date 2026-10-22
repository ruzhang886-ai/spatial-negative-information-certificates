from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.gain_aware_verifier import sampled_gain_aware_multiplicity
from src.policies import choose_pair_candidate, choose_spatial_candidate
from src.simulator import CostConfig, Mission, ModelConfig, generate_scenario


DEFAULT_START_SEED = 14500
DEFAULT_SEEDS = 200
BEAMWIDTHS_DEG = (65.0, 90.0, 120.0)
PATH_LOSS_EXPONENT = 2.5
MAX_ATTENUATION_DB = 30.0


def model_config(beamwidth: float) -> ModelConfig:
    return ModelConfig(
        miss_probability=0.0,
        antenna_mode="parabolic_halfplane",
        antenna_half_power_beamwidth_deg=beamwidth,
        antenna_max_attenuation_db=MAX_ATTENUATION_DB,
        path_loss_exponent=PATH_LOSS_EXPONENT,
        reliable_pair_radius_m=999.0,
    )


def audit_candidate(candidate, model, *, spatial_q2: bool, subdivisions, angle_step):
    if candidate is None:
        return None
    if spatial_q2:
        polygon = candidate.certificate.base.far_polygon
        history = candidate.certificate.base.origin
        points = candidate.ordered_points
        required = 3
    else:
        polygon = candidate.certificate.far_polygon
        history = candidate.certificate.origin
        points = candidate.ordered_points
        required = 1
    audit = sampled_gain_aware_multiplicity(
        polygon,
        history,
        points,
        required,
        model,
        position_subdivisions=subdivisions,
        orientation_step_deg=angle_step,
    )
    return {
        "valid_on_sampled_states": audit.valid_on_sampled_states,
        "minimum_guaranteed_multiplicity": audit.minimum_guaranteed_multiplicity,
        "required_multiplicity": audit.required_multiplicity,
        "feasible_states_checked": audit.feasible_states_checked,
        "position_samples_checked": audit.position_samples_checked,
        "witness_source": audit.witness_source,
        "witness_boresight_deg": audit.witness_boresight_deg,
        "witness_lower_radius_m": audit.witness_lower_radius_m,
        "witness_detection_count": audit.witness_detection_count,
        "minimum_required_order_margin_m": (
            audit.minimum_required_order_margin_m
        ),
        "margin_witness_source": audit.margin_witness_source,
        "margin_witness_boresight_deg": audit.margin_witness_boresight_deg,
    }


def summarize(rows, field):
    available = [row[field] for row in rows if row[field] is not None]
    valid = [row for row in available if row["valid_on_sampled_states"]]
    counts = {}
    for row in available:
        key = str(row["minimum_guaranteed_multiplicity"])
        counts[key] = counts.get(key, 0) + 1
    return {
        "scenarios": len(rows),
        "candidate_available": len(available),
        "candidate_availability_rate": len(available) / len(rows),
        "sampled_valid": len(valid),
        "sampled_valid_rate_all_scenarios": len(valid) / len(rows),
        "sampled_valid_rate_available": (
            len(valid) / len(available) if available else None
        ),
        "minimum_multiplicity_distribution": counts,
        "mean_feasible_states_checked": (
            statistics.mean(row["feasible_states_checked"] for row in available)
            if available
            else None
        ),
        "minimum_sampled_margin_m": (
            min(row["minimum_required_order_margin_m"] for row in available)
            if available
            else None
        ),
        "median_sampled_margin_m": (
            statistics.median(
                row["minimum_required_order_margin_m"] for row in available
            )
            if available
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--position-subdivisions", type=int, default=8)
    parser.add_argument("--orientation-step-deg", type=float, default=3.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "runs" / "gain_aware_state_audit_14500_14699.json",
    )
    args = parser.parse_args()

    rows = []
    for beamwidth in BEAMWIDTHS_DEG:
        model = model_config(beamwidth)
        for seed in range(args.start_seed, args.start_seed + args.seeds):
            scenario = generate_scenario(seed, model)

            pair_mission = Mission(scenario, "full_pair", model, CostConfig())
            pair_candidate = choose_pair_candidate(pair_mission)
            pair_audit = audit_candidate(
                pair_candidate,
                model,
                spatial_q2=False,
                subdivisions=args.position_subdivisions,
                angle_step=args.orientation_step_deg,
            )

            spatial_mission = Mission(
                scenario, "spatial_multicover_q2", model, CostConfig()
            )
            spatial_candidate = choose_spatial_candidate(spatial_mission, q=2)
            spatial_audit = audit_candidate(
                spatial_candidate,
                model,
                spatial_q2=True,
                subdivisions=args.position_subdivisions,
                angle_step=args.orientation_step_deg,
            )
            rows.append(
                {
                    "seed": seed,
                    "beamwidth_deg": beamwidth,
                    "pair": pair_audit,
                    "spatial_q2": spatial_audit,
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
        "label": "sampled_gain_aware_state_audit_of_legacy_selected_candidates",
        "scope_warning": (
            "Finite position-orientation audit only; zero sampled violations "
            "is not a continuous-state proof."
        ),
        "seed_range": [args.start_seed, args.start_seed + args.seeds - 1],
        "beamwidths_deg": BEAMWIDTHS_DEG,
        "position_subdivisions": args.position_subdivisions,
        "orientation_step_deg": args.orientation_step_deg,
        "path_loss_exponent": PATH_LOSS_EXPONENT,
        "max_attenuation_db": MAX_ATTENUATION_DB,
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
