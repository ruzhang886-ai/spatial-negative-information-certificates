from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.gain_aware_verifier import certify_gain_aware_multiplicity
from src.policies import choose_pair_candidate, choose_spatial_candidate
from src.simulator import CostConfig, Mission, ModelConfig, generate_scenario


DEFAULT_START_SEED = 15100
DEFAULT_SEEDS = 10
BEAMWIDTHS_DEG = (60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0)
MAX_ATTENUATIONS_DB = (20.0, 30.0)
PATH_LOSS_EXPONENTS = (2.0, 2.5, 3.0)


def model_config(beamwidth, maximum_attenuation, path_loss_exponent):
    return ModelConfig(
        miss_probability=0.0,
        antenna_mode="parabolic_halfplane",
        antenna_half_power_beamwidth_deg=beamwidth,
        antenna_max_attenuation_db=maximum_attenuation,
        path_loss_exponent=path_loss_exponent,
        reliable_pair_radius_m=999.0,
    )


def run_certificate(candidate, model, *, spatial):
    if candidate is None:
        return {
            "status": "no_candidate",
            "elapsed_s": 0.0,
            "cells_visited": 0,
            "maximum_queue_size": 0,
        }
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
    started = time.perf_counter()
    result = certify_gain_aware_multiplicity(
        polygon,
        history,
        points,
        required,
        model,
        geometric_visibility_multiplicity=required,
        position_tolerance_m=2.0,
        orientation_tolerance_deg=0.5,
        max_cells=300000,
        numeric_safety_margin_m=1e-6,
        counterexample_position_subdivisions=10,
        counterexample_orientation_step_deg=2.0,
    )
    elapsed = time.perf_counter() - started
    return {
        "status": result.status,
        "elapsed_s": elapsed,
        "cells_visited": result.cells_visited,
        "cells_certified": result.cells_certified,
        "cells_discarded_as_infeasible": result.cells_discarded_as_infeasible,
        "maximum_queue_size": result.maximum_queue_size,
    }


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def summarize(rows, key):
    results = [row[key] for row in rows]
    statuses = {}
    for result in results:
        statuses[result["status"]] = statuses.get(result["status"], 0) + 1
    elapsed = [result["elapsed_s"] for result in results]
    cells = [result["cells_visited"] for result in results]
    return {
        "runs": len(results),
        "statuses": statuses,
        "certification_rate": statuses.get("certified", 0) / len(results),
        "counterexample_rate": statuses.get("counterexample", 0) / len(results),
        "unresolved_rate": statuses.get("unresolved", 0) / len(results),
        "median_elapsed_s": statistics.median(elapsed),
        "p95_elapsed_s": percentile(elapsed, 95),
        "max_elapsed_s": max(elapsed),
        "mean_cells_visited": statistics.mean(cells),
        "max_cells_visited": max(cells),
    }


def key_for(beamwidth, maximum_attenuation, path_loss_exponent):
    return (
        f"bw={beamwidth:.1f}|amax={maximum_attenuation:.1f}|"
        f"n={path_loss_exponent:.1f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "runs" / "gain_parameter_sensitivity_15100_15109.json",
    )
    args = parser.parse_args()

    rows = []
    overall_started = time.perf_counter()
    for maximum_attenuation in MAX_ATTENUATIONS_DB:
        for path_loss_exponent in PATH_LOSS_EXPONENTS:
            for beamwidth in BEAMWIDTHS_DEG:
                model = model_config(
                    beamwidth, maximum_attenuation, path_loss_exponent
                )
                for seed in range(args.start_seed, args.start_seed + args.seeds):
                    scenario = generate_scenario(seed, model)

                    started = time.perf_counter()
                    pair_candidate = choose_pair_candidate(
                        Mission(scenario, "full_pair", model, CostConfig())
                    )
                    pair_selection_s = time.perf_counter() - started
                    pair_result = run_certificate(
                        pair_candidate, model, spatial=False
                    )

                    started = time.perf_counter()
                    spatial_candidate = choose_spatial_candidate(
                        Mission(
                            scenario,
                            "spatial_multicover_q2",
                            model,
                            CostConfig(),
                        ),
                        q=2,
                    )
                    spatial_selection_s = time.perf_counter() - started
                    spatial_result = run_certificate(
                        spatial_candidate, model, spatial=True
                    )

                    rows.append(
                        {
                            "seed": seed,
                            "beamwidth_deg": beamwidth,
                            "maximum_attenuation_db": maximum_attenuation,
                            "path_loss_exponent": path_loss_exponent,
                            "pair_selection_s": pair_selection_s,
                            "spatial_q2_selection_s": spatial_selection_s,
                            "pair": pair_result,
                            "spatial_q2": spatial_result,
                        }
                    )
    overall_elapsed = time.perf_counter() - overall_started

    summary = {}
    for maximum_attenuation in MAX_ATTENUATIONS_DB:
        for path_loss_exponent in PATH_LOSS_EXPONENTS:
            for beamwidth in BEAMWIDTHS_DEG:
                subset = [
                    row
                    for row in rows
                    if row["beamwidth_deg"] == beamwidth
                    and row["maximum_attenuation_db"] == maximum_attenuation
                    and row["path_loss_exponent"] == path_loss_exponent
                ]
                summary[
                    key_for(
                        beamwidth, maximum_attenuation, path_loss_exponent
                    )
                ] = {
                    "pair": summarize(subset, "pair"),
                    "spatial_q2": summarize(subset, "spatial_q2"),
                    "median_pair_selection_s": statistics.median(
                        row["pair_selection_s"] for row in subset
                    ),
                    "median_spatial_q2_selection_s": statistics.median(
                        row["spatial_q2_selection_s"] for row in subset
                    ),
                }

    payload = {
        "label": "gain_aware_parameter_and_runtime_sensitivity",
        "seed_range": [args.start_seed, args.start_seed + args.seeds - 1],
        "beamwidths_deg": BEAMWIDTHS_DEG,
        "maximum_attenuations_db": MAX_ATTENUATIONS_DB,
        "path_loss_exponents": PATH_LOSS_EXPONENTS,
        "certificate_position_tolerance_m": 2.0,
        "certificate_orientation_tolerance_deg": 0.5,
        "certificate_numeric_safety_margin_m": 1e-6,
        "overall_wall_clock_s": overall_elapsed,
        "runtime_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
        },
        "summary": summary,
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    compact = {
        "overall_wall_clock_s": overall_elapsed,
        "summary": summary,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
