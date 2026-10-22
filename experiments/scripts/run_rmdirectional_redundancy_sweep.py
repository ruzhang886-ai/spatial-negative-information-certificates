from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.geometry import bearing_wedge_triangle
from src.paired_certificate import build_spatial_redundancy_certificate
from src.rmdirectional_adapter import (
    load_sample,
    nearest_building_height_m,
    nearest_path_gain_db,
    open_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="探索空间冗余阶数与点间距的外部传播取舍。")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("primary_result", type=Path)
    parser.add_argument("--q-values", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--spacing-factors", type=float, nargs="+", default=(0.04, 0.08, 0.12))
    parser.add_argument("--output", type=Path, default=BASE / "runs" / "rmdirectional_redundancy_sweep.json")
    return parser.parse_args()


def nearest_neighbor_order(points: np.ndarray, start: np.ndarray) -> list[int]:
    remaining = set(range(len(points)))
    order = []
    current = np.asarray(start, dtype=float)
    while remaining:
        index = min(remaining, key=lambda candidate: float(np.linalg.norm(points[candidate] - current)))
        order.append(index)
        current = points[index]
        remaining.remove(index)
    return order


def evaluate_candidate(sample, history, threshold: float, q: int, spacing_factor: float) -> dict:
    source = sample.tx_coords[:2]
    history = np.asarray(history, dtype=float)
    distance = float(np.linalg.norm(source - history))
    bearing = math.degrees(math.atan2(source[1] - history[1], source[0] - history[0]))
    feasible = bearing_wedge_triangle(history, bearing, 1.0, max_range=distance * 1.001)
    longitudinal = 0.55 * distance
    transverse = max(longitudinal * math.tan(math.radians(1.0)) + 0.5, 0.15 * distance)
    spacing = max(2.0, spacing_factor * distance)
    certificate = build_spatial_redundancy_certificate(
        feasible,
        history,
        bearing,
        1.0,
        longitudinal,
        transverse,
        spacing,
        reliable_radius=distance,
        tolerated_false_negatives=q,
    )
    result = {"valid": certificate.valid, "evaluable": False}
    if not certificate.valid:
        return result
    points = np.asarray(certificate.points, dtype=float)
    outcomes = []
    for point in points:
        building = nearest_building_height_m(sample, point)
        gain = nearest_path_gain_db(sample, point)
        if building is None or gain is None or building > 0:
            return result
        outcomes.append(gain >= threshold)
    result["evaluable"] = True
    result["all_negative"] = not any(outcomes)
    order = nearest_neighbor_order(points, history)
    current = history.copy()
    measured = 0
    travel = 0.0
    for index in order:
        travel += float(np.linalg.norm(points[index] - current))
        current = points[index]
        measured += 1
        if outcomes[index]:
            break
    result["measurements_until_stop"] = measured
    result["travel_until_stop_m"] = travel
    return result


def candidate_grid(q_values: list[int], spacing_factors: list[float]) -> list[tuple[int, float]]:
    result = []
    for q in q_values:
        if q == 0:
            result.append((q, spacing_factors[0]))
        else:
            result.extend((q, factor) for factor in spacing_factors)
    return result


def key(q: int, factor: float) -> str:
    return f"q{q}_spacing_{factor:.2f}r"


def main() -> None:
    args = parse_args()
    primary = json.loads(args.primary_result.read_text(encoding="utf-8"))
    candidates = candidate_grid(list(args.q_values), list(args.spacing_factors))
    counters: dict[tuple[float, int, float], dict[str, float]] = {}
    with open_dataset(args.dataset) as dataset:
        cached_id = None
        sample = None
        for row in primary["rows"]:
            threshold = float(row["threshold_db"])
            sample_id = tuple(row["sample_id"])
            if sample_id != cached_id:
                sample = load_sample(dataset, sample_id)
                cached_id = sample_id
            for history in row["histories"]:
                for q, factor in candidates:
                    counter = counters.setdefault(
                        (threshold, q, factor),
                        {
                            "opportunities": 0,
                            "valid": 0,
                            "evaluable": 0,
                            "all_negative": 0,
                            "measurements": 0.0,
                            "travel_m": 0.0,
                        },
                    )
                    counter["opportunities"] += 1
                    outcome = evaluate_candidate(
                        sample,
                        history["history"],
                        threshold,
                        q,
                        factor,
                    )
                    counter["valid"] += int(outcome["valid"])
                    if outcome["evaluable"]:
                        counter["evaluable"] += 1
                        counter["all_negative"] += int(outcome["all_negative"])
                        counter["measurements"] += outcome["measurements_until_stop"]
                        counter["travel_m"] += outcome["travel_until_stop_m"]

    summaries = []
    for (threshold, q, factor), counter in sorted(counters.items()):
        opportunities = int(counter["opportunities"])
        evaluable = int(counter["evaluable"])
        all_negative = int(counter["all_negative"])
        summaries.append(
            {
                "threshold_db": threshold,
                "q": q,
                "points": 2 * (q + 1),
                "spacing_factor": factor,
                "configuration": key(q, factor),
                "opportunities": opportunities,
                "geometrically_valid": int(counter["valid"]),
                "evaluable": evaluable,
                "evaluable_rate": evaluable / opportunities if opportunities else None,
                "all_negative": all_negative,
                "all_negative_rate_given_evaluable": all_negative / evaluable if evaluable else None,
                "all_negative_per_opportunity": all_negative / opportunities if opportunities else None,
                "mean_measurements_until_stop": counter["measurements"] / evaluable if evaluable else None,
                "mean_travel_until_stop_m": counter["travel_m"] / evaluable if evaluable else None,
            }
        )
    payload = {
        "label": "Exploratory RMDirectionalBerlin redundancy and spacing sweep",
        "status": "development-only; the 200 primary-audit tiles are not a holdout for this sweep",
        "dataset": str(args.dataset.resolve()),
        "primary_result": str(args.primary_result.resolve()),
        "q_values": args.q_values,
        "spacing_factors": args.spacing_factors,
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
