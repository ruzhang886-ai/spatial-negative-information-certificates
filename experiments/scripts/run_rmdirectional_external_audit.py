from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.geometry import bearing_wedge_triangle
from src.paired_certificate import (
    build_pair_certificate,
    build_spatial_redundancy_certificate,
)
from src.rmdirectional_adapter import (
    discover_sample_ids,
    load_sample,
    nearest_building_height_m,
    nearest_path_gain_db,
    open_dataset,
)


DEFAULT_THRESHOLDS_DB = (-115.0, -105.0, -95.0)
SELECTION_SALT = "rmdirectional-berlin-external-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用RMDirectionalBerlin射线追踪图审查历史正观测的径向外推与横向证书。"
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="RMBerlinDirectional官方ZIP文件或选择性解压后的目录",
    )
    parser.add_argument("--sample-limit", type=int, default=200)
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="在确定性地理分层顺序中跳过的样本数。",
    )
    parser.add_argument("--histories-per-sample", type=int, default=8)
    parser.add_argument("--thresholds-db", type=float, nargs="+", default=DEFAULT_THRESHOLDS_DB)
    parser.add_argument("--bearing-error-deg", type=float, default=1.0)
    parser.add_argument("--minimum-history-distance", type=float, default=30.0)
    parser.add_argument("--maximum-history-distance", type=float, default=120.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "runs" / "rmdirectional_external_audit.json",
    )
    return parser.parse_args()


def deterministic_subset(points: np.ndarray, limit: int, salt: str) -> np.ndarray:
    if len(points) <= limit:
        return points
    scores = []
    for point in points:
        token = f"{salt}|{int(point[0])}|{int(point[1])}".encode()
        scores.append(hashlib.sha256(token).digest())
    order = sorted(range(len(points)), key=lambda index: scores[index])
    return points[order[:limit]]


def geographically_stratified_sample_ids(
    sample_ids: list[tuple[str, str, str, str]],
    limit: int,
) -> list[tuple[str, str, str, str]]:
    """优先从不同城市切片各取一个样本，再按轮次补充。"""
    groups: dict[tuple[str, str, str], list[tuple[str, str, str, str]]] = {}
    for sample_id in sample_ids:
        groups.setdefault(sample_id[:3], []).append(sample_id)

    def digest(tokens) -> bytes:
        text = "|".join((SELECTION_SALT, *tokens))
        return hashlib.sha256(text.encode("utf-8")).digest()

    group_order = sorted(groups, key=digest)
    candidates = {
        group: sorted(groups[group], key=digest)
        for group in group_order
    }
    selected: list[tuple[str, str, str, str]] = []
    round_index = 0
    while len(selected) < limit:
        added = False
        for group in group_order:
            if round_index < len(candidates[group]):
                selected.append(candidates[group][round_index])
                added = True
                if len(selected) == limit:
                    return selected
        if not added:
            break
        round_index += 1
    return selected


def point_is_positive(sample, point, threshold_db: float) -> bool | None:
    building_height = nearest_building_height_m(sample, point)
    if building_height is None or building_height > 0:
        return None
    value = nearest_path_gain_db(sample, point)
    return None if value is None else value >= threshold_db


def audit_history(sample, history, threshold_db: float, bearing_error_deg: float) -> dict:
    source = sample.tx_coords[:2]
    history = np.asarray(history, dtype=float)
    distance = float(np.linalg.norm(source - history))
    bearing = math.degrees(math.atan2(source[1] - history[1], source[0] - history[0]))
    feasible = bearing_wedge_triangle(
        history,
        bearing,
        bearing_error_deg,
        max_range=distance * 1.001,
    )
    longitudinal = 0.55 * distance
    transverse = max(
        longitudinal * math.tan(math.radians(bearing_error_deg)) + 0.5,
        0.15 * distance,
    )
    pair = build_pair_certificate(
        feasible,
        history,
        bearing,
        bearing_error_deg,
        longitudinal,
        transverse,
        reliable_radius=distance,
    )
    result = {
        "history": [float(value) for value in history],
        "history_distance_m": distance,
        "pair_valid": pair.valid,
        "pair_points_evaluable": False,
        "pair_all_negative": None,
        "spatial_q2_valid": False,
        "spatial_q2_points_evaluable": False,
        "spatial_q2_all_negative": None,
    }
    if pair.valid:
        outcomes = [
            point_is_positive(sample, point, threshold_db)
            for point in (pair.q_minus, pair.q_plus)
        ]
        result["pair_points_evaluable"] = all(value is not None for value in outcomes)
        if result["pair_points_evaluable"]:
            result["pair_all_negative"] = not any(outcomes)

    spacing = max(2.0, 0.04 * distance)
    spatial = build_spatial_redundancy_certificate(
        feasible,
        history,
        bearing,
        bearing_error_deg,
        longitudinal,
        transverse,
        spacing,
        reliable_radius=distance,
        tolerated_false_negatives=2,
    )
    result["spatial_q2_valid"] = spatial.valid
    if spatial.valid:
        outcomes = [point_is_positive(sample, point, threshold_db) for point in spatial.points]
        result["spatial_q2_points_evaluable"] = all(value is not None for value in outcomes)
        if result["spatial_q2_points_evaluable"]:
            result["spatial_q2_all_negative"] = not any(outcomes)
    return result


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for threshold in sorted({row["threshold_db"] for row in rows}):
        selected = [row for row in rows if row["threshold_db"] == threshold]
        histories = [history for row in selected for history in row["histories"]]
        pair = [h for h in histories if h["pair_valid"] and h["pair_points_evaluable"]]
        q2 = [h for h in histories if h["spatial_q2_valid"] and h["spatial_q2_points_evaluable"]]
        common = [
            h
            for h in histories
            if h["pair_valid"]
            and h["pair_points_evaluable"]
            and h["spatial_q2_valid"]
            and h["spatial_q2_points_evaluable"]
        ]
        summary[str(threshold)] = {
            "samples": len(selected),
            "positive_histories": len(histories),
            "geometrically_valid_pair_certificates": sum(h["pair_valid"] for h in histories),
            "evaluable_pair_certificates": len(pair),
            "pair_all_negative": sum(bool(h["pair_all_negative"]) for h in pair),
            "pair_all_negative_rate": (
                sum(bool(h["pair_all_negative"]) for h in pair) / len(pair) if pair else None
            ),
            "geometrically_valid_spatial_q2_certificates": sum(
                h["spatial_q2_valid"] for h in histories
            ),
            "evaluable_spatial_q2_certificates": len(q2),
            "spatial_q2_all_negative": sum(bool(h["spatial_q2_all_negative"]) for h in q2),
            "spatial_q2_all_negative_rate": (
                sum(bool(h["spatial_q2_all_negative"]) for h in q2) / len(q2) if q2 else None
            ),
            "common_evaluable_certificates": len(common),
            "common_pair_all_negative": sum(
                bool(h["pair_all_negative"]) for h in common
            ),
            "common_pair_all_negative_rate": (
                sum(bool(h["pair_all_negative"]) for h in common) / len(common)
                if common
                else None
            ),
            "common_spatial_q2_all_negative": sum(
                bool(h["spatial_q2_all_negative"]) for h in common
            ),
            "common_spatial_q2_all_negative_rate": (
                sum(bool(h["spatial_q2_all_negative"]) for h in common) / len(common)
                if common
                else None
            ),
        }
    return summary


def main() -> None:
    args = parse_args()
    if args.sample_limit <= 0 or args.sample_offset < 0 or args.histories_per_sample <= 0:
        raise ValueError("样本数和每样本历史点数必须为正。")
    rows = []
    with open_dataset(args.dataset_root) as dataset:
        all_ids = discover_sample_ids(dataset)
        ordered = geographically_stratified_sample_ids(
            all_ids, args.sample_offset + args.sample_limit
        )
        ids = ordered[args.sample_offset : args.sample_offset + args.sample_limit]
        for sample_id in ids:
            sample = load_sample(dataset, sample_id)
            grid = np.indices(sample.path_gain_db.shape).reshape(2, -1).T
            distances = np.linalg.norm(grid - sample.tx_coords[:2], axis=1)
            distance_mask = (
                (distances >= args.minimum_history_distance)
                & (distances <= args.maximum_history_distance)
            )
            accessible = sample.building_height_m.reshape(-1) <= 0
            for threshold in args.thresholds_db:
                positive = sample.path_gain_db.reshape(-1) >= threshold
                histories = grid[distance_mask & positive & accessible]
                histories = deterministic_subset(
                    histories,
                    args.histories_per_sample,
                    salt=f"{'_'.join(sample_id)}|{threshold}",
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "threshold_db": float(threshold),
                        "antenna_id": sample.antenna_id,
                        "tx_coords": [float(value) for value in sample.tx_coords],
                        "histories": [
                            audit_history(
                                sample,
                                history,
                                float(threshold),
                                args.bearing_error_deg,
                            )
                            for history in histories
                        ],
                    }
                )
    payload = {
        "label": "RMDirectionalBerlin external ray-tracing stress audit",
        "dataset_source": str(args.dataset_root.resolve()),
        "sample_limit": args.sample_limit,
        "sample_offset": args.sample_offset,
        "available_samples": len(all_ids),
        "selection_method": "deterministic geographic-prefix stratification",
        "selection_salt": SELECTION_SALT,
        "selected_sample_ids": ids,
        "selected_geographic_tiles": len({sample_id[:3] for sample_id in ids}),
        "histories_per_sample": args.histories_per_sample,
        "thresholds_db": [float(value) for value in args.thresholds_db],
        "bearing_error_deg": args.bearing_error_deg,
        "interpretation": (
            "An all-negative event is an empirical counterexample to transferring the "
            "history-implied radial certificate to this ray-traced map. It is not a proof "
            "about the original idealized sensing model or a full mission simulation."
        ),
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
