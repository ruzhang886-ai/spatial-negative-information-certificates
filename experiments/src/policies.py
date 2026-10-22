"""第一轮四种成对补测消融策略。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .geometry import minimum_enclosing_circle, polygon_area
from .gain_aware_verifier import certify_gain_aware_multiplicity
from .paired_certificate import PairCertificate, build_pair_certificate
from .paired_certificate import (
    SpatialRedundancyCertificate,
    build_spatial_redundancy_certificate,
)
from .simulator import CostConfig, Mission, ModelConfig, Scenario


CORE_POLICIES = (
    "no_negative_information",
    "single_probe_safe",
    "pair_no_cancel",
    "full_pair",
    "gain_aware_full_pair",
)

SPATIAL_POLICIES = {
    "spatial_multicover_q1": 1,
    "spatial_multicover_q2": 2,
    "gain_aware_spatial_q2": 2,
}

TEMPORAL_POLICIES = {
    "temporal_repeat_q1": 1,
    "temporal_repeat_q2": 2,
}

POLICIES = (
    CORE_POLICIES
    + ("repeated_pair_q1",)
    + tuple(TEMPORAL_POLICIES)
    + tuple(SPATIAL_POLICIES)
)


@dataclass(frozen=True)
class PairCandidate:
    certificate: PairCertificate
    ordered_points: tuple[np.ndarray, np.ndarray]
    score_s: float
    area_fraction_removed: float


@dataclass(frozen=True)
class SpatialCandidate:
    certificate: SpatialRedundancyCertificate
    ordered_points: tuple[np.ndarray, ...]
    score_s: float
    area_fraction_removed: float


def _nearest_neighbor_order(points, start) -> tuple[np.ndarray, ...]:
    remaining = [np.asarray(point, dtype=float).copy() for point in points]
    current = np.asarray(start, dtype=float)
    ordered = []
    while remaining:
        index = min(range(len(remaining)), key=lambda i: np.linalg.norm(remaining[i] - current))
        current = remaining.pop(index)
        ordered.append(current)
    return tuple(ordered)


def _side_points_use_distinct_persistent_cells(
    certificate: SpatialRedundancyCertificate,
    correlation_length: float,
) -> bool:
    """检查每一侧的冗余点是否分别落入不同的持久漏检单元。"""
    if correlation_length <= 0:
        return False

    def cell(point) -> tuple[int, int]:
        index = np.floor(np.asarray(point, dtype=float) / correlation_length).astype(int)
        return int(index[0]), int(index[1])

    for side in (
        certificate.negative_side_points,
        certificate.positive_side_points,
    ):
        if len({cell(point) for point in side}) != len(side):
            return False
    return True


def _side_points_have_minimum_separation(
    certificate: SpatialRedundancyCertificate,
    minimum_distance: float,
) -> bool:
    """连续相关场中，每一侧的证书点至少相隔一个相关尺度。"""
    if minimum_distance <= 0:
        return False
    for side in (
        certificate.negative_side_points,
        certificate.positive_side_points,
    ):
        for index, first in enumerate(side):
            for second in side[index + 1 :]:
                if np.linalg.norm(first - second) < minimum_distance - 1e-9:
                    return False
    return True


def choose_pair_candidate(mission: Mission) -> PairCandidate | None:
    """从最近三次成功测向构造满足连续证书的最低代理成本点对。"""
    polygon = mission.state.polygon
    current = mission.position
    before_area = max(polygon_area(polygon), 1e-12)
    _, before_radius = minimum_enclosing_circle(polygon)
    best: PairCandidate | None = None

    for origin, theta in mission.state.positive_observations[-3:]:
        angle = math.radians(theta)
        direction = np.array([math.cos(angle), math.sin(angle)])
        projections = (polygon - origin) @ direction
        low = max(0.0, float(np.min(projections)))
        high = float(np.max(projections))
        if high - low < 0.1:
            continue
        for fraction in (0.35, 0.50, 0.65):
            a = low + fraction * (high - low)
            wedge_width = a * math.tan(math.radians(mission.model.bearing_error_deg))
            for scale in (0.16, 0.28):
                b = max(
                    wedge_width + 1.0,
                    min(200.0, max(30.0, scale * (high - low))),
                )
                certificate = build_pair_certificate(
                    polygon,
                    origin,
                    theta,
                    mission.model.bearing_error_deg,
                    a,
                    b,
                    mission.model.reliable_pair_radius_m,
                )
                if not certificate.valid:
                    continue
                points = [certificate.q_minus.copy(), certificate.q_plus.copy()]
                if any(
                    any(np.linalg.norm(point - previous) < 0.5 for previous in mission.measured_points)
                    for point in points
                ):
                    continue
                points.sort(key=lambda point: np.linalg.norm(point - current))
                path = float(
                    np.linalg.norm(points[0] - current)
                    + np.linalg.norm(points[1] - points[0])
                )
                near_center, near_radius = minimum_enclosing_circle(
                    certificate.near_polygon
                )
                continuation = float(np.linalg.norm(points[1] - near_center)) / mission.costs.speed_mps
                score = (
                    path / mission.costs.speed_mps
                    + 2 * mission.costs.measure_s
                    + continuation
                    + 2 * min(before_radius, near_radius) / mission.costs.speed_mps
                )
                removed = 1.0 - polygon_area(certificate.near_polygon) / before_area
                candidate = PairCandidate(
                    certificate=certificate,
                    ordered_points=(points[0], points[1]),
                    score_s=score,
                    area_fraction_removed=removed,
                )
                if best is None or candidate.score_s < best.score_s:
                    best = candidate
    return best


def choose_spatial_candidate(mission: Mission, q: int = 1) -> SpatialCandidate | None:
    """选择两侧各 q+1 个不同空间点的最低代理成本证书。"""
    polygon = mission.state.polygon
    current = mission.position
    before_area = max(polygon_area(polygon), 1e-12)
    _, before_radius = minimum_enclosing_circle(polygon)
    best: SpatialCandidate | None = None

    for origin, theta in mission.state.positive_observations[-3:]:
        angle = math.radians(theta)
        direction = np.array([math.cos(angle), math.sin(angle)])
        projections = (polygon - origin) @ direction
        low = max(0.0, float(np.min(projections)))
        high = float(np.max(projections))
        if high - low < 0.1:
            continue
        for fraction in (0.35, 0.50, 0.65):
            a = low + fraction * (high - low)
            wedge_width = a * math.tan(math.radians(mission.model.bearing_error_deg))
            for scale in (0.16, 0.28):
                inner = max(
                    wedge_width + 1.0,
                    min(180.0, max(30.0, scale * (high - low))),
                )
                spacing_values = [25.0, 50.0, 75.0]
                if mission.model.miss_mode == "correlated_shadowing":
                    spacing_values.extend(
                        [
                            mission.model.miss_correlation_m,
                            1.5 * mission.model.miss_correlation_m,
                        ]
                    )
                for spacing in sorted(set(spacing_values)):
                    certificate = build_spatial_redundancy_certificate(
                        polygon,
                        origin,
                        theta,
                        mission.model.bearing_error_deg,
                        a,
                        inner,
                        spacing,
                        mission.model.reliable_pair_radius_m,
                        q,
                    )
                    if not certificate.valid:
                        continue
                    if (
                        mission.model.miss_mode == "persistent_location"
                        and not _side_points_use_distinct_persistent_cells(
                            certificate,
                            mission.model.miss_correlation_m,
                        )
                    ):
                        continue
                    if (
                        mission.model.miss_mode == "correlated_shadowing"
                        and not _side_points_have_minimum_separation(
                            certificate,
                            mission.model.miss_correlation_m,
                        )
                    ):
                        continue
                    if any(
                        any(
                            np.linalg.norm(point - previous) < 0.5
                            for previous in mission.measured_points
                        )
                        for point in certificate.points
                    ):
                        continue
                    ordered = _nearest_neighbor_order(certificate.points, current)
                    path = float(np.linalg.norm(ordered[0] - current))
                    path += sum(
                        float(np.linalg.norm(b - a_point))
                        for a_point, b in zip(ordered, ordered[1:])
                    )
                    near_center, near_radius = minimum_enclosing_circle(
                        certificate.base.near_polygon
                    )
                    continuation = float(np.linalg.norm(ordered[-1] - near_center)) / mission.costs.speed_mps
                    score = (
                        path / mission.costs.speed_mps
                        + len(ordered) * mission.costs.measure_s
                        + continuation
                        + 2 * min(before_radius, near_radius) / mission.costs.speed_mps
                    )
                    removed = 1.0 - polygon_area(certificate.base.near_polygon) / before_area
                    candidate = SpatialCandidate(
                        certificate=certificate,
                        ordered_points=ordered,
                        score_s=score,
                        area_fraction_removed=removed,
                    )
                    if best is None or candidate.score_s < best.score_s:
                        best = candidate
    return best


def _gain_certificate_result(candidate, mission: Mission, *, spatial: bool):
    """非理想方向图下，连续证书失败或未决时禁止硬删除。"""
    if spatial:
        polygon = candidate.certificate.base.far_polygon
        history = candidate.certificate.base.origin
        points = candidate.ordered_points
        required = candidate.certificate.guaranteed_detection_multiplicity
    else:
        polygon = candidate.certificate.far_polygon
        history = candidate.certificate.origin
        points = candidate.ordered_points
        required = 1
    return certify_gain_aware_multiplicity(
        polygon,
        history,
        points,
        required,
        mission.model,
        geometric_visibility_multiplicity=required,
        position_tolerance_m=2.0,
        orientation_tolerance_deg=0.5,
        max_cells=300000,
        counterexample_position_subdivisions=10,
        counterexample_orientation_step_deg=2.0,
    )


def run_policy(
    scenario: Scenario,
    policy: str,
    model: ModelConfig | None = None,
    costs: CostConfig | None = None,
) -> dict:
    """运行单场景单策略，返回可序列化指标。"""
    if policy not in POLICIES:
        raise ValueError(f"未知策略：{policy}")
    model = model or ModelConfig()
    costs = costs or CostConfig()
    mission = Mission(scenario, policy, model, costs)

    for _ in range(model.max_pair_rounds):
        if mission.invalid_state:
            return mission.metrics.serializable()
        if mission.try_terminal_clear():
            return mission.metrics.serializable()
        if policy in SPATIAL_POLICIES:
            q = SPATIAL_POLICIES[policy]
            spatial_candidate = choose_spatial_candidate(mission, q=q)
            if spatial_candidate is None:
                break
            if (
                policy == "gain_aware_spatial_q2"
                and mission.model.antenna_mode == "parabolic_halfplane"
            ):
                gain_result = _gain_certificate_result(
                    spatial_candidate, mission, spatial=True
                )
                mission.metrics.events.append(
                    {
                        "action": "gain_certificate_checked",
                        "status": gain_result.status,
                        "cells_visited": gain_result.cells_visited,
                        "required_multiplicity": (
                            gain_result.required_multiplicity
                        ),
                    }
                )
                if gain_result.status != "certified":
                    break
            mission.metrics.pair_rounds += 1
            mission.metrics.certificate_executions += 1
            mission.metrics.events.append(
                {
                    "action": "spatial_multicover_selected",
                    "tolerated_false_negatives": q,
                    "score_s": spatial_candidate.score_s,
                    "area_fraction_removed_if_all_miss": spatial_candidate.area_fraction_removed,
                    "points": [point.tolist() for point in spatial_candidate.ordered_points],
                }
            )
            received = False
            failed_sites: set[tuple] = set()
            for point in spatial_candidate.ordered_points:
                result = mission.measure(point)
                if (
                    result.false_negative
                    and mission.model.miss_mode
                    in ("persistent_location", "correlated_shadowing")
                ):
                    failed_sites.add(mission.failure_site_key(point))
                if mission.invalid_state:
                    return mission.metrics.serializable()
                if result.kind != "no_signal":
                    mission.metrics.early_cancellations += 1
                    received = True
                    break
            if not received:
                mission.metrics.all_negative_certificates += 1
                mission.metrics.max_failed_cells_in_certificate = max(
                    mission.metrics.max_failed_cells_in_certificate,
                    len(failed_sites),
                )
                if len(failed_sites) > q:
                    mission.metrics.budget_exceeded_certificates += 1
                mission.apply_pair_cut(spatial_candidate.certificate.base.near_polygon)
            continue

        candidate = choose_pair_candidate(mission)
        if candidate is None:
            break
        if (
            policy == "gain_aware_full_pair"
            and mission.model.antenna_mode == "parabolic_halfplane"
        ):
            gain_result = _gain_certificate_result(
                candidate, mission, spatial=False
            )
            mission.metrics.events.append(
                {
                    "action": "gain_certificate_checked",
                    "status": gain_result.status,
                    "cells_visited": gain_result.cells_visited,
                    "required_multiplicity": gain_result.required_multiplicity,
                }
            )
            if gain_result.status != "certified":
                break
        mission.metrics.pair_rounds += 1
        certificate = candidate.certificate
        first, second = candidate.ordered_points
        mission.metrics.events.append(
            {
                "action": "pair_selected",
                "score_s": candidate.score_s,
                "area_fraction_removed_if_double_miss": candidate.area_fraction_removed,
                "points": [first.tolist(), second.tolist()],
            }
        )

        if policy in TEMPORAL_POLICIES:
            q = TEMPORAL_POLICIES[policy]
            mission.metrics.certificate_executions += 1
            received = False
            for point in (first, second):
                for _ in range(q + 1):
                    result = mission.measure(point)
                    if mission.invalid_state:
                        return mission.metrics.serializable()
                    if result.kind != "no_signal":
                        mission.metrics.early_cancellations += 1
                        received = True
                        break
                if received:
                    break
            if not received:
                mission.metrics.all_negative_certificates += 1
                mission.apply_pair_cut(certificate.near_polygon)
            continue

        if policy == "single_probe_safe":
            mission.measure(first)
            if mission.invalid_state:
                return mission.metrics.serializable()
            continue

        first_result = mission.measure(first)
        if mission.invalid_state:
            return mission.metrics.serializable()
        if policy in (
            "full_pair",
            "gain_aware_full_pair",
            "repeated_pair_q1",
        ) and first_result.kind != "no_signal":
            mission.metrics.early_cancellations += 1
            continue

        second_result = mission.measure(second)
        if mission.invalid_state:
            return mission.metrics.serializable()
        if policy == "repeated_pair_q1" and second_result.kind != "no_signal":
            mission.metrics.early_cancellations += 1
            continue
        if policy == "repeated_pair_q1" and second_result.kind == "no_signal":
            repeated_first = mission.measure(first)
            if mission.invalid_state:
                return mission.metrics.serializable()
            if repeated_first.kind != "no_signal":
                mission.metrics.early_cancellations += 1
                continue
            repeated_second = mission.measure(second)
            if mission.invalid_state:
                return mission.metrics.serializable()
            if repeated_second.kind != "no_signal":
                mission.metrics.early_cancellations += 1
                continue
            mission.apply_pair_cut(certificate.near_polygon)
            continue

        if policy in ("pair_no_cancel", "full_pair", "gain_aware_full_pair"):
            if first_result.kind == "no_signal" and second_result.kind == "no_signal":
                mission.apply_pair_cut(certificate.near_polygon)

        # no_negative_information 与 pair_no_cancel 执行相同两点；前者忽略双无信号。

    if not mission.metrics.completed:
        if mission.invalid_state:
            return mission.metrics.serializable()
        if not mission.try_terminal_clear():
            mission.fallback()
    return mission.metrics.serializable()
