"""成对主动补测的连续充分条件。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .geometry import as_polygon, clip_halfplane, max_distance_to_polygon_vertices


@dataclass(frozen=True)
class PairCertificate:
    """一次远侧排除所需的全部可审计量。"""

    origin: np.ndarray
    direction: np.ndarray
    sideways: np.ndarray
    longitudinal_offset: float
    transverse_offset: float
    q_minus: np.ndarray
    q_plus: np.ndarray
    near_polygon: np.ndarray
    far_polygon: np.ndarray
    far_max_distance: float
    reliable_radius: float
    valid: bool
    reason: str


@dataclass(frozen=True)
class SpatialRedundancyCertificate:
    """横截线两侧各 q+1 个不同位置形成的空间检测重数证书。"""

    base: PairCertificate
    tolerated_false_negatives: int
    negative_side_points: tuple[np.ndarray, ...]
    positive_side_points: tuple[np.ndarray, ...]
    far_max_distance: float
    valid: bool
    reason: str

    @property
    def points(self) -> tuple[np.ndarray, ...]:
        return self.negative_side_points + self.positive_side_points

    @property
    def guaranteed_detection_multiplicity(self) -> int:
        return self.tolerated_false_negatives + 1


def build_pair_certificate(
    feasible_polygon,
    origin,
    bearing_deg: float,
    bearing_error_deg: float,
    longitudinal_offset: float,
    transverse_offset: float,
    reliable_radius: float,
    *,
    strict_margin: float = 1e-9,
) -> PairCertificate:
    """构造并核验一个成对补测证书。

    远侧集合满足 d·(G-S) >= a。只有横向条件 b > a tan(delta)
    和两个补测点到整个远侧凸多边形的最大距离均不超过可靠半径时，
    `valid` 才为真。
    """
    polygon = as_polygon(feasible_polygon)
    origin = np.asarray(origin, dtype=float).reshape(2)
    if len(polygon) < 3:
        raise ValueError("可行区域必须是非退化凸多边形。")
    if not 0 <= bearing_error_deg < 90:
        raise ValueError("测向误差半角必须位于 [0, 90) 度。")
    if longitudinal_offset <= 0 or transverse_offset <= 0 or reliable_radius <= 0:
        raise ValueError("纵向距离、横向距离和可靠半径必须为正。")

    angle = math.radians(float(bearing_deg))
    direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)
    sideways = np.array([-direction[1], direction[0]], dtype=float)
    a = float(longitudinal_offset)
    b = float(transverse_offset)

    split_limit = float(direction @ origin) + a
    near_polygon = clip_halfplane(polygon, direction, split_limit)
    far_polygon = clip_halfplane(polygon, -direction, -split_limit)

    q_minus = origin + a * direction - b * sideways
    q_plus = origin + a * direction + b * sideways

    empty_reason = ""
    if len(near_polygon) == 0:
        empty_reason = "近侧区域为空。"
    elif len(far_polygon) == 0:
        empty_reason = "远侧区域为空。"

    required_b = a * math.tan(math.radians(float(bearing_error_deg)))
    baseline_ok = b > required_b + strict_margin

    if len(far_polygon):
        far_max_distance = max(
            max_distance_to_polygon_vertices(q_minus, far_polygon),
            max_distance_to_polygon_vertices(q_plus, far_polygon),
        )
        distance_ok = far_max_distance <= reliable_radius + 1e-9
    else:
        far_max_distance = math.inf
        distance_ok = False

    valid = not empty_reason and baseline_ok and distance_ok
    if empty_reason:
        reason = empty_reason
    elif not baseline_ok:
        reason = "横向间隔未严格超过测向扇形在横截线上的宽度。"
    elif not distance_ok:
        reason = "至少一个补测点不能保证覆盖整个远侧区域的接收距离。"
    else:
        reason = "满足连续横向条件和连续远侧距离条件。"

    return PairCertificate(
        origin=origin,
        direction=direction,
        sideways=sideways,
        longitudinal_offset=a,
        transverse_offset=b,
        q_minus=q_minus,
        q_plus=q_plus,
        near_polygon=near_polygon,
        far_polygon=far_polygon,
        far_max_distance=far_max_distance,
        reliable_radius=float(reliable_radius),
        valid=valid,
        reason=reason,
    )


def directional_signal(source, emission_direction, point, reception_radius: float) -> bool:
    """理想半平面定向源是否能被指定点接收。"""
    source = np.asarray(source, dtype=float).reshape(2)
    emission_direction = np.asarray(emission_direction, dtype=float).reshape(2)
    point = np.asarray(point, dtype=float).reshape(2)
    direction_norm = float(np.linalg.norm(emission_direction))
    if direction_norm <= 0:
        raise ValueError("发射方向必须非零。")
    emission_direction = emission_direction / direction_norm
    return (
        np.linalg.norm(point - source) <= reception_radius + 1e-9
        and float((point - source) @ emission_direction) >= -1e-9
    )


def pair_must_receive_for_state(
    certificate: PairCertificate,
    source,
    emission_direction,
    reception_radius: float,
) -> bool:
    """离线检查某个真实状态是否至少被证书点对中的一点接收。"""
    if not certificate.valid:
        raise ValueError("只有有效证书才能调用保证检查。")
    return directional_signal(
        source, emission_direction, certificate.q_minus, reception_radius
    ) or directional_signal(
        source, emission_direction, certificate.q_plus, reception_radius
    )


def build_spatial_redundancy_certificate(
    feasible_polygon,
    origin,
    bearing_deg: float,
    bearing_error_deg: float,
    longitudinal_offset: float,
    inner_transverse_offset: float,
    transverse_spacing: float,
    reliable_radius: float,
    tolerated_false_negatives: int,
) -> SpatialRedundancyCertificate:
    """构造两侧各 q+1 个不同位置的连续多重覆盖证书。"""
    if tolerated_false_negatives < 0:
        raise ValueError("可容忍假阴性数不能为负。")
    if transverse_spacing <= 0:
        raise ValueError("不同空间点之间的间距必须为正。")

    base = build_pair_certificate(
        feasible_polygon,
        origin,
        bearing_deg,
        bearing_error_deg,
        longitudinal_offset,
        inner_transverse_offset,
        reliable_radius,
    )
    count = tolerated_false_negatives + 1
    negative = tuple(
        base.origin
        + base.longitudinal_offset * base.direction
        - (base.transverse_offset + index * transverse_spacing) * base.sideways
        for index in range(count)
    )
    positive = tuple(
        base.origin
        + base.longitudinal_offset * base.direction
        + (base.transverse_offset + index * transverse_spacing) * base.sideways
        for index in range(count)
    )

    if len(base.far_polygon):
        far_max_distance = max(
            max_distance_to_polygon_vertices(point, base.far_polygon)
            for point in negative + positive
        )
    else:
        far_max_distance = math.inf

    if not base.valid:
        valid = False
        reason = f"基础横截证书无效：{base.reason}"
    elif far_max_distance > reliable_radius + 1e-9:
        valid = False
        reason = "至少一个空间冗余点不能保证覆盖整个远侧区域的接收距离。"
    else:
        valid = True
        reason = (
            f"两侧各有{count}个不同位置，连续状态的最小理想检测重数至少为{count}。"
        )

    return SpatialRedundancyCertificate(
        base=base,
        tolerated_false_negatives=tolerated_false_negatives,
        negative_side_points=negative,
        positive_side_points=positive,
        far_max_distance=far_max_distance,
        valid=valid,
        reason=reason,
    )


def spatial_detection_count_for_state(
    certificate: SpatialRedundancyCertificate,
    source,
    emission_direction,
    reception_radius: float,
) -> int:
    """离线计算某真实状态被多少个不同空间动作理想检测。"""
    if not certificate.valid:
        raise ValueError("只有有效空间冗余证书才能计算保证重数。")
    return sum(
        directional_signal(source, emission_direction, point, reception_radius)
        for point in certificate.points
    )
