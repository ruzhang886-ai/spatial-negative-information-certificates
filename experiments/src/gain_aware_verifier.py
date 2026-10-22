"""非理想方向图下的位置—朝向联合有限状态核验器。

本模块只提供数值审查，不把有限采样结果当作连续状态证明。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .geometry import as_polygon
from .simulator import ModelConfig


@dataclass(frozen=True)
class SampledGainAudit:
    """候选测点组在有限位置—朝向状态集上的检测重数审查。"""

    required_multiplicity: int
    minimum_guaranteed_multiplicity: int
    feasible_states_checked: int
    position_samples_checked: int
    valid_on_sampled_states: bool
    witness_source: tuple[float, float] | None
    witness_boresight_deg: float | None
    witness_lower_radius_m: float | None
    witness_detection_count: int | None
    minimum_required_order_margin_m: float
    margin_witness_source: tuple[float, float] | None
    margin_witness_boresight_deg: float | None


@dataclass(frozen=True)
class ContinuousGainCertificate:
    """区间分支核验结果；只有 status=certified 才是连续域证书。"""

    status: str
    required_multiplicity: int
    cells_visited: int
    cells_certified: int
    cells_discarded_as_infeasible: int
    maximum_queue_size: int
    unresolved_position_radius_m: float | None = None
    unresolved_orientation_halfwidth_deg: float | None = None
    unresolved_source_center: tuple[float, float] | None = None
    unresolved_boresight_center_deg: float | None = None
    counterexample_source: tuple[float, float] | None = None
    counterexample_boresight_deg: float | None = None
    counterexample_detection_count: int | None = None


def parabolic_range_factor_from_angle(angle_deg, config: ModelConfig):
    """由离轴角计算相对主轴有效接收半径比例。"""
    angle = np.asarray(angle_deg, dtype=float)
    attenuation_db = np.minimum(
        12.0
        * (angle / config.antenna_half_power_beamwidth_deg) ** 2,
        config.antenna_max_attenuation_db,
    )
    factor = 10.0 ** (
        -attenuation_db / (10.0 * config.path_loss_exponent)
    )
    return np.where(angle <= 90.0 + 1e-10, factor, 0.0)


def sample_convex_polygon(polygon, subdivisions: int = 12) -> np.ndarray:
    """在凸多边形边界和内部生成确定性三角网格样本。"""
    polygon = as_polygon(polygon)
    if subdivisions < 2:
        raise ValueError("位置采样细分数至少为2。")
    center = np.mean(polygon, axis=0)
    samples: dict[tuple[float, float], np.ndarray] = {}

    def add(point) -> None:
        point = np.asarray(point, dtype=float).reshape(2)
        key = (round(float(point[0]), 9), round(float(point[1]), 9))
        samples[key] = point

    add(center)
    for vertex in polygon:
        add(vertex)
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        for i in range(subdivisions + 1):
            for j in range(subdivisions + 1 - i):
                weight_first = i / subdivisions
                weight_second = j / subdivisions
                weight_center = 1.0 - weight_first - weight_second
                add(
                    weight_center * center
                    + weight_first * first
                    + weight_second * second
                )
    return np.asarray(list(samples.values()), dtype=float)


def _angle_deg(vectors: np.ndarray, directions: np.ndarray) -> np.ndarray:
    vector_norms = np.linalg.norm(vectors, axis=-1)
    direction_norms = np.linalg.norm(directions, axis=-1)
    denominator = np.maximum(vector_norms * direction_norms, 1e-15)
    cosine = np.sum(vectors * directions, axis=-1) / denominator
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _critical_boresights(source, history_point, probe_points) -> np.ndarray:
    """加入主轴和半平面边界方向，强化有限角度扫描。"""
    source = np.asarray(source, dtype=float).reshape(2)
    targets = np.vstack(
        [np.asarray(history_point, dtype=float).reshape(1, 2), probe_points]
    )
    offsets = targets - source
    nonzero = np.linalg.norm(offsets, axis=1) > 1e-10
    angles = np.degrees(np.arctan2(offsets[nonzero, 1], offsets[nonzero, 0]))
    critical = []
    for angle in angles:
        critical.extend((angle - 90.0, angle, angle + 90.0))
    return np.mod(np.asarray(critical, dtype=float), 360.0)


def gain_aware_detection_count(
    source,
    boresight_deg: float,
    history_point,
    probe_points,
    config: ModelConfig,
) -> tuple[bool, int, float]:
    """计算一个精确状态是否与历史正观测相容及其保证检测重数。"""
    source = np.asarray(source, dtype=float).reshape(2)
    history = np.asarray(history_point, dtype=float).reshape(2)
    probes = np.asarray(probe_points, dtype=float).reshape(-1, 2)
    radians = math.radians(float(boresight_deg))
    direction = np.array([math.cos(radians), math.sin(radians)])

    history_offset = history - source
    history_distance = float(np.linalg.norm(history_offset))
    if history_distance <= config.near_radius_m + 1e-9:
        return False, 0, math.inf
    history_angle = float(
        _angle_deg(history_offset.reshape(1, 2), direction.reshape(1, 2))[0]
    )
    history_factor = float(parabolic_range_factor_from_angle(history_angle, config))
    if history_factor <= 0:
        return False, 0, math.inf
    required_by_history = history_distance / history_factor
    if required_by_history > config.max_reception_m + 1e-9:
        return False, 0, required_by_history
    lower_radius = max(config.min_reception_m, required_by_history)

    count = 0
    for probe in probes:
        offset = probe - source
        distance = float(np.linalg.norm(offset))
        if distance <= config.near_radius_m + 1e-9:
            count += 1
            continue
        angle = float(
            _angle_deg(offset.reshape(1, 2), direction.reshape(1, 2))[0]
        )
        factor = float(parabolic_range_factor_from_angle(angle, config))
        if factor > 0 and distance <= lower_radius * factor + 1e-9:
            count += 1
    return True, count, lower_radius


def sampled_gain_aware_multiplicity(
    feasible_polygon,
    history_point,
    probe_points,
    required_multiplicity: int,
    config: ModelConfig,
    *,
    position_subdivisions: int = 12,
    orientation_step_deg: float = 2.0,
) -> SampledGainAudit:
    """核验有限状态集上每个可行状态的保证检测重数。

    历史点已经成功接收，因此每个位置—朝向状态都给出主轴接收
    半径的条件下界。新测点只在最小相容半径下仍能接收时计入
    “保证检测”。
    """
    if config.antenna_mode != "parabolic_halfplane":
        raise ValueError("增益感知核验器只用于抛物线半平面方向图。")
    if required_multiplicity < 1:
        raise ValueError("要求的检测重数至少为1。")
    if orientation_step_deg <= 0 or orientation_step_deg > 30:
        raise ValueError("方向采样步长必须位于(0,30]度。")

    history = np.asarray(history_point, dtype=float).reshape(2)
    probes = np.asarray(probe_points, dtype=float).reshape(-1, 2)
    if len(probes) < required_multiplicity:
        raise ValueError("测点数量小于要求的检测重数。")

    positions = sample_convex_polygon(
        feasible_polygon, subdivisions=position_subdivisions
    )
    base_angles = np.arange(0.0, 360.0, orientation_step_deg)
    minimum_count = len(probes)
    minimum_order_margin = math.inf
    feasible_states = 0
    checked_positions = 0
    witness = None
    margin_witness = None

    for source in positions:
        history_distance = float(np.linalg.norm(history - source))
        if history_distance <= config.near_radius_m + 1e-9:
            continue
        if history_distance > config.max_reception_m + 1e-9:
            continue

        critical = _critical_boresights(source, history, probes)
        boresight_deg = np.unique(
            np.round(np.concatenate([base_angles, critical]), decimals=10)
        )
        radians = np.radians(boresight_deg)
        directions = np.column_stack([np.cos(radians), np.sin(radians)])

        history_vectors = np.repeat(
            (history - source).reshape(1, 2), len(directions), axis=0
        )
        history_angles = _angle_deg(history_vectors, directions)
        history_factor = parabolic_range_factor_from_angle(
            history_angles, config
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            history_required_radius = np.where(
                history_factor > 0,
                history_distance / history_factor,
                np.inf,
            )
        compatible = history_required_radius <= config.max_reception_m + 1e-9
        if not np.any(compatible):
            continue

        checked_positions += 1
        directions = directions[compatible]
        compatible_boresights = boresight_deg[compatible]
        lower_radius = np.maximum(
            config.min_reception_m,
            history_required_radius[compatible],
        )
        feasible_states += len(directions)

        margins = np.empty((len(directions), len(probes)), dtype=float)
        for probe_index, probe in enumerate(probes):
            probe_offset = probe - source
            probe_distance = float(np.linalg.norm(probe_offset))
            if probe_distance <= config.near_radius_m + 1e-9:
                margins[:, probe_index] = math.inf
                continue
            probe_vectors = np.repeat(
                probe_offset.reshape(1, 2), len(directions), axis=0
            )
            probe_angles = _angle_deg(probe_vectors, directions)
            probe_factor = parabolic_range_factor_from_angle(
                probe_angles, config
            )
            margins[:, probe_index] = lower_radius * probe_factor - probe_distance

        counts = np.sum(margins >= -1e-9, axis=1)
        required_order_margin = np.partition(
            margins, -required_multiplicity, axis=1
        )[:, -required_multiplicity]
        margin_index = int(np.argmin(required_order_margin))
        local_margin = float(required_order_margin[margin_index])
        if local_margin < minimum_order_margin:
            minimum_order_margin = local_margin
            margin_witness = (
                source.copy(),
                float(compatible_boresights[margin_index]),
            )

        local_index = int(np.argmin(counts))
        local_minimum = int(counts[local_index])
        if local_minimum < minimum_count:
            minimum_count = local_minimum
            witness = (
                source.copy(),
                float(compatible_boresights[local_index]),
                float(lower_radius[local_index]),
                local_minimum,
            )

    if feasible_states == 0:
        raise ValueError("有限状态集中没有与历史成功观测相容的状态。")
    if witness is None:
        witness = (positions[0].copy(), 0.0, config.min_reception_m, minimum_count)
    if margin_witness is None:
        margin_witness = (positions[0].copy(), 0.0)

    return SampledGainAudit(
        required_multiplicity=required_multiplicity,
        minimum_guaranteed_multiplicity=minimum_count,
        feasible_states_checked=feasible_states,
        position_samples_checked=checked_positions,
        valid_on_sampled_states=minimum_count >= required_multiplicity,
        witness_source=(float(witness[0][0]), float(witness[0][1])),
        witness_boresight_deg=float(witness[1]),
        witness_lower_radius_m=float(witness[2]),
        witness_detection_count=int(witness[3]),
        minimum_required_order_margin_m=float(minimum_order_margin),
        margin_witness_source=(
            float(margin_witness[0][0]), float(margin_witness[0][1])
        ),
        margin_witness_boresight_deg=float(margin_witness[1]),
    )


def _circular_difference_deg(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _target_bounds_for_cell(
    triangle: np.ndarray,
    boresight_center_deg: float,
    boresight_halfwidth_deg: float,
    target: np.ndarray,
) -> tuple[float, float, float, float]:
    """目标距离和离轴角的保守上下界。"""
    center = np.mean(triangle, axis=0)
    position_radius = float(np.max(np.linalg.norm(triangle - center, axis=1)))
    center_vector = np.asarray(target, dtype=float).reshape(2) - center
    center_distance = float(np.linalg.norm(center_vector))
    distance_low = max(0.0, center_distance - position_radius)
    distance_high = center_distance + position_radius
    if center_distance <= position_radius + 1e-12:
        direction_uncertainty = 180.0
        direction_angle = 0.0
    else:
        direction_uncertainty = math.degrees(
            math.asin(min(1.0, position_radius / center_distance))
        )
        direction_angle = math.degrees(
            math.atan2(center_vector[1], center_vector[0])
        )
    center_difference = _circular_difference_deg(
        direction_angle, boresight_center_deg
    )
    total_uncertainty = boresight_halfwidth_deg + direction_uncertainty
    angle_low = max(0.0, center_difference - total_uncertainty)
    angle_high = min(180.0, center_difference + total_uncertainty)
    return distance_low, distance_high, angle_low, angle_high


def _subdivide_triangle(triangle: np.ndarray) -> tuple[np.ndarray, ...]:
    first, second, third = triangle
    first_second = (first + second) / 2.0
    second_third = (second + third) / 2.0
    third_first = (third + first) / 2.0
    return (
        np.array([first, first_second, third_first]),
        np.array([first_second, second, second_third]),
        np.array([third_first, second_third, third]),
        np.array([first_second, second_third, third_first]),
    )


def certify_gain_aware_multiplicity(
    feasible_polygon,
    history_point,
    probe_points,
    required_multiplicity: int,
    config: ModelConfig,
    *,
    geometric_visibility_multiplicity: int = 0,
    initial_orientation_bins: int = 16,
    position_tolerance_m: float = 1.0,
    orientation_tolerance_deg: float = 0.25,
    max_cells: int = 500000,
    numeric_safety_margin_m: float = 1e-6,
    counterexample_position_subdivisions: int = 10,
    counterexample_orientation_step_deg: float = 2.0,
) -> ContinuousGainCertificate:
    """用保守区间界核验全部连续位置—朝向状态。

    返回 `certified` 表示每个与历史正观测相容的连续状态都至少被
    required_multiplicity 个新测点保证接收；`counterexample` 表示
    已找到一个精确可行状态违反条件；`unresolved` 表示区间界或预算
    尚不足，不能据此执行硬删除。
    """
    if config.antenna_mode != "parabolic_halfplane":
        raise ValueError("连续增益证书只用于抛物线半平面方向图。")
    if initial_orientation_bins < 4:
        raise ValueError("初始朝向区间数至少为4。")
    polygon = as_polygon(feasible_polygon)
    history = np.asarray(history_point, dtype=float).reshape(2)
    probes = np.asarray(probe_points, dtype=float).reshape(-1, 2)
    if len(probes) < required_multiplicity:
        raise ValueError("测点数量小于要求的检测重数。")
    if geometric_visibility_multiplicity < 0:
        raise ValueError("几何可见重数不能为负。")
    if geometric_visibility_multiplicity > len(probes):
        raise ValueError("几何可见重数不能超过测点总数。")
    if numeric_safety_margin_m < 0:
        raise ValueError("数值安全余量不能为负。")

    sampled = sampled_gain_aware_multiplicity(
        polygon,
        history,
        probes,
        required_multiplicity,
        config,
        position_subdivisions=counterexample_position_subdivisions,
        orientation_step_deg=counterexample_orientation_step_deg,
    )
    if not sampled.valid_on_sampled_states:
        compatible, count, _ = gain_aware_detection_count(
            sampled.witness_source,
            sampled.witness_boresight_deg,
            history,
            probes,
            config,
        )
        if compatible and count < required_multiplicity:
            return ContinuousGainCertificate(
                status="counterexample",
                required_multiplicity=required_multiplicity,
                cells_visited=0,
                cells_certified=0,
                cells_discarded_as_infeasible=0,
                maximum_queue_size=0,
                counterexample_source=sampled.witness_source,
                counterexample_boresight_deg=sampled.witness_boresight_deg,
                counterexample_detection_count=count,
            )

    polygon_center = np.mean(polygon, axis=0)
    triangles = [
        np.array(
            [polygon_center, polygon[index], polygon[(index + 1) % len(polygon)]]
        )
        for index in range(len(polygon))
    ]
    orientation_width = 360.0 / initial_orientation_bins
    queue = []
    for triangle in triangles:
        for index in range(initial_orientation_bins):
            queue.append(
                (
                    triangle,
                    (index + 0.5) * orientation_width,
                    orientation_width / 2.0,
                )
            )

    visited = 0
    certified = 0
    discarded = 0
    maximum_queue = len(queue)
    while queue:
        maximum_queue = max(maximum_queue, len(queue))
        triangle, beta_center, beta_halfwidth = queue.pop()
        visited += 1
        if visited > max_cells:
            center = np.mean(triangle, axis=0)
            radius = float(np.max(np.linalg.norm(triangle - center, axis=1)))
            return ContinuousGainCertificate(
                status="unresolved",
                required_multiplicity=required_multiplicity,
                cells_visited=visited,
                cells_certified=certified,
                cells_discarded_as_infeasible=discarded,
                maximum_queue_size=maximum_queue,
                unresolved_position_radius_m=radius,
                unresolved_orientation_halfwidth_deg=beta_halfwidth,
                unresolved_source_center=(float(center[0]), float(center[1])),
                unresolved_boresight_center_deg=float(beta_center),
            )

        history_low, _, history_angle_low, _ = _target_bounds_for_cell(
            triangle, beta_center, beta_halfwidth, history
        )
        history_factor_upper = float(
            parabolic_range_factor_from_angle(history_angle_low, config)
        )
        if history_factor_upper <= 0:
            discarded += 1
            continue
        history_required_lower = history_low / history_factor_upper
        if (
            history_required_lower
            > config.max_reception_m + numeric_safety_margin_m
        ):
            discarded += 1
            continue
        lower_radius_bound = max(
            config.min_reception_m, history_required_lower
        )

        guaranteed = 0
        for probe in probes:
            if np.linalg.norm(probe - history) <= 1e-10:
                guaranteed += 1
                continue
            _, distance_high, _, angle_high = _target_bounds_for_cell(
                triangle, beta_center, beta_halfwidth, probe
            )
            if distance_high <= config.near_radius_m + 1e-9:
                guaranteed += 1
                continue
            if angle_high > 90.0 + 1e-10:
                continue
            factor_lower = float(
                parabolic_range_factor_from_angle(angle_high, config)
            )
            if (
                factor_lower > 0
                and distance_high + numeric_safety_margin_m
                <= lower_radius_bound * factor_lower
            ):
                guaranteed += 1

        if guaranteed >= required_multiplicity:
            certified += 1
            continue

        # 横向几何定理保证至少一侧的指定数量测点位于发射半平面内。
        # 若本单元内所有测点即使取90度边缘增益也满足距离条件，则
        # 不需要固定由哪一侧负责，即可处理半平面边界上的接力覆盖。
        if geometric_visibility_multiplicity >= required_multiplicity:
            edge_factor = float(
                parabolic_range_factor_from_angle(90.0, config)
            )
            all_edge_safe = True
            for probe in probes:
                _, distance_high, _, _ = _target_bounds_for_cell(
                    triangle, beta_center, beta_halfwidth, probe
                )
                if (
                    distance_high + numeric_safety_margin_m
                    > lower_radius_bound * edge_factor
                ):
                    all_edge_safe = False
                    break
            if all_edge_safe:
                certified += 1
                continue

        center = np.mean(triangle, axis=0)
        position_radius = float(
            np.max(np.linalg.norm(triangle - center, axis=1))
        )
        if (
            position_radius <= position_tolerance_m
            and beta_halfwidth <= orientation_tolerance_deg
        ):
            return ContinuousGainCertificate(
                status="unresolved",
                required_multiplicity=required_multiplicity,
                cells_visited=visited,
                cells_certified=certified,
                cells_discarded_as_infeasible=discarded,
                maximum_queue_size=maximum_queue,
                unresolved_position_radius_m=position_radius,
                unresolved_orientation_halfwidth_deg=beta_halfwidth,
                unresolved_source_center=(float(center[0]), float(center[1])),
                unresolved_boresight_center_deg=float(beta_center),
            )

        position_ratio = position_radius / max(position_tolerance_m, 1e-12)
        orientation_ratio = beta_halfwidth / max(
            orientation_tolerance_deg, 1e-12
        )
        if position_ratio >= orientation_ratio:
            for child in _subdivide_triangle(triangle):
                queue.append((child, beta_center, beta_halfwidth))
        else:
            child_halfwidth = beta_halfwidth / 2.0
            queue.append(
                (
                    triangle,
                    (beta_center - child_halfwidth) % 360.0,
                    child_halfwidth,
                )
            )
            queue.append(
                (
                    triangle,
                    (beta_center + child_halfwidth) % 360.0,
                    child_halfwidth,
                )
            )

    return ContinuousGainCertificate(
        status="certified",
        required_multiplicity=required_multiplicity,
        cells_visited=visited,
        cells_certified=certified,
        cells_discarded_as_infeasible=discarded,
        maximum_queue_size=maximum_queue,
    )
