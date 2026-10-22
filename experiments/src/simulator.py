"""首次成功测向后的单源定位与清除实验器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
from statistics import NormalDist

import numpy as np

from .geometry import (
    bearing_wedge_triangle,
    cell_intersects_polygon,
    intersect_convex,
    minimum_enclosing_circle,
    point_in_convex_polygon,
    regular_outer_polygon,
)


@dataclass(frozen=True)
class CostConfig:
    speed_mps: float = 5.0
    measure_s: float = 5.0
    clear_success_s: float = 5.0
    clear_failure_s: float = 3.0


@dataclass(frozen=True)
class ModelConfig:
    bearing_error_deg: float = 1.0
    min_reception_m: float = 1000.0
    max_reception_m: float = 1500.0
    near_radius_m: float = 5.0
    clear_radius_m: float = 20.0
    terminal_radius_m: float = 19.8
    source_domain_radius_m: float = 1800.0
    reliable_pair_radius_m: float = 999.0
    max_pair_rounds: int = 10
    fallback_cell_m: float = 25.0
    polygon_sides: int = 128
    miss_probability: float = 0.0
    miss_mode: str = "independent_action"
    miss_correlation_m: float = 25.0
    shadowing_features: int = 512
    antenna_mode: str = "uniform_halfplane"
    antenna_half_power_beamwidth_deg: float = 65.0
    antenna_max_attenuation_db: float = 30.0
    path_loss_exponent: float = 2.5


@dataclass(frozen=True)
class Scenario:
    seed: int
    source: tuple[float, float]
    emission_direction: tuple[float, float]
    reception_radius_m: float
    initial_position: tuple[float, float]
    initial_bearing_deg: float


@dataclass(frozen=True)
class Observation:
    kind: str
    bearing_deg: float | None = None
    ideal_detectable: bool = False
    false_negative: bool = False


@dataclass
class RunMetrics:
    policy: str
    seed: int
    completed: bool = False
    unsafe_exclusion: bool = False
    time_s: float = 0.0
    distance_m: float = 0.0
    measures: int = 0
    clear_successes: int = 0
    clear_failures: int = 0
    pair_rounds: int = 0
    pair_cuts: int = 0
    early_cancellations: int = 0
    fallback_used: bool = False
    fallback_clear_attempts: int = 0
    false_negative_measurements: int = 0
    distinct_false_negative_cells: int = 0
    distinct_false_negative_sites: int = 0
    certificate_executions: int = 0
    all_negative_certificates: int = 0
    max_failed_cells_in_certificate: int = 0
    budget_exceeded_certificates: int = 0
    terminal_radius_m: float | None = None
    events: list[dict] = field(default_factory=list)

    def serializable(self) -> dict:
        return asdict(self)


class SpatialErrorField:
    """使相同场景、坐标和重复序号在不同策略中共享随机数。"""

    def __init__(self, seed: int, bound_deg: float):
        self.seed = int(seed)
        self.bound_deg = float(bound_deg)

    def unit(self, point, *, occurrence: int = 0, salt: str = "bearing") -> float:
        point = np.asarray(point, dtype=float).reshape(2)
        key = (
            f"{self.seed}:{salt}:{occurrence}:{point[0]:.6f}:{point[1]:.6f}"
        ).encode()
        integer = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        return integer / float(2**64 - 1)

    def at(self, point) -> float:
        return (2 * self.unit(point, salt="bearing") - 1) * self.bound_deg


class CorrelatedShadowingField:
    """具有指数空间相关函数的连续高斯阴影场。

    标准化场的目标相关函数为 exp(-distance / correlation_length)。
    通过二维 Cauchy 频率的随机傅里叶展开生成；同一坐标在一次
    场景中始终返回相同值，避免把空间阴影误当作独立时间噪声。
    """

    def __init__(
        self,
        seed: int,
        correlation_length_m: float,
        features: int = 512,
    ):
        if correlation_length_m <= 0:
            raise ValueError("连续阴影场的相关尺度必须为正。")
        if features < 16:
            raise ValueError("连续阴影场至少需要16个傅里叶特征。")
        self.correlation_length_m = float(correlation_length_m)
        self.features = int(features)
        seed_sequence = np.random.SeedSequence([int(seed), 0x53484144])
        rng = np.random.default_rng(seed_sequence)
        numerator = rng.normal(size=(self.features, 2))
        denominator = np.abs(rng.normal(size=(self.features, 1)))
        denominator = np.maximum(denominator, 1e-12)
        self.frequencies = (
            numerator / denominator / self.correlation_length_m
        )
        scale = 1.0 / math.sqrt(self.features)
        self.cosine_weights = rng.normal(scale=scale, size=self.features)
        self.sine_weights = rng.normal(scale=scale, size=self.features)

    def at(self, point) -> float:
        point = np.asarray(point, dtype=float).reshape(2)
        phase = self.frequencies @ point
        return float(
            self.cosine_weights @ np.cos(phase)
            + self.sine_weights @ np.sin(phase)
        )

    def unit(self, point) -> float:
        """把标准高斯阴影值映射为边缘均匀分位数。"""
        return NormalDist().cdf(self.at(point))


def antenna_range_factor(
    point,
    source,
    emission_direction,
    config: ModelConfig,
) -> float:
    """返回给定离轴角下相对主瓣方向的可检测距离比例。"""
    if config.antenna_mode == "uniform_halfplane":
        return 1.0
    if config.antenna_mode != "parabolic_halfplane":
        raise ValueError(f"未知天线模式：{config.antenna_mode}")
    if config.antenna_half_power_beamwidth_deg <= 0:
        raise ValueError("半功率波束宽度必须为正。")
    if config.antenna_max_attenuation_db < 0:
        raise ValueError("最大方向图衰减不能为负。")
    if config.path_loss_exponent <= 0:
        raise ValueError("路径损耗指数必须为正。")

    point = np.asarray(point, dtype=float).reshape(2)
    source = np.asarray(source, dtype=float).reshape(2)
    emission = np.asarray(emission_direction, dtype=float).reshape(2)
    emission /= np.linalg.norm(emission)
    offset = point - source
    distance = float(np.linalg.norm(offset))
    if distance <= 1e-12:
        return 1.0
    cosine = float(np.clip((offset / distance) @ emission, -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cosine))
    if angle_deg > 90.0 + 1e-9:
        return 0.0
    attenuation_db = min(
        12.0
        * (angle_deg / config.antenna_half_power_beamwidth_deg) ** 2,
        config.antenna_max_attenuation_db,
    )
    return 10.0 ** (
        -attenuation_db / (10.0 * config.path_loss_exponent)
    )


class IdealDirectionalSensor:
    def __init__(self, scenario: Scenario, config: ModelConfig):
        self.source = np.asarray(scenario.source, dtype=float)
        self.emission = np.asarray(scenario.emission_direction, dtype=float)
        self.emission /= np.linalg.norm(self.emission)
        self.reception_radius = float(scenario.reception_radius_m)
        self.config = config
        self.error = SpatialErrorField(scenario.seed, config.bearing_error_deg)
        self.shadowing = (
            CorrelatedShadowingField(
                scenario.seed,
                config.miss_correlation_m,
                config.shadowing_features,
            )
            if config.miss_mode == "correlated_shadowing"
            else None
        )

    def measure(self, point, *, occurrence: int = 0) -> Observation:
        point = np.asarray(point, dtype=float).reshape(2)
        distance = float(np.linalg.norm(point - self.source))
        if distance <= self.config.near_radius_m + 1e-9:
            return Observation("near", ideal_detectable=True)
        visible = float((point - self.source) @ self.emission) >= -1e-9
        range_factor = antenna_range_factor(
            point,
            self.source,
            self.emission,
            self.config,
        )
        detectable_radius = self.reception_radius * range_factor
        if distance <= detectable_radius + 1e-9 and visible:
            if self.config.miss_mode == "independent_action":
                miss_value = self.error.unit(point, occurrence=occurrence, salt="miss")
            elif self.config.miss_mode == "persistent_location":
                length = self.config.miss_correlation_m
                if length <= 0:
                    raise ValueError("位置相关漏检的相关尺度必须为正。")
                bucket = np.floor(point / length) * length + length / 2
                miss_value = self.error.unit(bucket, occurrence=0, salt="persistent_miss")
            elif self.config.miss_mode == "correlated_shadowing":
                miss_value = self.shadowing.unit(point)
            else:
                raise ValueError(f"未知漏检模式：{self.config.miss_mode}")
            if miss_value < self.config.miss_probability:
                return Observation(
                    "no_signal",
                    ideal_detectable=True,
                    false_negative=True,
                )
            true_bearing = math.degrees(
                math.atan2(self.source[1] - point[1], self.source[0] - point[0])
            )
            return Observation(
                "direction",
                (true_bearing + self.error.at(point)) % 360,
                ideal_detectable=True,
            )
        return Observation("no_signal")

    def clear(self, point) -> bool:
        point = np.asarray(point, dtype=float).reshape(2)
        return bool(
            np.linalg.norm(point - self.source) <= self.config.clear_radius_m + 1e-9
        )


class FeasiblePositionState:
    """位置外包多边形；方向状态仅由证书定理隐式使用。"""

    def __init__(self, scenario: Scenario, config: ModelConfig):
        self.config = config
        initial = np.asarray(scenario.initial_position, dtype=float)
        domain = regular_outer_polygon(
            np.zeros(2), config.source_domain_radius_m, config.polygon_sides
        )
        wedge = bearing_wedge_triangle(
            initial,
            scenario.initial_bearing_deg,
            config.bearing_error_deg,
            config.max_reception_m,
        )
        self.polygon = intersect_convex(domain, wedge)
        self.positive_observations: list[tuple[np.ndarray, float]] = [
            (initial.copy(), float(scenario.initial_bearing_deg))
        ]

    def update(self, point, observation: Observation) -> None:
        point = np.asarray(point, dtype=float).reshape(2)
        if observation.kind == "direction":
            wedge = bearing_wedge_triangle(
                point,
                float(observation.bearing_deg),
                self.config.bearing_error_deg,
                self.config.max_reception_m,
            )
            self.polygon = intersect_convex(self.polygon, wedge)
            self.positive_observations.append((point.copy(), float(observation.bearing_deg)))
        elif observation.kind == "near":
            near = regular_outer_polygon(
                point, self.config.near_radius_m, self.config.polygon_sides
            )
            self.polygon = intersect_convex(self.polygon, near)
        elif observation.kind != "no_signal":
            raise ValueError(f"未知观测类型：{observation.kind}")
        # 在含漏检的实验中，错误硬删除可能使后续正观测与当前集合矛盾。
        # 空集由评估器记为方法失效，不在此处抛出异常。

    def contains(self, source) -> bool:
        return point_in_convex_polygon(source, self.polygon, eps=2e-6)

    def cover(self) -> tuple[np.ndarray, float]:
        return minimum_enclosing_circle(self.polygon)


class Mission:
    def __init__(
        self,
        scenario: Scenario,
        policy: str,
        model: ModelConfig,
        costs: CostConfig,
    ):
        self.scenario = scenario
        self.model = model
        self.costs = costs
        self.sensor = IdealDirectionalSensor(scenario, model)
        self.state = FeasiblePositionState(scenario, model)
        self.position = np.asarray(scenario.initial_position, dtype=float)
        self.source = np.asarray(scenario.source, dtype=float)
        self.metrics = RunMetrics(policy=policy, seed=scenario.seed)
        self.measured_points: list[np.ndarray] = [self.position.copy()]
        self.measure_occurrences: dict[tuple[float, float], int] = {}
        self.false_negative_cells: set[tuple[int, int]] = set()
        self.false_negative_sites: set[tuple] = set()
        if not self.state.contains(self.source):
            raise RuntimeError("初始位置外包区域未包含真实源。")

    def move_cost(self, point) -> None:
        point = np.asarray(point, dtype=float).reshape(2)
        distance = float(np.linalg.norm(point - self.position))
        self.metrics.distance_m += distance
        self.metrics.time_s += distance / self.costs.speed_mps
        self.position = point.copy()

    def measure(self, point) -> Observation:
        self.move_cost(point)
        key = (round(float(self.position[0]), 6), round(float(self.position[1]), 6))
        occurrence = self.measure_occurrences.get(key, 0)
        self.measure_occurrences[key] = occurrence + 1
        observation = self.sensor.measure(point, occurrence=occurrence)
        self.metrics.time_s += self.costs.measure_s
        self.metrics.measures += 1
        if observation.false_negative:
            self.metrics.false_negative_measurements += 1
            if self.model.miss_mode == "persistent_location":
                self.false_negative_cells.add(self.persistent_cell(point))
                self.metrics.distinct_false_negative_cells = len(
                    self.false_negative_cells
                )
            if self.model.miss_mode in (
                "persistent_location",
                "correlated_shadowing",
            ):
                self.false_negative_sites.add(self.failure_site_key(point))
                self.metrics.distinct_false_negative_sites = len(
                    self.false_negative_sites
                )
        self.measured_points.append(self.position.copy())
        self.state.update(point, observation)
        if not self.state.contains(self.source):
            self.metrics.unsafe_exclusion = True
        self.metrics.events.append(
            {
                "action": "measure",
                "point": self.position.tolist(),
                "result": observation.kind,
                "bearing_deg": observation.bearing_deg,
                "ideal_detectable": observation.ideal_detectable,
                "false_negative": observation.false_negative,
                "time_s": self.metrics.time_s,
            }
        )
        return observation

    def persistent_cell(self, point) -> tuple[int, int]:
        """返回位置相关漏检模型中的空间单元编号。"""
        length = self.model.miss_correlation_m
        if length <= 0:
            raise ValueError("位置相关漏检的相关尺度必须为正。")
        index = np.floor(np.asarray(point, dtype=float) / length).astype(int)
        return int(index[0]), int(index[1])

    def failure_site_key(self, point) -> tuple:
        """返回错误预算核验使用的持久失效位置编号。"""
        if self.model.miss_mode == "persistent_location":
            return ("cell",) + self.persistent_cell(point)
        if self.model.miss_mode == "correlated_shadowing":
            point = np.asarray(point, dtype=float).reshape(2)
            return ("point", round(float(point[0]), 6), round(float(point[1]), 6))
        raise ValueError("当前漏检模式没有持久失效位置编号。")

    @property
    def invalid_state(self) -> bool:
        return len(self.state.polygon) == 0 or self.metrics.unsafe_exclusion

    def apply_pair_cut(self, near_polygon) -> None:
        self.state.polygon = np.asarray(near_polygon, dtype=float).copy()
        self.metrics.pair_cuts += 1
        if not self.state.contains(self.source):
            self.metrics.unsafe_exclusion = True
        self.metrics.events.append(
            {
                "action": "certified_far_exclusion",
                "vertices_after": len(self.state.polygon),
                "time_s": self.metrics.time_s,
            }
        )

    def clear(self, point, fallback: bool = False) -> bool:
        self.move_cost(point)
        success = self.sensor.clear(point)
        self.metrics.time_s += (
            self.costs.clear_success_s if success else self.costs.clear_failure_s
        )
        if success:
            self.metrics.clear_successes += 1
            self.metrics.completed = True
        else:
            self.metrics.clear_failures += 1
        if fallback:
            self.metrics.fallback_clear_attempts += 1
        self.metrics.events.append(
            {
                "action": "clear",
                "point": self.position.tolist(),
                "success": success,
                "fallback": fallback,
                "time_s": self.metrics.time_s,
            }
        )
        return success

    def try_terminal_clear(self) -> bool:
        center, radius = self.state.cover()
        self.metrics.terminal_radius_m = radius
        if radius <= self.model.terminal_radius_m + 1e-9:
            success = self.clear(center)
            if not success:
                self.metrics.unsafe_exclusion = True
            return True
        return False

    def fallback(self) -> None:
        self.metrics.fallback_used = True
        polygon = self.state.polygon
        low, high = np.min(polygon, axis=0), np.max(polygon, axis=0)
        size = self.model.fallback_cell_m
        points = []
        x_values = np.arange(math.floor(low[0] / size) * size, high[0] + size, size)
        y_values = np.arange(math.floor(low[1] / size) * size, high[1] + size, size)
        for ix, x in enumerate(x_values):
            ordered_y = y_values if ix % 2 == 0 else y_values[::-1]
            for y in ordered_y:
                point = np.array([x, y], dtype=float)
                if cell_intersects_polygon(point, size, polygon):
                    points.append(point)
        if points and np.linalg.norm(points[-1] - self.position) < np.linalg.norm(
            points[0] - self.position
        ):
            points.reverse()
        for point in points:
            if self.clear(point, fallback=True):
                return
        self.metrics.events.append(
            {
                "action": "fallback_exhausted",
                "time_s": self.metrics.time_s,
            }
        )


def generate_scenario(seed: int, config: ModelConfig) -> Scenario:
    """生成原点能够成功得到首次方向观测的随机场景。"""
    rng = np.random.default_rng(seed)
    initial = np.zeros(2)
    for _ in range(100000):
        radius = config.max_reception_m * math.sqrt(float(rng.random()))
        angle = float(rng.uniform(0, 2 * math.pi))
        source = radius * np.array([math.cos(angle), math.sin(angle)])
        reception = float(rng.uniform(config.min_reception_m, config.max_reception_m))
        beta = float(rng.uniform(0, 2 * math.pi))
        emission = np.array([math.cos(beta), math.sin(beta)])
        if radius <= config.near_radius_m or radius > reception:
            continue
        if float((initial - source) @ emission) < 0:
            continue
        if (
            radius
            > reception
            * antenna_range_factor(initial, source, emission, config)
            + 1e-9
        ):
            continue
        field = SpatialErrorField(seed, config.bearing_error_deg)
        true_bearing = math.degrees(math.atan2(source[1], source[0]))
        measured = (true_bearing + field.at(initial)) % 360
        return Scenario(
            seed=seed,
            source=(float(source[0]), float(source[1])),
            emission_direction=(float(emission[0]), float(emission[1])),
            reception_radius_m=reception,
            initial_position=(0.0, 0.0),
            initial_bearing_deg=measured,
        )
    raise RuntimeError("未能生成满足首次成功观测条件的场景。")
