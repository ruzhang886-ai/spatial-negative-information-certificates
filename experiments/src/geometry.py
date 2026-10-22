"""连续凸多边形几何工具。"""

from __future__ import annotations

import numpy as np


EPS = 1e-10


def as_polygon(vertices) -> np.ndarray:
    """返回形状为 (n, 2) 的浮点顶点数组。"""
    polygon = np.asarray(vertices, dtype=float)
    if polygon.size == 0:
        return np.empty((0, 2), dtype=float)
    polygon = polygon.reshape(-1, 2)
    if not np.all(np.isfinite(polygon)):
        raise ValueError("多边形顶点必须为有限数值。")
    return polygon


def clip_halfplane(vertices, normal, limit, eps: float = EPS) -> np.ndarray:
    """用闭半平面 normal·x <= limit 裁剪凸多边形。"""
    polygon = as_polygon(vertices)
    if len(polygon) == 0:
        return polygon

    normal = np.asarray(normal, dtype=float).reshape(2)
    if not np.all(np.isfinite(normal)) or np.linalg.norm(normal) <= eps:
        raise ValueError("半平面法向量必须非零且有限。")
    if not np.isfinite(limit):
        raise ValueError("半平面界限必须为有限数值。")

    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_value = float(normal @ previous - limit)
    previous_inside = previous_value <= eps

    for current in polygon:
        current_value = float(normal @ current - limit)
        current_inside = current_value <= eps

        if current_inside != previous_inside:
            direction = current - previous
            denominator = float(normal @ direction)
            if abs(denominator) > eps:
                ratio = (limit - float(normal @ previous)) / denominator
                output.append(previous + ratio * direction)

        if current_inside:
            output.append(current.copy())

        previous = current
        previous_inside = current_inside

    if not output:
        return np.empty((0, 2), dtype=float)

    cleaned: list[np.ndarray] = []
    for point in output:
        if not cleaned or np.linalg.norm(point - cleaned[-1]) > eps:
            cleaned.append(point)
    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= eps:
        cleaned.pop()
    return np.asarray(cleaned, dtype=float).reshape(-1, 2)


def max_distance_to_polygon_vertices(point, vertices) -> float:
    """返回点到凸多边形的最大距离；凸函数最大值由顶点取得。"""
    polygon = as_polygon(vertices)
    if len(polygon) == 0:
        raise ValueError("不能计算空多边形的最大距离。")
    point = np.asarray(point, dtype=float).reshape(2)
    return float(np.max(np.linalg.norm(polygon - point, axis=1)))


def signed_area(vertices) -> float:
    """返回多边形有向面积；正值表示逆时针。"""
    polygon = as_polygon(vertices)
    if len(polygon) < 3:
        return 0.0
    shifted = np.roll(polygon, -1, axis=0)
    return 0.5 * float(
        np.sum(polygon[:, 0] * shifted[:, 1] - polygon[:, 1] * shifted[:, 0])
    )


def polygon_area(vertices) -> float:
    return abs(signed_area(vertices))


def intersect_convex(subject, clipper) -> np.ndarray:
    """返回两个凸多边形的交集。"""
    result = as_polygon(subject)
    boundary = as_polygon(clipper)
    if len(result) == 0 or len(boundary) < 3:
        return np.empty((0, 2), dtype=float)
    ccw = signed_area(boundary) >= 0
    for start, end in zip(boundary, np.roll(boundary, -1, axis=0)):
        edge = end - start
        normal = np.array([edge[1], -edge[0]], dtype=float)
        if not ccw:
            normal = -normal
        result = clip_halfplane(result, normal, float(normal @ start))
        if len(result) == 0:
            break
    return result


def regular_outer_polygon(center, radius: float, sides: int = 128) -> np.ndarray:
    """构造外切于指定圆的正多边形。"""
    if radius <= 0 or sides < 3:
        raise ValueError("半径必须为正，边数至少为3。")
    center = np.asarray(center, dtype=float).reshape(2)
    circumradius = radius / np.cos(np.pi / sides)
    angles = (np.arange(sides) + 0.5) * 2 * np.pi / sides
    return center + circumradius * np.column_stack((np.cos(angles), np.sin(angles)))


def bearing_wedge_triangle(
    origin, bearing_deg: float, half_angle_deg: float, max_range: float
) -> np.ndarray:
    """构造距离不超过 max_range 的测向扇形外包三角形。"""
    if not 0 <= half_angle_deg < 90 or max_range <= 0:
        raise ValueError("测向半角必须位于 [0,90) 度且距离必须为正。")
    origin = np.asarray(origin, dtype=float).reshape(2)
    center = np.deg2rad(float(bearing_deg))
    delta = np.deg2rad(float(half_angle_deg))
    length = max_range / np.cos(delta)
    lower = origin + length * np.array([np.cos(center - delta), np.sin(center - delta)])
    upper = origin + length * np.array([np.cos(center + delta), np.sin(center + delta)])
    return np.asarray([origin, lower, upper], dtype=float)


def point_in_convex_polygon(point, vertices, eps: float = 1e-8) -> bool:
    """检查点是否位于凸多边形内部或边界。"""
    point = np.asarray(point, dtype=float).reshape(2)
    polygon = as_polygon(vertices)
    if len(polygon) == 0:
        return False
    if len(polygon) == 1:
        return bool(np.linalg.norm(point - polygon[0]) <= eps)
    if len(polygon) == 2:
        edge = polygon[1] - polygon[0]
        if abs(np.cross(edge, point - polygon[0])) > eps * max(1.0, np.linalg.norm(edge)):
            return False
        return bool(
            float((point - polygon[0]) @ (point - polygon[1])) <= eps
        )
    signs = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        signs.append(float(np.cross(end - start, point - start)))
    return min(signs) >= -eps or max(signs) <= eps


def minimum_enclosing_circle(vertices) -> tuple[np.ndarray, float]:
    """计算有限顶点集的最小包围圆，并以全部顶点复核半径。"""
    points = as_polygon(vertices)
    if len(points) == 0:
        raise ValueError("空点集没有最小包围圆。")
    order = np.random.default_rng(1729).permutation(len(points))
    shuffled = points[order]
    center = shuffled[0].copy()
    radius = 0.0
    for i, a in enumerate(shuffled):
        if np.linalg.norm(a - center) <= radius + 1e-9:
            continue
        center, radius = a.copy(), 0.0
        for j in range(i):
            b = shuffled[j]
            if np.linalg.norm(b - center) <= radius + 1e-9:
                continue
            center = (a + b) / 2
            radius = float(np.linalg.norm(a - b) / 2)
            for k in range(j):
                c = shuffled[k]
                if np.linalg.norm(c - center) <= radius + 1e-9:
                    continue
                u, v = b - a, c - a
                determinant = 2 * float(u[0] * v[1] - u[1] * v[0])
                if abs(determinant) <= 1e-12 * max(
                    1.0, float(np.linalg.norm(u) * np.linalg.norm(v))
                ):
                    pairs = ((a, b), (a, c), (b, c))
                    p, q = max(pairs, key=lambda pair: np.linalg.norm(pair[0] - pair[1]))
                    center = (p + q) / 2
                else:
                    uu, vv = float(u @ u), float(v @ v)
                    center = a + np.array(
                        [v[1] * uu - u[1] * vv, u[0] * vv - v[0] * uu]
                    ) / determinant
                radius = max(
                    float(np.linalg.norm(a - center)),
                    float(np.linalg.norm(b - center)),
                    float(np.linalg.norm(c - center)),
                )
    verified_radius = float(np.max(np.linalg.norm(points - center, axis=1)))
    return center, verified_radius


def cell_intersects_polygon(center, cell_size: float, vertices) -> bool:
    """判断轴对齐正方形网格单元是否与凸多边形相交。"""
    center = np.asarray(center, dtype=float).reshape(2)
    half = cell_size / 2
    clipped = as_polygon(vertices)
    for normal in (
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, -1.0]),
    ):
        clipped = clip_halfplane(clipped, normal, float(normal @ center) + half)
        if len(clipped) == 0:
            return False
    return True
