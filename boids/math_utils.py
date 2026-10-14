"""向量与转向力计算工具。

供 :mod:`boids.flock` 使用，包含环形坐标环绕、向量长度限制，
以及「逃离一组点」和「避让障碍物」两种通用转向力。
"""

from __future__ import annotations

import numpy as np

from .params import Params


def wrap_coord(value: np.ndarray, size: float) -> np.ndarray:
    """把坐标环绕到 [-size/2, size/2] 区间（环形边界）。"""
    return (value + size / 2.0) % size - size / 2.0


def limit_magnitude(vector: np.ndarray, max_norm: float) -> np.ndarray:
    """把向量长度限制在 max_norm 以内（按最后一维缩放）。"""
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norm, 1e-9))
    return vector * scale


def away_steering(
    position: np.ndarray,
    velocity: np.ndarray,
    points: np.ndarray,
    radius: float,
    weight: float,
    speed: float,
    params: Params,
) -> np.ndarray:
    """计算逃离一组点（半径 radius 内）的转向力，距离越近权重越大。"""
    if points.size == 0:
        return np.zeros(2)
    d_vec = points - position                 # (K,2)
    d_vec[:, 0] = wrap_coord(d_vec[:, 0], params.width)
    d_vec[:, 1] = wrap_coord(d_vec[:, 1], params.height)
    d = np.linalg.norm(d_vec, axis=1)         # (K,)
    close = d < radius
    if not close.any():
        return np.zeros(2)
    d = d[close]
    away = -d_vec[close]
    unit = away / np.maximum(d[:, None], 1e-9)
    w = 1.0 / np.maximum(d, 1e-6)
    direction = (unit * w[:, None]).sum(axis=0)
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    return weight * (direction * speed - velocity)


def obstacle_steering(
    position: np.ndarray,
    velocity: np.ndarray,
    obstacles: list,
    params: Params,
    speed: float,
) -> np.ndarray:
    """计算避让障碍物的转向力（进入「半径 + 余量」内才生效）。"""
    steer = np.zeros(2)
    for ob in obstacles:
        d_vec = ob.position - position
        d_vec[0] = wrap_coord(d_vec[0], params.width)
        d_vec[1] = wrap_coord(d_vec[1], params.height)
        d = np.linalg.norm(d_vec)
        avoid_r = ob.radius + params.obstacle_margin
        if 1e-9 < d < avoid_r:
            away = -d_vec / d
            strength = (avoid_r - d) / avoid_r   # 越近越强
            steer += params.w_obstacle * (away * speed - velocity) * strength
    return steer
