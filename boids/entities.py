"""仿真中的数据实体：Boid、障碍物、捕食者。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .params import Params


@dataclass
class Boid:
    """单个 Boid 的初始状态（numpy 一维向量）。

    注意：进入 Flock 后，全体状态统一存放在 (N,2) 数组中以便向量化计算。
    """

    position: np.ndarray  # (2,) x, y
    velocity: np.ndarray  # (2,) vx, vy


@dataclass
class Obstacle:
    """圆形障碍物：位置 + 半径。"""

    position: np.ndarray  # (2,) x, y
    radius: float


class Predator:
    """捕食者群体：位置/速度数组，负责自身初始化。"""

    def __init__(self, params: Params, rng: np.random.Generator):
        self.params = params
        self.rng = rng
        self.positions = np.zeros((params.num_predators, 2))
        self.velocities = np.zeros((params.num_predators, 2))
        self.reset()

    def reset(self):
        """重置到随机位置与随机方向。"""
        p = self.params
        half_w, half_h = p.width / 2.0, p.height / 2.0
        self.positions = self.rng.uniform(
            [-half_w, -half_h], [half_w, half_h], size=(p.num_predators, 2)
        )
        angles = self.rng.uniform(0.0, 2.0 * math.pi, size=p.num_predators)
        self.velocities = (
            np.stack([np.cos(angles), np.sin(angles)], axis=1)
            * (p.predator_speed / 2.0)
        )
