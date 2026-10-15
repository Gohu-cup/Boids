"""捕食者：追捕、避障与捕食行为。

:class:`Predator` 同时持有自身状态（位置/速度数组）与行为逻辑，
通过 ``step(flock, obstacles)`` 自主追捕最近 Boid，并与 :class:`~boids.flock.Flock`
通过 ``respawn`` 协作完成「吃掉即重生」。
"""

from __future__ import annotations

import math

import numpy as np

from .flock import Flock
from .params import Params
from .steering import away_steering, obstacle_steering
from .vecmath import EPS, limit_magnitude, wrap_coord


class Predator:
    """捕食者群体：位置/速度数组，拥有追捕与捕食行为。"""

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

    def step(self, flock: Flock, obstacles: list) -> int:
        """追捕最近 Boid、彼此分离、避障并捕食。返回本帧捕食数。"""
        p = self.params
        k = self.positions.shape[0]
        eaten = 0
        for j in range(k):
            acc = np.zeros(2)

            # 追最近的 Boid
            to_boids = flock.positions - self.positions[j]
            to_boids[:, 0] = wrap_coord(to_boids[:, 0], p.width)
            to_boids[:, 1] = wrap_coord(to_boids[:, 1], p.height)
            bd = np.linalg.norm(to_boids, axis=1)
            if bd.size:
                i_min = int(np.argmin(bd))
                if bd[i_min] > EPS:
                    desired = to_boids[i_min] / bd[i_min] * p.predator_speed
                    acc += desired - self.velocities[j]

            # 捕食者之间分离
            if k > 1:
                other = np.ones(k, dtype=bool)
                other[j] = False
                acc += away_steering(
                    self.positions[j],
                    self.velocities[j],
                    self.positions[other],
                    p.separation_radius,
                    p.w_separation,
                    p.predator_speed,
                    p,
                )

            # 避障
            if obstacles:
                acc += obstacle_steering(
                    self.positions[j],
                    self.velocities[j],
                    obstacles,
                    p,
                    p.predator_speed,
                )

            # 更新速度与位置
            self.velocities[j] = limit_magnitude(
                self.velocities[j] + limit_magnitude(acc, p.max_force),
                p.predator_speed,
            )
            self.positions[j] = self.positions[j] + self.velocities[j]
            self.positions[j, 0] = wrap_coord(self.positions[j, 0], p.width)
            self.positions[j, 1] = wrap_coord(self.positions[j, 1], p.height)

            # 捕食判定：追上即吃掉（重生）
            to_boids = flock.positions - self.positions[j]
            to_boids[:, 0] = wrap_coord(to_boids[:, 0], p.width)
            to_boids[:, 1] = wrap_coord(to_boids[:, 1], p.height)
            bd = np.linalg.norm(to_boids, axis=1)
            if bd.size and bd.min() < p.catch_radius:
                flock.respawn(int(np.argmin(bd)))
                eaten += 1
        return eaten
