"""群体仿真核心。

:class:`Flock` 只负责 Boid 群体：持有位置/速度/种群数组，
向量化计算分离、对齐、聚合三条规则，并叠加避障、逃离捕食者、
鼠标驱赶三种外力。捕食者自身的追捕行为见 :mod:`boids.predator`。
"""

from __future__ import annotations

import math

import numpy as np

from .entities import Obstacle
from .params import Params
from .steering import away_steering, obstacle_steering
from .vecmath import EPS, EPS_DIST, limit_magnitude, wrap_coord


class Flock:
    """群体：持有全体 Boid 的位置/速度/种群数组，向量化计算规则。"""

    def __init__(self, params: Params, rng: np.random.Generator):
        self.params = params
        self.rng = rng
        self.positions, self.velocities = self._random_state()
        self.species = np.arange(params.num_boids) % params.num_species

    # -- 初始化 ------------------------------------------------------------
    def _random_state(self):
        """生成随机初始位置与速度（位置在窗口内，速度方向随机）。"""
        p = self.params
        half_w, half_h = p.width / 2.0, p.height / 2.0
        positions = self.rng.uniform(
            [-half_w, -half_h], [half_w, half_h], size=(p.num_boids, 2)
        )
        angles = self.rng.uniform(0.0, 2.0 * math.pi, size=p.num_boids)
        velocities = (p.max_speed / 2.0) * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        )
        return positions, velocities

    def reset(self):
        """重置为随机初始状态（种群划分保持不变）。"""
        self.positions, self.velocities = self._random_state()

    def set_multi_species(self, enabled: bool):
        """开启/关闭多种群：开启时按索引均分，关闭时全部归为单种群。"""
        if enabled:
            self.species = np.arange(self.params.num_boids) % self.params.num_species
        else:
            self.species = np.zeros(self.params.num_boids, dtype=int)

    def respawn(self, indices):
        """把指定索引的 Boid 重置到随机位置与速度（种群不变）。"""
        idx = np.atleast_1d(indices)
        if idx.size == 0:
            return
        p = self.params
        half_w, half_h = p.width / 2.0, p.height / 2.0
        self.positions[idx] = self.rng.uniform(
            [-half_w, -half_h], [half_w, half_h], size=(idx.size, 2)
        )
        angles = self.rng.uniform(0.0, 2.0 * math.pi, size=idx.size)
        self.velocities[idx] = (p.max_speed / 2.0) * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        )

    # -- 规则计算 ----------------------------------------------------------
    def step(
        self,
        obstacles: list[Obstacle],
        predator_positions: np.ndarray | None,
        mouse_pos: np.ndarray | None,
    ) -> None:
        """推进 Boid 一帧：计算合力 → 更新速度/位置。

        - ``obstacles``：空列表表示无障碍；
        - ``predator_positions``：None 表示无捕食者；
        - ``mouse_pos``：None 表示无鼠标驱赶。
        """
        p = self.params
        pos = self.positions
        vel = self.velocities
        n = p.num_boids
        same = self.species[:, None] == self.species[None, :]   # (N,N)

        # 环形位移矩阵 D[i,j] = wrap(pos_j - pos_i)，形状 (N,N,2)
        diff = pos[None, :, :] - pos[:, None, :]   # (N,N,2)
        diff[:, :, 0] = wrap_coord(diff[:, :, 0], p.width)
        diff[:, :, 1] = wrap_coord(diff[:, :, 1], p.height)
        dist = np.linalg.norm(diff, axis=2)        # (N,N) 环形距离
        np.fill_diagonal(dist, np.inf)             # 排除自身

        flock_neighbor = same & (dist < p.perception_radius)  # 对齐/聚合（同群）
        nearby = dist < p.separation_radius                   # 分离（全体）

        steering = np.zeros((n, 2))
        for i in range(n):
            acc = np.zeros(2)

            # 分离：远离过近的邻居（全体，不分群），距离越近权重越大
            if nearby[i].any():
                d = dist[i, nearby[i]]
                away = -diff[i, nearby[i]]
                unit = away / np.maximum(d[:, None], EPS)
                w = 1.0 / np.maximum(d, EPS_DIST)
                direction = (unit * w[:, None]).sum(axis=0)
                direction = direction / (np.linalg.norm(direction) + EPS)
                acc += p.w_separation * (direction * p.max_speed - vel[i])

            # 对齐 / 聚合：仅同种群
            if flock_neighbor[i].any():
                avg_vel = vel[flock_neighbor[i]].mean(axis=0)
                sp = np.linalg.norm(avg_vel)
                if sp > EPS:
                    acc += p.w_alignment * (avg_vel / sp * p.max_speed - vel[i])

                center = diff[i, flock_neighbor[i]].mean(axis=0)
                dc = np.linalg.norm(center)
                if dc > EPS:
                    acc += p.w_cohesion * (center / dc * p.max_speed - vel[i])

            # 逃离捕食者
            if predator_positions is not None and predator_positions.size:
                acc += away_steering(
                    pos[i], vel[i], predator_positions,
                    p.fear_radius, p.w_flee, p.max_speed, p,
                )

            # 避障
            if obstacles:
                acc += obstacle_steering(
                    pos[i], vel[i], obstacles, p, p.max_speed
                )

            # 鼠标驱赶（悬停常开）
            if mouse_pos is not None:
                acc += away_steering(
                    pos[i], vel[i], mouse_pos[None, :],
                    p.repel_radius, p.w_repel, p.max_speed, p,
                )

            steering[i] = acc

        # 转向力截断，随后更新速度并限制最大速度
        steering = limit_magnitude(steering, p.max_force)
        vel = limit_magnitude(vel + steering, p.max_speed)

        # 更新位置并环绕边界
        pos = pos + vel
        pos[:, 0] = wrap_coord(pos[:, 0], p.width)
        pos[:, 1] = wrap_coord(pos[:, 1], p.height)

        self.velocities = vel
        self.positions = pos
