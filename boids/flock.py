"""群体仿真核心。

:class:`Flock` 持有全体 Boid 的位置/速度/种群数组，向量化计算
分离、对齐、聚合三条规则，并处理避障、逃离捕食者、鼠标驱赶，
以及捕食者追捕与捕食判定。
"""

from __future__ import annotations

import math

import numpy as np

from .entities import Boid, Predator
from .math_utils import away_steering, limit_magnitude, obstacle_steering, wrap_coord
from .params import Params


class Flock:
    """群体：持有全体 Boid 的位置/速度/种群数组，向量化计算规则。"""

    def __init__(self, params: Params, rng: np.random.Generator):
        self.params = params
        self.rng = rng
        self.positions, self.velocities = self._random_state()
        self.species = np.arange(params.num_boids) % params.num_species

    # -- 初始化 ------------------------------------------------------------
    def _random_state(self):
        """生成随机初始位置与速度。"""
        p = self.params
        half_w, half_h = p.width / 2.0, p.height / 2.0
        boids = []
        for _ in range(p.num_boids):
            # 位置：窗口内随机
            position = self.rng.uniform([-half_w, -half_h], [half_w, half_h])
            # 速度：随机方向，大小为最大速度的一半
            angle = self.rng.uniform(0.0, 2.0 * math.pi)
            velocity = (p.max_speed / 2.0) * np.array(
                [math.cos(angle), math.sin(angle)]
            )
            boids.append(Boid(position=position, velocity=velocity))
        positions = np.array([b.position for b in boids], dtype=float)
        velocities = np.array([b.velocity for b in boids], dtype=float)
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

    def _respawn(self, i: int):
        """把被吃掉的 Boid 重置到随机位置与速度（种群不变）。"""
        p = self.params
        half_w, half_h = p.width / 2.0, p.height / 2.0
        self.positions[i] = self.rng.uniform([-half_w, -half_h], [half_w, half_h])
        angle = self.rng.uniform(0.0, 2.0 * math.pi)
        self.velocities[i] = (p.max_speed / 2.0) * np.array(
            [math.cos(angle), math.sin(angle)]
        )

    # -- 捕食者更新 --------------------------------------------------------
    def _update_predators(
        self, predators: Predator, obstacles: list, obstacles_enabled: bool
    ) -> int:
        """更新捕食者：追最近 Boid、彼此分离、避障、捕食判定。返回捕食数。"""
        p = self.params
        k = predators.positions.shape[0]
        eaten = 0
        for j in range(k):
            acc = np.zeros(2)

            # 追最近的 Boid
            to_boids = self.positions - predators.positions[j]
            to_boids[:, 0] = wrap_coord(to_boids[:, 0], p.width)
            to_boids[:, 1] = wrap_coord(to_boids[:, 1], p.height)
            bd = np.linalg.norm(to_boids, axis=1)
            if bd.size:
                i_min = int(np.argmin(bd))
                if bd[i_min] > 1e-9:
                    desired = to_boids[i_min] / bd[i_min] * p.predator_speed
                    acc += desired - predators.velocities[j]

            # 捕食者之间分离
            if k > 1:
                other = np.ones(k, dtype=bool)
                other[j] = False
                acc += away_steering(
                    predators.positions[j],
                    predators.velocities[j],
                    predators.positions[other],
                    p.separation_radius,
                    p.w_separation,
                    p.predator_speed,
                    p,
                )

            # 避障
            if obstacles_enabled and obstacles:
                acc += obstacle_steering(
                    predators.positions[j],
                    predators.velocities[j],
                    obstacles,
                    p,
                    p.predator_speed,
                )

            # 更新速度与位置
            predators.velocities[j] = limit_magnitude(
                predators.velocities[j] + limit_magnitude(acc, p.max_force),
                p.predator_speed,
            )
            predators.positions[j] = (
                predators.positions[j] + predators.velocities[j]
            )
            predators.positions[j, 0] = wrap_coord(
                predators.positions[j, 0], p.width
            )
            predators.positions[j, 1] = wrap_coord(
                predators.positions[j, 1], p.height
            )

            # 捕食判定：追上即吃掉（重生）
            to_boids = self.positions - predators.positions[j]
            to_boids[:, 0] = wrap_coord(to_boids[:, 0], p.width)
            to_boids[:, 1] = wrap_coord(to_boids[:, 1], p.height)
            bd = np.linalg.norm(to_boids, axis=1)
            if bd.size and bd.min() < p.catch_radius:
                self._respawn(int(np.argmin(bd)))
                eaten += 1
        return eaten

    # -- 规则计算 ----------------------------------------------------------
    def step(
        self,
        obstacles: list,
        predators: Predator,
        mouse_pos: np.ndarray | None,
        obstacles_enabled: bool = True,
        predators_enabled: bool = True,
    ) -> int:
        """推进一帧：计算合力 → 更新速度/位置 → 更新捕食者。返回本帧捕食数。"""
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
                unit = away / np.maximum(d[:, None], 1e-9)
                w = 1.0 / np.maximum(d, 1e-6)
                direction = (unit * w[:, None]).sum(axis=0)
                direction = direction / (np.linalg.norm(direction) + 1e-9)
                acc += p.w_separation * (direction * p.max_speed - vel[i])

            # 对齐 / 聚合：仅同种群
            if flock_neighbor[i].any():
                avg_vel = vel[flock_neighbor[i]].mean(axis=0)
                sp = np.linalg.norm(avg_vel)
                if sp > 1e-9:
                    acc += p.w_alignment * (avg_vel / sp * p.max_speed - vel[i])

                center = diff[i, flock_neighbor[i]].mean(axis=0)
                dc = np.linalg.norm(center)
                if dc > 1e-9:
                    acc += p.w_cohesion * (center / dc * p.max_speed - vel[i])

            # 逃离捕食者
            if predators_enabled and predators.positions.size:
                acc += away_steering(
                    pos[i], vel[i], predators.positions,
                    p.fear_radius, p.w_flee, p.max_speed, p,
                )

            # 避障
            if obstacles_enabled and obstacles:
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

        # 捕食者更新
        if predators_enabled and predators.positions.size:
            return self._update_predators(predators, obstacles, obstacles_enabled)
        return 0
