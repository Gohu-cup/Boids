"""仿真门面。

:class:`Simulation` 把群体、捕食者、障碍物与交互开关收敛到一个接口，
供渲染层（:mod:`boids.visualizer`）与测试通过同一入口驱动仿真，
而不必了解内部对象的协作细节。
"""

from __future__ import annotations

import numpy as np

from .entities import Obstacle
from .flock import Flock
from .params import Params
from .predator import Predator


class Simulation:
    """世界门面：统一管理群体、捕食者、障碍物与开关状态。"""

    def __init__(
        self,
        params: Params | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.params = params if params is not None else Params()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.flock = Flock(self.params, self.rng)
        self.predators = Predator(self.params, self.rng)
        self.obstacles: list[Obstacle] = [
            Obstacle(np.array(pos, dtype=float), self.params.obstacle_radius)
            for pos in self.params.preset_obstacles
        ]
        self.obstacles_enabled = True
        self.predators_enabled = False
        self.multi_species = True
        self.eaten_total = 0

    # -- 只读状态（供渲染层读取） ------------------------------------------
    @property
    def boid_positions(self) -> np.ndarray:
        return self.flock.positions

    @property
    def boid_velocities(self) -> np.ndarray:
        return self.flock.velocities

    @property
    def boid_species(self) -> np.ndarray:
        return self.flock.species

    @property
    def predator_positions(self) -> np.ndarray:
        return self.predators.positions

    @property
    def predator_velocities(self) -> np.ndarray:
        return self.predators.velocities

    # -- 指令 --------------------------------------------------------------
    def step(self, mouse_pos: np.ndarray | None = None) -> int:
        """推进一帧，返回本帧捕食数。"""
        obstacles = self.obstacles if self.obstacles_enabled else []
        predator_positions = (
            self.predators.positions if self.predators_enabled else None
        )

        self.flock.step(obstacles, predator_positions, mouse_pos)

        eaten = 0
        if self.predators_enabled:
            eaten = self.predators.step(self.flock, obstacles)
        self.eaten_total += eaten
        return eaten

    def reset(self):
        """重置群体与捕食者，清空捕食计数。"""
        self.flock.reset()
        self.predators.reset()
        self.eaten_total = 0

    def change_speed(self, delta: float):
        self.params.max_speed = max(0.5, self.params.max_speed + delta)

    def toggle_obstacles(self):
        self.obstacles_enabled = not self.obstacles_enabled

    def toggle_predators(self):
        self.predators_enabled = not self.predators_enabled

    def toggle_multi_species(self):
        self.multi_species = not self.multi_species
        self.flock.set_multi_species(self.multi_species)

    def add_obstacle(self, x: float, y: float):
        self.obstacles.append(
            Obstacle(np.array([x, y], dtype=float), self.params.obstacle_radius)
        )

    def remove_obstacle(self, x: float, y: float) -> int | None:
        """删除距离 (x,y) 最近的障碍物，返回其索引；无则返回 None。"""
        if not self.obstacles:
            return None
        pts = np.array([o.position for o in self.obstacles])
        d = np.linalg.norm(pts - np.array([x, y]), axis=1)
        i = int(np.argmin(d))
        del self.obstacles[i]
        return i
