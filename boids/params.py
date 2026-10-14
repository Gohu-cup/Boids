"""仿真参数配置。

所有可调参数集中在 :class:`Params` 中，便于调优。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Params:
    """Boids 仿真全部可调参数。"""

    # 基础
    num_boids: int = 50              # Boid 数量
    max_speed: float = 4.0           # 最大速度（像素/帧）
    max_force: float = 0.2           # 最大转向力（加速度上限）
    perception_radius: float = 50.0  # 对齐 / 聚合感知半径
    separation_radius: float = 20.0  # 分离最小距离
    w_separation: float = 1.8        # 分离权重
    w_alignment: float = 1.0         # 对齐权重
    w_cohesion: float = 1.0          # 聚合权重
    width: int = 900                 # 窗口宽度（像素）
    height: int = 700                # 窗口高度（像素）
    bg_color: str = "black"          # 背景色
    boid_color: str = "white"        # Boid 颜色（单种群时）
    frame_ms: int = 16               # 帧间隔（毫秒，约 60 FPS）

    # 捕食者
    num_predators: int = 2           # 捕食者数量
    predator_speed: float = 4.5      # 捕食者最大速度（略快于 Boid）
    fear_radius: float = 80.0        # Boid 逃离捕食者的恐惧半径
    catch_radius: float = 10.0       # 捕食判定距离
    w_flee: float = 2.5              # 逃离捕食者权重
    predator_color: str = "red"      # 捕食者颜色

    # 障碍物
    obstacle_radius: float = 30.0    # 障碍物半径
    obstacle_margin: float = 40.0    # 避障感知余量（半径外的缓冲带）
    w_obstacle: float = 2.0          # 避障权重
    obstacle_color: str = "gray"     # 障碍物描边颜色
    preset_obstacles: tuple = (
        (-200.0, -100.0), (150.0, 120.0), (0.0, 180.0),
    )                                # 预设障碍物位置

    # 多种群
    num_species: int = 3             # 种群数量
    species_colors: tuple = ("white", "cyan", "orange")  # 各种群颜色

    # 鼠标驱赶
    repel_radius: float = 80.0       # 驱赶半径
    w_repel: float = 2.5             # 驱赶权重
