"""Boids 群体行为仿真包。

基于 Craig Reynolds 的 Boids 模型，使用 numpy 向量化计算规则，
并用 turtle 实时渲染群体涌现行为。

模块划分：
- params      : 参数配置
- entities    : 数据实体（Boid / Obstacle / Predator）
- math_utils  : 向量与转向力工具
- flock       : 群体仿真核心
- visualizer  : turtle 可视化与交互
"""

from .entities import Boid, Obstacle, Predator
from .flock import Flock
from .params import Params
from .visualizer import Visualizer, main

__all__ = [
    "Boid",
    "Flock",
    "Obstacle",
    "Params",
    "Predator",
    "Visualizer",
    "main",
]
