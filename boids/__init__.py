"""Boids 群体行为仿真包。

基于 Craig Reynolds 的 Boids 模型，使用 numpy 向量化计算规则，
并用 turtle 实时渲染群体涌现行为。

模块划分：
- params      : 参数配置
- entities    : 纯数据实体（Obstacle）
- vecmath     : 纯向量几何工具
- steering    : Reynolds 转向力
- flock       : Boid 群体规则
- predator    : 捕食者追捕行为
- simulation  : 世界门面（统一协调与开关状态）
- visualizer  : turtle 渲染与交互
"""

from .entities import Obstacle
from .flock import Flock
from .params import Params
from .predator import Predator
from .simulation import Simulation
from .visualizer import Visualizer, main

__all__ = [
    "Flock",
    "Obstacle",
    "Params",
    "Predator",
    "Simulation",
    "Visualizer",
    "main",
]
