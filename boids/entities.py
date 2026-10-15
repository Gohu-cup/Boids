"""仿真中的纯数据实体。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Obstacle:
    """圆形障碍物：位置 + 半径。"""

    position: np.ndarray  # (2,) x, y
    radius: float
