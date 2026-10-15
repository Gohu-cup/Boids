"""纯向量几何工具。

仅依赖 numpy，不含任何仿真领域知识，可独立复用与测试。
"""

from __future__ import annotations

import numpy as np

# 向量归一化保护量（最小可接受长度，避免除零）
EPS = 1e-9
# 距离加权保护量（最小可接受距离）
EPS_DIST = 1e-6


def wrap_coord(value: np.ndarray, size: float) -> np.ndarray:
    """把坐标环绕到 [-size/2, size/2] 区间（环形边界）。"""
    return (value + size / 2.0) % size - size / 2.0


def limit_magnitude(vector: np.ndarray, max_norm: float) -> np.ndarray:
    """把向量长度限制在 max_norm 以内（按最后一维缩放）。"""
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norm, EPS))
    return vector * scale
