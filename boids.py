"""Boids 群体行为仿真

基于 Craig Reynolds 的 Boids 模型，使用 numpy 向量化计算三条规则，
并用 turtle 实时渲染群体涌现行为：

- 分离（Separation）：远离太近的邻居，避免碰撞
- 对齐（Alignment）：趋向邻域内平均速度方向，保持队形一致
- 聚合（Cohesion）：趋向邻域质心，保持群体聚集

运行方式：
    python boids.py

交互按键：
    空格       暂停 / 继续
    ↑ / ↓     增大 / 减小最大速度
    R          重置为随机初始状态
"""

from __future__ import annotations

import math
import turtle
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# 参数配置
# ---------------------------------------------------------------------------
@dataclass
class Params:
    """Boids 仿真全部可调参数，集中于此便于调优。"""

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
    boid_color: str = "white"        # Boid 颜色
    frame_ms: int = 16               # 帧间隔（毫秒，约 60 FPS）


# ---------------------------------------------------------------------------
# 单个 Boid 的初始状态
# ---------------------------------------------------------------------------
@dataclass
class Boid:
    """单个 Boid 的初始状态（numpy 一维向量）。

    注意：进入 Flock 后，全体状态统一存放在 (N,2) 数组中以便向量化计算。
    """

    position: np.ndarray  # (2,) x, y
    velocity: np.ndarray  # (2,) vx, vy


# ---------------------------------------------------------------------------
# 向量工具函数
# ---------------------------------------------------------------------------
def _wrap(value: np.ndarray, size: float) -> np.ndarray:
    """把坐标环绕到 [-size/2, size/2] 区间（环形边界）。"""
    return (value + size / 2.0) % size - size / 2.0


def _limit(vector: np.ndarray, max_norm: float) -> np.ndarray:
    """把向量长度限制在 max_norm 以内（按最后一维缩放）。"""
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norm, 1e-9))
    return vector * scale


# ---------------------------------------------------------------------------
# 群体
# ---------------------------------------------------------------------------
class Flock:
    """群体：持有全体 Boid 的位置/速度数组，向量化计算三条规则。"""

    def __init__(self, params: Params, rng: np.random.Generator):
        self.params = params
        self.rng = rng
        self.positions, self.velocities = self._random_state()

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
        """重置为随机初始状态。"""
        self.positions, self.velocities = self._random_state()

    # -- 规则计算 ----------------------------------------------------------
    def step(self):
        """推进一帧：计算三规则合力 → 更新速度 → 更新位置。"""
        p = self.params
        pos = self.positions
        vel = self.velocities
        n = p.num_boids

        # 环形位移矩阵 D[i,j] = wrap(pos_j - pos_i)，形状 (N,N,2)
        diff = pos[None, :, :] - pos[:, None, :]   # (N,N,2)
        diff[:, :, 0] = _wrap(diff[:, :, 0], p.width)
        diff[:, :, 1] = _wrap(diff[:, :, 1], p.height)
        dist = np.linalg.norm(diff, axis=2)        # (N,N) 环形距离
        np.fill_diagonal(dist, np.inf)             # 排除自身

        neighbor = dist < p.perception_radius       # 对齐/聚合邻域
        nearby = dist < p.separation_radius         # 分离邻域

        steering = np.zeros((n, 2))
        for i in range(n):
            acc = np.zeros(2)

            # 分离：远离过近的邻居，距离越近权重越大
            if nearby[i].any():
                d = dist[i, nearby[i]]
                away = -diff[i, nearby[i]]          # 从 i 指向远离邻居的方向
                unit = away / np.maximum(d[:, None], 1e-9)
                weight = 1.0 / np.maximum(d, 1e-6)
                direction = (unit * weight[:, None]).sum(axis=0)
                direction = direction / (np.linalg.norm(direction) + 1e-9)
                desired = direction * p.max_speed
                acc += p.w_separation * (desired - vel[i])

            if neighbor[i].any():
                # 对齐：趋向邻域平均速度方向
                avg_vel = vel[neighbor[i]].mean(axis=0)
                speed = np.linalg.norm(avg_vel)
                if speed > 1e-9:
                    desired = avg_vel / speed * p.max_speed
                    acc += p.w_alignment * (desired - vel[i])

                # 聚合：趋向邻域质心（环形质心，相对自身计算避免边界误差）
                center_disp = diff[i, neighbor[i]].mean(axis=0)
                d_center = np.linalg.norm(center_disp)
                if d_center > 1e-9:
                    desired = center_disp / d_center * p.max_speed
                    acc += p.w_cohesion * (desired - vel[i])

            steering[i] = acc

        # 转向力截断，随后更新速度并限制最大速度
        steering = _limit(steering, p.max_force)
        vel = _limit(vel + steering, p.max_speed)

        # 更新位置并环绕边界
        pos = pos + vel
        pos[:, 0] = _wrap(pos[:, 0], p.width)
        pos[:, 1] = _wrap(pos[:, 1], p.height)

        self.velocities = vel
        self.positions = pos


# ---------------------------------------------------------------------------
# turtle 可视化
# ---------------------------------------------------------------------------
class Visualizer:
    """turtle 可视化：屏幕初始化、每帧渲染、键盘交互、主循环。"""

    def __init__(self, params: Params):
        self.params = params
        self.flock = Flock(params, np.random.default_rng())
        self.paused = False

        half_w, half_h = params.width / 2.0, params.height / 2.0
        self.screen = turtle.Screen()
        self.screen.setup(params.width, params.height)
        self.screen.setworldcoordinates(-half_w, -half_h, half_w, half_h)
        self.screen.bgcolor(params.bg_color)
        self.screen.title("Boids 群体行为仿真")
        self.screen.tracer(0)  # 关闭自动刷新，手动 update 避免闪烁

        self.turtles = [
            self._make_boid_turtle() for _ in range(params.num_boids)
        ]
        self.status = self._make_status_turtle()
        self._bind_keys()

    # -- 构建 turtle 对象 --------------------------------------------------
    def _make_boid_turtle(self):
        t = turtle.Turtle()
        t.shape("triangle")
        t.color(self.params.boid_color)
        t.shapesize(0.35, 0.6)   # 小三角
        t.penup()
        t.speed(0)
        return t

    def _make_status_turtle(self):
        t = turtle.Turtle()
        t.hideturtle()
        t.penup()
        t.color("gray")
        t.goto(-self.params.width / 2.0 + 10, self.params.height / 2.0 - 20)
        return t

    # -- 键盘交互 ----------------------------------------------------------
    def _bind_keys(self):
        self.screen.onkey(self.toggle_pause, "space")
        self.screen.onkey(lambda: self.change_speed(0.5), "Up")
        self.screen.onkey(lambda: self.change_speed(-0.5), "Down")
        self.screen.onkey(self.reset, "r")
        self.screen.onkey(self.reset, "R")
        self.screen.listen()

    def toggle_pause(self):
        self.paused = not self.paused

    def change_speed(self, delta: float):
        self.params.max_speed = max(0.5, self.params.max_speed + delta)

    def reset(self):
        self.flock.reset()

    # -- 渲染 --------------------------------------------------------------
    def _render(self):
        for t, (x, y), (vx, vy) in zip(
            self.turtles, self.flock.positions, self.flock.velocities
        ):
            t.goto(x, y)
            t.setheading(math.degrees(math.atan2(vy, vx)))
        self._update_status()
        self.screen.update()

    def _update_status(self):
        state = "暂停" if self.paused else "运行"
        self.status.clear()
        self.status.write(
            f"{state} | 速度上限: {self.params.max_speed:.1f} | "
            "空格暂停 ↑↓变速 R重置",
            font=("Arial", 12, "normal"),
        )

    # -- 主循环 ------------------------------------------------------------
    def run(self):
        def loop():
            if not self.paused:
                self.flock.step()
            self._render()  # 暂停时也刷新状态栏
            self.screen.ontimer(loop, self.params.frame_ms)

        loop()
        self.screen.mainloop()


def main():
    viz = Visualizer(Params())
    viz.run()


if __name__ == "__main__":
    main()
