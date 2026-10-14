"""Boids 群体行为仿真

基于 Craig Reynolds 的 Boids 模型，使用 numpy 向量化计算规则，
并用 turtle 实时渲染群体涌现行为：

- 分离（Separation）：远离太近的邻居，避免碰撞
- 对齐（Alignment）：趋向邻域内平均速度方向，保持队形一致
- 聚合（Cohesion）：趋向邻域质心，保持群体聚集

扩展功能：
- 障碍物：圆形，左键点击放置、右键删除，Boid 与捕食者均避让
- 捕食者：追最近的 Boid，追上后吃掉（Boid 随机位置重生）
- 多种群：3 个种群、颜色区分，群内对齐/聚合、跨群仅分离
- 鼠标驱赶：鼠标悬停即驱赶半径内 Boid

运行方式：
    python boids.py

交互按键：
    空格       暂停 / 继续
    ↑ / ↓     增大 / 减小最大速度
    O          切换障碍物（默认开）
    P          切换捕食者（默认关）
    M          切换多种群（默认开）
    R          重置为随机初始状态
    鼠标左键   放置障碍物（O 开启时）
    鼠标右键   删除最近的障碍物（O 开启时）
    鼠标悬停   驱赶附近的 Boid
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


@dataclass
class Obstacle:
    """圆形障碍物：位置 + 半径。"""

    position: np.ndarray  # (2,) x, y
    radius: float


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


def _away_steering(
    position: np.ndarray,
    velocity: np.ndarray,
    points: np.ndarray,
    radius: float,
    weight: float,
    speed: float,
    params: Params,
) -> np.ndarray:
    """计算逃离一组点（半径 radius 内）的转向力，距离越近权重越大。"""
    if points.size == 0:
        return np.zeros(2)
    d_vec = points - position                 # (K,2)
    d_vec[:, 0] = _wrap(d_vec[:, 0], params.width)
    d_vec[:, 1] = _wrap(d_vec[:, 1], params.height)
    d = np.linalg.norm(d_vec, axis=1)         # (K,)
    close = d < radius
    if not close.any():
        return np.zeros(2)
    d = d[close]
    away = -d_vec[close]
    unit = away / np.maximum(d[:, None], 1e-9)
    w = 1.0 / np.maximum(d, 1e-6)
    direction = (unit * w[:, None]).sum(axis=0)
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    return weight * (direction * speed - velocity)


def _obstacle_steering(
    position: np.ndarray,
    velocity: np.ndarray,
    obstacles: list,
    params: Params,
    speed: float,
) -> np.ndarray:
    """计算避让障碍物的转向力（进入「半径 + 余量」内才生效）。"""
    steer = np.zeros(2)
    for ob in obstacles:
        d_vec = ob.position - position
        d_vec[0] = _wrap(d_vec[0], params.width)
        d_vec[1] = _wrap(d_vec[1], params.height)
        d = np.linalg.norm(d_vec)
        avoid_r = ob.radius + params.obstacle_margin
        if 1e-9 < d < avoid_r:
            away = -d_vec / d
            strength = (avoid_r - d) / avoid_r   # 越近越强
            steer += params.w_obstacle * (away * speed - velocity) * strength
    return steer


# ---------------------------------------------------------------------------
# 捕食者
# ---------------------------------------------------------------------------
class Predator:
    """捕食者群体：位置/速度数组，负责自身初始化。"""

    def __init__(self, params: Params, rng: np.random.Generator):
        self.params = params
        self.rng = rng
        self.positions = np.zeros((params.num_predators, 2))
        self.velocities = np.zeros((params.num_predators, 2))
        self.reset()

    def reset(self):
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


# ---------------------------------------------------------------------------
# 群体
# ---------------------------------------------------------------------------
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
    def _update_predators(self, predators, obstacles, obstacles_enabled):
        """更新捕食者：追最近 Boid、彼此分离、避障、捕食判定。返回捕食数。"""
        p = self.params
        k = predators.positions.shape[0]
        eaten = 0
        for j in range(k):
            acc = np.zeros(2)

            # 追最近的 Boid
            to_boids = self.positions - predators.positions[j]
            to_boids[:, 0] = _wrap(to_boids[:, 0], p.width)
            to_boids[:, 1] = _wrap(to_boids[:, 1], p.height)
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
                acc += _away_steering(
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
                acc += _obstacle_steering(
                    predators.positions[j],
                    predators.velocities[j],
                    obstacles,
                    p,
                    p.predator_speed,
                )

            # 更新速度与位置
            predators.velocities[j] = _limit(
                predators.velocities[j] + _limit(acc, p.max_force),
                p.predator_speed,
            )
            predators.positions[j] = (
                predators.positions[j] + predators.velocities[j]
            )
            predators.positions[j, 0] = _wrap(predators.positions[j, 0], p.width)
            predators.positions[j, 1] = _wrap(predators.positions[j, 1], p.height)

            # 捕食判定：追上即吃掉（重生）
            to_boids = self.positions - predators.positions[j]
            to_boids[:, 0] = _wrap(to_boids[:, 0], p.width)
            to_boids[:, 1] = _wrap(to_boids[:, 1], p.height)
            bd = np.linalg.norm(to_boids, axis=1)
            if bd.size and bd.min() < p.catch_radius:
                self._respawn(int(np.argmin(bd)))
                eaten += 1
        return eaten

    # -- 规则计算 ----------------------------------------------------------
    def step(
        self,
        obstacles: list,
        predators,
        mouse_pos,
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
        diff[:, :, 0] = _wrap(diff[:, :, 0], p.width)
        diff[:, :, 1] = _wrap(diff[:, :, 1], p.height)
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
                acc += _away_steering(
                    pos[i], vel[i], predators.positions,
                    p.fear_radius, p.w_flee, p.max_speed, p,
                )

            # 避障
            if obstacles_enabled and obstacles:
                acc += _obstacle_steering(pos[i], vel[i], obstacles, p, p.max_speed)

            # 鼠标驱赶（悬停常开）
            if mouse_pos is not None:
                acc += _away_steering(
                    pos[i], vel[i], mouse_pos[None, :],
                    p.repel_radius, p.w_repel, p.max_speed, p,
                )

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

        # 捕食者更新
        if predators_enabled and predators.positions.size:
            return self._update_predators(predators, obstacles, obstacles_enabled)
        return 0


# ---------------------------------------------------------------------------
# turtle 可视化
# ---------------------------------------------------------------------------
class Visualizer:
    """turtle 可视化：屏幕初始化、每帧渲染、键盘与鼠标交互、主循环。"""

    def __init__(self, params: Params):
        self.params = params
        self.rng = np.random.default_rng()
        self.flock = Flock(params, self.rng)
        self.predators = Predator(params, self.rng)
        self.paused = False
        self.obstacles_enabled = True
        self.predators_enabled = False
        self.multi_species = True
        self.mouse_pos = None
        self.eaten_total = 0

        half_w, half_h = params.width / 2.0, params.height / 2.0
        self.screen = turtle.Screen()
        self.screen.setup(params.width, params.height)
        self.screen.setworldcoordinates(-half_w, -half_h, half_w, half_h)
        self.screen.bgcolor(params.bg_color)
        self.screen.title("Boids 群体行为仿真")
        self.screen.tracer(0)  # 关闭自动刷新，手动 update 避免闪烁

        # Boid、捕食者、障碍物、状态栏
        self.turtles = [self._make_boid_turtle() for _ in range(params.num_boids)]
        self.predator_turtles = [
            self._make_predator_turtle() for _ in range(params.num_predators)
        ]
        self.obstacles = [
            Obstacle(np.array(pos, dtype=float), params.obstacle_radius)
            for pos in params.preset_obstacles
        ]
        self.obstacle_turtles = [
            self._make_obstacle_turtle(ob) for ob in self.obstacles
        ]
        self.status = self._make_status_turtle()

        self._sync_species_colors()
        self._sync_predators()
        self._bind_keys()
        self._bind_mouse()

    # -- 构建 turtle 对象 --------------------------------------------------
    def _make_boid_turtle(self):
        t = turtle.Turtle()
        t.shape("triangle")
        t.color(self.params.boid_color)
        t.shapesize(0.35, 0.6)   # 小三角
        t.penup()
        t.speed(0)
        return t

    def _make_predator_turtle(self):
        t = turtle.Turtle()
        t.shape("triangle")
        t.color(self.params.predator_color)
        t.shapesize(0.5, 0.9)    # 稍大三角
        t.penup()
        t.speed(0)
        t.hideturtle()           # 初始隐藏，P 开启时显示
        return t

    def _make_obstacle_turtle(self, obstacle):
        t = turtle.Turtle()
        t.hideturtle()
        t.speed(0)
        t.penup()
        t.color(self.params.obstacle_color)
        self._draw_obstacle(t, obstacle)
        return t

    def _draw_obstacle(self, t, obstacle):
        """（重新）绘制圆形障碍物描边。"""
        t.clear()
        t.penup()
        t.goto(obstacle.position[0], obstacle.position[1] - obstacle.radius)
        t.pendown()
        t.circle(obstacle.radius)
        t.penup()

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
        self.screen.onkey(self.toggle_obstacles, "o")
        self.screen.onkey(self.toggle_obstacles, "O")
        self.screen.onkey(self.toggle_predators, "p")
        self.screen.onkey(self.toggle_predators, "P")
        self.screen.onkey(self.toggle_multi_species, "m")
        self.screen.onkey(self.toggle_multi_species, "M")
        self.screen.onkey(self.reset, "r")
        self.screen.onkey(self.reset, "R")
        self.screen.listen()

    # -- 鼠标交互 ----------------------------------------------------------
    def _bind_mouse(self):
        # 左/右键点击（world 坐标）用于放置/删除障碍物
        self.screen.onclick(self._on_left_click, btn=1)
        self.screen.onclick(self._on_right_click, btn=3)
        # 悬停：绑定底层 Tk canvas 的 Motion/Leave 事件跟踪鼠标
        canvas = self.screen.getcanvas()
        canvas.bind("<Motion>", self._on_mouse_move)
        canvas.bind("<Leave>", self._on_mouse_leave)

    def _on_mouse_move(self, event):
        """把画布像素坐标转为世界坐标（x 线性、y 反转）。"""
        p = self.params
        cw = max(self.screen.getcanvas().winfo_width(), 1)
        ch = max(self.screen.getcanvas().winfo_height(), 1)
        x = -p.width / 2.0 + (event.x / cw) * p.width
        y = p.height / 2.0 - (event.y / ch) * p.height
        self.mouse_pos = np.array([x, y], dtype=float)

    def _on_mouse_leave(self, _event):
        self.mouse_pos = None

    def _on_left_click(self, x, y):
        if self.obstacles_enabled:
            self.add_obstacle(x, y)

    def _on_right_click(self, x, y):
        if self.obstacles_enabled:
            self.remove_obstacle(x, y)

    def add_obstacle(self, x: float, y: float):
        ob = Obstacle(np.array([x, y], dtype=float), self.params.obstacle_radius)
        self.obstacles.append(ob)
        self.obstacle_turtles.append(self._make_obstacle_turtle(ob))

    def remove_obstacle(self, x: float, y: float):
        if not self.obstacles:
            return
        pts = np.array([o.position for o in self.obstacles])
        d = np.linalg.norm(pts - np.array([x, y]), axis=1)
        i = int(np.argmin(d))
        self.obstacle_turtles[i].clear()
        del self.obstacles[i]
        del self.obstacle_turtles[i]

    # -- 开关回调 ----------------------------------------------------------
    def toggle_pause(self):
        self.paused = not self.paused

    def change_speed(self, delta: float):
        self.params.max_speed = max(0.5, self.params.max_speed + delta)

    def toggle_obstacles(self):
        self.obstacles_enabled = not self.obstacles_enabled
        self._sync_obstacles()

    def toggle_predators(self):
        self.predators_enabled = not self.predators_enabled
        self._sync_predators()

    def toggle_multi_species(self):
        self.multi_species = not self.multi_species
        self.flock.set_multi_species(self.multi_species)
        self._sync_species_colors()

    def reset(self):
        self.flock.reset()
        self.predators.reset()
        self.eaten_total = 0

    # -- 显示同步 ----------------------------------------------------------
    def _sync_species_colors(self):
        colors = self.params.species_colors
        for t, s in zip(self.turtles, self.flock.species):
            if self.multi_species:
                t.color(colors[s % len(colors)])
            else:
                t.color(self.params.boid_color)

    def _sync_predators(self):
        for t in self.predator_turtles:
            if self.predators_enabled:
                t.showturtle()
            else:
                t.hideturtle()

    def _sync_obstacles(self):
        for t, ob in zip(self.obstacle_turtles, self.obstacles):
            if self.obstacles_enabled:
                self._draw_obstacle(t, ob)
            else:
                t.clear()

    # -- 渲染 --------------------------------------------------------------
    def _render(self):
        for t, (x, y), (vx, vy) in zip(
            self.turtles, self.flock.positions, self.flock.velocities
        ):
            t.goto(x, y)
            t.setheading(math.degrees(math.atan2(vy, vx)))
        for t, (x, y), (vx, vy) in zip(
            self.predator_turtles, self.predators.positions, self.predators.velocities
        ):
            t.goto(x, y)
            t.setheading(math.degrees(math.atan2(vy, vx)))
        self._update_status()
        self.screen.update()

    def _update_status(self):
        state = "暂停" if self.paused else "运行"
        obs = "开" if self.obstacles_enabled else "关"
        pre = "开" if self.predators_enabled else "关"
        mul = "开" if self.multi_species else "关"
        self.status.clear()
        self.status.write(
            f"{state} | 速度:{self.params.max_speed:.1f} | "
            f"O障碍[{obs}] P捕食[{pre}] M多群[{mul}] | 捕食:{self.eaten_total} | "
            "空格暂停 ↑↓变速 R重置",
            font=("Arial", 11, "normal"),
        )

    # -- 主循环 ------------------------------------------------------------
    def run(self):
        def loop():
            if not self.paused:
                self.eaten_total += self.flock.step(
                    obstacles=self.obstacles,
                    predators=self.predators,
                    mouse_pos=self.mouse_pos,
                    obstacles_enabled=self.obstacles_enabled,
                    predators_enabled=self.predators_enabled,
                )
            self._render()  # 暂停时也刷新状态栏
            self.screen.ontimer(loop, self.params.frame_ms)

        loop()
        self.screen.mainloop()


def main():
    viz = Visualizer(Params())
    viz.run()


if __name__ == "__main__":
    main()
