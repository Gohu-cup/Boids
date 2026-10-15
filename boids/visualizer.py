"""turtle 可视化层。

:class:`Visualizer` 只负责渲染与交互：通过 :class:`~boids.simulation.Simulation`
的门面接口读取状态、下发指令，并把视觉同步（障碍物显隐、颜色等）保持在本层。
"""

from __future__ import annotations

import math
import turtle

import numpy as np

from .params import Params
from .simulation import Simulation


class Visualizer:
    """turtle 可视化：屏幕初始化、每帧渲染、键盘与鼠标交互、主循环。"""

    def __init__(self, params: Params | None = None):
        self.params = params if params is not None else Params()
        self.simulation = Simulation(self.params)
        self.paused = False
        self.mouse_pos = None

        half_w, half_h = self.params.width / 2.0, self.params.height / 2.0
        self.screen = turtle.Screen()
        self.screen.setup(self.params.width, self.params.height)
        self.screen.setworldcoordinates(-half_w, -half_h, half_w, half_h)
        self.screen.bgcolor(self.params.bg_color)
        self.screen.title("Boids 群体行为仿真")
        self.screen.tracer(0)  # 关闭自动刷新，手动 update 避免闪烁

        # Boid、捕食者、障碍物、状态栏
        self.turtles = [self._make_boid_turtle() for _ in range(self.params.num_boids)]
        self.predator_turtles = [
            self._make_predator_turtle() for _ in range(self.params.num_predators)
        ]
        self.obstacle_turtles = [
            self._make_obstacle_turtle(ob) for ob in self.simulation.obstacles
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
        if self.simulation.obstacles_enabled:
            self.add_obstacle(x, y)

    def _on_right_click(self, x, y):
        if self.simulation.obstacles_enabled:
            self.remove_obstacle(x, y)

    def add_obstacle(self, x: float, y: float):
        self.simulation.add_obstacle(x, y)
        self.obstacle_turtles.append(
            self._make_obstacle_turtle(self.simulation.obstacles[-1])
        )

    def remove_obstacle(self, x: float, y: float):
        i = self.simulation.remove_obstacle(x, y)
        if i is not None:
            self.obstacle_turtles[i].clear()
            del self.obstacle_turtles[i]

    # -- 开关回调（委托给 Simulation + 视觉同步） --------------------------
    def toggle_pause(self):
        self.paused = not self.paused

    def change_speed(self, delta: float):
        self.simulation.change_speed(delta)

    def toggle_obstacles(self):
        self.simulation.toggle_obstacles()
        self._sync_obstacles()

    def toggle_predators(self):
        self.simulation.toggle_predators()
        self._sync_predators()

    def toggle_multi_species(self):
        self.simulation.toggle_multi_species()
        self._sync_species_colors()

    def reset(self):
        self.simulation.reset()

    # -- 显示同步 ----------------------------------------------------------
    def _sync_species_colors(self):
        colors = self.params.species_colors
        for t, s in zip(self.turtles, self.simulation.boid_species):
            if self.simulation.multi_species:
                t.color(colors[s % len(colors)])
            else:
                t.color(self.params.boid_color)

    def _sync_predators(self):
        for t in self.predator_turtles:
            if self.simulation.predators_enabled:
                t.showturtle()
            else:
                t.hideturtle()

    def _sync_obstacles(self):
        for t, ob in zip(self.obstacle_turtles, self.simulation.obstacles):
            if self.simulation.obstacles_enabled:
                self._draw_obstacle(t, ob)
            else:
                t.clear()

    # -- 渲染 --------------------------------------------------------------
    def _render(self):
        sim = self.simulation
        for t, (x, y), (vx, vy) in zip(
            self.turtles, sim.boid_positions, sim.boid_velocities
        ):
            t.goto(x, y)
            t.setheading(math.degrees(math.atan2(vy, vx)))
        for t, (x, y), (vx, vy) in zip(
            self.predator_turtles, sim.predator_positions, sim.predator_velocities
        ):
            t.goto(x, y)
            t.setheading(math.degrees(math.atan2(vy, vx)))
        self._update_status()
        self.screen.update()

    def _update_status(self):
        sim = self.simulation
        state = "暂停" if self.paused else "运行"
        obs = "开" if sim.obstacles_enabled else "关"
        pre = "开" if sim.predators_enabled else "关"
        mul = "开" if sim.multi_species else "关"
        self.status.clear()
        self.status.write(
            f"{state} | 速度:{self.params.max_speed:.1f} | "
            f"O障碍[{obs}] P捕食[{pre}] M多群[{mul}] | 捕食:{sim.eaten_total} | "
            "空格暂停 ↑↓变速 R重置",
            font=("Arial", 11, "normal"),
        )

    # -- 主循环 ------------------------------------------------------------
    def run(self):
        def loop():
            if not self.paused:
                self.simulation.step(self.mouse_pos)
            self._render()  # 暂停时也刷新状态栏
            self.screen.ontimer(loop, self.params.frame_ms)

        loop()
        self.screen.mainloop()


def main():
    """程序入口：以默认参数启动仿真。"""
    viz = Visualizer()
    viz.run()
