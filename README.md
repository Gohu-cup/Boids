# Boids 群体行为仿真

基于 Craig Reynolds 的 Boids 模型，使用 **numpy 向量化**计算三条规则（分离 / 对齐 / 聚合），并用 **turtle** 实时渲染群体涌现行为。

## 环境要求

- Python 3.14（已由 `.venv` 提供）
- numpy ≥ 2.5
- tkinter（`python3.14-tk`）

## 运行

```bash
.venv/bin/python boids.py
```

运行后会打开 900×700 的窗口，50 个白色三角（Boid）随机散布，随时间逐渐形成同向飞行的群体，并可在边缘环绕。

## 交互按键

| 按键 | 功能 |
|------|------|
| 空格 | 暂停 / 继续 |
| `↑` / `↓` | 增大 / 减小最大速度 |
| `R` | 重置为随机初始状态 |

## 算法说明

每个 Boid 每帧依据邻域内其他个体计算三条规则的合力：

- **分离（Separation）**：`separation_radius` 内的邻居产生排斥，距离越近权重越大，避免碰撞；
- **对齐（Alignment）**：`perception_radius` 内邻居的平均速度方向即为期望方向；
- **聚合（Cohesion）**：趋向 `perception_radius` 内邻居的环形质心，保持群体聚集。

合力经 `max_force` 截断后叠加到速度上，速度再截断到 `max_speed`，位置按窗口边界**环绕（wrap-around）**。邻域距离同样采用环形距离，避免边缘群体被错误分割。

## 参数表

所有参数集中在 `boids.py` 顶部的 `Params` 类中，可直接修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_boids` | 50 | Boid 数量 |
| `max_speed` | 4.0 | 最大速度（像素/帧） |
| `max_force` | 0.2 | 最大转向力（加速度上限） |
| `perception_radius` | 50 | 对齐 / 聚合感知半径 |
| `separation_radius` | 20 | 分离最小距离 |
| `w_separation` | 1.8 | 分离权重 |
| `w_alignment` | 1.0 | 对齐权重 |
| `w_cohesion` | 1.0 | 聚合权重 |
| `width` / `height` | 900 / 700 | 窗口尺寸（像素） |
| `bg_color` / `boid_color` | black / white | 背景与 Boid 颜色 |
| `frame_ms` | 16 | 帧间隔（约 60 FPS） |

## 代码结构

- `Params`：全部可调参数；
- `Boid`：单个个体的初始状态（位置、速度）；
- `Flock`：持有全体 `(N,2)` 位置/速度数组，向量化计算三规则；
- `Visualizer`：turtle 屏幕初始化、每帧渲染、键盘绑定与主循环；
- `main()`：入口。

## 扩展方向

- 障碍物 / 捕食者 / 多种群；
- 速度矢量与感知范围可视化（教学演示）；
- 按速度或群体着色；
- 随机扰动（jitter）使运动更自然。
