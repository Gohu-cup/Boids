# Boids 群体行为仿真

基于 Craig Reynolds 的 Boids 模型，使用 **numpy 向量化**计算行为规则，并用 **turtle** 实时渲染群体涌现行为。除基础三条规则外，还实现了**障碍物 / 捕食者 / 多种群 / 鼠标驱赶**四个扩展。

## 环境要求

- Python 3.14（已由 `.venv` 提供）
- numpy ≥ 2.5
- tkinter（`python3.14-tk`）

## 运行

```bash
.venv/bin/python main.py
# 或
.venv/bin/python -m boids
```

运行后打开 900×700 的窗口，50 个三角（Boid）随机散布，随时间逐渐形成同向飞行的群体，并可在边缘环绕。

## 交互

### 键盘

| 按键 | 功能 | 默认 |
|------|------|------|
| 空格 | 暂停 / 继续 | — |
| `↑` / `↓` | 增大 / 减小最大速度 | — |
| `O` | 切换障碍物（含避障与显示） | 开 |
| `P` | 切换捕食者 | 关 |
| `M` | 切换多种群（3 群配色） | 开 |
| `R` | 重置为随机初始状态 | — |

### 鼠标

| 操作 | 功能 |
|------|------|
| 左键点击 | 放置圆形障碍物（`O` 开启时） |
| 右键点击 | 删除距离最近的障碍物（`O` 开启时） |
| 悬停移动 | 驱赶半径内 Boid（常开） |

## 算法说明

### 基础三规则

- **分离（Separation）**：`separation_radius` 内的邻居产生排斥，距离越近权重越大，避免碰撞；
- **对齐（Alignment）**：`perception_radius` 内邻居的平均速度方向即为期望方向；
- **聚合（Cohesion）**：趋向 `perception_radius` 内邻居的环形质心，保持群体聚集。

### 扩展规则

- **障碍物避让**：Boid 与捕食者进入「障碍半径 + `obstacle_margin` 余量」内即产生排斥力，越近越强；
- **捕食者追捕**：捕食者追最近 Boid，彼此分离并同样避障；追上（距离 `< catch_radius`）后该 Boid 在随机位置重生，总数守恒；
- **逃离捕食者**：捕食者进入 `fear_radius` 内的 Boid 产生强排斥；
- **多种群**：`num_species` 个种群颜色区分，对齐/聚合只在**同种群**内计算，分离对全体 Boid 生效；
- **鼠标驱赶**：鼠标进入 `repel_radius` 内的 Boid 产生强排斥（悬停即触发）。

合力经 `max_force` 截断后叠加到速度上，速度再截断到 `max_speed`，位置按窗口边界**环绕（wrap-around）**。邻域距离同样采用环形距离，避免边缘群体被错误分割。

## 参数表

所有参数集中在 `boids.py` 顶部的 `Params` 类中，可直接修改：

### 基础

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_boids` | 50 | Boid 数量 |
| `max_speed` | 4.0 | Boid 最大速度（像素/帧） |
| `max_force` | 0.2 | 最大转向力（加速度上限） |
| `perception_radius` | 50 | 对齐 / 聚合感知半径 |
| `separation_radius` | 20 | 分离最小距离 |
| `w_separation` / `w_alignment` / `w_cohesion` | 1.8 / 1.0 / 1.0 | 三条规则权重 |
| `width` / `height` | 900 / 700 | 窗口尺寸（像素） |
| `bg_color` / `boid_color` | black / white | 背景与单种群 Boid 颜色 |
| `frame_ms` | 16 | 帧间隔（约 60 FPS） |

### 捕食者

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_predators` | 2 | 捕食者数量 |
| `predator_speed` | 4.5 | 捕食者最大速度 |
| `fear_radius` | 80 | Boid 逃离捕食者的恐惧半径 |
| `catch_radius` | 10 | 捕食判定距离 |
| `w_flee` | 2.5 | 逃离捕食者权重 |
| `predator_color` | red | 捕食者颜色 |

### 障碍物

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `obstacle_radius` | 30 | 障碍物半径 |
| `obstacle_margin` | 40 | 避障感知余量（半径外的缓冲带） |
| `w_obstacle` | 2.0 | 避障权重 |
| `obstacle_color` | gray | 障碍物描边颜色 |
| `preset_obstacles` | 3 处 | 预设障碍物位置 |

### 多种群

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_species` | 3 | 种群数量 |
| `species_colors` | white / cyan / orange | 各种群颜色 |

### 鼠标驱赶

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `repel_radius` | 80 | 驱赶半径 |
| `w_repel` | 2.5 | 驱赶权重 |

## 代码结构

代码按职责拆分为 `boids` 包（`main.py` 为顶层入口）：

```
boids/
├── __init__.py   # 包导出（Params / Flock / Visualizer 等）
├── __main__.py   # 支持 python -m boids 运行
├── params.py     # Params：全部可调参数
├── entities.py   # Boid / Obstacle / Predator 数据实体
├── math_utils.py # 向量工具：坐标环绕、限长、逃离/避障转向力
├── flock.py      # Flock：仿真核心（三规则 + 捕食者更新）
└── visualizer.py # Visualizer：turtle 渲染与键盘/鼠标交互
```

依赖方向：`params` ← `entities` / `math_utils` ← `flock` ← `visualizer`，无循环依赖。

## 扩展方向

- 速度矢量与感知范围可视化（教学演示）；
- 按速度或种群密度着色；
- 随机扰动（jitter）使运动更自然；
- 移动障碍物、捕食者协作围猎、Boid 逃逸策略等更复杂行为。
