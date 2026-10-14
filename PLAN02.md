# Boids 扩展功能实施计划：障碍物 / 捕食者 / 多种群 / 鼠标驱赶

## 概述
在现有 `boids.py`（numpy 向量化三规则 + turtle 渲染）基础上新增四个扩展，并通过键盘切换按需启用，避免画面拥挤。**完成后同步更新 `README.md`**。

- 障碍物：圆形，左键点击放置、右键删除，Boid 与捕食者均避让
- 捕食者：追最近的 Boid，追上后吃掉（Boid 随机位置重生，总数守恒）
- 多种群：3 个种群、颜色区分，群内对齐/聚合、跨群仅分离
- 鼠标驱赶：鼠标悬停即驱赶半径内 Boid（常开）

## 行为与交互变更

### 键盘开关（新增，默认状态）
| 按键 | 功能 | 默认 |
|------|------|------|
| `O` | 切换障碍物（含避障与显示） | 开 |
| `P` | 切换捕食者 | 关 |
| `M` | 切换多种群 | 开 |
| 空格 / ↑↓ / R | 原有暂停 / 变速 / 重置，保持不变 | — |

- 鼠标**左键**点击：`O` 开启时放置圆形障碍；**右键**：删除距离最近的一个障碍。
- 鼠标**悬停**：常开，驱赶半径内 Boid（无需按键）。

### 多种群
- 3 个种群，颜色 `white`/`cyan`/`orange`，按索引均分 50 个 Boid。
- 对齐、聚合只在**同种群**内计算；分离对**全体 Boid**（不分种群）。
- `M` 关闭时全部归为单种群（白色）；开启时重新按 3 群配色。

### 捕食者
- 2 个红色、稍大三角，`max_speed=4.5`（略快于 Boid）。
- 追捕：找环形距离最近的 Boid，转向逼近。
- 捕食者之间互相分离，且同样避让障碍物、环绕边界。
- Boid 逃离：捕食者进入 `fear_radius` 内的 Boid 产生强排斥。
- 捕食：距离 `< catch_radius` 时该 Boid 位置/速度重置为随机（重生），状态栏显示累计捕食次数。

### 障碍物
- 圆形（turtle 用 `circle` 绘制），半径 30，灰色描边。
- 预设 3 个（约 (-200,-100)、(150,120)、(0,180)）。
- 避障：Boid 与捕食者进入「障碍半径 + 40 感知余量」内产生排斥力。
- `O` 关闭时障碍物隐藏且不再参与避障；开启时恢复。

### 鼠标驱赶（悬停）
- 技术：`screen.getcanvas().bind("<Motion>", handler)` 捕获像素坐标，按 `setworldcoordinates` 的线性映射转为世界坐标（x 线性、y 反转），越界事件做保护。
- 驱赶：鼠标进入 `repel_radius=80` 内的 Boid 产生强排斥（权重与逃离捕食者同级）。

## 数据结构与接口变更
- `Params` 新增字段：捕食者（`num_predators`、`predator_speed`、`fear_radius`、`catch_radius`、`w_flee`、`predator_color`）、障碍物（`obstacle_radius`、`obstacle_margin`、`w_obstacle`、预设位置）、多种群（`num_species`、`species_colors`）、鼠标（`repel_radius`、`w_repel`）。
- `Flock` 增加 `species` 数组 `(N,)`；`step()` 增加入参 `obstacles`、`predators`、`mouse_pos`，并返回/统计被吃事件。
- 新增轻量 `Predator`（位置/速度数组）与 `Obstacle`（位置 + 半径）数据结构，由 `Visualizer` 持有并随帧传入。
- `Visualizer` 增加捕食者 turtle、障碍物 turtle 的创建/显隐，以及鼠标事件与 O/P/M 绑定；状态栏扩展显示开关状态与捕食次数。

## 参数默认值（低风险假设）
- `num_predators=2`、`predator_speed=4.5`、`fear_radius=80`、`catch_radius=10`、`w_flee=2.5`
- `obstacle_radius=30`、`obstacle_margin=40`、`w_obstacle=2.0`
- `num_species=3`、`species_colors=["white","cyan","orange"]`
- `repel_radius=80`、`w_repel=2.5`
- 新增权重与分离/对齐/聚合求和后统一经 `max_force` 截断（沿用现有 `_limit`）。

## 测试与验收
1. `ruff check boids.py` 通过；`import boids` 无报错。
2. 无窗口逻辑测试：多群下 200 帧无 NaN、位置在窗口内、速度 ≤ `max_speed`；捕食者开启后 boid 总数恒为 50（吃掉即重生）。
3. GUI 冒烟测试（自动开窗渲染后关闭）：
   - O/P/M 开关切换生效，状态栏文字正确；
   - 左键放置障碍、右键删除障碍正常，Boid 绕开障碍；
   - 捕食者逼近时 Boid 逃离，追上后重生、计数增加；
   - 悬停鼠标时附近 Boid 被驱散；
   - 空格/↑↓/R 原功能不回归。
4. `README.md` 更新：新增交互按键表、四类扩展的算法说明、新参数表，并将「扩展方向」中已实现条目移除。

## 明确假设
- 捕食者默认关闭、障碍物与多种群默认开启（按 `P` 体验捕食，避免启动即被搅乱）。
- 捕食者不捕食其他捕食者；被吃 Boid 直接在随机位置重生，不缩短捕食者追击流程。
- 障碍物固定为圆形，不随帧移动。
- 鼠标驱赶为「悬停即驱赶」，无按住阈值。
