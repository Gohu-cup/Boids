# Boids 算法实现与可视化 —— 实施计划

## 概述
在 `/home/zefeng/Documents/pi_workspace/Boids算法实现与可视化` 下，用 Python 标准库 `turtle` + 已装的 `numpy` 实现经典 Boids 三规则（分离/对齐/聚合）仿真：50 个 Boid，朝向三角形渲染 + 键盘交互，输出可运行、可扩展的完整程序。

## 前置环境（阻塞项，需用户执行）
- 当前 `.venv`（Python 3.14.8，deadsnakes PPA）缺 `tkinter`，导致 `import turtle` 失败。
- 需安装匹配版本的包（我无法非交互 sudo，**由你执行**）：
  `sudo apt-get install -y python3.14-tk`
- 装完我验证 `.venv/bin/python -c "import turtle"`。
- 已确认：图形环境可用（`DISPLAY=:0`，Xwayland）；tcl/tk 8.6 运行库已存在，无需额外安装。

## 文件组织（本次新建）
```
Boids算法实现与可视化/
├── boids.py    # 主程序（参数配置、Boid、Flock、可视化、交互、main 入口）
├── README.md   # 使用说明 + 参数表
└── .venv/      # 已存在（Python 3.14.8 + numpy 2.5.4）
```

## 代码结构与接口
- `Params`（`@dataclass`）：全部可调参数集中一处，便于调优。
- `Boid`：单个个体，持有 `position`、`velocity`（numpy 一维向量）。
- `Flock`：持有全体 `(N,2)` 位置/速度数组，向量化计算三规则与位置更新。
- `Visualizer`：turtle 屏幕初始化、每帧渲染、键盘绑定、`ontimer` 主循环。
- `main()`：程序入口，组装以上模块。

## 算法（numpy 向量化）
- 邻域距离用**环形（toroidal）距离**，与环绕边界保持一致，避免边缘群体分裂。
- 分离：`SEPARATION_RADIUS` 内排斥力；对齐：`PERCEPTION_RADIUS` 内平均速度；聚合：`PERCEPTION_RADIUS` 内质心方向。
- 合力 = `W_SEPARATION·分离 + W_ALIGNMENT·对齐 + W_COHESION·聚合`，截断到 `MAX_FORCE`。
- 速度截断到 `MAX_SPEED`；位置按窗口边界**环绕（wrap-around）**。

## 参数默认值（低风险假设，写入 Params 便于改）
- `NUM_BOIDS=50`、`MAX_SPEED=4.0`、`MAX_FORCE=0.2`
- `PERCEPTION_RADIUS=50`、`SEPARATION_RADIUS=20`
- `W_SEPARATION=1.8`、`W_ALIGNMENT=1.0`、`W_COHESION=1.0`
- 窗口 900×700，深色背景，单色三角
- 初始：随机位置 + 随机方向，速度大小 = `MAX_SPEED/2`

## 渲染
- `screen.tracer(0)` + `update()` 手动刷新，`ontimer` 约 60 FPS 驱动主循环，避免闪烁。
- 每帧：将每个 turtle 的朝向设为速度方向、位置设为当前坐标（50 个小三角 `shapesize`）。

## 交互（键盘）
- `空格`：暂停/继续
- `↑` / `↓`：增大/减小最大速度
- `R`：重置为随机初始状态

## 测试与验收
1. `import turtle` 成功（安装 `python3.14-tk` 后）。
2. `python boids.py` 无报错，窗口正常打开。
3. 涌现行为：个体互不重叠（分离）、同向飞行（对齐）、逐渐聚群（聚合）、边缘环绕。
4. 空格暂停/恢复、↑/↓ 变速、R 重置均生效。
5. 50 个 Boid 下帧率流畅、无明显卡顿。

## 明确假设与默认
- 边界采用**环绕**（非反弹）。
- 可视化仅三角形朝向（不含速度矢量/感知圈，默认关闭可后续扩展）。
- 单文件 `boids.py` 内按类模块化组织（满足可读性与二次开发），不拆多模块包。
- 不引入 jitter 噪声；障碍物、捕食者、多种群等列入后续扩展，不在本次范围。
