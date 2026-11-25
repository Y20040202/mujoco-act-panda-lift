# mujoco-act-panda-lift
基于 MuJoCo 与 LeRobot 的 Panda 机械臂视觉抓取项目，涵盖专家示范采集、ACT 模仿学习训练和闭环仿真评测。
## 阶段 1 · 首条 Panda Lift 专家示范

使用 NVIDIA EGL 和 `hard_reset=False`，在 seed 2000 下成功采集 75 帧。
RGB 图像、机器人状态、动作和时间戳检查通过，五个抽查帧能看到夹爪接近、抓取和抬升过程。

- 原始数据：[episode_002000.npz](data/raw/preview-egl-soft-20261123-193423/episode_002000.npz)
- 复用入口：[collect_panda_lift_soft_reset.py](scripts/collect_panda_lift_soft_reset.py)
- 上游版本：[upstream-commit.txt](evidence/upstream-commit.txt)

![Panda Lift 首条专家示范](evidence/preview-egl-soft-20261123-193423-panel.png)

## 阶段 2 · 示范质检与 LeRobot 转换

将 7 条专家示范按完整回合划分，逐条检查图像、状态、动作与时间戳是否对齐。转换为 LeRobot 后，回读每条回合的闭合指令帧，核对动作、状态和时间戳。**7 条回合均通过回读检查。**

| 划分 | 种子 | 回合数 | 帧数 |
| --- | --- | ---: | ---: |
| 训练 | `2100–2104` | 5 | 394 |
| 验证 | `2200–2201` | 2 | 161 |

- [第二阶段记录](evidence/d02/stage-2-progress.md) · [回合索引](evidence/d02/episode-index.csv) · [转换验收日志](evidence/d02/lerobot-check.txt)
- [单条检查](scripts/inspect_episode.py) · [批量检查](scripts/audit_d02.py) · [转换与回读](scripts/convert_d02.py)

![七条示范的首帧、闭合指令帧和末帧](evidence/d02/rgb-panel.png)

## 阶段 3 · ACT 短训与检查点重载

从新进程加载自训的 `step_000100`，每次执行 20 步，在 `2300–2302` 成功 **1/3**。同一批种子上，1000 步权重成功 **0/3**。

- [阶段 3 记录与复跑配置](evidence/d03/README.md) · [终端转录的逐回合指标](evidence/d03/closed-loop-transcript.csv) · [100 步检查点路径](evidence/d03/checkpoint-step-000100.txt)
- [2301 成功回放](evidence/d03/videos/seed_002301-success.mp4) · [2301 失败回放](evidence/d03/videos/seed_002301-failure.mp4) · [抓取关键帧对照](evidence/d03/seed-2301-comparison.png)

## 阶段 4 · v1 扩采训练与新旧权重回归

`v1` 自训 **3,000 步**，batch size **32**，最终选用 `step_003000`。验证 L1 从第 0 步的 **0.823419** 降至第 3000 步的 **0.114877**。固定执行 **20 步**，在同一批 `1010–1019` 初态上，旧权重成功 **2/10**，`v1` 成功 **5/10**。

- [阶段 4 训练与回归记录](evidence/v1/regression/v1-exec20-20261125-195734/README.md) · [训练配置原件](evidence/v1/training/v1_3000_20261125_194654/run_config.json) · [训练指标原件](evidence/v1/training/v1_3000_20261125_194654/metrics.csv)
- [20 步回归终端汇总](evidence/v1/regression/v1-exec20-20261125-195734/eval-summary.txt) · [v1 采集、训练与配对评测入口](scripts/run_panda_lift_v1.py)

## 阶段 5 · 执行步数消融

固定同一份 `v1` 检查点和 `1010–1019` 的初态，比较每次预测后执行 1 步和 5 步。

| 执行长度 | 成功次数 | 按动作队列估算的前向次数 |
| ---: | ---: | ---: |
| 1 步 | 2/10 | 2,558 |
| 5 步 | 5/10 | 391 |

前向总次数同时受执行长度和回合步数影响。

- [逐种子配对结果](evidence/v1/paired/v1_3000_20261125_194654-step_003000-1010-1019/paired-comparison.csv) · [消融汇总](evidence/v1/paired/v1_3000_20261125_194654-step_003000-1010-1019/paired-summary.txt)

## 最终独立评测 · 100 个新种子

冻结 `step_003000` 和每次执行 5 步的配置，在 `3100–3199` 成功 **37/100**。成功回合平均用 77.2 步，63 个失败回合全部达到 300 步上限。动作裁剪只出现在夹爪通道。

- [复跑命令、失败分析和原始证据](evidence/v1/final/exec5-3100-3199-20261125-202416/report.md)
- [逐回合 CSV](evidence/v1/final/exec5-3100-3199-20261125-202416/exec5.csv) · [原始评测日志](evidence/v1/final/exec5-3100-3199-20261125-202416/eval.log) · [评测汇总](evidence/v1/final/exec5-3100-3199-20261125-202416/eval-summary.txt) · [检查点路径](evidence/v1/final/exec5-3100-3199-20261125-202416/checkpoint.txt)
- [成功回放 · 3107](evidence/v1/final/exec5-3100-3199-20261125-202416/videos/seed_003107.mp4) · [失败回放 · 3111](evidence/v1/final/exec5-3100-3199-20261125-202416/videos/seed_003111.mp4)
