# mujoco-act-panda-lift
基于 MuJoCo 与 LeRobot 的 Panda 机械臂视觉抓取项目，涵盖专家示范采集、ACT 模仿学习训练和闭环仿真评测。
## 阶段 1 · 首条 Panda Lift 专家示范

使用 NVIDIA EGL 和 `hard_reset=False`，在 seed 2000 下成功采集 75 帧。
RGB 图像、机器人状态、动作和时间戳检查通过，五个抽查帧能看到夹爪接近、抓取和抬升过程。

- 原始数据：[episode_002000.npz](data/raw/preview-egl-soft-20261123-193423/episode_002000.npz)
- 复用入口：[collect_panda_lift_soft_reset.py](scripts/collect_panda_lift_soft_reset.py)
- 上游版本：[upstream-commit.txt](evidence/upstream-commit.txt)

![Panda Lift 首条专家示范](evidence/preview-egl-soft-20261123-193423-panel.png)
