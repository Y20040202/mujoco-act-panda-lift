# Panda Lift v1 最终评测归档

## 实验设置

2026 年 11 月 25 日，在 Ubuntu 24.04 的 RTX 4090 服务器上运行。MuJoCo 使用 NVIDIA EGL 和 `hard_reset=False`。ACT 使用冻结的 `v1` 检查点，每次预测后执行 5 步。测试种子为 `3100–3199`，每回合最多 300 步，记录前 20 个回合的录像。

最终检查点为 `step_003000`。本次训练使用 3,000 步、batch size 32、4 个数据加载进程和训练种子 0。检查点的验证集 L1 为 `0.114877`，训练配置记录为每次预测 20 个动作，最终评测设置为每次执行 5 步。

上游实现固定为 `2932ee3cd0309d58d47afbc441375d88008c86f2`。任务要求方块中心高于桌面 0.08 m 并连续保持 10 步。本次桌面高度为 0.800 m。

## 实测结果

| 指标 | 结果 |
| --- | ---: |
| 成功回合 | 37/100 |
| 成功率的 Wilson 95% 区间 | 28.2%–46.8% |
| 全部回合平均步数 | 217.56 |
| 成功回合平均步数 / 中位步数 | 77.2 / 70 |
| 失败回合 | 63 次达到 300 步上限 |
| 发生动作裁剪的平均步占比 | 98.8% |
| 客户端动作请求平均耗时 | 2.28 ms |

动作裁剪只出现在夹爪通道。成功组的逐回合平均夹爪裁剪率是 98.0%，失败组是 99.2%。该统计对每步判断是否越界，越界幅度需要结合 `maximum_action_excess` 检查。上表的耗时对应客户端请求动作，覆盖从动作队列取动作的请求。

| 失败表现 | 回合数 | 复查种子 |
| --- | ---: | --- |
| 最高高度比初始高度增加不到 1 mm | 34 | `3100` |
| 高度明显增加，峰值仍低于绝对高度 0.880 m | 21 | `3102` |
| 峰值超过绝对高度 0.880 m，最终未达到保持条件 | 8 | `3111` |

失败且结束高度仍高于 0.850 m 的 20 个回合，方块最终均向 X 负方向偏移。该组横向 X 位移中位数为 −33.7 mm，成功组为 −1.0 mm。复看 `3111` 可以检查偏移发生在抬升、夹持滑动或位置回调的哪个阶段。建议用 `3107`（66 步成功）的录像作为动作对照。

## 复跑

在远程 Ubuntu 上，从仓库根目录使用现有 `panda_sim` 和 `panda_act`。采集、训练与 1 步和 5 步配对评测的入口如下：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate panda_act
python scripts/run_panda_lift_v1.py prepare
python scripts/run_panda_lift_v1.py train --steps 3000 --batch-size 32 --num-workers 4
python scripts/run_panda_lift_v1.py evaluate
```

复跑最终评测时，终端 A 从服务器留存的 `checkpoint.txt` 加载归档对应的检查点。先看到策略服务显示 `executed action steps: 5` 和 `status: ready`。

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate panda_act
cd "$HOME/GitHub/mujoco-act-panda-lift"
project="$HOME/embodied_internship/mujoco-act-panda-lift"
checkpoint="$(cat evidence/v1/final/exec5-3100-3199-20261125-202416/checkpoint.txt)"
test -d "$checkpoint/pretrained_model" &&
HF_HUB_OFFLINE=1 PYTHONPATH="$project/external/lerobot-mujoco-imitation/src" \
python -u -m lerobot_mujoco.inference.policy_server \
    --checkpoint-dir "$checkpoint" \
    --socket-path "/tmp/panda-v1-final-$(id -u).sock" \
    --device cuda --action-steps 5
```

终端 B 使用相同 socket 和软重置入口，输出到新的复跑目录。

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate panda_sim
project="$HOME/embodied_internship/mujoco-act-panda-lift"
repo="$HOME/GitHub/mujoco-act-panda-lift"
out="$project/results/v1/replay-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
set -o pipefail
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
PYTHONPATH="$project/external/lerobot-mujoco-imitation/src:$repo/scripts" \
python -u - \
    --socket-path "/tmp/panda-v1-final-$(id -u).sock" \
    --seed-start 3100 --num-episodes 100 --max-episode-steps 300 \
    --output-csv "$out/exec5.csv" --video-dir "$out/videos" --record-episodes 20 \
    <<'PY' 2>&1 | tee "$out/eval.log"
import robosuite
from collect_panda_lift_soft_reset import make_with_soft_reset
from lerobot_mujoco.evaluation.closed_loop import main

robosuite.make = make_with_soft_reset
main()
PY
```

## 原始证据

- [逐回合 CSV](exec5.csv) 是服务器原件，100 条记录与此前由粘贴文本恢复的记录逐行一致。[原始评测日志](eval.log) 和 [评测汇总](eval-summary.txt) 均记录 37/100。
- [检查点路径](checkpoint.txt) 对应 [训练配置](../../training/v1_3000_20261125_194654/run_config.json) 和 [训练指标](../../training/v1_3000_20261125_194654/metrics.csv)。
- 执行步数消融原件包括 [执行 1 步](../../paired/v1_3000_20261125_194654-step_003000-1010-1019/exec1.csv)、[执行 5 步](../../paired/v1_3000_20261125_194654-step_003000-1010-1019/exec5.csv)、[逐种子配对表](../../paired/v1_3000_20261125_194654-step_003000-1010-1019/paired-comparison.csv)和 [消融汇总](../../paired/v1_3000_20261125_194654-step_003000-1010-1019/paired-summary.txt)。已核对两种执行长度下 10 个种子及方块初始坐标相符。
- [3107 成功回放](videos/seed_003107.mp4) 验证为 66 帧，[3111 失败回放](videos/seed_003111.mp4) 为 300 帧。画面中显示的种子与 CSV 一致。
