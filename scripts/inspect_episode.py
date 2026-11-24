import sys

import numpy as np


def check_alignment(images, states, actions, timestamps):
    lengths = {
        "images": len(images),
        "states": len(states),
        "actions": len(actions),
        "timestamps": len(timestamps),
    }
    # TODO 1：四个长度必须相等，且帧数大于 0。
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1 or lengths["actions"] == 0:
        raise ValueError(f"帧数不一致或回合为空:{lengths}")
    
    # 不满足时抛出 ValueError，并带上 lengths，方便定位缺了哪个字段。
    


def first_close_frame(actions):
    grip = actions[:, 3]
    # TODO 2：找到夹爪指令第一次从负数变成正数的位置。
    # 返回变化后的帧号。找不到时抛出 ValueError。
    transitions = (grip[:-1] < 0) & (grip[1:] > 0)
    indices = np.flatnonzero(transitions)
    if len(indices) == 0:
        raise ValueError("找不到夹爪从负指令切换到正指令的帧")
    return int(indices[0]) + 1

if __name__ == "__main__":
    with np.load(sys.argv[1], allow_pickle=False) as ep:
        actions = ep["actions"]
        if "--drop-last-action" in sys.argv[2:]:
            actions = actions[:-1]

        check_alignment(ep["images"], ep["states"], actions, ep["timestamps"])
        frame = first_close_frame(actions)
        print("闭合前一帧和闭合帧的夹爪指令：", actions[frame - 1:frame + 1, 3])
        z = ep["cube_positions"][:, 2]

        print("seed =", ep["seed"].item(), "frames =", len(actions))
        print("close_frame =", frame, "time_s =", float(ep["timestamps"][frame]))
        print("fsm_state =", ep["fsm_states"][frame])
        print("lift_m =", float(z[frame:].max() - z[frame]))