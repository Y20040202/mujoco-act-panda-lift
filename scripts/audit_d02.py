import csv
from pathlib import Path

import numpy as np
from PIL import Image

from inspect_episode import check_alignment, first_close_frame

raw = Path.home() / "embodied_internship/mujoco-act-panda-lift/data/raw/d02"
evidence = Path(__file__).resolve().parents[1] / "evidence" / "d02"
rows, panels = [], []

for split, seeds in (("train", range(2100, 2105)), ("val", range(2200, 2202))):
    for index, seed in enumerate(seeds):
        path = raw / split / f"episode_{seed:06d}.npz"

        with np.load(path, allow_pickle=False) as ep:
            images, states = ep["images"], ep["states"]
            actions, times = ep["actions"], ep["timestamps"]
            n = len(actions)

            check_alignment(images, states, actions, times)
            assert ep["seed"].item() == seed, path
            assert ep["success"].item() and ep["steps"].item() == n, path
            assert images.shape == (n, 96, 96, 3), path
            assert states.shape == (n, 12) and actions.shape == (n, 4), path
            assert np.isfinite(states).all() and np.isfinite(actions).all(), path
            np.testing.assert_allclose(times, np.arange(n) / 20, rtol=0, atol=1e-6)
            assert (images.max(axis=(1, 2, 3)) > 0).all(), f"发现全黑帧：{path}"

            frame = first_close_frame(actions)
            z = ep["cube_positions"][:, 2]
            lift = float(z[frame:].max() - z[frame])

            rows.append((split, index, seed, n, frame, float(times[frame]), lift))
            panels.append(np.hstack(images[[0, frame, n - 1]]))
            print(f"{split} seed={seed} frames={n} close={frame} lift={lift:.3f} m")

evidence.mkdir(parents=True, exist_ok=True)
with (evidence / "episode-index.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(("split", "episode_index", "seed", "frames",
                     "close_frame", "close_time_s", "lift_delta_m"))
    writer.writerows(rows)

Image.fromarray(np.vstack(panels)).save(evidence / "rgb-panel.png")