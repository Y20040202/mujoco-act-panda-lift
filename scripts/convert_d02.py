import csv
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_mujoco.datasets import convert_raw_episodes

project = Path.home() / "embodied_internship/mujoco-act-panda-lift"
repo = Path(__file__).resolve().parents[1]

with (repo / "evidence/d02/episode-index.csv").open(encoding="utf-8") as file:
    index_rows = list(csv.DictReader(file))

for split, expected in (("train", 5), ("val", 2)):
    rows = [row for row in index_rows if row["split"] == split]
    assert len(rows) == expected, (split, len(rows), expected)

    paths = [
        project / f"data/raw/d02/{split}/episode_{int(row['seed']):06d}.npz"
        for row in rows
    ]
    root = project / f"data/lerobot/d02/{split}"
    repo_id = f"Y20040202/panda-lift-d02-{split}"

    if not root.exists():
        convert_raw_episodes(raw_paths=paths, dataset_root=root, repo_id=repo_id)

    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    assert dataset.meta.total_episodes == expected
    assert len(dataset) == sum(int(row["frames"]) for row in rows)

    offset = 0
    for row, path in zip(rows, paths, strict=True):
        frame = int(row["close_frame"])
        global_index = offset + frame
        sample = dataset[global_index]

        assert int(sample["episode_index"]) == int(row["episode_index"])
        assert int(sample["frame_index"]) == frame
        assert tuple(sample["observation.images.front"].shape) == (3, 96, 96)

        with np.load(path, allow_pickle=False) as ep:
            np.testing.assert_allclose(sample["action"], ep["actions"][frame])
            np.testing.assert_allclose(
                sample["observation.state"], ep["states"][frame]
            )
            np.testing.assert_allclose(
                float(sample["timestamp"]), ep["timestamps"][frame], atol=1e-6
            )

        print(f"PASS {split} seed={row['seed']} frame={frame} global_index={global_index}")
        offset += int(row["frames"])

    print(f"{split}: {expected} episodes, {len(dataset)} frames")