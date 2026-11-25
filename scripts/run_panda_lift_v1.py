"""Collect, train and evaluate Panda Lift v1 on the Ubuntu GPU server."""
import argparse
import csv
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

SPLITS = {"train": (0, 50), "val": (500, 10)}
EVAL_SEEDS = range(1010, 1020)
EVAL_PROGRAM = """
import robosuite
from collect_panda_lift_soft_reset import make_with_soft_reset
from lerobot_mujoco.evaluation.closed_loop import main
robosuite.make = make_with_soft_reset
main()
"""


def run_logged(command, log, env):
    print(f"Log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        with subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
        return process.returncode


def repo_id(split):
    return f"Y20040202/panda-lift-v1-{split}"


def episode_files(folder):
    return sorted(path for path in folder.glob("episode_*.npz")
                  if re.fullmatch(r"episode_[0-9]{6}\.npz", path.name))


def prepare_data(project, repo, conda, env, evidence):
    import numpy as np
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot_mujoco.datasets import convert_raw_episodes, validate_raw_episode

    raw = project / "data/raw/v1"
    output = project / "data/lerobot/v1"
    output.mkdir(parents=True, exist_ok=True)
    coverage = []
    panels = {}
    selected = {}
    frames = {}
    for split, (start, count) in SPLITS.items():
        folder = raw / split
        candidates = episode_files(folder)
        if len(candidates) < count:
            attempted = [int(path.stem[-6:]) for path in candidates]
            for previous in evidence.glob(f"{split}-*.csv"):
                with previous.open(encoding="utf-8", newline="") as handle:
                    attempted.extend(int(row["seed"]) for row in csv.DictReader(handle))
            next_seed = max([start - 1, *attempted]) + 1
            missing = count - len(candidates)
            manifest = evidence / f"{split}-{next_seed}-manifest.csv"
            command = [
                conda, "run", "--no-capture-output", "-n", "panda_sim", "python",
                str(repo / "scripts/collect_panda_lift_soft_reset.py"),
                "--output-dir", str(folder), "--manifest", str(manifest),
                "--seed-start", str(next_seed), "--num-episodes", str(missing), "--resume",
            ]
            log = evidence / f"{split}-{next_seed}.log"
            if run_logged(command, log, env):
                raise RuntimeError(f"Collection failed. See {log}")
            candidates = episode_files(folder)
        if len(candidates) < count:
            raise RuntimeError(f"{split}: {len(candidates)}/{count} successful files. "
                               "See the collection log. Rerun prepare to fill the gap.")
        selected[split] = candidates[:count]
        frames[split] = []
        panels[split] = []
        for path in selected[split]:
            length = validate_raw_episode(path)
            with np.load(path, allow_pickle=False) as episode:
                seed = int(episode["seed"].item())
                images = episode["images"]
                if not length or seed != int(path.stem[-6:]):
                    raise ValueError(f"Invalid seed or empty episode: {path}")
                if (images.max(axis=(1, 2, 3)) == 0).any():
                    raise ValueError(f"Black frame in {path}")
                if not np.isfinite(episode["states"]).all() or not np.isfinite(episode["actions"]).all():
                    raise ValueError(f"Non-finite state or action in {path}")
                panels[split].append(np.hstack(images[[0, length // 2, length - 1]]))
                x, y, z = episode["cube_positions"][0]
                coverage.append((split, seed, length, float(x), float(y), float(z), path.name))
            frames[split].append(length)

    seed_sets = [{row[1] for row in coverage if row[0] == split} for split in SPLITS]
    if seed_sets[0] & seed_sets[1] or (seed_sets[0] | seed_sets[1]) & (set(EVAL_SEEDS) | {2300, 2301, 2302}):
        raise ValueError("The train, validation, regression or paired evaluation seeds overlap.")

    for split, paths in selected.items():
        Image.fromarray(np.vstack(panels[split])).save(evidence / f"{split}-review.png")
        root = output / split
        if not root.exists():
            with tempfile.TemporaryDirectory(prefix=f"{split}-", dir=output) as temporary:
                staging = Path(temporary) / split
                convert_raw_episodes(raw_paths=paths, dataset_root=staging, repo_id=repo_id(split))
                staging.rename(root)
        dataset = LeRobotDataset(repo_id=repo_id(split), root=root)
        if dataset.meta.total_episodes != len(paths) or len(dataset) != sum(frames[split]):
            raise ValueError(f"Episode or frame count mismatch in {root}")
        offset = 0
        for index, (path, count) in enumerate(zip(paths, frames[split], strict=True)):
            sample = dataset[offset]
            if int(sample["episode_index"]) != index or int(sample["frame_index"]) != 0:
                raise ValueError(f"Episode order mismatch at {path}")
            if tuple(sample["observation.images.front"].shape) != (3, 96, 96):
                raise ValueError(f"Unexpected decoded image shape at {path}")
            with np.load(path, allow_pickle=False) as episode:
                np.testing.assert_allclose(sample["observation.state"], episode["states"][0])
                np.testing.assert_allclose(sample["action"], episode["actions"][0])
            offset += count
        print(f"PASS {split}: {len(paths)} episodes, {len(dataset)} frames")

    with (evidence / "initial-xy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "seed", "frames", "cube_x", "cube_y", "cube_z", "file"))
        writer.writerows(coverage)

    train_rows = [row for row in coverage if row[0] == "train"]
    first_frames = []
    for row in (min(train_rows, key=lambda item: item[3]), max(train_rows, key=lambda item: item[3])):
        with np.load(raw / "train" / row[-1], allow_pickle=False) as episode:
            first_frames.append(episode["images"][0])
        print(f"x extreme: seed={row[1]} x={row[3]:.4f} y={row[4]:.4f}")
    Image.fromarray(np.hstack(first_frames)).resize((768, 384), Image.Resampling.NEAREST).save(
        evidence / "train-x-extremes.png"
    )


def select_checkpoint(run):
    candidates = []
    with (run / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(row["step"])
            checkpoint = run / "checkpoints" / f"step_{step:06d}"
            if step > 0 and row["val_l1_loss"] and (checkpoint / "pretrained_model").is_dir():
                loss = float(row["val_l1_loss"])
                if math.isfinite(loss):
                    candidates.append((loss, step, checkpoint))
    if not candidates:
        raise RuntimeError(f"No saved checkpoint with finite validation L1 in {run}")
    return min(candidates)


def train(project, evidence, env, steps, batch_size, num_workers):
    datasets = project / "data/lerobot/v1"
    run = project / "checkpoints/act" / f"v1_{steps}_{datetime.now():%Y%m%d_%H%M%S}"
    command = [
        sys.executable, "-u", "-m", "lerobot_mujoco.training.train_act",
        "--train-dataset-root", str(datasets / "train"), "--train-repo-id", repo_id("train"),
        "--val-dataset-root", str(datasets / "val"), "--val-repo-id", repo_id("val"),
        "--run-dir", str(run), "--steps", str(steps), "--batch-size", str(batch_size),
        "--num-workers", str(num_workers), "--device", "cuda", "--seed", "0",
        "--log-every", "50", "--val-every", "500", "--val-batches", "0",
        "--checkpoint-every", "500",
    ]
    status = run_logged(command, evidence / f"{run.name}-train.log", env)
    checkpoint = run / "checkpoints" / f"step_{steps:06d}"
    if status or not (checkpoint / "pretrained_model").is_dir():
        raise RuntimeError(f"Training did not save its final checkpoint at {checkpoint}")
    for name in ("metrics.csv", "run_config.json"):
        shutil.copy2(run / name, evidence / f"{run.name}-{name}")
    loss, step, selected = select_checkpoint(run)
    (evidence / "latest-checkpoint.txt").write_text(str(selected) + "\n", encoding="utf-8")
    print(f"PASS training: {checkpoint}")
    print(f"Selected checkpoint: {selected} (step={step}, val_l1_loss={loss:.6f})")


def evaluate(project, repo, conda, env, checkpoint):
    name = f"{checkpoint.parents[1].name}-{checkpoint.name}-{datetime.now():%Y%m%d_%H%M%S}"
    evidence = repo / "evidence/v1/paired" / name
    results = project / "results/v1" / name
    evidence.mkdir(parents=True)
    runs = {}

    for horizon in (1, 5):
        server_log = evidence / f"exec{horizon}-server.log"
        result_csv = evidence / f"exec{horizon}.csv"
        video_dir = results / f"exec{horizon}"
        with tempfile.TemporaryDirectory(prefix="panda-act-") as temporary:
            socket = Path(temporary) / "policy.sock"
            command = [
                sys.executable, "-u", "-m", "lerobot_mujoco.inference.policy_server",
                "--checkpoint-dir", str(checkpoint), "--socket-path", str(socket),
                "--device", "cuda", "--action-steps", str(horizon),
            ]
            with server_log.open("w", encoding="utf-8") as handle:
                server = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
                try:
                    for _ in range(180):
                        if "status: ready" in server_log.read_text(encoding="utf-8", errors="replace"):
                            break
                        if server.poll() is not None:
                            raise RuntimeError(f"Policy server exited. See {server_log}")
                        time.sleep(1)
                    else:
                        raise RuntimeError(f"Policy server did not become ready. See {server_log}")
                    command = [
                        conda, "run", "--no-capture-output", "-n", "panda_sim",
                        "python", "-u", "-c", EVAL_PROGRAM, "--socket-path", str(socket),
                        "--seed-start", str(EVAL_SEEDS.start), "--num-episodes", str(len(EVAL_SEEDS)),
                        "--max-episode-steps", "300",
                        "--output-csv", str(result_csv), "--video-dir", str(video_dir),
                        "--record-episodes", str(len(EVAL_SEEDS)),
                    ]
                    if run_logged(command, evidence / f"exec{horizon}.log", env):
                        raise RuntimeError(f"Evaluation failed. See {evidence / f'exec{horizon}.log'}")
                finally:
                    if server.poll() is None:
                        server.send_signal(signal.SIGINT)
                        try:
                            server.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            server.terminate()
                            server.wait(timeout=10)

        with result_csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if [int(row["seed"]) for row in rows] != list(EVAL_SEEDS):
            raise ValueError(f"Missing or unexpected seed in {result_csv}")
        runs[horizon] = rows

    write_comparison(evidence, runs)
    report = [
        sys.executable, "-m", "lerobot_mujoco.evaluation.analyze_ablation",
        "--run", f"1={evidence / 'exec1.csv'}", "--run", f"5={evidence / 'exec5.csv'}",
        "--output", str(evidence / "action-step-comparison.md"),
    ]
    if run_logged(report, evidence / "analysis.log", env):
        raise RuntimeError(f"Paired report failed. See {evidence / 'analysis.log'}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        for horizon, rows in runs.items():
            directory = results / f"exec{horizon}"
            for outcome in ("True", "False"):
                match = next((row for row in rows if row["success"] == outcome), None)
                if match:
                    video = directory / f"seed_{int(match['seed']):06d}.mp4"
                    converted = video.with_name(f"{video.stem}_h264.mp4")
                    status = subprocess.run(
                        [ffmpeg, "-v", "error", "-n", "-i", str(video), "-an", "-c:v", "libx264",
                         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(converted)],
                        check=False,
                    )
                    if status.returncode == 0:
                        print(f"H.264 playback: {converted}")
    print(f"V1 tables: {evidence}")
    print(f"V1 videos: {results}")


def write_comparison(evidence, runs):
    fields = ("seed", "exec1_success", "exec5_success", "exec1_steps", "exec5_steps",
              "exec1_model_calls", "exec5_model_calls", "exec1_lift_m", "exec5_lift_m")
    pairs = []
    for first, fifth in zip(runs[1], runs[5], strict=True):
        if first["seed"] != fifth["seed"] or any(
            not math.isclose(float(first[f"initial_cube_{axis}"]), float(fifth[f"initial_cube_{axis}"]),
                             abs_tol=1e-6, rel_tol=0) for axis in "xyz"
        ):
            raise ValueError(f"Unmatched initial states for seed {first['seed']}")
        steps = [int(row["steps"]) for row in (first, fifth)]
        lift = [float(row["maximum_cube_height"]) - float(row["initial_cube_z"])
                for row in (first, fifth)]
        pairs.append((first["seed"], first["success"], fifth["success"], *steps,
                      steps[0], (steps[1] + 4) // 5, *lift))
    with (evidence / "paired-comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(pairs)
    lines = []
    for index, horizon in enumerate((1, 5)):
        success = sum(row[index + 1] == "True" for row in pairs)
        calls = sum(row[index + 5] for row in pairs)
        lines.append(f"exec{horizon}: success={success}/{len(pairs)}, model_calls_from_action_queue={calls}")
    (evidence / "paired-summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("steps and batch-size must be positive, and num-workers must be non-negative")

    project = Path.home() / "embodied_internship/mujoco-act-panda-lift"
    repo = Path(__file__).resolve().parents[1]
    upstream = project / "external/lerobot-mujoco-imitation/src"
    if not (repo / "scripts/collect_panda_lift_soft_reset.py").is_file() or not upstream.is_dir():
        parser.error("Place this script in the GitHub repo's scripts/ folder and check the existing upstream checkout.")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(upstream), str(repo / "scripts"), env.get("PYTHONPATH", ""))).rstrip(os.pathsep)
    env.update(HF_HUB_OFFLINE="1", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    sys.path.insert(0, str(upstream))
    conda = env.get("CONDA_EXE", str(Path.home() / "miniconda3/bin/conda"))
    evidence = repo / "evidence/v1"
    evidence.mkdir(parents=True, exist_ok=True)

    if args.stage == "prepare":
        prepare_data(project, repo, conda, env, evidence)
    elif args.stage == "train":
        train(project, evidence, env, args.steps, args.batch_size, args.num_workers)
    else:
        recorded = evidence / "latest-checkpoint.txt"
        if args.checkpoint is None and not recorded.is_file():
            parser.error("Run the train stage first, or give the checkpoint directory with --checkpoint.")
        checkpoint = args.checkpoint or Path(recorded.read_text(encoding="utf-8").strip())
        if not (checkpoint / "pretrained_model").is_dir():
            parser.error(f"Missing checkpoint files at {checkpoint}")
        evaluate(project, repo, conda, env, checkpoint.resolve())


if __name__ == "__main__":
    main()
