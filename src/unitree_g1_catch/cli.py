"""CLI wrappers that register the external task before delegating to mjlab."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from unitree_g1_catch import (
  task_id,
  experiment_name,
  register_tasks,
)

CACHE_ROOT = Path(__file__).resolve().parents[2] / ".cache"


def _configure_cache_paths() -> None:
  os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
  import warp as wp

  wp.config.kernel_cache_dir = str(CACHE_ROOT / "warp")


def train_main() -> None:
  _configure_cache_paths()
  register_tasks()
  from mjlab.scripts.train import main

  main()


def play_main() -> None:
  _configure_cache_paths()
  register_tasks()
  from mjlab.scripts.play import main

  main()


def train_pipeline_main() -> None:
  """Train phase 1, then warm-start phase 2 from its final checkpoint."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--motion-file", required=True, type=Path)
  parser.add_argument("--phase-1-iterations", type=int, default=20_000)
  parser.add_argument("--phase-2-iterations", type=int, default=20_000)
  parser.add_argument("--num-envs", type=int, default=2_048)
  parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
  parser.add_argument("--run-name", default="pipeline")
  parser.add_argument("--inspire", action="store_true", help="Use Inspire hand in the environment")
  parser.add_argument("--obs-version", type=int, default=0, help="Observation version")
  parser.add_argument("--reward-version", type=int, default=0, help="Reward version")
  args = parser.parse_args()

  motion_file = args.motion_file.expanduser().resolve()
  if not motion_file.is_file():
    parser.error(f"motion file does not exist: {motion_file}")
  if args.phase_1_iterations <= 0 or args.phase_2_iterations <= 0:
    parser.error("iteration counts must be positive")
  if args.num_envs <= 0:
    parser.error("num-envs must be positive")

  train_executable = Path(sys.executable).with_name("g1-catch-train")
  if not train_executable.is_file():
    parser.error(f"training executable does not exist: {train_executable}")

  log_root = args.log_root.expanduser().resolve()
  flat_task_id = task_id(throw=False, inspire=args.inspire, obs_ver=args.obs_version, reward_ver=args.reward_version)
  throw_task_id = task_id(throw=True, inspire=args.inspire, obs_ver=args.obs_version, reward_ver=args.reward_version)

  phase_1_root = log_root / experiment_name(throw=False, inspire=args.inspire, obs_ver=args.obs_version, reward_ver=args.reward_version)
  previous_runs = set(phase_1_root.iterdir()) if phase_1_root.exists() else set()
  phase_1_name = f"{args.run_name}-phase1"
  subprocess.run(
    [
      train_executable,
      flat_task_id,
      "--env.commands.motion.motion-file",
      motion_file,
      "--env.scene.num-envs",
      str(args.num_envs),
      "--agent.max-iterations",
      str(args.phase_1_iterations),
      "--agent.run-name",
      phase_1_name,
      "--agent.logger",
      "tensorboard",
      "--log-root",
      log_root,
    ],
    check=True,
  )

  phase_1_runs = [
    path
    for path in phase_1_root.iterdir()
    if path not in previous_runs and path.name.endswith(f"_{phase_1_name}")
  ]
  if len(phase_1_runs) != 1:
    raise RuntimeError(f"Expected one new phase-1 run, found {len(phase_1_runs)}")
  checkpoints = list(phase_1_runs[0].glob("model_*.pt"))
  if not checkpoints:
    raise RuntimeError(f"No phase-1 checkpoint found in {phase_1_runs[0]}")
  checkpoint = max(checkpoints, key=lambda path: int(path.stem.removeprefix("model_")))
  print(f"[INFO] Starting phase 2 from: {checkpoint}")

  subprocess.run(
    [
      train_executable,
      throw_task_id,
      "--agent.pretrained-actor-checkpoint",
      checkpoint,
      "--env.commands.throw.motion-cfg.motion-file",
      motion_file,
      "--env.scene.num-envs",
      str(args.num_envs),
      "--agent.max-iterations",
      str(args.phase_2_iterations),
      "--agent.run-name",
      f"{args.run_name}-phase2",
      "--agent.logger",
      "tensorboard",
      "--log-root",
      log_root,
    ],
    check=True,
  )
