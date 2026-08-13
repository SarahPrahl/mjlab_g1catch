"""CLI wrappers that register the external task before delegating to mjlab."""

import os
from pathlib import Path

from unitree_g1_catch import register_tasks

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
