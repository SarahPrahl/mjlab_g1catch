"""Unitree G1 reactive ball-catching task for mjlab."""

def register_tasks() -> None:
  """Register this package's tasks after mjlab's built-in tasks."""
  import mjlab.tasks  # noqa: F401

  import unitree_g1_catch.tasks  # noqa: F401

def task_id(throw: bool = False, inspire: bool = False, obs_ver: int = 0, reward_ver: int = 0) -> str:
  """Return the task ID for the given configuration."""
  return f"Mjlab-Catch-{'Throws-' if throw else 'Flat-'}Unitree-G1-{'Inspire-' if inspire else ''}o{obs_ver}-r{reward_ver}"

def experiment_name(throw: bool = False, inspire: bool = False, obs_ver: int = 0, reward_ver: int = 0) -> str:
  """Return the experiment name for the given configuration."""
  return f"g1_{'inspire_' if inspire else ''}{'throw_' if throw else ''}catch_o{obs_ver}_r{reward_ver}"

__all__ = [
  "register_tasks",
  "task_id",
  "experiment_name",
]
