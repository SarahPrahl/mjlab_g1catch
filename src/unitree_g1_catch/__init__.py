"""Unitree G1 reactive ball-catching task for mjlab."""

TASK_ID = "Mjlab-Catch-Flat-Unitree-G1-Inspire"


def register_tasks() -> None:
  """Register this package's tasks after mjlab's built-in tasks."""
  import mjlab.tasks  # noqa: F401

  import unitree_g1_catch.tasks  # noqa: F401


__all__ = ["TASK_ID", "register_tasks"]
