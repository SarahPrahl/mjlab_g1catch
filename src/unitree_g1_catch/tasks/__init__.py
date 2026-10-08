"""Register the G1 reactive catching tasks with mjlab."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from unitree_g1_catch import (
  TASK_ID,
  TASK_ID_STANDARD,
  THROW_TASK_ID,
  THROW_TASK_ID_STANDARD,
)

from .env_cfg import (
  unitree_g1_catch_env_cfg,
  unitree_g1_throw_env_cfg,
)
from .rl_cfg import (
  unitree_g1_inspire_catch_ppo_runner_cfg,
  unitree_g1_inspire_throw_ppo_runner_cfg,
)
from .runner import ThrowOnPolicyRunner

# Imitation learning task for the G1 Inspire hand, used for warm-starting the RL task.
register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=unitree_g1_catch_env_cfg(inspire=True),
  play_env_cfg=unitree_g1_catch_env_cfg(play=True, inspire=True),
  rl_cfg=unitree_g1_inspire_catch_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# Imitation learning task for the standard G1, used for warm-starting the RL task.
register_mjlab_task(
  task_id=TASK_ID_STANDARD,
  env_cfg=unitree_g1_catch_env_cfg(inspire=False),
  play_env_cfg=unitree_g1_catch_env_cfg(play=True, inspire=False),
  rl_cfg=unitree_g1_inspire_catch_ppo_runner_cfg(), # TODO: check
  runner_cls=MotionTrackingOnPolicyRunner,
)

# RL task for the G1 Inspire hand, which is the main task for this package.
register_mjlab_task(
  task_id=THROW_TASK_ID,
  env_cfg=unitree_g1_throw_env_cfg(inspire=True),
  play_env_cfg=unitree_g1_throw_env_cfg(play=True, inspire=True),
  rl_cfg=unitree_g1_inspire_throw_ppo_runner_cfg(),
  runner_cls=ThrowOnPolicyRunner,
)

# RL task for the standard G1, which is the main task for this package.
register_mjlab_task(
  task_id=THROW_TASK_ID_STANDARD,
  env_cfg=unitree_g1_throw_env_cfg(inspire=False),
  play_env_cfg=unitree_g1_throw_env_cfg(play=True, inspire=False),
  rl_cfg=unitree_g1_inspire_throw_ppo_runner_cfg(),
  runner_cls=ThrowOnPolicyRunner,
)
