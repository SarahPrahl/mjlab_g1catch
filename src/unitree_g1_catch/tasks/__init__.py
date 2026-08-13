"""Register the G1 + Inspire reactive catching task with mjlab."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from unitree_g1_catch import TASK_ID

from .env_cfg import unitree_g1_inspire_catch_env_cfg
from .rl_cfg import unitree_g1_inspire_catch_ppo_runner_cfg

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=unitree_g1_inspire_catch_env_cfg(),
  play_env_cfg=unitree_g1_inspire_catch_env_cfg(play=True),
  rl_cfg=unitree_g1_inspire_catch_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
