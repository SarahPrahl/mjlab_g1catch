"""Register the G1 reactive catching tasks with mjlab."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from unitree_g1_catch import task_id

from .env_cfg import (
  unitree_g1_catch_env_cfg,
  unitree_g1_throw_env_cfg,
)
from .obs_cfgs import ACTOR_OBSERVATION_TERMS
from .reward_cfgs import REWARD_TERMS_WEIGHTS_CATCH
from .rl_cfg import (
  unitree_g1_catch_ppo_runner_cfg,
  unitree_g1_throw_ppo_runner_cfg,
)
from .runner import ThrowOnPolicyRunner

for inspire in (True, False):
  for obs_version in range(len(ACTOR_OBSERVATION_TERMS)):
    for reward_version in range(len(REWARD_TERMS_WEIGHTS_CATCH)):
      # Imitation learning task for the G1 Inspire hand, used for warm-starting the RL task.
      register_mjlab_task(
        task_id=task_id(
          throw=False, inspire=inspire, obs_ver=obs_version, reward_ver=reward_version
        ),
        env_cfg=unitree_g1_catch_env_cfg(
          inspire=inspire, obs_version=obs_version, reward_version=reward_version
        ),
        play_env_cfg=unitree_g1_catch_env_cfg(
          play=True,
          inspire=inspire,
          obs_version=obs_version,
          reward_version=reward_version,
        ),
        rl_cfg=unitree_g1_catch_ppo_runner_cfg(
          inspire=inspire, obs_ver=obs_version, reward_ver=reward_version
        ),  # TODO
        runner_cls=MotionTrackingOnPolicyRunner,
      )

      # RL task for the G1 Inspire hand, which is the main task for this package.
      register_mjlab_task(
        task_id=task_id(
          throw=True, inspire=inspire, obs_ver=obs_version, reward_ver=reward_version
        ),
        env_cfg=unitree_g1_throw_env_cfg(
          inspire=inspire, obs_version=obs_version, reward_version=reward_version
        ),
        play_env_cfg=unitree_g1_throw_env_cfg(
          play=True,
          inspire=inspire,
          obs_version=obs_version,
          reward_version=reward_version,
        ),
        rl_cfg=unitree_g1_throw_ppo_runner_cfg(
          inspire=inspire, obs_ver=obs_version, reward_ver=reward_version
        ),  # TODO
        runner_cls=ThrowOnPolicyRunner,
      )
