"""PPO configuration for reactive G1 + Inspire ball catching."""

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from unitree_g1_catch import experiment_name


@dataclass
class ThrowOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  """PPO config with an optional actor-only warm start."""

  pretrained_actor_checkpoint: str | None = None


def unitree_g1_catch_ppo_runner_cfg(
  inspire: bool = True, obs_ver: int = 0, reward_ver: int = 0
) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.8,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name(
      throw=False, inspire=inspire, obs_ver=obs_ver, reward_ver=reward_ver
    ),
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def unitree_g1_throw_ppo_runner_cfg(
  inspire: bool = True, obs_ver: int = 0, reward_ver: int = 0
) -> ThrowOnPolicyRunnerCfg:
  return ThrowOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.003,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name(
      throw=True, inspire=inspire, obs_ver=obs_ver, reward_ver=reward_ver
    ),
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
