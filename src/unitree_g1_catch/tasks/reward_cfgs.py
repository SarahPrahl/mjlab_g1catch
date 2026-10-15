from mjlab.managers import RewardTermCfg, SceneEntityCfg
from mjlab.tasks.tracking import mdp as tracking_mdp

from unitree_g1_catch import mdp as catch_mdp
from unitree_g1_catch.assets import Humanoid

# Additional rewards for imitation learning & weights
REWARD_TERMS_WEIGHTS_CATCH = [
  # v0
  {
    "ball_position_tracking": 1.0,
    "ball_velocity_tracking": 0.25,
    "palm_ball_proximity": 1.0,
    "ball_contact": 2.0,
  },
  # v1
  {
    "ball_position_tracking": 1.0,
    "ball_velocity_tracking": 0.25,
    "palm_ball_proximity": 1.0,
    "ball_contact": 2.0,
  },
]

# Rewards for RL + weights
REWARD_TERMS_WEIGHTS_THROW = [
  # v0
  {
    "idle_posture": 2.0,
    "ball_released": 2.0,
    "standing_posture": 3.5,
    "base_motion": -1.0,
    "controlled_arm_posture": 2.0,
    "arm_motion": -2.0,
    "palm_contact": 4.0,
    "interception": 1.5,
    "catch_pose": 2.0,
    "stable_grasp": 5.0,
    "catch_success": 10.0,
    "body_ball_contact": -3.0,
    "joint_torques_l2": -1.0e-5,
    "joint_limit": -10.0,
    "self_collisions": -10.0,
    "finger_posture": 1.0,
    "finger_contact": 4.0,
  },
  # v1
  {
    "idle_posture": 2.0,
    "ball_released": 2.0,
    "standing_posture": 3.5,
    "base_motion": -1.0,
    "controlled_arm_posture": 2.0,
    "arm_motion": -2.0,
    "palm_contact": 4.0,
    "interception": 1.5,
    "catch_pose": 2.0,
    "stable_grasp": 5.0,
    "catch_success": 10.0,
    "body_ball_contact": -3.0,
    "joint_torques_l2": -1.0e-4,
    "joint_limit": -10.0,
    "self_collisions": -10.0,
    "finger_posture": 1.0,
    "finger_contact": 4.0,
  },
]

# Rewards to keep from IL for RL
REHERSAL_REWARD_NAMES = [
  # v0
  [
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
    "ball_position_tracking",
    "ball_velocity_tracking",
    "palm_ball_proximity",
    "ball_contact",
  ],
  # v1
  [
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
    "ball_position_tracking",
    "ball_velocity_tracking",
    "palm_ball_proximity",
    "ball_contact",
  ],
]


def all_rewards_dict(
  command_name: str,
  humanoid: Humanoid,
) -> dict[str, RewardTermCfg]:
  """
  Return a dictionary with all Reward Term Configurations possibly used for catching and random throws.
  """
  palm_cfg = (
    SceneEntityCfg("robot", site_names=humanoid.palm_sites)
    if humanoid.palm_sites
    else SceneEntityCfg("robot", geom_names=humanoid.palm_collisions)
  )
  body_joint_cfg = SceneEntityCfg("robot", joint_names=humanoid.actuated_joints)
  arm_cfg = SceneEntityCfg("robot", joint_names=humanoid.arm_joints)
  arm_ready_cfg = SceneEntityCfg("robot", joint_names=humanoid.arm_ready_joints)
  elbow_cfg = SceneEntityCfg("robot", joint_names=humanoid.elbow_joints)
  finger_cfg = SceneEntityCfg("robot", joint_names=humanoid.finger_joints)
  lower_body_cfg = SceneEntityCfg("robot", joint_names=humanoid.lower_body_joints)

  return {
    # How well the ball position fits to the motion data
    "ball_position_tracking": RewardTermCfg(
      func=catch_mdp.ball_position_tracking_exp,
      weight=1.0,
      params={"command_name": command_name, "std": 0.25},
    ),
    # How well the ball velocity fits to the motion data
    "ball_velocity_tracking": RewardTermCfg(
      func=catch_mdp.ball_velocity_tracking_exp,
      weight=0.25,
      params={"command_name": command_name, "std": 1.5},
    ),
    # Minimize distance of both palms to the ball
    "palm_ball_proximity": RewardTermCfg(
      func=catch_mdp.palm_ball_proximity_exp,
      weight=1.0,
      params={
        "command_name": command_name,
        "std": 0.3,
        "asset_cfg": palm_cfg,
      },
    ),
    # Reward contact of both hands to the ball
    "ball_contact": RewardTermCfg(
      func=catch_mdp.ball_contact_reward,
      weight=2.0,
      params={
        "command_name": command_name,
        "left_sensor_name": "left_ball_contact",
        "right_sensor_name": "right_ball_contact",
      },
    ),
    # Reward default pose
    "idle_posture": RewardTermCfg(
      func=catch_mdp.idle_posture_exp,
      weight=2.0,
      params={
        "command_name": command_name,
        "asset_cfg": arm_cfg,
        "posture_std": 0.4,
        "velocity_std": 1.0,
      },
    ),
    # Reward releasing the ball after catching
    "ball_released": RewardTermCfg(
      func=catch_mdp.ball_released_reward,
      weight=2.0,
      params={"command_name": command_name},
    ),
    # Reward upright standing posture
    "standing_posture": RewardTermCfg(
      func=catch_mdp.standing_posture_exp,
      weight=3.5,
      params={
        "command_name": command_name,
        "target_height_m": 0.76,
        "tilt_std": 0.35,
        "height_std": 0.12,
        "posture_std": 0.7,
        "asset_cfg": lower_body_cfg,
      },
    ),
    # Penalize motion of the robots base/root
    "base_motion": RewardTermCfg(
      func=catch_mdp.base_motion_l2,
      weight=-1.0,
      params={
        "command_name": command_name,
        "position_std": 0.2,
        "velocity_std": 0.3,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    # Keep arms quiet and extended before interception and while holding.
    "controlled_arm_posture": RewardTermCfg(
      func=catch_mdp.controlled_arm_posture_exp,
      weight=2.0,
      params={
        "command_name": command_name,
        "ready_std": 0.4,
        "elbow_target_rad": 0.25,
        "elbow_std": 0.4,
        "velocity_std": 1.0,
        "ready_cfg": arm_ready_cfg,
        "elbow_cfg": elbow_cfg,
        "arm_cfg": arm_cfg,
      },
    ),
    # Penalize exceeding arm motion
    "arm_motion": RewardTermCfg(
      func=catch_mdp.arm_motion_l2,
      weight=-2.0,
      params={
        "command_name": command_name,
        "velocity_limit_rad_s": 2.0,
        "acceleration_limit_rad_s2": 20.0,
        "asset_cfg": arm_cfg,
      },
    ),
    # Reward palm-face contacts whose normals point into the two-hand grasp.
    "palm_contact": RewardTermCfg(
      func=catch_mdp.palm_contact_reward,
      weight=4.0,
      params={"command_name": command_name},
    ),
    # Bring the palms to the ball, with an extra bonus for velocity matching.
    "interception": RewardTermCfg(
      func=catch_mdp.interception_exp,
      weight=1.5,
      params={
        "command_name": command_name,
        "speed_std": 0.5,
        "distance_std": 0.35,
      },
    ),
    # Reward an upright, demonstration-like catch with the palms below the ball.
    "catch_pose": RewardTermCfg(
      func=catch_mdp.catch_pose_reward,
      weight=2.0,
      params={"command_name": command_name},
    ),
    # Reward physically containing and slowing the ball after palm contact.
    "stable_grasp": RewardTermCfg(
      func=catch_mdp.stable_grasp_reward,
      weight=5.0,
      params={"command_name": command_name},
    ),
    # One success pulse per episode, even though holding/release continue.
    "catch_success": RewardTermCfg(
      func=catch_mdp.catch_success_reward,
      weight=10.0,
      params={"command_name": command_name},
    ),
    # Penalize the ball touching any robot body other than the hands
    "body_ball_contact": RewardTermCfg(
      func=catch_mdp.body_ball_contact_penalty,
      weight=-3.0,
      params={"command_name": command_name, "sensor_name": "body_ball_contact"},
    ),
    # Penalize joint torques applied on the articulation using L2 squared kernel.
    "joint_torques_l2": RewardTermCfg(
      func=tracking_mdp.joint_torques_l2,
      weight=-1.0e-5,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=humanoid.joint_names)
      },
    ),
    # Penalize joint positions if they cross the soft limits.
    "joint_limit": RewardTermCfg(
      func=tracking_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": body_joint_cfg},
    ),
    # Penaliue self-collisions
    "self_collisions": RewardTermCfg(
      func=tracking_mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    # Reward finger joint angles close to the predefined grip posture
    "finger_posture": RewardTermCfg(
      func=catch_mdp.finger_posture_exp,
      weight=1.0,
      params={
        "command_name": command_name,
        "open_std": 0.15,
        "grip_std": 0.25,
        "velocity_limit_rad_s": 1.0,
        "velocity_std": 0.5,
        "grip_targets": humanoid.finger_grip_targets,
        "asset_cfg": finger_cfg,
      },
    ),
    # Reward distal finger contacts while both true palms hold the ball in-box
    "finger_contact": RewardTermCfg(
      func=catch_mdp.finger_contact_reward,
      weight=4.0,
      params={
        "command_name": command_name,
        "left_sensor_name": "left_finger_ball_contact",
        "right_sensor_name": "right_finger_ball_contact",
        "contacts_per_hand": 2,
      },
    ),
  }


def catch_reward_cfg(
  command_name: str,
  humanoid: Humanoid,
  version: int = 0,
  keep_prev: bool = False,
  play: bool = False,
  prev_rewards: dict[str, RewardTermCfg] | None = None,
) -> dict[str, RewardTermCfg]:
  """
  Returns the full reward configuration as a dictionary of reward names and RewardTermCfg.
  Params:
      command_name: Name of the current task command.
      humanoid: Instance of the used Humanoid Class.
      version: Integer of the desired version of reward configuration.
      keep_prev: Whether to keep the rewards from prev_rewards.
      play: If the tasked is replayed instead of trained.
      prev_rewards: Dictionary of RewardTermCfg to retain or extract rehersal rewards.
  """

  prev_rewards = {} if prev_rewards is None else prev_rewards

  rewards = {}
  if keep_prev:
    rewards.update(prev_rewards)
  # Add all reward terms of the desired reward version
  rewards.update(
    _reward_terms(
      command_name=command_name,
      humanoid=humanoid,
      version=version,
    )
  )

  # For RL training with diverse throws keep the defined rehersal rewards from the previous rewards
  if command_name == "throw" and not play:
    rewards.update(
      _rehersal_reward_terms(
        command_name=command_name, version=version, prev_rewards=prev_rewards
      )
    )

  return rewards


def _reward_terms(
  command_name: str,
  humanoid: Humanoid,
  version: int = 0,
) -> dict[str, RewardTermCfg]:
  """Return the Reward Configuration based on the desired version."""
  # Get all possible rewards
  all_rewards = all_rewards_dict(
    command_name=command_name,
    humanoid=humanoid,
  )

  reward_terms = dict(
    REWARD_TERMS_WEIGHTS_THROW[version]
    if command_name == "throw"
    else REWARD_TERMS_WEIGHTS_CATCH[version]
  )

  # Remove rewards requiring finger joints, if the robot does not have finger joints
  if not humanoid.finger_joints:
    reward_terms.pop("finger_posture", None)
    reward_terms.pop("finger_contact", None)

  # update the weights of each reward term
  for key, weight in reward_terms.items():
    all_rewards[key].weight = weight

  # extract the desired terms
  return {key: all_rewards[key] for key in reward_terms.keys()}


def _rehersal_reward_terms(
  command_name: str,
  version: int = 0,
  prev_rewards: dict[str, RewardTermCfg] | None = None,
) -> dict[str, RewardTermCfg]:
  """
  Return specified rewards that are used during IL. IL Rewards are contained in prev_rewards.
  """

  if prev_rewards is None:
    prev_rewards = {}
  rehersal_rewards = {}

  # Add all desired rehersal rewards as rehersal_reward with the parameters of the previous rewards
  for name in REHERSAL_REWARD_NAMES[version]:
    previous_term = prev_rewards[name]
    reward_params = {
      key: value for key, value in previous_term.params.items() if key != "command_name"
    }
    rehersal_rewards[f"rehersal_{name}"] = RewardTermCfg(
      func=catch_mdp.rehearsal_reward,
      weight=previous_term.weight,
      params={
        "command_name": command_name,
        "reward_func": previous_term.func,
        **reward_params,
      },
    )
  return rehersal_rewards
