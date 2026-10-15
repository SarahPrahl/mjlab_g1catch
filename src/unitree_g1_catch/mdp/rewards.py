"""Reward terms for ball-conditioned G1 tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.string import resolve_matching_names_values

from .commands import BallThrowCommand, CatchMotionCommand, RehearsalThrowCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import RewardTermCfg


def _command(
  env: ManagerBasedRlEnv, command_name: str
) -> CatchMotionCommand | BallThrowCommand:
  return cast(
    CatchMotionCommand | BallThrowCommand,
    env.command_manager.get_term(command_name),
  )


def rehearsal_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  reward_func: Any,
  **reward_params: Any,
) -> torch.Tensor:
  """Apply an existing phase-one reward only to rehearsal environments."""
  command = _command(env, command_name)
  assert isinstance(command, RehearsalThrowCommand)
  value = reward_func(env, command_name=command_name, **reward_params)
  return value * command.rehearsal_weight


def _throw_command(env: ManagerBasedRlEnv, command_name: str) -> BallThrowCommand:
  command = _command(env, command_name)
  assert isinstance(command, BallThrowCommand)
  return command


def ball_position_tracking_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Track the recorded ball trajectory after it becomes visible."""
  command = _command(env, command_name)
  assert isinstance(command, (CatchMotionCommand, RehearsalThrowCommand))
  error = torch.sum(
    torch.square(command.ball_reference_pos_w - command.ball_pos_w), dim=-1
  )
  return torch.exp(-error / std**2) * command.ball_visible.float()


def ball_velocity_tracking_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Track the recorded ball linear velocity after it becomes visible."""
  command = _command(env, command_name)
  assert isinstance(command, (CatchMotionCommand, RehearsalThrowCommand))
  error = torch.sum(
    torch.square(command.ball_reference_lin_vel_w - command.ball_lin_vel_w), dim=-1
  )
  return torch.exp(-error / std**2) * command.ball_visible.float()


def palm_ball_proximity_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Bring both palm centers toward the physical ball."""
  command = _command(env, command_name)
  robot: Entity = env.scene[asset_cfg.name]
  if asset_cfg.site_names is not None:
    palm_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids]
  elif asset_cfg.geom_names is not None:
    palm_pos_w = robot.data.geom_pos_w[:, asset_cfg.geom_ids]
  else:
    raise ValueError("palm_ball_proximity_exp requires either site_names or geom_names")
  error = torch.sum(torch.square(palm_pos_w - command.ball_pos_w[:, None, :]), dim=-1)
  return torch.exp(-error.mean(dim=-1) / std**2) * command.ball_visible.float()


def ball_contact_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  left_sensor_name: str,
  right_sensor_name: str,
) -> torch.Tensor:
  """Reward simultaneous contact between the ball and both hands."""
  command = _command(env, command_name)
  left: ContactSensor = env.scene[left_sensor_name]
  right: ContactSensor = env.scene[right_sensor_name]
  assert left.data.found is not None and right.data.found is not None
  left_contact = left.data.found.any(dim=-1).float()
  right_contact = right.data.found.any(dim=-1).float()
  contact_score = 0.25 * (left_contact + right_contact) + 0.5 * (
    left_contact * right_contact
  )
  return contact_score * command.ball_visible.float()


def palm_contact_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Reward palm-face contacts whose normals point into the two-hand grasp."""
  command = _throw_command(env, command_name)
  score = 0.5 * (command.left_palm_contact.float() + command.right_palm_contact.float())
  return (
    score
    * command.ball_in_catch_zone.float()
    * command.catch_pose_score
    * command.synthetic_weight
  )


def finger_contact_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  left_sensor_name: str,
  right_sensor_name: str,
  contacts_per_hand: int,
) -> torch.Tensor:
  """Reward distal finger contacts while both true palms hold the ball in-box."""
  command = _throw_command(env, command_name)
  left: ContactSensor = env.scene[left_sensor_name]
  right: ContactSensor = env.scene[right_sensor_name]
  assert left.data.found is not None and right.data.found is not None
  left_score = torch.clamp(
    (left.data.found > 0).sum(dim=-1).float() / contacts_per_hand, max=1.0
  )
  right_score = torch.clamp(
    (right.data.found > 0).sum(dim=-1).float() / contacts_per_hand, max=1.0
  )
  score = 0.5 * (left_score + right_score)
  return (
    score
    * (command.left_palm_contact & command.right_palm_contact).float()
    * command.stable_grasp.float()
    * command.catch_pose_score
    * command.synthetic_weight
  )


class finger_posture_exp:
  """Open for interception, then close gently toward a recorded grip pose."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    robot: Entity = env.scene[cfg.params["asset_cfg"].name]
    grip_target_patterns = tuple(cfg.params["grip_targets"].keys())
    joint_ids, joint_names = robot.find_joints(
      grip_target_patterns, preserve_order=True
    )
    _, _, targets = resolve_matching_names_values(
      data=cfg.params["grip_targets"],
      list_of_strings=joint_names,
    )
    self.joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
    self.grip_targets = torch.tensor(targets, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    open_std: float,
    grip_std: float,
    velocity_limit_rad_s: float,
    velocity_std: float,
    grip_targets: dict[str, float],
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    del grip_targets
    command = _throw_command(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, self.joint_ids]
    joint_vel = robot.data.joint_vel[:, self.joint_ids]
    open_error = torch.mean(
      torch.square(joint_pos - robot.data.default_joint_pos[:, self.joint_ids]),
      dim=-1,
    )
    grip_error = torch.mean(torch.square(joint_pos - self.grip_targets), dim=-1)
    excess_speed = torch.relu(torch.abs(joint_vel) - velocity_limit_rad_s)
    grip_score = torch.exp(
      -grip_error / grip_std**2
      - torch.mean(torch.square(excess_speed), dim=-1) / velocity_std**2
    )
    return (
      torch.exp(-open_error / open_std**2) * command.finger_open_weight
      + grip_score * command.finger_grip_weight * command.ball_in_catch_zone.float()
    ) * command.synthetic_weight


def standing_posture_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  target_height_m: float,
  tilt_std: float,
  height_std: float,
  posture_std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Keep the base upright and the lower body near its standing pose."""
  robot: Entity = env.scene[asset_cfg.name]
  tilt_error = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=-1)
  height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  height_error = torch.square(height - target_height_m)
  posture_error = torch.mean(
    torch.square(
      robot.data.joint_pos[:, asset_cfg.joint_ids]
      - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    ),
    dim=-1,
  )
  return (
    torch.exp(
      -tilt_error / tilt_std**2
      - height_error / height_std**2
      - posture_error / posture_std**2
    )
    * _throw_command(env, command_name).synthetic_weight
  )


def base_motion_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  position_std: float,
  velocity_std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize walking, drift, and horizontal base velocity."""
  robot: Entity = env.scene[asset_cfg.name]
  position = robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  velocity = robot.data.root_link_lin_vel_b[:, :2]
  return (
    torch.sum(torch.square(position), dim=-1) / position_std**2
    + torch.sum(torch.square(velocity), dim=-1) / velocity_std**2
  ) * _throw_command(env, command_name).synthetic_weight


def controlled_arm_posture_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  ready_std: float,
  elbow_target_rad: float,
  elbow_std: float,
  velocity_std: float,
  ready_cfg: SceneEntityCfg,
  elbow_cfg: SceneEntityCfg,
  arm_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Keep arms quiet and extended before interception and while holding."""
  command = _throw_command(env, command_name)
  robot: Entity = env.scene[arm_cfg.name]
  ready_error = torch.mean(
    torch.square(
      robot.data.joint_pos[:, ready_cfg.joint_ids]
      - robot.data.default_joint_pos[:, ready_cfg.joint_ids]
    ),
    dim=-1,
  )
  elbow_error = torch.mean(
    torch.square(robot.data.joint_pos[:, elbow_cfg.joint_ids] - elbow_target_rad),
    dim=-1,
  )
  velocity_error = torch.mean(
    torch.square(robot.data.joint_vel[:, arm_cfg.joint_ids]), dim=-1
  )
  quiet_extended = torch.exp(
    -elbow_error / elbow_std**2 - velocity_error / velocity_std**2
  )
  ready = (
    torch.exp(-ready_error / ready_std**2)
    * quiet_extended
    * command.pre_catch_pose_weight
  )
  holding = quiet_extended * command.stable_grasp.float()
  return (
    torch.maximum(ready, holding)
    * command.synthetic_weight
    * command.catch_enabled.float()
  )


def idle_posture_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  posture_std: float,
  velocity_std: float,
) -> torch.Tensor:
  """Return arms to the default side pose without moving quickly."""
  command = _throw_command(env, command_name)
  robot: Entity = env.scene[asset_cfg.name]
  error = (
    robot.data.joint_pos[:, asset_cfg.joint_ids]
    - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  velocity = robot.data.joint_vel[:, asset_cfg.joint_ids]
  return (
    torch.exp(
      -error.square().mean(-1) / posture_std**2
      - velocity.square().mean(-1) / velocity_std**2
    )
    * (~command.catch_enabled).float()
    * command.synthetic_weight
  )


def ball_released_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """After a catch, reward a dropped ball, not just open fingers supporting it."""
  command = _throw_command(env, command_name)
  left: ContactSensor = env.scene["left_ball_contact"]
  right: ContactSensor = env.scene["right_ball_contact"]
  assert left.data.found is not None and right.data.found is not None
  no_contact = ~left.data.found.any(-1) & ~right.data.found.any(-1)
  below = command.ball_pos_w[:, 2] < command.palm_midpoint_w[:, 2] - 0.15
  return (
    no_contact & below & command.caught & ~command.catch_enabled
  ).float() * command.synthetic_weight


def arm_motion_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  velocity_limit_rad_s: float,
  acceleration_limit_rad_s2: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize only arm motion exceeding the configured soft safety limits."""
  robot: Entity = env.scene[asset_cfg.name]
  velocity = robot.data.joint_vel[:, asset_cfg.joint_ids]
  acceleration = robot.data.joint_acc[:, asset_cfg.joint_ids]
  velocity_excess = torch.relu(torch.abs(velocity) - velocity_limit_rad_s)
  acceleration_excess = torch.relu(torch.abs(acceleration) - acceleration_limit_rad_s2)
  velocity_cost = torch.mean(
    torch.square(velocity_excess / velocity_limit_rad_s), dim=-1
  )
  acceleration_cost = torch.mean(
    torch.square(acceleration_excess / acceleration_limit_rad_s2), dim=-1
  )
  return (velocity_cost + 0.25 * acceleration_cost) * _throw_command(
    env, command_name
  ).synthetic_weight


def interception_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  speed_std: float,
  distance_std: float,
) -> torch.Tensor:
  """Bring the palms to the ball, with an extra bonus for velocity matching."""
  command = _throw_command(env, command_name)
  speed_score = torch.exp(
    -torch.square(command.ball_hand_relative_speed) / speed_std**2
  )
  proximity = torch.exp(-torch.square(command.ball_palm_distance) / distance_std**2)
  return (
    proximity
    * speed_score
    * command.ball_visible.float()
    * command.synthetic_weight
    * command.catch_enabled.float()
  )


def catch_pose_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reward an upright, demonstration-like catch with the palms below the ball."""
  command = _throw_command(env, command_name)
  return command.catch_pose_score * command.synthetic_weight


def stable_grasp_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reward physically containing and slowing the ball after palm contact."""
  command = _throw_command(env, command_name)
  return (
    command.stable_grasp.float() * command.catch_pose_score * command.synthetic_weight
  )


def catch_success_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """One success pulse per episode, even though holding/release continue."""
  command = _throw_command(env, command_name)
  return command.catch_success_pulse.float() * command.synthetic_weight / env.step_dt


def body_ball_contact_penalty(
  env: ManagerBasedRlEnv, command_name: str, sensor_name: str
) -> torch.Tensor:
  """Penalize the ball touching any robot body other than the hands."""
  command = _throw_command(env, command_name)
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (
    sensor.data.found.any(dim=-1).float()
    * command.ball_visible.float()
    * command.synthetic_weight
  )
