"""Reward terms for ball-conditioned G1 tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from .commands import CatchMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> CatchMotionCommand:
  return cast(CatchMotionCommand, env.command_manager.get_term(command_name))


def ball_position_tracking_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Track the recorded ball trajectory after it becomes visible."""
  command = _command(env, command_name)
  error = torch.sum(
    torch.square(command.ball_reference_pos_w - command.ball_pos_w), dim=-1
  )
  return torch.exp(-error / std**2) * command.ball_visible.float()


def ball_velocity_tracking_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Track the recorded ball linear velocity after it becomes visible."""
  command = _command(env, command_name)
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
  palm_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids]
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
