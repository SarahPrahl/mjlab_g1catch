"""Runtime observations for ball-conditioned motion imitation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from .commands import BallThrowCommand, CatchMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(
  env: ManagerBasedRlEnv, command_name: str
) -> CatchMotionCommand | BallThrowCommand:
  return cast(
    CatchMotionCommand | BallThrowCommand,
    env.command_manager.get_term(command_name),
  )


def _ball_observed(
  command: CatchMotionCommand | BallThrowCommand,
  occlude_in_catch_zone: bool,
) -> torch.Tensor:
  visible = command.ball_visible
  if not occlude_in_catch_zone or not isinstance(command, BallThrowCommand):
    return visible
  return visible & ~command.ball_observation_occluded


def ball_position_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  occlude_in_catch_zone: bool = False,
) -> torch.Tensor:
  """Ball position relative to the robot anchor, in the anchor frame."""
  command = _command(env, command_name)
  relative_w = command.ball_pos_w - command.robot_anchor_pos_w
  relative_b = quat_apply(quat_inv(command.robot_anchor_quat_w), relative_w)
  visible = _ball_observed(command, occlude_in_catch_zone)
  return torch.where(visible[:, None], relative_b, 0.0)


def ball_velocity_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  occlude_in_catch_zone: bool = False,
) -> torch.Tensor:
  """Ball velocity relative to the robot anchor, in the anchor frame."""
  command = _command(env, command_name)
  relative_w = command.ball_lin_vel_w - command.robot_anchor_lin_vel_w
  relative_b = quat_apply(quat_inv(command.robot_anchor_quat_w), relative_w)
  visible = _ball_observed(command, occlude_in_catch_zone)
  return torch.where(visible[:, None], relative_b, 0.0)


def ball_visible(
  env: ManagerBasedRlEnv,
  command_name: str,
  occlude_in_catch_zone: bool = False,
) -> torch.Tensor:
  """Return whether the physical ball is currently available to the tracker."""
  command = _command(env, command_name)
  return _ball_observed(command, occlude_in_catch_zone).float().unsqueeze(-1)


def robot_anchor_height(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Robot anchor height above the local environment origin."""
  command = _command(env, command_name)
  height = command.robot_anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return height.unsqueeze(-1)


def reference_ball_state_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Privileged reference ball state relative to the reference anchor."""
  command = _command(env, command_name)
  assert isinstance(command, CatchMotionCommand)
  position_w = command.ball_reference_pos_w - command.anchor_pos_w
  velocity_w = command.ball_reference_lin_vel_w - command.anchor_lin_vel_w
  inverse_anchor = quat_inv(command.anchor_quat_w)
  position_b = quat_apply(inverse_anchor, position_w)
  velocity_b = quat_apply(inverse_anchor, velocity_w)
  visible = command.ball_visible.float().unsqueeze(-1)
  return torch.cat((position_b, velocity_b, visible), dim=-1)
