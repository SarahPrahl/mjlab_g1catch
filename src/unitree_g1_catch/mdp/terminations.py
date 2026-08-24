"""Termination conditions for synthetic physical-ball throws."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from .commands import BallThrowCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> BallThrowCommand:
  return cast(BallThrowCommand, env.command_manager.get_term(command_name))


def catch_succeeded(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """End an episode after the ball has been held stably for the required time."""
  return _command(env, command_name).caught


def throw_missed(
  env: ManagerBasedRlEnv,
  command_name: str,
  minimum_ball_height_m: float,
  maximum_ball_distance_m: float,
) -> torch.Tensor:
  """End an episode once the launched ball hits the floor or leaves reach."""
  command = _command(env, command_name)
  relative_height = command.ball_pos_w[:, 2] - env.scene.env_origins[:, 2]
  distance = torch.norm(command.ball_pos_w - command.robot_anchor_pos_w, dim=-1)
  return (
    command.launched
    & ~command.caught
    & ((relative_height < minimum_ball_height_m) | (distance > maximum_ball_distance_m))
  )
