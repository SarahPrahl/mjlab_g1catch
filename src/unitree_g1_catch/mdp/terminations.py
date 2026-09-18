"""Termination conditions for synthetic physical-ball throws."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from .commands import BallThrowCommand, RehearsalThrowCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> BallThrowCommand:
  return cast(BallThrowCommand, env.command_manager.get_term(command_name))


def rehearsal_termination(
  env: ManagerBasedRlEnv,
  command_name: str,
  termination_func: Any,
  **termination_params: Any,
) -> torch.Tensor:
  """Apply an existing phase-one termination only to rehearsal environments."""
  command = _command(env, command_name)
  assert isinstance(command, RehearsalThrowCommand)
  value = termination_func(env, command_name=command_name, **termination_params)
  return value & command.rehearsal


def catch_succeeded(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """End an episode after the ball has been held stably for the required time."""
  command = _command(env, command_name)
  return command.caught & command.synthetic_weight.bool()


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
  too_far_after_arrival = (command.time_to_arrival <= 0.0) & (
    distance > maximum_ball_distance_m
  )
  return (
    command.launched
    & command.synthetic_weight.bool()
    & ~command.caught
    & ((relative_height < minimum_ball_height_m) | too_far_after_arrival)
  )
