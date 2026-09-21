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


def release_finished(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Allow a fixed release/rest window after disarming, also if release fails."""
  command = _command(env, command_name)
  elapsed = env.episode_length_buf * env.step_dt
  return (
    (elapsed >= command.disable_time + command.cfg.release_time_s)
    & command.synthetic_weight.bool()
    & (command.cfg.catch_enabled_override is None)
  )


def rehearsal_time_out(
  env: ManagerBasedRlEnv,
  command_name: str,
  duration_s: float,
) -> torch.Tensor:
  """Keep reference episode duration independent of the new release window."""
  command = _command(env, command_name)
  assert isinstance(command, RehearsalThrowCommand)
  return command.rehearsal & (env.episode_length_buf * env.step_dt >= duration_s)


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
    & command.catch_enabled
    & command.synthetic_weight.bool()
    & ((relative_height < minimum_ball_height_m) | too_far_after_arrival)
  )
