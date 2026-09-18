"""Curriculum terms for the catch environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def terrain_mix(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  flat_steps: int,
  curriculum_steps: int,
) -> torch.Tensor:
  """Grow the uneven-terrain share from zero to its configured proportion."""
  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  generator = terrain.cfg.terrain_generator
  assert generator is not None

  if isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]

  progress = min(
    max(env.common_step_counter - flat_steps, 0)
    / max(curriculum_steps - flat_steps, 1),
    1.0,
  )
  names = list(generator.sub_terrains)
  flat_index = names.index("flat")
  rough_index = names.index("slight_undulations")
  total_proportion = sum(cfg.proportion for cfg in generator.sub_terrains.values())
  rough_probability = (
    progress
    * generator.sub_terrains["slight_undulations"].proportion
    / total_proportion
  )

  rough = torch.rand(len(env_ids), device=env.device) < rough_probability
  terrain.terrain_types[env_ids] = torch.where(rough, rough_index, flat_index)
  maximum_level = int(progress * (terrain.terrain_origins.shape[0] - 1))
  levels = torch.randint(maximum_level + 1, (len(env_ids),), device=env.device)
  terrain.terrain_levels[env_ids] = torch.where(rough, levels, 0)
  terrain.env_origins[env_ids] = terrain.terrain_origins[
    terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
  ]
  return torch.tensor(rough_probability, device=env.device)
